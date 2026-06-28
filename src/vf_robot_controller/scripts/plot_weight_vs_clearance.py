#!/usr/bin/env python3
"""Plot per-critic weight vs nearest-obstacle clearance.

Two data sources, same code path:
  --source oracle  : reads `oracle_weights` from vf_fixedwt H5s (recovered
                     targets — answers "does the supervision signal vary
                     with clearance?").
  --source deploy  : reads `critic_weights_applied` from inferencewt H5s
                     (live NN predictions at deploy time — answers "does
                     the trained NN reproduce the pattern?").

Per critic, one panel: scatter + binned mean curve + Spearman correlation
with min-clearance. The obstacle critic (WeightedCostCritic) is the headline
test — its Spearman should be strongly negative if adaptation worked.

Usage:
  plot_weight_vs_clearance.py /path/to/dir --source oracle --out oracle.png
  plot_weight_vs_clearance.py /path/to/dir --source deploy --out deploy.png
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import h5py
import numpy as np
from scipy.stats import spearmanr


def _channel_slice(f: h5py.File, name: str) -> Optional[Tuple[int, int]]:
    names = [n.decode() if isinstance(n, bytes) else n
             for n in f.attrs["channel_names"]]
    dims = list(f.attrs["channel_dims"])
    if name not in names:
        return None
    idx = names.index(name)
    start = int(sum(dims[:idx]))
    return start, int(dims[idx])


def _proximity_signal(f: h5py.File) -> Optional[Tuple[np.ndarray, str, int]]:
    """Return (signal, label, expected_sign).

    Rosette layout is N × (composite, clearance_2d, clutter_density). When
    `clearance_2d` is filled, lower clearance = closer obstacle, so the
    obstacle-critic weight should *anti*-correlate with it (sign = -1).

    Some map backends do not fill clearance/clutter and only the composite
    GCF field carries the signal: higher composite = more danger = closer
    obstacle, so the obstacle-critic weight should *positively* correlate
    (sign = +1). The script picks whichever field actually varies.
    """
    sl = _channel_slice(f, "gcf_rosette")
    if sl is None:
        return None
    start, width = sl
    if width % 3 != 0:
        return None
    feats = f["features"][:, start:start + width]
    composite = feats[:, 0::3]
    clearance = feats[:, 1::3]
    if float(clearance.std()) > 1e-6:
        return clearance.min(axis=1), "min clearance [m]", -1
    if float(composite.std()) > 1e-6:
        return composite.max(axis=1), "max GCF composite (obstacle risk)", +1
    return None


def _collect(root: str, source: str):
    """Walk `root` for .h5 files; return (clearance, weights, critic_names).

    `clearance`: (R,)  ;  `weights`: (R, K)  ;  R = total valid rows.
    """
    weight_key = "oracle_weights" if source == "oracle" else "critic_weights_applied"
    all_x, all_w, critic_names = [], [], None
    label, sign, n_files, n_used = None, None, 0, 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".h5"):
                continue
            n_files += 1
            path = os.path.join(dirpath, name)
            try:
                with h5py.File(path, "r") as f:
                    if weight_key not in f:
                        continue
                    prox = _proximity_signal(f)
                    if prox is None:
                        continue
                    x, lbl, sgn = prox
                    w = f[weight_key][...]
                    if w.shape[0] != x.shape[0]:
                        continue
                    if critic_names is None:
                        cn = f.attrs.get("critic_names")
                        critic_names = [c.decode() if isinstance(c, bytes) else c
                                        for c in cn]
                        label, sign = lbl, sgn
                    all_x.append(x.astype(np.float64))
                    all_w.append(w.astype(np.float64))
                    n_used += 1
            except (OSError, KeyError) as e:
                print(f"  skip {path}: {e}", file=sys.stderr)
    if not all_x:
        return None, None, None, None, None, n_files, 0
    x = np.concatenate(all_x)
    weights = np.concatenate(all_w, axis=0)
    finite = np.isfinite(x) & np.all(np.isfinite(weights), axis=1)
    return x[finite], weights[finite], critic_names, label, sign, n_files, n_used


def _binned_mean(x: np.ndarray, y: np.ndarray, nbins: int = 20):
    """Equal-width binning of y over x; returns (centers, means, stds)."""
    lo, hi = np.quantile(x, [0.01, 0.99])
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(nbins, np.nan)
    stds = np.full(nbins, np.nan)
    for i in range(nbins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.sum() > 5:
            means[i] = y[mask].mean()
            stds[i] = y[mask].std()
    return centers, means, stds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Directory to walk for .h5 files")
    ap.add_argument("--source", choices=["oracle", "deploy"], required=True)
    ap.add_argument("--out", default=None, help="Output PNG (default: derived)")
    ap.add_argument("--max-points", type=int, default=20000,
                    help="Scatter subsample cap for plot legibility")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"[plot_w_vs_clr] walking {args.root}, source={args.source}")
    x, weights, critic_names, xlabel, sign, n_files, n_used = _collect(
        args.root, args.source)
    if x is None:
        print(f"[plot_w_vs_clr] no usable H5s ({n_files} scanned). "
              f"For --source oracle, runs need an `oracle_weights` dataset "
              f"(produced by scripts/run_oracle.py).", file=sys.stderr)
        return 1
    K = weights.shape[1]
    print(f"[plot_w_vs_clr] {n_used}/{n_files} files used, "
          f"{x.size} rows, K={K} critics, x-axis = {xlabel}")
    expected = ("ρ < 0" if sign < 0 else "ρ > 0")

    rng = np.random.default_rng(0)
    if x.size > args.max_points:
        sub = rng.choice(x.size, args.max_points, replace=False)
    else:
        sub = np.arange(x.size)

    cols = 4
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows),
                             squeeze=False)
    title = ("Oracle target weights vs proximity"
             if args.source == "oracle"
             else "Deployed NN weights vs proximity")
    fig.suptitle(f"{title}  ({n_used} episodes, {x.size} rows)  x = {xlabel}",
                 fontsize=12)

    summary = []
    for k in range(K):
        ax = axes[k // cols][k % cols]
        cname = (critic_names[k] if critic_names and k < len(critic_names)
                 else f"critic_{k}")
        rho, pval = spearmanr(x, weights[:, k])
        summary.append((cname, rho, pval))
        ax.scatter(x[sub], weights[sub, k], s=2, alpha=0.15, c="steelblue")
        centers, means, _ = _binned_mean(x, weights[:, k])
        ax.plot(centers, means, "-", color="crimson", lw=2)
        ax.set_title(f"{cname}\nSpearman ρ={rho:+.3f}  p={pval:.1e}",
                     fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("weight", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    for k in range(K, rows * cols):
        axes[k // cols][k % cols].axis("off")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor="steelblue", markeredgecolor="steelblue", alpha=0.6,
               label="per-step weight (one dot = one MPPI cycle)"),
        Line2D([0], [0], color="crimson", lw=2,
               label="binned mean of weight over proximity bins"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = args.out or f"weight_vs_clearance_{args.source}.png"
    fig.savefig(out, dpi=140)
    print(f"[plot_w_vs_clr] wrote {out}")

    print(f"\nSpearman ρ({xlabel}, weight) per critic")
    print(f"{'critic':<36s}  {'ρ':>7s}  {'p':>10s}")
    for cname, rho, pval in summary:
        flag = "  <-- HEADLINE" if "Cost" in cname else ""
        print(f"{cname:<36s}  {rho:+7.3f}  {pval:10.2e}{flag}")
    print(f"\nAdaptation evidence: expect {expected} for WeightedCostCritic "
          f"(obstacle critic).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
