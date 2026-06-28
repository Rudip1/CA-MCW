#!/usr/bin/env python3
"""comparison_figures.py — cross-controller comparison suite.

Compares a controller set directly — no screening / top-K funnel. Two
scopes:

  --scope headline   6 controllers: 3 classical (mppi / dwb / rpp)
                     + fixedwt + imitationwt + the TOPSIS rank-1 trained
                     controller. Full figure set — per-tier bar panels,
                     XTE violin / profile / envelope, trajectory overlays.

  --scope full       Every controller in the dataset. Only the figure
                     types that stay legible at ~24 controllers:
                     per-tier bar panels + XTE violin.

Per-tier bar panels use ``renderers_bars.render_tier_panel`` — bar +
Wilson 95% CI for rate metrics, bar + bootstrap 95% CI for continuous
metrics. XTE and trajectory figures use ``renderers_xte`` /
``renderers_trajectories``.

Tier 6 (Adaptation) is restricted to controllers that emit a critic
weight vector — the trained families + fixedwt + imitation; the classical
planners are dropped from that panel (they have no weights).

Output (under ``<agg-dir>/comparison[_all]/`` unless ``--out``):
  tiers/         tier_1_outcome.pdf ... tier_6_adaptation.pdf
  xte/           xte_violin.pdf, and [headline only] xte_overview.pdf
                 (3-panel envelope|profile|violin) + per-goal profiles
                 + per-controller envelopes
  trajectories/  trajectories_<goal>.pdf            [headline only]
  comparison_summary.csv

Usage::

    python3 -m vf_robot_utils.analysis.figures.comparison_figures \\
        --results .../results.csv \\
        --agg-dir .../_aggregate/thesis_eval \\
        --root .../batch/house_my1_map --scope headline
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vf_robot_utils.analysis.style import (
    apply_style, controller_color, order_by_category,
    category_separators, category_spans, SEPARATOR_COLOR,
)
from vf_robot_utils.analysis.csv_pipeline import (
    METRIC_CATALOG, PROPORTION_METRICS, ALL_CONTROLLERS, TOPSIS_CRITERIA,
)
from vf_robot_utils.analysis.csv_pipeline.controller_selection import (
    build_decision_matrix, correlation_filter, _vector_normalize,
    _flip_lower_better, entropy_weights, topsis, apply_sr_gate,
)

# Controllers whose mean success rate is at or below this are gated out
# of the TOPSIS ranking (the agreed selection pipeline ranks only
# controllers that reliably reach the goal). They still appear in the
# comparison figures — just not in the winner ranking.
_SR_GATE: float = 0.5
from vf_robot_utils.analysis.figures.renderers_bars import (
    render_tier_panel, _short,
)
from vf_robot_utils.analysis.figures.renderers_trajectories import (
    _load_map, _autodetect_map_yaml, render_per_goal, render_tour_overview,
)
from vf_robot_utils.analysis.figures.renderers_xte import (
    render_violin, render_profile, render_envelope, render_xte_overview,
)


TIER_ORDER: list[str] = [
    "Outcome", "Efficiency", "Safety",
    "Motion quality", "Path adherence", "Adaptation",
]
TIER_SLUG: dict[str, str] = {
    "Outcome": "outcome", "Efficiency": "efficiency", "Safety": "safety",
    "Motion quality": "motion_quality", "Path adherence": "path_adherence",
    "Adaptation": "adaptation",
}
TIER_METRICS: dict[str, list[str]] = {
    "Outcome": ["t2_success", "t2_timeout", "t2_aborted", "t2_rescued",
                "t2_gpe_m", "t2_goe_rad"],
    "Efficiency": ["t2_spl_safe", "t2_spl", "t2_duration_s",
                   "t2_actual_path_m", "t3_mean_lin_vel",
                   "t3_stall_fraction"],
    "Safety": ["t1_collision", "t1_mmc_m", "t1_mdo_m",
               "t1_p5_clear_m", "t1_near_miss_rate", "t1_svr"],
    "Motion quality": ["t4_mean_jerk", "t4_max_jerk", "t4_ang_vel_std",
                       "t4_cmd_accel_rms", "t4_cmd_sign_flips",
                       "t4_cmd_ang_sign_flips"],
    "Path adherence": ["t6_mean_xte_m", "t6_median_xte_m", "t6_rmse_xte_m",
                       "t6_max_xte_m", "t6_p95_xte_m",
                       "t6_path_coverage_frac"],
    "Adaptation": ["t5_mean_entropy_norm", "t5_total_variation",
                   "t5_tv_per_second", "t5_dominant_critic_fraction"],
}

# Controllers with no critic-weight vector — dropped from the Adaptation
# tier (trained families + fixedwt + imitation are kept).
_NO_WEIGHTS: tuple[str, ...] = ("mppi", "dwb", "rpp", "graceful")

_LABEL: dict[str, str] = {m.col: m.label for m in METRIC_CATALOG}
_DIR: dict[str, int] = {m.col: m.dir for m in METRIC_CATALOG}
_DROP_INF: dict[str, bool] = {m.col: m.drop_inf for m in METRIC_CATALOG}


def _has_data(df: pd.DataFrame, ctrl: str, col: str, drop_inf: bool) -> bool:
    s = pd.to_numeric(df.loc[df["controller"] == ctrl, col],
                      errors="coerce").to_numpy(dtype=np.float64)
    s = s[np.isfinite(s) if drop_inf else ~np.isnan(s)]
    return s.size > 0


def _tier_metric_tuples(
    tier: str, df: pd.DataFrame, controllers: list[str],
) -> list[tuple[str, str, bool, bool, int]]:
    """(col, label, is_prop, drop_inf, direction) for populated metrics."""
    out: list[tuple[str, str, bool, bool, int]] = []
    for col in TIER_METRICS[tier]:
        if col not in df.columns:
            continue
        drop_inf = _DROP_INF.get(col, False)
        if not any(_has_data(df, c, col, drop_inf) for c in controllers):
            continue
        out.append((col, _LABEL.get(col, col), col in PROPORTION_METRICS,
                    drop_inf, _DIR.get(col, 0)))
    return out


def _headline_controllers(agg_dir: Path,
                          present: set[str]) -> tuple[list[str], str]:
    """6-controller headline set; reference = TOPSIS rank-1 trained."""
    sel = pd.read_csv(agg_dir / "controller_selection.csv")
    top1 = str(sel.sort_values("rank").iloc[0]["Controller"])
    wanted = ["mppi", "dwb", "rpp", "fixedwt", "imitationwt_normal_v1", top1]
    return [c for c in wanted if c in present], top1


def _compute_topsis(df: pd.DataFrame,
                    controllers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """TOPSIS ranking over a controller set, with the success-rate gate.

    Reuses the entropy-weighted TOPSIS pipeline from controller_selection
    (SR gate -> decision matrix -> correlation prune -> vector normalise
    -> flip lower-better -> entropy weights -> closeness coefficient).

    Controllers with mean success rate <= ``_SR_GATE`` are gated out
    before ranking — both because an unreliable controller has no claim
    to "winner", and because a lone gated controller's collision /
    failure column would otherwise dominate the entropy weights and
    collapse the ranking.

    Returns ``(ranked_df, gated_out)`` — ``ranked_df`` has columns
    ``Controller, C_star, S_plus, S_minus, rank`` sorted best first.
    """
    kept = apply_sr_gate(df, controllers, _SR_GATE)
    gated = [c for c in controllers if c not in kept]
    mat = build_decision_matrix(df, kept, list(TOPSIS_CRITERIA))
    cols, _, _ = correlation_filter(mat)
    mat = mat[cols]
    norm = _flip_lower_better(_vector_normalize(mat))
    weights = entropy_weights(norm)
    return topsis(norm, weights), gated


def render_topsis_ranking(df: pd.DataFrame, display_controllers: list[str],
                          compute_controllers: list[str],
                          out_pdf: Path, out_csv: Path,
                          label_map: dict[str, str] | None = None,
                          legend_label_map: dict[str, str] | None = None,
                          highlight: str | None = None,
                          scope: str = "full") -> None:
    """TOPSIS closeness-coefficient bar panel — same style as the tiers.

    TOPSIS is computed once over the full controller set
    (``compute_controllers``) — entropy-weighted TOPSIS is unstable on a
    tiny subset — and the full ranking is written to ``out_csv``. The
    figure shows only ``display_controllers`` as a single vertical bar
    panel in the tier-figure style: controllers in category order
    (classical | fixedwt | trained | imitation) with red dashed
    separators + category labels, slanted x-labels, thin black box. The
    winner's bar is starred; a success-gated controller has no bar.
    """
    ranked, gated = _compute_topsis(df, compute_controllers)
    ranked.to_csv(out_csv, index=False, float_format="%.5f")
    label_map = label_map or {}
    legend_label_map = legend_label_map or label_map
    cstar_of = dict(zip(ranked["Controller"], ranked["C_star"]))

    controllers = order_by_category(display_controllers)
    ranked_set = [c for c in controllers if c in cstar_of]
    winner = max(ranked_set, key=lambda c: cstar_of[c]) if ranked_set else None
    n = len(controllers)
    many = n > 12

    apply_style()
    fig_w = 13.0 if many else 7.6
    fig_h = 6.4 if many else 5.8       # extra room for the legend strip
    tick_fs = 7.0 if many else 12.0
    cat_fs = 5.0 if many else 9.0
    val_fs = 6.5 if many else 12.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    xs = np.arange(n, dtype=float)
    vals = np.array([cstar_of.get(c, np.nan) for c in controllers])
    colors = [controller_color(c, scope=scope, highlight=highlight)
              for c in controllers]
    edge_w = [1.8 if c == highlight else 0.4 for c in controllers]
    ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=edge_w,
           width=0.82, zorder=2)

    for x, c in zip(xs, controllers):
        if c in cstar_of:
            v = cstar_of[c]
            ax.text(x, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=val_fs,
                    fontweight=("bold" if c == winner else "normal"))
            if c == winner:
                ax.text(x, v + 0.075, "rank 1", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#d62728")
        else:
            ax.text(x, 0.08, "gated\n(SR $\\leq$ 0.5)", ha="center", va="bottom",
                    fontsize=11, style="italic", color="#444444",
                    fontweight="bold")

    ax.set_ylim(0, 1.18)
    # Extra x-margin so an edge category label (e.g. a single-bar
    # "Imitation" group) stays centred without clipping the axes.
    ax.set_xlim(-0.85, (n - 1) + 0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([label_map.get(c, _short(c)) for c in controllers],
                       rotation=70, ha="right", fontsize=tick_fs)
    for tl, c in zip(ax.get_xticklabels(), controllers):
        if c == highlight:
            tl.set_fontweight("bold")
            tl.set_fontsize(tick_fs + 2.0)
            tl.set_color("#000000")
    ax.set_ylabel("TOPSIS closeness coefficient $C^*$", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("TOPSIS ranking ($\\uparrow$ better) — rank-1: "
                 f"{label_map.get(winner, _short(winner)) if winner else 'n/a'}",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for sp in ax.spines.values():           # thin black panel box
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_color("black")
    for sx in category_separators(controllers):
        ax.axvline(sx, color=SEPARATOR_COLOR, linestyle=(0, (4, 3)),
                   linewidth=1.0, zorder=1)
    for centre, clabel in category_spans(controllers):
        ax.text(centre, 0.97, clabel, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=cat_fs,
                fontweight="bold", color=SEPARATOR_COLOR)

    # Shared legend below the bars: same controller→colour mapping as the
    # tier panels (Figs 5.2–5.6). One handle per controller; the highlight
    # is bold and gets a thicker swatch edge.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=controller_color(c, scope=scope, highlight=highlight),
                     edgecolor="black",
                     linewidth=(1.8 if c == highlight else 0.4),
                     label=legend_label_map.get(c, _short(c)))
               for c in controllers]
    ncol = min(3, len(controllers)) if not many else min(6, len(controllers))
    leg = fig.legend(handles=handles, loc="lower center", ncol=ncol,
                     frameon=False, fontsize=10.5,
                     bbox_to_anchor=(0.5, 0.0),
                     handlelength=2.2, handleheight=1.5,
                     columnspacing=2.0, labelspacing=0.7)
    for txt, c in zip(leg.get_texts(), controllers):
        if c == highlight:
            txt.set_fontweight("bold")

    # Leave space at the bottom for the legend strip; rect tuple is
    # (left, bottom, right, top) in normalised figure coordinates.
    legend_in = 1.0 if many else 0.9
    fig.tight_layout(rect=(0.0, legend_in / fig_h, 1.0, 1.0))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[comparison] TOPSIS — winner: {winner}  "
          f"({len(ranked_set)} ranked, {len(gated)} gated)")


_VOLUMETRIC_IDX = 9   # disabled critic — column is identically zero; skipped


def _h5_proximity(f: h5py.File) -> np.ndarray | None:
    """Per-step obstacle-proximity from the gcf_rosette feature channel.

    Rosette layout is N x (composite, clearance_2d, clutter_density). The
    clearance_2d sub-channels are unpopulated in these logs, so the
    obstacle signal is the GCF *composite* (columns 0::3) — a unitless
    obstacle-risk score the controller actually perceives: higher = the
    robot is closer to an obstacle. Returns the per-step max composite,
    or ``None`` if the channel is absent / flat.
    """
    names = [n.decode() if isinstance(n, (bytes, np.bytes_)) else str(n)
             for n in f.attrs.get("channel_names", [])]
    dims = list(f.attrs.get("channel_dims", []))
    if "gcf_rosette" not in names:
        return None
    i = names.index("gcf_rosette")
    start = int(sum(dims[:i]))
    width = int(dims[i])
    if width % 3 != 0 or "features" not in f:
        return None
    feats = f["features"][:, start:start + width].astype(np.float64)
    composite = feats[:, 0::3]
    if float(np.nanstd(composite)) <= 1e-6:
        return None
    return np.nanmax(composite, axis=1)


def _collect_weight_share(paths: Iterable[str], weight_key: str):
    """Pool (proximity, normalised-weight-share) over a set of HDF5 files.

    Each per-step weight vector is normalised to sum 1 — a 'weight share'
    — so the meta-critic output (sum != 1) and the QP oracle weights
    (already normalised) are on one comparable scale. Returns
    ``(proximity[R], share[R, K], critic_names)`` or ``(None, None, None)``.
    """
    xs: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    critic_names: list[str] | None = None
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                if weight_key not in f:
                    continue
                clr = _h5_proximity(f)
                if clr is None:
                    continue
                w = f[weight_key][...].astype(np.float64)
                if w.ndim != 2 or w.shape[0] != clr.shape[0]:
                    continue
                if critic_names is None:
                    critic_names = [
                        c.decode() if isinstance(c, (bytes, np.bytes_))
                        else str(c)
                        for c in f.attrs.get("critic_names", [])]
                xs.append(clr)
                ws.append(w)
        except (OSError, KeyError):
            continue
    if not xs:
        return None, None, None
    x = np.concatenate(xs)
    W = np.concatenate(ws, axis=0)
    fin = np.isfinite(x) & np.all(np.isfinite(W), axis=1)
    x, W = x[fin], W[fin]
    s = W.sum(axis=1, keepdims=True)
    W = W / np.where(s > 1e-9, s, 1.0)        # -> weight share, sums to 1
    return x, W, critic_names


def _binned_curve(x: np.ndarray, y: np.ndarray, edges: np.ndarray,
                  min_n: int = 10) -> np.ndarray:
    """Mean of ``y`` per proximity bin; NaN where a bin is too sparse."""
    centers = len(edges) - 1
    out = np.full(centers, np.nan)
    bidx = np.clip(np.digitize(x, edges) - 1, 0, centers - 1)
    for b in range(centers):
        v = y[bidx == b]
        if v.size >= min_n:
            out[b] = float(v.mean())
    return out


def render_critic_adaptation(df: pd.DataFrame, root: Path, winner: str,
                             out_pdf: Path,
                             label_map: dict[str, str] | None = None,
                             orientation: str = "vertical") -> None:
    """Per-critic small-multiples: critic weight SHARE vs obstacle clearance.

    Three lines per critic, all on one comparable scale (weight share —
    each per-step weight vector normalised to sum 1):

      oracle  — QP cost-optimal weights (``oracle_weights`` in the
                training-tree vf_fixedwt HDF5s): the *correct* weighting.
      winner  — the deployed controller's meta-critic output.
      fixedwt — uniform share (no meta-critic): the no-adaptation floor.

    Where the winner line tracks the oracle line, the controller adapts
    *correctly* — it has recovered the cost-optimal context response.
    """
    label_map = label_map or {}

    # Training tree (oracle weights live in vf_fixedwt training HDF5s).
    parts = root.parts
    if "vf_data_evaluation" in parts:
        training_root = Path(*parts[:parts.index("vf_data_evaluation")]) \
            / "vf_data_training"
    else:
        training_root = root.parent / "vf_data_training"

    win_paths = list(df.loc[df["controller"] == winner, "hdf5_path"])
    x_win, W_win, cnames = _collect_weight_share(win_paths,
                                                 "critic_weights_applied")
    if x_win is None:
        print(f"[comparison] no usable {winner} HDF5 — skipping tier 6",
              file=sys.stderr)
        return

    oracle_paths = [str(p) for p in sorted(training_root.rglob("run_*.h5"))]
    x_ora, W_ora, _ = _collect_weight_share(oracle_paths, "oracle_weights")

    apply_style()
    K = W_win.shape[1]
    critics = [(i, cnames[i] if cnames and i < len(cnames) else f"critic {i}")
               for i in range(K) if i != _VOLUMETRIC_IDX]
    n = len(critics)
    if orientation == "horizontal":
        # Wide grid for the poster: two rows so the panels form a broad,
        # short strip that fills the column width. K=10 -> 5 cols x 2 rows.
        nrows = 2
        ncols = math.ceil(n / nrows)
    else:
        # Portrait-favouring grid: at \includegraphics[width=\textwidth] a
        # taller-than-wide source PDF scales to a larger printed figure than
        # a wide-and-short one. K=10 -> 3 cols x 4 rows (2 slots hidden);
        # K=14 -> 4 cols x 4 rows.
        ncols = 3 if n <= 12 else 4
        nrows = math.ceil(n / ncols)

    c_lo = max(0.0, float(np.percentile(x_win, 1)))
    c_hi = float(np.percentile(x_win, 98))
    edges = np.linspace(c_lo, c_hi, 9)           # 8 bins — winner set is small
    centers = (edges[:-1] + edges[1:]) / 2.0
    fw_share = 1.0 / K                           # uniform — no adaptation

    # Per-critic panels: enlarged so each subplot is readable in print.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.85 * ncols, 3.4 * nrows),
                             squeeze=False)
    flat = list(axes.flat)
    win_color = controller_color(winner, scope="headline", highlight=winner)
    win_lbl = label_map.get(winner, _short(winner)) + " (deployed)"

    for idx, (ci, cname) in enumerate(critics):
        ax = flat[idx]
        w_curve = _binned_curve(x_win, W_win[:, ci], edges)
        ax.plot(centers, w_curve, color=win_color, linewidth=2.0,
                marker="o", markersize=3.5, label=win_lbl, zorder=4)
        if x_ora is not None and ci < W_ora.shape[1]:
            o_curve = _binned_curve(x_ora, W_ora[:, ci], edges)
            ax.plot(centers, o_curve, color="#000000", linewidth=1.8,
                    linestyle="-.", marker="^", markersize=3.5,
                    label="oracle (QP optimal)", zorder=3)
        ax.plot([c_lo, c_hi], [fw_share, fw_share], color="#999999",
                linewidth=1.5, linestyle="--",
                label="fixedwt (no adaptation)", zorder=2)
        short = cname.replace("Weighted", "").replace("Critic", "")
        ax.set_title(short, fontsize=13, fontweight="bold", pad=4)
        ax.set_xlim(c_lo, c_hi)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.25)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("obstacle proximity  (GCF, higher = closer)",
                          fontsize=12)
        if idx % ncols == 0:
            ax.set_ylabel("critic weight share\n(of total, sums to 1)",
                          fontsize=12)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.8)
            sp.set_color("black")
    for j in range(n, nrows * ncols):
        flat[j].set_visible(False)

    # Compact legend strip at the bottom; the full descriptive caption
    # lives in the LaTeX figure environment, not inside the PDF.
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Tier 6: Adaptation — critic weight share vs obstacle "
                 "proximity (deployed vs QP-optimal)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    oracle_n = 0 if x_ora is None else int(x_ora.size)
    print(f"[comparison] wrote {out_pdf}  ({n} critics; winner steps="
          f"{int(x_win.size)}, oracle steps={oracle_n})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--agg-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True,
                   help="batch/<map> directory — used to autodetect the map.")
    p.add_argument("--scope", choices=["headline", "full"], required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    df = pd.read_csv(args.results)
    if "timestamp" in df.columns:
        df = (df.sort_values("timestamp")
                .drop_duplicates(subset=["controller", "goal_folder"],
                                 keep="last")
                .reset_index(drop=True))
    n_goals = int(df["goal_folder"].nunique())
    present = set(df["controller"].unique())

    # TOPSIS rank-1 trained controller — emphasised in every figure.
    sel = pd.read_csv(args.agg_dir / "controller_selection.csv")
    top1 = str(sel.sort_values("rank").iloc[0]["Controller"])
    highlight = top1 if top1 in present else None

    if args.scope == "headline":
        controllers, top1 = _headline_controllers(args.agg_dir, present)
        out_dir = args.out or (args.agg_dir / "comparison")
        # Short slanted x-axis labels (kept compact); the full descriptive
        # names go in the colour legend.
        label_map = {
            "fixedwt":                "FW-MPPI",
            "imitationwt_normal_v1":  "IL-V",
            top1:                     "CA-MCW",
        }
        legend_label_map = {
            "fixedwt":                "FW-MPPI (fixed-weight baseline)",
            "imitationwt_normal_v1":  "IL-V (imitation baseline)",
            top1:                     "CA-MCW (this thesis)",
        }
        print(f"[comparison] reference (TOPSIS rank 1)={top1}")
    else:
        controllers = [c for c in ALL_CONTROLLERS if c in present]
        out_dir = args.out or (args.agg_dir / "comparison_all")
        # Full set keeps the canonical short names for the trained-grid
        # variants (rawwt_*, oraclewt_*) but renames the three named
        # controllers to the thesis acronyms for consistency with the
        # headline figures.
        label_map = {
            "fixedwt":                "FW-MPPI",
            "imitationwt_normal_v1":  "IL-V",
            top1:                     "CA-MCW",
        }
        legend_label_map = {
            "fixedwt":                "FW-MPPI (fixed-weight baseline)",
            "imitationwt_normal_v1":  "IL-V (imitation baseline)",
            top1:                     "CA-MCW (this thesis)",
        }
    if not controllers:
        print("[comparison] no controllers found in results.csv",
              file=sys.stderr)
        return 1
    print(f"[comparison] scope={args.scope}  controllers={controllers}")

    tiers_dir = out_dir / "tiers"
    tiers_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-tier bar panels (tiers 1–5; tier 6 is a dedicated figure) ──────
    summary_rows: list[dict] = []
    for i, tier in enumerate(TIER_ORDER, start=1):
        if tier == "Adaptation":
            continue          # rendered below as the critic-adaptation grid
        panel_ctrls = controllers
        if not panel_ctrls:
            continue
        metrics = _tier_metric_tuples(tier, df, panel_ctrls)
        if not metrics:
            print(f"[comparison] {tier}: no populated metrics, skipped")
            continue
        out_pdf = tiers_dir / f"tier_{i}_{TIER_SLUG[tier]}.pdf"
        rows = render_tier_panel(
            df, panel_ctrls, metrics, out_pdf,
            tier_header=f"Tier {i}: {tier}",
            label_map=label_map, legend_label_map=legend_label_map,
            highlight=highlight,
            use_legend=(args.scope == "headline"),
            scope=args.scope)
        for r in rows:
            r["Tier"] = tier
        summary_rows.extend(rows)
        print(f"[comparison] wrote {out_pdf}")

    summary_csv = out_dir / "comparison_summary.csv"
    cols = ["Tier", "Metric", "Label", "Controller", "n",
            "mean", "ci_lo", "ci_hi"]
    pd.DataFrame(summary_rows).reindex(columns=cols).to_csv(
        summary_csv, index=False, float_format="%.5f")
    print(f"[comparison] wrote {summary_csv}")

    # ── Tier 6: Adaptation — per-critic weight vs clearance grid ───────────
    if highlight is not None:
        render_critic_adaptation(df, args.root, highlight,
                                 tiers_dir / "tier_6_adaptation.pdf",
                                 label_map=label_map)

    # ── TOPSIS ranking — declares the overall winner ───────────────────────
    # Computed once over every controller present (TOPSIS is unstable on a
    # tiny subset); the figure shows the comparison set with global ranks.
    compute_set = [c for c in ALL_CONTROLLERS if c in present]
    render_topsis_ranking(df, controllers, compute_set,
                          out_dir / "topsis_ranking.pdf",
                          out_dir / "topsis_ranking.csv",
                          label_map=label_map,
                          legend_label_map=legend_label_map,
                          highlight=highlight, scope=args.scope)
    print(f"[comparison] wrote {out_dir / 'topsis_ranking.pdf'}")

    # ── XTE violin (both scopes — scales fine to many controllers) ─────────
    xte_dir = out_dir / "xte"
    xte_dir.mkdir(parents=True, exist_ok=True)
    render_violin(df, controllers, xte_dir / "xte_violin.pdf",
                  label_map=label_map, highlight=highlight,
                  show_category_labels=True, scope=args.scope)
    print(f"[comparison] wrote {xte_dir / 'xte_violin.pdf'}")

    # ── XTE overview + per-goal / per-controller detail + trajectories ─────
    # Produced for both scopes — same figure set, only the controller
    # count differs.
    goals = sorted(df["goal_folder"].unique())
    render_xte_overview(df, controllers, xte_dir / "xte_overview.pdf",
                        label_map=label_map,
                        legend_label_map=legend_label_map,
                        highlight=highlight, scope=args.scope)
    for goal in goals:                       # goal carries the "goal_" prefix
        render_profile(df, goal, controllers,
                       xte_dir / f"xte_profile_{goal}.pdf",
                       label_map=label_map, highlight=highlight)
    for ctrl in controllers:
        render_envelope(df, ctrl, xte_dir / f"xte_envelope_{ctrl}.pdf")
    print(f"[comparison] wrote xte_overview.pdf + per-goal profiles "
          f"({len(goals)}) + per-controller envelopes ({len(controllers)})")

    map_yaml = _autodetect_map_yaml(args.root.name)
    if map_yaml is None:
        print("[comparison] no map YAML — skipping trajectory overlays",
              file=sys.stderr)
    else:
        traj_dir = out_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        img_tuple = _load_map(map_yaml)
        suffix = f"{len(controllers)}-controller comparison"
        for goal in goals:
            render_per_goal(df, goal, controllers, img_tuple,
                            traj_dir / f"trajectories_{goal}.pdf",
                            title_suffix=suffix,
                            scope=args.scope, highlight=highlight)
        render_tour_overview(df, goals, controllers, img_tuple,
                             traj_dir / "tour_overview.pdf",
                             label_map=legend_label_map,
                             highlight=highlight, scope=args.scope)
        print(f"[comparison] wrote trajectory overlays ({len(goals)}) "
              f"+ tour_overview.pdf")

    print(f"[comparison] done -> {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
