#!/usr/bin/env python3
"""controller_selection.py — TOPSIS pipeline for the single best trained controller.

The primary, defensible winner-selection method. An equal-weight
composite-z score could not answer "where do the weights come from?",
so this module derives weights objectively from the data
(Shannon-entropy method), ranks controllers by TOPSIS
closeness coefficient ``C* in [0,1]``, cross-checks against the
Pareto front, and runs a sensitivity analysis over alternative
weighting schemes.

Pipeline
--------
0. **Decision matrix** — 18 adaptive trained controllers × 6 criteria
   (per-controller means, NaN-safe):
     - t2_spl              (Efficiency, higher better)
     - t1_mmc_m            (Safety,    higher better, inf-filtered)
     - t1_collision        (Safety,    lower  better)
     - t4_mean_jerk        (Motion,    lower  better)
     - t6_mean_xte_m       (Path,      lower  better)
     - t2_duration_s       (Efficiency time, lower better)

1. **SR > gate** — exclude controllers below ``--sr-gate`` (default 0:
   only excludes 100% failure). With n=3 we keep everything; bump to
   0.5 when n grows.

2. **Correlation check** — Pearson ``|r|`` between every pair of
   criteria; if ``|r| > 0.7`` the criterion with the lower information
   content (smaller std after min-max norm) is dropped. Likely cut:
   t2_duration_s vs t2_spl.

3. **Vector normalisation** — divide each criterion column by its
   L2 norm (the canonical TOPSIS step before weighting). Lower-better
   criteria are flipped to higher-better by ``x -> (max(x) - x + min(x))``
   AFTER normalisation, so the ideal solution is always the column
   maximum.

4. **Entropy weighting** — Shannon-entropy objective weights:
       p_ij  = x_ij / sum_i x_ij
       e_j   = -(1/ln(n)) * sum_i p_ij * ln(p_ij)
       w_j   = (1 - e_j) / sum_j (1 - e_j)
   Highly-discriminating criteria (low entropy across controllers)
   get higher weight. No human numbers.

5. **TOPSIS** —
       v_ij     = w_j * x_norm_ij
       v_plus_j = max_i v_ij   (positive ideal)
       v_minus_j= min_i v_ij   (negative ideal)
       S_plus_i = sqrt(sum_j (v_ij - v_plus_j)^2)
       S_minus_i= sqrt(sum_j (v_ij - v_minus_j)^2)
       C_i      = S_minus_i / (S_plus_i + S_minus_i)
   Rank descending by C_i; top-1 is the recommended controller.

6. **Pareto check** — non-dominated front across the 6 criteria
   (direction-aware). Sanity test: the TOPSIS winner should sit on
   the front. If not, log a warning and discuss in the paper.

7. **Sensitivity analysis** — re-rank under
     (a) entropy weights (baseline),
     (b) equal weights,
     (c) domain-priority weights (safety + outcome heavy),
     (d) entropy +/- 20% per criterion (Monte-Carlo, 200 draws).
   Reports rank-1 stability: fraction of perturbations that keep the
   baseline-top-1 in rank 1.

Outputs (in ``--out`` dir, default ``<results.csv parent>``):
  controller_selection.csv     ranked TOPSIS table + raw criteria
  pareto_front.json            non-dominated controller list + notes
  sensitivity.csv              alt-weighting ranks side by side
  weights.json                 entropy weights + correlation matrix

Usage
-----
    python3 -m vf_robot_utils.analysis.csv_pipeline.controller_selection \\
        --results .../results.csv \\
        [--controllers <list>] \\
        [--sr-gate 0] \\
        [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from vf_robot_utils.analysis.csv_pipeline import (
    METRIC_CATALOG,
    TOPSIS_CRITERIA,
    TRAINED_CONTROLLERS,
    family_of,
)


# Direction lookup built once from METRIC_CATALOG so this module never
# duplicates the +1 / -1 metadata.
_DIR_BY_COL: dict[str, int] = {m.col: m.dir for m in METRIC_CATALOG}
_DROP_INF_BY_COL: dict[str, bool] = {m.col: m.drop_inf for m in METRIC_CATALOG}


# Domain-priority weights for sensitivity analysis. Reflects navigation
# common sense (safety + completion matter more than smoothness/time).
# Sums to 1.0; same ordering as TOPSIS_CRITERIA.
_DOMAIN_WEIGHTS: dict[str, float] = {
    "t2_spl":         0.25,   # efficient success
    "t1_mmc_m":       0.20,   # safety (clearance)
    "t1_collision":   0.25,   # safety (collisions are veto-like)
    "t4_mean_jerk":   0.10,   # motion comfort
    "t6_mean_xte_m":  0.10,   # path tracking
    "t2_duration_s":  0.10,   # time
}


# ---------------------------------------------------------------------------
# Step 0 — Decision matrix
# ---------------------------------------------------------------------------

def _per_controller_mean(
    df: pd.DataFrame, controllers: list[str], col: str, drop_inf: bool,
) -> dict[str, float]:
    """NaN-safe per-controller mean. Excludes inf for clearance metrics."""
    out: dict[str, float] = {}
    for c in controllers:
        s = pd.to_numeric(
            df.loc[df["controller"] == c, col], errors="coerce",
        ).to_numpy(dtype=np.float64)
        s = s[np.isfinite(s) if drop_inf else ~np.isnan(s)]
        out[c] = float(s.mean()) if s.size else float("nan")
    return out


def build_decision_matrix(
    df: pd.DataFrame, controllers: list[str], criteria: list[str],
) -> pd.DataFrame:
    """Return a (controllers x criteria) DataFrame of per-controller means."""
    cols = {}
    for col in criteria:
        cols[col] = _per_controller_mean(
            df, controllers, col, _DROP_INF_BY_COL.get(col, False),
        )
    out = pd.DataFrame(cols, index=controllers)
    out.index.name = "Controller"
    return out


# ---------------------------------------------------------------------------
# Step 1 — Outcome gate
# ---------------------------------------------------------------------------

def apply_sr_gate(
    df: pd.DataFrame, controllers: list[str], sr_gate: float,
) -> list[str]:
    """Return controllers with mean(t2_success) > sr_gate."""
    keep: list[str] = []
    dropped: dict[str, float] = {}
    for c in controllers:
        s = pd.to_numeric(
            df.loc[df["controller"] == c, "t2_success"], errors="coerce",
        ).to_numpy(dtype=np.float64)
        s = s[~np.isnan(s)]
        sr = float(s.mean()) if s.size else 0.0
        if sr > sr_gate:
            keep.append(c)
        else:
            dropped[c] = sr
    if dropped:
        print(f"[selection] SR gate (>{sr_gate}) dropped {len(dropped)}: "
              + ", ".join(f"{c}={v:.2f}" for c, v in dropped.items()))
    return keep


# ---------------------------------------------------------------------------
# Step 2 — Correlation filter
# ---------------------------------------------------------------------------

def correlation_filter(
    matrix: pd.DataFrame, threshold: float = 0.7,
) -> tuple[list[str], np.ndarray, list[tuple[str, str, float]]]:
    """Drop the weaker of any pair with |Pearson r| > threshold.

    "Weaker" = lower stdev after min-max normalisation (less
    information content for downstream weighting).

    Returns
    -------
    kept_cols : list of column names that survived
    corr      : NxN correlation matrix in matrix.columns order
    dropped   : list of (kept_col, dropped_col, r) tuples for the log
    """
    cols = list(matrix.columns)
    sub = matrix.dropna(how="any")
    if sub.shape[0] < 3:
        return cols, np.full((len(cols), len(cols)), np.nan), []
    corr = sub.corr(method="pearson").to_numpy()

    # min-max normalised std as "information content"
    info: dict[str, float] = {}
    for c in cols:
        v = sub[c].to_numpy()
        rng = v.max() - v.min()
        v_n = (v - v.min()) / rng if rng > 0 else v * 0.0
        info[c] = float(v_n.std(ddof=0))

    dropped: list[tuple[str, str, float]] = []
    kept = set(cols)
    for i, j in combinations(range(len(cols)), 2):
        ci, cj = cols[i], cols[j]
        if ci not in kept or cj not in kept:
            continue
        r = corr[i, j]
        if not np.isfinite(r) or abs(r) <= threshold:
            continue
        weak = ci if info[ci] < info[cj] else cj
        strong = cj if weak == ci else ci
        kept.discard(weak)
        dropped.append((strong, weak, float(r)))

    kept_cols = [c for c in cols if c in kept]
    return kept_cols, corr, dropped


# ---------------------------------------------------------------------------
# Step 3 — Normalisation + direction flip
# ---------------------------------------------------------------------------

def _vector_normalize(matrix: pd.DataFrame) -> pd.DataFrame:
    """L2-normalise each column. Imputes column-mean for any remaining NaN
    so TOPSIS arithmetic is well-defined; controllers with column NaN
    receive a neutral value (no advantage, no penalty)."""
    out = matrix.copy()
    for c in out.columns:
        v = out[c].to_numpy(dtype=np.float64)
        # Imputation: column mean over non-NaN entries.
        mu = np.nanmean(v) if np.any(np.isfinite(v)) else 0.0
        v = np.where(np.isnan(v), mu, v)
        norm = np.linalg.norm(v)
        out[c] = v / norm if norm > 0 else v
    return out


def _flip_lower_better(matrix: pd.DataFrame) -> pd.DataFrame:
    """Translate lower-better columns so the column maximum is always
    the ideal point. Applied AFTER vector normalisation to preserve the
    L2 structure."""
    out = matrix.copy()
    for c in out.columns:
        if _DIR_BY_COL.get(c, +1) == -1:
            v = out[c].to_numpy()
            out[c] = (v.max() + v.min()) - v
    return out


# ---------------------------------------------------------------------------
# Step 4 — Entropy weights
# ---------------------------------------------------------------------------

def entropy_weights(matrix: pd.DataFrame) -> np.ndarray:
    """Shannon-entropy weights over a non-negative matrix.

    matrix is the post-normalisation, post-direction-flip table:
    every column is already "higher is better" and >= 0.
    """
    arr = matrix.to_numpy(dtype=np.float64)
    n = arr.shape[0]
    weights = np.zeros(arr.shape[1])
    for j in range(arr.shape[1]):
        col = arr[:, j].copy()
        total = col.sum()
        if total <= 0:
            weights[j] = 0.0
            continue
        p = col / total
        # 0 * log(0) := 0 ; mask out zeros before np.log
        nonzero = p > 0
        p_log = np.zeros_like(p)
        p_log[nonzero] = p[nonzero] * np.log(p[nonzero])
        ent = -p_log.sum() / np.log(n) if n > 1 else 0.0
        weights[j] = max(0.0, 1.0 - ent)
    s = weights.sum()
    return weights / s if s > 0 else np.full(weights.shape, 1.0 / weights.size)


# ---------------------------------------------------------------------------
# Step 5 — TOPSIS
# ---------------------------------------------------------------------------

def topsis(
    matrix: pd.DataFrame, weights: np.ndarray,
) -> pd.DataFrame:
    """Apply weights, find ideal points, compute closeness coefficient.

    Expects a matrix where every column is already higher-is-better
    (post-flip) and L2-normalised.
    """
    weighted = matrix.to_numpy() * weights[np.newaxis, :]
    pis = weighted.max(axis=0)   # positive ideal
    nis = weighted.min(axis=0)   # negative ideal
    s_plus  = np.sqrt(((weighted - pis) ** 2).sum(axis=1))
    s_minus = np.sqrt(((weighted - nis) ** 2).sum(axis=1))
    denom = s_plus + s_minus
    closeness = np.where(denom > 0, s_minus / denom, 0.0)
    out = pd.DataFrame(
        {"Controller": matrix.index, "C_star": closeness,
         "S_plus": s_plus, "S_minus": s_minus},
    )
    out = out.sort_values("C_star", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


# ---------------------------------------------------------------------------
# Step 6 — Pareto front
# ---------------------------------------------------------------------------

def pareto_front(matrix: pd.DataFrame) -> list[str]:
    """Non-dominated set across all columns. Expects a higher-is-better
    (post-flip) matrix."""
    controllers = list(matrix.index)
    arr = matrix.to_numpy(dtype=np.float64)
    non_dominated: list[str] = []
    for i in range(arr.shape[0]):
        dominated = False
        for k in range(arr.shape[0]):
            if k == i:
                continue
            # k dominates i if k >= i in all dims and > i in at least one.
            geq = arr[k] >= arr[i] - 1e-12
            grt = arr[k] >  arr[i] + 1e-12
            if geq.all() and grt.any():
                dominated = True
                break
        if not dominated:
            non_dominated.append(controllers[i])
    return non_dominated


# ---------------------------------------------------------------------------
# Step 7 — Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    matrix: pd.DataFrame,
    baseline_weights: np.ndarray,
    n_perturbations: int = 200,
    pct: float = 0.20,
    rng_seed: int = 17,
) -> tuple[pd.DataFrame, dict]:
    """Re-rank under alternative weights + Monte-Carlo entropy perturbations.

    Returns
    -------
    table : DataFrame with rank under each weighting scheme
    stats : dict with rank-1 stability fraction + Spearman vs baseline
    """
    cols = list(matrix.columns)
    rng = np.random.default_rng(rng_seed)

    schemes: dict[str, np.ndarray] = {
        "entropy":         baseline_weights,
        "equal":           np.full(len(cols), 1.0 / len(cols)),
        "domain_priority": np.array(
            [_DOMAIN_WEIGHTS.get(c, 1.0 / len(cols)) for c in cols],
        ),
    }
    # Re-normalise domain_priority in case some criteria were filtered.
    s = schemes["domain_priority"].sum()
    if s > 0:
        schemes["domain_priority"] = schemes["domain_priority"] / s

    ranks_by_scheme: dict[str, pd.Series] = {}
    for name, w in schemes.items():
        tbl = topsis(matrix, w).set_index("Controller")
        ranks_by_scheme[name] = tbl["rank"]

    # Monte-Carlo perturbations of baseline_weights.
    baseline_top1 = topsis(matrix, baseline_weights).iloc[0]["Controller"]
    top1_hits = 0
    for _ in range(n_perturbations):
        noise = 1.0 + rng.uniform(-pct, pct, size=len(cols))
        w = baseline_weights * noise
        w = w / w.sum() if w.sum() > 0 else baseline_weights
        top = topsis(matrix, w).iloc[0]["Controller"]
        if top == baseline_top1:
            top1_hits += 1

    table = pd.DataFrame(ranks_by_scheme)
    table.index.name = "Controller"
    table = table.sort_values("entropy")

    # Spearman of alt-scheme ranks vs entropy ranks.
    base = ranks_by_scheme["entropy"].rank()
    spearman = {
        name: float(s.rank().corr(base, method="spearman"))
        for name, s in ranks_by_scheme.items()
    }
    stats = {
        "baseline_top1": baseline_top1,
        "perturbation_top1_stability": float(top1_hits) / n_perturbations,
        "spearman_vs_entropy": spearman,
        "n_perturbations": int(n_perturbations),
        "perturbation_pct": float(pct),
    }
    return table, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_selection(
    df: pd.DataFrame,
    controllers: list[str],
    criteria: list[str],
    sr_gate: float,
) -> dict:
    """End-to-end pipeline; returns all artefacts in a single dict."""
    # 0. Decision matrix on the raw criteria.
    raw = build_decision_matrix(df, controllers, criteria)

    # 1. SR gate.
    gated = apply_sr_gate(df, controllers, sr_gate)
    if not gated:
        raise SystemExit(
            f"[selection] all controllers failed SR>{sr_gate} gate. "
            "Either there's no successful run yet, or the gate is too high.",
        )
    raw = raw.loc[gated]

    # 2. Correlation filter.
    kept_cols, corr_matrix, corr_dropped = correlation_filter(raw, 0.7)
    if corr_dropped:
        for s, d, r in corr_dropped:
            print(f"[selection] correlation drop: keep={s}  drop={d}  r={r:+.2f}")
    raw_kept = raw[kept_cols]

    # 3. Normalise + direction flip.
    normed = _vector_normalize(raw_kept)
    flipped = _flip_lower_better(normed)

    # 4. Entropy weights on the flipped matrix.
    w = entropy_weights(flipped)
    weights = dict(zip(kept_cols, w.tolist()))

    # 5. TOPSIS.
    ranking = topsis(flipped, w)

    # 6. Pareto cross-check.
    front = pareto_front(flipped)
    topsis_winner = ranking.iloc[0]["Controller"]
    winner_on_front = topsis_winner in front

    # 7. Sensitivity.
    sens_table, sens_stats = sensitivity_analysis(flipped, w)

    return {
        "raw_matrix":     raw_kept,
        "normed_matrix":  flipped,
        "weights":        weights,
        "correlation":    {
            "columns": list(raw.columns),
            "matrix":  corr_matrix.tolist(),
            "dropped": [
                {"kept": s, "dropped": d, "r": r}
                for s, d, r in corr_dropped
            ],
        },
        "ranking":        ranking,
        "pareto_front":   front,
        "winner_on_front": winner_on_front,
        "sensitivity":    sens_table,
        "sensitivity_stats": sens_stats,
        "gated_controllers": gated,
        "kept_criteria":     kept_cols,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_results_csv() -> Path:
    from vf_robot_utils.constants import EVALUATION_ROOT
    candidates = sorted(
        Path(EVALUATION_ROOT).rglob("_aggregate/*/results.csv")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No results.csv under {EVALUATION_ROOT} — "
            "run aggregate_csv first.",
        )
    return candidates[-1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=None)
    p.add_argument(
        "--controllers", nargs="*", default=None,
        help="Default: 18 adaptive (oraclewt_*/rawwt_*). Pass any subset "
             "of ALL_CONTROLLERS to widen the search.",
    )
    p.add_argument(
        "--criteria", nargs="*", default=None,
        help="Override TOPSIS_CRITERIA (column names). Default: csv_pipeline.TOPSIS_CRITERIA.",
    )
    p.add_argument(
        "--sr-gate", type=float, default=0.0,
        help="Drop controllers with mean(t2_success) <= this value. "
             "Default 0.0 only drops total-failures. Use 0.5 when n grows.",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    results_csv = args.results or _find_results_csv()
    out_dir = args.out or results_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    if "timestamp" in df.columns:
        df = (df.sort_values("timestamp")
                .drop_duplicates(subset=["controller", "goal_folder"],
                                 keep="last")
                .reset_index(drop=True))

    controllers = args.controllers or [
        c for c in TRAINED_CONTROLLERS if c.startswith(("oraclewt_", "rawwt_"))
    ]
    criteria = args.criteria or list(TOPSIS_CRITERIA)

    print(f"[selection] results={results_csv}")
    print(f"[selection] controllers={len(controllers)}  criteria={criteria}")

    result = run_selection(df, controllers, criteria, args.sr_gate)

    # ── controller_selection.csv: ranked TOPSIS + raw criteria ──────────
    rank = result["ranking"].copy()
    rank["Family"] = [family_of(c) for c in rank["Controller"]]
    raw_named = result["raw_matrix"].reset_index()
    rank = rank.merge(raw_named, on="Controller", how="left")
    rank["winner_on_pareto_front"] = (
        rank["Controller"] == result["ranking"].iloc[0]["Controller"]
    ) & result["winner_on_front"]
    out_csv = out_dir / "controller_selection.csv"
    rank.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"[selection] wrote {out_csv}")

    # ── pareto_front.json ───────────────────────────────────────────────
    pareto_path = out_dir / "pareto_front.json"
    pareto_path.write_text(json.dumps({
        "front": result["pareto_front"],
        "winner": result["ranking"].iloc[0]["Controller"],
        "winner_on_front": bool(result["winner_on_front"]),
        "criteria": result["kept_criteria"],
    }, indent=2))
    print(f"[selection] wrote {pareto_path}")

    # ── sensitivity.csv ─────────────────────────────────────────────────
    sens = result["sensitivity"].reset_index()
    sens["Family"] = [family_of(c) for c in sens["Controller"]]
    sens_path = out_dir / "sensitivity.csv"
    sens.to_csv(sens_path, index=False)
    print(f"[selection] wrote {sens_path}")

    # ── weights.json (entropy weights + correlation matrix) ─────────────
    weights_path = out_dir / "weights.json"
    weights_path.write_text(json.dumps({
        "entropy_weights":   result["weights"],
        "kept_criteria":     result["kept_criteria"],
        "correlation":       result["correlation"],
        "sensitivity_stats": result["sensitivity_stats"],
        "gated_controllers": result["gated_controllers"],
        "sr_gate":           args.sr_gate,
    }, indent=2))
    print(f"[selection] wrote {weights_path}")

    # ── headline ────────────────────────────────────────────────────────
    print()
    print(f"[selection] winner: {result['ranking'].iloc[0]['Controller']}  "
          f"C*={result['ranking'].iloc[0]['C_star']:.4f}  "
          f"on_pareto_front={result['winner_on_front']}")
    print(f"[selection] sensitivity top-1 stability: "
          f"{result['sensitivity_stats']['perturbation_top1_stability']:.2%} "
          f"(n={result['sensitivity_stats']['n_perturbations']}, "
          f"+/-{int(result['sensitivity_stats']['perturbation_pct']*100)}%)")
    print(result["ranking"].head(5)[["Controller", "C_star", "rank"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
