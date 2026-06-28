"""Statistical helpers for evaluation aggregation.

Primitives, each independently importable:

- ``bootstrap_ci``           percentile or BCa bootstrap CI.
- ``wilson_ci``              closed-form Wilson CI for a proportion.
- ``newcombe_wilson_diff_ci`` Newcombe-Wilson CI for diff of proportions.
- ``cliffs_delta``           non-parametric effect size in [-1, +1].
- ``cliffs_delta_ci``        bootstrap CI for Cliff's delta.
- ``cliffs_delta_magnitude`` Romano et al. 2006 magnitude classes.
- ``adjust_pvalues``         Holm / BH multiple-comparison correction.
- ``friedman_with_nemenyi``  Friedman + Nemenyi post-hoc with CD.
- ``paired_wilcoxon``        paired Wilcoxon with optional correction.
- ``mann_whitney_per_controller`` pairwise unpaired Mann-Whitney.

scipy / statsmodels are imported lazily inside each function so this
module stays importable when those packages are absent (the function
then raises a clear ImportError when called). Wilson and Cliff's delta
do NOT require scipy.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    B: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    method: str = "percentile",
) -> tuple[float, float]:
    """Bootstrap (lo, hi) confidence interval for ``statistic``.

    ``method='percentile'`` (default) runs an in-house resampler; this is
    the path used by aggregation code so the function stays importable
    without scipy.

    ``method='bca'`` delegates to ``scipy.stats.bootstrap`` with
    bias-correction and acceleration. Bias-corrected and accelerated
    CIs are the default recommendation for skewed distributions and
    small N. Falls back to percentile if scipy is unavailable.

    Returns (nan, nan) for empty input.
    """
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        v = float(statistic(a))
        return v, v

    if method not in ("percentile", "bca"):
        raise ValueError(
            f"method must be 'percentile' or 'bca', got {method!r}"
        )

    if method == "bca":
        try:
            from scipy.stats import bootstrap as _scipy_bootstrap
        except ImportError:
            method = "percentile"
        else:
            res = _scipy_bootstrap(
                (a,),
                statistic=statistic,
                n_resamples=B,
                confidence_level=1.0 - alpha,
                method="BCa",
                random_state=seed,
            )
            ci = res.confidence_interval
            return float(ci.low), float(ci.high)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(B, a.size))
    samples = a[idx]
    stats_arr = np.apply_along_axis(statistic, 1, samples)
    lo = float(np.quantile(stats_arr, alpha / 2.0))
    hi = float(np.quantile(stats_arr, 1.0 - alpha / 2.0))
    return lo, hi


# ── Wilson CI for proportions ────────────────────────────────────────────────

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Closed-form Wilson score interval for a binomial proportion.

    More accurate than the normal-approximation interval for small n or
    extreme p, especially at p=0 or p=n. Wilson never produces bounds
    outside [0, 1].

    Reference: Wilson 1927, Brown/Cai/DasGupta 2001.
    """
    if n <= 0:
        return float("nan"), float("nan")
    if k < 0 or k > n:
        raise ValueError(f"k={k} must be in [0, n={n}]")

    # Two-sided z-score for (1-alpha) coverage.
    z = _normal_inv_cdf(1.0 - alpha / 2.0)
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    se = math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    half = (z * se) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return lo, hi


def newcombe_wilson_diff_ci(
    k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05,
) -> tuple[float, float]:
    """Newcombe-Wilson CI for the difference of two independent proportions.

    Method 10 from Newcombe 1998 — combines the two single-proportion
    Wilson CIs in a way that respects [-1, +1] bounds and behaves well
    at extreme proportions.
    """
    l1, u1 = wilson_ci(k1, n1, alpha=alpha)
    l2, u2 = wilson_ci(k2, n2, alpha=alpha)
    p1 = k1 / n1 if n1 > 0 else float("nan")
    p2 = k2 / n2 if n2 > 0 else float("nan")
    if not (math.isfinite(p1) and math.isfinite(p2)):
        return float("nan"), float("nan")

    delta = math.sqrt(((p1 - l1) ** 2) + ((u2 - p2) ** 2))
    epsilon = math.sqrt(((u1 - p1) ** 2) + ((p2 - l2) ** 2))
    diff = p1 - p2
    return max(-1.0, diff - delta), min(1.0, diff + epsilon)


def _normal_inv_cdf(p: float) -> float:
    """Standard-normal inverse CDF without scipy.

    Beasley-Springer-Moro approximation; absolute error < 4.5e-4.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p={p} must lie in (0, 1)")
    # Beasley-Springer-Moro coefficients.
    a = (-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00)
    d = (7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00)
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5] \
            / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q \
            / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) \
        / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


# ── Cliff's delta ────────────────────────────────────────────────────────────

def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta — non-parametric effect size in [-1, +1].

    delta = (#(x > y) - #(x < y)) / (n_x * n_y).

    Positive delta means x tends to be greater than y.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Vectorised pairwise comparison.
    diff = a[:, None] - b[None, :]
    n_pos = float(np.count_nonzero(diff > 0.0))
    n_neg = float(np.count_nonzero(diff < 0.0))
    return (n_pos - n_neg) / (a.size * b.size)


def cliffs_delta_ci(
    x: Sequence[float], y: Sequence[float],
    B: int = 10_000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap percentile CI for Cliff's delta."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    deltas = np.empty(B, dtype=np.float64)
    for i in range(B):
        ai = rng.integers(0, a.size, size=a.size)
        bi = rng.integers(0, b.size, size=b.size)
        deltas[i] = cliffs_delta(a[ai], b[bi])
    lo = float(np.quantile(deltas, alpha / 2.0))
    hi = float(np.quantile(deltas, 1.0 - alpha / 2.0))
    return lo, hi


def cliffs_delta_magnitude(d: float) -> str:
    """Romano et al. 2006 magnitude thresholds for |delta|.

    < 0.147 negligible, < 0.33 small, < 0.474 medium, >= 0.474 large.
    """
    if not math.isfinite(d):
        return "undefined"
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.330:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


# ── Multiple-comparison correction ──────────────────────────────────────────

def adjust_pvalues(pvals: Sequence[float], method: str = "bh") -> list[float]:
    """Adjust p-values for multiple comparisons.

    method='bh'   — Benjamini-Hochberg (false discovery rate).
    method='holm' — Holm-Bonferroni (family-wise error rate).

    Uses ``statsmodels.stats.multitest.multipletests`` under the hood when
    available; falls back to a small in-house implementation otherwise so
    the figure pipeline keeps working without statsmodels.
    """
    if method not in ("bh", "holm"):
        raise ValueError(f"method must be 'bh' or 'holm', got {method!r}")
    p = np.asarray(pvals, dtype=np.float64)
    if p.size == 0:
        return []
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")

    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return _adjust_pvalues_inhouse(p.tolist(), method)

    sm_method = {"bh": "fdr_bh", "holm": "holm"}[method]
    _, p_adj, _, _ = multipletests(p, method=sm_method)
    return p_adj.tolist()


def _adjust_pvalues_inhouse(p: list[float], method: str) -> list[float]:
    """Fallback BH / Holm implementation with no external deps."""
    n = len(p)
    order = sorted(range(n), key=lambda i: p[i])
    sorted_p = [p[i] for i in order]
    out = [0.0] * n

    if method == "holm":
        # Holm-Bonferroni step-down.
        prev = 0.0
        for rank, i in enumerate(order):
            adj = (n - rank) * sorted_p[rank]
            adj = min(adj, 1.0)
            adj = max(adj, prev)
            out[i] = adj
            prev = adj
    else:
        # BH step-up: q_i = min_{j>=i} (n / rank_j) * p_j.
        adjusted_sorted = [0.0] * n
        running_min = 1.0
        for rank in range(n - 1, -1, -1):
            cand = (n / (rank + 1)) * sorted_p[rank]
            running_min = min(running_min, cand)
            adjusted_sorted[rank] = min(running_min, 1.0)
        for rank, i in enumerate(order):
            out[i] = adjusted_sorted[rank]
    return out


# ── Friedman + Nemenyi ──────────────────────────────────────────────────────

def friedman_with_nemenyi(
    df: pd.DataFrame, metric: str,
    scenario_col: str = "scenario", controller_col: str = "controller",
    alpha: float = 0.05,
) -> dict:
    """Friedman omnibus + Nemenyi post-hoc on per-scenario summaries.

    Procedure:
      1. Pivot ``df`` so rows are scenarios (blocks) and columns are
         controllers (treatments). Each cell holds the scenario-mean of
         ``metric``.
      2. Run Friedman's test on the (n_scenarios x n_controllers) matrix.
      3. If H0 is rejected, compute average ranks per controller and
         Nemenyi's critical difference at level ``alpha``:

             CD = q_alpha * sqrt(K (K+1) / (6 N))

         where K = number of controllers, N = number of scenarios.

    Returns a dict with:
      - ``friedman_stat``, ``friedman_p``, ``n_scenarios``, ``n_controllers``
      - ``avg_ranks``    (controller -> mean rank)
      - ``cd``           (Nemenyi critical difference, NaN if N or K too small)
      - ``nemenyi``      (DataFrame: ctrl_i, ctrl_j, |rank_i - rank_j|, sig)
    """
    try:
        from scipy import stats as _stats
    except ImportError as e:
        raise ImportError(
            "friedman_with_nemenyi requires scipy. Install scipy."
        ) from e

    pivot = df.pivot_table(
        index=scenario_col, columns=controller_col,
        values=metric, aggfunc="mean",
    ).dropna(axis=0, how="any")
    K = pivot.shape[1]
    N = pivot.shape[0]
    if N < 2 or K < 3:
        return {
            "friedman_stat": float("nan"), "friedman_p": float("nan"),
            "n_scenarios": int(N), "n_controllers": int(K),
            "avg_ranks": {}, "cd": float("nan"),
            "nemenyi": pd.DataFrame(
                columns=["ctrl_i", "ctrl_j", "rank_diff", "sig"]
            ),
        }

    cols_arr = [pivot[c].to_numpy() for c in pivot.columns]
    res = _stats.friedmanchisquare(*cols_arr)

    ranks = pivot.rank(axis=1).mean(axis=0)  # mean rank per controller
    cd = _nemenyi_critical_difference(K, N, alpha=alpha)
    cd_finite = math.isfinite(cd)

    rows = []
    cols = list(pivot.columns)
    for i, ci in enumerate(cols):
        for cj in cols[i + 1:]:
            rd = float(abs(ranks[ci] - ranks[cj]))
            rows.append({
                "ctrl_i": ci, "ctrl_j": cj,
                "rank_diff": rd,
                "sig": rd > cd if cd_finite else False,
            })

    return {
        "friedman_stat": float(res.statistic),
        "friedman_p": float(res.pvalue),
        "n_scenarios": int(N), "n_controllers": int(K),
        "avg_ranks": {c: float(r) for c, r in ranks.items()},
        "cd": float(cd),
        "nemenyi": pd.DataFrame(rows),
    }


# Studentised-range approximation for the q_alpha used in Nemenyi.
# Tabulated by Demsar 2006 for alpha=0.05 and alpha=0.10. Values picked
# from k=2..10; truncate to k=10 by replicating the q at k=10 for larger
# k (figure-quality only — beyond k=10 the Friedman+Nemenyi route is
# anyway not the right tool).
_NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164,
}
_NEMENYI_Q10 = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.460, 6: 2.589, 7: 2.693,
    8: 2.780, 9: 2.855, 10: 2.920,
}


def _nemenyi_critical_difference(K: int, N: int, alpha: float = 0.05) -> float:
    table = _NEMENYI_Q05 if abs(alpha - 0.05) < 1e-9 else (
        _NEMENYI_Q10 if abs(alpha - 0.10) < 1e-9 else None
    )
    if table is None or K < 2 or N < 2:
        return float("nan")
    q = table.get(K, table[10])
    return q * math.sqrt(K * (K + 1) / (6.0 * N))


# ── Paired Wilcoxon ──────────────────────────────────────────────────────────

def paired_wilcoxon(
    df: pd.DataFrame,
    metric: str,
    ctrl_a: str,
    ctrl_b: str,
    pair_key: tuple = ("planner", "run_id", "leg_id"),
    correct: str | None = None,
) -> dict:
    """Paired Wilcoxon signed-rank between two controllers.

    Inner-join ``df`` on ``pair_key`` for the two controllers, then run
    Wilcoxon on (a - b).

    ``correct`` is unused for a single comparison; the parameter exists
    so callers that loop over many pairs and pass the same value get a
    consistent signature. For a single pair the raw and adjusted p are
    identical. Use ``adjust_pvalues`` for the actual MC step at the
    figure-pipeline level.

    Returns dict with keys ``n_pairs``, ``statistic``, ``pvalue``,
    ``pvalue_adj``, ``median_diff``.
    """
    needed = list(pair_key) + ["controller", metric]
    sub = df[needed].copy()
    sub = sub[sub["controller"].isin([ctrl_a, ctrl_b])]
    sub = sub.dropna(subset=[metric])
    sub = sub[np.isfinite(sub[metric])]

    a = sub[sub["controller"] == ctrl_a].drop(columns=["controller"])
    b = sub[sub["controller"] == ctrl_b].drop(columns=["controller"])
    merged = a.merge(b, on=list(pair_key), suffixes=("_a", "_b"))

    n_pairs = len(merged)
    if n_pairs == 0:
        return {"n_pairs": 0, "statistic": float("nan"),
                "pvalue": float("nan"), "pvalue_adj": float("nan"),
                "median_diff": float("nan")}

    diffs = (merged[f"{metric}_a"] - merged[f"{metric}_b"]).to_numpy()
    median_diff = float(np.median(diffs))

    if np.allclose(diffs, 0.0):
        return {"n_pairs": n_pairs, "statistic": 0.0,
                "pvalue": 1.0, "pvalue_adj": 1.0,
                "median_diff": median_diff}

    try:
        from scipy import stats as _stats
    except ImportError as e:
        raise ImportError(
            "paired_wilcoxon requires scipy. Install scipy or remove this "
            "test from your figure pipeline."
        ) from e

    res = _stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    pvalue = float(res.pvalue)
    pvalue_adj = pvalue
    if correct in ("bh", "holm"):
        # Single comparison — adjustment is the identity. Returned for
        # symmetry with batched callers that DO accumulate p-values.
        pvalue_adj = adjust_pvalues([pvalue], method=correct)[0]
    return {
        "n_pairs": n_pairs,
        "statistic": float(res.statistic),
        "pvalue": pvalue,
        "pvalue_adj": pvalue_adj,
        "median_diff": median_diff,
    }


# ── Pairwise Mann-Whitney across controllers ────────────────────────────────

def mann_whitney_per_controller(
    df: pd.DataFrame,
    metric: str,
    controllers: list[str],
    correct: str | None = None,
) -> pd.DataFrame:
    """Pairwise (unpaired) Mann-Whitney U test across the given controllers.

    Returns a long DataFrame with columns
    ``[ctrl_i, ctrl_j, n_i, n_j, U, pvalue, pvalue_adj, median_i, median_j]``,
    one row per ordered pair (i < j).

    ``correct='bh' | 'holm'`` applies a multiple-comparison correction
    across all returned pairs; ``None`` leaves ``pvalue_adj == pvalue``.
    """
    try:
        from scipy import stats as _stats
    except ImportError as e:
        raise ImportError(
            "mann_whitney_per_controller requires scipy."
        ) from e

    rows = []
    for i, ci in enumerate(controllers):
        xi = df.loc[df["controller"] == ci, metric].to_numpy(dtype=float)
        xi = xi[np.isfinite(xi)]
        for cj in controllers[i + 1:]:
            xj = df.loc[df["controller"] == cj, metric].to_numpy(dtype=float)
            xj = xj[np.isfinite(xj)]

            if xi.size == 0 or xj.size == 0:
                med_i = float("nan") if xi.size == 0 else float(np.median(xi))
                med_j = float("nan") if xj.size == 0 else float(np.median(xj))
                rows.append({
                    "ctrl_i": ci, "ctrl_j": cj,
                    "n_i": int(xi.size), "n_j": int(xj.size),
                    "U": float("nan"), "pvalue": float("nan"),
                    "median_i": med_i, "median_j": med_j,
                })
                continue

            res = _stats.mannwhitneyu(xi, xj, alternative="two-sided")
            rows.append({
                "ctrl_i": ci, "ctrl_j": cj,
                "n_i": int(xi.size), "n_j": int(xj.size),
                "U": float(res.statistic), "pvalue": float(res.pvalue),
                "median_i": float(np.median(xi)),
                "median_j": float(np.median(xj)),
            })

    out = pd.DataFrame(rows)
    if correct in ("bh", "holm") and not out.empty:
        raw = out["pvalue"].fillna(1.0).tolist()
        out["pvalue_adj"] = adjust_pvalues(raw, method=correct)
    else:
        out["pvalue_adj"] = out["pvalue"]
    return out
