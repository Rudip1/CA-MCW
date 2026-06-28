"""statistical correctness tests.

Run via:
    colcon test --packages-select vf_robot_utils --pytest-args -k test_stats
"""
from __future__ import annotations

from pathlib import Path  # noqa: F401  (kept for tmp_path fixture typing)

import numpy as np
import pandas as pd
import pytest

from vf_robot_utils.analysis import aggregate as agg
from vf_robot_utils.analysis.statistical_tests import (
    adjust_pvalues,
    bootstrap_ci,
    cliffs_delta,
    cliffs_delta_ci,
    cliffs_delta_magnitude,
    friedman_with_nemenyi,
    mann_whitney_per_controller,
    newcombe_wilson_diff_ci,
    paired_wilcoxon,
    wilson_ci,
)


# ── 1. Sample std vs population std ──────────────────────────────────────────

def test_summary_uses_sample_std(tmp_path: Path) -> None:
    """The new summary writer must use ddof=1 (sample std), not ddof=0.

    For x = [1, 2, 3, 4, 5]:
      population std (ddof=0) ≈ 1.4142
      sample std     (ddof=1) ≈ 1.5811
    """
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    row = agg._summary_row('demo_metric', values)

    # row layout: [name, n, mean, std, sem, ci95_lo, ci95_hi, min, max]
    assert row[0] == 'demo_metric'
    assert row[1] == 5
    assert float(row[2]) == pytest.approx(3.0, abs=1e-6)

    sample_std = float(row[3])
    assert sample_std == pytest.approx(1.5811, abs=1e-3), (
        f'expected sample std ≈ 1.5811 (ddof=1), got {sample_std}; '
        'aggregate is still using population std (ddof=0).'
    )
    # Sanity: must NOT match population std.
    assert sample_std != pytest.approx(1.4142, abs=1e-3)


# ── 2. SEM column populated correctly ────────────────────────────────────────

def test_summary_sem_column() -> None:
    # row values are written with f'{:.4f}' precision, so check at 1e-3.
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    row = agg._summary_row('demo_metric', values)
    sample_std = float(row[3])
    sem = float(row[4])
    assert sem == pytest.approx(sample_std / np.sqrt(5), abs=1e-3)
    assert sem == pytest.approx(0.7071, abs=1e-3)


# ── 3. Bootstrap CI brackets the mean of a clean Gaussian ────────────────────

def test_bootstrap_ci_brackets_gaussian_mean() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=10.0, scale=2.0, size=200)
    lo, hi = bootstrap_ci(x, B=2000, seed=1)
    assert lo <= 10.0 <= hi, (
        f'true mean 10.0 not in bootstrap CI [{lo:.3f}, {hi:.3f}]'
    )
    # CI on a 200-sample Gaussian should be tight.
    assert (hi - lo) < 1.0


# ── 4. Paired Wilcoxon detects a clear pair ──────────────────────────────────

def test_paired_wilcoxon_detects_clear_pair() -> None:
    pytest.importorskip('scipy')
    rng = np.random.default_rng(42)
    n = 30

    rows = []
    for run_id in range(n):
        base = float(rng.normal(loc=0.5, scale=0.05))
        rows.append({
            'planner': 'NavFn', 'run_id': run_id, 'leg_id': 0,
            'controller': 'vf_inference',
            't1_mmc_m': base + 0.20,   # +0.2 m more clearance
        })
        rows.append({
            'planner': 'NavFn', 'run_id': run_id, 'leg_id': 0,
            'controller': 'vf_imitation',
            't1_mmc_m': base,
        })
    df = pd.DataFrame(rows)

    res = paired_wilcoxon(df, 't1_mmc_m', 'vf_inference', 'vf_imitation')
    assert res['n_pairs'] == n
    assert res['median_diff'] > 0.15
    assert res['pvalue'] < 0.01, f'expected p<0.01, got {res["pvalue"]}'


# ── 5. Paired Wilcoxon: identical samples → p > 0.5 ──────────────────────────

def test_paired_wilcoxon_identical_pair_high_p() -> None:
    pytest.importorskip('scipy')
    rng = np.random.default_rng(7)
    n = 25

    rows = []
    for run_id in range(n):
        v = float(rng.normal(loc=1.0, scale=0.1))
        rows.append({'planner': 'NavFn', 'run_id': run_id, 'leg_id': 0,
                     'controller': 'A', 'metric': v})
        rows.append({'planner': 'NavFn', 'run_id': run_id, 'leg_id': 0,
                     'controller': 'B', 'metric': v})
    df = pd.DataFrame(rows)

    res = paired_wilcoxon(df, 'metric', 'A', 'B')
    assert res['n_pairs'] == n
    assert res['median_diff'] == 0.0
    assert res['pvalue'] == pytest.approx(1.0, abs=1e-9), (
        f'identical samples should give p=1.0, got {res["pvalue"]}'
    )


# ── 6. Summary header has the new columns ────────────────────────────────────

def test_summary_header_extended() -> None:
    """Regression guard: any consumer of summary.csv expects this header."""
    assert agg._SUMMARY_HEADER == [
        'metric', 'n', 'mean', 'std', 'sem',
        'ci95_lo', 'ci95_hi', 'min', 'max',
    ]


# ── 7. _summary_row handles empty / single-element / nonfinite input ────────

def test_summary_row_edge_cases() -> None:
    # Empty
    row = agg._summary_row('m', np.array([]))
    assert row[1] == 0
    assert row[2] == ''        # blank cells for empty inputs
    # Single value
    row = agg._summary_row('m', np.array([7.0]))
    assert row[1] == 1
    assert float(row[2]) == 7.0
    assert float(row[3]) == 0.0  # std degenerate → 0
    assert float(row[4]) == 0.0  # SEM degenerate → 0
    assert float(row[5]) == 7.0  # CI collapses to the point
    assert float(row[6]) == 7.0
    # Non-finite stripped
    row = agg._summary_row('m', np.array([1.0, 2.0, np.inf, np.nan, 3.0]))
    assert row[1] == 3
    assert float(row[2]) == pytest.approx(2.0, abs=1e-6)


# ── E8.1 Wilson CI ──────────────────────────────────────────────────────────

def test_wilson_ci_brackets_exact_for_typical_case() -> None:
    """k=8, n=10 → Wilson 95% CI ≈ (0.4901, 0.9433) per Wikipedia table."""
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4901, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_wilson_ci_p_zero_no_lower_overflow() -> None:
    """k=0, n=10 — Wilson must keep lower bound at 0, not negative."""
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    # Upper bound from k=0,n=10 ≈ 0.2775.
    assert hi == pytest.approx(0.2775, abs=1e-3)


def test_wilson_ci_p_one_no_upper_overflow() -> None:
    """k=n — Wilson must keep upper bound at 1, not above."""
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0
    # Lower bound from k=10,n=10 ≈ 0.7225.
    assert lo == pytest.approx(0.7225, abs=1e-3)


def test_wilson_ci_n_one_extreme() -> None:
    """n=1 with k=1 → highly uncertain CI (≈0.21..1.0)."""
    lo, hi = wilson_ci(1, 1)
    assert hi == 1.0
    assert lo < 0.25


def test_wilson_ci_large_n_narrows() -> None:
    """n=100, k=50 → CI tight around 0.5."""
    lo, hi = wilson_ci(50, 100)
    assert (hi - lo) < 0.20
    assert 0.40 < lo < hi < 0.60


def test_wilson_ci_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        wilson_ci(11, 10)
    with pytest.raises(ValueError):
        wilson_ci(-1, 10)
    lo, hi = wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_newcombe_wilson_diff_zero_for_identical_proportions() -> None:
    lo, hi = newcombe_wilson_diff_ci(50, 100, 50, 100)
    assert lo < 0.0 < hi
    assert abs((lo + hi) / 2.0) < 0.02   # centred on zero


def test_newcombe_wilson_diff_excludes_zero_when_clearly_different() -> None:
    """80/100 vs 20/100 — CI should clearly exclude 0."""
    lo, hi = newcombe_wilson_diff_ci(80, 100, 20, 100)
    assert lo > 0.40
    assert hi < 0.80


def test_newcombe_wilson_diff_bounds_in_unit_range() -> None:
    """CI for difference of proportions must lie in [-1, 1]."""
    lo, hi = newcombe_wilson_diff_ci(99, 100, 1, 100)
    assert -1.0 <= lo <= hi <= 1.0


def test_newcombe_wilson_diff_handles_zero_n() -> None:
    lo, hi = newcombe_wilson_diff_ci(0, 0, 5, 10)
    assert np.isnan(lo) and np.isnan(hi)


# ── E8.2 Cliff's delta ──────────────────────────────────────────────────────

def test_cliffs_delta_disjoint_sets_extreme() -> None:
    """All-of-x > all-of-y → delta = +1."""
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)


def test_cliffs_delta_identical_distributions_zero() -> None:
    """Equal samples → delta = 0."""
    x = [1.0, 2.0, 3.0, 4.0]
    assert cliffs_delta(x, x) == pytest.approx(0.0, abs=1e-12)


def test_cliffs_delta_known_case() -> None:
    """[1,2,3] vs [2,3,4] — by-hand: 1 win, 6 losses, 2 ties out of 9.
    delta = (1 - 6) / 9 = -5/9.
    """
    d = cliffs_delta([1, 2, 3], [2, 3, 4])
    assert d == pytest.approx(-5.0 / 9.0, abs=1e-9)


def test_cliffs_delta_empty_input_nan() -> None:
    assert np.isnan(cliffs_delta([], [1, 2, 3]))


def test_cliffs_delta_magnitude_thresholds() -> None:
    assert cliffs_delta_magnitude(0.10) == "negligible"
    assert cliffs_delta_magnitude(0.20) == "small"
    assert cliffs_delta_magnitude(0.40) == "medium"
    assert cliffs_delta_magnitude(0.50) == "large"
    assert cliffs_delta_magnitude(-0.50) == "large"
    assert cliffs_delta_magnitude(float("nan")) == "undefined"


def test_cliffs_delta_ci_brackets_zero_when_same_distribution() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=80)
    y = rng.normal(0, 1, size=80)
    lo, hi = cliffs_delta_ci(x, y, B=500, seed=1)
    assert lo < 0.0 < hi


def test_cliffs_delta_ci_excludes_zero_when_clear_shift() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 1.0, size=80)
    y = rng.normal(0.0, 1.0, size=80)
    lo, hi = cliffs_delta_ci(x, y, B=500, seed=2)
    assert lo > 0.0


# ── E8.3 BCa bootstrap option ───────────────────────────────────────────────

def test_bootstrap_bca_matches_percentile_for_symmetric() -> None:
    """Symmetric data: BCa and percentile differ only marginally."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    x = rng.normal(loc=10.0, scale=2.0, size=200)
    p_lo, p_hi = bootstrap_ci(x, B=2000, seed=1, method="percentile")
    b_lo, b_hi = bootstrap_ci(x, B=2000, seed=1, method="bca")
    assert abs(b_lo - p_lo) < 0.5
    assert abs(b_hi - p_hi) < 0.5


def test_bootstrap_invalid_method_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0, 3.0], method="bogus")


# ── E8.4 Multiple-comparison correction ─────────────────────────────────────

def test_adjust_pvalues_bh_known_case() -> None:
    """BH on [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205].

    Hand-computed q-values via the step-up rule
        q_(i) = min_{j>=i} (n/j) * p_(j), n=8.

    j=1 (0.001): 8/1 * 0.001 = 0.0080  (running_min capped from above)
    j=2 (0.008): 8/2 * 0.008 = 0.0320
    j=3 (0.039): 8/3 * 0.039 = 0.1040 -> min with later -> 0.0672
    j=4 (0.041): 8/4 * 0.041 = 0.0820 -> min with later -> 0.0672
    j=5 (0.042): 8/5 * 0.042 = 0.0672 -> min with later -> 0.0672
    j=6 (0.060): 8/6 * 0.060 = 0.0800
    j=7 (0.074): 8/7 * 0.074 = 0.0846
    j=8 (0.205): 8/8 * 0.205 = 0.2050
    """
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205]
    q = adjust_pvalues(p, method="bh")
    expected = [0.0080, 0.0320, 0.0672, 0.0672, 0.0672,
                0.0800, 0.0846, 0.2050]
    for got, want in zip(q, expected):
        assert got == pytest.approx(want, abs=5e-3)


def test_adjust_pvalues_holm_known_case() -> None:
    """Holm on [0.01, 0.02, 0.03] for n=3 gives [0.03, 0.04, 0.04]."""
    q = adjust_pvalues([0.01, 0.02, 0.03], method="holm")
    assert q == pytest.approx([0.03, 0.04, 0.04], abs=1e-9)


def test_adjust_pvalues_empty() -> None:
    assert adjust_pvalues([]) == []


def test_adjust_pvalues_invalid_method() -> None:
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, 0.2], method="bogus")


def test_adjust_pvalues_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, 1.2])
    with pytest.raises(ValueError):
        adjust_pvalues([-0.1, 0.5])


# ── E8.5 Friedman + Nemenyi ─────────────────────────────────────────────────

def test_friedman_nemenyi_clear_winner() -> None:
    """One controller dominates — Friedman p < 0.01, ranks ordered."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    rows = []
    # 10 scenarios, 3 controllers; A wins consistently.
    for s in range(10):
        rows.append({"scenario": s, "controller": "A",
                     "metric": rng.normal(10.0, 0.5)})
        rows.append({"scenario": s, "controller": "B",
                     "metric": rng.normal(5.0, 0.5)})
        rows.append({"scenario": s, "controller": "C",
                     "metric": rng.normal(1.0, 0.5)})
    df = pd.DataFrame(rows)
    res = friedman_with_nemenyi(df, "metric")
    assert res["n_scenarios"] == 10
    assert res["n_controllers"] == 3
    assert res["friedman_p"] < 0.01
    # Higher metric → higher rank.
    ranks = res["avg_ranks"]
    assert ranks["A"] > ranks["B"] > ranks["C"]
    # Nemenyi finds at least one significant pair.
    assert res["nemenyi"]["sig"].any()


def test_friedman_nemenyi_no_difference() -> None:
    """All controllers ~equal — Friedman p high, no Nemenyi sig pairs.

    Use small noise so ranks vary slightly per scenario but no controller
    consistently beats the others (avoids degenerate-zero variance).
    """
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    rows = []
    for s in range(20):
        for c in ("A", "B", "C"):
            rows.append({"scenario": s, "controller": c,
                         "metric": float(rng.normal(0.0, 1.0))})
    df = pd.DataFrame(rows)
    res = friedman_with_nemenyi(df, "metric")
    assert res["friedman_p"] > 0.10
    assert not res["nemenyi"]["sig"].any()


def test_friedman_nemenyi_too_few_controllers() -> None:
    """K=2 → Friedman degenerates to Wilcoxon; we report NaNs gracefully."""
    df = pd.DataFrame([
        {"scenario": s, "controller": c, "metric": float(s + (c == "B"))}
        for s in range(5) for c in ("A", "B")
    ])
    res = friedman_with_nemenyi(df, "metric")
    assert np.isnan(res["friedman_stat"])
    assert res["nemenyi"].empty


# ── E8 wiring: paired_wilcoxon and Mann-Whitney expose adjusted p ──────────

def test_paired_wilcoxon_returns_pvalue_adj_field() -> None:
    pytest.importorskip("scipy")
    df = pd.DataFrame([
        {"planner": "NavFn", "run_id": i, "leg_id": 0,
         "controller": c, "m": v}
        for i in range(10) for c, v in (("A", 1.0 + i), ("B", 0.0 + i))
    ])
    res = paired_wilcoxon(df, "m", "A", "B", correct="bh")
    assert "pvalue_adj" in res
    # Single comparison → adjusted == raw.
    assert res["pvalue_adj"] == res["pvalue"]


def test_mann_whitney_per_controller_correct_bh_applies() -> None:
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    rows = []
    for c, mean in (("A", 0.0), ("B", 1.0), ("C", 2.0)):
        for _ in range(50):
            rows.append({"controller": c, "m": float(rng.normal(mean, 1.0))})
    df = pd.DataFrame(rows)
    res = mann_whitney_per_controller(df, "m", ["A", "B", "C"], correct="bh")
    assert "pvalue_adj" in res.columns
    # All adj p values must be >= raw p values (BH only inflates).
    assert (res["pvalue_adj"] >= res["pvalue"]).all()
