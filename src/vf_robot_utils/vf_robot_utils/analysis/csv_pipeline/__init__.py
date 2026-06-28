"""csv_pipeline — row-major CSV deliverables for the thesis evaluation.

Modules:
  metrics_from_h5.py       : episode-level metric extractor for the
                             current HDF5 schema (used by aggregate_csv).
  aggregate_csv.py         : walks vf_data/<map>/goal_*/<planner>/<ctrl>/
                             and writes the master results.csv.
  controller_selection.py  : entropy-weighted TOPSIS winner selection
                             over results.csv (primary selection method).

The comparison figures (figures/comparison_figures.py) consume
results.csv + controller_selection.csv directly.

Schema reference: EVALUATION_PLAN.md.

Source of truth: results.csv produced by aggregate_csv.
Catalog source of truth: METRIC_CATALOG below.
"""
from __future__ import annotations

from typing import NamedTuple


# ---------------------------------------------------------------------------
# Metric catalogue — canonical metric metadata used by every CSV / figure.
#
# Categories (benchmark-aligned, 2026-05-16):
#   Outcome             — binary task completion + failure modes (gate, not
#                         a TOPSIS criterion).
#   Efficiency          — SPL family + duration/path-length/stall.
#   Safety              — collision, clearance, near-miss, SVR.
#   Motion quality      — jerk, acceleration, sign-flips.
#   Path adherence      — XTE family vs each controller's own active /plan.
#   Adaptation          — entropy / total variation; **trained-only**
#   (diagnostic)          and structurally NaN for non-meta controllers,
#                         so excluded from the decision.
#
# The category name in `tier` is the only thing that maps to the new
# taxonomy; column keys (`t2_success`, etc.) are stable internal IDs so
# none of the existing pipeline code breaks.
#
# `t3_duration_s` and `t3_path_length_m` are *duplicates* of `t2_duration_s`
# / `t2_actual_path_m` produced by the same upstream attrs. They are flagged
# `dir=0` here so the composite/TOPSIS pipeline cannot double-count them,
# but the columns stay in results.csv for backwards compatibility.
#
# Conventions:
#   +1  higher-is-better
#   -1  lower-is-better
#    0  neutral / informational (NOT used in any composite or TOPSIS run)
#   drop_inf = True → clearance metrics: treat ±inf as categorical, strip
#                     before any mean / CI computation.
# ---------------------------------------------------------------------------

class _M(NamedTuple):
    tier: str       # category name (see header above)
    col: str        # results.csv column key
    label: str      # display name
    dir: int        # +1 higher-better, -1 lower-better, 0 neutral
    drop_inf: bool = False


METRIC_CATALOG: list[_M] = [
    # Outcome — binary task completion + failure modes. Used as a gate
    # (SR > 0) before TOPSIS, NOT as a TOPSIS criterion: SPL already
    # encodes success.
    _M("Outcome",       "t2_success",          "Success rate",                +1),
    _M("Outcome",       "t2_timeout",          "Timeout rate",                -1),
    _M("Outcome",       "t2_aborted",          "Abort rate",                  -1),
    _M("Outcome",       "t2_canceled",         "Cancel rate",                 -1),
    _M("Outcome",       "t2_rescued",          "Rescue rate",                 -1),
    _M("Outcome",       "t2_gpe_m",            "Goal pos error (m)",          -1),
    _M("Outcome",       "t2_goe_rad",          "Goal orient error (rad)",     -1),

    # Efficiency — SPL (Anderson 2018) and path/time economy.
    # t2_spl_safe = t2_spl * (1 - t1_collision); headline metric only,
    # NOT a TOPSIS criterion (would double-count with the Safety
    # category's collision rate). Plain t2_spl is the TOPSIS criterion.
    _M("Efficiency",    "t2_spl_safe",         "SPL (safety-gated)",          +1),
    _M("Efficiency",    "t2_spl",              "SPL",                         +1),
    _M("Efficiency",    "t2_spl_astar",        "SPL (A* reference)",          +1),
    _M("Efficiency",    "t2_duration_s",       "Episode duration (s)",        -1),
    _M("Efficiency",    "t2_actual_path_m",    "Actual path length (m)",      -1),
    _M("Efficiency",    "t2_optimal_path_m",   "Optimal path length (m)",      0),
    _M("Efficiency",    "t2_astar_path_m",     "A* path length (m)",           0),
    _M("Efficiency",    "t3_mean_lin_vel",     "Mean lin vel (m/s)",          +1),
    _M("Efficiency",    "t3_max_lin_vel",      "Max lin vel (m/s)",           +1),
    _M("Efficiency",    "t3_time_at_cruise_frac", "Cruise-time fraction",     +1),
    _M("Efficiency",    "t3_stall_fraction",   "Stall fraction",              -1),
    _M("Efficiency",    "t3_loop_time_mean_ms","Loop time mean (ms)",         -1),
    _M("Efficiency",    "t3_loop_time_p99_ms", "Loop time P99 (ms)",          -1),

    # t3_duration_s / t3_path_length_m are duplicates of t2_*; dir=0
    # so the TOPSIS / composite pipeline cannot double-count them.
    _M("Efficiency",    "t3_duration_s",       "Duration (dup of t2)",         0),
    _M("Efficiency",    "t3_path_length_m",    "Path length (dup of t2)",      0),
    _M("Efficiency",    "t3_mean_abs_ang_vel", "Mean |ang vel| (rad/s)",      -1),

    # Safety — collision avoidance and clearance margins.
    _M("Safety",        "t1_collision",        "Collision rate",              -1),
    _M("Safety",        "t1_mmc_m",            "Mean min clearance (m)",      +1, drop_inf=True),
    _M("Safety",        "t1_mdo_m",            "Min dist to obstacle (m)",    +1, drop_inf=True),
    _M("Safety",        "t1_p5_clear_m",       "P5 clearance (m)",            +1),
    _M("Safety",        "t1_near_miss_rate",   "Near-miss rate",              -1),
    _M("Safety",        "t1_svr",              "Safety violation rate",       -1),

    # Motion quality — smoothness of commanded and observed motion.
    _M("Motion quality", "t4_mean_jerk",       "Mean jerk (m/s³)",            -1),
    _M("Motion quality", "t4_max_jerk",        "Max jerk (m/s³)",             -1),
    _M("Motion quality", "t4_ang_vel_std",     "Ang vel std (rad/s)",         -1),
    _M("Motion quality", "t4_cmd_accel_rms",   "Cmd accel RMS (m/s²)",        -1),
    _M("Motion quality", "t4_cmd_sign_flips",  "Lin cmd sign flips",          -1),
    _M("Motion quality", "t4_cmd_ang_sign_flips","Ang cmd sign flips",        -1),

    # Path adherence — XTE vs each controller's own time-active /plan.
    _M("Path adherence", "t6_mean_xte_m",      "Mean XTE (m)",                -1),
    _M("Path adherence", "t6_median_xte_m",    "Median XTE (m)",              -1),
    _M("Path adherence", "t6_max_xte_m",       "Max XTE (m)",                 -1),
    _M("Path adherence", "t6_p95_xte_m",       "P95 XTE (m)",                 -1),
    _M("Path adherence", "t6_std_xte_m",       "Std XTE (m)",                 -1),
    _M("Path adherence", "t6_rmse_xte_m",      "RMSE XTE (m)",                -1),
    _M("Path adherence", "t6_mean_signed_xte_m","Mean signed XTE (m)",         0),
    _M("Path adherence", "t6_frechet_m",       "Frechet distance (m)",        -1),
    _M("Path adherence", "t6_path_coverage_frac","Path coverage frac",        +1),

    # Adaptation (diagnostic) — trained-only, structurally NaN for
    # classical baselines. Excluded from TOPSIS to keep the decision
    # cross-comparable across all controllers.
    _M("Adaptation",    "t5_mean_entropy",     "Mean weight entropy",         +1),
    _M("Adaptation",    "t5_mean_entropy_norm","Mean entropy (normalised)",   +1),
    _M("Adaptation",    "t5_total_variation",  "Total variation",             -1),
    _M("Adaptation",    "t5_tv_per_second",    "TV per second",               -1),
    _M("Adaptation",    "t5_dominant_critic_fraction","Dominant critic fraction", -1),
    _M("Adaptation",    "t5_n_weight_samples", "Weight samples (n)",           0),
    _M("Adaptation",    "t5_num_critics",      "Num critics",                  0),
]


# TOPSIS decision criteria — one per non-diagnostic category, minimally
# correlated. Used by controller_selection.py.
# (success-rate is the gate, not a criterion; collision rate sits in
#  Safety; t2_spl_safe is NOT here — it's the headline only.)
TOPSIS_CRITERIA: list[str] = [
    "t2_spl",            # Efficiency
    "t1_mmc_m",          # Safety   (clearance)
    "t1_collision",      # Safety   (collision rate)
    "t4_mean_jerk",      # Motion quality
    "t6_mean_xte_m",     # Path adherence
    "t2_duration_s",     # Efficiency (time) — correlation-checked against SPL
]


# Mapping from category name (canonical "tier") to its members. Built
# lazily so adding catalog entries doesn't require touching this code.
def metrics_by_category() -> dict[str, list[_M]]:
    out: dict[str, list[_M]] = {}
    for m in METRIC_CATALOG:
        out.setdefault(m.tier, []).append(m)
    return out


# Episode-level binary outcomes → Wilson 95 % CI in summary tables.
PROPORTION_METRICS: set[str] = {
    "t1_collision",
    "t2_success", "t2_timeout", "t2_aborted", "t2_canceled", "t2_rescued",
}


# ---------------------------------------------------------------------------
# Controller universe — single source of truth for the thesis evaluation.
# Stage 1 and Stage 2 use TRAINED_CONTROLLERS (19 rows).
# Stage 3 uses one top-trained + CLASSICAL_CONTROLLERS minus graceful.
# ---------------------------------------------------------------------------

TRAINED_CONTROLLERS: list[str] = (
    [f"oraclewt_{hp}_v{v}" for hp in ("normal", "tuned", "hardreg") for v in (1, 2, 3)]
    + [f"rawwt_{hp}_v{v}"    for hp in ("normal", "tuned", "hardreg") for v in (1, 2, 3)]
    + ["imitationwt_normal_v1"]
)

CLASSICAL_CONTROLLERS: list[str] = ["fixedwt", "mppi", "dwb", "rpp", "graceful"]

ALL_CONTROLLERS: list[str] = TRAINED_CONTROLLERS + CLASSICAL_CONTROLLERS


def family_of(controller: str) -> str:
    if controller in CLASSICAL_CONTROLLERS:
        return controller
    if controller.startswith("oraclewt_"):    return "oraclewt"
    if controller.startswith("rawwt_"):       return "rawwt"
    if controller.startswith("imitationwt_"): return "imitationwt"
    return "unknown"


__all__ = [
    "METRIC_CATALOG", "_M",
    "TOPSIS_CRITERIA",
    "metrics_by_category",
    "PROPORTION_METRICS",
    "TRAINED_CONTROLLERS", "CLASSICAL_CONTROLLERS", "ALL_CONTROLLERS",
    "family_of",
]
