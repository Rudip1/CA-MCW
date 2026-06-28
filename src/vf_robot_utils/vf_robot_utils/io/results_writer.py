"""Append-only master results.csv writer for evaluation runs.

One row per recorded leg (planner × controller × run_id × leg_id).
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path


# Canonical field list — derived from actual tier dataclass fields + identity cols.
# Tier5 list fields (mean_weights, std_weights) expanded to N_CRITICS=10 columns.
RESULTS_FIELDS: list[str] = [
    # ── identity ──────────────────────────────────────────────────────────────
    'run_id', 'leg_id', 'leg_start', 'leg_goal',
    'planner', 'controller', 'goal_folder', 'scenario', 'timestamp',
    'csv_path', 'csv_sha1', 'map_path', 'map_sha1',
    'global_paths_json', 'cached_path_length_m', 'cached_path_n_poses',

    # ── tier 1 — safety ───────────────────────────────────────────────────────
    't1_collision', 't1_mdo_m', 't1_mmc_m', 't1_p5_clear_m',
    't1_near_miss_rate', 't1_svr',
    't1_n_snaps', 't1_n_snaps_used', 't1_near_miss_threshold_m',

    # ── tier 2 — success ──────────────────────────────────────────────────────
    # t2_spl_safe = t2_spl * (1 - t1_collision). Paper's headline metric:
    # a collision forces SPL to 0 even if the goal was reached, making
    # safety non-negotiable at the top line. Plain t2_spl is the TOPSIS
    # criterion (collision rate is a separate Safety criterion).
    't2_success', 't2_timeout', 't2_aborted', 't2_canceled', 't2_rescued',
    't2_final_status', 't2_spl', 't2_spl_astar', 't2_spl_safe',
    't2_gpe_m', 't2_goe_rad',
    't2_actual_path_m', 't2_optimal_path_m', 't2_astar_path_m', 't2_duration_s',

    # ── tier 3 — efficiency ───────────────────────────────────────────────────
    't3_duration_s', 't3_path_length_m',
    't3_mean_lin_vel', 't3_max_lin_vel', 't3_mean_abs_ang_vel',
    't3_time_at_cruise_frac', 't3_stall_fraction',
    't3_loop_time_mean_ms', 't3_loop_time_p99_ms',

    # ── tier 4 — smoothness ───────────────────────────────────────────────────
    't4_mean_jerk', 't4_max_jerk',
    't4_ang_vel_std', 't4_cmd_accel_rms',
    't4_cmd_sign_flips', 't4_cmd_ang_sign_flips',

    # ── tier 5 — diagnostics (NaN for non-meta-critic controllers) ────────────
    't5_n_weight_samples', 't5_num_critics',
    *[f't5_mean_weights_{i}' for i in range(10)],
    *[f't5_std_weights_{i}'  for i in range(10)],
    't5_mean_entropy', 't5_mean_entropy_norm',
    't5_total_variation', 't5_tv_per_second',
    't5_dominant_critic_fraction',

    # ── tier 6 — path following ───────────────────────────────────────────────
    # t6_xte_ref logs which reference was used:
    #   "active_plan"  — sim_time-bisected /plan history (preferred)
    #   "final_plan"   — last /plan only (back-compat fallback)
    #   "straight_line"— legacy fallback for HDF5s without /plan logging
    't6_has_global_path', 't6_xte_ref',
    't6_mean_xte_m', 't6_median_xte_m', 't6_max_xte_m', 't6_p95_xte_m',
    't6_std_xte_m', 't6_rmse_xte_m', 't6_mean_signed_xte_m',
    't6_frechet_m',
    't6_path_coverage_frac',
    't6_n_global_poses', 't6_n_robot_poses',

    # ── data pointers ─────────────────────────────────────────────────────────
    'hdf5_path', 'notes',
]


def build_fieldnames() -> list[str]:
    return list(RESULTS_FIELDS)


class ResultsWriter:
    def __init__(self, path: Path, fieldnames: list[str] | None = None) -> None:
        self._path = Path(path)
        self._fieldnames = fieldnames or build_fieldnames()
        is_new = not self._path.exists() or self._path.stat().st_size == 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self._path, 'a', newline='')
        self._writer = csv.DictWriter(
            self._f, fieldnames=self._fieldnames, extrasaction='ignore')
        if is_new:
            self._writer.writeheader()
            self._f.flush()

    def append(self, row: dict) -> None:
        unknown = set(row.keys()) - set(self._fieldnames)
        if unknown:
            warnings.warn(f'ResultsWriter: dropping unknown keys: {sorted(unknown)}')
        filled = {k: '' for k in self._fieldnames}
        filled.update({k: v for k, v in row.items() if k in self._fieldnames})
        self._writer.writerow(filled)
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
