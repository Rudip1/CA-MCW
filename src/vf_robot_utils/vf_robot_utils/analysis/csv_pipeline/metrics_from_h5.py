#!/usr/bin/env python3
"""metrics_from_h5.py — episode-level metric extraction for the current HDF5 schema.

Background
----------
The legacy metrics modules (vf_robot_utils.metrics.tier*) read an obsolete
HDF5 layout (meta/, time_series/, costmaps/, global_path_poses). The
current `data_collector_node` writes a different schema: root-level
attrs + datasets (robot_pose, selected_action, goal, sim_time, features,
critic_costs, critic_weights_applied). This module recomputes every tier
metric we can derive from that current schema, producing a row dict whose
keys match `io.results_writer.RESULTS_FIELDS` so the rest of the pipeline
(controller_selection and the comparison figures, which all read
results.csv) works unchanged.

What we can compute, vs cannot
------------------------------
Computable from current schema:
  T1: t1_collision (via attrs + clearance), t1_mmc_m, t1_mdo_m,
      t1_p5_clear_m, t1_near_miss_rate
  T2: t2_success, t2_duration_s, t2_gpe_m, t2_goe_rad, t2_spl
      (Euclidean optimal), t2_actual_path_m, t2_final_status
  T3: t3_duration_s, t3_path_length_m, t3_mean_lin_vel, t3_max_lin_vel,
      t3_mean_abs_ang_vel, t3_stall_fraction, t3_time_at_cruise_frac
  T4: t4_mean_jerk, t4_max_jerk, t4_ang_vel_std, t4_cmd_accel_rms,
      t4_cmd_sign_flips, t4_cmd_ang_sign_flips
  T5: t5_n_weight_samples, t5_num_critics, t5_mean_weights_*,
      t5_std_weights_*, t5_mean_entropy, t5_mean_entropy_norm,
      t5_total_variation, t5_tv_per_second, t5_dominant_critic_fraction
      (NaN for baselines — no /vf/applied_weights)
  T6: t6_mean_xte_m, t6_max_xte_m, t6_p95_xte_m, t6_rmse_xte_m,
      t6_mean_signed_xte_m, t6_path_coverage_frac, t6_xte_ref
      Reference resolution (most-preferred first):
        (a) "active_plan"  — sim_time-bisected /plan that was live at
                             each robot step (collector logs every /plan
                             into global_path_plans/ as of 2026-05-16).
        (b) "final_plan"   — single global_path_poses if plans-group
                             absent.
        (c) "straight_line"— start→goal Euclidean fallback for legacy
                             HDF5s. The chosen tier is logged in
                             t6_xte_ref so figures can filter / annotate.

Not derivable without re-collecting:
  t1_svr (no /vf/safety_veto in the collector schema)
  t3_loop_time_* (no per-tick controller wall-clock log)
  t6_frechet_m (computed but only meaningful vs a real planner path)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEAR_MISS_THRESHOLD_M = 0.30        # project default
P5_CLEARANCE_QUANTILE = 0.05
STALL_LIN_VEL_M_S      = 0.05
CRUISE_LIN_VEL_M_S     = 0.20
N_CRITIC_COLS_IN_CSV   = 10         # results_writer hardcodes 10 weight cols


# ---------------------------------------------------------------------------
# Map cache — load + distance-transform once per workspace
# ---------------------------------------------------------------------------

@dataclass
class _ObstacleField:
    dist_m: np.ndarray   # (H, W) distance in metres to nearest obstacle cell
    res:    float
    origin_x: float
    origin_y: float
    height: int
    width:  int

    def clearance_at(self, wx: float, wy: float) -> float:
        """World (wx, wy) → metres to nearest obstacle. NaN if outside map."""
        col = int((wx - self.origin_x) / self.res)
        row = self.height - 1 - int((wy - self.origin_y) / self.res)
        if 0 <= row < self.height and 0 <= col < self.width:
            return float(self.dist_m[row, col])
        return float("nan")


_FIELD_CACHE: dict[str, _ObstacleField] = {}


def _load_obstacle_field(map_yaml: str) -> _ObstacleField:
    if map_yaml in _FIELD_CACHE:
        return _FIELD_CACHE[map_yaml]

    import yaml
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    with open(map_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    pgm = cfg["image"]
    if not os.path.isabs(pgm):
        pgm = os.path.join(os.path.dirname(os.path.abspath(map_yaml)), pgm)
    res = float(cfg["resolution"])
    origin_x = float(cfg["origin"][0])
    origin_y = float(cfg["origin"][1])
    negate = int(cfg.get("negate", 0))
    free_thresh = float(cfg.get("free_thresh", 0.25))

    # ROS map_server convention: occupancy = (255 - pixel)/255 → 0 free, 1 occ.
    raw = np.array(Image.open(pgm).convert("L"), dtype=np.float32)
    occ = (255.0 - raw) / 255.0
    if negate:
        occ = 1.0 - occ
    # Cells whose *occupied probability* is below free_thresh are navigable.
    free = occ < free_thresh
    # distance_transform_edt(input) returns the distance to the nearest
    # *zero* cell. We want distance from each cell to nearest obstacle,
    # so feed the FREE mask: free cells are nonzero, obstacle cells zero.
    dist_cells = distance_transform_edt(free.astype(np.uint8))
    dist_m = dist_cells * res

    field = _ObstacleField(
        dist_m=dist_m.astype(np.float32),
        res=res, origin_x=origin_x, origin_y=origin_y,
        height=int(dist_m.shape[0]), width=int(dist_m.shape[1]),
    )
    _FIELD_CACHE[map_yaml] = field
    return field


# ---------------------------------------------------------------------------
# Per-tier extractors — operate on already-loaded arrays / attrs
# ---------------------------------------------------------------------------

def _tier1(robot_pose: np.ndarray, field: Optional[_ObstacleField],
           collision_attr: int) -> dict:
    out = {
        "t1_collision":      bool(collision_attr > 0) if collision_attr is not None else False,
        "t1_mmc_m":          float("nan"),
        "t1_mdo_m":          float("nan"),
        "t1_p5_clear_m":     float("nan"),
        "t1_near_miss_rate": float("nan"),
        "t1_svr":            float("nan"),
        "t1_n_snaps":        int(robot_pose.shape[0]) if robot_pose is not None else 0,
        "t1_n_snaps_used":   0,
        "t1_near_miss_threshold_m": NEAR_MISS_THRESHOLD_M,
    }
    if field is None or robot_pose is None or robot_pose.shape[0] == 0:
        return out
    clearances = np.empty(robot_pose.shape[0], dtype=np.float64)
    for i in range(robot_pose.shape[0]):
        clearances[i] = field.clearance_at(
            float(robot_pose[i, 0]), float(robot_pose[i, 1])
        )
    valid = clearances[np.isfinite(clearances)]
    if valid.size == 0:
        return out
    out["t1_n_snaps_used"]   = int(valid.size)
    out["t1_mmc_m"]          = float(valid.mean())
    out["t1_mdo_m"]          = float(valid.min())
    out["t1_p5_clear_m"]     = float(np.quantile(valid, P5_CLEARANCE_QUANTILE))
    out["t1_near_miss_rate"] = float(np.mean(valid < NEAR_MISS_THRESHOLD_M))
    # Override collision detection with map-derived if not set by attr.
    if not out["t1_collision"]:
        out["t1_collision"] = bool(np.any(valid <= 0.0))
    return out


def _tier2(robot_pose: np.ndarray, goal: np.ndarray, sim_time: np.ndarray,
           attrs: dict) -> dict:
    out = {
        "t2_success":        bool(attrs.get("success", False)),
        "t2_timeout":        False,
        "t2_aborted":        False,
        "t2_canceled":       False,
        "t2_rescued":        False,
        "t2_final_status":   "unknown",
        "t2_spl":            float("nan"),
        "t2_spl_astar":      float("nan"),
        "t2_gpe_m":          float("nan"),
        "t2_goe_rad":        float("nan"),
        "t2_actual_path_m":  float(attrs.get("path_length_m", float("nan"))),
        "t2_optimal_path_m": float("nan"),
        "t2_astar_path_m":   float("nan"),
        "t2_duration_s":     float(attrs.get("time_to_goal_s", float("nan"))),
    }
    if robot_pose is None or robot_pose.shape[0] == 0:
        out["t2_final_status"] = "no_data"
        return out

    # Final pose error vs goal (use last-step goal — goal is constant per leg).
    gx, gy, gyaw = float(goal[-1, 0]), float(goal[-1, 1]), float(goal[-1, 2])
    fx, fy, fyaw = float(robot_pose[-1, 0]), float(robot_pose[-1, 1]), float(robot_pose[-1, 2])
    out["t2_gpe_m"]   = float(math.hypot(fx - gx, fy - gy))
    out["t2_goe_rad"] = float(abs(math.atan2(math.sin(fyaw - gyaw),
                                              math.cos(fyaw - gyaw))))

    # Optimal-path = straight-line from start to goal (Euclidean lower bound).
    sx, sy = float(robot_pose[0, 0]), float(robot_pose[0, 1])
    out["t2_optimal_path_m"] = float(math.hypot(gx - sx, gy - sy))

    # SPL = success * optimal / max(actual, optimal)
    actual = out["t2_actual_path_m"]
    opt    = out["t2_optimal_path_m"]
    if math.isfinite(actual) and math.isfinite(opt) and max(actual, opt) > 0:
        out["t2_spl"] = float(int(out["t2_success"]) * opt / max(actual, opt))

    out["t2_final_status"] = "succeeded" if out["t2_success"] else "timeout_or_abort"
    out["t2_timeout"] = not out["t2_success"]
    return out


def _tier3(selected_action: np.ndarray, sim_time: np.ndarray,
           path_length_m: float, duration_s: float) -> dict:
    out = {
        "t3_duration_s":         float(duration_s) if duration_s is not None else float("nan"),
        "t3_path_length_m":      float(path_length_m) if path_length_m is not None else float("nan"),
        "t3_mean_lin_vel":       float("nan"),
        "t3_max_lin_vel":        float("nan"),
        "t3_mean_abs_ang_vel":   float("nan"),
        "t3_time_at_cruise_frac": float("nan"),
        "t3_stall_fraction":     float("nan"),
        "t3_loop_time_mean_ms":  float("nan"),
        "t3_loop_time_p99_ms":   float("nan"),
    }
    if selected_action is None or selected_action.shape[0] == 0:
        return out
    vx = selected_action[:, 0].astype(np.float64)
    wz = selected_action[:, 2].astype(np.float64)
    speed = np.abs(vx)
    out["t3_mean_lin_vel"]       = float(speed.mean())
    out["t3_max_lin_vel"]        = float(speed.max())
    out["t3_mean_abs_ang_vel"]   = float(np.abs(wz).mean())
    out["t3_stall_fraction"]     = float(np.mean(speed < STALL_LIN_VEL_M_S))
    out["t3_time_at_cruise_frac"] = float(np.mean(speed >= CRUISE_LIN_VEL_M_S))
    return out


def _safe_diff_dt(y: np.ndarray, t: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """First-order forward diff y/dt, NaN where dt < eps or non-finite."""
    dy = np.diff(y)
    dt = np.diff(t)
    dt = np.where(np.isfinite(dt) & (dt > eps), dt, np.nan)
    return dy / dt


def _effective_time_axis(sim_time: np.ndarray, n_steps: int,
                         fallback_hz: float = 10.0) -> np.ndarray:
    """Return a finite-valued time axis. If sim_time is all-NaN (collector bug
    on some legacy runs) synthesize a uniform clock at `fallback_hz` Hz so
    derivative-based metrics stay numerically defined."""
    if sim_time is None or sim_time.size == 0:
        return np.arange(n_steps, dtype=np.float64) / fallback_hz
    t = np.asarray(sim_time, dtype=np.float64)
    if not np.any(np.isfinite(t)):
        return np.arange(n_steps, dtype=np.float64) / fallback_hz
    return t


def _tier4(selected_action: np.ndarray, sim_time: np.ndarray) -> dict:
    out = {
        "t4_mean_jerk":           float("nan"),
        "t4_max_jerk":            float("nan"),
        "t4_ang_vel_std":         float("nan"),
        "t4_cmd_accel_rms":       float("nan"),
        "t4_cmd_sign_flips":      0,
        "t4_cmd_ang_sign_flips":  0,
    }
    if selected_action is None or selected_action.shape[0] < 3:
        return out
    vx = selected_action[:, 0].astype(np.float64)
    wz = selected_action[:, 2].astype(np.float64)
    t  = _effective_time_axis(sim_time, vx.size)
    accel = _safe_diff_dt(vx, t)
    jerk  = _safe_diff_dt(accel, t[1:])
    accel_finite = accel[np.isfinite(accel)]
    jerk_finite  = jerk[np.isfinite(jerk)]
    out["t4_cmd_accel_rms"] = float(np.sqrt(np.mean(accel_finite ** 2))) if accel_finite.size else float("nan")
    out["t4_mean_jerk"]     = float(np.mean(np.abs(jerk_finite))) if jerk_finite.size else float("nan")
    out["t4_max_jerk"]      = float(np.max(np.abs(jerk_finite))) if jerk_finite.size else float("nan")
    out["t4_ang_vel_std"]   = float(wz.std(ddof=1)) if wz.size >= 2 else float("nan")

    # Sign flips: count sign changes (ignoring zero-crossings within +/-1e-3).
    def _flips(x: np.ndarray) -> int:
        sx = np.sign(np.where(np.abs(x) < 1e-3, 0.0, x))
        return int(np.sum(np.abs(np.diff(sx)) > 1.5))
    out["t4_cmd_sign_flips"]     = _flips(vx)
    out["t4_cmd_ang_sign_flips"] = _flips(wz)
    return out


def _tier5(critic_weights: np.ndarray, critic_names: list, sim_time: np.ndarray) -> dict:
    """Per-critic weight diagnostics. NaN for baselines (no weights)."""
    out: dict = {
        "t5_n_weight_samples": 0,
        "t5_num_critics":      0,
        "t5_mean_entropy":     float("nan"),
        "t5_mean_entropy_norm": float("nan"),
        "t5_total_variation":  float("nan"),
        "t5_tv_per_second":    float("nan"),
        "t5_dominant_critic_fraction": float("nan"),
    }
    for i in range(N_CRITIC_COLS_IN_CSV):
        out[f"t5_mean_weights_{i}"] = float("nan")
        out[f"t5_std_weights_{i}"]  = float("nan")

    if critic_weights is None or critic_weights.shape[0] == 0:
        return out
    W = critic_weights.astype(np.float64)
    # Drop rows where every weight is NaN (baselines write all-NaN rows).
    finite_rows = np.isfinite(W).all(axis=1)
    W = W[finite_rows]
    if W.size == 0:
        return out

    n_samples = int(W.shape[0])
    K = int(W.shape[1])
    out["t5_n_weight_samples"] = n_samples
    out["t5_num_critics"]      = K

    for i in range(min(N_CRITIC_COLS_IN_CSV, K)):
        out[f"t5_mean_weights_{i}"] = float(W[:, i].mean())
        out[f"t5_std_weights_{i}"]  = float(W[:, i].std(ddof=1)) if n_samples >= 2 else float("nan")

    # Row-normalize to a probability simplex for entropy.
    row_sums = W.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 1e-9, row_sums, 1.0)
    P = np.clip(W / safe, 1e-12, 1.0)
    H = -np.sum(P * np.log(P), axis=1)
    out["t5_mean_entropy"] = float(H.mean())
    out["t5_mean_entropy_norm"] = float(H.mean() / math.log(K)) if K > 1 else float("nan")

    # Total variation: sum of |W_t - W_{t-1}| across critics, summed over time.
    if n_samples >= 2:
        tv = float(np.sum(np.abs(np.diff(W, axis=0))))
        out["t5_total_variation"] = tv
        t_eff = _effective_time_axis(sim_time, n_samples)
        if t_eff.size >= 2 and math.isfinite(t_eff[-1] - t_eff[0]):
            dur = float(t_eff[-1] - t_eff[0])
            if dur > 0:
                out["t5_tv_per_second"] = tv / dur

    # Dominant critic fraction: per-step argmax shares; report the largest.
    dominant_idx = np.argmax(W, axis=1)
    counts = np.bincount(dominant_idx, minlength=K)
    out["t5_dominant_critic_fraction"] = float(counts.max() / n_samples)
    return out


def _point_to_polyline(
    points_xy: np.ndarray, polyline_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised perpendicular distance from each point to a polyline.

    Returns
    -------
    signed : (P,) float64
        Signed perpendicular distance to the *nearest* segment. Sign
        follows the segment's left-normal (positive = left of travel).
    abs_d : (P,) float64
        ``|signed|`` — the unsigned point-to-polyline distance.

    Memory is O(P * S); used in batches keyed by active-plan index so a
    full-episode call never materialises the full cross-product.
    """
    P = int(points_xy.shape[0])
    if polyline_xy.shape[0] < 2:
        if polyline_xy.shape[0] == 0:
            return np.zeros(P), np.full(P, np.nan)
        d = np.linalg.norm(points_xy - polyline_xy[0], axis=1)
        return np.zeros(P, dtype=np.float64), d.astype(np.float64)
    a = polyline_xy[:-1].astype(np.float64)
    b = polyline_xy[1:].astype(np.float64)
    seg = b - a
    seg_len_sq = (seg ** 2).sum(axis=1)
    seg_len_sq = np.where(seg_len_sq < 1e-12, 1e-12, seg_len_sq)

    rel = points_xy[:, None, :] - a[None, :, :]            # (P, S, 2)
    t = (rel * seg[None, :, :]).sum(axis=2) / seg_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * seg[None, :, :]
    diffs = points_xy[:, None, :] - closest                # (P, S, 2)
    dist_sq = (diffs ** 2).sum(axis=2)                     # (P, S)
    abs_d = np.sqrt(dist_sq.min(axis=1))

    nearest = np.argmin(dist_sq, axis=1)
    rows = np.arange(P)
    seg_at = seg[nearest]
    seg_len = np.sqrt((seg_at ** 2).sum(axis=1))
    seg_len = np.where(seg_len < 1e-12, 1.0, seg_len)
    nx = -seg_at[:, 1] / seg_len
    ny =  seg_at[:, 0] / seg_len
    diff_at = diffs[rows, nearest]
    signed = diff_at[:, 0] * nx + diff_at[:, 1] * ny
    return signed.astype(np.float64), abs_d.astype(np.float64)


def _xte_against_polyline(
    robot_xy: np.ndarray, polyline_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Single-polyline XTE plus arc-length-projection 'coverage'.

    Returns (signed_xte, abs_xte, along_frac_max) where ``along_frac_max``
    is the maximum fraction of the polyline arc-length traversed by the
    robot.
    """
    signed, abs_d = _point_to_polyline(robot_xy, polyline_xy)
    # Arc-length parameter of the nearest projection (for coverage).
    if polyline_xy.shape[0] < 2:
        return signed, abs_d, float("nan")
    seg = np.diff(polyline_xy.astype(np.float64), axis=0)
    seg_lens = np.sqrt((seg ** 2).sum(axis=1))
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = float(cum[-1])
    if total < 1e-6:
        return signed, abs_d, float("nan")
    # Re-project each point onto its nearest segment to get along_frac.
    a = polyline_xy[:-1].astype(np.float64)
    b = polyline_xy[1:].astype(np.float64)
    seg_v = b - a
    seg_len_sq = (seg_v ** 2).sum(axis=1)
    seg_len_sq = np.where(seg_len_sq < 1e-12, 1e-12, seg_len_sq)
    rel = robot_xy[:, None, :] - a[None, :, :]
    t = (rel * seg_v[None, :, :]).sum(axis=2) / seg_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * seg_v[None, :, :]
    dist_sq = ((robot_xy[:, None, :] - closest) ** 2).sum(axis=2)
    nearest = np.argmin(dist_sq, axis=1)
    along = cum[nearest] + t[np.arange(robot_xy.shape[0]), nearest] * seg_lens[nearest]
    along_frac = float(np.clip(along.max() / total, 0.0, 1.0))
    return signed, abs_d, along_frac


def _tier6(
    robot_pose: "np.ndarray | None",
    robot_times: "np.ndarray | None",
    goal: "np.ndarray | None",
    plan_history: "list[tuple[float, np.ndarray]] | None" = None,
    final_plan: "np.ndarray | None" = None,
) -> dict:
    """Tier-6 path-following metrics.

    Reference resolution (most-preferred first):

    1. ``plan_history`` — time-active plan: each robot pose is matched
       (by sim_time bisect) to the most-recent /plan that was published
       at or before its timestamp. XTE is computed against that plan's
       polyline. This is the **standard** per-controller execution
       fidelity metric and what the CSVs report when the collector
       captured /plan history.
    2. ``final_plan`` — final-plan fallback: when plan_history is absent
       but a single ``global_path_poses`` was stored.
    3. straight-line start→goal — last-resort fallback for HDF5s that
       predate the plan-logging fix.

    The chosen reference is reported in the ``t6_xte_ref`` column.
    Figures (``fig_*_xte_*``) use the same active-plan reference; cross
    controller performance comparison is made via TOPSIS over the
    Efficiency / Safety / Motion quality / Path adherence categories
    (see ``controller_selection.py``), not by comparing ``t6_*`` values
    across controllers against a shared fixed path. See
    ``evaluation_plots.txt``.
    """
    out = {
        "t6_has_global_path":   False,
        "t6_xte_ref":           "none",
        "t6_mean_xte_m":        float("nan"),
        "t6_median_xte_m":      float("nan"),
        "t6_max_xte_m":         float("nan"),
        "t6_p95_xte_m":         float("nan"),
        "t6_std_xte_m":         float("nan"),
        "t6_rmse_xte_m":        float("nan"),
        "t6_mean_signed_xte_m": float("nan"),
        "t6_frechet_m":         float("nan"),
        "t6_path_coverage_frac": float("nan"),
        "t6_n_global_poses":    0,
        "t6_n_robot_poses":     int(robot_pose.shape[0]) if robot_pose is not None else 0,
    }
    if robot_pose is None or robot_pose.shape[0] < 2:
        return out

    robot_xy = robot_pose[:, :2].astype(np.float64)

    # ----- Tier-A: time-active plan reference ------------------------------
    if (
        plan_history
        and robot_times is not None
        and robot_times.shape[0] == robot_xy.shape[0]
    ):
        plan_times = np.asarray([t for t, _ in plan_history], dtype=np.float64)
        plan_polys = [np.asarray(p, dtype=np.float64) for _, p in plan_history]
        # For each robot pose, which plan was active?  Largest index i such
        # that plan_times[i] <= robot_times[k]. Clip to [0, M-1] so robot
        # poses earlier than the first plan use plan 0 (the initial plan).
        idx = np.searchsorted(plan_times, robot_times, side="right") - 1
        idx = np.clip(idx, 0, len(plan_polys) - 1)

        signed = np.full(robot_xy.shape[0], np.nan, dtype=np.float64)
        abs_d  = np.full(robot_xy.shape[0], np.nan, dtype=np.float64)
        for i, poly in enumerate(plan_polys):
            if poly.shape[0] < 2:
                continue
            mask = (idx == i)
            if not mask.any():
                continue
            s_i, a_i = _point_to_polyline(robot_xy[mask], poly[:, :2])
            signed[mask] = s_i
            abs_d[mask]  = a_i

        valid = np.isfinite(abs_d)
        if valid.any():
            sv = signed[valid]; av = abs_d[valid]
            # Coverage against the final plan (most representative of the
            # route the robot actually committed to at the end).
            _, _, cov = _xte_against_polyline(robot_xy, plan_polys[-1][:, :2])
            out["t6_has_global_path"]   = True
            out["t6_xte_ref"]           = "active_plan"
            out["t6_mean_xte_m"]        = float(av.mean())
            out["t6_median_xte_m"]      = float(np.median(av))
            out["t6_max_xte_m"]         = float(av.max())
            out["t6_p95_xte_m"]         = float(np.quantile(av, 0.95))
            out["t6_std_xte_m"]         = float(av.std(ddof=1)) if av.size >= 2 else float("nan")
            out["t6_rmse_xte_m"]        = float(np.sqrt(np.mean(sv ** 2)))
            out["t6_mean_signed_xte_m"] = float(sv.mean())
            out["t6_path_coverage_frac"] = cov
            out["t6_n_global_poses"]    = int(sum(p.shape[0] for p in plan_polys))
            return out

    # ----- Tier-B: final-plan fallback -------------------------------------
    if final_plan is not None and final_plan.shape[0] >= 2:
        poly = np.asarray(final_plan, dtype=np.float64)[:, :2]
        signed, abs_d, cov = _xte_against_polyline(robot_xy, poly)
        out["t6_has_global_path"]   = True
        out["t6_xte_ref"]           = "final_plan"
        out["t6_mean_xte_m"]        = float(abs_d.mean())
        out["t6_median_xte_m"]      = float(np.median(abs_d))
        out["t6_max_xte_m"]         = float(abs_d.max())
        out["t6_p95_xte_m"]         = float(np.quantile(abs_d, 0.95))
        out["t6_std_xte_m"]         = float(abs_d.std(ddof=1)) if abs_d.size >= 2 else float("nan")
        out["t6_rmse_xte_m"]        = float(np.sqrt(np.mean(signed ** 2)))
        out["t6_mean_signed_xte_m"] = float(signed.mean())
        out["t6_path_coverage_frac"] = cov
        out["t6_n_global_poses"]    = int(poly.shape[0])
        return out

    # ----- Tier-C: straight-line fallback (legacy HDF5s) -------------------
    if goal is None:
        return out
    sx, sy = float(robot_pose[0, 0]), float(robot_pose[0, 1])
    gx, gy = float(goal[-1, 0]), float(goal[-1, 1])
    dx, dy = gx - sx, gy - sy
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return out
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    px = robot_pose[:, 0].astype(np.float64) - sx
    py = robot_pose[:, 1].astype(np.float64) - sy
    signed = px * nx + py * ny
    along  = px * ux + py * uy
    abs_xte = np.abs(signed)
    out["t6_has_global_path"]   = True
    out["t6_xte_ref"]           = "straight_line"
    out["t6_mean_xte_m"]        = float(abs_xte.mean())
    out["t6_median_xte_m"]      = float(np.median(abs_xte))
    out["t6_max_xte_m"]         = float(abs_xte.max())
    out["t6_p95_xte_m"]         = float(np.quantile(abs_xte, 0.95))
    out["t6_std_xte_m"]         = float(abs_xte.std(ddof=1)) if abs_xte.size >= 2 else float("nan")
    out["t6_rmse_xte_m"]        = float(np.sqrt(np.mean(signed ** 2)))
    out["t6_mean_signed_xte_m"] = float(signed.mean())
    out["t6_path_coverage_frac"] = float(np.clip(along / L, 0.0, 1.0).max())
    out["t6_n_global_poses"]    = 2
    return out


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def compute_row(
    h5_path: str | Path,
    map_yaml: Optional[str | Path] = None,
) -> dict:
    """Compute every available metric for one HDF5 file. Returns a row dict
    keyed by RESULTS_FIELDS columns (plus identity columns)."""
    h5_path = str(h5_path)
    field = _load_obstacle_field(str(map_yaml)) if map_yaml else None

    with h5py.File(h5_path, "r") as f:
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}
        robot_pose      = f["robot_pose"][:]      if "robot_pose"      in f else None
        selected_action = f["selected_action"][:] if "selected_action" in f else None
        goal_ds         = f["goal"][:]            if "goal"            in f else None
        sim_time        = f["sim_time"][:]        if "sim_time"        in f else None
        critic_weights  = f["critic_weights_applied"][:] if "critic_weights_applied" in f else None
        critic_names    = list(attrs.get("critic_names", []))

        # Tier-6 XTE references (new in 2026-05-16). Time-active first,
        # final-plan second; straight-line is the last-resort fallback
        # inside _tier6 itself for legacy HDF5s that have neither.
        plan_history: list[tuple[float, np.ndarray]] | None = None
        if "global_path_plans" in f:
            g = f["global_path_plans"]
            if "plan_times" in g:
                pts = g["plan_times"][:]
                plan_history = []
                for i, t in enumerate(pts):
                    key = f"plan_{i:05d}"
                    if key in g:
                        plan_history.append((float(t), g[key][:]))
        final_plan = f["global_path_poses"][:] if "global_path_poses" in f else None

    # Coerce attrs to native types where possible.
    def _A(k, default=None):
        v = attrs.get(k, default)
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode("utf-8", errors="replace")
        if isinstance(v, np.generic):
            return v.item()
        return v

    success      = bool(_A("success", False))
    path_length  = float(_A("path_length_m", float("nan")))
    duration_s   = float(_A("time_to_goal_s", float("nan")))
    coll_count   = int(_A("collision_count", 0))

    t1 = _tier1(robot_pose, field, coll_count)
    t2 = _tier2(robot_pose, goal_ds, sim_time, attrs={
        "success": success, "path_length_m": path_length,
        "time_to_goal_s": duration_s,
    })
    t3 = _tier3(selected_action, sim_time, path_length, duration_s)
    t4 = _tier4(selected_action, sim_time)
    t5 = _tier5(critic_weights, critic_names, sim_time)
    t6 = _tier6(robot_pose, sim_time, goal_ds, plan_history, final_plan)

    row: dict = {}
    for d in (t1, t2, t3, t4, t5, t6):
        row.update(d)
    row["hdf5_path"]   = h5_path
    row["t3_path_length_m"] = path_length  # alias the attr (clobber numeric)

    # Safety-gated SPL — paper's headline metric. A collision sets SPL to 0
    # even if the goal was reached, so the top-line number cannot reward
    # an unsafe success. Plain t2_spl is kept untouched for the TOPSIS
    # criterion (collision rate sits in the Safety category separately).
    spl = row.get("t2_spl", float("nan"))
    collided = bool(row.get("t1_collision", False))
    if isinstance(spl, float) and math.isnan(spl):
        row["t2_spl_safe"] = float("nan")
    else:
        row["t2_spl_safe"] = 0.0 if collided else float(spl)
    return row


__all__ = ["compute_row", "_load_obstacle_field"]
