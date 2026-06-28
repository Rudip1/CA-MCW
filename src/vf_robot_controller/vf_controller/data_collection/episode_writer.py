"""
episode_writer.py — Phase 7

Writes one HDF5 file per navigation episode, schema per docs/data_format.md.

This module is intentionally ROS-free so it can be unit-tested with synthetic
inputs. The ROS sidecar (data_collector_node.py) drives it.

Buffering policy:
  - All datasets are created with chunks of (chunk_rows, ...) and gzip-4
    compression (per phase rules: chunked streaming, do not flush per cycle).
  - The writer keeps per-dataset Python lists in memory; flush() appends them
    in one h5py write (cheap relative to per-row writes), and flush() is
    called by the sidecar every ~1 s and at episode end.

Note: don't crash on missing topics — pad with NaN/zeros.
"""
from __future__ import annotations

import datetime
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np


# Per-cycle row schema. Sidecar fills these from cached topic snapshots.
@dataclass
class CycleRow:
    features: np.ndarray            # shape (D,)  float32; NaN-padded if missing
    critic_costs: np.ndarray        # shape (K,)  float32
    critic_weights: np.ndarray      # shape (K,)  float32
    selected_action: np.ndarray     # shape (3,)  float32   [vx, vy, wz]
    robot_pose: np.ndarray          # shape (3,)  float32   [x, y, theta]
    goal: np.ndarray                # shape (3,)  float32   [x, y, theta]
    dynamic_obstacles: Optional[np.ndarray] = None  # (M, 5) float32 or None
    sim_time: float = float('nan')  # simulation clock seconds; NaN in real time


@dataclass
class EpisodeMetadata:
    scenario_id: str = "unknown_scenario"
    seed: int = 0
    controller_mode: str = "collect"
    weight_provider: str = "fixed"
    channels_config: str = "channels_v1"
    channel_names: Sequence[str] = field(default_factory=list)
    channel_dims: Sequence[int] = field(default_factory=list)
    critic_names: Sequence[str] = field(default_factory=list)


@dataclass
class EpisodeOutcome:
    success: bool = False
    collision_count: int = 0
    time_to_goal_s: float = float("nan")
    path_length_m: float = float("nan")
    mean_clearance_m: float = float("nan")
    goal_reached_at_step: int = -1


def _git_commit() -> str:
    """Capture HEAD commit at episode start (a known startup pitfall)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _isoformat_now() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


def episode_filename(scenario_id: str, seed: int,
                     timestamp: Optional[datetime.datetime] = None) -> str:
    """Per docs/data_format.md: {scenario_id}_seed{seed}_{YYYYMMDD_HHMMSS}.h5"""
    ts = timestamp or datetime.datetime.now()
    return "{sid}_seed{seed}_{stamp}.h5".format(
        sid=scenario_id, seed=seed, stamp=ts.strftime("%Y%m%d_%H%M%S"))


class EpisodeWriter:
    """
    HDF5 writer that streams episode rows to disk.

    Lifecycle:
      w = EpisodeWriter(path, feature_dim=D, critic_count=K, max_obstacles=M,
                        meta=meta)
      w.append(row)               # in-memory only
      w.flush()                   # write buffered rows to HDF5 (call ~1Hz)
      w.close(outcome=outcome)    # writes outcome attrs and closes file
    """

    DEFAULT_CHUNK_ROWS = 256
    GZIP_LEVEL = 4

    def __init__(
        self,
        path: str,
        feature_dim: int,
        critic_count: int,
        meta: EpisodeMetadata,
        max_obstacles: int = 0,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
    ) -> None:
        self.path = path
        self.feature_dim = int(max(1, feature_dim))
        self.critic_count = int(max(1, critic_count))
        self.max_obstacles = int(max(0, max_obstacles))
        self.chunk_rows = int(max(1, chunk_rows))

        self._rows: List[CycleRow] = []
        self._closed = False
        self._row_count = 0  # rows already flushed to disk

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        self.h5 = h5py.File(path, "w")
        self._create_datasets()
        self._write_start_attrs(meta)

    # ------------------------------------------------------------------ setup
    def _create_datasets(self) -> None:
        D = self.feature_dim
        K = self.critic_count
        M = self.max_obstacles
        gz = self.GZIP_LEVEL
        ck = self.chunk_rows

        kw = dict(compression="gzip", compression_opts=gz, dtype="float32")

        self.h5.create_dataset("features", shape=(0, D),
                               maxshape=(None, D), chunks=(ck, D), **kw)
        self.h5.create_dataset("critic_costs", shape=(0, K),
                               maxshape=(None, K), chunks=(ck, K), **kw)
        self.h5.create_dataset("critic_weights_applied", shape=(0, K),
                               maxshape=(None, K), chunks=(ck, K), **kw)
        self.h5.create_dataset("selected_action", shape=(0, 3),
                               maxshape=(None, 3), chunks=(ck, 3), **kw)
        self.h5.create_dataset("robot_pose", shape=(0, 3),
                               maxshape=(None, 3), chunks=(ck, 3), **kw)
        self.h5.create_dataset("goal", shape=(0, 3),
                               maxshape=(None, 3), chunks=(ck, 3), **kw)
        self.h5.create_dataset("sim_time", shape=(0,),
                               maxshape=(None,), chunks=(ck,),
                               compression="gzip", compression_opts=gz,
                               dtype="float64")
        if M > 0:
            self.h5.create_dataset(
                "dynamic_obstacles", shape=(0, M, 5),
                maxshape=(None, M, 5), chunks=(ck, M, 5), **kw)

    def _write_start_attrs(self, meta: EpisodeMetadata) -> None:
        a = self.h5.attrs
        a["scenario_id"] = meta.scenario_id
        a["seed"] = int(meta.seed)
        a["controller_mode"] = meta.controller_mode
        a["weight_provider"] = meta.weight_provider
        a["channels_config"] = meta.channels_config
        a["channel_names"] = np.array(
            list(meta.channel_names), dtype=h5py.string_dtype())
        a["channel_dims"] = np.array(list(meta.channel_dims), dtype=np.int32)
        a["critic_names"] = np.array(
            list(meta.critic_names), dtype=h5py.string_dtype())
        a["start_time_iso"] = _isoformat_now()
        a["git_commit"] = _git_commit()

    # ------------------------------------------------------------------ rows
    def append(self, row: CycleRow) -> None:
        if self._closed:
            raise RuntimeError("EpisodeWriter is closed")
        self._rows.append(row)

    def __len__(self) -> int:
        return self._row_count + len(self._rows)

    # --------------------------------------------------------------- flushing
    def flush(self) -> None:
        if self._closed or not self._rows:
            return

        n = len(self._rows)
        D = self.feature_dim
        K = self.critic_count
        M = self.max_obstacles

        feats = np.empty((n, D), dtype=np.float32)
        ccst = np.empty((n, K), dtype=np.float32)
        wts = np.empty((n, K), dtype=np.float32)
        act = np.empty((n, 3), dtype=np.float32)
        pose = np.empty((n, 3), dtype=np.float32)
        goal = np.empty((n, 3), dtype=np.float32)
        if M > 0:
            obs = np.full((n, M, 5), np.nan, dtype=np.float32)

        sim_t = np.empty((n,), dtype=np.float64)

        for i, r in enumerate(self._rows):
            feats[i] = self._fit(r.features, D)
            ccst[i] = self._fit(r.critic_costs, K)
            wts[i] = self._fit(r.critic_weights, K)
            act[i] = self._fit(r.selected_action, 3)
            pose[i] = self._fit(r.robot_pose, 3)
            goal[i] = self._fit(r.goal, 3)
            sim_t[i] = r.sim_time
            if M > 0 and r.dynamic_obstacles is not None:
                arr = np.asarray(r.dynamic_obstacles, dtype=np.float32)
                rows = min(arr.shape[0], M)
                if arr.ndim == 2 and arr.shape[1] == 5 and rows > 0:
                    obs[i, :rows, :] = arr[:rows, :]

        self._extend("features", feats)
        self._extend("critic_costs", ccst)
        self._extend("critic_weights_applied", wts)
        self._extend("selected_action", act)
        self._extend("robot_pose", pose)
        self._extend("goal", goal)
        self._extend("sim_time", sim_t)
        if M > 0:
            self._extend("dynamic_obstacles", obs)

        self._row_count += n
        self._rows.clear()
        self.h5.flush()

    @staticmethod
    def _fit(vec: np.ndarray, dim: int) -> np.ndarray:
        """Coerce vec to float32 length-`dim`. NaN-pad / truncate as needed."""
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size == dim:
            return v
        out = np.full((dim,), np.nan, dtype=np.float32)
        n = min(v.size, dim)
        if n > 0:
            out[:n] = v[:n]
        return out

    def _extend(self, name: str, block: np.ndarray) -> None:
        ds = self.h5[name]
        old = ds.shape[0]
        ds.resize(old + block.shape[0], axis=0)
        ds[old:old + block.shape[0]] = block

    # ---------------------------------------------------------- rich datasets
    def write_global_path(self, poses: list) -> None:
        """Store the global path computed by the planner (one-time write).

        ``poses`` is a list of [x, y, yaw] triples from global_path_cache.
        Written as ``global_path_poses`` (N, 3) float64 at the root level.
        No-op if already written or writer is closed.
        """
        if self._closed or "global_path_poses" in self.h5:
            return
        arr = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        self.h5.create_dataset("global_path_poses", data=arr,
                               compression="gzip", compression_opts=self.GZIP_LEVEL)
        self.h5.flush()

    def write_global_path_plans(
        self, history: "list[tuple[float, np.ndarray]]",
    ) -> None:
        """Store the full /plan history seen during the episode.

        ``history`` is a chronologically ordered list of
        ``(sim_time_seconds, poses_array)`` tuples, one entry per /plan
        message received while the writer was open. ``poses_array`` is
        a ``(N_i, 3)`` ``[x, y, yaw]`` polyline.

        Written under group ``global_path_plans/`` with:
          - ``plan_times`` : (M,) float64 — header.stamp of each plan
          - ``plan_NNNNN`` : (N_i, 3) float64 — pose polylines
                             (zero-padded index, matches ``plan_times``)

        No-op if already written, writer closed, or history empty.
        """
        if self._closed or "global_path_plans" in self.h5 or not history:
            return
        g = self.h5.create_group("global_path_plans")
        times = np.asarray([t for t, _ in history], dtype=np.float64)
        g.create_dataset(
            "plan_times", data=times,
            compression="gzip", compression_opts=self.GZIP_LEVEL,
        )
        for i, (_, poses) in enumerate(history):
            arr = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
            if arr.shape[0] == 0:
                continue
            g.create_dataset(
                f"plan_{i:05d}", data=arr,
                compression="gzip", compression_opts=self.GZIP_LEVEL,
            )
        self.h5.flush()

    def write_costmap_snapshot(
        self,
        step: int,
        data: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        sim_t: float = float('nan'),
    ) -> None:
        """Store one local costmap grid snapshot under ``/costmaps/snap_NNNNN``.

        ``data`` must be a (H, W) int8/uint8 array of OccupancyGrid values
        (0–100 = free–lethal, -1 = unknown).  Attributes record the sim time,
        grid resolution, grid origin in the map frame, and the robot pose at
        the snapshot moment so the grid can be re-projected during analysis.
        """
        if self._closed:
            return
        grp_name = f"costmaps/snap_{step:05d}"
        if grp_name in self.h5:
            return
        grp = self.h5.require_group("costmaps")
        snap = grp.create_dataset(
            f"snap_{step:05d}",
            data=np.asarray(data, dtype=np.int8),
            compression="gzip", compression_opts=self.GZIP_LEVEL,
        )
        snap.attrs["t"] = float(sim_t)
        snap.attrs["resolution"] = float(resolution)
        snap.attrs["origin_x"] = float(origin_x)
        snap.attrs["origin_y"] = float(origin_y)
        snap.attrs["robot_x"] = float(robot_x)
        snap.attrs["robot_y"] = float(robot_y)
        snap.attrs["robot_yaw"] = float(robot_yaw)
        self.h5.flush()

    # ----------------------------------------------------------------- close
    def close(self, outcome: Optional[EpisodeOutcome] = None) -> None:
        if self._closed:
            return
        try:
            self.flush()
            if outcome is not None:
                a = self.h5.attrs
                a["success"] = bool(outcome.success)
                a["collision_count"] = int(outcome.collision_count)
                a["time_to_goal_s"] = float(outcome.time_to_goal_s)
                a["path_length_m"] = float(outcome.path_length_m)
                a["mean_clearance_m"] = float(outcome.mean_clearance_m)
                a["goal_reached_at_step"] = int(outcome.goal_reached_at_step)
            a = self.h5.attrs
            a["end_time_iso"] = _isoformat_now()
            a["num_steps"] = int(self._row_count)
        finally:
            try:
                self.h5.close()
            except Exception:
                pass
            self._closed = True

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --------------------------------------------------------------- inspect
    def current_size_bytes(self) -> int:
        """Best-effort size of the on-disk file (post-flush)."""
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def num_steps(self) -> int:
        """Steps already flushed to disk."""
        return self._row_count


def open_episode_reader(path: str) -> Dict[str, np.ndarray]:
    """
    Tiny convenience reader. Phase 8 will replace with full EpisodeReader
    class. Returns a dict of dataset arrays + an `attrs` dict.
    """
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = f[k][...]
        out["attrs"] = {k: f.attrs[k] for k in f.attrs.keys()}  # type: ignore[assignment]
    return out
