"""
world_reconstructor.py — Phase 9.2

Reads one recorded HDF5 episode and exposes a per-timestep view of the
world that the replay oracle can query with **future** information
available. Specifically:

  * robot pose history (replay-frame ground truth)
  * dynamic-obstacle positions at any future step `t+τ`
  * per-id finite-difference velocities (vx_logged is currently NaN —
    see ``_on_obstacles_marker`` in ``data_collector_node.py``; we
    reconstruct velocity here)
  * goal pose
  * recorded critic costs / applied weights, for cross-checking only

This module is intentionally ROS-free. It does not load a costmap (the
HDF5 schema does not store one); ``WorldReplay.map_costmap`` is a stub
hook so the oracle MPPI can plug in a static-map loader later if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

import h5py
import numpy as np


# Default writer cadence (data_collector_node.py runs at 10 Hz). Used as a
# fallback when start/end timestamps cannot be parsed. Override via
# ``WorldReplay.dt`` after construction if your dataset was collected at a
# different rate.
DEFAULT_WRITER_DT = 0.1

# Sentinel ID we use when the recorded id is NaN (padding row).
_INVALID_ID = -1


@dataclass
class ObstacleSnapshot:
    """One timestep of dynamic obstacles, post-padding-removal.

    Attributes:
      ids:        (M_active,) int — marker IDs present at this step.
      positions:  (M_active, 2) float — [x, y] per active obstacle.
      velocities: (M_active, 2) float — finite-diff [vx, vy] per active
                  obstacle. NaN at the very first frame an ID is observed.
    """

    ids: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray

    @property
    def num_active(self) -> int:
        return int(self.ids.shape[0])

    def empty(self) -> bool:
        return self.num_active == 0


class WorldReplay:
    """Read-only view over one recorded episode.

    Construct via :func:`reconstruct`. Indexes are clamped to the episode
    horizon — ``state_at(num_steps)`` returns the last logged pose,
    ``obstacles_at(num_steps + 5)`` returns the last logged snapshot.
    This matches the agent's "no information past episode end" semantic
    rather than raising, which keeps the oracle's edge-of-episode logic
    simple.
    """

    def __init__(
        self,
        h5_path: str,
        *,
        robot_pose: np.ndarray,
        goal: np.ndarray,
        dynamic_obstacles: Optional[np.ndarray],
        critic_costs: np.ndarray,
        critic_weights_applied: np.ndarray,
        critic_names: List[str],
        scenario_id: str,
        dt: float,
    ) -> None:
        self.h5_path = h5_path
        self._robot_pose = np.asarray(robot_pose, dtype=np.float32)
        self._goal = np.asarray(goal, dtype=np.float32)
        self._dyn_raw = (
            np.asarray(dynamic_obstacles, dtype=np.float32)
            if dynamic_obstacles is not None
            else None
        )
        self._critic_costs = np.asarray(critic_costs, dtype=np.float32)
        self._critic_weights_applied = np.asarray(
            critic_weights_applied, dtype=np.float32
        )
        self.critic_names = list(critic_names)
        self.scenario_id = scenario_id
        self.dt = float(dt)

        self.num_steps = int(self._robot_pose.shape[0])
        self.has_dynamic_obstacles = self._dyn_raw is not None

        # Cache snapshot per timestep (small mem footprint vs recomputing
        # finite-diff on every query). Velocities are NaN at the first
        # frame an obstacle id is observed.
        self._snapshots: List[ObstacleSnapshot] = []
        if self.has_dynamic_obstacles:
            self._snapshots = self._build_snapshots()

    # ------------------------------------------------------------------ static
    @staticmethod
    def _clamp(t: int, hi: int) -> int:
        if t < 0:
            return 0
        if t >= hi:
            return hi - 1
        return t

    # ------------------------------------------------------------------ accessors
    def state_at(self, t: int) -> np.ndarray:
        """Robot pose [x, y, theta] at step t (clamped)."""
        ti = self._clamp(int(t), self.num_steps)
        return self._robot_pose[ti].copy()

    def goal_at(self, t: Optional[int] = None) -> np.ndarray:
        """Goal pose [x, y, theta]. Most episodes have a constant goal but
        the writer logs it per step, so we honour that. ``t=None`` returns
        the last logged goal."""
        if t is None:
            return self._goal[-1].copy()
        ti = self._clamp(int(t), self.num_steps)
        return self._goal[ti].copy()

    @property
    def goal(self) -> np.ndarray:
        """Convenience: final goal pose."""
        return self.goal_at(None)

    def obstacles_at(self, t: int) -> ObstacleSnapshot:
        """Dynamic-obstacle snapshot at step t (clamped). Empty snapshot
        if the episode logged no obstacles."""
        if not self.has_dynamic_obstacles:
            return _empty_snapshot()
        ti = self._clamp(int(t), self.num_steps)
        return self._snapshots[ti]

    def critic_costs_at(self, t: int) -> np.ndarray:
        ti = self._clamp(int(t), self.num_steps)
        return self._critic_costs[ti].copy()

    def critic_weights_applied_at(self, t: int) -> np.ndarray:
        ti = self._clamp(int(t), self.num_steps)
        return self._critic_weights_applied[ti].copy()

    @property
    def critic_count(self) -> int:
        return int(self._critic_costs.shape[1]) if self._critic_costs.ndim == 2 else 0

    def map_costmap(self):
        """Hook for an external costmap loader. Phase 9.2 returns None;
        Phase 9.3's MPPI plugs in a static-map reader if needed."""
        return None

    # ------------------------------------------------------------------ snapshot build
    def _build_snapshots(self) -> List[ObstacleSnapshot]:
        """Walk the (T, M, 5) raw obstacle tensor once and produce a list
        of per-timestep snapshots with finite-diff velocities.

        Each row in dynamic_obstacles is [id, x, y, NaN, NaN]. NaN id rows
        are padding. Velocity at step t for id i is ``(p[t] - p[prev])/dt``
        where ``prev`` is the most recent step where id i was active; NaN
        on first observation.
        """
        assert self._dyn_raw is not None
        T = self._dyn_raw.shape[0]
        snapshots: List[ObstacleSnapshot] = [None] * T  # type: ignore[list-item]
        last_seen: Dict[int, Tuple[int, np.ndarray]] = {}

        for t in range(T):
            row = self._dyn_raw[t]  # (M, 5)
            ids_raw = row[:, 0]
            valid = np.isfinite(ids_raw)
            if not valid.any():
                snapshots[t] = _empty_snapshot()
                continue

            ids = ids_raw[valid].astype(np.int64)
            positions = row[valid][:, 1:3].astype(np.float32)
            velocities = np.full_like(positions, np.nan, dtype=np.float32)

            for j in range(ids.shape[0]):
                oid = int(ids[j])
                pos = positions[j]
                if oid in last_seen:
                    prev_t, prev_pos = last_seen[oid]
                    dt_eff = self.dt * max(1, t - prev_t)
                    velocities[j] = (pos - prev_pos) / dt_eff
                last_seen[oid] = (t, pos.copy())

            snapshots[t] = ObstacleSnapshot(
                ids=ids,
                positions=positions,
                velocities=velocities,
            )
        return snapshots


# =============================================================================
# Module entry point
# =============================================================================

_warned_no_obstacles: set = set()


def reconstruct(h5_path: str, *, dt_override: Optional[float] = None) -> WorldReplay:
    """Load one HDF5 episode and return a :class:`WorldReplay`.

    Args:
      h5_path: path to a recorded episode file.
      dt_override: if provided, use this timestep instead of inferring from
        ``start_time_iso`` / ``end_time_iso``. Useful for synthetic test
        episodes whose wall-clock runtime is not representative.

    Notes:
      If the file was written with ``max_obstacles == 0`` (older episodes
      collected before the dynamic-obstacle channel was wired up), the
      returned replay reports ``has_dynamic_obstacles == False`` and
      ``obstacles_at(t)`` returns an empty snapshot. A one-shot warning is
      emitted per file path.
    """
    with h5py.File(h5_path, "r") as f:
        robot_pose = f["robot_pose"][...]
        goal = f["goal"][...]
        critic_costs = f["critic_costs"][...]
        critic_weights_applied = f["critic_weights_applied"][...]
        dynamic_obstacles = (
            f["dynamic_obstacles"][...] if "dynamic_obstacles" in f else None
        )
        scenario_id = _decode(f.attrs.get("scenario_id", b"unknown"))
        critic_names = _decode_str_array(f.attrs.get("critic_names"))
        dt = float(dt_override) if dt_override is not None else _infer_dt(f.attrs)

    if dynamic_obstacles is None and h5_path not in _warned_no_obstacles:
        warnings.warn(
            f"world_reconstructor: {h5_path} has no dynamic_obstacles dataset; "
            "oracle will run as if the world is static.",
            stacklevel=2,
        )
        _warned_no_obstacles.add(h5_path)

    return WorldReplay(
        h5_path=h5_path,
        robot_pose=robot_pose,
        goal=goal,
        dynamic_obstacles=dynamic_obstacles,
        critic_costs=critic_costs,
        critic_weights_applied=critic_weights_applied,
        critic_names=critic_names,
        scenario_id=scenario_id,
        dt=dt,
    )


# =============================================================================
# helpers
# =============================================================================

def _empty_snapshot() -> ObstacleSnapshot:
    return ObstacleSnapshot(
        ids=np.empty((0,), dtype=np.int64),
        positions=np.empty((0, 2), dtype=np.float32),
        velocities=np.empty((0, 2), dtype=np.float32),
    )


def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray) and v.dtype.kind in ("S", "O"):
        return v.tolist()[0].decode("utf-8", errors="replace") if v.size else "unknown"
    return str(v)


def _decode_str_array(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        out: List[str] = []
        for x in v.tolist():
            out.append(x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x))
        return out
    return [str(v)]


def _infer_dt(attrs) -> float:
    """Best-effort dt: prefer (end_time - start_time)/num_steps; fall back to
    DEFAULT_WRITER_DT.

    Clamped to [0.01, 1.0] s — controllers run at 5–100 Hz; any inferred
    value outside that band is almost certainly a synthetic file whose
    wall-clock runtime is not representative (e.g. unit tests, episodes
    aborted before the writer flushed end_time_iso)."""
    try:
        from datetime import datetime

        start = _decode(attrs.get("start_time_iso", b""))
        end = _decode(attrs.get("end_time_iso", b""))
        n = int(attrs.get("num_steps", 0))
        if start and end and n > 1:
            t0 = datetime.fromisoformat(start)
            t1 = datetime.fromisoformat(end)
            secs = (t1 - t0).total_seconds()
            if secs > 0:
                inferred = float(secs / max(1, n - 1))
                if 0.01 <= inferred <= 1.0:
                    return inferred
    except Exception:
        pass
    return DEFAULT_WRITER_DT


__all__ = ["WorldReplay", "ObstacleSnapshot", "reconstruct"]
