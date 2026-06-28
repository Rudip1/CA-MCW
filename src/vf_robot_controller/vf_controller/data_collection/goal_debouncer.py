"""
goal_debouncer.py — pure-Python state machine for goal debounce + cooldown.

Extracted from data_collector_node.py so the logic that fixes the
"many files per single goal" bug is unit-testable without rclpy.

Behaviour (mirrors plan.md §3)
------------------------------
The debouncer tracks at most three goals:

  committed    — the goal of the currently open episode (if any)
  pending      — a candidate goal not yet old enough to commit
  cooldown     — the just-closed goal, latched out for goal_cooldown_s

Every incoming proposal goes through the same pipeline:

  1. If a writer is open and the proposal matches `committed`, treat it as
     a replan refresh and return `Decision.REFRESH` (no episode change).
  2. If `cooldown` is active and the proposal matches it, return
     `Decision.IGNORE`. (Stops the post-close /plan re-opening the same goal.)
  3. If no `pending` exists, or it differs from this proposal, set
     `pending` to this proposal, stamp `pending_first_seen_t = now`,
     return `Decision.PENDING`.
  4. Else (proposal == pending) refresh `pending` (latest pose snapshot)
     and check duration: if `now - pending_first_seen_t >= debounce_s`,
     return `Decision.COMMIT` (caller opens an episode). Else `PENDING`.

The "same goal" predicate is XY-distance ≤ `dedup_radius_m` AND yaw delta
≤ `yaw_eps_rad`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class Decision(Enum):
    PENDING = "pending"     # debounce timer still ticking
    COMMIT = "commit"       # open an episode for the pending goal
    REFRESH = "refresh"     # replan of the active goal — keep going
    IGNORE = "ignore"       # cooldown filtered this out


def _angle_diff(a: float, b: float) -> float:
    """Signed delta b - a wrapped to (-pi, pi]."""
    return (b - a + math.pi) % (2.0 * math.pi) - math.pi


def _xy_close(a: np.ndarray, b: np.ndarray, r: float) -> bool:
    return float(np.linalg.norm(a[:2] - b[:2])) <= r


def same_goal(a: np.ndarray, b: np.ndarray,
              dedup_radius_m: float, yaw_eps_rad: float) -> bool:
    if not _xy_close(a, b, dedup_radius_m):
        return False
    return abs(_angle_diff(float(a[2]), float(b[2]))) <= yaw_eps_rad


@dataclass
class DebouncerConfig:
    debounce_s: float = 0.5
    cooldown_s: float = 2.0
    dedup_radius_m: float = 0.5
    yaw_eps_rad: float = 0.35   # ~20 deg


class GoalDebouncer:
    """State machine. ROS-free; the caller injects the wall clock via `now`.

    Usage:
        d = GoalDebouncer(DebouncerConfig())
        decision = d.propose(goal=np.array([x, y, yaw]), now=t)
        if decision is Decision.COMMIT:
            d.mark_committed(goal)
            ...open episode...
        # later:
        d.mark_closed(goal, now=t_close)
    """

    def __init__(self, config: Optional[DebouncerConfig] = None) -> None:
        self.cfg = config or DebouncerConfig()
        self._committed: Optional[np.ndarray] = None
        self._pending: Optional[np.ndarray] = None
        self._pending_first_seen_t: float = 0.0
        self._cooldown_goal: Optional[np.ndarray] = None
        self._cooldown_until_t: float = 0.0
        # Whether an episode is currently open (driven by mark_committed /
        # mark_closed). Used to distinguish REFRESH from PENDING.
        self._writer_open: bool = False

    # --------------------------------------------------------- inspectors
    @property
    def committed_goal(self) -> Optional[np.ndarray]:
        return None if self._committed is None else self._committed.copy()

    @property
    def pending_goal(self) -> Optional[np.ndarray]:
        return None if self._pending is None else self._pending.copy()

    @property
    def writer_open(self) -> bool:
        return self._writer_open

    # ------------------------------------------------------- driver entry
    def propose(self, goal: np.ndarray, *, now: float) -> Decision:
        """Process an inbound goal candidate."""
        cfg = self.cfg

        # 1. Replan refresh of the active goal.
        if (
            self._writer_open
            and self._committed is not None
            and _xy_close(goal, self._committed, cfg.dedup_radius_m)
        ):
            return Decision.REFRESH

        # 2. Cooldown filter for the just-closed goal.
        if (
            self._cooldown_goal is not None
            and now < self._cooldown_until_t
            and _xy_close(goal, self._cooldown_goal, cfg.dedup_radius_m)
        ):
            return Decision.IGNORE

        # 3. New pending or change of pending.
        if self._pending is None or not same_goal(
            self._pending, goal,
            cfg.dedup_radius_m, cfg.yaw_eps_rad,
        ):
            self._pending = goal.copy()
            self._pending_first_seen_t = now
            return Decision.PENDING

        # 4. Same as pending — refresh snapshot, evaluate duration.
        self._pending = goal.copy()
        if now - self._pending_first_seen_t >= cfg.debounce_s:
            self._enter_committed(goal)
            return Decision.COMMIT
        return Decision.PENDING

    def poll(self, *, now: float) -> Decision:
        """Re-check pending goal without a new proposal.

        Required because /goal_pose may publish only once and /plan may
        stop firing before debounce_s elapses; without this, a single
        published goal would never commit.
        """
        if self._pending is None or self._writer_open:
            return Decision.PENDING
        if now - self._pending_first_seen_t < self.cfg.debounce_s:
            return Decision.PENDING
        committed = self._pending
        self._enter_committed(committed)
        return Decision.COMMIT

    def _enter_committed(self, goal: np.ndarray) -> None:
        """Internal: a COMMIT was just emitted. Latch state so subsequent
        propose() calls of the same goal return REFRESH instead of
        re-emitting COMMIT. The caller is expected to follow up with
        ``mark_committed`` (idempotent confirmation) and eventually
        ``mark_closed``.
        """
        self._committed = goal.copy()
        self._writer_open = True
        self._pending = None
        self._pending_first_seen_t = 0.0

    # --------------------------------------------------- state transitions
    def mark_committed(self, goal: np.ndarray) -> None:
        """Confirmation that the caller has opened an episode for this goal.

        Idempotent: ``propose`` already latches state on COMMIT. This is
        a hook for the caller to refresh `committed` to its preferred
        snapshot (e.g. the pose from /goal_pose vs. the planner's slightly
        offset endpoint).
        """
        self._committed = goal.copy()
        self._writer_open = True
        self._pending = None
        self._pending_first_seen_t = 0.0

    def mark_closed(self, goal: Optional[np.ndarray], *, now: float) -> None:
        """Caller closed the active episode. Start cooldown for that goal."""
        # Prefer the explicitly-passed goal; fall back to last committed.
        latch = goal if goal is not None else self._committed
        if latch is not None:
            self._cooldown_goal = latch.copy()
            self._cooldown_until_t = now + self.cfg.cooldown_s
        self._committed = None
        self._writer_open = False

    def reset(self) -> None:
        self._committed = None
        self._pending = None
        self._pending_first_seen_t = 0.0
        self._cooldown_goal = None
        self._cooldown_until_t = 0.0
        self._writer_open = False


__all__ = [
    "Decision",
    "DebouncerConfig",
    "GoalDebouncer",
    "same_goal",
]
