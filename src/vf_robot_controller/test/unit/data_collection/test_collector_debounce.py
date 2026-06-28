"""Unit tests for vf_controller.data_collection.goal_debouncer.

Reproduces the "many files per single nav goal" bug as a deterministic
test (without rclpy) and proves debounce + cooldown fix it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vf_controller.data_collection.goal_debouncer import (
    DebouncerConfig,
    Decision,
    GoalDebouncer,
    same_goal,
)


# --------------------------------------------------------------- helpers
def _g(x, y, yaw=0.0):
    return np.array([x, y, yaw], dtype=np.float32)


def _drive(d: GoalDebouncer, goal, t0, n=100, dt=0.05):
    """Send `n` proposals with the same goal at dt-spaced timestamps.

    Returns the list of decisions and the timestamp of the last proposal.
    """
    decisions = []
    t = t0
    for _ in range(n):
        decisions.append(d.propose(goal, now=t))
        t += dt
    return decisions, t


# ------------------------------------------------------------- predicate
def test_same_goal_predicate_xy_and_yaw():
    a = _g(1.0, 1.0, 0.0)
    assert same_goal(a, _g(1.05, 1.05, 0.0), 0.5, 0.35)
    # XY too far.
    assert not same_goal(a, _g(2.0, 1.0, 0.0), 0.5, 0.35)
    # Yaw too different.
    assert not same_goal(a, _g(1.0, 1.0, math.pi / 2), 0.5, 0.35)
    # Wraparound: -pi vs +pi - 0.01 should be close.
    assert same_goal(_g(0, 0, -math.pi), _g(0, 0, math.pi - 0.01), 0.5, 0.35)


# ------------------------------------- 100 replans → exactly one commit
def test_hundred_replans_single_commit():
    d = GoalDebouncer(DebouncerConfig(debounce_s=0.5, cooldown_s=2.0,
                                      dedup_radius_m=0.5, yaw_eps_rad=0.35))
    goal = _g(5.0, 0.5, 0.0)
    decisions, _ = _drive(d, goal, t0=0.0, n=100, dt=0.05)
    n_commit = sum(1 for x in decisions if x is Decision.COMMIT)
    n_pending = sum(1 for x in decisions if x is Decision.PENDING)
    assert n_commit == 1, "exactly one COMMIT for a stable goal stream"
    assert n_pending >= 9  # 0.5 s / 0.05 s = 10 PENDINGs before COMMIT


# ----------------------------------------- replan after commit = REFRESH
def test_replans_after_commit_are_refresh():
    d = GoalDebouncer(DebouncerConfig(debounce_s=0.5, cooldown_s=2.0,
                                      dedup_radius_m=0.5, yaw_eps_rad=0.35))
    goal = _g(5.0, 0.5, 0.0)
    # Drive past commit.
    _drive(d, goal, t0=0.0, n=20, dt=0.05)  # commits at t≈0.5
    d.mark_committed(goal)
    # 100 more replans of the same goal (small XY noise) — all must be
    # REFRESH, never COMMIT.
    for i in range(100):
        noisy = _g(5.0 + 0.01 * (i % 3), 0.5, 0.0)
        dec = d.propose(noisy, now=2.0 + 0.05 * i)
        assert dec is Decision.REFRESH, f"step {i}: {dec}"


# ----------------------------------- close + same goal for cooldown_s = IGNORE
def test_post_close_cooldown_blocks_same_goal():
    cfg = DebouncerConfig(debounce_s=0.5, cooldown_s=2.0,
                          dedup_radius_m=0.5, yaw_eps_rad=0.35)
    d = GoalDebouncer(cfg)
    goal = _g(5.0, 0.5, 0.0)

    # Open + close the episode normally.
    _drive(d, goal, t0=0.0, n=20, dt=0.05)
    d.mark_committed(goal)
    d.mark_closed(goal, now=10.0)

    # Within the 2 s cooldown, 50 replans of the same goal must all IGNORE.
    decisions = []
    for i in range(50):
        decisions.append(d.propose(goal, now=10.0 + 0.02 * i))
    assert all(x is Decision.IGNORE for x in decisions), (
        "every replan within cooldown of the just-closed goal must IGNORE"
    )

    # After cooldown elapses, a stable stream commits exactly once again.
    after_decisions, _ = _drive(d, goal, t0=12.5, n=20, dt=0.05)
    assert sum(1 for x in after_decisions if x is Decision.COMMIT) == 1


# ----------------------- two distinct goals → 2 commits (no cross-talk)
def test_two_distinct_goals_open_two_episodes():
    cfg = DebouncerConfig(debounce_s=0.5, cooldown_s=2.0,
                          dedup_radius_m=0.5, yaw_eps_rad=0.35)
    d = GoalDebouncer(cfg)

    goal_a = _g(5.0, 0.5, 0.0)
    decisions_a, t = _drive(d, goal_a, t0=0.0, n=20, dt=0.05)
    assert sum(1 for x in decisions_a if x is Decision.COMMIT) == 1
    d.mark_committed(goal_a)
    d.mark_closed(goal_a, now=t + 1.0)

    # Wait past cooldown, then propose a *different* goal 4 m away.
    goal_b = _g(9.0, 0.5, 0.0)
    decisions_b, _ = _drive(d, goal_b, t0=t + 5.0, n=20, dt=0.05)
    assert sum(1 for x in decisions_b if x is Decision.COMMIT) == 1


# ---------------------------------------- /plan flicker between A and B
def test_alternating_proposals_resets_debounce():
    cfg = DebouncerConfig(debounce_s=0.5, cooldown_s=2.0,
                          dedup_radius_m=0.5, yaw_eps_rad=0.35)
    d = GoalDebouncer(cfg)

    a = _g(5.0, 0.5, 0.0)
    b = _g(9.0, 0.5, 0.0)
    # A flickering planner switches every 0.1 s for 1.0 s.
    decisions = []
    for i in range(10):
        decisions.append(d.propose(a if i % 2 == 0 else b, now=0.1 * i))
    # Neither stayed stable for 0.5 s — so no COMMIT.
    assert all(x is Decision.PENDING for x in decisions)


# ------------------------------------- poll() commits when /plan stops
def test_poll_commits_when_no_more_proposals():
    cfg = DebouncerConfig(debounce_s=0.5, cooldown_s=2.0,
                          dedup_radius_m=0.5, yaw_eps_rad=0.35)
    d = GoalDebouncer(cfg)
    goal = _g(5.0, 0.5, 0.0)

    # Single proposal (e.g. a one-shot /goal_pose), then silence.
    assert d.propose(goal, now=0.0) is Decision.PENDING
    # Before debounce_s elapses, poll says still pending.
    assert d.poll(now=0.4) is Decision.PENDING
    # After debounce_s, poll says commit.
    assert d.poll(now=0.6) is Decision.COMMIT


# ------------------------------------- yaw flip on Smac counts as new
def test_yaw_flip_does_not_match_committed():
    cfg = DebouncerConfig(debounce_s=0.5, cooldown_s=2.0,
                          dedup_radius_m=0.5, yaw_eps_rad=0.35)
    d = GoalDebouncer(cfg)
    goal_fwd = _g(5.0, 0.5, 0.0)
    d.mark_committed(goal_fwd)

    # Plan flips yaw 180 — but /plan dedup is XY-only on the committed
    # path so this should still REFRESH (we don't want yaw flips opening
    # a new file mid-flight). Yaw is a same_goal predicate term only for
    # debounce/pending tracking.
    flipped = _g(5.0, 0.5, math.pi)
    assert d.propose(flipped, now=1.0) is Decision.REFRESH


# -------------------------------------------- reset() clears everything
def test_reset_clears_all_state():
    cfg = DebouncerConfig()
    d = GoalDebouncer(cfg)
    goal = _g(5.0, 0.5, 0.0)
    d.propose(goal, now=0.0)
    d.mark_committed(goal)
    d.mark_closed(goal, now=10.0)
    d.reset()
    assert d.pending_goal is None
    assert d.committed_goal is None
    assert not d.writer_open
    # Same goal as the cleared cooldown should now be treated as a fresh
    # PENDING, not IGNORE.
    assert d.propose(goal, now=10.5) is Decision.PENDING


def test_xy_outside_dedup_is_a_new_pending():
    cfg = DebouncerConfig(debounce_s=0.5, cooldown_s=2.0,
                          dedup_radius_m=0.5, yaw_eps_rad=0.35)
    d = GoalDebouncer(cfg)
    a = _g(5.0, 0.5, 0.0)
    b = _g(7.0, 0.5, 0.0)  # 2 m away — different goal
    assert d.propose(a, now=0.0) is Decision.PENDING
    assert d.propose(b, now=0.05) is Decision.PENDING
    # Pending replaced; debounce timer reset; so even at t=0.5 we're
    # only 0.45 s past *b*'s first sighting → still PENDING.
    assert d.propose(b, now=0.5) is Decision.PENDING
    # 0.55 s after first sighting of b → COMMIT.
    assert d.propose(b, now=0.6) is Decision.COMMIT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
