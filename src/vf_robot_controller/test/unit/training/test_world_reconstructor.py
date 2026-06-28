"""
Phase 9.2 — unit tests for WorldReplay / world_reconstructor.

Builds a synthetic HDF5 via EpisodeWriter and checks the read-back semantics.
"""
import numpy as np
import pytest

from vf_controller.data_collection.episode_writer import (
    CycleRow, EpisodeMetadata, EpisodeOutcome, EpisodeWriter)
from vf_controller.training.oracle.world_reconstructor import (
    ObstacleSnapshot, WorldReplay, reconstruct)


def _meta(K: int = 3) -> EpisodeMetadata:
    return EpisodeMetadata(
        scenario_id="phase9_unit",
        seed=7,
        controller_mode="collect",
        weight_provider="fixed",
        channels_config="channels_v1",
        channel_names=["a", "b"],
        channel_dims=[3, 2],
        critic_names=["GoalCritic", "PathCritic", "ConstraintCritic"][:K],
    )


def _zeros_row(D: int, K: int) -> dict:
    return dict(
        features=np.zeros(D, dtype=np.float32),
        critic_costs=np.zeros(K, dtype=np.float32),
        critic_weights=np.full(K, 1.0 / K, dtype=np.float32),
        selected_action=np.zeros(3, dtype=np.float32),
        robot_pose=np.zeros(3, dtype=np.float32),
        goal=np.array([5.0, 0.0, 0.0], dtype=np.float32),
    )


def test_reconstruct_no_dynamic_obstacles(tmp_path):
    """Episode without dynamic_obstacles dataset must still reconstruct.
    Behaviour: has_dynamic_obstacles=False, obstacles_at returns empty."""
    p = str(tmp_path / "ep_static.h5")
    D, K, T = 5, 3, 4
    with EpisodeWriter(p, D, K, _meta(K), max_obstacles=0) as w:
        for t in range(T):
            row = _zeros_row(D, K)
            row["robot_pose"] = np.array([float(t) * 0.5, 0.0, 0.0], dtype=np.float32)
            w.append(CycleRow(dynamic_obstacles=None, **row))
        w.close(outcome=EpisodeOutcome(success=True, goal_reached_at_step=T - 1))

    with pytest.warns(UserWarning, match="no dynamic_obstacles"):
        rep = reconstruct(p)

    assert rep.num_steps == T
    assert rep.has_dynamic_obstacles is False
    assert rep.scenario_id == "phase9_unit"
    np.testing.assert_allclose(rep.state_at(2), [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(rep.goal, [5.0, 0.0, 0.0], atol=1e-6)
    snap = rep.obstacles_at(0)
    assert snap.empty()
    assert snap.num_active == 0


def test_reconstruct_with_persistent_obstacle_velocity(tmp_path):
    """Single obstacle moving at constant velocity along x.
    Velocity at t=0 must be NaN (first sighting), then equal to dx/dt
    for all subsequent steps."""
    p = str(tmp_path / "ep_one_obs.h5")
    D, K, T, M = 5, 3, 5, 2
    dt_writer = 0.1  # default
    vx_truth = 1.0  # m/s

    with EpisodeWriter(p, D, K, _meta(K), max_obstacles=M) as w:
        for t in range(T):
            row = _zeros_row(D, K)
            x = vx_truth * dt_writer * t
            obs = np.array([[42.0, x, 0.5, np.nan, np.nan]], dtype=np.float32)
            w.append(CycleRow(dynamic_obstacles=obs, **row))
        w.close(outcome=EpisodeOutcome(success=True))

    rep = reconstruct(p)
    assert rep.has_dynamic_obstacles is True

    snap0 = rep.obstacles_at(0)
    assert snap0.num_active == 1
    assert snap0.ids[0] == 42
    np.testing.assert_allclose(snap0.positions[0], [0.0, 0.5], atol=1e-6)
    assert np.isnan(snap0.velocities[0]).all(), "first sighting has NaN velocity"

    snap2 = rep.obstacles_at(2)
    assert snap2.num_active == 1
    np.testing.assert_allclose(
        snap2.velocities[0], [vx_truth, 0.0], atol=1e-3,
        err_msg="constant-velocity obstacle should yield clean finite-diff",
    )


def test_reconstruct_filters_padding_rows(tmp_path):
    """M=4 dataset where only 2 obstacles are active — the NaN padding
    rows must be stripped from the snapshot."""
    p = str(tmp_path / "ep_padded.h5")
    D, K, T, M = 5, 3, 3, 4
    with EpisodeWriter(p, D, K, _meta(K), max_obstacles=M) as w:
        for t in range(T):
            row = _zeros_row(D, K)
            obs = np.array(
                [
                    [1.0, 2.0 + t * 0.1, 0.0, np.nan, np.nan],
                    [2.0, -1.0, 1.5 - t * 0.1, np.nan, np.nan],
                ],
                dtype=np.float32,
            )
            w.append(CycleRow(dynamic_obstacles=obs, **row))
        w.close(outcome=EpisodeOutcome(success=True))

    rep = reconstruct(p)
    snap = rep.obstacles_at(1)
    assert snap.num_active == 2, "padding rows should be filtered"
    assert set(int(i) for i in snap.ids) == {1, 2}


def test_reconstruct_handles_intermittent_obstacle(tmp_path):
    """Obstacle 7 is observed at t=0 and t=2, missing at t=1. Velocity
    at t=2 is computed against t=0 with dt = 2 * dt_writer (gap-aware)."""
    p = str(tmp_path / "ep_gap.h5")
    D, K, T, M = 5, 3, 3, 1
    dt_writer = 0.1
    vx_eff = 0.5

    with EpisodeWriter(p, D, K, _meta(K), max_obstacles=M) as w:
        # t=0: obstacle 7 at x=0
        row0 = _zeros_row(D, K)
        obs0 = np.array([[7.0, 0.0, 0.0, np.nan, np.nan]], dtype=np.float32)
        w.append(CycleRow(dynamic_obstacles=obs0, **row0))
        # t=1: no obstacles (None — writer NaN-pads)
        row1 = _zeros_row(D, K)
        w.append(CycleRow(dynamic_obstacles=None, **row1))
        # t=2: obstacle 7 at x = vx_eff * 2 * dt_writer
        row2 = _zeros_row(D, K)
        x_t2 = vx_eff * (2 * dt_writer)
        obs2 = np.array([[7.0, x_t2, 0.0, np.nan, np.nan]], dtype=np.float32)
        w.append(CycleRow(dynamic_obstacles=obs2, **row2))
        w.close(outcome=EpisodeOutcome(success=True))

    rep = reconstruct(p)
    snap1 = rep.obstacles_at(1)
    assert snap1.num_active == 0, "no obstacle observed at t=1"

    snap2 = rep.obstacles_at(2)
    assert snap2.num_active == 1
    np.testing.assert_allclose(
        snap2.velocities[0], [vx_eff, 0.0], atol=1e-3,
        err_msg="gap-aware finite-diff should account for missing frame",
    )


def test_index_clamping(tmp_path):
    """state_at and obstacles_at clamp out-of-range indices instead of
    raising — keeps the oracle's edge-of-episode logic simple."""
    p = str(tmp_path / "ep_clamp.h5")
    D, K, T = 3, 2, 3
    with EpisodeWriter(p, D, K, _meta(K=2), max_obstacles=0) as w:
        for t in range(T):
            row = _zeros_row(D, K=2)
            row["robot_pose"] = np.array([float(t), 0.0, 0.0], dtype=np.float32)
            w.append(CycleRow(dynamic_obstacles=None, **row))
        w.close(outcome=EpisodeOutcome(success=True))

    rep = reconstruct(p)
    np.testing.assert_allclose(rep.state_at(-5), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(rep.state_at(99), [2.0, 0.0, 0.0])
