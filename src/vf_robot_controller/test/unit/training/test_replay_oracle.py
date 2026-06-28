"""
Phase 9.3 — unit tests for the pure-Python replay oracle.

Builds tiny synthetic episodes with known optimal behaviour, runs the
oracle with a small sample budget, and asserts that the recovered
weights move in the *expected direction* across critic dimensions.

These are sanity tests, not bit-exact: the oracle is a stochastic MPPI,
and the QP layer adds the KL prior. We assert ordering of weights, not
absolute values.
"""
import numpy as np
import pytest

from vf_controller.data_collection.episode_writer import (
    CycleRow, EpisodeMetadata, EpisodeOutcome, EpisodeWriter)
from vf_controller.training.oracle.replay_oracle import (
    MPPIConfig, ReplayOracle)
from vf_controller.training.oracle.world_reconstructor import reconstruct


# Critic-name set we'll use in synthetic episodes. Mirrors the recorded
# K=11 layout but pared down for speed.
CRITIC_NAMES_SHORT = [
    "WeightedConstraintCritic",      # 0
    "WeightedGoalCritic",            # 1
    "WeightedPathAlignCritic",       # 2
    "DynamicObstacleCritic",         # 3
]


def _meta(critic_names) -> EpisodeMetadata:
    return EpisodeMetadata(
        scenario_id="phase9_3_unit",
        seed=11,
        controller_mode="collect",
        weight_provider="fixed",
        channels_config="channels_v1",
        channel_names=["a", "b"],
        channel_dims=[3, 2],
        critic_names=critic_names,
    )


def _zero_row(D: int, K: int) -> dict:
    return dict(
        features=np.zeros(D, dtype=np.float32),
        critic_costs=np.zeros(K, dtype=np.float32),
        critic_weights=np.full(K, 1.0 / K, dtype=np.float32),
        selected_action=np.zeros(3, dtype=np.float32),
    )


def _write_episode(path, *, T, K, names, robot_traj, goal_xy, obstacles=None,
                   max_obstacles=0):
    """obstacles: list of T entries, each (M, 5) array or None."""
    D = 5
    with EpisodeWriter(path, D, K, _meta(names), max_obstacles=max_obstacles) as w:
        for t in range(T):
            row = _zero_row(D, K)
            row["robot_pose"] = np.array(
                [robot_traj[t, 0], robot_traj[t, 1], 0.0], dtype=np.float32)
            row["goal"] = np.array(
                [goal_xy[0], goal_xy[1], 0.0], dtype=np.float32)
            obs = None if obstacles is None else obstacles[t]
            w.append(CycleRow(dynamic_obstacles=obs, **row))
        w.close(outcome=EpisodeOutcome(success=True, goal_reached_at_step=T - 1))


@pytest.fixture(autouse=True)
def _need_cvxpy():
    pytest.importorskip("cvxpy")


def test_open_path_emphasises_goal_or_path(tmp_path):
    """Open environment, goal straight ahead, no dynamic obstacles.
    The oracle's i_star = trajectory closest to goal end-pose. Recovered
    weights should put more mass on Goal/PathAlign than on the
    dynamic-obstacle critic (which has no signal here)."""
    p = str(tmp_path / "open.h5")
    T, K = 3, 4
    robot_traj = np.zeros((T, 2))
    _write_episode(
        p, T=T, K=K, names=CRITIC_NAMES_SHORT,
        robot_traj=robot_traj, goal_xy=(5.0, 0.0),
        obstacles=None, max_obstacles=0,
    )

    rep = reconstruct(p, dt_override=0.1)
    cfg = MPPIConfig(
        horizon=20, n_samples=128, dt=0.1,
        rng_seed=0, qp_T=0.3, qp_lam=0.1,
    )
    oracle = ReplayOracle(rep, CRITIC_NAMES_SHORT, cfg)
    weights = oracle.compute_oracle_weights()

    assert weights.shape == (T, K)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-3)

    # Average over timesteps for a stable signal.
    mean_w = weights.mean(axis=0)
    # Goal (idx 1) + PathAlign (idx 2) should outweigh DynamicObstacle (idx 3),
    # which has no signal in a static episode.
    geometric = mean_w[1] + mean_w[2]
    dynamic = mean_w[3]
    assert geometric > dynamic + 0.05, (
        f"expected goal+path > dynamic in open episode; got mean_w={mean_w}"
    )


def test_dynamic_critic_weight_lifts_when_path_is_blocked(tmp_path):
    """Comparative: same robot/goal in two episodes, one with an obstacle
    parked on the straight-line path, one without. The recovered weight
    on DynamicObstacleCritic should be **higher** in the blocked episode
    than the open one. This is the right shape of signal for the
    Phase 9.7 context-bucket scatter (acceptance criterion #3)."""

    def _run(path, obstacles, max_obstacles):
        T, K = 2, 4
        robot_traj = np.zeros((T, 2))
        _write_episode(
            path, T=T, K=K, names=CRITIC_NAMES_SHORT,
            robot_traj=robot_traj, goal_xy=(5.0, 0.0),
            obstacles=obstacles, max_obstacles=max_obstacles,
        )
        rep = reconstruct(path, dt_override=0.1)
        cfg = MPPIConfig(
            horizon=30, n_samples=256, dt=0.1,
            collision_radius=0.4, obstacle_radius=0.4,
            obstacle_kernel_sigma=0.8,
            rng_seed=0, qp_T=0.3, qp_lam=0.1,
        )
        return ReplayOracle(rep, CRITIC_NAMES_SHORT, cfg).compute_oracle_weights()

    open_w = _run(str(tmp_path / "open.h5"), obstacles=None, max_obstacles=0)

    obs = [
        np.array([[1.0, 1.5, 0.0, np.nan, np.nan]], dtype=np.float32)
        for _ in range(2)
    ]
    blocked_w = _run(str(tmp_path / "blocked.h5"), obstacles=obs, max_obstacles=1)

    assert open_w.shape == blocked_w.shape == (2, 4)
    np.testing.assert_allclose(open_w.sum(axis=1), 1.0, atol=1e-3)
    np.testing.assert_allclose(blocked_w.sum(axis=1), 1.0, atol=1e-3)

    # DynamicObstacleCritic is index 3. In the open episode it has no
    # signal so the KL prior dominates → ~uniform weight; in the blocked
    # episode the future-aware critic is the one critic that reliably
    # rewards i_star, so it should carry materially more mass.
    open_dyn = open_w[:, 3].mean()
    blocked_dyn = blocked_w[:, 3].mean()
    assert blocked_dyn > open_dyn + 0.02, (
        f"expected dynamic-obstacle weight to lift in blocked scenario; "
        f"open mean={open_dyn:.4f}, blocked mean={blocked_dyn:.4f}"
    )


def test_K_matches_recorded_critic_names(tmp_path):
    """Sanity: the oracle outputs K columns matching the supplied
    critic_names, not the registry's full size."""
    p = str(tmp_path / "k_check.h5")
    K = 11
    names = [
        "WeightedConstraintCritic", "WeightedCostCritic", "WeightedGoalCritic",
        "WeightedGoalAngleCritic", "WeightedPathAlignCritic",
        "WeightedPathFollowCritic", "WeightedPathAngleCritic",
        "WeightedPreferForwardCritic", "CorridorCritic", "VolumetricCritic",
        "DynamicObstacleCritic",
    ]
    T = 2
    robot_traj = np.zeros((T, 2))
    _write_episode(
        p, T=T, K=K, names=names,
        robot_traj=robot_traj, goal_xy=(3.0, 0.0),
        obstacles=None, max_obstacles=0,
    )

    rep = reconstruct(p, dt_override=0.1)
    cfg = MPPIConfig(horizon=10, n_samples=64, dt=0.1, rng_seed=1)
    oracle = ReplayOracle(rep, names, cfg)
    weights = oracle.compute_oracle_weights()
    assert weights.shape == (T, K)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-3)
    assert (weights >= 0).all()
