"""
replay_oracle.py — Phase 9.3

Pure-Python MPPI replay oracle. For each recorded timestep:

  1. Sample N candidate control sequences (vx, wz) from the recorded
     pose at step t.
  2. Roll out diff-drive kinematics to produce N trajectories.
  3. Score each trajectory with the K critics named in the recorded
     ``critic_names`` attr, using a name-driven Python port registry.
  4. Apply the **oracle cost function** (collision check against
     ``WorldReplay.obstacles_at(t+τ)`` plus terminal goal distance) to
     pick ``i_star``. This is independent of the runtime softmax — the
     oracle uses ground-truth future, the QP recovers what runtime
     weights would have justified the same pick.
  5. Solve the convex QP (``weight_recovery_qp.recover_weights``) to
     obtain ``w_star ∈ Δ^K``.

The result is a per-timestep ``oracle_weights`` array of shape
``(num_steps, K)`` — the supervised target for Phase 9.5 wiring.

Critic ports are *approximations* of upstream ``nav2_mppi_controller``
critics. They are not bit-for-bit faithful; the oracle's job is to
expose plausible per-critic gradients so the QP can recover a
meaningful weight vector. Critics that depend on perception (CostCritic,
CorridorCritic, VolumetricCritic) return constant zero — the KL prior
keeps their weight near uniform, which is the right behaviour when the
oracle has no signal for them.

Path source: straight line from current pose to the recorded goal. This
is the v1 simplification flagged in plan.md §9.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .weight_recovery_qp import recover_weights
from .world_reconstructor import ObstacleSnapshot, WorldReplay


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MPPIConfig:
    """Tuneables for the offline oracle MPPI rollout.

    Defaults track ``vf_controller_fixed.yaml`` where it makes sense
    (vx_max, wz_max) but use a much smaller sample count — the oracle
    is offline so we can afford to be careful per timestep without
    matching the C++ runtime's 2000-sample batch.
    """

    horizon: int = 30                  # steps to roll out
    n_samples: int = 256               # candidate trajectories per timestep
    dt: float = 0.05                   # rollout integration step (matches model_dt)
    vx_max: float = 0.5
    vx_min: float = -0.35
    wz_max: float = 1.9
    sigma_vx: float = 0.20             # gaussian sample std for vx
    sigma_wz: float = 0.80             # gaussian sample std for wz
    collision_radius: float = 0.40     # robot footprint approximation [m]
    obstacle_radius: float = 0.30      # obstacle inflation [m]
    obstacle_kernel_sigma: float = 0.6 # soft-kernel width for the per-critic cost [m]
    collision_penalty: float = 1.0e6   # oracle hard-cost for collisions
    oracle_margin_weight: float = 5.0  # weight on the oracle's soft margin term
    rng_seed: int = 0
    # QP knobs (forwarded to recover_weights)
    qp_T: float = 0.3
    qp_lam: float = 0.1


# =============================================================================
# Critic port registry
# =============================================================================
#
# Each port computes a non-negative cost (lower = better, per upstream MPPI
# convention) for one trajectory. The signature is uniform so the oracle can
# loop over them by name from the HDF5 ``critic_names`` attr.
#
# A few of the runtime critics depend on the costmap or GCF features that
# the oracle has no offline access to. Those return 0 — see module
# docstring.

CriticPort = Callable[[np.ndarray, np.ndarray, "ScoreContext"], float]


@dataclass
class ScoreContext:
    """Per-step inputs the critic ports need."""

    start_pose: np.ndarray   # (3,) [x, y, theta]
    goal: np.ndarray         # (3,) [x, y, theta]
    config: MPPIConfig


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed angle from ``b`` to ``a``."""
    d = a - b
    return float(np.arctan2(np.sin(d), np.cos(d)))


def _path_unit(start: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, float]:
    """Unit vector + length for the straight-line plan from start to goal."""
    delta = goal[:2] - start[:2]
    L = float(np.linalg.norm(delta))
    if L < 1e-6:
        return np.array([1.0, 0.0]), 0.0
    return delta / L, L


# --- ports ------------------------------------------------------------------

def _port_constraint(
    traj: np.ndarray, controls: np.ndarray, ctx: ScoreContext,
) -> float:
    """Penalise commands beyond kinematic limits. Squared overshoot."""
    vx, wz = controls[:, 0], controls[:, 1]
    over_vx = np.maximum(0.0, np.abs(vx) - ctx.config.vx_max)
    over_wz = np.maximum(0.0, np.abs(wz) - ctx.config.wz_max)
    under_vx = np.maximum(0.0, ctx.config.vx_min - vx)
    return float(np.mean(over_vx ** 2 + under_vx ** 2 + over_wz ** 2))


def _port_goal(
    traj: np.ndarray, controls: np.ndarray, ctx: ScoreContext,
) -> float:
    """Squared distance from trajectory end to goal."""
    end_xy = traj[-1, :2]
    return float(np.sum((end_xy - ctx.goal[:2]) ** 2))


def _port_goal_angle(
    traj: np.ndarray, controls: np.ndarray, ctx: ScoreContext,
) -> float:
    """Heading error at trajectory end vs goal heading."""
    return float(_angle_diff(traj[-1, 2], ctx.goal[2]) ** 2)


def _port_path_align(
    traj: np.ndarray, controls: np.ndarray, ctx: ScoreContext,
) -> float:
    """Mean squared cross-track error vs straight-line plan."""
    u, L = _path_unit(ctx.start_pose, ctx.goal)
    if L == 0.0:
        return 0.0
    n = np.array([-u[1], u[0]])
    rel = traj[:, :2] - ctx.start_pose[:2][None, :]
    cross = rel @ n
    return float(np.mean(cross ** 2))


def _port_path_follow(
    traj: np.ndarray, controls: np.ndarray, ctx: ScoreContext,
) -> float:
    """Negative along-track progress at trajectory end (lower = farther)."""
    u, L = _path_unit(ctx.start_pose, ctx.goal)
    if L == 0.0:
        return 0.0
    rel = traj[-1, :2] - ctx.start_pose[:2]
    along = float(rel @ u)
    return float((L - along) ** 2)


def _port_path_angle(
    traj: np.ndarray, controls: np.ndarray, ctx: ScoreContext,
) -> float:
    """Heading vs plan tangent, averaged over trajectory."""
    u, L = _path_unit(ctx.start_pose, ctx.goal)
    if L == 0.0:
        return 0.0
    plan_theta = float(np.arctan2(u[1], u[0]))
    diffs = np.array([_angle_diff(t, plan_theta) for t in traj[:, 2]])
    return float(np.mean(diffs ** 2))


def _port_prefer_forward(
    traj: np.ndarray, controls: np.ndarray, ctx: ScoreContext,
) -> float:
    """Penalise reverse motion. Squared negative-vx component."""
    vx = controls[:, 0]
    return float(np.mean(np.minimum(0.0, vx) ** 2))


def _port_zero(
    traj: np.ndarray, controls: np.ndarray, ctx: ScoreContext,
) -> float:
    """Inert port for critics that need perception not exposed to the oracle."""
    return 0.0


# Registry maps recorded critic name -> port function. The "Weighted" prefix
# is stripped before lookup (the runtime wraps upstream critics under that
# prefix; the cost computation is the same).

_PORTS: Dict[str, CriticPort] = {
    "ConstraintCritic": _port_constraint,
    "GoalCritic": _port_goal,
    "GoalAngleCritic": _port_goal_angle,
    "PathAlignCritic": _port_path_align,
    "PathFollowCritic": _port_path_follow,
    "PathAngleCritic": _port_path_angle,
    "PreferForwardCritic": _port_prefer_forward,
    # Perception-dependent runtime critics — no offline signal. KL prior
    # keeps these near uniform; see module docstring.
    "CostCritic": _port_zero,
    "ObstaclesCritic": _port_zero,
    "TwirlingCritic": _port_zero,
    "VelocityDeadbandCritic": _port_zero,
    "CorridorCritic": _port_zero,
    "VolumetricCritic": _port_zero,
    # Dynamic-obstacle critic is special-cased in ReplayOracle._score_step
    # because it needs WorldReplay.obstacles_at(t+τ); see _dynamic_obstacle_cost.
    "DynamicObstacleCritic": _port_zero,
}


def _resolve_port(name: str) -> CriticPort:
    key = name[len("Weighted"):] if name.startswith("Weighted") else name
    return _PORTS.get(key, _port_zero)


def _is_dynamic_obstacle_critic(name: str) -> bool:
    return name.endswith("DynamicObstacleCritic")


# =============================================================================
# Oracle
# =============================================================================

class ReplayOracle:
    """Run pure-Python MPPI offline against a recorded episode."""

    def __init__(
        self,
        world: WorldReplay,
        critic_names: List[str],
        config: Optional[MPPIConfig] = None,
    ) -> None:
        self.world = world
        self.critic_names = list(critic_names)
        self.K = len(self.critic_names)
        self.config = config or MPPIConfig()
        self._ports = [_resolve_port(n) for n in self.critic_names]
        self._dyn_idx = [
            i for i, n in enumerate(self.critic_names)
            if _is_dynamic_obstacle_critic(n)
        ]
        self._rng = np.random.default_rng(self.config.rng_seed)

    # ------------------------------------------------------------------ public
    def compute_oracle_weights(self) -> np.ndarray:
        """Run the oracle over every recorded timestep.

        Returns:
          (T, K) float32 simplex array. Row t is the recovered weight
          vector w* for the trajectory the oracle would have picked.
        """
        T = self.world.num_steps
        out = np.full((T, self.K), 1.0 / max(self.K, 1), dtype=np.float32)
        for t in range(T):
            try:
                S, i_star = self._score_step(t)
            except Exception:
                # Per-timestep robustness: any failure leaves the row at
                # uniform (rather than poisoning the whole episode).
                continue
            try:
                w = recover_weights(
                    S, i_star,
                    T=self.config.qp_T,
                    lam=self.config.qp_lam,
                )
            except Exception:
                continue
            out[t] = np.asarray(w, dtype=np.float32)
        return out

    def score_step(self, t: int) -> Tuple[np.ndarray, int]:
        """Public access to one timestep's (S, i_star). Mostly for tests."""
        return self._score_step(t)

    # ----------------------------------------------------------------- internal
    def _score_step(self, t: int) -> Tuple[np.ndarray, int]:
        cfg = self.config
        start_pose = self.world.state_at(t)
        goal = self.world.goal_at(t)

        controls = self._sample_controls()                      # (N, H, 2)
        trajectories = self._rollout(start_pose, controls)      # (N, H+1, 3)

        ctx = ScoreContext(start_pose=start_pose, goal=goal, config=cfg)
        S = self._compute_S(trajectories, controls, ctx, t)     # (N, K)
        oracle_costs, surviving = self._oracle_costs(trajectories, t)
        i_star = self._pick_i_star(oracle_costs, surviving)
        return S, i_star

    # ---- sampling / rollout ------------------------------------------------
    def _sample_controls(self) -> np.ndarray:
        cfg = self.config
        N, H = cfg.n_samples, cfg.horizon
        vx = self._rng.normal(loc=0.5 * (cfg.vx_max + cfg.vx_min), scale=cfg.sigma_vx, size=(N, H))
        wz = self._rng.normal(loc=0.0, scale=cfg.sigma_wz, size=(N, H))
        out = np.stack([vx, wz], axis=-1).astype(np.float32)
        return out

    def _rollout(self, start_pose: np.ndarray, controls: np.ndarray) -> np.ndarray:
        cfg = self.config
        N, H, _ = controls.shape
        traj = np.empty((N, H + 1, 3), dtype=np.float32)
        traj[:, 0, :] = start_pose
        for k in range(H):
            x = traj[:, k, 0]
            y = traj[:, k, 1]
            th = traj[:, k, 2]
            vx = controls[:, k, 0]
            wz = controls[:, k, 1]
            traj[:, k + 1, 0] = x + vx * np.cos(th) * cfg.dt
            traj[:, k + 1, 1] = y + vx * np.sin(th) * cfg.dt
            traj[:, k + 1, 2] = th + wz * cfg.dt
        return traj

    # ---- per-critic scoring -----------------------------------------------
    def _compute_S(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        ctx: ScoreContext,
        t: int,
    ) -> np.ndarray:
        N = trajectories.shape[0]
        K = self.K
        S = np.zeros((N, K), dtype=np.float64)

        for k, port in enumerate(self._ports):
            if k in self._dyn_idx:
                # Special-cased — needs world.obstacles_at(t+τ).
                S[:, k] = self._dynamic_obstacle_cost(trajectories, t)
            else:
                for n in range(N):
                    S[n, k] = port(trajectories[n], controls[n], ctx)
        return S

    def _dynamic_obstacle_cost(
        self, trajectories: np.ndarray, t0: int,
    ) -> np.ndarray:
        """Runtime-equivalent dynamic-obstacle cost.

        Uses the PRESENT-frame obstacle snapshot at t0 (with finite-diff
        velocities from the world reconstructor) and predicts forward
        with constant velocity. This mirrors what the live runtime
        ``DynamicObstacleCritic`` does — the oracle's ground-truth-future
        advantage shows up in :meth:`_oracle_costs`, not here. The QP
        recovers a weight that makes the runtime-style softmax pick
        i_star; that's the meaningful signal."""
        N = trajectories.shape[0]
        if not self.world.has_dynamic_obstacles:
            return np.zeros(N, dtype=np.float64)

        snap = self.world.obstacles_at(t0)
        if snap.empty():
            return np.zeros(N, dtype=np.float64)

        cfg = self.config
        H = trajectories.shape[1]  # horizon + 1
        sigma2 = max(cfg.obstacle_kernel_sigma ** 2, 1e-6)
        dt = cfg.dt
        vel = np.where(np.isfinite(snap.velocities), snap.velocities, 0.0)

        cost = np.zeros(N, dtype=np.float64)
        for k in range(H):
            obs_xy = snap.positions + vel * (k * dt)  # (M, 2)
            traj_xy = trajectories[:, k, :2]
            d2 = ((traj_xy[:, None, :] - obs_xy[None, :, :]) ** 2).sum(axis=-1)
            cost += np.exp(-d2 / sigma2).sum(axis=1)
        return cost

    # ---- oracle cost / i_star --------------------------------------------
    def _oracle_costs(
        self, trajectories: np.ndarray, t0: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ground-truth-aware oracle cost: hard collision check using
        **actual** future obstacle positions, plus a soft margin term
        that punishes trajectories passing close to obstacles, plus
        terminal goal distance.

        The soft-margin term is what makes the oracle prefer wide-berth
        trajectories over "shoot the gap" ones; without it, i_star tends
        to be the trajectory that just barely survives while minimising
        terminal goal cost, which leaves the per-critic signal flat.

        Returns:
          oracle_costs:  (N,) ground-truth-aware cost.
          surviving:     (N,) bool, True if no future-obstacle collision.
        """
        cfg = self.config
        N, H, _ = trajectories.shape
        goal = self.world.goal_at(t0)[:2]
        end_xy = trajectories[:, -1, :2]
        terminal = np.sum((end_xy - goal[None, :]) ** 2, axis=1)

        surviving = np.ones(N, dtype=bool)
        margin_cost = np.zeros(N, dtype=np.float64)
        if self.world.has_dynamic_obstacles:
            infl = (cfg.collision_radius + cfg.obstacle_radius) ** 2
            sigma2 = max(cfg.obstacle_kernel_sigma ** 2, 1e-6)
            for k in range(H):
                snap = self.world.obstacles_at(t0 + k)
                if snap.empty():
                    continue
                obs_xy = snap.positions
                traj_xy = trajectories[:, k, :2]
                d2 = ((traj_xy[:, None, :] - obs_xy[None, :, :]) ** 2).sum(axis=-1)
                hits = (d2 < infl).any(axis=1)
                surviving &= ~hits
                # Sum exp-kernel proximity over all obstacles, all times.
                margin_cost += np.exp(-d2 / sigma2).sum(axis=1)

        oracle_cost = terminal + cfg.oracle_margin_weight * margin_cost
        oracle_cost = np.where(
            surviving, oracle_cost, oracle_cost + cfg.collision_penalty,
        )
        return oracle_cost, surviving

    @staticmethod
    def _pick_i_star(oracle_costs: np.ndarray, surviving: np.ndarray) -> int:
        if surviving.any():
            masked = np.where(surviving, oracle_costs, np.inf)
            return int(np.argmin(masked))
        return int(np.argmin(oracle_costs))


__all__ = ["ReplayOracle", "MPPIConfig", "ScoreContext"]
