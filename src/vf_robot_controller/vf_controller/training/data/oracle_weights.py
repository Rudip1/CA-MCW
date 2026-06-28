"""
oracle_weights.py — derive ground-truth target weights for the INFERENCE
network from an HDF5 episode.

Two label sources coexist:

  * **Phase 8 (hindsight)** — fall back to the recorded
    ``critic_weights_applied`` (behaviour cloning on whatever weight provider
    drove the COLLECT episode). Used when an episode has no oracle labels.
  * **Phase 9 (oracle)** — when ``scripts/run_oracle.py`` has augmented the
    HDF5 with an ``oracle_weights`` dataset (per-timestep simplex recovered
    by the convex QP in :mod:`vf_controller.training.oracle.weight_recovery_qp`),
    use it directly. This is the publishable thesis target.

Both paths return the same ``(target, mask)`` shape so the trainer is
unchanged. The mask is True wherever the target row is meaningful — a row
that had to be synthesised by uniform fall-back (NaN, zero-sum, etc.) is
False so the loss can opt to ignore it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def derive_inference_targets(
    critic_costs: np.ndarray,
    critic_weights_applied: np.ndarray,
    *,
    oracle_weights: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute INFERENCE targets per timestep.

    Args:
      critic_costs:           (T, K) float32 — per-critic raw cost (kept for
                              symmetry; the Phase 9 QP runs in
                              :mod:`scripts.run_oracle`, not here).
      critic_weights_applied: (T, K) float32 — weights applied this cycle.
                              Used only when ``oracle_weights`` is None.
      oracle_weights:         optional (T, K) float32 — per-timestep simplex
                              recovered offline. When provided, this function
                              degenerates to a row-finiteness check and mask
                              construction — the rows are taken verbatim.

    Returns:
      target:                 (T, K) float32, rows sum to 1 (probability
                              simplex), aligned to ``critic_weights_applied``.
      mask:                   (T,)   bool    — True where the target is real;
                              fallback / NaN rows are False.
    """
    cw = np.asarray(critic_weights_applied, dtype=np.float32)
    if cw.ndim != 2:
        raise ValueError(f"critic_weights_applied must be (T, K), got {cw.shape}")
    T, K = cw.shape

    if oracle_weights is not None:
        return _from_oracle_dataset(oracle_weights, T=T, K=K, eps=eps)

    # Phase 8 hindsight fallback.
    target = np.zeros_like(cw, dtype=np.float32)
    mask = np.zeros(T, dtype=bool)
    for t in range(T):
        row = cw[t]
        if np.any(np.isnan(row)):
            target[t, :] = 1.0 / K
            continue
        s = float(row.sum())
        if s <= eps:
            target[t, :] = 1.0 / K
            continue
        target[t, :] = row / s
        mask[t] = True
    return target, mask


def _from_oracle_dataset(
    oracle_weights: np.ndarray,
    *,
    T: int,
    K: int,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Phase 9 path: take the dataset verbatim and validate per-row.

    The HDF5 augmentation by :mod:`scripts.run_oracle` already writes a
    simplex per row, but we re-normalise defensively — small numerical drift
    from the QP solver shouldn't propagate into the loss. Rows whose oracle
    output was non-finite or zero-sum (a per-timestep solver failure that
    fell back inside ``recover_weights``) get a uniform target with mask
    False, matching the Phase 8 contract.
    """
    ow = np.asarray(oracle_weights, dtype=np.float32)
    if ow.ndim != 2 or ow.shape[1] != K:
        raise ValueError(
            f"oracle_weights must be (T, {K}), got {ow.shape}"
        )
    T_ow = ow.shape[0]
    target = np.zeros((T_ow, K), dtype=np.float32)
    mask = np.zeros(T_ow, dtype=bool)

    for t in range(T_ow):
        row = ow[t]
        if not np.all(np.isfinite(row)):
            target[t, :] = 1.0 / K
            continue
        # Clip tiny negatives from solver numerics; renormalise.
        row = np.clip(row, 0.0, None)
        s = float(row.sum())
        if s <= eps:
            target[t, :] = 1.0 / K
            continue
        target[t, :] = row / s
        mask[t] = True

    # If the augmented file's T differs from features T (rare but possible
    # if the augmentation step ran on a partially-flushed file), pad with
    # uniform-fallback rows so the caller can still align by min-length.
    if T_ow < T:
        pad = np.full((T - T_ow, K), 1.0 / K, dtype=np.float32)
        target = np.concatenate([target, pad], axis=0)
        mask = np.concatenate([mask, np.zeros(T - T_ow, dtype=bool)], axis=0)
    return target, mask


def derive_raw_critic_targets(
    critic_costs: np.ndarray,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive training targets from raw critic costs without an oracle.

    label_k = softmax(critic_costs_k / temperature)

    Semantics: a critic with higher cost is being violated more → assign it
    more weight so MPPI puts greater effort into satisfying it.  Temperature
    controls sharpness (lower T → more peaked; higher T → more uniform).
    Matches the runtime MPPI softmax temperature when set to the same value
    as ``temperature`` in vf_fixedwt.yaml (default 0.3).

    No oracle run required — labels are derived on-the-fly from the costs
    already present in every collect HDF5.

    Returns:
      target: (T, K) float32 — probability simplex per row.
      mask:   (T,)   bool   — False for rows with any non-finite cost (NaN/Inf
              rows fall back to uniform 1/K and are excluded from the loss).
    """
    cc = np.asarray(critic_costs, dtype=np.float32)
    if cc.ndim != 2:
        raise ValueError(f"critic_costs must be (T, K), got {cc.shape}")
    T, K = cc.shape

    valid = np.all(np.isfinite(cc), axis=1)   # (T,) — rows where all K costs finite
    target = np.full((T, K), 1.0 / K, dtype=np.float32)

    if valid.any():
        x = cc[valid] / temperature                          # (T_v, K)
        x = x - x.max(axis=1, keepdims=True)                # numerical stability
        exp_x = np.exp(x)                                    # (T_v, K)
        row_sums = exp_x.sum(axis=1, keepdims=True)          # (T_v, 1)
        nz = row_sums.squeeze(1) > eps                       # (T_v,)
        rows = np.full_like(exp_x, 1.0 / K)
        rows[nz] = exp_x[nz] / row_sums[nz]
        target[valid] = rows

    return target, valid


def derive_imitation_targets(selected_action: np.ndarray) -> np.ndarray:
    """
    IMITATION target = (vx, wz) from the recorded selected_action.

    Args:
      selected_action: (T, 3) float32 — [vx, vy, wz] per cycle.

    Returns:
      target:          (T, 2) float32 — [vx, wz] per cycle.
    """
    sa = np.asarray(selected_action, dtype=np.float32)
    if sa.ndim != 2 or sa.shape[1] < 3:
        raise ValueError(f"selected_action must be (T, 3+), got {sa.shape}")
    target = np.zeros((sa.shape[0], 2), dtype=np.float32)
    target[:, 0] = sa[:, 0]  # vx
    target[:, 1] = sa[:, 2]  # wz
    return target
