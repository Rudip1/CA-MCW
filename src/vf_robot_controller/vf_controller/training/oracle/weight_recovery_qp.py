"""
weight_recovery_qp.py — convex QP for recovering ideal critic weights.

Phase 9: given a critic-cost matrix S in R^{N x K} (N candidate trajectories,
K critics; lower-is-better per upstream MPPI convention) and the index
``i_star`` of the trajectory the *oracle* picked offline, recover a weight
simplex w in R^K under which MPPI's softmax-NLL selection rule would have
also picked i_star with high probability.

The optimization problem (see the oracle-QP derivation notes):

    minimize   S[i*] @ w / T  +  log_sum_exp(-S @ w / T)  +  lam * KL(w || 1/K)
    s.t.       w >= 0,  sum(w) == 1

Convex (linear + log-sum-exp + KL), unique global minimum for any lam > 0.
Solved via cvxpy. We import cvxpy lazily so that this module is importable
in colcon-test environments where cvxpy is not provisioned (mirrors the
torch / onnxruntime pattern used elsewhere in vf_controller.training).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


_CVXPY_IMPORT_ERROR: Optional[Exception] = None
try:  # lazy / soft import — see module docstring
    import cvxpy as cp  # type: ignore
except Exception as exc:  # pragma: no cover - exercised only on cvxpy-less hosts
    cp = None  # type: ignore[assignment]
    _CVXPY_IMPORT_ERROR = exc


_DEFAULT_T = 0.3
_DEFAULT_LAM = 0.1


def recover_weights(
    S: np.ndarray,
    i_star: int,
    *,
    T: float = _DEFAULT_T,
    lam: float = _DEFAULT_LAM,
    K: Optional[int] = None,
    solver: Optional[str] = None,
) -> np.ndarray:
    """Recover the K-dim weight simplex that maximises P(i_star).

    Args:
      S:       (N, K) critic-cost matrix. Lower is better.
      i_star:  index of the oracle-chosen trajectory in [0, N).
      T:       softmax temperature; match MPPI's runtime value
               (vf_fixedwt.yaml temperature, default 0.3 as of 2026-05-08).
      lam:     KL prior weight. Higher pulls w toward uniform. Tune in [0.01, 1.0].
      K:       override critic count; defaults to S.shape[1].
      solver:  cvxpy solver name. Defaults to SCS for K<=20.

    Returns:
      np.ndarray, shape (K,), non-negative, sums to 1. On solver failure
      returns the uniform simplex (1/K) — never NaN — so a single bad
      timestep does not poison the entire oracle pass.

    Raises:
      RuntimeError: if cvxpy is not installed in this Python environment.
    """
    if cp is None:
        raise RuntimeError(
            "cvxpy is required for recover_weights() but is not installed "
            "in this Python environment. Install it in the training env "
            f"(e.g. conda dl): pip install cvxpy. Original import error: {_CVXPY_IMPORT_ERROR!r}"
        )

    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"S must be 2-D, got shape {S.shape}")
    N, K_inferred = S.shape
    K_eff = K if K is not None else K_inferred
    if K_eff != K_inferred:
        raise ValueError(f"K mismatch: arg={K_eff}, S.shape[1]={K_inferred}")
    if not (0 <= i_star < N):
        raise ValueError(f"i_star {i_star} out of range [0, {N})")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if lam < 0.0:
        raise ValueError(f"lam must be non-negative, got {lam}")

    w = cp.Variable(K_eff, nonneg=True)
    nll = (S[i_star] @ w) / T + cp.log_sum_exp(-(S @ w) / T)
    prior = np.full(K_eff, 1.0 / K_eff)
    kl = cp.sum(cp.kl_div(w, prior))
    obj = cp.Minimize(nll + lam * kl)
    prob = cp.Problem(obj, [cp.sum(w) == 1])

    chosen = solver or _pick_default_solver()
    try:
        prob.solve(solver=chosen)
    except Exception:  # pragma: no cover - solver hiccup
        return np.full(K_eff, 1.0 / K_eff, dtype=np.float32)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return np.full(K_eff, 1.0 / K_eff, dtype=np.float32)

    out = np.asarray(w.value, dtype=np.float64)
    out = np.clip(out, 0.0, None)
    s = out.sum()
    if not np.isfinite(s) or s <= 0.0:
        return np.full(K_eff, 1.0 / K_eff, dtype=np.float32)
    return (out / s).astype(np.float32)


def _pick_default_solver() -> str:
    """Pick a solver that's compatible with the log-sum-exp + KL objective.

    OSQP is QP-only and would reject our objective. SCS and CLARABEL both
    handle exponential cones; ECOS is the legacy fallback.
    """
    if cp is None:
        return "SCS"
    installed = set(cp.installed_solvers())
    for name in ("CLARABEL", "SCS", "ECOS"):
        if name in installed:
            return name
    return "SCS"


__all__ = ["recover_weights"]
