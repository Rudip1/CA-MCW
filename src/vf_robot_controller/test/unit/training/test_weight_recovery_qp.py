"""
Phase 9.1 — unit tests for the oracle weight-recovery QP.

Mirrors the four mandatory tests in
``the oracle-QP derivation notes``. cvxpy is gated by
pytest.importorskip, the same pattern used elsewhere in this package for
torch / onnxruntime — colcon test stays green on a cvxpy-less CI image.
"""
import numpy as np
import pytest


def _recover(*args, **kwargs):
    """Skip-if-missing wrapper around ``recover_weights``.

    The module imports cleanly without cvxpy, but ``recover_weights`` raises
    RuntimeError on call. Skip here rather than at import so the file is
    collected in either environment.
    """
    pytest.importorskip("cvxpy")
    from vf_controller.training.oracle.weight_recovery_qp import recover_weights
    return recover_weights(*args, **kwargs)


def test_recover_known_two_critic_problem():
    """Sanity: oracle picks a trajectory that's best on critic 0 — recovered
    weight should put noticeably more mass on critic 0."""
    S = np.array(
        [
            [3.0, 1.0],
            [2.0, 2.0],
            [0.5, 4.0],  # i_star — best on critic 0
            [1.5, 3.0],
            [4.0, 0.5],
        ],
        dtype=np.float64,
    )
    w = _recover(S, i_star=2, T=0.3, lam=0.05)
    assert w.shape == (2,)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-3)
    assert w[0] > w[1] + 0.2, f"expected critic 0 to dominate, got w={w}"


def test_uniform_scores_returns_uniform_weights():
    """Degenerate: every trajectory has identical scores. The likelihood
    is flat in w, so the KL prior pulls the optimum to the simplex centre."""
    K = 4
    S = np.full((10, K), 2.5)
    w = _recover(S, i_star=3, T=0.3, lam=0.1)
    assert w.shape == (K,)
    assert w == pytest.approx(np.full(K, 1.0 / K), abs=0.05)


def test_one_dominant_critic():
    """Sparse: critic 0 sweeps a wide range while the others are uniform
    noise. With weak prior, the QP should put >0.5 mass on critic 0."""
    rng = np.random.default_rng(0)
    K = 5
    N = 20
    S = rng.uniform(1.0, 1.1, size=(N, K))
    S[:, 0] = np.linspace(5.0, 0.0, N)  # trajectory N-1 is best on critic 0
    w = _recover(S, i_star=N - 1, T=0.3, lam=0.01)
    assert w.shape == (K,)
    assert w.sum() == pytest.approx(1.0, abs=1e-3)
    assert w[0] > 0.5, f"expected critic 0 to dominate, got w={w}"


def test_handles_large_score_scales():
    """Numerical stability: score columns at wildly different magnitudes
    must not produce NaN or non-simplex output. The recovery may be
    biased — that's the whole point of normalising upstream — but it
    must remain a valid probability vector."""
    S = np.array(
        [
            [1e3, 1e-3, 1.0],
            [1e2, 1e-2, 2.0],
            [1e1, 1e-1, 3.0],
        ],
        dtype=np.float64,
    )
    for i in range(3):
        w = _recover(S, i_star=i, T=0.3, lam=0.1)
        assert np.all(np.isfinite(w)), f"NaN/inf in w for i_star={i}: {w}"
        assert (w >= -1e-6).all(), f"negative weight for i_star={i}: {w}"
        assert w.sum() == pytest.approx(1.0, abs=1e-3)
