"""
Phase 9.5 — verify that oracle labels (when present in the HDF5) are
preferred over hindsight labels, and that legacy episodes still work.

Touches:
  * derive_inference_targets() oracle / hindsight branches
  * EpisodeReader.oracle_weights / has_oracle_weights
  * EpisodeFeatureDataset oracle-vs-hindsight episode counters
"""
import numpy as np
import pytest

from vf_controller.data_collection.episode_writer import (
    CycleRow, EpisodeMetadata, EpisodeOutcome, EpisodeWriter)
from vf_controller.training.data.log_reader import EpisodeReader
from vf_controller.training.data.oracle_weights import (
    derive_inference_targets)


CRITICS = [
    "WeightedConstraintCritic",
    "WeightedGoalCritic",
    "WeightedPathAlignCritic",
    "DynamicObstacleCritic",
]


def _meta(K=4):
    return EpisodeMetadata(
        scenario_id="phase9_5_unit",
        seed=1,
        controller_mode="collect",
        weight_provider="fixed",
        channels_config="channels_v1",
        channel_names=["a", "b"],
        channel_dims=[3, 2],
        critic_names=CRITICS[:K],
    )


def _write_basic(path, T=4, K=4, applied=None):
    if applied is None:
        applied = np.full((T, K), 1.0 / K, dtype=np.float32)
    with EpisodeWriter(path, 5, K, _meta(K), max_obstacles=0) as w:
        for t in range(T):
            w.append(CycleRow(
                features=np.zeros(5, dtype=np.float32),
                critic_costs=np.zeros(K, dtype=np.float32),
                critic_weights=applied[t].astype(np.float32),
                selected_action=np.zeros(3, dtype=np.float32),
                robot_pose=np.array([float(t)*0.1, 0.0, 0.0], dtype=np.float32),
                goal=np.array([3.0, 0.0, 0.0], dtype=np.float32),
                dynamic_obstacles=None,
            ))
        w.close(outcome=EpisodeOutcome(success=True))


def _attach_oracle(path, weights):
    """Append an oracle_weights dataset to an existing file."""
    import h5py
    with h5py.File(path, "a") as f:
        if "oracle_weights" in f:
            del f["oracle_weights"]
        f.create_dataset(
            "oracle_weights",
            data=np.asarray(weights, dtype=np.float32),
            compression="gzip", compression_opts=4,
        )


# --- derive_inference_targets ------------------------------------------------

def test_oracle_branch_takes_precedence_when_provided():
    """When oracle_weights is passed, the result is the oracle simplex
    verbatim (re-normalised) — critic_weights_applied is ignored."""
    K = 4
    applied = np.full((3, K), 1.0 / K, dtype=np.float32)  # uniform
    oracle = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25],
    ], dtype=np.float32)

    tgt, mask = derive_inference_targets(
        np.zeros((3, K), dtype=np.float32), applied, oracle_weights=oracle)

    np.testing.assert_allclose(tgt, oracle, atol=1e-6)
    assert mask.tolist() == [True, True, True]


def test_oracle_nan_row_falls_back_to_uniform():
    """A NaN row in oracle_weights yields uniform target with mask=False."""
    K = 3
    applied = np.full((2, K), 1.0 / K, dtype=np.float32)
    oracle = np.array([
        [0.5, 0.3, 0.2],
        [np.nan, 0.5, 0.5],
    ], dtype=np.float32)

    tgt, mask = derive_inference_targets(
        np.zeros((2, K), dtype=np.float32), applied, oracle_weights=oracle)

    np.testing.assert_allclose(tgt[0], [0.5, 0.3, 0.2], atol=1e-6)
    np.testing.assert_allclose(tgt[1], [1/3, 1/3, 1/3], atol=1e-6)
    assert mask.tolist() == [True, False]


def test_hindsight_path_unchanged_when_oracle_not_provided():
    """No oracle_weights → behaviour matches Phase 8: row-normalise applied,
    NaN rows become uniform-fallback (mask=False)."""
    K = 3
    applied = np.array([
        [2.0, 1.0, 1.0],          # sums to 4 -> normalised
        [np.nan, 0.0, 0.0],       # NaN -> uniform
        [0.0, 0.0, 0.0],          # zero -> uniform
    ], dtype=np.float32)
    tgt, mask = derive_inference_targets(np.zeros((3, K), dtype=np.float32), applied)

    np.testing.assert_allclose(tgt[0], [0.5, 0.25, 0.25], atol=1e-6)
    np.testing.assert_allclose(tgt[1], [1/3, 1/3, 1/3], atol=1e-6)
    np.testing.assert_allclose(tgt[2], [1/3, 1/3, 1/3], atol=1e-6)
    assert mask.tolist() == [True, False, False]


def test_oracle_dataset_with_wrong_K_raises():
    K = 4
    applied = np.full((2, K), 1.0 / K, dtype=np.float32)
    bad_oracle = np.full((2, K + 1), 1.0 / (K + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="oracle_weights must be"):
        derive_inference_targets(
            np.zeros((2, K), dtype=np.float32), applied, oracle_weights=bad_oracle)


# --- EpisodeReader -----------------------------------------------------------

def test_episode_reader_exposes_oracle_dataset(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_basic(p, T=3, K=4)
    with EpisodeReader(p) as r:
        assert r.oracle_weights is None
        assert r.has_oracle_weights is False

    oracle_truth = np.full((3, 4), 0.25, dtype=np.float32)
    oracle_truth[0, 0] = 0.7
    oracle_truth[0, 1:] = 0.1
    _attach_oracle(p, oracle_truth)

    with EpisodeReader(p) as r:
        assert r.has_oracle_weights is True
        np.testing.assert_allclose(r.oracle_weights, oracle_truth, atol=1e-6)


# --- EpisodeFeatureDataset ---------------------------------------------------

def test_dataset_uses_oracle_when_present_and_counts_episodes(tmp_path):
    """Mixed corpus: one episode with oracle_weights, one without.
    Counters reflect the split; oracle rows beat hindsight rows verbatim.

    Torch-free: this path exercises only ``__init__`` + the .targets array,
    not ``__getitem__``, so it runs on the apt CI image."""
    from vf_controller.training.data.dataset import EpisodeFeatureDataset

    p_hind = str(tmp_path / "hind.h5")
    p_oracle = str(tmp_path / "oracle.h5")
    K = 4
    # Hindsight episode: applied = uniform → derived target is uniform.
    _write_basic(p_hind, T=3, K=K)
    # Oracle episode: applied stays uniform but we write a non-uniform oracle
    # dataset; the dataset must surface oracle rows verbatim.
    _write_basic(p_oracle, T=3, K=K)
    oracle_truth = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25],
    ], dtype=np.float32)
    _attach_oracle(p_oracle, oracle_truth)

    ds = EpisodeFeatureDataset([p_hind, p_oracle], mode="inference")
    assert ds.n_oracle_episodes == 1
    assert ds.n_hindsight_episodes == 1
    assert len(ds) == 6  # 3 hindsight + 3 oracle rows

    # Hindsight rows = uniform; oracle rows = oracle_truth verbatim. The
    # dataset concatenates in path order [hind, oracle] so rows 3..5 are oracle.
    targets = ds.targets
    np.testing.assert_allclose(targets[:3], 1.0 / K, atol=1e-6)
    np.testing.assert_allclose(targets[3:6], oracle_truth, atol=1e-6)
