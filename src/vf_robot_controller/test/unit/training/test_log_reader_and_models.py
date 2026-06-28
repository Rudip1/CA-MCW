"""
Phase 8 - smoke tests for the training pipeline.

The training scripts require torch / onnx / onnxruntime, which the apt CI
image deliberately does not ship (the user runs training in a separate
conda env). These tests:

  * always run a log_reader round-trip on a synthetic .h5 file (h5py-only)
  * skip the network-forward and ONNX-export tests if torch is missing,
    so the package's colcon test stays green on a torch-less machine
"""
import os
import tempfile

import h5py
import numpy as np
import pytest

from vf_controller.data_collection.episode_writer import (
    CycleRow, EpisodeMetadata, EpisodeOutcome, EpisodeWriter)
from vf_controller.training.data.log_reader import (
    EpisodeReader, MultiEpisodeIndex)


def _write_synthetic_episode(path: str, T: int = 4, D: int = 17, K: int = 3):
    meta = EpisodeMetadata(
        scenario_id="phase8_unit",
        seed=11,
        controller_mode="collect",
        weight_provider="fixed",
        channels_config="channels_v1",
        channel_names=["a", "b"],
        channel_dims=[8, 9],  # sums to 17
        critic_names=["WeightedConstraintCritic", "WeightedCostCritic",
                      "CorridorCritic"],
    )
    with EpisodeWriter(path, D, K, meta) as w:
        for _ in range(T):
            w.append(CycleRow(
                features=np.random.randn(D).astype(np.float32),
                critic_costs=np.random.randn(K).astype(np.float32),
                critic_weights=np.ones(K, dtype=np.float32),
                selected_action=np.array([0.1, 0.0, 0.05], dtype=np.float32),
                robot_pose=np.array([0.5, 1.0, 0.0], dtype=np.float32),
                goal=np.array([5.0, 1.0, 0.0], dtype=np.float32),
                dynamic_obstacles=None,
            ))
        w.close(outcome=EpisodeOutcome(
            success=True, time_to_goal_s=0.5,
            path_length_m=0.1, mean_clearance_m=0.4,
            collision_count=0, goal_reached_at_step=T - 1))


def test_episode_reader_round_trip(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_synthetic_episode(p, T=5, D=17, K=3)
    r = EpisodeReader(p)
    try:
        assert r.num_steps == 5
        assert r.feature_dim == 17
        assert r.critic_count == 3
        assert r.scenario_id == "phase8_unit"
        assert r.attrs.controller_mode == "collect"
        assert list(r.critic_names) == [
            "WeightedConstraintCritic", "WeightedCostCritic", "CorridorCritic"]
        assert r.features.shape == (5, 17)
        assert r.critic_costs.shape == (5, 3)
        assert r.critic_weights_applied.shape == (5, 3)
        assert r.selected_action.shape == (5, 3)
    finally:
        r.close()


def test_multi_episode_index_dimensions(tmp_path):
    for i in range(3):
        _write_synthetic_episode(str(tmp_path / f"ep_{i}.h5"), T=4, D=17, K=3)
    idx = MultiEpisodeIndex.from_directory(str(tmp_path))
    assert len(idx) == 3
    assert idx.feature_dim() == 17
    assert idx.critic_count() == 3


def test_multi_episode_index_empty_directory_yields_zero(tmp_path):
    idx = MultiEpisodeIndex.from_directory(str(tmp_path))
    assert len(idx) == 0


def test_multi_episode_index_recurses_into_session_folders(tmp_path):
    """Default workflow: --data-dir <vf_data/vf_data_training> finds episodes nested
    under manual/<session>/ AND batch/<session>/."""
    manual = tmp_path / "manual" / "house_my1_map_20260503_120000"
    manual.mkdir(parents=True)
    auto = tmp_path / "batch" / "house_my1_map_20260503_180000_corridor"
    auto.mkdir(parents=True)
    for i in range(2):
        _write_synthetic_episode(
            str(manual / f"ep_{i:03d}.h5"), T=4, D=11, K=3,
        )
    for i in range(3):
        _write_synthetic_episode(
            str(auto / f"ep_{i:03d}_NavFn.h5"), T=4, D=11, K=3,
        )
    # Top-level: no .h5 directly here, so the recursive fallback fires.
    idx = MultiEpisodeIndex.from_directory(str(tmp_path))
    assert len(idx) == 5
    assert idx.feature_dim() == 11

    # Session-folder targeting: same call, single session, count matches.
    idx_one = MultiEpisodeIndex.from_directory(str(manual))
    assert len(idx_one) == 2


def test_multi_episode_index_prefers_flat_when_present(tmp_path):
    """When .h5 files sit directly in the dir, do NOT recurse — preserves
    old behaviour and avoids picking up unrelated episodes from siblings."""
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_synthetic_episode(str(tmp_path / "flat.h5"), T=4, D=11, K=3)
    _write_synthetic_episode(str(nested / "should_not_be_seen.h5"),
                             T=4, D=11, K=3)
    idx = MultiEpisodeIndex.from_directory(str(tmp_path))
    assert len(idx) == 1


# --- Network forward / ONNX export: only run when torch is available -----
# Uses an importorskip inside each test so that module collection succeeds
# even on torch-less environments (the ament pytest runner treats "module
# entirely skipped" as exit code 4 == failure).


def test_inference_net_forward_pass():
    torch = pytest.importorskip("torch")
    from vf_controller.training.models.inference_net import InferenceNet
    net = InferenceNet(
        channel_dims=[8, 9],
        per_channel_hidden=4,
        fusion_hidden=[16, 8],
        n_critics=3,
        dropout=0.0,
    )
    net.eval()
    x = torch.randn(2, 17)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 3)
    assert torch.all(y > 0).item()  # softplus -> strictly positive


def test_imitation_net_forward_pass_obeys_limits():
    torch = pytest.importorskip("torch")
    from vf_controller.training.models.imitation_net import ImitationNet
    net = ImitationNet(
        channel_dims=[8, 9],
        per_channel_hidden=4,
        fusion_hidden=[16, 8],
        vx_max=0.30, vx_min=-0.20,
        wz_max=1.0,
        dropout=0.0,
    )
    net.eval()
    x = torch.randn(4, 17) * 5  # large inputs to push tanh saturation
    with torch.no_grad():
        y = net(x)
    assert y.shape == (4, 2)
    # vx within [vx_min, vx_max], wz within [-wz_max, wz_max]
    assert torch.all(y[:, 0] <= 0.30 + 1e-5).item()
    assert torch.all(y[:, 0] >= -0.20 - 1e-5).item()
    assert torch.all(y[:, 1].abs() <= 1.0 + 1e-5).item()
