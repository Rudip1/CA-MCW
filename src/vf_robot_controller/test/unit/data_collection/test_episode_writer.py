"""
Phase 7 — unit tests for vf_controller.data_collection.episode_writer.

These exercise the HDF5 writer with synthetic per-cycle rows so we can
validate the file schema (datasets, shapes, attrs, chunked writes) without
needing a live ROS stack.
"""
import os
import tempfile

import h5py
import numpy as np
import pytest

from vf_controller.data_collection.episode_writer import (
    CycleRow, EpisodeMetadata, EpisodeOutcome, EpisodeWriter,
    episode_filename, open_episode_reader)


def _make_meta() -> EpisodeMetadata:
    return EpisodeMetadata(
        scenario_id="unit_scenario",
        seed=7,
        controller_mode="collect",
        weight_provider="fixed",
        channels_config="channels_v1",
        channel_names=["robot_state", "context"],
        channel_dims=[9, 9],
        critic_names=["WeightedConstraintCritic", "WeightedCostCritic",
                      "CorridorCritic"],
    )


def _row(D: int, K: int, with_obstacles: bool = False,
         obstacles_n: int = 0) -> CycleRow:
    return CycleRow(
        features=np.random.randn(D).astype(np.float32),
        critic_costs=np.random.randn(K).astype(np.float32),
        critic_weights=np.ones(K, dtype=np.float32),
        selected_action=np.array([0.1, 0.0, 0.05], dtype=np.float32),
        robot_pose=np.array([1.0, 2.0, 0.3], dtype=np.float32),
        goal=np.array([5.0, 5.0, 0.0], dtype=np.float32),
        dynamic_obstacles=(np.random.randn(obstacles_n, 5).astype(np.float32)
                           if with_obstacles and obstacles_n > 0 else None),
    )


def test_episode_filename_format():
    name = episode_filename("mycorridor", 42)
    assert name.startswith("mycorridor_seed42_")
    assert name.endswith(".h5")


def test_writer_creates_datasets_and_attrs(tmp_path):
    path = str(tmp_path / "ep.h5")
    D, K = 18, 3
    with EpisodeWriter(path, D, K, _make_meta()) as w:
        w.append(_row(D, K))
        w.flush()

    assert os.path.exists(path)
    with h5py.File(path, "r") as f:
        for ds in ("features", "critic_costs", "critic_weights_applied",
                   "selected_action", "robot_pose", "goal"):
            assert ds in f, "missing dataset %s" % ds
        assert f["features"].shape == (1, D)
        assert f["critic_costs"].shape == (1, K)
        assert f["selected_action"].shape == (1, 3)
        # Chunked + gzip
        assert f["features"].chunks is not None
        assert f["features"].compression == "gzip"
        # Group attrs
        assert f.attrs["scenario_id"] == "unit_scenario"
        assert int(f.attrs["seed"]) == 7
        assert f.attrs["controller_mode"] == "collect"
        assert f.attrs["channels_config"] == "channels_v1"
        names = [s for s in f.attrs["critic_names"]]
        assert "CorridorCritic" in names


def test_writer_streams_many_rows(tmp_path):
    path = str(tmp_path / "ep_long.h5")
    D, K = 9, 3
    n_rows = 1024  # exceeds default chunk_rows (256)
    with EpisodeWriter(path, D, K, _make_meta(), chunk_rows=128) as w:
        for _ in range(n_rows):
            w.append(_row(D, K))
            # Flush periodically to mirror the sidecar's 1 Hz flush cadence
            if len(w) % 200 == 0:
                w.flush()
        w.close(outcome=EpisodeOutcome(success=True, time_to_goal_s=12.5,
                                       path_length_m=4.2,
                                       goal_reached_at_step=n_rows - 1))

    with h5py.File(path, "r") as f:
        assert f["features"].shape == (n_rows, D)
        assert f["robot_pose"].shape == (n_rows, 3)
        assert bool(f.attrs["success"]) is True
        assert int(f.attrs["goal_reached_at_step"]) == n_rows - 1
        assert pytest.approx(float(f.attrs["time_to_goal_s"])) == 12.5
        assert int(f.attrs["num_steps"]) == n_rows


def test_writer_pads_short_rows_with_nan(tmp_path):
    path = str(tmp_path / "ep_pad.h5")
    D, K = 12, 4
    short = CycleRow(
        features=np.zeros(5, dtype=np.float32),       # too short
        critic_costs=np.zeros(K, dtype=np.float32),
        critic_weights=np.zeros(K, dtype=np.float32),
        selected_action=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        robot_pose=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        goal=np.array([1.0, 1.0, 0.0], dtype=np.float32),
    )
    with EpisodeWriter(path, D, K, _make_meta()) as w:
        w.append(short)
        w.flush()

    with h5py.File(path, "r") as f:
        feats = f["features"][...]
        assert feats.shape == (1, D)
        # First 5 dims preserved, remainder NaN-padded.
        np.testing.assert_array_equal(feats[0, :5], np.zeros(5, np.float32))
        assert np.all(np.isnan(feats[0, 5:]))


def test_writer_with_dynamic_obstacles(tmp_path):
    path = str(tmp_path / "ep_obs.h5")
    D, K, M = 9, 3, 4
    with EpisodeWriter(path, D, K, _make_meta(), max_obstacles=M) as w:
        # First row: 2 obstacles. Second row: none.
        w.append(_row(D, K, with_obstacles=True, obstacles_n=2))
        w.append(_row(D, K, with_obstacles=False))
        w.flush()

    with h5py.File(path, "r") as f:
        assert "dynamic_obstacles" in f
        ds = f["dynamic_obstacles"][...]
        assert ds.shape == (2, M, 5)
        # Row 0 slots 0..1 are valid (non-NaN); slots 2..3 are NaN.
        assert not np.any(np.isnan(ds[0, :2, :]))
        assert np.all(np.isnan(ds[0, 2:, :]))
        # Row 1 entirely NaN.
        assert np.all(np.isnan(ds[1]))


def test_open_episode_reader_smoke(tmp_path):
    path = str(tmp_path / "ep_reader.h5")
    D, K = 18, 3
    with EpisodeWriter(path, D, K, _make_meta()) as w:
        for _ in range(5):
            w.append(_row(D, K))
        w.close(outcome=EpisodeOutcome(success=True))

    out = open_episode_reader(path)
    assert "features" in out
    assert out["features"].shape == (5, D)
    assert out["attrs"]["scenario_id"] == "unit_scenario"
    assert bool(out["attrs"]["success"]) is True
