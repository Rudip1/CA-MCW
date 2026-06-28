"""
Phase 9.4 — end-to-end tests for scripts/run_oracle.py.

Exercises the CLI on synthetic episodes:
  * single-file augmentation
  * directory mirror layout
  * idempotency on second run (skip), and --force override
  * exit codes
"""
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# Make scripts/ importable as a sibling package.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from vf_controller.data_collection.episode_writer import (  # noqa: E402
    CycleRow, EpisodeMetadata, EpisodeOutcome, EpisodeWriter)


@pytest.fixture(autouse=True)
def _need_cvxpy():
    pytest.importorskip("cvxpy")


def _import_cli():
    """Import scripts.run_oracle. Done lazily so non-cvxpy collectors
    don't trip on the heavy oracle imports."""
    sys.path.insert(0, str(_REPO / "scripts"))
    import run_oracle  # type: ignore[import-not-found]
    return run_oracle


CRITICS = [
    "WeightedConstraintCritic",
    "WeightedGoalCritic",
    "WeightedPathAlignCritic",
    "DynamicObstacleCritic",
]


def _write_synthetic(path: str, T: int = 3, K: int = 4) -> None:
    meta = EpisodeMetadata(
        scenario_id="phase9_4_unit",
        seed=3,
        controller_mode="collect",
        weight_provider="fixed",
        channels_config="channels_v1",
        channel_names=["a", "b"],
        channel_dims=[3, 2],
        critic_names=CRITICS[:K],
    )
    with EpisodeWriter(path, 5, K, meta, max_obstacles=0) as w:
        for t in range(T):
            w.append(CycleRow(
                features=np.zeros(5, dtype=np.float32),
                critic_costs=np.zeros(K, dtype=np.float32),
                critic_weights=np.full(K, 1.0 / K, dtype=np.float32),
                selected_action=np.zeros(3, dtype=np.float32),
                robot_pose=np.array([float(t) * 0.1, 0.0, 0.0], dtype=np.float32),
                goal=np.array([3.0, 0.0, 0.0], dtype=np.float32),
                dynamic_obstacles=None,
            ))
        w.close(outcome=EpisodeOutcome(success=True, goal_reached_at_step=T - 1))


def _common_argv(in_path, out_path):
    """Smallest oracle config so tests run in <1 s."""
    return [
        "--in", str(in_path),
        "--out", str(out_path),
        "--horizon", "8",
        "--samples", "32",
        "--seed", "0",
    ]


def test_single_file_augmentation(tmp_path):
    cli = _import_cli()
    src = tmp_path / "ep_single.h5"
    dst = tmp_path / "out" / "ep_single.h5"
    _write_synthetic(str(src), T=3, K=4)

    rc = cli.main(_common_argv(src, dst))
    assert rc == 0, "CLI should exit 0 on clean run"
    assert dst.exists(), "Output file must be created"

    with h5py.File(dst, "r") as f:
        assert "oracle_weights" in f, "augmentation dataset missing"
        w = f["oracle_weights"][...]
        assert w.shape == (3, 4)
        assert np.all(w >= 0)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-3)
        assert "oracle_horizon" in f.attrs
        assert int(f.attrs["oracle_horizon"]) == 8
        assert int(f.attrs["oracle_samples"]) == 32
        # original payload preserved
        assert "features" in f and f["features"].shape == (3, 5)


def test_idempotent_skip_then_force(tmp_path):
    cli = _import_cli()
    src = tmp_path / "ep_skip.h5"
    dst = tmp_path / "ep_skip_aug.h5"
    _write_synthetic(str(src), T=2, K=4)

    assert cli.main(_common_argv(src, dst)) == 0
    with h5py.File(dst, "r") as f:
        first_run_iso = str(f.attrs["oracle_run_iso"])
        first_weights = f["oracle_weights"][...].copy()

    # Second run without --force should skip — file mtime / weights unchanged.
    assert cli.main(_common_argv(src, dst)) == 0
    with h5py.File(dst, "r") as f:
        assert str(f.attrs["oracle_run_iso"]) == first_run_iso
        np.testing.assert_array_equal(f["oracle_weights"][...], first_weights)

    # With --force, the run must execute again and update the timestamp.
    rc = cli.main(_common_argv(src, dst) + ["--force"])
    assert rc == 0
    with h5py.File(dst, "r") as f:
        assert str(f.attrs["oracle_run_iso"]) != first_run_iso, (
            "--force should rewrite oracle_run_iso"
        )


def test_directory_mirror_layout(tmp_path):
    cli = _import_cli()
    src_root = tmp_path / "in_episodes"
    out_root = tmp_path / "out_episodes"
    nested = src_root / "batch" / "session_A"
    nested.mkdir(parents=True)
    _write_synthetic(str(nested / "ep_001.h5"), T=2, K=4)
    _write_synthetic(str(src_root / "ep_top.h5"), T=2, K=4)

    rc = cli.main(_common_argv(src_root, out_root))
    assert rc == 0
    # Mirror layout: top-level + nested both reproduced.
    assert (out_root / "ep_top.h5").exists()
    assert (out_root / "batch" / "session_A" / "ep_001.h5").exists()
    for p in (out_root / "ep_top.h5",
              out_root / "batch" / "session_A" / "ep_001.h5"):
        with h5py.File(p, "r") as f:
            assert "oracle_weights" in f


def test_missing_input_returns_nonzero(tmp_path):
    cli = _import_cli()
    bad = tmp_path / "does_not_exist.h5"
    out = tmp_path / "out.h5"
    with pytest.raises((FileNotFoundError, SystemExit)):
        cli.main(_common_argv(bad, out))


def test_in_place_augmentation_no_copy(tmp_path):
    """When --in == --out, the file is augmented in place rather than
    copied → unique inode, content preserved."""
    cli = _import_cli()
    p = tmp_path / "ep_inplace.h5"
    _write_synthetic(str(p), T=2, K=4)
    inode_before = os.stat(p).st_ino

    rc = cli.main(_common_argv(p, p))
    assert rc == 0
    assert os.stat(p).st_ino == inode_before, (
        "in-place augmentation must not replace the file"
    )
    with h5py.File(p, "r") as f:
        assert "oracle_weights" in f
        assert "features" in f  # original payload preserved
