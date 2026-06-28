#!/usr/bin/env python3
"""
scripts/run_oracle.py — Phase 9.4

Run the offline replay oracle over one HDF5 episode (or a directory of
them) and write augmented HDF5(s) with an extra ``oracle_weights``
dataset and provenance attrs.

Usage::

    python3 -m scripts.run_oracle \
        --in  <vf_data/vf_data_training_dir_or_file.h5> \
        --out <augmented_dir_or_file.h5> \
        [--horizon 40] [--samples 1024] [--workers 4] \
        [--qp-T 0.3] [--qp-lam 0.1] \
        [--oracle-margin-weight 5.0] [--obstacle-kernel-sigma 0.6] \
        [--seed 0] [--force]

Input handling:
  --in is a file:       process that one episode → --out is the file path.
  --in is a directory:  walk recursively for ``*.h5``; --out must also be a
                        directory and the layout is mirrored.

Idempotency:
  An output file with an ``oracle_weights`` dataset already present is
  skipped unless ``--force`` is set. The skip is reported on stdout but
  is not an error.

Augmentation written:
  Dataset      ``oracle_weights``         (T, K) float32, simplex per row
  Attribute    ``oracle_horizon``         int
  Attribute    ``oracle_samples``         int
  Attribute    ``oracle_qp_T``            float
  Attribute    ``oracle_qp_lam``          float
  Attribute    ``oracle_margin_weight``   float
  Attribute    ``oracle_kernel_sigma``    float
  Attribute    ``oracle_seed``            int
  Attribute    ``oracle_run_iso``         ISO-8601 timestamp
  Attribute    ``oracle_run_git``         short commit at run time
"""
from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

# Local package imports — added to path below for direct ``python3
# scripts/run_oracle.py`` invocation; ``-m scripts.run_oracle`` already
# resolves them via the package layout.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vf_controller.training.oracle.replay_oracle import (  # noqa: E402
    MPPIConfig, ReplayOracle)
from vf_controller.training.oracle.world_reconstructor import (  # noqa: E402
    reconstruct)


ORACLE_DATASET = "oracle_weights"


@dataclass
class RunOptions:
    horizon: int
    samples: int
    qp_T: float
    qp_lam: float
    oracle_margin_weight: float
    obstacle_kernel_sigma: float
    seed: int
    force: bool

    def to_mppi_config(self) -> MPPIConfig:
        return MPPIConfig(
            horizon=self.horizon,
            n_samples=self.samples,
            qp_T=self.qp_T,
            qp_lam=self.qp_lam,
            oracle_margin_weight=self.oracle_margin_weight,
            obstacle_kernel_sigma=self.obstacle_kernel_sigma,
            rng_seed=self.seed,
        )


# =============================================================================
# Per-file processing
# =============================================================================

def _critic_names(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as f:
        names = f.attrs.get("critic_names", None)
        if names is None:
            return []
        out: List[str] = []
        for x in (names.tolist() if hasattr(names, "tolist") else names):
            out.append(x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x))
        return out


def _has_oracle(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return ORACLE_DATASET in f
    except Exception:
        return False


def _git_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO), stderr=subprocess.DEVNULL, timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _augment_in_place(out_path: str, weights: np.ndarray, opts: RunOptions) -> None:
    """Write oracle_weights + provenance into an existing output HDF5."""
    with h5py.File(out_path, "a") as f:
        if ORACLE_DATASET in f:
            del f[ORACLE_DATASET]
        f.create_dataset(
            ORACLE_DATASET,
            data=weights.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        a = f.attrs
        a["oracle_horizon"] = int(opts.horizon)
        a["oracle_samples"] = int(opts.samples)
        a["oracle_qp_T"] = float(opts.qp_T)
        a["oracle_qp_lam"] = float(opts.qp_lam)
        a["oracle_margin_weight"] = float(opts.oracle_margin_weight)
        a["oracle_kernel_sigma"] = float(opts.obstacle_kernel_sigma)
        a["oracle_seed"] = int(opts.seed)
        a["oracle_run_iso"] = datetime.datetime.now(
            tz=datetime.timezone.utc).isoformat()
        a["oracle_run_git"] = _git_short()


def process_one(in_path: str, out_path: str, opts: RunOptions) -> Tuple[str, str]:
    """Run the oracle on one HDF5 episode.

    Returns a (status, message) pair. Status is one of:
      "ok"       — augmentation written successfully
      "skipped"  — output already augmented and --force not set
      "empty"    — episode has no critic_names; nothing to recover
      "error"    — unexpected failure (message contains short reason)
    """
    try:
        if (not opts.force) and _has_oracle(out_path):
            return ("skipped", out_path)

        names = _critic_names(in_path)
        if not names:
            return ("empty", in_path)

        # Copy input → output (preserve the original episode payload), then
        # write augmentation in place. ``shutil.copy2`` preserves mtime.
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        if os.path.abspath(in_path) != os.path.abspath(out_path):
            shutil.copy2(in_path, out_path)

        replay = reconstruct(in_path)
        oracle = ReplayOracle(replay, names, opts.to_mppi_config())
        weights = oracle.compute_oracle_weights()

        _augment_in_place(out_path, weights, opts)
        return ("ok", f"{out_path} [T={weights.shape[0]} K={weights.shape[1]}]")
    except Exception as exc:  # pragma: no cover - reported, not raised
        return ("error", f"{in_path}: {exc.__class__.__name__}: {exc}")


# =============================================================================
# Path resolution
# =============================================================================

def _list_h5(in_path: Path) -> List[Path]:
    if in_path.is_file():
        return [in_path]
    if in_path.is_dir():
        return sorted(in_path.rglob("*.h5"))
    raise FileNotFoundError(f"--in does not exist: {in_path}")


def _plan_pairs(in_path: Path, out_path: Path) -> List[Tuple[Path, Path]]:
    sources = _list_h5(in_path)
    if not sources:
        return []

    if in_path.is_file():
        if out_path.is_dir() or str(out_path).endswith(os.sep):
            return [(in_path, out_path / in_path.name)]
        return [(in_path, out_path)]

    # in_path is a dir — mirror layout under out_path.
    pairs: List[Tuple[Path, Path]] = []
    for src in sources:
        rel = src.relative_to(in_path)
        pairs.append((src, out_path / rel))
    return pairs


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_oracle",
        description="Phase 9 replay oracle — augments HDF5 episodes with "
                    "per-timestep ideal critic weights via convex QP.",
    )
    p.add_argument("--in", dest="in_path", required=True,
                   help="Input HDF5 file or directory of episodes.")
    p.add_argument("--out", dest="out_path", required=True,
                   help="Output HDF5 file or directory; layout is mirrored "
                        "when --in is a directory.")
    p.add_argument("--horizon", type=int, default=40,
                   help="MPPI rollout horizon (steps). Default: 40.")
    p.add_argument("--samples", type=int, default=1024,
                   help="Trajectory samples per timestep. Default: 1024. "
                        "Plan §9.3 suggests 40000 for production runs.")
    p.add_argument("--qp-T", type=float, default=0.3,
                   help="Softmax temperature for the QP. Match runtime "
                        "vf_fixedwt.yaml temperature (default 0.3).")
    p.add_argument("--qp-lam", type=float, default=0.1,
                   help="KL prior weight for the QP. Default: 0.1.")
    p.add_argument("--oracle-margin-weight", type=float, default=5.0,
                   help="Weight on the oracle's soft-margin term in i_star "
                        "selection. Default: 5.0.")
    p.add_argument("--obstacle-kernel-sigma", type=float, default=0.6,
                   help="Gaussian kernel std (m) for the dynamic-obstacle "
                        "soft cost. Default: 0.6.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base RNG seed. Each file uses seed+file_index so "
                        "parallel runs are deterministic.")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel workers. Default: 1 (serial).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite oracle_weights even if present.")
    return p


def _worker(task: Tuple[str, str, RunOptions]) -> Tuple[str, str]:
    in_path, out_path, opts = task
    try:
        return process_one(in_path, out_path, opts)
    except Exception:  # pragma: no cover - last-resort guard
        return ("error", traceback.format_exc(limit=2))


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    base_opts = RunOptions(
        horizon=args.horizon,
        samples=args.samples,
        qp_T=args.qp_T,
        qp_lam=args.qp_lam,
        oracle_margin_weight=args.oracle_margin_weight,
        obstacle_kernel_sigma=args.obstacle_kernel_sigma,
        seed=args.seed,
        force=args.force,
    )

    pairs = _plan_pairs(in_path, out_path)
    if not pairs:
        print(f"[run_oracle] no .h5 episodes found under {in_path}")
        return 1

    # Per-file seeds: deterministic across parallel runs.
    tasks = []
    for idx, (src, dst) in enumerate(pairs):
        opts = RunOptions(**{**asdict(base_opts), "seed": base_opts.seed + idx})
        tasks.append((str(src), str(dst), opts))

    print(f"[run_oracle] processing {len(tasks)} episode(s) "
          f"with workers={args.workers} samples={args.samples} "
          f"horizon={args.horizon}")

    results: List[Tuple[str, str]] = []
    if args.workers > 1 and len(tasks) > 1:
        with mp.get_context("spawn").Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(_worker, tasks):
                _print_result(r)
                results.append(r)
    else:
        for t in tasks:
            r = _worker(t)
            _print_result(r)
            results.append(r)

    n_ok = sum(1 for s, _ in results if s == "ok")
    n_skip = sum(1 for s, _ in results if s == "skipped")
    n_empty = sum(1 for s, _ in results if s == "empty")
    n_err = sum(1 for s, _ in results if s == "error")
    print(f"[run_oracle] done — ok={n_ok} skipped={n_skip} "
          f"empty={n_empty} errors={n_err}")
    return 0 if n_err == 0 else 2


def _print_result(r: Tuple[str, str]) -> None:
    status, msg = r
    tag = {
        "ok": "[ok]      ",
        "skipped": "[skip]    ",
        "empty": "[empty]   ",
        "error": "[error]   ",
    }.get(status, "[?]       ")
    print(tag + msg)


if __name__ == "__main__":
    raise SystemExit(main())
