#!/usr/bin/env python3
"""scripts/phase9_acceptance.py — Phase 9.7 acceptance verifier.

Checks the four code-side gates from `plan.md` Phase 9.7:

  1. Recovered-weights distribution non-uniformity
       mean Shannon entropy < log(K) - 0.2
  2. Context-conditioned weight variation
       Spearman corr(goal_distance, goal_critic_weight) < 0  with p < 0.01
  3. Inference val_ce floor
       min val_ce across seeds < 0.7 * log(K)
  4. (Acceptance #5 — Gazebo eval beating MPPI — is NOT checked here.
      That gate lives in vf_robot_utils Phase E10 and runs against the
      full evaluation harness.)

Reads:
  - All `*.h5` under <root> recursively (default: $PWD/vf_data/vf_data_training).
  - All `<MODELS_ROOT>/train_log_inference_*.csv` for #3.

Usage:
  python3 scripts/phase9_acceptance.py [--rootvf_data/vf_data_training] [--models-dir models]
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import stats


GOAL_CRITIC_INDEX = 2  # WeightedGoalCritic in the standard 11-critic order.


def _walk_h5(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.h5") if p.is_file())


def _episode_has_oracle(p: Path) -> bool:
    try:
        with h5py.File(str(p), "r") as f:
            return "oracle_weights" in f
    except Exception:
        return False


def _entropy_per_row(W: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -np.sum(W * np.log(W + eps), axis=1)


def gate_1_entropy(files: list[Path]) -> tuple[bool, str]:
    """Mean entropy of recovered weights stays below log(K) - 0.2."""
    rows = []
    K = None
    for p in files:
        with h5py.File(str(p), "r") as f:
            if "oracle_weights" not in f:
                continue
            W = f["oracle_weights"][...].astype(np.float64)
            if K is None:
                K = W.shape[1]
            rows.append(_entropy_per_row(W))
    if not rows:
        return False, "no oracle_weights found"
    H = np.concatenate(rows)
    mean_H = float(H.mean())
    threshold = math.log(K) - 0.2
    ok = mean_H < threshold
    return ok, (
        f"mean entropy = {mean_H:.4f} bits   "
        f"threshold = log(K={K}) - 0.2 = {threshold:.4f}   "
        f"{'PASS' if ok else 'FAIL'}"
    )


def gate_2_correlation(files: list[Path]) -> tuple[bool, str]:
    """Spearman: goal_distance vs goal-critic weight should be negative,
    p < 0.01.
    """
    gd, gw = [], []
    for p in files:
        with h5py.File(str(p), "r") as f:
            if "oracle_weights" not in f:
                continue
            if "robot_pose" not in f or "goal" not in f:
                continue
            W = f["oracle_weights"][...]
            pose = f["robot_pose"][...]
            goal = f["goal"][...]
            T = min(W.shape[0], pose.shape[0], goal.shape[0])
            d = np.linalg.norm(goal[:T, :2] - pose[:T, :2], axis=1)
            gd.append(d)
            gw.append(W[:T, GOAL_CRITIC_INDEX])
    if not gd:
        return False, "no episodes had robot_pose + goal + oracle_weights"
    gd_all = np.concatenate(gd)
    gw_all = np.concatenate(gw)
    rho, p = stats.spearmanr(gd_all, gw_all)
    ok = (rho < 0.0) and (p < 0.01)
    return ok, (
        f"Spearman rho = {rho:+.4f}   p = {p:.2e}   "
        f"n = {len(gd_all)}   {'PASS' if ok else 'FAIL'}"
    )


def gate_3_val_ce(models_dir: Path, K: int) -> tuple[bool, str]:
    """Minimum val_ce across seeded train logs must be < 0.7 * log(K)."""
    logs = sorted(models_dir.glob("train_log_inference_seed*.csv"))
    if not logs:
        return False, f"no seed training logs in {models_dir}"
    best_per_seed = {}
    for p in logs:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            ces = []
            for row in reader:
                # 'ce' column may be absent if schema changes; fall back to val_loss.
                v = row.get("ce") or row.get("val_ce") or row.get("val_loss")
                if v is None or v == "":
                    continue
                try:
                    ces.append(float(v))
                except ValueError:
                    pass
            if ces:
                best_per_seed[p.stem] = min(ces)
    if not best_per_seed:
        return False, "no parseable val_ce values"
    min_ce = min(best_per_seed.values())
    threshold = 0.7 * math.log(K)
    ok = min_ce < threshold
    detail = ", ".join(f"{k}={v:.3f}" for k, v in best_per_seed.items())
    return ok, (
        f"min val_ce = {min_ce:.4f}   threshold = 0.7*log(K={K}) = {threshold:.4f}   "
        f"{'PASS' if ok else 'FAIL'}\n  per-seed best: {detail}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.cwd() / "vf_data/vf_data_training"))
    ap.add_argument("--models-dir", default=None,
                    help="Defaults to src/vf_robot_controller/models")
    args = ap.parse_args()

    root = Path(args.root)
    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        models_dir = (
            Path(__file__).resolve().parent.parent / "models"
        )

    files = [p for p in _walk_h5(root) if _episode_has_oracle(p)]
    if not files:
        print(f"[acceptance] no oracle-augmented episodes under {root}")
        return 1

    # Pick K from the first episode that has any.
    with h5py.File(str(files[0]), "r") as f:
        K = int(f["oracle_weights"].shape[1])

    print(f"[acceptance] {len(files)} oracle-augmented episodes under {root}")
    print(f"[acceptance] K = {K}   models_dir = {models_dir}")
    print()

    gates = [
        ("9.7-2  weight entropy non-uniform     ", gate_1_entropy(files)),
        ("9.7-3  goal-distance Spearman corr    ", gate_2_correlation(files)),
        ("9.7-4  val_ce below ln(K) floor       ", gate_3_val_ce(models_dir, K)),
    ]

    all_pass = True
    for label, (ok, msg) in gates:
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] {label} :: {msg}")
        if not ok:
            all_pass = False

    print()
    print("=" * 72)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 72)
    print("Note: gate 9.7-1 (unit tests) verified separately via colcon test.")
    print("Note: gate 9.7-5 (Gazebo eval beats MPPI) deferred to "
          "vf_robot_utils Phase E10.")
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
