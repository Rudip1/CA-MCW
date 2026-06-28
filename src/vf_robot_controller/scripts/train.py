#!/usr/bin/env python3
"""
scripts/train.py — unified training entrypoint.

Dispatches to the INFERENCE, IMITATION, or RAW_CRITICS trainer.
Works without sourcing install/setup.bash — adds both source packages to
sys.path automatically so the dl conda env is sufficient.

Usage (from workspace root ~/CA-MCW):
  python3 src/vf_robot_controller/scripts/train.py --mode raw_critics \\
      --data-dir $PWD/vf_data/vf_data_training --temperature 0.3 --epochs 40

  python3 src/vf_robot_controller/scripts/train.py --mode inference
  python3 src/vf_robot_controller/scripts/train.py --mode imitation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# ── path bootstrap ────────────────────────────────────────────────────────────
# Resolve workspace root (two levels up from scripts/): .../CA-MCW
_SCRIPTS = Path(__file__).resolve().parent          # .../scripts
_PKG     = _SCRIPTS.parent                          # .../vf_robot_controller
_SRC     = _PKG.parent                              # .../src
_WS      = _SRC.parent                              # .../CA-MCW

for _p in [
    str(_PKG),                                              # vf_controller.*
    str(_SRC / "vf_robot_utils"),                           # vf_robot_utils.*
    str(_WS / "install" / "vf_robot_controller" /
        "local" / "lib" / "python3.10" / "dist-packages"), # colcon-installed
    str(_WS / "install" / "vf_robot_utils" /
        "local" / "lib" / "python3.10" / "dist-packages"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Training dispatcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  raw_critics   oracle-free: labels = softmax(critic_costs / temperature)
  inference     oracle labels (run_oracle.py must have augmented HDF5s first)
  imitation     behaviour cloning on (vx, wz)
""",
    )
    p.add_argument("--mode",
                   choices=["inference", "imitation", "raw_critics"],
                   required=True)
    args, rest = p.parse_known_args(argv)

    if args.mode == "inference":
        from vf_controller.training.train_inference import main as run_train
    elif args.mode == "raw_critics":
        from vf_controller.training.train_raw_critics import main as run_train
    else:
        from vf_controller.training.train_imitation import main as run_train
    return run_train(rest)


if __name__ == "__main__":
    raise SystemExit(main())
