#!/usr/bin/env python3
"""
scripts/compute_normalization.py — compute feature stats on a directory of
HDF5 episodes and save them to feature_norm.json.

Both INFERENCE and IMITATION reuse the saved file (no need to recompute).

Usage:
  scripts/compute_normalization.py --data-dir ~/CA-MCW/vf_data/vf_data_training \
                                   --out feature_norm.json
"""
from __future__ import annotations

import argparse

from vf_controller.training.data.log_reader import (EpisodeReader,
                                                    MultiEpisodeIndex)
from vf_controller.training.data.normalization import \
    compute_stats_from_episodes


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="feature_norm.json")
    args = p.parse_args(argv)

    idx = MultiEpisodeIndex.from_directory(args.data_dir)
    if len(idx) == 0:
        print(f"[compute_normalization] no .h5 files in {args.data_dir}")
        return 2

    eps = [EpisodeReader(e.path) for e in idx.entries]
    try:
        stats = compute_stats_from_episodes(eps)
    finally:
        for e in eps:
            e.close()

    stats.save(args.out)
    print(f"[compute_normalization] saved {args.out}  "
          f"D={stats.mean.shape[0]}  rows={sum(e.num_steps for e in idx.entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
