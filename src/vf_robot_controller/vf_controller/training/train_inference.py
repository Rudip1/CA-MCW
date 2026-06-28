"""
train_inference.py — entrypoint for the INFERENCE network (thesis contribution).

Reads HDF5 episodes from a directory, splits by episode, trains the
InferenceNet, and exports `meta_critic.onnx` + `meta_critic.pt` +
`meta_critic.onnx.metadata.json` into `--out-dir`.

Usage (zero-config, uses workspace-relative defaults):
  python -m vf_controller.training.train_inference --epochs 20

  Defaults:
    --data-dir = <workspace_root>/vf_data/vf_data_training/  (TRAINING_ROOT; VF_DATA_ROOT)
    --out-dir  = <workspace_root>/src/vf_robot_controller/models  (MODELS_ROOT)

  Override either with explicit paths or the VF_DATA_ROOT / VF_MODELS_ROOT
  environment variables.

(or via `scripts/train.py --mode inference [--data-dir ... --out-dir ...]`).
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch

from vf_robot_utils.constants import TRAINING_ROOT, MODELS_METACRITIC_ORACLE_ROOT

_DEFAULT_DATA_DIR = str(TRAINING_ROOT)

from .data.dataset import build_datasets
from .data.log_reader import EpisodeReader, MultiEpisodeIndex
from .data.normalization import (FeatureNormalizationStats,
                                 compute_stats_from_episodes)
from .export.to_onnx import (export_to_onnx, verify_torch_vs_onnx,
                             write_metadata)
from .models.inference_net import InferenceNet
from .training.log_writer import save_training_log
from .training.trainer import TrainConfig, train_inference as run_train


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 8 INFERENCE training")
    p.add_argument("--run-name", required=True,
                   help="Run tag, e.g. batchmanual_pravin_2026_05_11. "
                        "Final folder = models/metacritic_oracle_wt/<channels>_<run-name>/. "
                        "Fails if the folder already exists.")
    p.add_argument("--channels", choices=["v1", "v2", "v3"], default="v3",
                   help="Channel set to train on. v1=126 dims, v2=130, v3=170. "
                        "Slices the v3 HDF5 prefix; default v3 (no slicing).")
    p.add_argument("--parent-dir", default=str(MODELS_METACRITIC_ORACLE_ROOT),
                   help="Parent directory where the run folder is created. "
                        "(default: models/metacritic_oracle_wt/)")
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
                   help="Directory (or tree) of .h5 episodes "
                        "(default: vf_data/vf_data_training/; override with VF_DATA_ROOT)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay (L2). Bump to 1e-3 for small-data "
                        "overfitting. Default 1e-4.")
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=0.01)
    p.add_argument("--sparsity-weight", type=float, default=0.001)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--per-channel-hidden", type=int, default=32)
    p.add_argument("--fusion-hidden", default="256,128,64")
    p.add_argument("--dropout", type=float, default=0.1)
    # Phase 9.5: oracle-by-default. The hindsight target (recorded
    # critic_weights_applied) was the Phase-8 fallback; with the oracle
    # pipeline live, training on hindsight silently produces a degenerate
    # net that re-emits the FixedWeightProvider policy. Default to refusing
    # any episode without an oracle_weights dataset; the user opts in
    # explicitly with --allow-hindsight to mix or fall back.
    p.add_argument(
        "--allow-hindsight",
        action="store_true",
        help="Allow training on hindsight (Phase-8) labels for episodes "
             "without an oracle_weights dataset. Off by default — train "
             "fails fast if any episode in the corpus has no oracle labels.",
    )
    return p


def _parse_hidden(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    folder_name = f"{args.channels}_{args.run_name}"
    out_dir = os.path.join(args.parent_dir, folder_name)
    if os.path.exists(out_dir):
        print(f"[train_inference] ERROR: run folder already exists: {out_dir}")
        print("[train_inference] Use a different --run-name to avoid overwriting.")
        return 1
    os.makedirs(out_dir)

    idx = MultiEpisodeIndex.from_directory(args.data_dir)
    if len(idx) == 0:
        print(f"[train_inference] no .h5 files in {args.data_dir}")
        return 2

    n_critics = idx.critic_count()
    print(f"[train_inference] {len(idx)} episodes, channels={args.channels}, "
          f"K={n_critics}")

    # Compute normalization stats on the TRAIN split only.
    from .data.dataset import split_by_episode
    splits = split_by_episode(
        [e.path for e in idx.entries],
        val_fraction=args.val_fraction, seed=args.seed)

    train_eps = [EpisodeReader(p) for p in splits.train_paths]
    try:
        norm = compute_stats_from_episodes(train_eps, channels=args.channels)
    finally:
        for ep in train_eps:
            ep.close()

    norm_path = os.path.join(out_dir, "feature_norm.json")
    norm.save(norm_path)
    print(f"[train_inference] saved norm stats to {norm_path}")

    # Build the actual datasets (train + val) with normalization applied.
    from .data.dataset import EpisodeFeatureDataset
    train_ds = EpisodeFeatureDataset(
        splits.train_paths, mode="inference", norm=norm, channels=args.channels)
    val_ds = EpisodeFeatureDataset(
        splits.val_paths, mode="inference", norm=norm, channels=args.channels)
    in_dim = train_ds.in_dim
    print(f"[train_inference] train={len(train_ds)} rows, val={len(val_ds)} rows, "
          f"sliced in_dim={in_dim}")

    # Phase 9.5 corpus-mix audit. Refuse to silently train on hindsight
    # labels — that path produces a degenerate net that re-emits the
    # FixedWeightProvider policy.
    n_oracle = train_ds.n_oracle_episodes + val_ds.n_oracle_episodes
    n_hind = train_ds.n_hindsight_episodes + val_ds.n_hindsight_episodes
    print(f"[train_inference] label source: oracle={n_oracle} ep, "
          f"hindsight={n_hind} ep")
    if n_hind > 0 and not args.allow_hindsight:
        print(
            "[train_inference] aborting: {} episode(s) have no oracle_weights "
            "dataset. Run scripts/run_oracle.py first to augment them, or "
            "pass --allow-hindsight to train on the Phase-8 fallback "
            "(degenerate target, debug only).".format(n_hind)
        )
        return 3

    # Channel dims (for the channel-wise frontend). If absent, frontend falls
    # back to a flat MLP automatically.
    channel_dims = train_ds.channel_dims or None

    model = InferenceNet(
        in_dim=in_dim,
        n_critics=n_critics,
        channel_dims=channel_dims,
        per_channel_hidden=args.per_channel_hidden,
        fusion_hidden=_parse_hidden(args.fusion_hidden),
        dropout=args.dropout,
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ce_weight=args.ce_weight,
        kl_weight=args.kl_weight,
        sparsity_weight=args.sparsity_weight,
        device=args.device,
    )
    result = run_train(model, train_ds, val_ds, cfg)

    # Persist training history to disk so figure pipelines and multi-seed
    # analysis (plan.md Phase E11.1) can read it back. Without this the only
    # training record is stdout.
    save_training_log(result.history, out_dir, kind="inference",
                      run_id=folder_name)

    # Restore best weights for export.
    if result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)

    pt_path = os.path.join(out_dir, "meta_critic.pt")
    torch.save(model.state_dict(), pt_path)

    onnx_path = os.path.join(out_dir, "meta_critic.onnx")
    export_to_onnx(model, in_dim, onnx_path, output_name="weights")
    write_metadata(
        onnx_path,
        model_kind="meta_critic_inference",
        in_dim=in_dim,
        target_dim=n_critics,
        channel_names=train_ds.channel_names,
        channel_dims=train_ds.channel_dims,
        critic_names=train_ds.critic_names,
        norm_path=norm_path,
        training_run_id=folder_name,
    )

    ok = verify_torch_vs_onnx(model, onnx_path, in_dim, n_samples=8)
    print(f"[train_inference] export -> {onnx_path}  verify={'OK' if ok else 'SKIP'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
