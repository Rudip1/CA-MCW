"""
train_raw_critics.py — oracle-free training for the InferenceNet.

Labels are derived on-the-fly from raw critic costs already present in every
collect HDF5, so no run_oracle.py step is needed.

  label_k = softmax(critic_costs_k / temperature)

High cost → high weight → MPPI focuses more on satisfying that critic.
Temperature matches the runtime MPPI softmax temperature (vf_fixedwt.yaml
``temperature`` key, default 0.3). Higher values → more uniform labels →
easier to learn but less discriminative.

The trained model is exported as meta_critic.onnx / meta_critic.pt —
identical format to the oracle-trained model, so the same bringup YAML
deploys it (just point inference_model_type:= at the right family).

Usage:
  python3 src/vf_robot_controller/scripts/train.py --mode raw_critics \\
      --run-name my_run [--data-dir <path>] [--temperature 0.3] [--epochs 40]
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch

from vf_robot_utils.constants import TRAINING_ROOT, MODELS_METACRITIC_RAW_ROOT

_DEFAULT_DATA_DIR = str(TRAINING_ROOT)

from .data.dataset import EpisodeFeatureDataset, split_by_episode
from .data.log_reader import EpisodeReader, MultiEpisodeIndex
from .data.normalization import (FeatureNormalizationStats,
                                 compute_stats_from_episodes)
from .export.to_onnx import export_to_onnx, verify_torch_vs_onnx, write_metadata
from .models.inference_net import InferenceNet
from .training.log_writer import save_training_log
from .training.trainer import TrainConfig, train_inference as run_train


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Oracle-free RAW_CRITICS training: labels from softmax(critic_costs/T)")
    p.add_argument("--run-name", required=True,
                   help="Run tag, e.g. batchmanual_pravin_2026_05_11. "
                        "Final folder = models/metacritic_raw_wt/<channels>_<run-name>/. "
                        "Fails if the folder already exists.")
    p.add_argument("--channels", choices=["v1", "v2", "v3"], default="v3",
                   help="Channel set to train on. v1=126 dims, v2=130, v3=170. "
                        "Slices the v3 HDF5 prefix; default v3 (no slicing).")
    p.add_argument("--parent-dir", default=str(MODELS_METACRITIC_RAW_ROOT),
                   help="Parent directory where the run folder is created. "
                        "(default: models/metacritic_raw_wt/)")
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
                   help="Directory (or tree) of .h5 collect episodes "
                        "(default: vf_data/vf_data_training/; override with VF_DATA_ROOT)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Softmax temperature for label derivation. "
                        "Set to your runtime qp_T (e.g. 0.3) so labels match "
                        "the MPPI trajectory-selection temperature. (default: 1.0)")
    p.add_argument("--epochs", type=int, default=40)
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
    return p


def _parse_hidden(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    folder_name = f"{args.channels}_{args.run_name}"
    out_dir = os.path.join(args.parent_dir, folder_name)
    if os.path.exists(out_dir):
        print(f"[train_raw_critics] ERROR: run folder already exists: {out_dir}")
        print("[train_raw_critics] Use a different --run-name to avoid overwriting.")
        return 1
    os.makedirs(out_dir)

    idx = MultiEpisodeIndex.from_directory(args.data_dir)
    if len(idx) == 0:
        print(f"[raw_critics] no .h5 files found under {args.data_dir}")
        return 2

    n_critics = idx.critic_count()
    print(f"[raw_critics] {len(idx)} episodes  channels={args.channels}  "
          f"K={n_critics}  temperature={args.temperature}")

    splits = split_by_episode(
        [e.path for e in idx.entries],
        val_fraction=args.val_fraction, seed=args.seed)

    # Norm stats from train split only.
    train_eps = [EpisodeReader(p) for p in splits.train_paths]
    try:
        norm = compute_stats_from_episodes(train_eps, channels=args.channels)
    finally:
        for ep in train_eps:
            ep.close()

    norm_path = os.path.join(out_dir, "feature_norm.json")
    norm.save(norm_path)
    print(f"[raw_critics] norm stats -> {norm_path}")

    train_ds = EpisodeFeatureDataset(
        splits.train_paths, mode="raw_critics", norm=norm,
        raw_critics_temperature=args.temperature, channels=args.channels)
    val_ds = EpisodeFeatureDataset(
        splits.val_paths, mode="raw_critics", norm=norm,
        raw_critics_temperature=args.temperature, channels=args.channels)
    in_dim = train_ds.in_dim

    n_train_valid = int(train_ds.masks.sum())
    n_val_valid = int(val_ds.masks.sum())
    print(f"[raw_critics] train={len(train_ds)} rows ({n_train_valid} valid)  "
          f"val={len(val_ds)} rows ({n_val_valid} valid)")

    if n_train_valid == 0:
        print("[raw_critics] no valid training rows — check critic_costs in your HDF5s")
        return 3

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

    save_training_log(result.history, out_dir, kind="raw_critics",
                      run_id=folder_name)

    if result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)

    pt_path = os.path.join(out_dir, "meta_critic.pt")
    torch.save(model.state_dict(), pt_path)

    onnx_path = os.path.join(out_dir, "meta_critic.onnx")
    export_to_onnx(model, in_dim, onnx_path, output_name="weights")
    write_metadata(
        onnx_path,
        model_kind="meta_critic_raw_critics",
        in_dim=in_dim,
        target_dim=n_critics,
        channel_names=train_ds.channel_names,
        channel_dims=train_ds.channel_dims,
        critic_names=train_ds.critic_names,
        norm_path=norm_path,
        training_run_id=folder_name,
    )

    ok = verify_torch_vs_onnx(model, onnx_path, in_dim, n_samples=8)
    print(f"[raw_critics] export -> {onnx_path}  verify={'OK' if ok else 'SKIP'}")
    print(f"[raw_critics] best val_loss={result.best_val_loss:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
