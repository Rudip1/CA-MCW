"""
train_imitation.py — entrypoint for the IMITATION network (BC baseline).

Reads HDF5 episodes from a directory, splits by episode, trains the
ImitationNet, and exports `imitation.onnx` + `imitation.pt` +
`imitation.onnx.metadata.json` into `--out-dir`.

Defaults: --data-dir = <workspace>/vf_data/vf_data_training/ (override: VF_DATA_ROOT),
          --out-dir  = <workspace>/src/vf_robot_controller/models
"""
from __future__ import annotations

import argparse
import os
from typing import List

import torch

from vf_robot_utils.constants import TRAINING_ROOT, MODELS_IMITATION_ROOT


_DEFAULT_DATA_DIR = str(TRAINING_ROOT)

from .data.dataset import EpisodeFeatureDataset, split_by_episode
from .data.log_reader import EpisodeReader, MultiEpisodeIndex
from .data.normalization import compute_stats_from_episodes
from .export.to_onnx import (export_to_onnx, verify_torch_vs_onnx,
                             write_metadata)
from .models.imitation_net import ImitationNet
from .training.log_writer import save_training_log
from .training.trainer_imitation import (ImitationTrainConfig,
                                         train_imitation as run_train)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 8 IMITATION training")
    p.add_argument("--run-name", required=True,
                   help="Run tag, e.g. batchmanual_pravin_2026_05_11. "
                        "Final folder = models/imitation_wt/<channels>_<run-name>/. "
                        "Fails if the folder already exists.")
    p.add_argument("--channels", choices=["v1", "v2", "v3"], default="v3",
                   help="Channel set to train on. v1=126 dims, v2=130, v3=170. "
                        "Slices the v3 HDF5 prefix; default v3 (no slicing).")
    p.add_argument("--parent-dir", default=str(MODELS_IMITATION_ROOT),
                   help="Parent directory where the run folder is created. "
                        "(default: models/imitation_wt/)")
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
                   help="Directory (or tree) of .h5 episodes "
                        "(default: vf_data/vf_data_training/; override with VF_DATA_ROOT)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay (L2). Bump to 1e-3 for small-data "
                        "overfitting. Default 1e-4.")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--vx-max", type=float, default=0.30)
    p.add_argument("--wz-max", type=float, default=1.00)
    p.add_argument("--per-channel-hidden", type=int, default=32)
    p.add_argument("--fusion-hidden", default="256,128,64")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--zero-channels", default="",
                   help="Comma-separated channel names to mask to zero in the "
                        "training features (after channel-set slicing). Use this "
                        "for channels that are only populated at training time. "
                        "Example: --zero-channels critic_history makes the "
                        "model independent of MPPI's per-cycle critic costs so "
                        "it can drive when MPPI is off (vf_imitationwt / PASSIVE).")
    return p


def _parse_hidden(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    zero_channels: List[str] = [
        s.strip() for s in args.zero_channels.split(",") if s.strip()
    ]
    folder_name = f"{args.channels}_{args.run_name}"
    out_dir = os.path.join(args.parent_dir, folder_name)
    if os.path.exists(out_dir):
        print(f"[train_imitation] ERROR: run folder already exists: {out_dir}")
        print("[train_imitation] Use a different --run-name to avoid overwriting.")
        return 1
    os.makedirs(out_dir)

    idx = MultiEpisodeIndex.from_directory(args.data_dir)
    if len(idx) == 0:
        print(f"[train_imitation] no .h5 files in {args.data_dir}")
        return 2

    print(f"[train_imitation] {len(idx)} episodes, channels={args.channels} "
          f"(raw in_dim={idx.feature_dim()})")

    splits = split_by_episode(
        [e.path for e in idx.entries],
        val_fraction=args.val_fraction, seed=args.seed)

    train_eps = [EpisodeReader(p) for p in splits.train_paths]
    try:
        norm = compute_stats_from_episodes(
            train_eps, channels=args.channels,
            zero_channels=zero_channels,
        )
    finally:
        for ep in train_eps:
            ep.close()
    if zero_channels:
        print(f"[train_imitation] zeroing channels at train+norm time: "
              f"{zero_channels}")

    norm_path = os.path.join(out_dir, "feature_norm.json")
    norm.save(norm_path)
    print(f"[train_imitation] saved norm stats to {norm_path}")

    train_ds = EpisodeFeatureDataset(
        splits.train_paths, mode="imitation", norm=norm,
        channels=args.channels, zero_channels=zero_channels)
    val_ds = EpisodeFeatureDataset(
        splits.val_paths, mode="imitation", norm=norm,
        channels=args.channels, zero_channels=zero_channels)
    in_dim = train_ds.in_dim
    print(f"[train_imitation] train={len(train_ds)} rows, val={len(val_ds)} rows, "
          f"sliced in_dim={in_dim}")

    channel_dims = train_ds.channel_dims or None
    model = ImitationNet(
        in_dim=in_dim,
        channel_dims=channel_dims,
        per_channel_hidden=args.per_channel_hidden,
        fusion_hidden=_parse_hidden(args.fusion_hidden),
        dropout=args.dropout,
        vx_max=args.vx_max,
        wz_max=args.wz_max,
    )

    cfg = ImitationTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    result = run_train(model, train_ds, val_ds, cfg)

    # Persist training history to disk so figure pipelines and multi-seed
    # analysis (plan.md Phase E11.1) can read it back. Without this the only
    # training record is stdout.
    save_training_log(result.history, out_dir, kind="imitation",
                      run_id=folder_name)

    if result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)

    pt_path = os.path.join(out_dir, "imitation.pt")
    torch.save(model.state_dict(), pt_path)

    onnx_path = os.path.join(out_dir, "imitation.onnx")
    export_to_onnx(model, in_dim, onnx_path, output_name="cmd_vel")
    write_metadata(
        onnx_path,
        model_kind="imitation",
        in_dim=in_dim,
        target_dim=2,
        channel_names=train_ds.channel_names,
        channel_dims=train_ds.channel_dims,
        critic_names=train_ds.critic_names,
        norm_path=norm_path,
        training_run_id=folder_name,
        zero_channels=zero_channels,
    )

    ok = verify_torch_vs_onnx(model, onnx_path, in_dim, n_samples=8)
    print(f"[train_imitation] export -> {onnx_path}  verify={'OK' if ok else 'SKIP'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
