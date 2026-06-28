#!/usr/bin/env python3
"""
scripts/export_model.py — Phase 8 ONNX export entrypoint.

Re-exports an already-trained .pt checkpoint to .onnx (and writes the sibling
metadata.json). Useful when the trainer was run with a stale ONNX opset and
the user wants to re-emit without re-training.

Usage:
  scripts/export_model.py --mode inference --ckpt models/meta_critic.pt \
                          --out  models/meta_critic.onnx \
                          --in-dim 170 --n-critics 11 --channel-dims 9,9,14,...
  scripts/export_model.py --mode imitation --ckpt models/imitation.pt \
                          --out  models/imitation.onnx --in-dim 170
"""
from __future__ import annotations

import argparse
import sys

import torch

from vf_controller.training.export.to_onnx import (export_to_onnx,
                                                   verify_torch_vs_onnx,
                                                   write_metadata)
from vf_controller.training.models.imitation_net import ImitationNet
from vf_controller.training.models.inference_net import InferenceNet


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["inference", "imitation"], required=True)
    p.add_argument("--ckpt", required=True, help=".pt state_dict file")
    p.add_argument("--out", required=True, help="output .onnx path")
    p.add_argument("--in-dim", type=int, required=True)
    p.add_argument("--n-critics", type=int, default=11)
    p.add_argument("--channel-dims", default="",
                   help="comma-separated channel dims for the channel-wise frontend")
    p.add_argument("--vx-max", type=float, default=0.30)
    p.add_argument("--wz-max", type=float, default=1.00)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args(argv)

    cd = ([int(x) for x in args.channel_dims.split(",") if x]
          if args.channel_dims else None)

    if args.mode == "inference":
        model = InferenceNet(in_dim=args.in_dim, n_critics=args.n_critics,
                             channel_dims=cd)
        out_name = "weights"
    else:
        model = ImitationNet(in_dim=args.in_dim, channel_dims=cd,
                             vx_max=args.vx_max, wz_max=args.wz_max)
        out_name = "cmd_vel"

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    export_to_onnx(model, args.in_dim, args.out, output_name=out_name)
    write_metadata(
        args.out,
        model_kind=f"{args.mode}{'_inference' if args.mode == 'inference' else ''}",
        in_dim=args.in_dim,
        target_dim=args.n_critics if args.mode == "inference" else 2,
        channel_dims=cd or [],
    )
    if not args.no_verify:
        ok = verify_torch_vs_onnx(model, args.out, args.in_dim, n_samples=8)
        print(f"[export_model] verify={'OK' if ok else 'SKIP'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
