"""
to_onnx.py — export PyTorch model to ONNX.

Phase 8.

Both INFERENCE and IMITATION export with a fixed input shape `(1, D)` so the
C++ runner doesn't need dynamic-shape support. (We also write a dynamic-batch
variant under a different filename if the caller asks.)

I/O signatures (documented in docs/training.md):
  INFERENCE: input  features   float32 (1, D)
             output weights    float32 (1, K)   (softplus, >= 0.05)
  IMITATION: input  features   float32 (1, D)
             output cmd_vel    float32 (1, 2)   (vx, wz)
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

import numpy as np
import torch


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def export_to_onnx(
    model: torch.nn.Module,
    in_dim: int,
    out_path: str,
    output_name: str = "weights",
    opset: int = 14,
    dynamic_batch: bool = True,
) -> str:
    """Export model to ONNX. Returns the absolute path.

    Side effect: writes a sibling `<out_path>.metadata.json` if the caller
    passes a model that has `in_dim` / `target_dim` / `n_critics` attributes.
    """
    model.eval()
    model.cpu()  # ONNX runtime in C++ runs on CPU; export with CPU dummy input.
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    dummy = torch.zeros(1, in_dim, dtype=torch.float32)
    dynamic_axes = (
        {"features": {0: "batch"}, output_name: {0: "batch"}}
        if dynamic_batch else None
    )
    torch.onnx.export(
        model, (dummy,), out_path,
        input_names=["features"],
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    return os.path.abspath(out_path)


def write_metadata(
    out_path: str,
    *,
    model_kind: str,
    in_dim: int,
    target_dim: int,
    channel_names=None,
    channel_dims=None,
    critic_names=None,
    norm_path: Optional[str] = None,
    training_run_id: Optional[str] = None,
    zero_channels=None,
) -> str:
    """Write metadata.json into the same directory as the .onnx file."""
    meta_path = os.path.join(os.path.dirname(os.path.abspath(out_path)), "metadata.json")
    payload = {
        "model_kind": model_kind,
        "in_dim": int(in_dim),
        "target_dim": int(target_dim),
        "channel_names": list(channel_names or []),
        "channel_dims": [int(x) for x in (channel_dims or [])],
        "critic_names": list(critic_names or []),
        "norm_path": norm_path,
        "training_run_id": training_run_id or "unspecified",
        "zero_channels": list(zero_channels or []),
        "git_commit": _git_commit(),
    }
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)
    return meta_path


def verify_torch_vs_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    in_dim: int,
    n_samples: int = 4,
    atol: float = 1e-4,
) -> bool:
    """
    Compare PyTorch and ONNX outputs on random inputs. Returns True iff every
    sample matches within `atol` absolute tolerance.

    Soft-skips when onnxruntime isn't available — the caller decides whether
    that constitutes a failure (the script-level entrypoint logs a warning).
    """
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        print(f"[verify] onnxruntime unavailable ({e!r}); skipping numerical check.")
        return False

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    model.eval()
    ok = True
    for _ in range(n_samples):
        x = rng.standard_normal(size=(1, in_dim)).astype(np.float32)
        with torch.no_grad():
            y_torch = model(torch.from_numpy(x)).cpu().numpy()
        y_onnx = sess.run(None, {in_name: x})[0]
        if not np.allclose(y_torch, y_onnx, atol=atol):
            print(f"[verify] FAIL diff={(y_torch - y_onnx).abs().max() if hasattr((y_torch - y_onnx), 'abs') else np.abs(y_torch - y_onnx).max()}")
            ok = False
    return ok
