"""
losses.py — Phase 8 loss functions.

INFERENCE loss components:
  ce_kl_sparsity_loss(logits, target_simplex, kl_weight, sparsity_weight)

IMITATION loss components:
  velocity_mse_loss(pred_xy, target_xy)

Where ``target_simplex`` is a row-stochastic (T, K) tensor produced by
oracle_weights.derive_inference_targets.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def cross_entropy_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft-target cross-entropy. logits: (B, K), target: (B, K) row-stochastic."""
    log_p = F.log_softmax(logits, dim=-1)
    return -(target * log_p).sum(dim=-1).mean()


def kl_to_uniform(logits: torch.Tensor) -> torch.Tensor:
    """KL(softmax(logits) || uniform). Penalises collapse onto a single critic."""
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    K = logits.shape[-1]
    uniform_log = -torch.log(torch.tensor(float(K), device=logits.device))
    # KL(p || u) = sum p (log p - log u) = sum p log p - log u
    return (p * (log_p - uniform_log)).sum(dim=-1).mean()


def negative_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Negative entropy of softmax(logits). Encourages confidence."""
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    return (p * log_p).sum(dim=-1).mean()


def ce_kl_sparsity_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ce_weight: float = 1.0,
    kl_weight: float = 0.01,
    sparsity_weight: float = 0.001,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined INFERENCE loss.

    Returns (scalar loss, components dict for logging).
    """
    if mask is not None and mask.numel() == logits.shape[0]:
        m = mask.bool()
        if m.any():
            logits = logits[m]
            target = target[m]
        else:
            zero = torch.zeros((), device=logits.device, requires_grad=True)
            return zero, {"ce": 0.0, "kl": 0.0, "sparsity": 0.0}

    ce = cross_entropy_to_target(logits, target)
    kl = kl_to_uniform(logits)
    sp = negative_entropy(logits)
    total = ce_weight * ce + kl_weight * kl + sparsity_weight * sp
    return total, {
        "ce": float(ce.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "sparsity": float(sp.detach().cpu()),
    }


def velocity_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """IMITATION loss = MSE(vx) + MSE(wz). Logs per-axis loss for debug."""
    vx_mse = F.mse_loss(pred[:, 0], target[:, 0])
    wz_mse = F.mse_loss(pred[:, 1], target[:, 1])
    total = vx_mse + wz_mse
    return total, {
        "vx_mse": float(vx_mse.detach().cpu()),
        "wz_mse": float(wz_mse.detach().cpu()),
    }
