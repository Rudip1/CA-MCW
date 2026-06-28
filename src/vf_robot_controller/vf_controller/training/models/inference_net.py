"""
inference_net.py — Phase 8 INFERENCE network (the thesis contribution).

Architecture: channel-wise input MLP per docs/architecture.md, fused MLP head,
softplus output (non-negative critic-weight multipliers, no critic ever fully
silenced).

The frontend is identical to imitation_net.py — the only difference is the
output head. We do NOT subclass; sharing inheritance across the two networks
violates design anti-pattern #7 (don't try to unify them at runtime).
The duplication is deliberate and small.

Reference: docs/architecture.md (Phase 10 channel-wise MLP target shape).
For Phase 8 we accept a sensible default channel-dim layout (channels_v3
totals ~170) and ship with optional `channel_dims` so future channel sets
slot in without code changes.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ChannelWiseFrontend(nn.Module):
    """One small MLP per channel, outputs concatenated, then a fusion MLP."""

    def __init__(
        self,
        in_dim: int,
        channel_dims: Optional[Sequence[int]] = None,
        per_channel_hidden: int = 32,
        fusion_hidden: Sequence[int] = (256, 128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not channel_dims or sum(channel_dims) != in_dim:
            # Fall back to "one big channel" when the layout isn't supplied
            # — same total parameter count as the flat MLP baseline.
            channel_dims = [in_dim]
        self.channel_dims = list(channel_dims)
        self.per_channel: nn.ModuleList = nn.ModuleList(
            [_mlp(d, [per_channel_hidden], per_channel_hidden, dropout)
             for d in self.channel_dims]
        )
        fused_in = per_channel_hidden * len(self.channel_dims)
        self.fusion = _mlp(fused_in, list(fusion_hidden), fusion_hidden[-1], dropout)
        self.out_dim = fusion_hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice into channels along last dim; run each through its head.
        offsets = [0]
        for d in self.channel_dims:
            offsets.append(offsets[-1] + d)
        outs = []
        for i, head in enumerate(self.per_channel):
            outs.append(head(x[:, offsets[i]:offsets[i + 1]]))
        fused = torch.cat(outs, dim=-1)
        return self.fusion(fused)


class InferenceNet(nn.Module):
    """
    Outputs K non-negative weights via softplus + per-network temperature.

    Why softplus and not softmax:
      The IWeightProvider contract is "per-critic multipliers", not "a
      probability distribution over critics". Softmax would force the weights
      to sum to 1, which collapses the YAML-tuned base weights. Softplus keeps
      them in O(1) range while never going to zero (avoids fully silencing a
      critic — see the design notes cost-magnitude rule).

    Loss-side note:
      The trainer.py CE loss treats the prediction AS a probability for the
      cross-entropy term — we apply log-softmax internally there. The exported
      ONNX model still emits softplus weights (the C++ provider expects O(1)
      multipliers, not log-probs).
    """

    def __init__(
        self,
        in_dim: int,
        n_critics: int,
        channel_dims: Optional[Sequence[int]] = None,
        per_channel_hidden: int = 32,
        fusion_hidden: Sequence[int] = (256, 128, 64),
        dropout: float = 0.1,
        softplus_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.frontend = ChannelWiseFrontend(
            in_dim=in_dim,
            channel_dims=channel_dims,
            per_channel_hidden=per_channel_hidden,
            fusion_hidden=fusion_hidden,
            dropout=dropout,
        )
        self.head = nn.Linear(self.frontend.out_dim, n_critics)
        self.n_critics = int(n_critics)
        self.in_dim = int(in_dim)
        self.softplus_beta = float(softplus_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.frontend(x)
        raw = self.head(feat)
        # softplus -> always > 0; small positive offset guarantees no critic
        # ever drops below ~0.05 multiplier (anti-collapse).
        return F.softplus(raw, beta=self.softplus_beta) + 0.05

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-softplus logits — used by the trainer to derive a CE loss
        (we softmax these for the probability target)."""
        feat = self.frontend(x)
        return self.head(feat)
