"""
imitation_net.py — Phase 8 IMITATION network (behaviour-cloning baseline).

Architecture matches inference_net.py's frontend exactly. The only difference
is the head — 2 outputs (vx, wz) bounded by tanh + scale.

The default scales are the virofighter limits from the design notes:
  vx in [-0.20, 0.30]   (so we tanh into 0.30 and offset)
  wz in [-1.00, 1.00]

We use a symmetric tanh * vx_max for vx (i.e. tanh in [-1, 1] -> [-vx_max, +vx_max])
and clip the lower bound at vx_min in the controller. This keeps the network
output strictly differentiable while honoring the design notes "needs reverse to
recover from dead ends".

NOTE: this module is intentionally NOT a subclass of InferenceNet. Sharing a
common base would invite anti-pattern #7 (unifying INFERENCE and IMITATION at
runtime). The two networks are trained, exported, and consumed independently.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .inference_net import ChannelWiseFrontend


class ImitationNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        channel_dims: Optional[Sequence[int]] = None,
        per_channel_hidden: int = 32,
        fusion_hidden: Sequence[int] = (256, 128, 64),
        dropout: float = 0.1,
        vx_max: float = 0.30,
        wz_max: float = 1.00,
    ) -> None:
        super().__init__()
        self.frontend = ChannelWiseFrontend(
            in_dim=in_dim,
            channel_dims=channel_dims,
            per_channel_hidden=per_channel_hidden,
            fusion_hidden=fusion_hidden,
            dropout=dropout,
        )
        self.head = nn.Linear(self.frontend.out_dim, 2)
        self.vx_max = float(vx_max)
        self.wz_max = float(wz_max)
        self.in_dim = int(in_dim)
        self.target_dim = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.frontend(x)
        raw = self.head(feat)
        vx = torch.tanh(raw[:, 0]) * self.vx_max
        wz = torch.tanh(raw[:, 1]) * self.wz_max
        return torch.stack([vx, wz], dim=-1)
