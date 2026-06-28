"""
mlp_baseline.py — flat-MLP INFERENCE baseline.

Phase 8. The phase-8 thesis spec also mentions a "flat baseline" form of the
network (input -> 256 -> 128 -> 64 -> K -> softmax) as a comparison against
the channel-wise frontend in inference_net.py. We keep both; the entrypoint
in scripts/train.py defaults to the channel-wise variant.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_critics: int,
        hidden: Sequence[int] = (256, 128, 64),
        dropout: float = 0.1,
        softplus_beta: float = 1.0,
    ) -> None:
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_critics)
        self.softplus_beta = float(softplus_beta)
        self.in_dim = int(in_dim)
        self.n_critics = int(n_critics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.body(x))
        return F.softplus(raw, beta=self.softplus_beta) + 0.05

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))
