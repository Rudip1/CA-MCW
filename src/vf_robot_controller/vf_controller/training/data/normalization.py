"""
normalization.py — compute & apply per-channel mean/std statistics.

Phase 8.

Stats are computed once on the training-split episodes via Welford streaming
(no full-dataset memory footprint) and persisted to JSON. Both INFERENCE and
IMITATION reuse the same file. The C++ inference path (OnnxWeightProvider)
applies the same normalization before invoking the model.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from .channel_sets import resolve_slice


@dataclass
class FeatureNormalizationStats:
    """Per-feature mean/std plus the channel boundary metadata."""
    mean: np.ndarray  # shape (D,)
    std: np.ndarray   # shape (D,)  (clamped >= eps)
    channel_names: list
    channel_dims: list

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "mean": self.mean.astype(float).tolist(),
                "std": self.std.astype(float).tolist(),
                "channel_names": list(self.channel_names),
                "channel_dims": [int(x) for x in self.channel_dims],
            }, f)

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizationStats":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            channel_names=list(data.get("channel_names", [])),
            channel_dims=[int(x) for x in data.get("channel_dims", [])],
        )


class WelfordStreamingStats:
    """Numerically-stable online mean/var (Welford 1962) over D dims."""

    def __init__(self, dim: int) -> None:
        self.n = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._m2 = np.zeros(dim, dtype=np.float64)
        self.dim = dim

    def update_block(self, block: np.ndarray) -> None:
        """Update with a (B, D) block. NaN rows are skipped silently."""
        if block.size == 0:
            return
        block = np.asarray(block, dtype=np.float64).reshape(-1, self.dim)
        valid = ~np.any(np.isnan(block), axis=1)
        if not np.any(valid):
            return
        block = block[valid]
        # Vectorised Welford using the parallel-aggregation form:
        n_b = block.shape[0]
        mean_b = block.mean(axis=0)
        # Variance in block (population)
        m2_b = ((block - mean_b) ** 2).sum(axis=0)
        n = self.n + n_b
        delta = mean_b - self._mean
        self._mean = self._mean + delta * (n_b / n)
        self._m2 = self._m2 + m2_b + (delta ** 2) * (self.n * n_b / n)
        self.n = n

    @property
    def mean(self) -> np.ndarray:
        return self._mean.astype(np.float32)

    def std(self, eps: float = 1e-3) -> np.ndarray:
        if self.n < 2:
            return np.full(self.dim, 1.0, dtype=np.float32)
        var = self._m2 / max(1, self.n - 1)
        return np.sqrt(np.maximum(var, eps * eps)).astype(np.float32)


def compute_stats_from_episodes(
    episodes: Iterable["object"],  # Iterable[EpisodeReader]
    channel_names: Optional[Sequence[str]] = None,
    channel_dims: Optional[Sequence[int]] = None,
    eps: float = 1e-3,
    channels: Optional[str] = None,
    zero_channels: Optional[Sequence[str]] = None,
) -> FeatureNormalizationStats:
    """Stream rows from each episode and accumulate Welford stats.

    If ``zero_channels`` is given, those channel ranges are zeroed in each
    block before being fed to Welford. The std-eps clamp prevents zero
    division at inference time.
    """
    # Local import to avoid a circular dependency at module load.
    from .dataset import _apply_zero_channels

    welford: Optional[WelfordStreamingStats] = None
    cn: list = list(channel_names) if channel_names else []
    cd: list = list(channel_dims) if channel_dims else []
    n_feat: Optional[int] = None
    zc: list = list(zero_channels or [])

    for ep in episodes:
        feats = ep.features
        if feats.size == 0:
            continue
        if channels is not None and n_feat is None:
            sl = resolve_slice(
                channels,
                ep.attrs.channel_names or [],
                ep.attrs.channel_dims or [],
            )
            n_feat = sl.n_feat
            cn = list(sl.channel_names)
            cd = list(sl.channel_dims)
        if n_feat is not None:
            feats = feats[:, :n_feat]
        if welford is None:
            welford = WelfordStreamingStats(feats.shape[1])
            if not cn and ep.attrs.channel_names:
                cn = list(ep.attrs.channel_names)
            if not cd and ep.attrs.channel_dims:
                cd = list(ep.attrs.channel_dims)
        if zc:
            feats = _apply_zero_channels(feats, cn, cd, zc)
        welford.update_block(feats)

    if welford is None:
        # Fallback for pathological empty corpus: emit unit stats.
        return FeatureNormalizationStats(
            mean=np.zeros(1, dtype=np.float32),
            std=np.ones(1, dtype=np.float32),
            channel_names=cn, channel_dims=cd,
        )

    return FeatureNormalizationStats(
        mean=welford.mean,
        std=welford.std(eps=eps),
        channel_names=cn,
        channel_dims=cd,
    )
