"""
dataset.py — PyTorch Dataset wrapping HDF5 logs with per-channel normalization.

Phase 8.

Two dataset classes share the same multi-file backend:

  EpisodeFeatureDataset  — yields (features, target) pairs.
                            target is either:
                              * (K,) probability simplex over critic weights
                                (mode == "inference"), or
                              * (2,) [vx, wz] (mode == "imitation").

In-memory layout: each .h5 episode is read into a CPU-resident float32 array
once at __init__; batches are produced by indexing. The dataset is bounded by
the total number of cycles across the corpus (≤ 1M rows for the Phase 8 50k-
sample target) so RAM is fine.

Why episode-level splits and not per-frame:
  Frames inside one episode are correlated (same map, same robot, etc.) — a
  frame-level random split leaks information across train/val and over-states
  validation accuracy. Episode-level splits keep correlated frames together.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover — torch absent on cluster sometimes
    Dataset = object  # type: ignore[assignment, misc]
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

from .channel_sets import ChannelSlice, resolve_slice
from .log_reader import EpisodeReader, MultiEpisodeIndex
from .normalization import FeatureNormalizationStats
from .oracle_weights import (derive_imitation_targets, derive_inference_targets,
                              derive_raw_critic_targets)


@dataclass
class DatasetSplits:
    train_paths: List[str]
    val_paths: List[str]


def _apply_zero_channels(
    feats: np.ndarray,
    channel_names: Sequence[str],
    channel_dims: Sequence[int],
    zero_channels: Sequence[str],
) -> np.ndarray:
    """Zero the column ranges in `feats` that correspond to `zero_channels`.

    Loud: any name in `zero_channels` that isn't found raises — silent typos
    would leave critic_history alive and the model would still collapse at
    deployment, which is the very failure mode this is meant to prevent.
    """
    if not zero_channels:
        return feats
    if len(channel_names) != len(channel_dims):
        raise ValueError(
            f"channel_names/channel_dims length mismatch: "
            f"{len(channel_names)} vs {len(channel_dims)}"
        )
    unknown = set(zero_channels) - set(channel_names)
    if unknown:
        raise ValueError(
            f"zero_channels {sorted(unknown)} not found in active layout "
            f"{list(channel_names)}"
        )
    feats = feats.copy()
    offset = 0
    for name, dim in zip(channel_names, channel_dims):
        if name in zero_channels:
            feats[:, offset:offset + dim] = 0.0
        offset += int(dim)
    return feats


def split_by_episode(
    paths: Sequence[str],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> DatasetSplits:
    """Episode-level train/val split. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    pool = list(paths)
    rng.shuffle(pool)
    if not pool:
        return DatasetSplits([], [])
    n_val = max(1, int(round(len(pool) * val_fraction)))
    return DatasetSplits(train_paths=pool[n_val:], val_paths=pool[:n_val])


class EpisodeFeatureDataset(Dataset):  # type: ignore[misc]
    """
    Reads each .h5 file into memory once; concatenates across episodes.

    `mode` selects the target:
        "inference" -> (K,) probability simplex over critics
        "imitation" -> (2,) [vx, wz]
    """

    def __init__(
        self,
        paths: Sequence[str],
        mode: str = "inference",
        norm: Optional[FeatureNormalizationStats] = None,
        raw_critics_temperature: float = 1.0,
        channels: Optional[str] = None,
        zero_channels: Optional[Sequence[str]] = None,
    ) -> None:
        if mode not in ("inference", "imitation", "raw_critics"):
            raise ValueError(
                f"mode must be 'inference', 'imitation', or 'raw_critics', got {mode!r}"
            )
        self._raw_critics_temperature = raw_critics_temperature
        self.mode = mode
        self.norm = norm
        self.channels = channels
        # Channel names whose feature dims will be forced to zero at load time.
        # Use case: the imitation policy must not learn to depend on channels
        # that are only populated when MPPI runs (e.g. critic_history), because
        # at deployment in PASSIVE mode that channel is identically zero — the
        # model would otherwise be permanently out-of-distribution and collapse
        # to a near-zero fixed point.
        self.zero_channels: List[str] = list(zero_channels or [])

        feats_list: List[np.ndarray] = []
        targets_list: List[np.ndarray] = []
        masks_list: List[np.ndarray] = []
        critic_names: List[str] = []
        channel_names: List[str] = []
        channel_dims: List[int] = []
        slice_: Optional[ChannelSlice] = None
        # Phase 9: count how many episodes contributed oracle vs hindsight
        # labels so the trainer can log it.
        n_oracle = 0
        n_hindsight = 0

        for p in paths:
            with EpisodeReader(p) as ep:
                feats = np.asarray(ep.features, dtype=np.float32)
                if feats.size == 0:
                    continue
                if channels is not None:
                    if slice_ is None:
                        slice_ = resolve_slice(
                            channels,
                            ep.attrs.channel_names or [],
                            ep.attrs.channel_dims or [],
                        )
                    feats = feats[:, : slice_.n_feat]
                # Zero out user-requested channels in the sliced feature view.
                if self.zero_channels:
                    eff_names = (list(slice_.channel_names) if slice_ is not None
                                 else list(ep.attrs.channel_names or []))
                    eff_dims  = (list(slice_.channel_dims)  if slice_ is not None
                                 else list(ep.attrs.channel_dims  or []))
                    feats = _apply_zero_channels(
                        feats, eff_names, eff_dims, self.zero_channels)
                if mode == "inference":
                    oracle = ep.oracle_weights
                    tgt, mask = derive_inference_targets(
                        ep.critic_costs,
                        ep.critic_weights_applied,
                        oracle_weights=oracle,
                    )
                    if oracle is not None:
                        n_oracle += 1
                    else:
                        n_hindsight += 1
                elif mode == "raw_critics":
                    tgt, mask = derive_raw_critic_targets(
                        ep.critic_costs,
                        temperature=self._raw_critics_temperature,
                    )
                else:
                    tgt = derive_imitation_targets(ep.selected_action)
                    mask = np.ones(tgt.shape[0], dtype=bool)
                # Truncate to common length defensively (writer can leave
                # ragged tails on Ctrl-C in rare cases).
                T = min(feats.shape[0], tgt.shape[0])
                feats_list.append(feats[:T])
                targets_list.append(tgt[:T])
                masks_list.append(mask[:T])
                if not critic_names and ep.critic_names:
                    critic_names = list(ep.critic_names)
                if not channel_names and ep.attrs.channel_names:
                    channel_names = (
                        list(slice_.channel_names) if slice_ is not None
                        else list(ep.attrs.channel_names)
                    )
                if not channel_dims and ep.attrs.channel_dims:
                    channel_dims = (
                        list(slice_.channel_dims) if slice_ is not None
                        else list(ep.attrs.channel_dims)
                    )

        self.n_oracle_episodes = n_oracle
        self.n_hindsight_episodes = n_hindsight

        if feats_list:
            self.features = np.concatenate(feats_list, axis=0)
            self.targets = np.concatenate(targets_list, axis=0)
            self.masks = np.concatenate(masks_list, axis=0)
        else:
            self.features = np.zeros((0, 1), dtype=np.float32)
            self.targets = np.zeros((0, 2 if mode == "imitation" else 1), dtype=np.float32)
            self.masks = np.zeros((0,), dtype=bool)

        self.critic_names = critic_names
        self.channel_names = channel_names
        self.channel_dims = channel_dims

        # NaN-mask features (data writer NaN-pads short rows). We replace NaN
        # with zero AFTER applying normalization so the zero-fill is on the
        # normalized scale.
        self._nan_rows = np.any(np.isnan(self.features), axis=1)

    # ----------------------------------------------------------------- pytorch
    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.targets[idx]
        m = bool(self.masks[idx]) if self.masks.size else True
        if self.norm is not None:
            x = self.norm.apply(x)
        # Zero-fill NaNs after normalization so they don't bias the network.
        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
        if _HAS_TORCH:
            return (
                torch.from_numpy(np.asarray(x, dtype=np.float32)),
                torch.from_numpy(np.asarray(y, dtype=np.float32)),
                torch.tensor(m, dtype=torch.bool),
            )
        return x, y, m

    # --------------------------------------------------------------- helpers
    @property
    def in_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.targets.shape[1]) if self.targets.ndim == 2 else 1


def build_datasets(
    h5_dir: str,
    mode: str,
    val_fraction: float = 0.1,
    seed: int = 0,
    norm: Optional[FeatureNormalizationStats] = None,
    raw_critics_temperature: float = 1.0,
) -> Tuple[EpisodeFeatureDataset, EpisodeFeatureDataset, DatasetSplits]:
    """One-call helper: scan dir, split, build train/val datasets."""
    idx = MultiEpisodeIndex.from_directory(h5_dir)
    splits = split_by_episode([e.path for e in idx.entries], val_fraction, seed)
    kw = dict(mode=mode, norm=norm, raw_critics_temperature=raw_critics_temperature)
    train = EpisodeFeatureDataset(splits.train_paths, **kw)
    val = EpisodeFeatureDataset(splits.val_paths, **kw)
    return train, val, splits
