"""
channel_sets.py — map a channel-set name (v1/v2/v3) to a feature-vector slice.

Channels are nested prefixes: v1 is the first 6 channels of v3, v2 is the first
7. PACKAGE.txt:328-331 is the canonical reference; this module just enforces
the same contract at training time so the same v3 HDF5 corpus can train v1, v2,
and v3 models.

Resolution is by channel *name* and not positional, so we fail loudly if the
HDF5 doesn't carry the expected channels in the expected order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


CHANNEL_SETS: dict = {
    "v1": [
        "robot_state", "context", "path_geometry", "gcf_rosette",
        "critic_history", "obstacle_dynamics",
    ],
    "v2": [
        "robot_state", "context", "path_geometry", "gcf_rosette",
        "critic_history", "obstacle_dynamics", "reynolds",
    ],
    "v3": [
        "robot_state", "context", "path_geometry", "gcf_rosette",
        "critic_history", "obstacle_dynamics", "reynolds", "slam_persistent",
    ],
}


@dataclass
class ChannelSlice:
    """Result of resolving a channels choice against an HDF5's channel layout."""
    n_feat: int                # number of feature dims to keep (prefix length)
    channel_names: List[str]   # sliced channel names (in order)
    channel_dims: List[int]    # sliced channel dims (in order)


def resolve_slice(
    channels: str,
    h5_channel_names: Sequence[str],
    h5_channel_dims: Sequence[int],
) -> ChannelSlice:
    """Compute prefix length so feats[:, :n_feat] gives the requested channel set.

    Raises ValueError if the HDF5 channel order doesn't match the expected v1/v2/v3
    prefix — refusing to silently mis-slice features.
    """
    if channels not in CHANNEL_SETS:
        raise ValueError(
            f"unknown channels {channels!r}; expected one of {sorted(CHANNEL_SETS)}"
        )
    expected = CHANNEL_SETS[channels]
    h5_names = list(h5_channel_names)
    h5_dims = list(h5_channel_dims)

    if len(h5_names) != len(h5_dims):
        raise ValueError(
            f"HDF5 channel_names/channel_dims length mismatch "
            f"({len(h5_names)} vs {len(h5_dims)})"
        )
    if len(h5_names) < len(expected):
        raise ValueError(
            f"HDF5 has only {len(h5_names)} channels {h5_names!r}, but "
            f"{channels!r} needs at least {len(expected)}: {expected!r}"
        )
    h5_prefix = h5_names[: len(expected)]
    if h5_prefix != expected:
        raise ValueError(
            f"HDF5 channel prefix {h5_prefix!r} does not match expected "
            f"{channels!r} prefix {expected!r}; refusing to slice"
        )

    kept_dims = h5_dims[: len(expected)]
    return ChannelSlice(
        n_feat=int(sum(kept_dims)),
        channel_names=expected,
        channel_dims=[int(x) for x in kept_dims],
    )
