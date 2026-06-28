"""
log_reader.py — reads HDF5 episode files written by data_collector_node.

Phase 8 implementation. Pure-Python; ROS-free; pulls only h5py + numpy.

Schema lives in docs/data_format.md. The writer is
vf_controller.data_collection.episode_writer.EpisodeWriter.

Two reader surfaces:
  EpisodeReader      — single-file, attribute-rich access
  MultiEpisodeIndex  — pre-scanned shape index over a directory of files,
                       used by the Dataset wrapper in data/dataset.py.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


def _decode_attr(v: Any) -> Any:
    """Decode h5py attribute values: bytes -> str, ndarray of bytes -> list[str]."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        if v.dtype.kind in ("S", "O"):
            return [
                (x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x))
                for x in v.tolist()
            ]
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


@dataclass
class EpisodeAttrs:
    scenario_id: str = "unknown"
    seed: int = 0
    controller_mode: str = "collect"
    weight_provider: str = "fixed"
    channels_config: str = "channels_v1"
    channel_names: List[str] = None  # type: ignore[assignment]
    channel_dims: List[int] = None  # type: ignore[assignment]
    critic_names: List[str] = None  # type: ignore[assignment]
    start_time_iso: str = ""
    end_time_iso: str = ""
    git_commit: str = ""
    success: Optional[bool] = None
    collision_count: int = 0
    time_to_goal_s: float = float("nan")
    path_length_m: float = float("nan")
    mean_clearance_m: float = float("nan")
    goal_reached_at_step: int = -1
    num_steps: int = 0


class EpisodeReader:
    """
    Reads one HDF5 episode file. Lazily memmaps datasets via h5py — the file
    handle stays open until the reader is closed (or used as a context manager).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._f = h5py.File(path, "r")
        self.attrs = self._read_attrs(self._f)

    # -------------------------------------------------------------- properties
    @property
    def features(self) -> np.ndarray:
        return self._f["features"][...]

    @property
    def critic_costs(self) -> np.ndarray:
        return self._f["critic_costs"][...]

    @property
    def critic_weights_applied(self) -> np.ndarray:
        return self._f["critic_weights_applied"][...]

    @property
    def selected_action(self) -> np.ndarray:
        return self._f["selected_action"][...]

    @property
    def robot_pose(self) -> np.ndarray:
        return self._f["robot_pose"][...]

    @property
    def goal(self) -> np.ndarray:
        return self._f["goal"][...]

    @property
    def dynamic_obstacles(self) -> Optional[np.ndarray]:
        if "dynamic_obstacles" in self._f:
            return self._f["dynamic_obstacles"][...]
        return None

    @property
    def oracle_weights(self) -> Optional[np.ndarray]:
        """Phase 9 augmentation. Present only when ``scripts/run_oracle.py``
        has written labels into this file."""
        if "oracle_weights" in self._f:
            return self._f["oracle_weights"][...]
        return None

    @property
    def has_oracle_weights(self) -> bool:
        return "oracle_weights" in self._f

    @property
    def num_steps(self) -> int:
        return int(self._f["features"].shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self._f["features"].shape[1])

    @property
    def critic_count(self) -> int:
        return int(self._f["critic_costs"].shape[1])

    @property
    def critic_names(self) -> List[str]:
        return list(self.attrs.critic_names or [])

    @property
    def scenario_id(self) -> str:
        return self.attrs.scenario_id

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _read_attrs(f: h5py.File) -> EpisodeAttrs:
        a = f.attrs
        out = EpisodeAttrs()
        if "scenario_id" in a:
            out.scenario_id = str(_decode_attr(a["scenario_id"]))
        if "seed" in a:
            out.seed = int(_decode_attr(a["seed"]))
        if "controller_mode" in a:
            out.controller_mode = str(_decode_attr(a["controller_mode"]))
        if "weight_provider" in a:
            out.weight_provider = str(_decode_attr(a["weight_provider"]))
        if "channels_config" in a:
            out.channels_config = str(_decode_attr(a["channels_config"]))
        if "channel_names" in a:
            out.channel_names = list(_decode_attr(a["channel_names"]) or [])
        else:
            out.channel_names = []
        if "channel_dims" in a:
            out.channel_dims = [int(x) for x in _decode_attr(a["channel_dims"]) or []]
        else:
            out.channel_dims = []
        if "critic_names" in a:
            out.critic_names = list(_decode_attr(a["critic_names"]) or [])
        else:
            out.critic_names = []
        if "start_time_iso" in a:
            out.start_time_iso = str(_decode_attr(a["start_time_iso"]))
        if "end_time_iso" in a:
            out.end_time_iso = str(_decode_attr(a["end_time_iso"]))
        if "git_commit" in a:
            out.git_commit = str(_decode_attr(a["git_commit"]))
        if "success" in a:
            out.success = bool(_decode_attr(a["success"]))
        if "collision_count" in a:
            out.collision_count = int(_decode_attr(a["collision_count"]))
        if "time_to_goal_s" in a:
            out.time_to_goal_s = float(_decode_attr(a["time_to_goal_s"]))
        if "path_length_m" in a:
            out.path_length_m = float(_decode_attr(a["path_length_m"]))
        if "mean_clearance_m" in a:
            out.mean_clearance_m = float(_decode_attr(a["mean_clearance_m"]))
        if "goal_reached_at_step" in a:
            out.goal_reached_at_step = int(_decode_attr(a["goal_reached_at_step"]))
        if "num_steps" in a:
            out.num_steps = int(_decode_attr(a["num_steps"]))
        return out

    # ------------------------------------------------------------------- close
    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self) -> "EpisodeReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# =============================================================================
# Multi-file index (used by dataset.py to build train/val splits per episode).
# =============================================================================

@dataclass
class EpisodeIndexEntry:
    path: str
    num_steps: int
    feature_dim: int
    critic_count: int
    scenario_id: str
    seed: int
    success: Optional[bool]


class MultiEpisodeIndex:
    """Pre-scans a directory of .h5 files. Caches shape + a few attrs per file.

    Used to plan train/val episode splits without holding all data in memory.
    """

    def __init__(self, paths: Sequence[str]) -> None:
        self.entries: List[EpisodeIndexEntry] = []
        for p in paths:
            try:
                with EpisodeReader(p) as ep:
                    self.entries.append(EpisodeIndexEntry(
                        path=p,
                        num_steps=ep.num_steps,
                        feature_dim=ep.feature_dim,
                        critic_count=ep.critic_count,
                        scenario_id=ep.scenario_id,
                        seed=ep.attrs.seed,
                        success=ep.attrs.success,
                    ))
            except Exception as e:  # pragma: no cover — corrupt-file tolerant
                print(f"[MultiEpisodeIndex] skipping {p}: {e}")

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def total_rows(self) -> int:
        return sum(e.num_steps for e in self.entries)

    def feature_dim(self) -> int:
        if not self.entries:
            return 0
        return self.entries[0].feature_dim

    def critic_count(self) -> int:
        if not self.entries:
            return 0
        return self.entries[0].critic_count

    @classmethod
    def from_directory(
        cls, dirpath: str, pattern: str = "*.h5",
    ) -> "MultiEpisodeIndex":
        """Discover .h5 episodes under a directory.

        Strategy:
          1. Try ``<dirpath>/<pattern>`` first — preserves O(N) behaviour
             when called on a single leaf folder.
          2. If that finds nothing, recurse with ``rglob`` so a top-level
             ``vf_data/vf_data_training/`` returns every episode file regardless of
             nesting depth.  Works with both ``run_*.h5`` (new naming) and
             legacy ``ep_*.h5`` filenames since pattern defaults to ``*.h5``.
        """
        flat = sorted(glob.glob(os.path.join(dirpath, pattern)))
        if flat:
            return cls(flat)
        # Recursive walk: handles vf_data/vf_data_training/{mode}/{map}/{goal}/{...}/*.h5
        # and legacy vf_data/vf_data_training/{manual,batch}/<session>/*.h5.
        nested = sorted(
            glob.glob(os.path.join(dirpath, "**", pattern), recursive=True)
        )
        return cls(nested)
