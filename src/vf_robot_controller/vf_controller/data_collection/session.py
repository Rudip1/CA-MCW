"""
session.py — session folder, session.json, manifest.csv helpers.

New layout (vf_data):
    A *leaf* is the planner/controller folder for one goal. It lives at:
      vf_data/vf_data_training/{mode}/{map}/{goal_folder}/{Planner}/{controller}/
    Each leaf owns:
      session.json   written when the first episode opens for that leaf.
      manifest.csv   appended once per closed episode.
      run_*.h5       per-run HDF5 episode files.

    Use ``vfdata_leaf_for()`` to build the leaf path, then pass it as
    ``leaf_dir`` to ``write_session_json()`` and ``append_manifest_row()``.

Legacy layout (vf_data/vf_data_training — deprecated, kept for backwards compat):
    ``session_dir_for()`` builds a per-session folder under
    ``vf_data/vf_data_training/{manual,batch}/``. New code should use the vf_data
    helpers above.

This module is ROS-free so it can be unit-tested without rclpy.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


SESSION_KIND_MANUAL = "manual"
SESSION_KIND_BATCH = "batch"


def _stamp(now: Optional[datetime.datetime] = None) -> str:
    return (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")


# ── vf_data leaf path helper ──────────────────────────────────────────────────

def vfdata_leaf_for(
    training_root: str | Path,
    mode: str,
    map_name: str,
    goal_folder: str,
    planner: str,
    controller: str,
) -> Path:
    """Build the vf_data leaf directory path for one goal/planner/controller.

    Layout:
      {training_root}/{mode}/{map_name}/{goal_folder}/{planner}/{controller}/

    Does NOT create the directory on disk. Call ``Path.mkdir(parents=True,
    exist_ok=True)`` on the result before writing to it.

    Args:
        training_root: typically ``TRAINING_ROOT`` from constants.
        mode:           ``"manual"`` or ``"batch"``.
        map_name:       snake_case world name.
        goal_folder:    output of ``goal_folder_name(gx, gy, gt)``.
        planner:        PascalCase Nav2 planner key (e.g. ``"NavFn"``).
        controller:     lowercase controller name (e.g. ``"vf_fixedwt"``).
    """
    return (
        Path(training_root) / mode / map_name / goal_folder / planner / controller
    )


def run_filename(now: Optional[datetime.datetime] = None) -> str:
    """``run_YYYYMMDD_HHMMSS.h5`` — canonical collected episode filename."""
    return (now or datetime.datetime.now()).strftime("run_%Y%m%d_%H%M%S.h5")


# ── Legacy session path helper (deprecated) ───────────────────────────────────

def session_dir_for(
    episodes_root: str | Path,
    session_kind: str,
    map_name: str,
    suffix: str = "",
    now: Optional[datetime.datetime] = None,
) -> Path:
    """Build a session directory path. Does not create it on disk.

    Layout:
      <episodes_root>/<session_kind>/<map_name>_<YYYYMMDD_HHMMSS>[_suffix]
    """
    if session_kind not in (SESSION_KIND_MANUAL, SESSION_KIND_BATCH):
        raise ValueError(
            "session_kind must be 'manual' or 'batch', "
            "got %r" % session_kind
        )
    name = "%s_%s" % (map_name or "unknown", _stamp(now))
    if suffix:
        name = "%s_%s" % (name, suffix)
    return Path(episodes_root) / session_kind / name


@dataclass
class SessionInfo:
    """Once-per-session metadata. Written to session.json."""

    session_kind: str
    map_name: str
    started_at_iso: str
    controller_mode: str
    weight_provider: str
    channels_config: str
    scenario_id: str
    seed: int
    episode_timeout_s: float
    write_period_s: float
    goal_radius_m: float
    git_commit: str
    extra: Dict[str, object] = field(default_factory=dict)


def write_session_json(session_dir: str | Path, info: SessionInfo) -> Path:
    """Write session.json idempotently (only if missing)."""
    p = Path(session_dir) / "session.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return p
    data = asdict(info)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return p


# Manifest schema. Order is the column order in manifest.csv.
MANIFEST_FIELDS: List[str] = [
    "episode_index",
    "h5_filename",
    "scenario_id",
    "seed",
    "controller_mode",
    "channels_config",
    "start_x", "start_y", "start_yaw",
    "goal_x", "goal_y", "goal_yaw",
    "success",
    "close_reason",
    "n_steps",
    "duration_s",
    "path_length_m",
    "size_bytes",
    "started_at_iso",
    "ended_at_iso",
]


def append_manifest_row(session_dir: str | Path, row: Dict[str, object]) -> Path:
    """Append one manifest row. Creates the file with header if missing."""
    p = Path(session_dir) / "manifest.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    is_new = (not p.exists()) or p.stat().st_size == 0
    # Coerce booleans/floats to safe scalars for csv
    safe = {k: ("" if v is None else v) for k, v in row.items()}
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore",
        )
        if is_new:
            w.writeheader()
        w.writerow(safe)
    return p


def count_episodes(session_dir: str | Path) -> int:
    p = Path(session_dir)
    if not p.is_dir():
        return 0
    return len(sorted(p.glob("ep_*.h5")))


def episode_filename(index: int, scenario_id: str,
                     now: Optional[datetime.datetime] = None) -> str:
    """ep_<NNN>_<scenario>_<YYYYMMDD_HHMMSS>.h5"""
    return "ep_%03d_%s_%s.h5" % (
        index, scenario_id or "scenario", _stamp(now),
    )


def latest_session(episodes_root: str | Path,
                   session_kind: Optional[str] = None) -> Optional[Path]:
    """Return the most-recently-modified session folder, or None."""
    root = Path(episodes_root)
    if not root.is_dir():
        return None
    kinds = ([session_kind] if session_kind
             else [SESSION_KIND_MANUAL, SESSION_KIND_BATCH])
    candidates: List[Path] = []
    for kind in kinds:
        d = root / kind
        if not d.is_dir():
            continue
        for s in d.iterdir():
            if s.is_dir():
                candidates.append(s)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def discover_episodes(root: str | Path,
                      pattern: str = "*.h5") -> List[Path]:
    """Recursively discover .h5 files under a root.

    Honours the new ``vf_data/vf_data_training/{manual,batch}/<session>/ep_*.h5``
    layout while still finding any flat ``.h5`` placed directly under the
    root (e.g. legacy data).
    """
    root_p = Path(root)
    if not root_p.exists():
        return []
    if root_p.is_file():
        return [root_p]
    return sorted(root_p.rglob(pattern))
