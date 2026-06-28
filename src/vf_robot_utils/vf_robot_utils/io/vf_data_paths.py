"""
vf_data_paths.py — canonical path construction for vf_data/.

All training HDF5 episodes live under vf_data/vf_data_training/ and all
evaluation artifacts live under vf_data/vf_data_evaluation/. Both subtrees
share the same folder hierarchy down to the controller level, so paths
are interchangeable by swapping the top-level word.

Import these functions whenever building a path inside vf_data/; never
construct the goal_folder name with raw f-strings (the 'n'-prefix rule for
negatives is easy to get wrong).

See vf_data/README_vf_data.md and vf_data/plan.md for the full spec.
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Optional


# Lazy import so this module can be unit-tested without the full ROS install.
def _training_root() -> Path:
    from vf_robot_utils.constants import TRAINING_ROOT
    return TRAINING_ROOT


def _evaluation_root() -> Path:
    from vf_robot_utils.constants import EVALUATION_ROOT
    return EVALUATION_ROOT


# ── Goal folder encoding ───────────────────────────────────────────────────────

def goal_folder_name(gx: float, gy: float, gt: float) -> str:
    """Encode a navigation goal pose as a filesystem-safe folder name.

    Format: ``goal_x{X}_y{Y}_t{T}`` where negative values are prefixed with
    ``n`` instead of ``-`` to avoid shell flag-parsing issues.

    Examples:
        goal_folder_name(3.54, 1.33, 2.55)   -> "goal_x3.54_y1.33_t2.55"
        goal_folder_name(-3.54, -1.33, 0.78) -> "goal_xn3.54_yn1.33_t0.78"
    """
    def _enc(v: float) -> str:
        s = f"{abs(v):.2f}"
        return f"n{s}" if v < 0.0 else s
    return f"goal_x{_enc(gx)}_y{_enc(gy)}_t{_enc(gt)}"


_GOAL_PATTERN = re.compile(
    r"^goal_x(n?)([0-9]+(?:\.[0-9]+)?)"
    r"_y(n?)([0-9]+(?:\.[0-9]+)?)"
    r"_t(n?)([0-9]+(?:\.[0-9]+)?)$"
)


def parse_goal_folder(name: str) -> tuple[float, float, float]:
    """Inverse of goal_folder_name. Raises ValueError on bad input."""
    m = _GOAL_PATTERN.fullmatch(name)
    if not m:
        raise ValueError(
            f"Cannot parse goal folder name: {name!r}. "
            "Expected goal_x[n]X_y[n]Y_t[n]T"
        )
    def _dec(neg: str, val: str) -> float:
        return -float(val) if neg else float(val)
    return (
        _dec(m.group(1), m.group(2)),
        _dec(m.group(3), m.group(4)),
        _dec(m.group(5), m.group(6)),
    )


# ── Run timestamp ──────────────────────────────────────────────────────────────

def run_ts(now: Optional[datetime.datetime] = None) -> str:
    """Return ``run_YYYYMMDD_HHMMSS`` — canonical prefix for training H5
    filenames and evaluation run directories."""
    return (now or datetime.datetime.now()).strftime("run_%Y%m%d_%H%M%S")


# ── Leaf path builders ─────────────────────────────────────────────────────────

def training_leaf(
    mode: str,
    map_name: str,
    goal_folder: str,
    planner: str,
    controller: str,
    root: Optional[Path] = None,
) -> Path:
    """Canonical path for one planner/controller leaf under vf_data_training/.

    Layout:
        {root}/vf_data_training/{mode}/{map_name}/{goal_folder}/{Planner}/{controller}/

    Args:
        mode:        "manual" or "batch".
        map_name:    snake_case world name (e.g. "hospital_map").
        goal_folder: output of goal_folder_name() (e.g. "goal_x3.54_y1.33_t2.55").
        planner:     PascalCase Nav2 planner key (e.g. "NavFn").
        controller:  lowercase controller name (e.g. "vf_collect").
        root:        override TRAINING_ROOT (useful in tests).
    """
    r = Path(root) if root is not None else _training_root()
    return r / mode / map_name / goal_folder / planner / controller


def evaluation_leaf(
    mode: str,
    map_name: str,
    goal_folder: str,
    planner: str,
    controller: str,
    root: Optional[Path] = None,
) -> Path:
    """Canonical path for one planner/controller leaf under vf_data_evaluation/.

    Layout:
        {root}/vf_data_evaluation/{mode}/{map_name}/{goal_folder}/{Planner}/{controller}/

    Each run within this leaf is a subdirectory: ``run_YYYYMMDD_HHMMSS/``
    containing episode.h5, metrics.json, per_episode.csv, and figures/.
    Cross-run summary.csv lives directly in this leaf directory.
    """
    r = Path(root) if root is not None else _evaluation_root()
    return r / mode / map_name / goal_folder / planner / controller


def goal_analysis_dir(
    mode: str,
    map_name: str,
    goal_folder: str,
    root: Optional[Path] = None,
) -> Path:
    """Path to the _analysis/ directory for cross-controller comparison.

    Layout:
        {root}/vf_data_evaluation/{mode}/{map_name}/{goal_folder}/_analysis/

    This directory is a sibling to all planner folders under goal_folder
    and contains comparison_table.csv, report.md, figures/, and logs/.
    """
    r = Path(root) if root is not None else _evaluation_root()
    return r / mode / map_name / goal_folder / "_analysis"


def evaluation_cache_dir(
    mode: str,
    map_name: str,
    root: Optional[Path] = None,
) -> Path:
    """Path to the per-map cache directory for global_paths JSON files.

    Layout:
        {root}/vf_data_evaluation/{mode}/{map_name}/_cache/
    """
    r = Path(root) if root is not None else _evaluation_root()
    return r / mode / map_name / "_cache"


def evaluation_aggregate_dir(
    mode: str,
    map_name: str,
    csv_stem: str,
    root: Optional[Path] = None,
) -> Path:
    """Path to the cross-goal aggregate directory for one CSV-runner sweep.

    Layout:
        {root}/vf_data_evaluation/{mode}/{map_name}/_aggregate/{csv_stem}/

    Contains: results.csv (flat table across all goals/planners/controllers),
    figures/ (headline bars, scenario heatmap, etc.).
    """
    r = Path(root) if root is not None else _evaluation_root()
    return r / mode / map_name / "_aggregate" / csv_stem
