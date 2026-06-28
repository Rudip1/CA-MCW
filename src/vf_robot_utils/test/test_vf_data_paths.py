"""Unit tests for vf_data_paths.py (Phase R1).

Tests the canonical path construction functions that encode the
vf_data/vf_data_training/ and vf_data/vf_data_evaluation/ hierarchy.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from vf_robot_utils.io.vf_data_paths import (
    training_leaf,
    evaluation_leaf,
    evaluation_aggregate_dir,
    evaluation_cache_dir,
    goal_analysis_dir,
    goal_folder_name,
    parse_goal_folder,
    run_ts,
)


# ── goal_folder_name ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("gx,gy,gt,expected", [
    (3.54,  1.33,  2.55,  "goal_x3.54_y1.33_t2.55"),
    (-3.54, -1.33, 0.78,  "goal_xn3.54_yn1.33_t0.78"),
    (7.05,  -8.70, 1.57,  "goal_x7.05_yn8.70_t1.57"),
    (8.58,   7.20, -0.50, "goal_x8.58_y7.20_tn0.50"),
    (0.0,    0.0,  0.0,   "goal_x0.00_y0.00_t0.00"),
])
def test_goal_folder_name(gx, gy, gt, expected):
    assert goal_folder_name(gx, gy, gt) == expected


def test_goal_folder_name_no_minus_sign():
    name = goal_folder_name(-1.0, -2.5, -3.14)
    assert "-" not in name


# ── parse_goal_folder ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("gx,gy,gt", [
    (3.54, 1.33, 2.55),
    (-3.54, -1.33, 0.78),
    (0.0, 0.0, 0.0),
    (7.05, -8.70, 1.57),
])
def test_parse_goal_folder_round_trip(gx, gy, gt):
    name = goal_folder_name(gx, gy, gt)
    pgx, pgy, pgt = parse_goal_folder(name)
    assert abs(pgx - gx) < 0.001
    assert abs(pgy - gy) < 0.001
    assert abs(pgt - gt) < 0.001


def test_parse_goal_folder_bad_input_raises():
    with pytest.raises(ValueError):
        parse_goal_folder("goal_x-3.54_y1.33_t2.55")   # minus sign is wrong


def test_parse_goal_folder_bad_prefix_raises():
    with pytest.raises(ValueError):
        parse_goal_folder("Goal_x3.54_y1.33_t2.55")


# ── run_ts ────────────────────────────────────────────────────────────────────

def test_run_ts_format():
    ts = run_ts()
    assert ts.startswith("run_")
    assert len(ts) == len("run_YYYYMMDD_HHMMSS")


def test_run_ts_uses_provided_datetime():
    now = datetime.datetime(2026, 5, 4, 16, 14, 13)
    assert run_ts(now) == "run_20260504_161413"


# ── training_leaf ────────────────────────────────────────────────────────────

def test_training_leaf_structure(tmp_path):
    gf = goal_folder_name(3.54, 1.33, 2.55)
    leaf = training_leaf(
        "batch", "hospital_map", gf, "NavFn", "vf_collect",
        root=tmp_path,
    )
    assert leaf == tmp_path / "batch" / "hospital_map" / gf / "NavFn" / "vf_collect"


def test_training_leaf_mode_manual(tmp_path):
    gf = goal_folder_name(1.0, 2.0, 0.0)
    leaf = training_leaf("manual", "house_my1_map", gf, "ThetaStar", "mppi", root=tmp_path)
    assert "manual" in leaf.parts
    assert "house_my1_map" in leaf.parts


# ── evaluation_leaf ────────────────────────────────────────────────────────────

def test_evaluation_leaf_structure(tmp_path):
    gf = goal_folder_name(-3.54, -1.33, 0.78)
    leaf = evaluation_leaf(
        "batch", "hospital_map", gf, "SmacPlannerHybridA", "mppi",
        root=tmp_path,
    )
    assert leaf == (
        tmp_path / "batch" / "hospital_map" / gf
        / "SmacPlannerHybridA" / "mppi"
    )


def test_collected_and_evaluated_same_relative_structure(tmp_path):
    gf = goal_folder_name(1.0, 2.0, 0.5)
    train_root = tmp_path / "vf_data_training"
    eval_root = tmp_path / "vf_data_evaluation"
    c = training_leaf("batch", "house_my1_map", gf, "NavFn", "vf_collect", root=train_root)
    e = evaluation_leaf("batch", "house_my1_map", gf, "NavFn", "vf_collect", root=eval_root)
    # Same relative path below root.
    assert c.relative_to(train_root) == e.relative_to(eval_root)


# ── goal_analysis_dir ─────────────────────────────────────────────────────────

def test_goal_analysis_dir_uses_underscore_prefix(tmp_path):
    gf = goal_folder_name(3.0, 4.0, 1.5)
    d = goal_analysis_dir("batch", "house_my1_map", gf, root=tmp_path)
    assert d.name == "_analysis"
    assert d.parent.name == gf


# ── evaluation_cache_dir / evaluation_aggregate_dir ─────────────────────────────

def test_evaluation_cache_dir(tmp_path):
    d = evaluation_cache_dir("batch", "house_my1_map", root=tmp_path)
    assert d.name == "_cache"
    assert d.parent.name == "house_my1_map"


def test_evaluation_aggregate_dir(tmp_path):
    d = evaluation_aggregate_dir("batch", "house_my1_map", "evaluate_goals", root=tmp_path)
    assert d.name == "evaluate_goals"
    assert d.parent.name == "_aggregate"
    assert d.parent.parent.name == "house_my1_map"
