"""Unit tests for vf_controller.data_collection.session."""
from __future__ import annotations

import csv
import datetime
import json
import os
import time

import pytest

from vf_controller.data_collection.session import (
    SESSION_KIND_BATCH,
    SESSION_KIND_MANUAL,
    SessionInfo,
    append_manifest_row,
    count_episodes,
    discover_episodes,
    episode_filename,
    latest_session,
    session_dir_for,
    write_session_json,
)


# --------------------------------------------------------------- session_dir_for
def test_session_dir_for_layout(tmp_path):
    fixed = datetime.datetime(2026, 5, 3, 14, 25, 33)
    p = session_dir_for(
        episodes_root=tmp_path,
        session_kind=SESSION_KIND_MANUAL,
        map_name="house_my1_map",
        now=fixed,
    )
    assert p == tmp_path / "manual" / "house_my1_map_20260503_142533"


def test_session_dir_for_with_suffix(tmp_path):
    fixed = datetime.datetime(2026, 5, 3, 14, 25, 33)
    p = session_dir_for(
        episodes_root=tmp_path,
        session_kind=SESSION_KIND_BATCH,
        map_name="house_my1_map",
        suffix="collect_goals",
        now=fixed,
    )
    assert p == (
        tmp_path / "batch" / "house_my1_map_20260503_142533_collect_goals"
    )


def test_session_dir_for_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        session_dir_for(tmp_path, session_kind="other", map_name="m")


# --------------------------------------------------------------- session.json
def test_write_session_json_idempotent(tmp_path):
    info = SessionInfo(
        session_kind=SESSION_KIND_MANUAL,
        map_name="house_my1_map",
        started_at_iso="2026-05-03T14:25:33+00:00",
        controller_mode="collect",
        weight_provider="fixed",
        channels_config="channels_v1",
        scenario_id="manual_run",
        seed=0,
        episode_timeout_s=180.0,
        write_period_s=0.05,
        goal_radius_m=0.4,
        git_commit="deadbee",
    )
    p1 = write_session_json(tmp_path, info)
    assert p1.exists()
    body1 = p1.read_text()
    # Second call must not overwrite (idempotency).
    info2 = SessionInfo(**{**info.__dict__, "scenario_id": "different"})
    p2 = write_session_json(tmp_path, info2)
    assert p2 == p1
    assert p2.read_text() == body1
    parsed = json.loads(body1)
    assert parsed["session_kind"] == "manual"
    assert parsed["map_name"] == "house_my1_map"
    assert parsed["scenario_id"] == "manual_run"  # original, not overwritten


# ----------------------------------------------------------- manifest.csv
def test_append_manifest_row_writes_header_once(tmp_path):
    row = {
        "episode_index": 0,
        "h5_filename": "ep_000.h5",
        "scenario_id": "smoke",
        "seed": 0,
        "controller_mode": "collect",
        "channels_config": "channels_v1",
        "start_x": 1.0, "start_y": 2.0, "start_yaw": 0.0,
        "goal_x": 5.0, "goal_y": 2.0, "goal_yaw": 0.0,
        "success": True,
        "close_reason": "goal_reached",
        "n_steps": 1234,
        "duration_s": 61.7,
        "path_length_m": 4.2,
        "size_bytes": 1024 * 1024,
        "started_at_iso": "2026-05-03T14:30:00+00:00",
        "ended_at_iso":   "2026-05-03T14:31:01+00:00",
    }
    append_manifest_row(tmp_path, row)
    append_manifest_row(tmp_path, {**row, "episode_index": 1,
                                   "h5_filename": "ep_001.h5"})
    p = tmp_path / "manifest.csv"
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["episode_index"] == "0"
    assert rows[1]["h5_filename"] == "ep_001.h5"
    assert rows[0]["success"] == "True"
    assert int(rows[0]["size_bytes"]) == 1024 * 1024


def test_append_manifest_drops_unknown_columns(tmp_path):
    append_manifest_row(tmp_path, {
        "episode_index": 0, "h5_filename": "x.h5",
        "junk_field": "should_be_dropped",
        "n_steps": 10,
    })
    text = (tmp_path / "manifest.csv").read_text()
    assert "junk_field" not in text


# ------------------------------------------------------------- episode_filename
def test_episode_filename_pattern():
    fixed = datetime.datetime(2026, 5, 3, 14, 30, 12)
    name = episode_filename(7, "scenario_a", now=fixed)
    assert name == "ep_007_scenario_a_20260503_143012.h5"


def test_episode_filename_index_padding():
    fixed = datetime.datetime(2026, 5, 3, 14, 30, 12)
    assert episode_filename(0, "x", now=fixed).startswith("ep_000_")
    assert episode_filename(123, "x", now=fixed).startswith("ep_123_")


# ---------------------------------------------------------------- helpers
def test_count_episodes(tmp_path):
    assert count_episodes(tmp_path) == 0
    (tmp_path / "ep_000.h5").write_text("x")
    (tmp_path / "ep_001.h5").write_text("y")
    (tmp_path / "session.json").write_text("{}")
    assert count_episodes(tmp_path) == 2


def test_latest_session_picks_newest(tmp_path):
    a = tmp_path / "manual" / "map_a"
    b = tmp_path / "manual" / "map_b"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    # Make 'b' newer.
    later = time.time() + 10.0
    os.utime(b, (later, later))
    assert latest_session(tmp_path) == b


def test_latest_session_kind_filter(tmp_path):
    m = tmp_path / "manual" / "map_m"
    a = tmp_path / "batch" / "map_a"
    m.mkdir(parents=True)
    a.mkdir(parents=True)
    later = time.time() + 10.0
    os.utime(a, (later, later))
    # batch is newer overall, but kind=manual should still find m.
    assert latest_session(tmp_path) == a
    assert latest_session(tmp_path, "manual") == m


def test_latest_session_none_for_empty(tmp_path):
    assert latest_session(tmp_path) is None


def test_discover_episodes_recursive(tmp_path):
    s1 = tmp_path / "manual" / "s1"
    s2 = tmp_path / "batch" / "s2"
    s1.mkdir(parents=True)
    s2.mkdir(parents=True)
    (s1 / "ep_000.h5").write_text("a")
    (s2 / "ep_000.h5").write_text("b")
    (s2 / "ep_001.h5").write_text("c")
    found = discover_episodes(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["ep_000.h5", "ep_000.h5", "ep_001.h5"]
    # Single-file case.
    one = discover_episodes(s1 / "ep_000.h5")
    assert len(one) == 1


def test_discover_episodes_nonexistent_returns_empty(tmp_path):
    assert discover_episodes(tmp_path / "does_not_exist") == []
