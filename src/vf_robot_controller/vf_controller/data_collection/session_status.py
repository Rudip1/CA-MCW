#!/usr/bin/env python3
"""
session_status.py — live progress monitor for a data-collection session.

Two modes:

  $ ros2 run vf_robot_controller vf_session_status
  $ vf_session_status                                  # picks latest session
  $ vf_session_status --session <path-to-session-dir>  # specific session
  $ vf_session_status --watch                          # reprint every 1 s

Reads session.json + manifest.csv from disk plus stat()'s the currently-open
episode (the .h5 file with the most recent mtime that does not yet appear in
manifest.csv) so it works without any ROS subscription.

Optionally subscribes to ``/vf/collector_status`` (Float32MultiArray
published by data_collector_node) when run under ros2 run, for instantaneous
in-flight cycle counts.

Pure-Python prints — no curses, no extra deps.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from vf_controller.data_collection.session import (
        latest_session,
        SESSION_KIND_BATCH,
        SESSION_KIND_MANUAL,
    )
except ImportError:  # standalone invocation without the package installed
    SESSION_KIND_MANUAL = "manual"
    SESSION_KIND_BATCH = "batch"

    def latest_session(root, session_kind=None):
        root_p = Path(root)
        if not root_p.is_dir():
            return None
        kinds = ([session_kind] if session_kind
                 else [SESSION_KIND_MANUAL, SESSION_KIND_BATCH])
        cands = []
        for k in kinds:
            d = root_p / k
            if not d.is_dir():
                continue
            cands.extend(s for s in d.iterdir() if s.is_dir())
        return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _training_root() -> Path:
    """Resolve the workspace TRAINING_ROOT without forcing a hard import."""
    try:
        from vf_robot_utils.constants import TRAINING_ROOT
        return Path(TRAINING_ROOT)
    except ImportError:
        env = os.environ.get("VF_DATA_ROOT")
        if env:
            return Path(env) / "vf_data_training"
        return Path.home() / "CA-MCW" / "vf_data" / "vf_data_training"


def _load_manifest(session_dir: Path) -> List[Dict[str, str]]:
    p = session_dir / "manifest.csv"
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _load_session_json(session_dir: Path) -> Dict[str, object]:
    p = session_dir / "session.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _open_episode(session_dir: Path,
                  manifest: List[Dict[str, str]]) -> Optional[Path]:
    """Return the path of an .h5 in session_dir not yet referenced by the
    manifest, or None.

    The collector writes manifest rows on close, so any .h5 not in the
    manifest is the in-flight episode (or, after a crash, an orphan that
    will need cleanup).
    """
    closed = {row["h5_filename"] for row in manifest if row.get("h5_filename")}
    h5s = sorted(session_dir.glob("ep_*.h5"))
    for p in h5s:
        if p.name not in closed:
            return p
    return None


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f TB" % n


def _human_secs(s: float) -> str:
    if s < 60:
        return "%.0fs" % s
    if s < 3600:
        return "%dm %02ds" % (s // 60, s % 60)
    return "%dh %02dm" % (s // 3600, (s % 3600) // 60)


def _stat(session_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, str]],
                                       Optional[Path]]:
    info = _load_session_json(session_dir)
    manifest = _load_manifest(session_dir)
    open_ep = _open_episode(session_dir, manifest)
    return info, manifest, open_ep


def _print_status(session_dir: Path) -> None:
    info, manifest, open_ep = _stat(session_dir)
    n_total = len(manifest)
    n_succ = sum(1 for r in manifest if str(r.get("success")).lower() == "true")
    n_fail = n_total - n_succ
    total_steps = sum(int(r.get("n_steps", 0) or 0) for r in manifest)
    total_dur = sum(float(r.get("duration_s", 0) or 0.0) for r in manifest)
    total_size = sum(int(r.get("size_bytes", 0) or 0) for r in manifest)

    started = info.get("started_at_iso", "?")
    age_s = 0.0
    if isinstance(started, str) and started != "?":
        try:
            t0 = datetime.datetime.fromisoformat(started)
            now = datetime.datetime.now(tz=t0.tzinfo)
            age_s = (now - t0).total_seconds()
        except Exception:
            pass

    print("=" * 72)
    print("session: %s" % session_dir)
    print("  kind:        %s" % info.get("session_kind", "?"))
    print("  map:         %s" % info.get("map_name", "?"))
    print("  controller:  %s   weight_provider: %s   channels: %s"
          % (info.get("controller_mode", "?"),
             info.get("weight_provider", "?"),
             info.get("channels_config", "?")))
    print("  started:     %s   (running %s)" % (started, _human_secs(age_s)))
    print("-" * 72)
    print("episodes:     %d  (success=%d, failed=%d)" % (n_total, n_succ, n_fail))
    print("total cycles: %d  (~%s of recording at %s Hz)"
          % (total_steps, _human_secs(total_dur),
             "20" if total_dur > 0 else "?"))
    print("total size:   %s" % _human_bytes(total_size))
    if open_ep is not None:
        try:
            sz = open_ep.stat().st_size
            mt = open_ep.stat().st_mtime
            elapsed = max(0.0, time.time() - mt)
        except FileNotFoundError:
            sz = 0
            elapsed = 0.0
        print("-" * 72)
        print("open episode: %s" % open_ep.name)
        print("  size:    %s   last write: %s ago" % (
            _human_bytes(sz), _human_secs(elapsed),
        ))
    else:
        print("-" * 72)
        print("open episode: <none — waiting for next /goal_pose>")
    print("=" * 72)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Live status for a data-collection session.",
    )
    p.add_argument("--session", type=str, default=None,
                   help="Path to a session folder. "
                        "Default: latest under TRAINING_ROOT.")
    p.add_argument("--root", type=str, default=None,
                   help="Override training root "
                        "(default: $VF_DATA_ROOT/vf_data_training).")
    p.add_argument("--kind", choices=("manual", "batch", None),
                   default=None,
                   help="Restrict latest-session search to one kind.")
    p.add_argument("--watch", action="store_true",
                   help="Reprint every 1 s until Ctrl-C.")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Watch interval seconds.")
    args = p.parse_args(argv)

    if args.session:
        sd = Path(args.session).resolve()
    else:
        root = Path(args.root) if args.root else _training_root()
        sd_path = latest_session(root, args.kind)
        if sd_path is None:
            print(
                "[vf_session_status] no session found under %s" % root,
                file=sys.stderr,
            )
            return 2
        sd = sd_path

    if not sd.is_dir():
        print(
            "[vf_session_status] not a directory: %s" % sd,
            file=sys.stderr,
        )
        return 2

    if not args.watch:
        _print_status(sd)
        return 0

    try:
        while True:
            os.system("clear")
            _print_status(sd)
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
