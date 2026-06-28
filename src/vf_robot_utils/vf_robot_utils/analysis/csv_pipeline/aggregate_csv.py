#!/usr/bin/env python3
"""aggregate_csv.py — walk a vfdata eval tree, emit results.csv.

Schema-aware replacement for analysis.aggregate when the on-disk data
uses the current `data_collector_node` HDF5 layout (root-level datasets
+ attrs). Calls `metrics_from_h5.compute_row` for each .h5, fills the
identity columns (planner, controller, goal_folder, etc.), and writes
to `<root>/_aggregate/<csv-stem>/results.csv` using the canonical
RESULTS_FIELDS schema from `io.results_writer`.

Usage::

    python3 -m vf_robot_utils.analysis.csv_pipeline.aggregate_csv \\
        --root vf_data/vf_data_evaluation/batch/house_my1_map \\
        [--controllers <subset>] \\
        [--map-yaml maps/house_my1_map/house_my1_map.yaml] \\
        [--csv-stem thesis_eval] \\
        [--xte-ref active_plan ...]

``--xte-ref`` drops HDF5s whose computed ``t6_xte_ref`` is not in the
allow-list. Use ``--xte-ref active_plan`` to exclude legacy HDF5s that
predate the ``global_path_plans/`` schema (2026-05-16) and would
otherwise contaminate cross-controller tier-6 comparisons.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from vf_robot_utils.analysis.csv_pipeline import ALL_CONTROLLERS
from vf_robot_utils.analysis.csv_pipeline.metrics_from_h5 import compute_row
from vf_robot_utils.io.results_writer import ResultsWriter, RESULTS_FIELDS


def _autodetect_map_yaml(map_dir_name: str) -> Path | None:
    """Look for maps/<map_dir_name>/<map_dir_name>.yaml under VF_WORKSPACE_ROOT."""
    try:
        from vf_robot_utils.constants import MAPS_ROOT
        cand = Path(MAPS_ROOT) / map_dir_name / f"{map_dir_name}.yaml"
        if cand.exists():
            return cand
    except Exception:
        pass
    return None


def aggregate(
    root: Path,
    controllers: list[str] | None,
    map_yaml: Path | None,
    csv_stem: str,
    xte_ref_filter: list[str] | None = None,
) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(root)

    map_dir_name = root.name  # e.g. 'house_my1_map'
    if map_yaml is None:
        map_yaml = _autodetect_map_yaml(map_dir_name)
    if map_yaml is None:
        print(f"[aggregate_csv] WARNING: no map YAML for '{map_dir_name}'; "
              "tier-1 clearance metrics will be NaN.")

    out_dir = root / "_aggregate" / csv_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    if results_csv.exists():
        results_csv.unlink()

    writer = ResultsWriter(results_csv)
    controllers = controllers or ALL_CONTROLLERS
    allow_refs = set(xte_ref_filter) if xte_ref_filter else None
    if allow_refs:
        print(f"[aggregate_csv] xte-ref filter active: keep only {sorted(allow_refs)}")
    n_rows = 0
    n_skipped = 0
    n_filtered = 0

    for goal_dir in sorted(root.glob("goal_*")):
        if not goal_dir.is_dir():
            continue
        goal_folder = goal_dir.name
        for planner_dir in sorted(goal_dir.iterdir()):
            if not planner_dir.is_dir() or planner_dir.name.startswith("_"):
                continue
            planner = planner_dir.name
            for ctrl_dir in sorted(planner_dir.iterdir()):
                if not ctrl_dir.is_dir():
                    continue
                ctrl = ctrl_dir.name
                if ctrl not in controllers:
                    continue
                for h5 in sorted(ctrl_dir.glob("run_*.h5")):
                    try:
                        row = compute_row(h5, map_yaml=map_yaml)
                    except Exception as e:
                        print(f"[skip] {h5}: {type(e).__name__}: {e}")
                        n_skipped += 1
                        continue
                    if allow_refs is not None:
                        ref = row.get("t6_xte_ref")
                        if ref not in allow_refs:
                            n_filtered += 1
                            continue
                    row.update({
                        "planner":     planner,
                        "controller":  ctrl,
                        "goal_folder": goal_folder,
                        "scenario":    goal_folder,  # alias for existing figures
                        "hdf5_path":   str(h5),
                        "run_id":      0,
                        "leg_id":      0,
                        "timestamp":   h5.stem.replace("run_", ""),
                    })
                    writer.append(row)
                    n_rows += 1

    writer.close()
    msg = f"[aggregate_csv] wrote {results_csv}  rows={n_rows}  skipped={n_skipped}"
    if allow_refs is not None:
        msg += f"  filtered_by_xte_ref={n_filtered}"
    print(msg)
    return results_csv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True,
                   help="vf_data/vf_data_evaluation/batch/<map>")
    p.add_argument("--controllers", nargs="*", default=None)
    p.add_argument("--map-yaml", type=Path, default=None)
    p.add_argument("--csv-stem", default="thesis_eval")
    p.add_argument("--xte-ref", nargs="*", default=None,
                   choices=["active_plan", "final_plan", "straight_line"],
                   help="Keep only HDF5s whose computed t6_xte_ref is in "
                        "this list. Use 'active_plan' alone to exclude "
                        "legacy straight_line HDF5s — mixing XTE references "
                        "taints cross-controller tier-6 comparisons.")
    args = p.parse_args(argv)
    aggregate(args.root, args.controllers, args.map_yaml, args.csv_stem,
              xte_ref_filter=args.xte_ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
