"""Parse and validate input run CSVs into typed RunSpec objects."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


class CsvSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float

    def to_str(self) -> str:
        return f"{self.x},{self.y},{self.yaw}"


@dataclass(frozen=True)
class RunSpec:
    run_id: int
    notes: str
    start: Pose
    goals: tuple  # tuple[Pose, ...]

    def legs(self) -> list[tuple[Pose, Pose]]:
        """Return [(start, g1), (g1, g2), ...]."""
        chain = [self.start, *self.goals]
        return list(zip(chain[:-1], chain[1:]))


def load_runs_csv(path: str | Path) -> list[RunSpec]:
    path = Path(path)
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Validate required base columns
        required = ['run_id', 'notes', 'start_x', 'start_y', 'start_yaw']
        for col in required:
            if col not in fieldnames:
                raise CsvSchemaError(
                    f"{path}: missing required column '{col}' "
                    f"(have: {fieldnames})")

        # Discover goal triples: g1_x/y/yaw, g2_x/y/yaw, ...
        n_goals = 0
        while True:
            i = n_goals + 1
            if all(f'g{i}_{ax}' in fieldnames for ax in ('x', 'y', 'yaw')):
                n_goals += 1
            else:
                break
        if n_goals == 0:
            raise CsvSchemaError(
                f"{path}: no goal columns found (need at least g1_x, g1_y, g1_yaw)")

        # Validate contiguity: no gaps allowed
        for i in range(1, n_goals + 1):
            for ax in ('x', 'y', 'yaw'):
                col = f'g{i}_{ax}'
                if col not in fieldnames:
                    raise CsvSchemaError(
                        f"{path}: gap in goal columns — '{col}' is missing "
                        f"but earlier goal columns exist")

        rows: list[RunSpec] = []
        seen_ids: set[int] = set()

        for row_idx, row in enumerate(reader):
            # parse run_id
            try:
                run_id = int(row['run_id'])
            except (ValueError, KeyError):
                raise CsvSchemaError(
                    f"{path} row {row_idx}: cannot parse run_id={row.get('run_id')!r}")
            if run_id in seen_ids:
                raise CsvSchemaError(
                    f"{path} row {row_idx}: duplicate run_id={run_id}")
            seen_ids.add(run_id)

            # parse start
            try:
                start = Pose(
                    x=float(row['start_x']),
                    y=float(row['start_y']),
                    yaw=float(row['start_yaw']),
                )
            except (ValueError, KeyError) as e:
                raise CsvSchemaError(
                    f"{path} row {row_idx} (run_id={run_id}): bad start pose — {e}")

            # parse goals (stop at first empty triple)
            goals: list[Pose] = []
            for i in range(1, n_goals + 1):
                gx_s = row.get(f'g{i}_x', '').strip()
                gy_s = row.get(f'g{i}_y', '').strip()
                gyaw_s = row.get(f'g{i}_yaw', '').strip()
                if not gx_s:
                    break  # trailing empty columns — truncate
                try:
                    goals.append(Pose(
                        x=float(gx_s),
                        y=float(gy_s),
                        yaw=float(gyaw_s),
                    ))
                except ValueError as e:
                    raise CsvSchemaError(
                        f"{path} row {row_idx} (run_id={run_id}): "
                        f"bad g{i} pose — {e}")

            if not goals:
                raise CsvSchemaError(
                    f"{path} row {row_idx} (run_id={run_id}): no valid goals")

            if len(goals) * len(goals[0].to_str()) > 0:  # max-legs guard
                max_legs = 999
                if len(goals) > max_legs:
                    raise CsvSchemaError(
                        f"{path} row {row_idx} (run_id={run_id}): "
                        f"too many goals ({len(goals)} > {max_legs}); "
                        f"episode_id encoding requires ≤ {max_legs} legs per run")

            rows.append(RunSpec(
                run_id=run_id,
                notes=row.get('notes', ''),
                start=start,
                goals=tuple(goals),
            ))

    return rows
