# runs/ — legacy sample CSVs

> **ARCHIVED (2026-05-16).** The `csv_runner` / `evaluate_from_csv`
> pipeline these files belonged to has been removed. Everything now flows
> through `tour_runner` and the `vf_data_{training,evaluation}/batch`
> launches. The CSVs here (`smoke_test.csv`, `collect_corridor.csv`,
> `eval_corridor.csv`) are kept only as sample tour rows — nothing in the
> current pipeline reads this directory.

## Current workflow

Tour rows are recorded into, and replayed from, per-map CSVs:

```
maps/<map>/training_goalposes_collect.csv
maps/<map>/evaluation_goalposes_collect.csv
```

Record them with the `*_goalposes_collect` launches; replay one row with
the batch launches. Full guide: `../README.md` §1.2-1.6.

## Tour-CSV schema (still canonical)

One row = one tour: a start pose followed by one or more goal poses.
`tour_runner` repositions to `start_*`, then drives `NavigateToPose`
through `g1, g2, …` in order.

```
run_id,notes,start_x,start_y,start_yaw,g1_x,g1_y,g1_yaw[,g2_x,g2_y,g2_yaw,...]
```

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `run_id` | int | yes | Unique integer ID; selects the row at replay (`run_id:=`) |
| `notes` | str | yes | Human-readable label |
| `start_x/y/yaw` | float | yes | Tour start pose (map frame, yaw in radians) |
| `g1_x/y/yaw` | float | yes | First goal |
| `gN_x/y/yaw` | float | no | Nth goal — contiguous triples, no gaps |

Rules:
- `run_id` unique across rows.
- Goal columns come in complete triples `g{i}_{x,y,yaw}`; no gaps.
- Trailing empty columns are truncated (variable tour length per row).

A replay writes one HDF5 per goal under
`vf_data/vf_data_{training,evaluation}/batch/<map>/<goal>/<Planner>/<variant>/run_*.h5`
— see `../README.md` §1.5.
