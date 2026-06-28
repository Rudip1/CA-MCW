# vf_robot_utils

Evaluation, data-collection, and analysis harness for the VF adaptive-MPPI
navigation thesis. Reads HDF5 episode logs + ONNX models from
[`vf_robot_controller`](../vf_robot_controller), drives closed-loop sweeps
through Gazebo + Nav2, and renders the thesis comparison figures and
summary tables.

> Deep internals (math, control flow, HDF5 schema field-by-field, statistical
> definitions, every metric formula) live in
> the package source. Read it before changing any aggregator,
> figure, or metric module.

---

## Quick map of what this package does

```
maps/<map>/training_goalposes_collect.csv     ← record tours in RViz
maps/<map>/evaluation_goalposes_collect.csv   ← record tours in RViz
            │
            ▼
   batch launch files
     vf_data_training/batch/vf_data_training_batch_fixedwt.launch.py
     vf_data_evaluation/batch/vf_data_evaluation_batch_{family}.launch.py
            │
            ├── bringup_launch.py     ← Gazebo + Nav2 + controller + sidecar
            ├── data_collector_node   ← writes one HDF5 per goal
            └── tour_runner           ← drives NavigateToPose through CSV row
            │
            ▼
   vf_data/vf_data_training/batch/<map>/<goal>/<Planner>/<controller>/run_*.h5
   vf_data/vf_data_evaluation/batch/<map>/<goal>/<Planner>/<variant>/run_*.h5
            │
            ▼
   tools/run_comparison.sh       (headline 6-controller comparison)
   tools/run_comparison_all.sh   (every-controller comparison, appendix)
     → aggregate (6-tier metric taxonomy) → results.csv
     → TOPSIS selection (controller_selection.csv + sensitivity)
     → comparison figures + comparison_summary.csv
```

All output paths live inside the workspace under `vf_data/`. Override the
whole store with `export VF_DATA_ROOT=/mnt/ssd/vf_data`. `vf_data/` is
gitignored.

---

# Part 1 — User guide

## 1.1 Build

```bash
cd ~/CA-MCW
colcon build --packages-select vf_robot_utils --symlink-install
source install/setup.bash
```

`--symlink-install` means Python edits don't need a rebuild. Re-run
`colcon build` after `setup.py`, `package.xml`, or any launch file under
`launch/` changes.

## 1.2 Record a goal-pose tour (one-time per map)

No robot, no Nav2 — only `map_server`, RViz, and the `pose_recorder` node.
Each Ctrl-C appends one row (auto-incrementing `run_id`) to a wide CSV
under `~/CA-MCW/maps/<map>/`.

```bash
# Training tour  →  maps/<map>/training_goalposes_collect.csv
ros2 launch vf_robot_utils training_goalposes_collect.launch.py \
    map_name:=house_my1_map

# Evaluation tour →  maps/<map>/evaluation_goalposes_collect.csv
ros2 launch vf_robot_utils evaluation_goalposes_collect.launch.py \
    map_name:=house_my1_map
```

In RViz:

1. **2D Pose Estimate** click → tour start (writes `start_x,start_y,start_yaw`)
2. **Nav2 Goal** clicks → waypoints `g1`, `g2`, … (one column triple each)
3. Ctrl-C → writes the row, increments `run_id`, exits

CSV format (wide):
```
run_id, notes, start_x, start_y, start_yaw,
g1_x, g1_y, g1_yaw, g2_x, g2_y, g2_yaw, ...
```

Both the training and evaluation CSVs use the same schema; the only
difference is which file you read from at replay time.

## 1.3 Collect training data (fixed weights)

This is what produced the 85-episode corpus that trained every imitation /
raw / oracle model. One launch starts Gazebo, brings up Nav2 with
`controller:=vf_fixedwt`, the `data_collector_node`, and `tour_runner`
which drives `NavigateToPose` through one CSV row.

```bash
# T1 — Gazebo (separate terminal)
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py

# T2 — bringup + collector + tour driver (one launch)
ros2 launch vf_robot_utils vf_data_training_batch_fixedwt.launch.py \
    map:=house_my1_map  planner:=NavFn  run_id:=0
```

Output: one HDF5 per goal in
```
vf_data/vf_data_training/batch/house_my1_map/<goal>/NavFn/vf_fixedwt/run_<ts>.h5
```
plus a `session.json` and `manifest.csv` describing the session.

**Optional knobs** (full list in `launch_builders.py` / launch file):

| Arg | Default | Purpose |
|---|---|---|
| `reposition_first` | `true` | Drive to row's `start_*` before sending `g1` |
| `post_reposition_stabilize_s` | `5.0` | Pause after reaching start, before first goal |
| `settle_s` | `3.0` | Pause between consecutive goals |
| `inter_leg_pause_s` | `1.0` | Buffer after CLOSE before next OPEN (collector flush) |
| `per_goal_timeout_s` | `180.0` | Hard timeout per `NavigateToPose` |
| `headless` | `false` | True = unattended (no RViz, gazebo server-only) |

To replay several rows in a loop the shell wrapper at
`launch/vf_data_training/batch/batch_data_training.sh` calls the launch
sequentially.

## 1.4 Collect evaluation data — the three model families + four baselines

After training (see [`vf_robot_controller`](../vf_robot_controller#train)),
each `(channel, hyperparam)` pair produces one ONNX in:

| Family | ONNX path |
|---|---|
| `imitationwt` | `models/imitation_wt/<ch>_<hp>/imitation.onnx` |
| `rawwt`       | `models/metacritic_raw_wt/<ch>_<hp>/meta_critic.onnx` |
| `oraclewt`    | `models/metacritic_oracle_wt/<ch>_<hp>/meta_critic.onnx` |

Note: `imitation.onnx` artefacts (all 9) were retrained 2026-05-11
with `critic_history` masked to zero — the channel is silent in
`vf_imitationwt` (MPPI off), and the original imitation sweep collapsed
to a zero-twist fixed point at deploy time. Metadata records
`"zero_channels": ["critic_history"]`. Pre-fix artefacts have been
deleted. Oracle and raw critic models were NOT retrained — they only
run with MPPI active so the channel is healthy at both train and
deploy time.

Each family has a thin launch wrapper that spins up bringup with the
correct controller + sidecar + ONNX, plus the `data_collector_node` and
`tour_runner`. One launch = one CSV row × one planner × one variant.

```bash
# T1 — Gazebo
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py

# T2 — pick a family and run one tour row
ros2 launch vf_robot_utils vf_data_evaluation_batch_imitationwt.launch.py \
    map:=house_my1_map planner:=NavFn run_id:=0 hp:=hardreg ch:=v3

ros2 launch vf_robot_utils vf_data_evaluation_batch_rawwt.launch.py \
    map:=house_my1_map planner:=NavFn run_id:=0 hp:=normal ch:=v1

ros2 launch vf_robot_utils vf_data_evaluation_batch_oraclewt.launch.py \
    map:=house_my1_map planner:=NavFn run_id:=0 hp:=tuned ch:=v2
```

The variant tag burns `family_hp_ch` into the output path:
```
vf_data/vf_data_evaluation/batch/house_my1_map/<goal>/NavFn/imitationwt_hardreg_v3/run_<ts>.h5
```

For nav2 baselines (no ONNX, no sidecar) and the `vf_fixedwt` ablation
baseline, use the matching wrappers:

```bash
# vf_fixedwt — same as training, but logged under vf_data_evaluation/
ros2 launch vf_robot_utils vf_data_evaluation_batch_fixedwt.launch.py \
    map:=house_my1_map planner:=NavFn run_id:=0

# stock Nav2 baselines
ros2 launch vf_robot_utils vf_data_evaluation_batch_mppi.launch.py \
    map:=house_my1_map planner:=NavFn run_id:=0
ros2 launch vf_robot_utils vf_data_evaluation_batch_dwb.launch.py \
    map:=house_my1_map planner:=NavFn run_id:=0
ros2 launch vf_robot_utils vf_data_evaluation_batch_rpp.launch.py \
    map:=house_my1_map planner:=NavFn run_id:=0
ros2 launch vf_robot_utils vf_data_evaluation_batch_graceful.launch.py \
    map:=house_my1_map planner:=NavFn run_id:=0
```

All eight wrappers share the same arg surface and the same output layout;
they differ only in the bringup controller and (for trained families) the
ONNX path. Each launch shuts itself down as soon as `tour_runner` exits,
so you can loop over them in a shell script — see § 1.6 below.

**Required args (every eval launch):**

| Arg | Notes |
|---|---|
| `map` | Map folder under `MAPS_ROOT` (e.g. `house_my1_map`) |
| `planner` | One of `NavFn`, `SmacPlanner2D`, `SmacPlannerHybrid`, `SmacLattice`, `ThetaStar` |
| `run_id` | Integer row in `evaluation_goalposes_collect.csv` |
| `hp` (trained_wt only) | `normal`, `tuned`, `hardreg` |
| `ch` (trained_wt only) | `v1`, `v2`, `v3` — channel set the model was trained on |

## 1.5 Output layout

```
vf_data/
├── vf_data_training/batch/<map>/<goal>/<Planner>/vf_fixedwt/
│   ├── run_<YYYYMMDD_HHMMSS>.h5      one per goal
│   ├── session.json                  one per launch session
│   └── manifest.csv                  one row per closed episode
│
└── vf_data_evaluation/batch/<map>/<goal>/<Planner>/<variant>/
    ├── run_<YYYYMMDD_HHMMSS>.h5
    ├── session.json
    └── manifest.csv

# `<variant>` is one of:
#   imitationwt_<hp>_<ch>   rawwt_<hp>_<ch>   oraclewt_<hp>_<ch>
#   fixedwt   mppi   dwb   rpp   graceful
```

The HDF5 schema is identical across training and evaluation (same writer:
`vf_robot_controller`'s `data_collector_node`). Full per-field schema in
the source code § HDF5.

## 1.6 Sweep — collect evaluation data

`tools/sweep_eval.sh` is the recommended entry point. It drives one
cohort per invocation (trained variants, baselines, or a single
variant), never mixes the two cohorts, never deletes existing HDF5,
prints a pre-flight count, checks Gazebo is running, and logs each run.

```bash
# Gazebo first, in a separate terminal:
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py

# then, in another terminal — one cohort per invocation:
bash tools/sweep_eval.sh trained                  # 19 trained variants
bash tools/sweep_eval.sh baselines                # 5 baseline controllers
bash tools/sweep_eval.sh rawwt_hardreg_v3         # a single variant
bash tools/sweep_eval.sh trained --list           # preview, launch nothing

# Let Nav2's BT keep replanning + running recoveries until goal achieved:
bash tools/sweep_eval.sh baselines --no-timeout
# Or set the two caps explicitly (seconds, forwarded as launch args):
bash tools/sweep_eval.sh baselines --per-goal-timeout 600 --episode-timeout 600
```

Flags (any order, after the cohort/variant):

| Flag | Effect |
|---|---|
| `--list` | Pre-flight table only — show counts, exit without launching |
| `--no-timeout` | Shortcut for `--per-goal-timeout 99999 --episode-timeout 99999` |
| `--per-goal-timeout <s>` | Forwarded as `per_goal_timeout_s:=<s>` (tour_runner per-`NavigateToPose` cap; default 180) |
| `--episode-timeout <s>` | Forwarded as `episode_timeout_s:=<s>` (collector-side episode cap; default 180) |

Env: `MAP`, `PLANNER`, `RUN`, `HEADLESS`, `RVIZ`, `LAUNCH_TIMEOUT` (script-side outer cap, default 1800 s; independent of the Nav2-side timeouts above).

Or loop the launch wrappers by hand — this is what `sweep_eval.sh`
does internally:

```bash
export MAP=house_my1_map
export PLANNER=NavFn
export RUN=0

# 18 metacritic variants — rawwt + oraclewt, 3 hyperparams x 3 channel sets
for FAMILY in rawwt oraclewt; do
  for HP in normal tuned hardreg; do
    for CH in v1 v2 v3; do
      echo "=== $FAMILY $HP $CH ==="
      ros2 launch vf_robot_utils \
          vf_data_evaluation_batch_${FAMILY}.launch.py \
          map:=$MAP planner:=$PLANNER run_id:=$RUN hp:=$HP ch:=$CH \
          headless:=false rviz:=true
      sleep 5
    done
  done
done
```
```bash
# imitationwt has a single variant (normal / v1 only) -> 19 trained total
export MAP=house_my1_map
export PLANNER=NavFn
export RUN=0
ros2 launch vf_robot_utils vf_data_evaluation_batch_imitationwt.launch.py \
    map:=$MAP planner:=$PLANNER run_id:=$RUN hp:=normal ch:=v1 \
    headless:=false rviz:=true
sleep 5
```
```bash
# 5 baselines (no hp/ch)
export MAP=house_my1_map
export PLANNER=NavFn
export RUN=0
for FAMILY in fixedwt mppi dwb rpp graceful; do
  ros2 launch vf_robot_utils \
      vf_data_evaluation_batch_${FAMILY}.launch.py \
      map:=$MAP planner:=$PLANNER run_id:=$RUN \
      headless:=false rviz:=true
  sleep 5
done
```

Wall time: ~3–6 min per variant. Nineteen trained variants + five
baselines per map ≈ 70–145 min. Run trained and baseline cohorts in
separate sweeps; do not interleave them.

## 1.7 Analyse — one command

```bash
# Headline 6-controller comparison (3 classical + fixedwt + imitation
# + TOPSIS rank-1 trained):
bash tools/run_comparison.sh

# Every-controller comparison (thesis appendix):
bash tools/run_comparison_all.sh
```

Pure Python, no Gazebo, no ROS 2 nav stack. Each script aggregates the
HDF5 episodes, runs TOPSIS selection, then renders the comparison
figures. Outputs land under

```
vf_data/vf_data_evaluation/batch/<map>/_aggregate/<stem>/
├── results.csv                    (master per-episode table)
├── controller_selection.csv       (TOPSIS — the selection method)
├── pareto_front.json              (non-dominated controllers)
├── sensitivity.csv                (TOPSIS robustness under alt weights)
├── weights.json                   (entropy weights + correlation matrix)
├── comparison/                    (headline 6-controller set)
│   ├── tiers/         tier_1_outcome.pdf ... tier_6_adaptation.pdf
│   ├── topsis_ranking.pdf
│   ├── xte/           violin / profile / envelope
│   ├── trajectories/  per-goal overlays
│   └── comparison_summary.csv
└── comparison_all/                (every controller — appendix)
    ├── tiers/   per-tier bar panels
    ├── topsis_ranking.pdf
    ├── xte/     xte_violin.pdf
    └── comparison_summary.csv
```

See `evaluation_plots.txt` at the package root for a per-figure
catalog (what each PDF shows, where it fits in the thesis chapter,
caption templates).

Override the map or stem via env vars:

```bash
MAP=house_my1_map STEM=thesis_eval bash tools/run_comparison.sh
```

You can also invoke the pipeline modules by name:

```bash
python3 -m vf_robot_utils.analysis.csv_pipeline.aggregate_csv \
    --root vf_data/vf_data_evaluation/batch/house_my1_map --csv-stem thesis_eval

python3 -m vf_robot_utils.analysis.csv_pipeline.controller_selection --results ...
python3 -m vf_robot_utils.analysis.figures.comparison_figures \
    --results ... --agg-dir ... --root ... --scope headline
```

## 1.8 Sync results into the thesis

`tools/sync_to_thesis.sh` copies the analysis output into the thesis
tree so the Results chapter can be written and re-figured:

```bash
bash tools/sync_to_thesis.sh
```

- comparison figures → `<thesis-repo>/figures/ch5/comparison{,_all}/`
- `results.csv` + `controller_selection.csv` + the comparison summary
  CSVs + `evaluation_plots.txt` → `references/docs/` — the context
  bundle for thesis writing
- `references/docs/SNAPSHOT.md` — sync date, git SHA, per-controller `n`

It copies only; it never edits LaTeX. The tier table in
`ch5_results.tex` is regenerated separately.

---

# Part 2 — Developer guide

## 2.1 Hard rules

1. **Never write under `$HOME`.** All artefacts go under `TRAINING_ROOT`
   or `EVALUATION_ROOT` (= `vf_data/vf_data_training/`,
   `vf_data/vf_data_evaluation/`). Override the whole store with
   `VF_DATA_ROOT`. Per-tree overrides: `VF_MAPS_ROOT`, `VF_MODELS_ROOT`,
   `VF_WORKSPACE_ROOT`. Never hard-code an absolute path.
2. **Cross-episode stats use `ddof=1`.** Sample std + SEM + bootstrap 95 %
   CI. `np.std(..., ddof=0)` is forbidden for summary stats; it's fine
   for within-episode signal stats *if explicitly commented*.
3. **Drop `inf` and `NaN` before any mean / CI.** `t1_mmc_m == inf`
   means "no lethal cell within search radius" — that's a categorical
   signal, not a clearance value.
4. **Every error-bar must be labelled** in the figure caption with which
   estimator (sample std, SEM, Wilson 95 %, bootstrap 95 %). Bar charts
   without error bars are not allowed in thesis figures.
5. **Critic order in `launch_builders.py`** must match
   `vf_*wt.yaml` in `vf_robot_bringup/config/nav2/controllers/`. If a
   critic is added or removed in YAML, update the constant. The HDF5
   `critic_costs` columns are positional.
6. **No emojis** in code, commit messages, or figure captions.

## 2.2 Package layout

```
vf_robot_utils/
├── launch_builders.py          shared LaunchDescription factory for the
│                               8 vf_data_evaluation_batch_*.launch.py
│                               wrappers (one function: family → LD)
├── constants.py                workspace-rooted paths + model dirs
├── io/
│   ├── csv_schema.py           load_runs_csv() + RunSpec dataclass
│   ├── hash_utils.py           SHA-1 of files (cache validation)
│   ├── results_writer.py       append-only results.csv writer
│   └── vf_data_paths.py        canonical path helpers
├── analysis/
│   ├── statistical_tests.py    Wilcoxon, Mann-Whitney, bootstrap, Wilson,
│   │                           Newcombe-Wilson, Cliff's delta, BH-adjusted q
│   ├── style.py                matplotlib style + controller_color()
│   ├── csv_pipeline/           CSV / metric extraction pipeline
│   │   ├── __init__.py             METRIC_CATALOG (6-tier taxonomy) +
│   │   │                           TOPSIS_CRITERIA + controller universe
│   │   ├── metrics_from_h5.py      per-episode metrics from HDF5 schema;
│   │   │                           emits t1_..t6_ column groups + t2_spl_safe
│   │   ├── aggregate_csv.py        walks tree → master results.csv
│   │   │                           (--xte-ref filter for legacy HDF5s)
│   │   └── controller_selection.py TOPSIS + entropy weights + Pareto +
│   │                               sensitivity — the selection method
│   ├── figures/                Comparison figure pipeline
│   │   ├── comparison_figures.py     orchestrator (--scope headline|full)
│   │   ├── renderers_bars.py         per-tier bar-panel renderer
│   │   ├── renderers_xte.py          XTE violin / profile / envelope
│   │   └── renderers_trajectories.py per-goal trajectory overlay
│   └── (statistical_tests.py, style.py — shared helpers)
├── mapgen/map_loader.py        load .yaml + .pgm into a 2-D grid
└── tools/
    ├── tour_runner.py          NavigateToPose driver (1 console_script)
    └── pose_recorder.py        RViz click -> CSV row (1 console_script)

launch/
├── goalposes_collect/
│   ├── training_goalposes_collect.launch.py
│   └── evaluation_goalposes_collect.launch.py
├── vf_data_training/batch/
│   ├── vf_data_training_batch_fixedwt.launch.py
│   └── batch_data_training.sh    shell loop wrapper
└── vf_data_evaluation/batch/
    ├── vf_data_evaluation_batch_imitationwt.launch.py    (trained_wt)
    ├── vf_data_evaluation_batch_rawwt.launch.py          (trained_wt)
    ├── vf_data_evaluation_batch_oraclewt.launch.py       (trained_wt)
    ├── vf_data_evaluation_batch_fixedwt.launch.py        (baseline)
    ├── vf_data_evaluation_batch_mppi.launch.py           (baseline)
    ├── vf_data_evaluation_batch_dwb.launch.py            (baseline)
    ├── vf_data_evaluation_batch_rpp.launch.py            (baseline)
    └── vf_data_evaluation_batch_graceful.launch.py       (baseline)

tools/
├── sweep_eval.sh               data-collection sweep (trained|baselines|variant)
├── run_comparison.sh           aggregate → TOPSIS → headline 6-controller figures
├── run_comparison_all.sh       same, every controller (thesis appendix)
└── sync_to_thesis.sh           copy figures + CSVs into the thesis tree

test/                           pytest suite (51 test functions; flake8
                                stricter than the codebase — aspirational)
runs/                           sample input CSVs (collect_corridor, etc.)
scenarios/                      scenario YAMLs (unused by the new
                                pipeline; kept for reference)
RESEARCH_NOTES.md               methodology references (Wilson, Newcombe,
                                bootstrap, Cliff's δ, BH-adjusted p)
the source code                     deep internals — math, flows, schemas
```

## 2.3 Add a new eval family

Edit `vf_robot_utils/launch_builders.py`:

1. Add a row to `_FAMILY` with the right `class` (`trained_wt` or
   `baseline`), bringup controller name, sidecar type, and HDF5
   `controller_mode` / `weight_provider` tags.
2. If it's a trained family, make sure the ONNX path is resolvable by
   `bringup_launch.py` under the conventional folder.
3. Create a one-line wrapper at
   `launch/vf_data_evaluation/batch/vf_data_evaluation_batch_<family>.launch.py`
   that calls `build_eval_batch_launch_description("<family>")`.

No other code change is needed — collector, tour runner, and output paths
all flow from the family entry.

## 2.4 Add a new metric

1. Add the column key (`t<N>_<name>`) to **both** `RESULTS_FIELDS` in
   `io/results_writer.py` *and* `METRIC_CATALOG` in
   `analysis/csv_pipeline/__init__.py` (with the correct category,
   direction, and `drop_inf` flag). The writer silently drops unknown
   keys, and figures iterate the catalog — missing either side is a
   silent bug.
2. Add the computation to `metrics_from_h5._tier<N>(...)`.
3. Old HDF5s get NaN automatically (writer uses
   `extrasaction='ignore'`).
4. Add a unit test in `test/`.

## 2.5 Run the tests

```bash
colcon test --packages-select vf_robot_utils
colcon test-result --test-result-base build/vf_robot_utils
```

The `test_flake8` failure is pre-existing: the codebase uses
column-aligned `=` assignments which trigger `E221`. It does not block
the functional tests.

---

# Part 3 — Reference

## 3.1 Metric taxonomy (high level)

Metrics are grouped into six **categories** in `METRIC_CATALOG`.
Column keys retain their `t1_..t6_` prefix (stable internal IDs);
only the category labels are new. See the `metric-tier-conventions`
skill for the complete table.

| Category | Example columns | TOPSIS role |
|---|---|---|
| **Outcome** | `t2_success`, failure flags, `t2_gpe_m` | gate (`SR > 0` excludes a controller) |
| **Efficiency** | `t2_spl`, `t2_spl_safe` (headline), `t2_duration_s`, `t2_actual_path_m`, `t3_mean_lin_vel`, `t3_stall_fraction` | criteria: `t2_spl`, `t2_duration_s` |
| **Safety** | `t1_collision`, `t1_mmc_m`, `t1_mdo_m`, `t1_p5_clear_m`, `t1_near_miss_rate`, `t1_svr` | criteria: `t1_mmc_m`, `t1_collision` |
| **Motion quality** | `t4_mean_jerk`, `t4_max_jerk`, `t4_ang_vel_std`, `t4_cmd_accel_rms`, `t4_cmd_sign_flips` | criterion: `t4_mean_jerk` |
| **Path adherence** | `t6_mean_xte_m`, `t6_max_xte_m`, `t6_rmse_xte_m`, `t6_frechet_m`, `t6_path_coverage_frac` | criterion: `t6_mean_xte_m` |
| **Adaptation** *(diagnostic)* | `t5_mean_entropy`, `t5_mean_entropy_norm`, `t5_total_variation`, `t5_dominant_critic_fraction` | excluded (trained-only) |

**Headline metric:** `t2_spl_safe = t2_spl × (1 − t1_collision)` —
a collision forces SPL to 0 so the top-line cannot reward an unsafe
success.

**Primary selection:** TOPSIS with Shannon-entropy weights over the
six criteria above (`controller_selection.py`). Pareto-front and
±20% MC sensitivity are cross-checks.

**Path adherence reference:** each controller's own time-active
`/plan` (sim_time-bisected over HDF5 `global_path_plans/`).
Cross-controller comparison goes through TOPSIS, never through raw
`t6_*` values.

Full formulas, units, and edge-case rules in the source code § Metrics.

## 3.2 Statistical conventions

- Sample std (`ddof=1`), SEM (`σ/√n`), bootstrap 95 % CI (`B = 10 000`,
  percentile method, fixed seed).
- Binomial outcomes (SR, CR): **Wilson 95 %** interval; **Newcombe-Wilson**
  for the difference of two proportions.
- Paired controller comparisons: **paired Wilcoxon signed-rank**, joined on
  `(planner, run_id, leg_id)`.
- Unpaired: **Mann-Whitney U**.
- Multi-pair: Benjamini-Hochberg-adjusted q-values.
- Effect size: **Cliff's δ** (preferred over Cohen's d for non-normal data).
- Bar charts always carry error bars. `np.std(...)` without `ddof=1` is
  forbidden for cross-episode stats.

## 3.3 Environment variables

| Variable | Default | Effect |
|---|---|---|
| `VF_WORKSPACE_ROOT` | nearest dir with `src/` + `install/` | Workspace marker |
| `VF_DATA_ROOT` | `<workspace>/vf_data` | Whole data store |
| `VF_MAPS_ROOT` | `<workspace>/maps` | 2-D maps + per-map CSVs |
| `VF_MODELS_ROOT` | `<workspace>/src/vf_robot_controller/models` | Trained weights |

## 3.4 Console scripts

| Command | Purpose |
|---|---|
| `ros2 run vf_robot_utils tour_runner` | Drive `NavigateToPose` through one CSV row (called by every batch launch) |
| `ros2 run vf_robot_utils pose_recorder` | Standalone RViz-click → CSV writer (called by `*_goalposes_collect` launches) |

## 3.5 Where to look when something breaks

| Symptom | First check |
|---|---|
| Launch shuts down immediately, "Reposition failed: rejected" | Map TF not ready when `tour_runner` sent first goal — increase `nav2_ready_timeout_s` or `post_reposition_stabilize_s` |
| All legs `final_status: timeout` | Bringup lifecycle not active. `ros2 lifecycle list /controller_server`. |
| Family-specific abort (only `imitationwt` or `rawwt` aborts) | Missing ONNX. `ls models/<family>/<ch>_<hp>/`. |
| `t1_mmc_m` is `inf` everywhere | Robot too far from any lethal cell in `SEARCH_RADIUS_M`. Drop rows or widen radius. |
| `t5_*` columns all NaN | Controller doesn't publish `/vf/applied_weights` — expected for stock baselines (`mppi`, `dwb`, `rpp`, `graceful`). |
| Empty box plot | `df[col].dropna()` empty — all values were `inf` (clearance) or `NaN` (SVR on non-VF). Filter before plotting. |
