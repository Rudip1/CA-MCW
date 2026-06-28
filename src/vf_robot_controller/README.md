# vf_robot_controller

Adaptive meta-critic Nav2 controller. Wraps the apt-installed
`nav2_mppi_controller` and adds:

- **3 GCF-aware custom critics** — Corridor, Volumetric, DynamicObstacle
- **Channel-wise feature pipeline** — `channels_v1` 126 dims, `channels_v2`
  +reynolds 130 dims, `channels_v3` +slam_persistent 170 dims
- **Geometric Complexity Field** — 2D + 3D + persistent SLAM-derived
- **Two learned controllers** — separate networks, separate runtimes:
  - `vf_inferencewt` — ONNX critic-weight modulation on top of MPPI (C++ in-process)
  - `vf_imitationwt` — direct (vx, wz) behavior cloning, MPPI bypassed (Python sidecar)

> Architecture, math, plugin internals, and HDF5 schema are in
> the package source. Full per-channel feature manifest (170 dims)
> is in the source code § "Per-slot feature manifest". Open future work:
> Phase 9 acceptance gates 9.7-3 / 9.7-5 (oracle Spearman + Gazebo eval),
> Phase 10 (channel-wise MLP + ablation matrix).

---

## Build

```bash
cd ~/CA-MCW
colcon build --packages-select vf_robot_controller --symlink-install
source install/setup.bash
colcon test --packages-select vf_robot_controller
```

C++ edits require rebuild. Python and YAML edits do not (`--symlink-install`).

---

## Quick reference — all controllers

Every controller needs Gazebo running first.

```bash
# Terminal 1 (always)
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py
```

```bash
# Terminal 2 — pick one controller
ros2 launch vf_robot_bringup bringup_launch.py \
    controller:=<MODE>  localization:=rtabmap_loc  map:=house_my1_map  use_sim_time:=true
```

| `<MODE>` | What runs | Sidecar | Publishes `/vf/per_critic_costs` | When to use |
|---|---|---|---|---|
| `vf_fixedwt`     | MPPI + 11 critics, fixed weights from YAML | none | yes | Ablation baseline + training data |
| `vf_inferencewt` | MPPI + critic weights from ONNX (C++ in-process) | `metacritic_inference_node` (raw) | yes | Thesis main result |
| `vf_imitationwt` | C++ returns zero; sidecar publishes /cmd_vel_nav directly | `imitation_inference_node` | no (MPPI never runs) | End-to-end velocity learner |
| `mppi`, `dwb`, `rpp`, `graceful` | Stock Nav2 baselines | none | no | Reference comparisons |

`/vf/per_critic_costs` is published unconditionally by `vf_fixedwt` and
`vf_inferencewt` (post-M10 — the C++ collect-mode gate was removed).
`vf_imitationwt` skips MPPI entirely, so the topic is silent.

Then drive the robot — manually (RViz "Nav2 Goal") or automatically
(see [Batch CSV sweeps](#batch-csv-sweeps)).

### Inference-side launches

For testing the sidecars in isolation (without bringing up Nav2):

```bash
# Meta-critic sidecar (raw weights — default per model_defaults.py)
ros2 launch vf_robot_controller metacritic_inference_launch.py

# Meta-critic sidecar (oracle weights)
ros2 launch vf_robot_controller metacritic_inference_launch.py \
    inference_model_type:=oracle  inference_weights:=oracle_manual_v1

# Imitation sidecar
ros2 launch vf_robot_controller imitation_inference_launch.py
```

Verify any controller:

```bash
ros2 topic hz /cmd_vel                       # ~20 Hz
ros2 topic hz /vf/features                   # ~20 Hz (vf_* controllers)
ros2 topic hz /vf/per_critic_costs           # ~20 Hz (vf_fixedwt, vf_inferencewt)
ros2 topic hz /vf/applied_weights            # ~20 Hz (vf_fixedwt, vf_inferencewt)
ros2 topic hz /vf_controller/meta_weights    # ~20 Hz (vf_inferencewt sidecar)
ros2 topic hz /cmd_vel_nav                   # ~20 Hz (vf_imitationwt sidecar)
```

---

## Collect HDF5 episodes (manual mode)

```bash
# T1 — Gazebo
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py
# T2 — fixedwt + RViz + data_collector_node
ros2 launch vf_robot_controller vf_data_training_manual_fixedwt.launch.py \
    map:=house_my1_map  planner:=NavFn  scenario_id:=corridor_traverse  seed:=42
# T3 — live monitor (optional)
ros2 run vf_robot_controller vf_session_status --watch
```

Then drive nav goals from RViz ("Nav2 Goal"). For unattended CSV-driven
tour replay (batch mode), see
[`../vf_robot_utils/README.md`](../vf_robot_utils/README.md) §
"Workflow — collect training data".

| Argument | Default | Notes |
|---|---|---|
| `map_name`            | `house_my1_map`   | RTAB-Map .db must exist in `~/CA-MCW/maps/<map>/` |
| `scenario_id`         | `manual_run`  | Tag burned into HDF5 attrs and filename |
| `seed`                | `0`           | Filename label, not a randomness seed |
| `training_root`      | `<workspace>/vf_data/vf_data_training` | Or `export VF_DATA_ROOT=...` |
| `session_suffix`      | `""`          | Optional tag appended to the session folder name |
| `episode_timeout_s`   | `180.0`       | Auto-close if goal not reached |
| `channels_config`     | `channels_v1` | `channels_v2` +reynolds; `channels_v3` +SLAM persistent |
| `goal_debounce_s`     | `0.5`         | Stable-goal time before opening an episode |
| `goal_cooldown_s`     | `2.0`         | After-close lockout for the same goal |
| `goal_dedup_radius_m` | `0.5`         | XY radius for the same-goal predicate |
| `goal_yaw_eps_rad`    | `0.35`        | Yaw tolerance for the same-goal predicate |

**Layout (one leaf per goal × planner × controller):**

```
vf_data/vf_data_training/manual/<map>/<goal_folder>/<Planner>/<controller>/
  session.json                    # written once on first episode open
  manifest.csv                    # one row per closed episode
  run_YYYYMMDD_HHMMSS.h5          # one file per nav goal
  ...
```

**Sample rate:** 20 Hz (`write_period_s=0.05`).

**One file per goal, strict.** Goal debounce + cooldown ensures a single
HDF5 per nav goal even when `/plan` replans every cycle and `/goal_pose`
yaw flips on Smac planners. Episodes close on goal-reach (`distance <
goal_radius_m=0.4` for 5 consecutive cycles), timeout, a new *different*
goal, or shutdown.

**Live monitoring:**

```bash
# CLI (no ROS dependency — reads disk state):
ros2 run vf_robot_controller vf_session_status --watch
# Or echo the publish topic:
ros2 topic echo /vf/collector_status   # [n_closed, total_steps, total_bytes,
                                       #  cur_steps, cur_bytes]
```

Schema details for `session.json`, `manifest.csv`, and the per-episode HDF5
are in the source code § "HDF5 schema" and § "Session layout".

---

## Train

Three modes — all output an ONNX + `.pt` + `metadata.json` + `feature_norm.json`
into `models/<family>/<run-name>/`.

```bash
# RAW_CRITICS — oracle-free, labels from softmax(critic_costs / temperature).
# Fastest path: collect → train with no intermediate oracle step.
# --temperature should match vf_fixedwt.yaml temperature (typically 0.3).
python3 src/vf_robot_controller/scripts/train.py --mode raw_critics \
    --data-dir ~/CA-MCW/vf_data/vf_data_training \
    --temperature 0.3 --epochs 40 --run-name my_run

# INFERENCE — oracle-labelled weights (Phase 9 QP-recovered ideal simplex).
# Requires scripts/run_oracle.py to have augmented every episode first.
python3 src/vf_robot_controller/scripts/train.py --mode inference \
    --data-dir ~/CA-MCW/vf_data/vf_data_training \
    --run-name my_run

# IMITATION — behaviour cloning on (vx, wz), no oracle needed.
# `--zero-channels critic_history` is REQUIRED: critic_history is silent
# in vf_imitationwt PASSIVE mode (MPPI off), and a model that learned to
# depend on it will collapse to zero twist at deployment. The trainer
# zeros the channel in both the Welford normaliser and the dataset; the
# metadata records "zero_channels": ["critic_history"] for traceability.
# Oracle and raw critic training do NOT need this flag — they only run
# when MPPI runs (vf_inferencewt), so the channel is healthy at both
# train and deploy time.
python3 src/vf_robot_controller/scripts/train.py --mode imitation \
    --data-dir ~/CA-MCW/vf_data/vf_data_training \
    --run-name my_run \
    --zero-channels critic_history
```

Common flags: `--epochs N`, `--batch-size N`, `--device cuda|cpu`,
`--val-fraction 0.1`, `--seed N`. Run `--help` on any mode for the full list.

Training output goes to:

| Mode | Default `--parent-dir` |
|--------|------------------------|
| `--mode raw_critics` | `models/metacritic_raw_wt/<run-name>/` |
| `--mode inference`   | `models/metacritic_oracle_wt/<run-name>/` |
| `--mode imitation`   | `models/imitation_wt/<run-name>/` |

Each script fails immediately if the run folder already exists — no silent overwrites.

Loss / training math → the source code § "Training".

### Oracle pre-step (Phase 9)

Required only for `--mode inference`. Runs the offline replay oracle and
augments each HDF5 with an `oracle_weights` dataset:

```bash
python3 src/vf_robot_controller/scripts/run_oracle.py \
    --in ~/CA-MCW/vf_data/vf_data_training \
    --out ~/CA-MCW/vf_data/vf_data_training   # in-place augmentation, idempotent
    --workers 4
```

Verify with `python3 scripts/phase9_acceptance.py` (gates 9.7-1, 9.7-2,
9.7-4 are coded; 9.7-3 / 9.7-5 are deferred to vf_robot_utils Phase E10).

---

## Inference (deploy a trained model)

Active model is selected by `vf_controller/model_defaults.py` (single
source of truth):

```python
DEFAULT_INFERENCE_TYPE    = "raw"            # "raw" | "oracle"
DEFAULT_INFERENCE_WEIGHTS = "raw_manual_v1"  # run folder under metacritic_<type>_wt/
DEFAULT_IMITATION_WEIGHTS = "manual_v1"      # run folder under imitation_wt/
```

Override at launch time:

```bash
# vf_inferencewt — oracle weights (auto-started sidecar)
ros2 launch vf_robot_bringup bringup_launch.py \
    controller:=vf_inferencewt  map:=house_my1_map  autostart_sidecar:=true \
    inference_model_type:=oracle  inference_weights:=oracle_manual_v1

# vf_imitationwt — switch run
ros2 launch vf_robot_bringup bringup_launch.py \
    controller:=vf_imitationwt  map:=house_my1_map  autostart_sidecar:=true \
    imitation_weights:=run2_2026_05_12

# Escape hatch — bypass folder resolution entirely
ros2 launch vf_robot_bringup bringup_launch.py \
    controller:=vf_inferencewt  map:=house_my1_map  autostart_sidecar:=true \
    onnx_path:=/abs/path/to/meta_critic.onnx
```

**Failure modes:**

- ONNX missing or features stale → meta-critic falls back silently to YAML
  `fixed_weights`; `/cmd_vel` never stops.
- Imitation sidecar prediction stale (no message within ~150 ms) →
  Python publishes zero twist rather than crashing controller_server.

---

## Batch CSV sweeps

For multi-controller × multi-planner benchmarks, this controller is driven
externally by `vf_robot_utils`. See
[`../vf_robot_utils/README.md`](../vf_robot_utils/README.md) for the recording
+ sweep pipeline.

---

## Where things live

| What | Path | Override |
|---|---|---|
| Manual collect sessions     | `vf_data/vf_data_training/manual/<map>/<goal>/<Planner>/<ctrl>/`     | `VF_DATA_ROOT=...` |
| Batch collect sessions      | `vf_data/vf_data_training/batch/<map>/<goal>/<Planner>/<ctrl>/`      | `VF_DATA_ROOT=...` |
| Inspection collect sessions | `vf_data/vf_data_training/inspection/<map>/<goal>/<Planner>/<ctrl>/` | (deferred) |
| Evaluated results           | `vf_data/vf_data_evaluation/batch/<map>/_aggregate/<csv_stem>/`      | `VF_DATA_ROOT=...` |
| ONNX models                | `src/vf_robot_controller/models/<family>/<run-name>/`        | `model_defaults.py` |
| Custom critic configs      | `../vf_robot_bringup/config/nav2/controllers/vf_*.yaml`      | — |

`vf_data/` is git-ignored at the workspace root. Each collect leaf contains
`session.json`, `manifest.csv`, `run_*.h5`. Manual sessions come from
`launch/vf_data_training/manual/vf_data_training_manual_fixedwt.launch.py`;
batch sessions from `vf_robot_utils`'s
`launch/vf_data_training/batch/vf_data_training_batch_fixedwt.launch.py`
(driven by `tour_runner`).

---

## Source layout

```
vf_robot_controller/
├── src/                       C++ libraries
│   ├── perception/            GCF, context, features, map backends, voxel filter
│   ├── meta_critic/           IWeightProvider + Fixed/Onnx/ImitationVelocity
│   ├── critics/               3 custom critics + 10 weighted-wrapper critics
│   ├── logging/               C++ helpers for episode metadata
│   └── controller/            VFController shim, mode dispatch
├── nodes/                     C++ ROS nodes (gcf, feature_extractor, context, map_backend)
├── vf_controller/             Python sub-package
│   ├── data_collection/       data_collector_node.py + episode_writer.py
│   ├── inference/             metacritic_inference_node.py, imitation_inference_node.py
│   └── training/              train_inference, train_imitation, train_raw_critics, oracle/
├── launch/
│   ├── core/                  perception, sidecar launches included by bringup
│   ├── vf_data_training/      vf_data_training_manual_fixedwt.launch.py + _legacy/
│   │   manual/                (manual mode — RViz-driven)
│   └── vf_navigation_*.launch.py   thin wrappers that re-export bringup_launch.py
├── scripts/                   train.py, run_oracle.py, phase9_acceptance.py, …
├── models/                    .onnx + .pt artefacts (binaries gitignored)
└── test/                      pytest + gtest
```

Detailed module-by-module breakdown → the source code.
