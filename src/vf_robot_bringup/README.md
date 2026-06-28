# vf_robot_bringup

Single launch entry point for ViroFighter navigation. Four orthogonal axes —
`robot:= controller:= localization:= planner:=` — compose into a complete Nav2 stack.

> Full architecture, parameter composition, robot profiles, and per-controller/planner YAML
> internals live in the package source.

---

## What this package does

```
bringup_launch.py
   ├── depth_to_scan        (vf_robot_slam)
   ├── localization backend (rtabmap_slam | rtabmap_loc | amcl | slam_toolbox)
   ├── Nav2 stack           (planner_server + controller_server + bt + behavior + smoother)
   ├── GCF perception       (vf_* controllers only — gcf_node + feature_extractor_node)
   ├── ONNX sidecar         (vf_inferencewt / vf_imitationwt — optional, autostart_sidecar:=true)
   └── RViz                 (gated by rviz:= and headless:=)
```

YAML composition at launch time:

```
nav2_base.yaml
  ⊕ controllers/<ctrl>.yaml   (controller fragment)
  ⊕ planners/<planner>.yaml   (planner fragment)
  ⊕ localization/amcl.yaml    (only for amcl / slam_toolbox)
  ⊕ robots/<robot>.yaml       (footprint, vel limits, per-family overrides)
  ──────────────────────────────────────────────────────
  /tmp/nav2_<robot>_<ctrl>_<planner>_<loc>.yaml  → passed to all Nav2 nodes
```

Merged by `vf_robot_bringup/launch_utils/compose_params.py`.

**This package does NOT:**
- Save HDF5 training data — use `vf_robot_controller/launch/collect/fixedwt_data.launch.py`
- Train models — use `vf_robot_controller/vf_controller/training/`

---

## Build

```bash
cd ~/CA-MCW
colcon build --packages-select vf_robot_bringup --symlink-install
source install/setup.bash
```

YAML and Python edits do not need rebuild with `--symlink-install`.

---

## Quick start

```bash
# T1 — Gazebo
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py
```

```bash
# T2 — bringup (NavFn planner, RTAB-Map localization, fixed weights)
ros2 launch vf_robot_bringup bringup_launch.py \
    controller:=vf_fixedwt  localization:=rtabmap_loc  map:=house_my1_map
```

Click **Nav2 Goal Pose** in RViz to drive.

---

## Arguments

```bash
ros2 launch vf_robot_bringup bringup_launch.py --show-args
```

| Argument | Choices / type | Default | Notes |
|---|---|---|---|
| `robot` | `virofighter`, `turtlebot3_waffle` | `virofighter` | Footprint, kinematics, vel limits |
| `controller` | see [Controllers](#controllers) | `vf_inferencewt` | Plugin loaded by `controller_server` |
| `localization` | `rtabmap_loc`, `rtabmap_slam`, `amcl`, `slam_toolbox` | `rtabmap_loc` | See [Localization modes](#localization-modes) |
| `planner` | `NavFn`, `SmacPlanner2D`, `SmacPlannerHybrid`, `SmacLattice`, `ThetaStar` | `NavFn` | Global planner plugin |
| `camera` | `d435i`, `d455`, `dual` | `dual` | depth_to_scan + RTAB-Map subscriptions |
| `scan_method` | `dimg`, `pc2scan` | `pc2scan` | `dimg` faster; D435i has < 1.1 m blind zone |
| `merge_scans` | `true`, `false` | `true` | Merge dual-camera scans into single `/scan` |
| `map` | bare name or abs path | `house_my1_map` | Bare: resolved under `$VF_MAPS_ROOT`; abs path: `dirname/basename` split |
| `new_map` | `true`, `false` | `true` | RTAB-Map SLAM: delete existing `.db` and start fresh |
| `use_sim_time` | `true`, `false` | `true` | `true` in Gazebo, `false` on real robot |
| `rviz` | `true`, `false` | `true` | Auto-disabled when `headless:=true` |
| `headless` | `true`, `false` | `false` | Suppress all bringup-owned GUIs |
| `autostart_sidecar` | `true`, `false` | `false` | Spawn ONNX sidecar Node directly from bringup |
| `inference_model_type` | `raw`, `oracle` | `raw` | Selects `metacritic_raw_wt/` or `metacritic_oracle_wt/` |
| `inference_weights` | run folder name | `raw_manual_v1` | Folder inside `metacritic_{type}_wt/` |
| `imitation_weights` | run folder name | `manual_v1` | Folder inside `imitation_wt/` |
| `onnx_path` | abs path | `""` | Escape hatch: bypasses folder resolution entirely |

Defaults for `inference_model_type`, `inference_weights`, and `imitation_weights` are read
from `vf_controller/model_defaults.py` — edit that file to change the active weights project-wide.

---

## Controllers

Seven controllers selectable via `controller:=`.

| `controller:=` | Plugin mode | Sidecar | Publishes `/vf/per_critic_costs` |
|---|---|---|---|
| `vf_fixedwt` | `fixedwt` | none | yes |
| `vf_inferencewt` | `inferencewt` | `metacritic_inference_node.py` | yes |
| `vf_imitationwt` | `imitationwt` | `imitation_inference_node.py` | no (MPPI doesn't run) |
| `mppi` | — | none | no |
| `dwb` | — | none | no |
| `rpp` | — | none | no |
| `graceful` | — | none | no |

Training data collection (`fixedwt_data.launch.py`) and LOO ablations are
launched from `vf_robot_controller/launch/` — not from bringup.

### ONNX sidecars

`vf_inferencewt` and `vf_imitationwt` require a Python sidecar.
Requires: `pip3 install onnxruntime` (system Python — no conda needed).

**Default weights** come from `vf_controller/model_defaults.py`. To use them with `autostart_sidecar:=true`:

```bash
# Inference — raw weights (default)
ros2 launch vf_robot_bringup bringup_launch.py \
    controller:=vf_inferencewt  map:=house_my1_map  autostart_sidecar:=true

# Inference — oracle weights
ros2 launch vf_robot_bringup bringup_launch.py \
    controller:=vf_inferencewt  map:=house_my1_map  autostart_sidecar:=true \
    inference_model_type:=oracle  inference_weights:=oracle_manual_v1

# Imitation — default weights
ros2 launch vf_robot_bringup bringup_launch.py \
    controller:=vf_imitationwt  map:=house_my1_map  autostart_sidecar:=true
```

**Manual start** (`autostart_sidecar:=false`, default) — bringup prints the `ros2 run` command:

```bash
# Then in a second terminal:
source ~/CA-MCW/install/setup.bash
ros2 run vf_robot_controller metacritic_inference_node.py --ros-args \
    -p onnx_path:=$HOME/CA-MCW/src/vf_robot_controller/models/metacritic_raw_wt/raw_manual_v1/meta_critic.onnx \
    -p norm_path:=$HOME/CA-MCW/src/vf_robot_controller/models/metacritic_raw_wt/raw_manual_v1/feature_norm.json
```

---

## Planners

| `planner:=` | Plugin class | Notes |
|---|---|---|
| `NavFn` | `nav2_navfn_planner/NavfnPlanner` | Default. Fast grid-based A*. |
| `SmacPlanner2D` | `nav2_smac_planner/SmacPlanner2D` | Smoother paths; slightly slower. |
| `SmacPlannerHybrid` | `nav2_smac_planner/SmacPlannerHybrid` | SE2 kinematic paths. ViroFighter: `min_turning_radius=0.05 m`. |
| `SmacLattice` | `nav2_smac_planner/SmacPlannerLattice` | Motion primitives. ViroFighter: `reverse_penalty=5.0`. |
| `ThetaStar` | `nav2_theta_star_planner/ThetaStarPlanner` | Any-angle shortcuts. Good for open areas. |

All register under the canonical instance name `"GridBased"` — the Nav2 BT XML never changes.

---

## Localization modes

### `rtabmap_slam` — build a new visual map

```bash
ros2 launch vf_robot_bringup bringup_launch.py \
    controller:=mppi  localization:=rtabmap_slam  map:=house_my1_map  new_map:=true
```

Drive around. Ctrl-C saves `$VF_MAPS_ROOT/house_my1_map/house_my1_map.db`.

### `rtabmap_loc` — localize in an existing `.db` (default)

```bash
ros2 launch vf_robot_bringup bringup_launch.py \
    controller:=vf_fixedwt  localization:=rtabmap_loc  map:=house_my1_map
```

`/map` publishes ~3 s after startup. Does not modify the `.db`.

### `amcl` — particle filter on a `.pgm`/`.yaml`

After launch: click **2D Pose Estimate** in RViz.

### `slam_toolbox` — 2D laser SLAM

Maps not auto-saved — serialise via:
```bash
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
  "filename: '$VF_MAPS_ROOT/house_my1_map/house_my1_map'"
```

---

## Diagnostics

```bash
ros2 topic hz /scan                        # ~30 Hz
ros2 topic hz /odom                        # ~50 Hz
ros2 topic hz /cmd_vel                     # ~20 Hz when goal active
ros2 topic hz /vf/features                 # ~20 Hz (vf_* controllers)
ros2 topic hz /vf/per_critic_costs         # ~20 Hz (vf_fixedwt / vf_inferencewt)
ros2 topic hz /vf_controller/meta_weights  # ~20 Hz (vf_inferencewt sidecar)

ros2 topic echo /vf/applied_weights        # live critic weight vector (all modes)
ros2 topic hz /vf/applied_weights          # should track controller_frequency (~20 Hz)
ros2 topic info /vf/applied_weights
```

If `/vf/applied_weights` and `/vf/per_critic_costs` run at ≪ 20 Hz, the
controller is starving (MPPI rollout cost > cycle budget). Symptoms include
"Control loop missed its desired rate" warnings and frequent BT recoveries.
See [Tuning notes](#tuning-notes) below.

If `controller:=vf_*` errors with `plugin not found`, rebuild:
```bash
colcon build --packages-select vf_robot_controller vf_robot_bringup --symlink-install
source install/setup.bash
```

---

## Tuning notes

### Custom BT XML

`config/nav2/bt_xml/navigate_to_pose_w_replanning_and_recovery_vf.xml` is
loaded in place of the upstream Nav2 default. Two differences:

- `RecoveryNode number_of_retries`: **6 → 12** (top-level NavigateRecovery).
  The slow MPPI cycle on a heavy ViroFighter through clutter routinely
  burns >6 strikes before completing long routes.
- `<Wait wait_duration="5"/>` removed from the RoundRobin RecoveryActions.
  The 5 s idle wasted the ProgressChecker movement window and re-tripped
  "stuck" on the very next attempt, snowballing into the abort cap.

`bt_navigator.ros__parameters.default_nav_to_pose_bt_xml` is rewritten at
launch time by `bringup_launch._compose_action` to the absolute path
under `share/vf_robot_bringup/config/nav2/bt_xml/`. The composed
`/tmp/nav2_*.yaml` and the `[vf_bringup]   bt_xml: ...` banner line confirm
the override is active.

### ProgressChecker (controller_server)

`config/nav2/nav2_base.yaml` — loosened from the TB3-class defaults:

| Key | Default | Current | Why |
|---|---|---|---|
| `required_movement_radius` | 0.5 m | **0.3 m** | ViroFighter `vx_max` is 0.30 m/s; 0.5 m in 10 s is unrealistic in clutter |
| `movement_time_allowance` | 10 s | **15 s** | Gives heavy MPPI cycles time to clear narrow gaps without "stuck" misfires |

### MPPI sizing (vf_fixedwt / vf_inferencewt)

`config/nav2/controllers/vf_fixedwt.yaml` — tuned so the controller can
hold its 20 Hz target on the workstation:

| Key | Was | Now |
|---|---|---|
| `batch_size` | 4000 | **2000** |
| `time_steps` | 56 | **40** |
| `visualize` | true | **false** |
| `publish_optimal_trajectory` | false | **true** (kept on for RViz inspection) |

If `ros2 topic hz /vf/applied_weights` still runs below ~10 Hz, the next
suspect is `VolumetricCritic` (8 000-point cloud iteration per cycle).

---

## Developer reference

| Path | Role |
|---|---|
| `launch/bringup_launch.py` | Entry point. OpaqueFunction compose + sidecar branches. |
| `launch/navigation_launch.py` | Nav2 stack include |
| `launch/localization_launch.py` | AMCL + map_server include |
| `launch/slam_launch.py` | SLAM Toolbox include |
| `config/nav2/nav2_base.yaml` | Nav2 skeleton; single NavFn default planner |
| `config/nav2/controllers/*.yaml` | Per-controller fragments (FollowPath block only) |
| `config/nav2/planners/*.yaml` | Per-planner fragments (GridBased block + planner_plugins) |
| `config/nav2/localization/*.yaml` | AMCL and SLAM Toolbox params |
| `config/nav2/bt_xml/*.xml` | Custom Nav2 behavior trees (path injected into bt_navigator at launch) |
| `config/robots/*.yaml` | Robot profile: rewrites + controller_overrides + planner_overrides |
| `vf_robot_bringup/launch_utils/compose_params.py` | Deep-merge pipeline with strict-mode validation |

Adding a new controller, planner, or robot: see the source code § "Adding axes".
