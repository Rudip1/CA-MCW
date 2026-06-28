<!--
Copyright  EUROKNOWS CO., LTD.
Licensed under the Apache License, Version 2.0 — see LICENSE.

Authors: Pravin Oli (pravin.oli.08@gmail.com, olipravin18@gmail.com)
Erasmus Mundus Joint Masters in Intelligent Field Robotics System (IFROS)
Universitat de Girona, Spain · Eötvös Loránd University, Hungary
-->

# `vf_robot_gazebo`

> ViroFighter UVC-1 — **Gazebo Classic 11 simulation package** for ROS 2 Humble.
> Hybrid `ament_cmake` + Python. URDF / meshes live in `vf_robot_description`
> (single source of truth) — this package only carries Gazebo worlds, models,
> launch files, and a small set of helper nodes / teleop scripts.

![demo](docs/vf_robot_gazebo.gif)

---

## Table of contents

1. [Quick start](#quick-start)
2. [Package layout](#package-layout)
3. [Environments and launch files](#environments-and-launch-files)
4. [Two pipelines: SDF vs Xacro](#two-pipelines-sdf-vs-xacro)
5. [Helper launches (shared)](#helper-launches-shared)
6. [Worlds, models, attributions](#worlds-models-attributions)
7. [Nodes & scripts](#nodes--scripts)
8. [`GAZEBO_*` environment variables](#gazebo_-environment-variables)
9. [TF tree](#tf-tree)
10. [Build & run](#build--run)
11. [Common arguments](#common-arguments)
12. [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
# 1. Install ROS deps once
sudo apt install \
  ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros \
  ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
  ros-humble-rqt-robot-steering ros-humble-rviz2 \
  ros-humble-xacro ros-humble-tf2-tools

# 2. Build
cd ~/CA-MCW
colcon build --packages-select vf_robot_description vf_robot_gazebo --symlink-install
source install/setup.bash

# 3. Launch any environment (xacro pipeline shown — sdf works the same way)
ros2 launch vf_robot_gazebo empty_world_xacro.launch.py
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py
ros2 launch vf_robot_gazebo hospital_euroknows_world_xacro.launch.py
ros2 launch vf_robot_gazebo hospital_my1_world_xacro.launch.py
ros2 launch vf_robot_gazebo hospital_my2_world_xacro.launch.py
ros2 launch vf_robot_gazebo hospital_Tommaso_hospital_world_xacro.launch.py
```

A Gazebo window with the ViroFighter and an `rqt_robot_steering` teleop GUI
should come up. Drive with the sliders.

---

## Package layout

```
vf_robot_gazebo/
├── CMakeLists.txt                         # ament_cmake + ament_cmake_python (hybrid)
├── package.xml                            # ament_cmake build type
├── setup.py                               # Python module install (no console_scripts)
├── docs/
│   └── vf_robot_gazebo.gif                # README demo
├── include/vf_robot_gazebo/
│   └── ultrasound.h
├── src/
│   └── ultrasound.cpp                     # C++ ultrasound aggregator
├── vf_robot_gazebo/                       # Python module (importable)
│   ├── __init__.py
│   └── scripts/
│       ├── teleop_twist_keyboard.py
│       └── ultrasound_py.py
├── launch/
│   ├── ARCHITECTURE.md                    # detailed pipeline notes
│   ├── vf_robot_state_publisher.launch.py # shared by both pipelines
│   ├── vf_spawn_xacro.launch.py           # spawn from /robot_description
│   ├── vf_spawn_sdf.launch.py             # spawn from model.sdf
│   ├── empty_world_launch/
│   │   ├── empty_world_sdf.launch.py
│   │   └── empty_world_xacro.launch.py
│   ├── hospital_euroknows_launch/
│   │   ├── hospital_euroknows_world_sdf.launch.py
│   │   └── hospital_euroknows_world_xacro.launch.py
│   ├── hospital_my1_launch/
│   │   ├── hospital_my1_world_sdf.launch.py
│   │   └── hospital_my1_world_xacro.launch.py
│   ├── hospital_my2_launch/
│   │   ├── hospital_my2_world_sdf.launch.py
│   │   └── hospital_my2_world_xacro.launch.py
│   ├── hospital_Tommaso_launch/
│   │   ├── hospital_Tommaso_hospital_world_sdf.launch.py
│   │   └── hospital_Tommaso_hospital_world_xacro.launch.py
│   └── house_my1_launch/
│       ├── house_my1_world_sdf.launch.py
│       └── house_my1_world_xacro.launch.py
├── models/
│   ├── uvc1_virofighter/                  # robot Gazebo model — model.sdf is AUTO-GENERATED
│   ├── uvc1_common/                       # shared meshes for model:// resolution
│   ├── hospital_euroknows_models/
│   ├── hospital_my1_models/
│   ├── hospital_my2_models/
│   ├── hospital_Tommaso_models/           # AWS RoboMaker hospital catalogue (see folder README)
│   └── house_my1_models/
├── worlds/
│   ├── empty_world.world
│   ├── hospital_euroknows_world/
│   ├── hospital_my1_world/
│   ├── hospital_my2_world/
│   ├── hospital_Tommaso_world/            # see folder README for upstream attribution
│   └── house_my1_world/
└── rviz/
    └── vf_robot_gazebo.rviz
```

> Top-level `worlds/*.world` files (`aisle.world`, `corridor*.world`,
> `pr1_world.world`, …) are kept around as historical/scratch assets but are
> **not referenced by any launch**. Likewise `models/uvc1_corridors*` and
> `models/uvc1_big_rect`. Don't rely on them in new launches.

---

## Environments and launch files

Each of the **six** environments below ships with **exactly two** launch
files: one for the SDF pipeline, one for the xacro pipeline. The two are
strictly parallel — same args, same nodes, only the robot-spawn path differs.

| Environment             | World file                                    | SDF launch                                          | Xacro launch                                          |
|-------------------------|-----------------------------------------------|-----------------------------------------------------|-------------------------------------------------------|
| empty                   | `worlds/empty_world.world`                    | `empty_world_sdf.launch.py`                         | `empty_world_xacro.launch.py`                         |
| hospital_euroknows      | `worlds/hospital_euroknows_world/…`           | `hospital_euroknows_world_sdf.launch.py`            | `hospital_euroknows_world_xacro.launch.py`            |
| hospital_my1            | `worlds/hospital_my1_world/…`                 | `hospital_my1_world_sdf.launch.py`                  | `hospital_my1_world_xacro.launch.py`                  |
| hospital_my2            | `worlds/hospital_my2_world/…`                 | `hospital_my2_world_sdf.launch.py`                  | `hospital_my2_world_xacro.launch.py`                  |
| hospital_Tommaso        | `worlds/hospital_Tommaso_world/hospital_Tommaso_hospital.world` | `hospital_Tommaso_hospital_world_sdf.launch.py`     | `hospital_Tommaso_hospital_world_xacro.launch.py`     |
| house_my1               | `worlds/house_my1_world/house_my1.world`      | `house_my1_world_sdf.launch.py`                     | `house_my1_world_xacro.launch.py`                     |

Common arguments: `x_pose`, `y_pose`, `z_pose`, `theta`, `use_sim_time`. See
[Common arguments](#common-arguments) below.

The Tommaso variant defaults to `y_pose:=2.0` and `z_pose:=0.2` to spawn
clear of obstacles and above the AWS triangulated floor mesh. The other five
envs default to `(0, 0, 0.1)` — change with `x_pose:=…` etc.

---

## Two pipelines: SDF vs Xacro

|                                | XACRO pipeline                              | SDF pipeline                              |
|--------------------------------|---------------------------------------------|-------------------------------------------|
| Robot description source       | `vf_robot_description/urdf/xacro/uvc1_virofighter.xacro` (processed at runtime) | `models/uvc1_virofighter/model.sdf` (pre-generated) |
| Spawn helper                   | `vf_spawn_xacro.launch.py`                  | `vf_spawn_sdf.launch.py`                  |
| Mesh URI scheme                | `package://vf_robot_description/meshes/…`   | `model://uvc1_common/meshes/…`            |
| Edit workflow                  | edit xacro → re-launch                      | edit xacro → run `xacro_to_sdf.sh` → re-launch |
| Startup time                   | slower (xacro processing)                   | faster                                    |
| Internal delays                | gzclient +3 s, spawn +5 s (race-condition fixes) | none                                      |
| Recommended for                | development, iteration                      | demos, headless, batch experiments        |

**Both pipelines** include `vf_robot_state_publisher.launch.py`, because the
TF tree (`base_footprint → base_link → wheels / sensors`) is always derived
from xacro — Gazebo's diff-drive plugin only publishes `odom → base_footprint`.

For the deeper rationale (timing delays, env-var pitfalls, spawn flow), see
[`launch/ARCHITECTURE.md`](launch/ARCHITECTURE.md).

---

## Helper launches (shared)

The three top-level helpers under `launch/` are included by every per-env
launcher; they are **not invoked directly**.

- `vf_robot_state_publisher.launch.py` — processes the canonical
  `uvc1_virofighter.xacro` via `xacro` and publishes `/robot_description` +
  TF for `base_link` and below.
- `vf_spawn_xacro.launch.py` — `spawn_entity.py -topic robot_description` at
  the requested `(x_pose, y_pose, z_pose, theta)`.
- `vf_spawn_sdf.launch.py` — `spawn_entity.py -file
  models/uvc1_virofighter/model.sdf` at the requested pose.

These helpers always point at the **canonical robot model** (single source
of truth). Per-environment customisation is done with launch arguments
(`y_pose`, `z_pose`, …), not by swapping models.

---

## Worlds, models, attributions

- `models/uvc1_virofighter/model.sdf` is **auto-generated** from
  `vf_robot_description/urdf/xacro/uvc1_virofighter.xacro` via
  `vf_robot_description/urdf/xacro/xacro_to_sdf.sh`. Never hand-edit it.
- `models/uvc1_common/` mirrors the meshes used by the SDF (resolved through
  `model://uvc1_common/meshes/…` in `model.sdf`).
- Per-world model catalogues live under `models/<env>_models/`. The
  per-launch `GAZEBO_MODEL_PATH` always lists, in this order:
  `models/<env>_models` → `models/` → `/usr/share/gazebo-11/models`.
- `models/hospital_Tommaso_models/` and `worlds/hospital_Tommaso_world/`
  carry their own `README.md` with upstream attribution
  ([Tommaso Vandermeer / Hospitalbot-Path-Planning](https://github.com/TommasoVandermeer/Hospitalbot-Path-Planning),
  itself based on the
  [AWS RoboMaker hospital world](https://github.com/aws-robotics/aws-robomaker-hospital-world)).
- The Tommaso world also includes 4 walking actors (`<actor>` blocks with
  built-in `<script>/<trajectory>`, no external plugin required).

---

## Nodes & scripts

| Executable                  | Source                                | Purpose                                     |
|-----------------------------|---------------------------------------|---------------------------------------------|
| `ultrasound_cpp`            | `src/ultrasound.cpp`                  | Aggregates 5 `sensor_msgs/Range` topics into one `vf_robot_messages/UltraSound`. |
| `ultrasound_py`             | `vf_robot_gazebo/scripts/ultrasound_py.py` | Same as above, Python implementation. Pick one. |
| `teleop_twist_keyboard`     | `vf_robot_gazebo/scripts/teleop_twist_keyboard.py` | Keyboard teleop fallback when `rqt_robot_steering` isn't desired. |

```bash
ros2 run vf_robot_gazebo ultrasound_cpp
ros2 run vf_robot_gazebo ultrasound_py
ros2 run vf_robot_gazebo teleop_twist_keyboard
```

---

## `GAZEBO_*` environment variables

Every per-env launch sets `GAZEBO_MODEL_PATH`, `GAZEBO_PLUGIN_PATH`, and
`GAZEBO_RESOURCE_PATH` via `SetEnvironmentVariable`. **Never replace —
always prepend**, otherwise Gazebo loses its built-in resources and
`gzserver` exits with code 255 / RTShaderSystem errors.

```python
SetEnvironmentVariable(
    name="GAZEBO_MODEL_PATH",
    value=os.pathsep.join([
        os.path.join(pkg_vf_gazebo, "models", "<env>_models"),  # env-specific
        os.path.join(pkg_vf_gazebo, "models"),                  # uvc1_common, uvc1_virofighter
        "/usr/share/gazebo-11/models",                          # Gazebo built-ins
    ]),
)
```

The xacro pipeline also prepends `os.path.dirname(pkg_desc)` so
`package://vf_robot_description/…` URIs resolve.

---

## TF tree

```
odom                                            (Gazebo diff_drive plugin)
└── base_footprint                              (Gazebo diff_drive plugin)
    └── base_link                               (robot_state_publisher, from xacro)
        ├── wheel_front_left_link
        ├── wheel_front_right_link
        ├── wheel_rear_left_link / wrist_rear_left_link
        ├── wheel_rear_right_link / wrist_rear_right_link
        ├── camera_d435i_link, camera_d455_link, fisheye_*_link
        ├── ultrasonic_*_link
        └── uvc_lights_link
```

Inspection commands:

```bash
ros2 run tf2_tools view_frames                  # full TF tree → frames.pdf
ros2 run tf2_ros tf2_echo odom base_footprint   # diff-drive output
ros2 run tf2_ros tf2_echo base_link camera_d435i_link
ros2 topic echo /robot_description --once       # confirm RSP published the URDF
```

---

## Build & run

```bash
# Build only this package and its description dependency
colcon build --packages-select vf_robot_description vf_robot_gazebo --symlink-install
source install/setup.bash

# Smoke test
ros2 pkg executables vf_robot_gazebo            # → teleop_twist_keyboard, ultrasound_cpp, ultrasound_py
ros2 launch vf_robot_gazebo empty_world_xacro.launch.py
```

---

## Common arguments

All twelve per-env launches accept the same args (defaults vary per env):

| Arg            | Type     | Default              | Notes |
|----------------|----------|----------------------|-------|
| `use_sim_time` | bool     | `true`               | Gazebo-clock slaving for downstream nodes. |
| `x_pose`       | float    | `0.0` (Tommaso: `0.0`, house_my1: `5.0`) | Robot spawn X. |
| `y_pose`       | float    | `0.0` (Tommaso: `2.0`, house_my1: `0.5`) | Robot spawn Y. |
| `z_pose`       | float    | `0.1` (Tommaso: `0.2`) | Robot spawn Z. Bump above 0.1 on triangulated mesh floors to avoid wheel-mesh penetration on spawn. |
| `theta`        | float    | `0.0` (house_my1: `3.14`) | Robot yaw at spawn (rad). |

Override on the command line:

```bash
ros2 launch vf_robot_gazebo hospital_Tommaso_hospital_world_sdf.launch.py \
  x_pose:=3.0 y_pose:=4.0 theta:=1.5708
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `gzserver` exits 255, RTShaderSystem error | A `SetEnvironmentVariable` somewhere replaced `GAZEBO_RESOURCE_PATH` instead of prepending. Check the launch you ran. |
| RViz shows no robot links / "no transform from X to Y" | `robot_state_publisher` failed. Check launch output and `ros2 topic echo /robot_description --once`. |
| No TF from `odom` to `base_footprint` | Robot didn't spawn — Gazebo plugin failed. Check `gzserver` terminal for plugin errors. |
| Sensors have no TF frames | `vf_robot_state_publisher.launch.py` not running, or xacro processing failed. |
| Wheels jitter / robot rotates on spawn in Tommaso | Triangulated AWS floor mesh — leave the world's existing physics tuning and floor-friction normalisation in place; raise `z_pose` if needed. |
| SDF pipeline shows old robot after xacro edit | Re-run `vf_robot_description/urdf/xacro/xacro_to_sdf.sh uvc1_virofighter.xacro` and rebuild. |
| `ros2 launch` can't find a launch file | The launch lives in a per-env subfolder under `launch/`; the install rule (`install(DIRECTORY launch …)`) recurses, so naming the file alone is enough — confirm you re-built and re-sourced after renaming/adding a launch. |
| `Could not find package vf_robot_messages` | Build it first: `colcon build --packages-select vf_robot_messages --symlink-install && source install/setup.bash`. |

---

## License & maintainer

Apache 2.0 — see [LICENSE](LICENSE).

**Maintainer:** Pravin Oli — `olipravin18@gmail.com`
Project: **CA-MCW** · package: `vf_robot_gazebo` · ROS 2 Humble · Gazebo Classic 11.
