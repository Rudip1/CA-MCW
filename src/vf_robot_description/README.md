<!--\nCopyright  EUROKNOWS CO., LTD.\n\nLicensed under the Apache License, Version 2.0 (the \"License\");\nyou may not use this file except in compliance with the License.\nYou may obtain a copy of the License at\n\n    http://www.apache.org/licenses/LICENSE-2.0\n\nUnless required by applicable law or agreed to in writing, software\ndistributed under the License is distributed on an \"AS IS\" BASIS,\nWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\nSee the License for the specific language governing permissions and\nlimitations under the License.\n\nAuthors: Pravin Oli (pravin.oli.08@gmail.com, olipravin18@gmail.com)\nCompany: EUROKNOWS CO., LTD.\nWebsite: https://www.euroknows.com/en/home/\n\nErasmus Mundus Joint Masters in Intelligent Field Robotics System (IFROS)\nhttps://ifrosmaster.org/\n\nUniversitat de Girona, Spain - https://www.udg.edu/en/\nEötvös Loránd University, Hungary - https://www.elte.hu/\n-->\n\n# `vf_robot_description`

> **ViroFighter UVC-1 Robot — URDF / Xacro Description Package**
> ROS 2 Humble · Hybrid C++ / Python · Gazebo Classic

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue?style=flat-square&logo=ros)](https://docs.ros.org/en/humble/)
[![Gazebo Classic](https://img.shields.io/badge/Gazebo-Classic%2011-orange?style=flat-square)](http://gazebosim.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](LICENSE)
[![Build](https://img.shields.io/badge/Build-ament__cmake-purple?style=flat-square)]()
[![Xacro](https://img.shields.io/badge/Model-Xacro%2FURDF-yellow?style=flat-square)]()
[![Meshes](https://img.shields.io/badge/Meshes-.dae%20%2F%20.stl-lightgrey?style=flat-square)]()

---

## Table of Contents

- [🌟 Overview](#-overview)
- [⚡ Quick Start](#-quick-start)
- [📁 Package Structure](#-package-structure)
- [🚀 Launch Files](#-launch-files)
- [🎯 RViz Configuration](#-rviz-configuration)
- [🗺️ TF Frame Guide](#️-tf-frame-guide)
- [📐 Robot Geometry](#-robot-geometry)
- [🌳 Kinematic Tree & TF Frames](#-kinematic-tree--tf-frames)
- [🔩 Links & Joints Reference](#-links--joints-reference)
- [📷 Sensors](#-sensors)
- [⚙️ Gazebo Plugins](#️-gazebo-plugins)
- [🗂️ Mesh Assets](#️-mesh-assets)
- [🔄 URDF / Xacro → SDF Converter](#-urdf--xacro--sdf-converter)
- [🐛 Troubleshooting](#-troubleshooting)
- [📄 License](#-license)

---

## 🌟 Overview

The **ViroFighter UVC-1** is a differential-drive ground robot designed for autonomous UVC disinfection. This package is the **single source of truth** for the robot's xacro/URDF model, meshes, and RViz configuration. Both `vf_robot_gazebo` and any other package in the workspace load the robot model from here — nothing is duplicated.

**What's inside:**

- 🏗️ Full kinematic model — base, 4-wheel drive, rear caster wrists
- 📷 Dual RealSense depth cameras (D455 + D435i)
- 👁️ Quad fisheye camera array (360° surround vision)
- 🔊 5× ultrasonic range sensors
- 💡 6× UVC light tubes
- 🎮 Gazebo Classic simulation plugins (diff drive, cameras, IMU, ultrasound)

![ViroFighter RViz Demo](docs/robot_rviz_demo.gif)

**Architecture overview:**

![URDF/Xacro → Gazebo flow](docs/urdf_xacro_gazebo_flow.svg)

---

## ⚡ Quick Start

### Prerequisites

```bash
sudo apt install \
  ros-humble-robot-state-publisher \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui \
  ros-humble-rviz2 \
  ros-humble-xacro
```

### Build

```bash
cd ~/CA-MCW
colcon build --packages-select vf_robot_description --symlink-install
source install/setup.bash
```

### Launch RViz (xacro — recommended)

```bash
ros2 launch vf_robot_description display_robot_xacro.launch.py
```

### Launch RViz (plain URDF)

```bash
ros2 launch vf_robot_description display_robot_urdf.launch.py
```

You should see the ViroFighter robot in RViz with joint slider controls.
Set **Fixed Frame** to `base_footprint` if RViz shows a transform warning.

---

## 📁 Package Structure

```
vf_robot_description/
├── 📂 blend/
│   └── virofighter_scripted_lowpoly_linux.blend
├── 📂 docs/
│   ├── frames_2026-03-20_14.53.50.gv   # TF tree Graphviz source
│   ├── robot_rviz_demo.gif              # RViz demo animation
│   ├── tf_frames-1.png                  # TF frame diagram
│   └── urdf_xacro_gazebo_flow.svg       # URDF/xacro→Gazebo architecture diagram
├── 📂 launch/
│   ├── display_robot_urdf.launch.py     # RViz via plain URDF
│   └── display_robot_xacro.launch.py    # RViz via xacro (recommended)
├── 📂 meshes/
│   ├── bases/
│   │   ├── virofighter_base.dae         # Main chassis mesh
│   │   ├── virofighter.dae
│   │   └── Cube.019.dae                 # Collision proxy mesh
│   ├── sensors/
│   │   ├── camera_d435i.dae
│   │   ├── camera_d455.stl
│   │   ├── camera_t265.dae
│   │   └── lds.stl
│   └── wheels/
│       ├── wheel_front_left.dae
│       ├── wheel_front_right.dae
│       ├── wheel_rear_left.dae
│       ├── wheel_rear_right.dae
│       ├── wrist_rear_left.dae          # Rear caster pivot
│       └── wrist_rear_right.dae
├── 📂 resource/
│   └── vf_robot_description             # ament resource marker
├── 📂 rviz/
│   └── model.rviz
├── 📂 urdf/
│   ├── 📂 urdf/                            # Plain URDF files
│   │   ├── common_properties.urdf
│   │   ├── uvc1_virofighter.urdf        # Full URDF with Gazebo plugins
│   │   ├── uvc1_virofighter_direct.urdf # Simplified URDF (no plugins)
│   │   └── urdf_to_sdf.sh               # URDF → SDF converter
│   └── 📂 xacro/                           # Xacro files (single source of truth)
│       ├── common_properties.xacro      # Shared material colours
│       ├── sensors.xacro                # All sensors + Gazebo sensor plugins
│       ├── uvc1_virofighter.xacro       # Root xacro (includes the two above)
│       └── xacro_to_sdf.sh              # Xacro → SDF converter
├── 📂 vf_robot_description/
│   └── __init__.py
├── CMakeLists.txt
├── LICENSE
├── package.xml
└── setup.py
```

> **Mesh path convention:** All URDF/xacro mesh references use
> `package://vf_robot_description/meshes/...`
> so both RViz and Gazebo resolve them through the ament index — no extra model paths needed.

---

## 🚀 Launch Files

### `display_robot_xacro.launch.py` — xacro (recommended)

Processes xacro at launch time so any edit to `.xacro` files is reflected immediately on the next launch without rebuilding (when using `--symlink-install`).

```bash
# Default — xacro + interactive joint sliders + RViz
ros2 launch vf_robot_description display_robot_xacro.launch.py

# Headless — no GUI, joints held at zero (use when Gazebo provides /joint_states)
ros2 launch vf_robot_description display_robot_xacro.launch.py use_gui:=false

# Custom xacro file
ros2 launch vf_robot_description display_robot_xacro.launch.py \
  xacro_file:=uvc1_virofighter.xacro
```

#### Launch arguments

| Argument | Default | Description |
|---|---|---|
| `xacro_file` | `uvc1_virofighter.xacro` | Filename inside `urdf/xacro/` |
| `use_gui` | `true` | Show joint slider GUI |
| `use_rviz` | `true` | Launch RViz2 |
| `rviz_config` | `rviz/model.rviz` | Full path to RViz config |

---

### `display_robot_urdf.launch.py` — plain URDF

```bash
# Default — plain URDF + interactive joint sliders + RViz
ros2 launch vf_robot_description display_robot_urdf.launch.py

# Use simplified URDF (no Gazebo plugin tags)
ros2 launch vf_robot_description display_robot_urdf.launch.py \
  urdf_file:=uvc1_virofighter_direct.urdf

# Headless
ros2 launch vf_robot_description display_robot_urdf.launch.py use_gui:=false
```

#### Launch arguments

| Argument | Default | Description |
|---|---|---|
| `urdf_file` | `uvc1_virofighter.urdf` | Filename inside `urdf/urdf/` |
| `use_gui` | `true` | Show joint slider GUI |
| `use_rviz` | `true` | Launch RViz2 |
| `rviz_config` | `rviz/model.rviz` | Full path to RViz config |

#### Nodes started (both launch files)

```
robot_state_publisher      ← reads URDF/xacro → publishes /tf + /tf_static
joint_state_publisher_gui  ← publishes /joint_states with GUI sliders
rviz2                      ← opens model.rviz config
```

---

## 🎯 RViz Configuration

The pre-configured `rviz/model.rviz` includes:

| Display | Settings |
|---|---|
| **Fixed Frame** | `base_footprint` |
| **RobotModel** | Description Topic: `/robot_description` |
| **TF** | All frames visible |
| **Grid** | Cell size 0.1 m, 10×10 |

**Fixed Frame selection guide:**

| Mode | Fixed Frame |
|---|---|
| RViz inspection only | `base_footprint` |
| Gazebo simulation running | `odom` |
| Full Nav2 stack | `map` |

---

## 🗺️ TF Frame Guide

```
[navigation]        [gazebo/hw]              [robot_state_publisher]
     │                   │                            │
     ▼                   ▼                            ▼
   map    ──────►   odom    ──────►   base_footprint ──► base_link ──► all links
              (amcl/slam)    (diff drive)        (fixed)        (RSP + joint_states)
```

### Who publishes what

| TF Transform | Publisher | When available |
|---|---|---|
| `map → odom` | `amcl` or `slam_toolbox` | Nav2 running + map loaded |
| `odom → base_footprint` | Gazebo diff drive plugin | Gazebo running |
| `base_footprint → base_link` | `robot_state_publisher` | Always — start this first |
| `base_link → wheel_*` | `robot_state_publisher` + `/joint_states` | Always |
| `base_link → camera_*` | `robot_state_publisher` (static) | Always |
| `base_link → ultrasonic_*` | `robot_state_publisher` (static) | Always |

### Verifying the TF tree

```bash
# Save TF tree to frames.pdf
ros2 run tf2_tools view_frames

# Check a specific transform
ros2 run tf2_ros tf2_echo base_link wheel_front_left_link

# List all static frames
ros2 topic echo /tf_static --once
```

---

## 📐 Robot Geometry

### Chassis Dimensions

| Parameter | Value |
|---|---|
| **Wheel separation (track width)** | `0.356 m` |
| **Wheel diameter (front)** | `0.165 m` (radius `0.08273 m`) |
| **Wheel diameter (rear)** | `0.101 m` (radius `0.05072 m`) |
| **Front wheel Y offset** | `±0.17827 m` from centre |
| **Front wheel Z height** | `0.082 m` above base_footprint |
| **Rear wrist X offset** | `-0.470 m` (rear of robot) |
| **Rear wrist Y offset** | `±0.16211 m` from centre |
| **Rear wrist Z height** | `0.130 m` above base_footprint |
| **Rear wheel drop** | `-0.080 m` below wrist pivot |
| **Base mesh origin offset** | `(-0.25, 0, 0.049)` from base_link |
| **Robot total height** | `~1.90 m` (UVC lights at top) |

### Drive Configuration

```
        FRONT (+X axis)
    ┌───────────┐
    │  [FL] [FR]│   ← front wheels: continuous joints, axis Y
    │           │      differential drive pair
    │           │
    │  [WL] [WR]│   ← rear wrists: continuous joints, axis Z (steering pivot)
    │  [RL] [RR]│   ← rear wheels: continuous joints, axis Y (passive casters)
    └───────────┘
        REAR (−X axis)
```

> **Drive type:** Differential drive using front wheels only.
> **Left = +Y axis, Right = −Y axis** (ROS REP-103 convention).
> **Turning:** Differential speed on `wheel_front_left` (+Y) and `wheel_front_right` (−Y).

---

## 🌳 Kinematic Tree & TF Frames

![TF Frame Tree](docs/tf_frames-1.png)

```
base_footprint                               ← RViz Fixed Frame
└── base_link  [fixed]                       ← Main chassis
    │
    ├── wheel_front_left_link                ← [continuous] axis: Y  | pos: (0, +0.178, 0.082)
    ├── wheel_front_right_link               ← [continuous] axis: Y  | pos: (0, −0.178, 0.082)
    │
    ├── wrist_rear_left_link                 ← [continuous] axis: Z  | pos: (−0.47, +0.162, 0.13)
    │   └── wheel_rear_left_link             ← [continuous] axis: Y  | pos: (+0.038, 0, −0.08) rel. wrist
    │
    ├── wrist_rear_right_link                ← [continuous] axis: Z  | pos: (−0.47, −0.162, 0.13)
    │   └── wheel_rear_right_link            ← [continuous] axis: Y  | pos: (+0.038, 0, −0.08) rel. wrist
    │
    ├── camera_d455_link  [fixed]            ← pos: (−0.525, 0, 0.429)  rpy: (0, 0, π)  rear-facing
    │   ├── camera_d455_color_optical_frame  [fixed]
    │   ├── camera_d455_depth_optical_frame  [fixed]
    │   └── camera_d455_imu_frame            [fixed]
    │
    ├── camera_d435i_link  [fixed]           ← pos: (0.045, 0, 1.773)  rpy: (0, π/3, 0)  tilted down
    │   ├── camera_d435i_color_optical_frame [fixed]
    │   ├── camera_d435i_depth_optical_frame [fixed]
    │   └── camera_d435i_imu_frame           [fixed]
    │
    ├── camera_fisheye_front_link  [fixed]   ← pos: (0.07,  +0.00, 1.845)
    ├── camera_fisheye_left_link   [fixed]   ← pos: (−0.2, +0.18, 1.845)
    ├── camera_fisheye_right_link  [fixed]   ← pos: (−0.2, −0.18, 1.845)
    ├── camera_fisheye_rear_link   [fixed]   ← pos: (−0.5,  +0.00, 1.845)
    │
    ├── ultrasonic_front_left_link  [fixed]  ← pos: (0.08, −0.129, 0.212)
    ├── ultrasonic_front_right_link [fixed]  ← pos: (0.08, +0.134, 0.212)
    ├── ultrasonic_side_left_link   [fixed]  ← pos: (−0.347, −0.215, 0.212)
    ├── ultrasonic_side_right_link  [fixed]  ← pos: (−0.347, +0.215, 0.212)
    ├── ultrasonic_rear_link        [fixed]  ← pos: (−0.562, 0.013, 0.331)
    │
    └── uvc_lights_link  [fixed]             ← pos: (−0.336, 0, 0.526)
        └── [6× UVC tube visuals embedded in single link]
```

> **Total links:** 30 · **Total joints:** 29 (6 continuous + 23 fixed)

---

## 🔩 Links & Joints Reference

### Movable Joints

| Joint Name | Type | Parent → Child | Axis | Position (xyz) |
|---|---|---|---|---|
| `wheel_front_left_joint` | continuous | `base_link` → `wheel_front_left_link` | Y | `(0, +0.178, 0.082)` |
| `wheel_front_right_joint` | continuous | `base_link` → `wheel_front_right_link` | Y | `(0, −0.178, 0.082)` |
| `wrist_rear_left_joint` | continuous | `base_link` → `wrist_rear_left_link` | Z | `(−0.47, +0.162, 0.13)` |
| `wheel_rear_left_joint` | continuous | `wrist_rear_left_link` → `wheel_rear_left_link` | Y | `(0.038, 0, −0.08)` |
| `wrist_rear_right_joint` | continuous | `base_link` → `wrist_rear_right_link` | Z | `(−0.47, −0.162, 0.13)` |
| `wheel_rear_right_joint` | continuous | `wrist_rear_right_link` → `wheel_rear_right_link` | Y | `(0.038, 0, −0.08)` |

### Link Inertia Summary

| Link | Mass (kg) | Notes |
|---|---|---|
| `base_link` | `10.011` | Main chassis |
| `wheel_front_left_link` | `7.903` | Heavy front wheels |
| `wheel_front_right_link` | `7.905` | Heavy front wheels |
| `wrist_rear_left_link` | `2.000` | Caster pivot |
| `wheel_rear_left_link` | `2.210` | Smaller rear wheels |
| `wrist_rear_right_link` | `2.000` | Caster pivot |
| `wheel_rear_right_link` | `2.210` | Smaller rear wheels |
| `camera_d455_link` | `0.159` | RealSense D455 |
| `camera_d435i_link` | `0.072` | RealSense D435i |
| `uvc_lights_link` | `0.900` | 6× UVC tube array |
| Fisheye cameras (×4) | `0.010` each | Sphere marker links |
| Ultrasonic sensors (×5) | `0.010` each | Sphere marker links |

---

## 📷 Sensors

### 1. Intel RealSense D455 — Rear-Facing Depth Camera

```
Position:  (−0.525, 0.000, 0.429) relative to base_link
Rotation:  (0, 0, π)  ← mounted facing REAR of robot
Geometry:  Box 29×124×25 mm
```

| Topic | Type | Rate |
|---|---|---|
| `/d455/color/image_raw` | `sensor_msgs/Image` | 30 Hz |
| `/d455/color/camera_info` | `sensor_msgs/CameraInfo` | 30 Hz |
| `/d455/depth/image_raw` | `sensor_msgs/Image` | 30 Hz |
| `/d455/depth/points` | `sensor_msgs/PointCloud2` | 30 Hz |
| `/d455/imu/data_raw` | `sensor_msgs/Imu` | 200 Hz |

| Camera Param | RGB | Depth |
|---|---|---|
| Resolution (real hardware) | 1280×800 | 1280×720 |
| Resolution (Gazebo sim) | 1280×800 | **424×240** (reduced for perf — see note) |
| H-FOV | 85.9° (1.5 rad) | 87.1° (1.52 rad) |
| Near clip | 0.05 m | 0.60 m |
| Far clip | 20.0 m | 6.0 m |

> **Gazebo depth resolution note:** The depth sensor in `sensors.xacro` is set to 424×240
> instead of 1280×720. At full resolution, Gazebo delivers ~1.5 Hz actual depth framerate,
> causing pointcloud timestamps to be 150+ seconds stale and making `pc2scan` TF lookups
> fail silently. 424×240 matches the RealSense D455 low-resolution hardware mode and
> delivers 15–30 Hz in Gazebo. RTAB-Map visual SLAM and Nav2 obstacle detection are
> unaffected (SLAM uses RGB; Nav2 uses the derived laserscan, not depth directly).

---

### 2. Intel RealSense D435i — Top-Mounted Depth Camera

```
Position:  (0.045, 0.000, 1.773) relative to base_link
Rotation:  (0, π/3, 0)  ← tilted 60° downward
Mesh:      meshes/sensors/camera_d435i.dae (scale 0.001)
```

| Topic | Type | Rate |
|---|---|---|
| `/d435i/color/image_raw` | `sensor_msgs/Image` | 30 Hz |
| `/d435i/color/camera_info` | `sensor_msgs/CameraInfo` | 30 Hz |
| `/d435i/depth/image_raw` | `sensor_msgs/Image` | 30 Hz |
| `/d435i/depth/points` | `sensor_msgs/PointCloud2` | 30 Hz |
| `/d435i/imu/data_raw` | `sensor_msgs/Imu` | 200 Hz |

---

### 3. Fisheye Camera Array — 360° Surround Vision

Four fisheye cameras mounted at `z = 1.845 m`:

```
  camera_fisheye_front  →  ( 0.07,  0.00, 1.845)  rpy: (0,  0.628, −0.026)
  camera_fisheye_left   →  (−0.20, +0.18, 1.845)  rpy: (0,  0.628, +π/2)
  camera_fisheye_right  →  (−0.20, −0.18, 1.845)  rpy: (0,  0.628, −π/2)
  camera_fisheye_rear   →  (−0.50,  0.00, 1.845)  rpy: (π,  2.513,  0)
```

| Camera Param | Value |
|---|---|
| H-FOV | 170° (2.967 rad) |
| Resolution | 640×480 |
| Rate | 60 Hz |
| Lens model | Custom polynomial (c1=1.05, c2=4, fun=tan) |

| Topic pattern | Description |
|---|---|
| `/fisheye/{front,left,right,rear}/image_raw` | Raw fisheye image |
| `/fisheye/{front,left,right,rear}/camera_info` | Camera calibration |

---

### 4. Ultrasonic Range Sensors — 5× Perimeter

All sensors: `sensor_msgs/Range`, FOV ±14.3°, range 0.01–0.75 m, rate 5 Hz.

| Sensor | Position (xyz) | Heading | Topic |
|---|---|---|---|
| `ultrasonic_front_left` | `(0.08, −0.129, 0.212)` | Forward | `/ultrasound/front_left` |
| `ultrasonic_front_right` | `(0.08, +0.134, 0.212)` | Forward | `/ultrasound/front_right` |
| `ultrasonic_side_left` | `(−0.347, −0.215, 0.212)` | Left (−90°) | `/ultrasound/left` |
| `ultrasonic_side_right` | `(−0.347, +0.215, 0.212)` | Right (+90°) | `/ultrasound/right` |
| `ultrasonic_rear` | `(−0.562, +0.013, 0.331)` | Rearward (180°) | `/ultrasound/rear` |

---

### 5. UVC Light Array — 6× Tubes

Six UVC emitter tubes modelled as blue cylinders within `uvc_lights_link`:

```
Parent joint offset:  (−0.336, 0, 0.526) from base_link
Tube radius: 0.01004 m  |  Tube length: 1.327 m

  light_1: (−0.104, −0.072, 0.640)    light_2: (−0.104, +0.073, 0.640)
  light_3: (+0.094, +0.138, 0.640)    light_4: (−0.023, +0.138, 0.640)
  light_5: (+0.094, −0.141, 0.640)    light_6: (−0.023, −0.141, 0.640)
```

---

## ⚙️ Gazebo Plugins

| Plugin | Library | Purpose |
|---|---|---|
| Differential Drive | `libgazebo_ros_diff_drive.so` | `/cmd_vel` → wheels, publishes `odom` TF |
| Joint State Publisher | `libgazebo_ros_joint_state_publisher.so` | Publishes `/joint_states` for all 6 joints |
| RGB Camera (D455) | `libgazebo_ros_camera.so` | RGB stream from rear D455 |
| Depth Camera (D455) | `libgazebo_ros_camera.so` | Depth + point cloud from rear D455 |
| IMU (D455) | `libgazebo_ros_imu_sensor.so` | IMU from D455, noise σ=0.0005 |
| RGB Camera (D435i) | `libgazebo_ros_camera.so` | RGB stream from top D435i |
| Depth Camera (D435i) | `libgazebo_ros_camera.so` | Depth + point cloud from top D435i |
| IMU (D435i) | `libgazebo_ros_imu_sensor.so` | IMU from D435i, noise σ=0.0005 |
| Fisheye ×4 | `libgazebo_ros_camera.so` | 170° wide-angle cameras |
| Ultrasonic ×5 | `libgazebo_ros_ray_sensor.so` | Ray-cast sonar sensors |

### Differential Drive Parameters

```xml
wheel_separation:  0.356 m
wheel_diameter:    0.165 m
max_wheel_torque:  20 Nm
update_rate:       100 Hz
left_joint:        wheel_front_left_joint    ← +Y side (ROS REP-103 left)
right_joint:       wheel_front_right_joint   ← −Y side (ROS REP-103 right)
robot_base_frame:  base_footprint
```

---

## 🗂️ Mesh Assets

All meshes live under `meshes/` and are referenced as
`package://vf_robot_description/meshes/...`

| File | Format | Used By | Scale in URDF |
|---|---|---|---|
| `bases/virofighter_base.dae` | Collada | `base_link` visual | `1.0 1.0 1.0` |
| `bases/Cube.019.dae` | Collada | `base_link` collision | `1.0 1.0 1.0` |
| `wheels/wheel_front_left.dae` | Collada | `wheel_front_left_link` | `0.001 0.001 0.001` |
| `wheels/wheel_front_right.dae` | Collada | `wheel_front_right_link` | `0.001 0.001 0.001` |
| `wheels/wheel_rear_left.dae` | Collada | `wheel_rear_left_link` | `0.001 0.001 0.001` |
| `wheels/wheel_rear_right.dae` | Collada | `wheel_rear_right_link` | `0.001 0.001 0.001` |
| `wheels/wrist_rear_left.dae` | Collada | `wrist_rear_left_link` | `0.001 0.001 0.001` |
| `wheels/wrist_rear_right.dae` | Collada | `wrist_rear_right_link` | `0.001 0.001 0.001` |
| `sensors/camera_d435i.dae` | Collada | `camera_d435i_link` | `0.001 0.001 0.001` |

> Wheel and sensor meshes are scaled `0.001` because they were exported from Blender in **millimetres**. The base mesh is in **metres** (scale `1.0`).

---

## 🔄 URDF / Xacro → SDF Converter

Two converter scripts live alongside their respective model files. Run either whenever you update the model and need to regenerate `model.sdf` for Gazebo.

### `urdf/xacro/xacro_to_sdf.sh` — xacro (recommended)

```bash
cd ~/CA-MCW/src/vf_robot_description/urdf/xacro
chmod +x xacro_to_sdf.sh   # once only
./xacro_to_sdf.sh uvc1_virofighter.xacro
```

### `urdf/urdf/urdf_to_sdf.sh` — plain URDF

```bash
cd ~/CA-MCW/src/vf_robot_description/urdf/urdf
chmod +x urdf_to_sdf.sh    # once only
./urdf_to_sdf.sh uvc1_virofighter.urdf
```

Both scripts save to the same target:
```
~/CA-MCW/src/vf_robot_gazebo/models/uvc1_virofighter/model.sdf
```

### What the scripts do

| Step | Action |
|---|---|
| 1 | Resolves xacro macros (xacro_to_sdf always; urdf_to_sdf if `xmlns:xacro` detected) |
| 2 | Validates with `check_urdf` |
| 3 | Converts resolved URDF → raw SDF via `gz sdf -p` |
| 4a | Fixes mesh URIs — rewrites to `model://uvc1_common/meshes/…` |
| 4b | **Auto-extracts** all `<gazebo>` plugin tags and re-injects into SDF |
| 4c | Adds `<?xml version="1.0"?>` declaration |
| 5 | Backs up old `model.sdf` → `model.sdf.bak`, saves new file |
| 6 | Shows a colour diff of what changed |

> `gz sdf -p` strips all `<gazebo>` plugin tags and corrupts `model://` URIs during conversion. Both scripts repair this automatically — never edit `model.sdf` manually.

---

## 🐛 Troubleshooting

### RViz shows "No transform from X to Y"

Match Fixed Frame to what is running:

| Running | Fixed Frame |
|---|---|
| `display_robot_*.launch.py` only | `base_footprint` |
| Gazebo simulation | `odom` |
| Full Nav2 | `map` |

---

### Meshes not visible in RViz

Check that URDF mesh paths use the correct package name:

```bash
grep "package://" urdf/urdf/uvc1_virofighter.urdf | head -3
# Should show: package://vf_robot_description/meshes/...
```

Fix if needed:

```bash
sed -i 's|package://uvc1_common/meshes/|package://vf_robot_description/meshes/|g' \
  urdf/urdf/uvc1_virofighter.urdf \
  urdf/urdf/uvc1_virofighter_direct.urdf
```

---

### Joint sliders don't move the robot

```bash
ros2 topic hz /joint_states   # should show ~10 Hz
ros2 topic hz /tf             # should show ~50 Hz
```

---

### Launch fails with YAML parse error on robot_description

The `ParameterValue` wrapper is required on ROS 2 Humble. Both launch files include it:

- `display_robot_xacro.launch.py` wraps the `xacro` command output
- `display_robot_urdf.launch.py` wraps the `cat` command output

```python
from launch_ros.descriptions import ParameterValue
robot_description_content = ParameterValue(
    Command([FindExecutable(name='xacro'), ' ', xacro_file_path]),
    value_type=str,
)
```

---

### Build fails: `must contain exactly one <name> tag`

```bash
sed -i 's|<n>vf_robot_description</n>|<name>vf_robot_description</name>|' \
  package.xml
```

---

### Build fails: `doesn't contain an __init__.py`

```bash
mkdir -p vf_robot_description
touch vf_robot_description/__init__.py
```

---

### Overriding old installed version

```bash
colcon build --packages-select vf_robot_description \
  --symlink-install \
  --allow-overriding vf_robot_description
```

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE)

---

## 👤 Maintainer

**Pravin Oli** — olipravin18@gmail.com · roboticsmaterials@gmail.com
Project: **CA-MCW** | Package: `vf_robot_description`
