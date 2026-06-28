<!--
Copyright  EUROKNOWS CO., LTD.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Authors: Pravin Oli (pravin.oli.08@gmail.com, olipravin18@gmail.com)
Company: EUROKNOWS CO., LTD.
Website: https://www.euroknows.com/en/home/

Erasmus Mundus Joint Masters in Intelligent Field Robotics System (IFROS)
https://ifrosmaster.org/

Universitat de Girona, Spain - https://www.udg.edu/en/
Eötvös Loránd University, Hungary - https://www.elte.hu/
-->

# 🤖 `vf_robot_slam`

> **ViroFighter UVC-1 Robot — SLAM & Localization Package**
> ROS 2 Humble · RTAB-Map · Intel RealSense D435i + D455

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue?style=flat-square&logo=ros)](https://docs.ros.org/en/humble/)
[![RTAB-Map](https://img.shields.io/badge/RTAB--Map-Visual%20SLAM-orange?style=flat-square)](http://introlab.github.io/rtabmap/)
[![Nav2](https://img.shields.io/badge/Nav2-Compatible-green?style=flat-square)](https://navigation.ros.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](LICENSE)

---

## 📋 Table of Contents

**For Users (Operators)**

1. [Overview](#-overview)
2. [Prerequisites & Build](#-prerequisites--build)
3. [Quick Start — SLAM in 4 Terminals](#-quick-start--slam-in-4-terminals)
4. [Complete Command Reference — SLAM](#-complete-command-reference--slam)
5. [Complete Command Reference — Localization](#-complete-command-reference--localization)
6. [Complete Command Reference — Depth to LaserScan](#-complete-command-reference--depth-to-laserscan)
7. [Map Management — Save, Export, Reuse](#-map-management--save-export-reuse)
8. [Operating Modes (4 Modes)](#️-operating-modes-4-modes)
9. [Validation Matrix — Who Provides What](#-validation-matrix--who-provides-what)
10. [Verification & Diagnostics](#-verification--diagnostics)
11. [Troubleshooting](#-troubleshooting)

**For Developers**

12. [Package Structure](#-package-structure)
13. [Architecture — Modular Launch Design](#️-architecture--modular-launch-design)
14. [Camera Configuration — D435i vs D455](#-camera-configuration--d435i-vs-d455)
15. [Two Depth-to-Scan Methods — dimg vs pc2scan](#-two-depth-to-scan-methods--dimg-vs-pc2scan)
16. [Custom Nodes — scan_merger.py & pc_to_scan.py](#-custom-nodes--scan_mergerpy--pc_to_scanpy)
17. [Dual Camera Scan Merging](#-dual-camera-scan-merging)
18. [Launch Arguments Reference](#️-launch-arguments-reference)
19. [Topics & TF Published](#-topics--tf-published)
20. [Nav2 Costmap Configuration](#-nav2-costmap-configuration)
21. [Hard-Won Lessons](#-hard-won-lessons)
22. [Dependencies](#-dependencies)

---

# FOR USERS (OPERATORS)

---

## 🌟 Overview

`vf_robot_slam` provides all SLAM, localization, and depth-to-laserscan capabilities for the ViroFighter UVC-1 robot. The robot has **no 2D lidar** — it relies entirely on Intel RealSense depth cameras for perception, making visual SLAM the core approach.

### Where this package fits in the system

The CA-MCW project is split into modular packages. Each owns a specific concern:

| Package | Status | Responsibility | Key outputs |
|---------|--------|---------------|-------------|
| `vf_robot_description` | ✅ Done | URDF/xacro, sensor frames, robot model | `/robot_description`, all static TF (`base_link → sensors`) |
| `vf_robot_gazebo` | ✅ Done | Gazebo simulation, sensor plugins, world files | `/odom`, `odom→base_footprint` TF, camera topics, `/clock` |
| **`vf_robot_slam`** | **✅ Done** | **SLAM, localization, depth-to-scan** | **`/map`, `map→odom` TF, `/scan`** |
| `vf_robot_bringup` | ✅ Done | Planners, controllers, costmaps, Nav2 bringup | Full Nav2 stack (all 4 modes) |

### What this package does

| Function | Launch File | What it produces |
|----------|------------|-----------------|
| Build a new map | `rtabmap_slam.launch.py` | `/map` topic, `map→odom` TF, `.db` database |
| Navigate in an existing map | `rtabmap_loc.launch.py` | `/map` topic, `map→odom` TF from loaded `.db` |
| Convert depth to LaserScan | `depth_to_scan.launch.py` | `/scan` topic for Nav2 / AMCL / SLAM Toolbox |

### Camera options for every launch

| Value | Camera(s) Used | Coverage |
|-------|---------------|----------|
| `camera:=d435i` | Front D435i only | ~87° front arc |
| `camera:=d455` | Rear D455 only | ~87° rear arc |
| `camera:=dual` | Both cameras | ~174° combined (front + rear) |

### The 4 operating modes at a glance

This package supports 4 mutually exclusive operating modes. Each mode has exactly ONE `/map` publisher and ONE `map→odom` TF publisher — never run two simultaneously.

| Mode | Map source | `map→odom` TF source | `/scan` source | Map file needed |
|------|-----------|---------------------|---------------|----------------|
| **1. RTAB-Map SLAM** | RTAB-Map (builds live) | RTAB-Map | `depth_to_scan` | None (creates `.db`) |
| **2. RTAB-Map Loc** | RTAB-Map (from `.db`) | RTAB-Map | `depth_to_scan` | `.db` from Mode 1 |
| **3. AMCL** | `map_server` (from `.pgm`) | AMCL | `depth_to_scan` | `.pgm` + `.yaml` from Mode 1 |
| **4. SLAM Toolbox** | SLAM Toolbox (builds live) | SLAM Toolbox | `depth_to_scan` | None (or serialized map) |

> See [Operating Modes](#️-operating-modes-4-modes) for full terminal commands and [Validation Matrix](#-validation-matrix--who-provides-what) for the complete cross-check table.

---

## ⚡ Prerequisites & Build

### Install dependencies

```bash
sudo apt install ros-humble-rtabmap-ros \
                 ros-humble-depthimage-to-laserscan \
                 ros-humble-topic-tools
```

### Build

```bash
cd ~/CA-MCW
colcon build --packages-select vf_robot_slam --symlink-install
source install/setup.bash
```

---

## 🚀 Quick Start — SLAM in 4 Terminals

The fastest way to build a map in Gazebo:

```bash
# Terminal 1: Gazebo simulation
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py

# Terminal 2: RTAB-Map SLAM (builds the map)
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=dual map_name:=house_my1_map

# Terminal 3: Depth to LaserScan (for Nav2 costmap — run alongside SLAM or localization)
ros2 launch vf_robot_slam depth_to_scan.launch.py camera:=dual

# Terminal 4: Drive the robot
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

**What happens:**
1. Gazebo starts the simulated world with the ViroFighter robot
2. RTAB-Map starts building a 3D map — the `map` TF frame appears after ~3–10 seconds
3. Depth-to-scan produces `/scan` for Nav2 costmaps
4. You drive the robot to explore the environment

**When done mapping:** press Ctrl+C in Terminal 2 — `house_my1_map.db` saves automatically.

---

## 🗺️ Complete Command Reference — SLAM

RTAB-Map SLAM builds a new map. Every command below assumes Gazebo is already running.

### Dual camera (recommended)

```bash
# New map — deletes any existing house_my1_map.db and starts fresh
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=dual map_name:=house_my1_map

# Continue an existing map — keeps existing house_my1_map.db and adds to it
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=dual map_name:=house_my1_map new_map:=false

# No RViz (headless / SSH)
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=dual map_name:=house_my1_map rviz:=false

# Real robot (not Gazebo)
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=dual map_name:=house_my1_map use_sim_time:=false
```

### D435i only (front camera)

```bash
# New map
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=d435i map_name:=house_my1_map

# Continue existing
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=d435i map_name:=house_my1_map new_map:=false

# Real robot
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=d435i map_name:=house_my1_map use_sim_time:=false
```

### D455 only (rear camera)

```bash
# New map
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=d455 map_name:=house_my1_map

# Continue existing
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=d455 map_name:=house_my1_map new_map:=false

# Real robot
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=d455 map_name:=house_my1_map use_sim_time:=false
```

### Custom maps directory

```bash
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=dual map_name:=warehouse maps_dir:=~/my_maps
# Saves to: ~/my_maps/warehouse/warehouse.db
```

### What SLAM produces

| Output | Location | Created when |
|--------|----------|-------------|
| `house_my1_map.db` | `~/CA-MCW/maps/house_my1_map/` | Automatically on Ctrl+C |
| `house_my1_map.pgm` | `~/CA-MCW/maps/house_my1_map/` | Manually (see Map Management) |
| `house_my1_map.yaml` | `~/CA-MCW/maps/house_my1_map/` | Manually (see Map Management) |
| `/map` topic | ROS network | While SLAM is running |
| `map→odom` TF | ROS network | While SLAM is running |

---

## 📍 Complete Command Reference — Localization

RTAB-Map Localization navigates within a previously built map. **Requires a `.db` file** from a prior SLAM session.

### Dual camera (recommended)

```bash
# Gazebo
ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=dual map_name:=house_my1_map

# No RViz
ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=dual map_name:=house_my1_map rviz:=false

# Real robot
ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=dual map_name:=house_my1_map use_sim_time:=false
```

### D435i only

```bash
ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=d435i map_name:=house_my1_map
ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=d435i map_name:=house_my1_map use_sim_time:=false
```

### D455 only

```bash
ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=d455 map_name:=house_my1_map
ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=d455 map_name:=house_my1_map use_sim_time:=false
```

### Custom maps directory

```bash
ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=dual map_name:=warehouse maps_dir:=~/my_maps
# Loads from: ~/my_maps/warehouse/warehouse.db
```

### Error: map database not found

If the `.db` file doesn't exist, the launch prints an error:

```
ERROR: Map database not found!
Expected:  /home/pravin/CA-MCW/maps/house_my1_map/house_my1_map.db

Build the map first:
  ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=dual map_name:=house_my1_map
```

---

## 🔄 Complete Command Reference — Depth to LaserScan

Converts depth camera data into `/scan` (LaserScan) for Nav2, AMCL, or SLAM Toolbox. Run this **alongside** SLAM or Localization.

### Two methods available

| Method | Best for | Speed in Gazebo | D435i floor handling |
|--------|----------|----------------|---------------------|
| `method:=dimg` | Gazebo simulation | 15–30 Hz ✅ | `range_min=1.1 m` hack (blind zone < 1.1 m) |
| `method:=pc2scan` | Real robot | 15–30 Hz ✅ (real), ~2–3 Hz (Gazebo) | World-space height filter (no blind zone) ✅ |

### method:=dimg — Dual camera

```bash
# Gazebo (recommended for simulation)
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=dimg camera:=dual

# Dual without merging (Nav2 multi-source costmap instead)
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=dimg camera:=dual merge_scans:=false
# Publishes /scan_d435i + /scan_d455 only — no /scan

# Real robot
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=dimg camera:=dual use_sim_time:=false
```

### method:=dimg — Single camera

```bash
# D455 only (rear, horizontal — best single-camera option for obstacles)
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=dimg camera:=d455

# D435i only (front, 60° tilt — has 1.1 m blind zone with dimg method)
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=dimg camera:=d435i

# Real robot
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=dimg camera:=d455 use_sim_time:=false
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=dimg camera:=d435i use_sim_time:=false
```

### method:=pc2scan — Dual camera

```bash
# Real robot (recommended for production)
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=pc2scan camera:=dual use_sim_time:=false

# Gazebo (works but slow ~2–3 Hz)
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=pc2scan camera:=dual

# Dual without merging
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=pc2scan camera:=dual merge_scans:=false
```

### method:=pc2scan — Single camera

```bash
# D455 only
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=pc2scan camera:=d455 use_sim_time:=false

# D435i only (no blind zone — pc2scan handles 60° tilt correctly)
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=pc2scan camera:=d435i use_sim_time:=false

# Gazebo
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=pc2scan camera:=d455
ros2 launch vf_robot_slam depth_to_scan.launch.py method:=pc2scan camera:=d435i
```

### What depth_to_scan produces

| Camera setting | merge_scans | Topics published |
|---------------|-------------|-----------------|
| `camera:=d435i` | n/a | `/scan` |
| `camera:=d455` | n/a | `/scan` |
| `camera:=dual` | `true` (default) | `/scan_d435i` + `/scan_d455` + `/scan` (merged) |
| `camera:=dual` | `false` | `/scan_d435i` + `/scan_d455` only |

> **Note:** When `merge_scans:=false`, there is no `/scan` topic. AMCL requires `/scan`, so use `merge_scans:=true` or configure Nav2 with multi-source costmap (see Developer section).

---

## 💾 Map Management — Save, Export, Reuse

### Map storage structure

All maps are stored **outside** the ROS package at workspace level:

```
~/CA-MCW/
├── src/vf_robot_slam/              # Package source (no maps here)
└── maps/                           # Auto-created by launch files
    ├── house_my1_map/
    │   ├── house_my1_map.db            # RTAB-Map database (auto-saved on Ctrl+C)
    │   ├── house_my1_map.pgm           # 2D occupancy grid (saved manually)
    │   └── house_my1_map.yaml          # Map metadata (saved manually)
    └── warehouse/
        ├── warehouse.db
        ├── warehouse.pgm
        └── warehouse.yaml
```

### Map file types

| File | Size | Used By | Created How |
|------|------|---------|-------------|
| `.db` | 10–500+ MB | RTAB-Map localization (Mode 2) | **Automatic** — saved on Ctrl+C |
| `.pgm` | ~100 KB | AMCL / Nav2 map_server (Mode 3) | **Manual** — see below |
| `.yaml` | ~200 B | AMCL / Nav2 map_server (Mode 3) | **Manual** — saved with `.pgm` |

### Step-by-step: Complete mapping workflow

**Step 1 — Start Gazebo**
```bash
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py
```

**Step 2 — Start SLAM**
```bash
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=dual map_name:=house_my1_map
```

**Step 3 — Verify SLAM is working** (wait ~10 seconds)
```bash
ros2 run tf2_ros tf2_echo map odom
# Should show real translation values, not "Could not transform"
```

**Step 4 — Drive the robot** to explore
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
# Drive slowly, cover all areas, return to start for loop closure
```

**Step 5 — Save the 2D map** (while SLAM is still running)
```bash
ros2 run nav2_map_server map_saver_cli -f ~/CA-MCW/maps/house_my1_map/house_my1_map
```
This creates `house_my1_map.pgm` + `house_my1_map.yaml`.

**Step 6 — Stop SLAM** with Ctrl+C
The `.db` file saves automatically.

**Step 7 — Use the map**
```bash
# Option A: RTAB-Map Localization (uses .db) — via vf_robot_bringup
ros2 launch vf_robot_bringup bringup_launch.py \
    robot:=virofighter controller:=mppi localization:=rtabmap_loc \
    camera:=dual map_name:=house_my1_map

# Option B: AMCL (uses .pgm/.yaml) — via vf_robot_bringup
ros2 launch vf_robot_bringup bringup_launch.py \
    robot:=virofighter controller:=mppi localization:=amcl \
    camera:=dual map:=~/CA-MCW/maps/house_my1_map/house_my1_map.yaml
```

### Alternative: Export 2D map from .db after SLAM

If you forgot to run `map_saver_cli` while SLAM was running:

```bash
rtabmap-databaseViewer ~/CA-MCW/maps/house_my1_map/house_my1_map.db
# Menu: File → Export 2D Grid Map → save as .pgm
```

### Continue mapping a previous session

```bash
ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=dual map_name:=house_my1_map new_map:=false
# Loads existing house_my1_map.db and adds new data to it
```

---

## 🗺️ Operating Modes (4 Modes)

This package supports 4 mutually exclusive operating modes. All modes are
orchestrated by `vf_robot_bringup` — pass `localization:=<mode>` to `bringup_launch.py`.
This package provides the SLAM/localization and depth-to-scan launches that bringup includes.

| Mode | What `vf_robot_slam` provides | What `vf_robot_bringup` adds |
|------|-------------------------------|------------------------------|
| **1. RTAB-Map SLAM** | `rtabmap_slam.launch.py` + `depth_to_scan.launch.py` | Nav2 planners, controllers, costmaps |
| **2. RTAB-Map Loc** | `rtabmap_loc.launch.py` + `depth_to_scan.launch.py` | Nav2 planners, controllers, costmaps |
| **3. AMCL** | `depth_to_scan.launch.py` only | `map_server`, AMCL, Nav2 planners |
| **4. SLAM Toolbox** | `depth_to_scan.launch.py` only | SLAM Toolbox, Nav2 planners |

**Key constraint:** Each mode has exactly ONE `/map` publisher and ONE `map→odom` TF publisher. Never run two modes simultaneously.

### Mode 1: RTAB-Map SLAM + Nav2

```bash
# Terminal 1: Gazebo
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py

# Terminal 2: Full bringup — SLAM + depth-to-scan + Nav2 (all-in-one)
ros2 launch vf_robot_bringup bringup_launch.py \
    robot:=virofighter controller:=mppi localization:=rtabmap_slam \
    camera:=dual map_name:=house_my1_map new_map:=true
```

### Mode 2: RTAB-Map Localization + Nav2

```bash
# Terminal 1: Gazebo
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py

# Terminal 2: Full bringup — Loc + depth-to-scan + Nav2
ros2 launch vf_robot_bringup bringup_launch.py \
    robot:=virofighter controller:=mppi localization:=rtabmap_loc \
    camera:=dual map_name:=house_my1_map
```

### Mode 3: AMCL + Nav2

```bash
# Terminal 1: Gazebo
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py

# Terminal 2: Full bringup — AMCL + depth-to-scan + Nav2 (needs .pgm/.yaml map)
ros2 launch vf_robot_bringup bringup_launch.py \
    robot:=virofighter controller:=mppi localization:=amcl \
    camera:=dual map:=~/CA-MCW/maps/house_my1_map/house_my1_map.yaml
```

### Mode 4: SLAM Toolbox + Nav2

```bash
# Terminal 1: Gazebo
ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py

# Terminal 2: Full bringup — SLAM Toolbox + depth-to-scan + Nav2
ros2 launch vf_robot_bringup bringup_launch.py \
    robot:=virofighter controller:=mppi localization:=slam_toolbox \
    camera:=dual
```

---

## 📊 Validation Matrix — Who Provides What

This is the master cross-check table. For Nav2 autonomous navigation to work, **every row must have a ✅ in the mode you're running.** If any row is missing, navigation will fail.

### Required topics and TF

| Requirement | Source package | Mode 1 (SLAM) | Mode 2 (Loc) | Mode 3 (AMCL) | Mode 4 (SLAM Toolbox) |
|-------------|--------------|----------------|---------------|----------------|----------------------|
| `/map` | varies | ✅ RTAB-Map | ✅ RTAB-Map (.db) | ✅ map_server (.pgm) | ✅ SLAM Toolbox |
| `map→odom` TF | varies | ✅ RTAB-Map | ✅ RTAB-Map | ✅ AMCL | ✅ SLAM Toolbox |
| `/scan` | `vf_robot_slam` | ✅ depth_to_scan | ✅ depth_to_scan | ✅ depth_to_scan | ✅ depth_to_scan |
| `/odom` | `vf_robot_gazebo` | ✅ Gazebo diff_drive | ✅ Gazebo diff_drive | ✅ Gazebo diff_drive | ✅ Gazebo diff_drive |
| `odom→base_footprint` TF | `vf_robot_gazebo` | ✅ Gazebo diff_drive | ✅ Gazebo diff_drive | ✅ Gazebo diff_drive | ✅ Gazebo diff_drive |
| `base_link→sensor` TFs | `vf_robot_description` | ✅ robot_state_pub | ✅ robot_state_pub | ✅ robot_state_pub | ✅ robot_state_pub |
| `/robot_description` | `vf_robot_description` | ✅ robot_state_pub | ✅ robot_state_pub | ✅ robot_state_pub | ✅ robot_state_pub |
| `/clock` | `vf_robot_gazebo` | ✅ Gazebo | ✅ Gazebo | ✅ Gazebo | ✅ Gazebo |
| Nav2 planners/controllers | `vf_robot_bringup` | ✅ bringup_launch.py | ✅ bringup_launch.py | ✅ bringup_launch.py | ✅ bringup_launch.py |

### Which package provides each launch

| Launch file | Package | Used in modes |
|-------------|---------|--------------|
| `house_my1_world_xacro.launch.py` | `vf_robot_gazebo` | All (simulation) |
| `rtabmap_slam.launch.py` | `vf_robot_slam` | Mode 1 only |
| `rtabmap_loc.launch.py` | `vf_robot_slam` | Mode 2 only |
| `depth_to_scan.launch.py` | `vf_robot_slam` | All modes |
| `bringup_launch.py localization:=amcl` | `vf_robot_bringup` | Mode 3 |
| `bringup_launch.py localization:=slam_toolbox` | `vf_robot_bringup` | Mode 4 |

### Map file requirements per mode

| Mode | Map file needed | Format | How to create it |
|------|----------------|--------|-----------------|
| 1. RTAB-Map SLAM | None | — | Mode 1 creates the map |
| 2. RTAB-Map Loc | `<name>.db` | RTAB-Map database | Auto-saved on Ctrl+C from Mode 1 |
| 3. AMCL | `<name>.pgm` + `<name>.yaml` | 2D occupancy grid | `map_saver_cli` during Mode 1, or export from `.db` |
| 4. SLAM Toolbox | None (or serialized map) | — | Mode 4 creates the map (or loads serialized) |

### Per-mode node flow (dual camera, Gazebo)

**Mode 1 — RTAB-Map SLAM**
```
Gazebo ──► /odom, odom→base_footprint TF
       ──► /d435i/rgb/..., /d435i/depth/..., /d455/rgb/..., /d455/depth/...

rgbd_sync_d435i ──► /rgbd_image/d435i ──┐
rgbd_sync_d455  ──► /rgbd_image/d455  ──┤
                                        └──► rtabmap (SLAM) ──► /map, map→odom TF, .db

pc_to_scan_d435i ──► /scan_d435i ──┐
pc_to_scan_d455  ──► /scan_d455  ──┤
                                   └──► scan_merger ──► /scan ──► [Nav2 costmap]
```

**Mode 2 — RTAB-Map Localization**
```
Gazebo ──► /odom, odom→base_footprint TF
       ──► /d435i/rgb/..., /d435i/depth/..., /d455/rgb/..., /d455/depth/...

rgbd_sync_d435i ──► /rgbd_image/d435i ──┐
rgbd_sync_d455  ──► /rgbd_image/d455  ──┤
                                        └──► rtabmap (LOC, loads .db) ──► /map, map→odom TF

pc_to_scan_d435i ──► /scan_d435i ──┐
pc_to_scan_d455  ──► /scan_d455  ──┤
                                   └──► scan_merger ──► /scan ──► [Nav2 costmap]
```

**Mode 3 — AMCL (future)**
```
Gazebo ──► /odom, odom→base_footprint TF
       ──► /d435i/depth/..., /d455/depth/...

pc_to_scan_d435i ──► /scan_d435i ──┐
pc_to_scan_d455  ──► /scan_d455  ──┤
                                   └──► scan_merger ──► /scan ──► AMCL ──► map→odom TF

map_server (loads .pgm/.yaml) ──► /map
```

**Mode 4 — SLAM Toolbox (future)**
```
Gazebo ──► /odom, odom→base_footprint TF
       ──► /d435i/depth/..., /d455/depth/...

pc_to_scan_d435i ──► /scan_d435i ──┐
pc_to_scan_d455  ──► /scan_d455  ──┤
                                   └──► scan_merger ──► /scan ──► SLAM Toolbox ──► /map, map→odom TF
```

### Quick self-check commands

After launching any mode, verify every requirement is met:

```bash
# 1. /map topic is publishing
ros2 topic hz /map

# 2. map→odom TF exists
ros2 run tf2_ros tf2_echo map odom

# 3. /scan topic is publishing
ros2 topic hz /scan

# 4. odom→base_footprint TF exists
ros2 run tf2_ros tf2_echo odom base_footprint

# 5. Full TF chain is connected (saves PDF)
ros2 run tf2_tools view_frames

# 6. All expected topics exist
ros2 topic list | grep -E "^/(map|scan|odom|robot_description|clock)$"
```

---

## ✅ Verification & Diagnostics

### Is SLAM working?

```bash
# Map frame should appear within ~10 seconds
ros2 run tf2_ros tf2_echo map odom

# RTAB-Map processing stats (should NOT time out)
ros2 topic echo /rtabmap/info --once

# RGBD sync rate (expect 15–30 Hz)
ros2 topic hz /rgbd_image/d455
ros2 topic hz /rgbd_image/d435i
```

### Is depth_to_scan working?

```bash
# Scan rate
ros2 topic hz /scan
ros2 topic hz /scan_d435i    # dual mode only
ros2 topic hz /scan_d455     # dual mode only
```

### TF inspection

```bash
# Full TF tree (saves PDF)
ros2 run tf2_tools view_frames

# Specific transforms
ros2 run tf2_ros tf2_echo map odom
ros2 run tf2_ros tf2_echo odom base_footprint
ros2 run tf2_ros tf2_echo base_link camera_d455_link
```

### Topic inspection

```bash
ros2 topic list | grep -E "scan|map|rtabmap|rgbd"
ros2 topic echo /odom --once | grep child_frame_id    # must be: base_footprint
ros2 topic echo /clock --once                          # Gazebo sim time
```

---

## 🐛 Troubleshooting

### `map` frame never appears / `tf2_echo map odom` times out

This is the most common issue. Work through in order:

**1. Verify `use_sim_time` is set.**
Gazebo sim time is ~1000 s. Wall time is ~1.77 billion seconds. Mismatch = every RGBD frame silently dropped.

```bash
ros2 topic echo /clock --once
ros2 topic echo /rgbd_image/d455 --once | grep sec
# Both sec values must be in the same range (~1000, not ~1.77 billion)
```

Fix: ensure `use_sim_time:=true` in the launch command (default for all launch files).

**2. Verify odometry frame ID.**
RTAB-Map uses `frame_id: base_footprint`. Wrong frame = silent TF lookup failure.

```bash
ros2 topic echo /odom --once | grep child_frame_id
# Must print: child_frame_id: base_footprint
```

**3. Verify RGBD sync rate.**

```bash
ros2 topic hz /rgbd_image/d455
# Expected: 15–30 Hz. If ~6 Hz → approx_sync_max_interval was wrong (fixed in current version).
```

**4. Check for message filter warnings.**

```bash
ros2 topic echo /rosout 2>/dev/null | grep -i "dropping\|warn\|transform"
```

**5. Check RTAB-Map is receiving frames.**

```bash
ros2 topic echo /rtabmap/info --once
# If this times out: zero frames processed → check topic remappings
```

---

### LaserScan detecting floor (D435i)

Expected with `method:=dimg`. The D435i at 1.773 m height + 60° tilt = scan plane hits floor at ~1.024 m.

**Solutions:**
- Use `method:=pc2scan` — world-space Z filter handles the tilt correctly, no blind zone
- Use `camera:=d455` or `camera:=dual` for obstacle detection
- With `method:=dimg`, `range_min=1.1 m` clips floor but creates a blind zone < 1.1 m

---

### RTAB-Map not detecting loop closures

- Drive slowly — motion blur kills visual features
- Ensure sufficient texture and lighting in the environment
- Return to a previously visited area from a similar angle
- Check `Vis/MinInliers` (currently 15) — lower for more permissive closure

---

### Scan is empty or at 0 Hz

**With `method:=dimg` (dual mode):**
The `depthimage_to_laserscan` node uses lazy subscription — it won't subscribe to depth input until something subscribes to its scan output. The launch file starts `scan_merger` BEFORE the converter nodes to handle this. If you're seeing zero Hz, verify `scan_merger` is running:

```bash
ros2 node list | grep scan_merger
```

**With `method:=pc2scan`:**
The `pc_to_scan.py` node needs TF to be available. Check:

```bash
ros2 run tf2_ros tf2_echo base_footprint camera_d455_depth_optical_frame
ros2 topic hz /d455/depth/d455_depth/points
```

---

### TF errors for camera frames in RViz

Camera frames are static transforms from `robot_state_publisher`. If missing:

```bash
ros2 topic echo /robot_description --once | head -5
ros2 run tf2_tools view_frames
```

---

# FOR DEVELOPERS

---

## 📁 Package Structure

```
vf_robot_slam/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   └── cameras/                          # Camera topic/frame reference (not loaded by launch)
│       ├── d435i.yaml                    #   D435i: topics, frames, physical specs
│       ├── d455.yaml                     #   D455: topics, frames, physical specs
│       └── dual.yaml                     #   Both cameras combined
├── launch/
│   ├── rtabmap_slam.launch.py            # Mode 1: RTAB-Map SLAM (top-level)
│   ├── rtabmap_loc.launch.py             # Mode 2: RTAB-Map Localization (top-level)
│   ├── depth_to_scan.launch.py           # Router → picks dimg or pc2scan (top-level)
│   └── include/                          # Reusable building blocks (NOT launched directly)
│       ├── rgbd_sync.launch.py           #   RGBD sync nodes (shared by slam + loc)
│       ├── depth_to_scan_dimg.launch.py  #   depthimage_to_laserscan method
│       └── depth_to_scan_pc2scan.launch.py # pc_to_scan.py method
├── rviz/
│   ├── rtabmap_slam.rviz                 # RViz config for SLAM mode
│   └── rtabmap_loc.rviz                  # RViz config for Localization mode
└── scripts/
    ├── scan_merger.py                    # Custom LaserScan merger (replaces ira_laser_tools)
    └── pc_to_scan.py                     # Custom PointCloud2→LaserScan (replaces pointcloud_to_laserscan)

# Maps stored OUTSIDE the package:
~/CA-MCW/maps/                      # Auto-created by launch files
```

### Why no YAML config files for RTAB-Map parameters?

All RTAB-Map parameters are defined **inline** in the launch files (`_get_rtabmap_slam_params()` and `_get_rtabmap_loc_params()`) with comments explaining *why* each value is set. This is intentional:

- ROS 2 YAML loading requires the exact node name as the top-level key — node names change depending on camera mode, so no single YAML works
- Inline parameters with comments serve as both config AND documentation
- No risk of YAML files drifting out of sync with actual launch behavior

### Why camera YAML files exist but aren't loaded by launch files?

The `config/cameras/` YAML files are **reference documentation** — verified topic names, frame IDs, and physical specs for each camera. They are not loaded by any launch file. Instead, topic names and frames are hardcoded in launch files where the context makes the purpose clear.

---

## 🏗️ Architecture — Modular Launch Design

### The include/ pattern

Launch files use `include/` to eliminate code duplication:

```
include/rgbd_sync.launch.py          ← Defines RGBD sync nodes ONCE
         │
         ├── rtabmap_slam.launch.py   Includes rgbd_sync + adds RTAB-Map SLAM node
         └── rtabmap_loc.launch.py    Includes rgbd_sync + adds RTAB-Map LOC node
```

```
depth_to_scan.launch.py               ← Router (the ONLY file users call)
         │
         │  argument: method:=dimg or method:=pc2scan
         │
         ├── include/depth_to_scan_dimg.launch.py      (depthimage_to_laserscan)
         └── include/depth_to_scan_pc2scan.launch.py   (pc_to_scan.py custom node)
```

### Why this matters for `vf_robot_bringup`

The bringup package only needs to know **three launch file names**:

| Navigation calls | From `vf_robot_slam` |
|------------------|---------------------|
| RTAB-Map SLAM | `rtabmap_slam.launch.py` |
| RTAB-Map Localization | `rtabmap_loc.launch.py` |
| Depth → LaserScan | `depth_to_scan.launch.py` |

It never needs to know about `include/` files, which method is used, or which custom nodes run internally.

### File-by-file summary

| File | Role | Key details |
|------|------|-------------|
| `rtabmap_slam.launch.py` | SLAM mode | Includes `rgbd_sync.launch.py` for dual; inline params via `_get_rtabmap_slam_params()`; creates map folder; `delete_db_on_start` controlled by `new_map` arg |
| `rtabmap_loc.launch.py` | Localization mode | Includes `rgbd_sync.launch.py` for dual; inline params via `_get_rtabmap_loc_params()`; guards against missing `.db` file |
| `depth_to_scan.launch.py` | Router | Pure routing — includes one of the two `include/` files based on `method` arg |
| `include/rgbd_sync.launch.py` | RGBD sync | `rtabmap_sync/rgbd_sync` nodes with correct remappings; `approx_sync_max_interval: 0.05` (0.0 defeats sync) |
| `include/depth_to_scan_dimg.launch.py` | dimg method | `depthimage_to_laserscan` nodes; startup order: merger FIRST, then converters (lazy subscription fix) |
| `include/depth_to_scan_pc2scan.launch.py` | pc2scan method | Custom `pc_to_scan.py` nodes; no startup order dependency (no lazy subscription) |
| `scripts/scan_merger.py` | Scan merger | Custom node replacing `ira_laser_tools`; merges `/scan_d435i` + `/scan_d455` → `/scan` |
| `scripts/pc_to_scan.py` | PC2 → scan | Custom node replacing `pointcloud_to_laserscan`; world-space Z height filter via TF2 |

---

## 📷 Camera Configuration — D435i vs D455

| Property | D435i (front) | D455 (rear) |
|----------|--------------|-------------|
| Position (m) | (0.045, 0, 1.773) | (−0.525, 0, 0.429) |
| Height | 1.773 m | 0.429 m |
| Tilt | 60° downward | Horizontal (0°) |
| Orientation | Faces front | Faces rear (180°) |
| Depth range | 0.6–6.0 m | 0.6–6.0 m |
| Horizontal FOV | 87° | 87° |
| Link frame | `camera_d435i_link` | `camera_d455_link` |
| Color optical frame | `camera_d435i_color_optical_frame` | `camera_d455_color_optical_frame` |
| Depth optical frame | `camera_d435i_depth_optical_frame` | `camera_d455_depth_optical_frame` |
| RGB topic | `/d435i/rgb/d435i_rgb/image_raw` | `/d455/rgb/d455_rgb/image_raw` |
| Depth topic | `/d435i/depth/d435i_depth/depth/image_raw` | `/d455/depth/d455_depth/depth/image_raw` |
| PointCloud2 topic | `/d435i/depth/d435i_depth/points` | `/d455/depth/d455_depth/points` |
| Camera info topic | `/d435i/depth/d435i_depth/depth/camera_info` | `/d455/depth/d455_depth/depth/camera_info` |
| IMU topic | `/d435i/imu/d435i_imu_controller/out` | `/d455/imu/d455_imu_controller/out` |
| RGBD sync output | `/rgbd_image/d435i` | `/rgbd_image/d455` |

### D435i floor issue explained

The D435i is at 1.773 m, tilted 60° down. With `depthimage_to_laserscan` (method:=dimg), the scan plane is locked to the camera's optical axis. Even the topmost image row is 31° below horizontal, so the scan always sees the floor.

**Floor intersection distance:** `1.773 / tan(60°) ≈ 1.024 m`

With `method:=dimg`, `range_min` is set to 1.1 m to clip floor returns. This creates a blind zone: obstacles closer than 1.1 m are invisible.

With `method:=pc2scan`, the custom `pc_to_scan.py` node transforms all points into `base_footprint` frame and filters by world-space Z height (0.02–2.0 m). The camera tilt is irrelevant — floor is excluded cleanly and `range_min` can be 0.1 m.

---

## 🔀 Two Depth-to-Scan Methods — dimg vs pc2scan

### method:=dimg — depthimage_to_laserscan

```
Depth IMAGE (2D) ──► collapse centre row ──► LaserScan
```

- Uses the standard `ros-humble-depthimage-to-laserscan` package
- Processes cheap 2D depth image — runs at 15–30 Hz even in Gazebo
- Scan plane locked to camera optical axis (problem for tilted cameras)
- D435i: `range_min=1.1 m`, `output_frame=base_footprint`
- D455: `range_min=0.6 m`, `output_frame=camera_d455_link`

**Startup order in dual mode:** `scan_merger` starts FIRST, then converter nodes. This is required because `depthimage_to_laserscan` uses lazy subscription — it won't subscribe to depth input until something subscribes to its scan output. If the merger isn't listening when converters start, they go idle.

### method:=pc2scan — pc_to_scan.py (custom)

```
PointCloud2 (3D) ──► TF transform to base_footprint ──► Z height filter ──► polar binning ──► LaserScan
```

- Uses custom `scripts/pc_to_scan.py` (NOT the `ros-humble-pointcloud-to-laserscan` package)
- Transforms full 3D pointcloud into `base_footprint` using TF2
- Filters by world-space Z: keeps points at 0.02–2.0 m (floor and ceiling excluded)
- `range_min=0.1 m` for both cameras — no blind zone
- **No startup order dependency** — uses normal rclpy subscriptions (no lazy pattern)

### Why pc_to_scan.py replaced pointcloud_to_laserscan

The standard `ros-humble-pointcloud-to-laserscan` node uses `message_filters::Subscriber` with a lazy subscription thread. In ROS 2 Humble + CycloneDDS, subscriptions created by the background thread after `spin()` starts are never processed by the executor. The node appears subscribed (`ros2 node info` confirms it), receives data at the DDS layer, but `cloudCallback` never fires.

The custom `pc_to_scan.py` uses normal `rclpy` subscriptions — no `message_filters`, no lazy subscription, deterministic behavior.

### Comparison matrix

| Feature | `dimg` | `pc2scan` |
|---------|--------|-----------|
| Underlying tool | `depthimage_to_laserscan` (apt package) | `pc_to_scan.py` (custom script) |
| Input data | 2D depth image | 3D PointCloud2 |
| Speed in Gazebo | 15–30 Hz ✅ | ~2–3 Hz (CPU-bound) |
| Speed on real robot | 15–30 Hz ✅ | 15–30 Hz ✅ |
| D435i floor handling | `range_min=1.1 m` (blind zone) | Height filter Z=[0.02, 2.0] m ✅ |
| D455 floor handling | No issue ✅ | No issue ✅ |
| Startup order matters? | Yes (merger before converters) | No |
| External dependency | `ros-humble-depthimage-to-laserscan` | None (pure Python + numpy) |

---

## 🔧 Custom Nodes — scan_merger.py & pc_to_scan.py

### scan_merger.py

**Purpose:** Merges `/scan_d435i` + `/scan_d455` into a single `/scan` topic. Replaces `ira_laser_tools` with zero external dependencies.

**How it works:**
1. Subscribes to two LaserScan topics
2. If input scans share the output frame AND scan geometry → fast path: per-bin `numpy.minimum()` (< 0.1 ms)
3. If frames differ → cross-frame path: TF2 transform + vectorized numpy projection (< 1 ms)
4. Publishes merged LaserScan at 30 Hz on timer

**Key design decisions:**
- No staleness check on input scans — Gazebo sim time jitter caused scans to be falsely rejected, dropping output to ~2 Hz
- TF lookups are cached after first query (camera frames are static URDF transforms)
- Uses newest input timestamp for the output header

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scan_topics` | `"/scan_d435i /scan_d455"` | Space-separated input topics |
| `output_topic` | `"/scan"` | Merged output topic |
| `output_frame` | `"base_footprint"` | frame_id for merged scan |
| `angle_min` | `-π` | Minimum angle (radians) |
| `angle_max` | `+π` | Maximum angle (radians) |
| `angle_increment` | `0.00581` (~0.33°) | Angular resolution |
| `range_min` | `0.1` | Minimum range (m) |
| `range_max` | `6.0` | Maximum range (m) |

### pc_to_scan.py

**Purpose:** Converts PointCloud2 to LaserScan with world-space height filtering. Replaces `ros-humble-pointcloud-to-laserscan`.

**How it works:**
1. Subscribes to PointCloud2 (RELIABLE QoS, matching Gazebo and RealSense)
2. Looks up TF from pointcloud frame → `base_footprint` (cached after first lookup)
3. Transforms all points using full 3×3 rotation + translation
4. Filters by world-space Z height (removes floor and ceiling)
5. Projects to 2D polar coordinates (range + angle)
6. Bins into angular slots, keeps closest range per bin
7. Publishes LaserScan in `base_footprint` frame

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_frame` | `"base_footprint"` | Output frame |
| `min_height` | `0.02` | Min world Z to keep (m) |
| `max_height` | `2.0` | Max world Z to keep (m) |
| `angle_min` | `-π` | Scan start angle (radians) |
| `angle_max` | `+π` | Scan end angle (radians) |
| `angle_increment` | `0.00581` (~0.33°) | Angular resolution |
| `range_min` | `0.1` | Min 2D range (m) |
| `range_max` | `6.0` | Max 2D range (m) |
| `transform_tolerance` | `0.1` | Max TF wait time (s) |

---

## 🔗 Dual Camera Scan Merging

In dual camera mode (`camera:=dual`), both cameras produce individual scans that are merged into a single `/scan` topic:

```
depth_to_scan.launch.py  camera:=dual  merge_scans:=true
│
├── /scan_d435i   (front arc, ~87°)  ─┐
│                                      ├── scan_merger.py ──► /scan (~174° merged)
└── /scan_d455    (rear arc, ~87°)   ─┘
```

With `merge_scans:=false`:
```
├── /scan_d435i   (front arc, ~87°)
└── /scan_d455    (rear arc, ~87°)
    (no /scan topic — use Nav2 multi-source costmap)
```

**`/scan` always exists** when `merge_scans:=true` (default). AMCL, SLAM Toolbox, and Nav2 always have a scan topic to work with.

---

## ⚙️ Launch Arguments Reference

### rtabmap_slam.launch.py

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `camera` | `d435i`, `d455`, `dual` | `dual` | Camera configuration |
| `map_name` | any string | `default_map` | Name for the map folder and `.db` file |
| `maps_dir` | path | `~/CA-MCW/maps` | Base directory for all maps |
| `new_map` | `true`, `false` | `true` | Delete existing `.db` (true) or continue (false) |
| `rviz` | `true`, `false` | `true` | Launch RViz |
| `use_sim_time` | `true`, `false` | `true` | Gazebo (true) or real robot (false) |

### rtabmap_loc.launch.py

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `camera` | `d435i`, `d455`, `dual` | `dual` | Camera configuration |
| `map_name` | any string | *(required)* | Name of the map to load |
| `maps_dir` | path | `~/CA-MCW/maps` | Base directory for maps |
| `rviz` | `true`, `false` | `true` | Launch RViz |
| `use_sim_time` | `true`, `false` | `true` | Gazebo (true) or real robot (false) |

### depth_to_scan.launch.py

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `method` | `dimg`, `pc2scan` | `dimg` | Conversion method |
| `camera` | `d435i`, `d455`, `dual` | `dual` | Camera configuration |
| `merge_scans` | `true`, `false` | `true` | Merge dual scans into `/scan` |
| `use_sim_time` | `true`, `false` | `true` | Gazebo (true) or real robot (false) |

---

## 📡 Topics & TF Published

### RTAB-Map SLAM / Localization

| Topic | Type | Description |
|-------|------|-------------|
| `/map` | `nav_msgs/OccupancyGrid` | 2D occupancy grid |
| `/rtabmap/cloud_map` | `sensor_msgs/PointCloud2` | 3D point cloud map |
| `/rtabmap/mapPath` | `nav_msgs/Path` | Robot trajectory |
| `/rtabmap/info` | `rtabmap_msgs/Info` | Per-frame stats |

### RGBD Sync (dual mode)

| Topic | Type | Description |
|-------|------|-------------|
| `/rgbd_image/d435i` | `rtabmap_msgs/RGBDImage` | Synced RGB+D from D435i |
| `/rgbd_image/d455` | `rtabmap_msgs/RGBDImage` | Synced RGB+D from D455 |

### Depth to LaserScan

| Topic | Type | Published when |
|-------|------|----------------|
| `/scan` | `sensor_msgs/LaserScan` | Single camera, or dual with `merge_scans:=true` |
| `/scan_d435i` | `sensor_msgs/LaserScan` | Dual mode only |
| `/scan_d455` | `sensor_msgs/LaserScan` | Dual mode only |

### TF chain

```
[RTAB-Map slam/loc]     [Gazebo / robot_localization]     [robot_state_publisher]
         │                          │                              │
         ▼                          ▼                              ▼
       map ───────────► odom ───────────► base_footprint ──► base_link
                                                                ├── camera_d435i_link
                                                                │     ├── color_optical_frame
                                                                │     ├── depth_optical_frame
                                                                │     └── imu_frame
                                                                ├── camera_d455_link
                                                                │     ├── color_optical_frame
                                                                │     ├── depth_optical_frame
                                                                │     └── imu_frame
                                                                ├── camera_fisheye_{front,left,rear,right}_link
                                                                │     └── *_optical_frame
                                                                ├── ultrasonic_{front_left,front_right,rear,side_left,side_right}_link
                                                                ├── uvc_lights_link
                                                                └── wheel_{front,rear}_{left,right}_link
```

| Transform | Publisher | Active in |
|-----------|-----------|-----------|
| `map → odom` | RTAB-Map | Modes 1 and 2 only |
| `odom → base_footprint` | Gazebo / robot_localization | Always |
| `base_footprint → base_link` | `robot_state_publisher` | Always (static) |
| `base_link → camera_*` | `robot_state_publisher` | Always (static) |

---

## 🔧 Nav2 Costmap Configuration

For dual camera mode with `merge_scans:=false`, configure Nav2 to consume both scan topics as separate observation sources:

```yaml
# In nav2_params.yaml
local_costmap:
  local_costmap:
    ros__parameters:
      plugins: ["obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        observation_sources: scan_d435i scan_d455
        scan_d435i:
          topic: /scan_d435i
          sensor_frame: camera_d435i_depth_optical_frame
          data_type: "LaserScan"
          clearing: true
          marking: true
          max_obstacle_height: 2.0
          min_obstacle_height: 0.0
        scan_d455:
          topic: /scan_d455
          sensor_frame: camera_d455_depth_optical_frame
          data_type: "LaserScan"
          clearing: true
          marking: true
          max_obstacle_height: 2.0
          min_obstacle_height: 0.0
```

This approach lets Nav2 handle multi-sensor fusion natively, avoiding the scan merger entirely for costmap purposes. AMCL still requires a single `/scan` topic, so use `merge_scans:=true` for Mode 3.

---

## 📝 Hard-Won Lessons

These are bugs we discovered and fixed during development. Documented here to prevent regression.

**1. `use_sim_time` MUST be true for Gazebo.**
Gazebo publishes sim timestamps ~1000 s. Wall time is ~1.77 billion seconds. Without `use_sim_time:=true`, RTAB-Map's message filter silently drops every RGBD frame. The `map` frame never appears and there are zero error messages.

**2. `frame_id` MUST be `base_footprint`, not `base_link`.**
Gazebo publishes odometry as `odom → base_footprint`. If RTAB-Map uses `frame_id: base_link`, TF lookups for `odom → base_link` fail silently — RTAB-Map never publishes `map → odom`.

**3. `depth/camera_info` MUST be remapped in rgbd_sync.**
Without remapping `depth/camera_info`, the `rgbd_sync` node produces `RGBDImage` messages with empty `frame_id` fields. RTAB-Map then fails TF lookups silently.

**4. `approx_sync_max_interval: 0.05`, never `0.0`.**
Setting `0.0` defeats approximate sync on some `rtabmap_sync` versions. Result: RGBD sync drops from 15–30 Hz to ~6 Hz.

**5. `depthimage_to_laserscan` lazy subscription startup order.**
In dual mode with `method:=dimg`, the `scan_merger` must start BEFORE the converter nodes. `depthimage_to_laserscan` uses lazy subscription — it won't subscribe to depth input until something subscribes to its scan output. If converters start first with zero subscribers, they go idle permanently.

**6. `pointcloud_to_laserscan` broken in Humble + CycloneDDS.**
The standard package uses `message_filters::Subscriber` with a background thread for lazy subscription. In CycloneDDS, subscriptions created by the background thread after `spin()` starts are never processed. Replaced with custom `pc_to_scan.py`.

**7. `scan_merger` staleness check breaks with Gazebo sim time.**
The original merger rejected scans older than 0.5 s. With Gazebo, clock propagation jitter between nodes causes timestamps to drift enough to trigger false rejections, dropping output to ~2 Hz. Fix: no staleness check — always use latest available scan.

**8. D455 `output_frame` must be `camera_d455_link`, not `base_footprint` (dimg method).**
`depthimage_to_laserscan` does NOT rotate angle values — it only stamps the `header.frame_id`. Using `base_footprint` as output_frame makes the scan appear to face forward when the camera faces rear. The `scan_merger` handles the cross-frame transformation.

**9. `Reg/Strategy` MUST be `"0"` (visual) in localization mode, NOT `"1"` (ICP).**
ICP registration needs a close initial pose to converge. In localization, every fresh Gazebo session starts with `odom=(0,0,0)` and the initial `map→odom` is whatever was saved last session — making the initial guess wrong by an arbitrary offset. ICP diverges immediately, no loop closure is accepted, `map→odom` never updates, and the scan creates phantom obstacles everywhere. Visual registration (`Reg/Strategy: "0"`) matches feature descriptors globally — no initial pose needed — and relocates the robot in 1–5 seconds. ICP works in SLAM because continuous odometry provides a good initial guess; it fails for cold-start localization.

**10. `Rtabmap/LoopThr` and `RGBD/OptimizeMaxError` interact on first lock.**
`LoopThr: 0.15` (too strict) rejects the first weak visual match. `OptimizeMaxError: 1.0` (too tight) rejects the subsequent graph optimization if odom drifted during startup. Use `LoopThr: 0.11` and `OptimizeMaxError: 3.0` for reliable first-lock at the start of a session.

---

## 📄 Dependencies

### Required (apt)

```bash
sudo apt install ros-humble-rtabmap-ros \
                 ros-humble-depthimage-to-laserscan \
                 ros-humble-topic-tools
```

### Python (included in package)

- `scripts/scan_merger.py` — custom LaserScan merger (replaces `ira_laser_tools`)
- `scripts/pc_to_scan.py` — custom PointCloud2→LaserScan converter (replaces `pointcloud_to_laserscan`)

Both use only `rclpy`, `numpy`, `sensor_msgs`, and `tf2_ros` — no additional pip packages needed.

### Optional (for improved multi-camera SLAM)

```bash
# OpenGV improves multi-camera pose estimation
cd ~
git clone https://github.com/laurentkneip/opengv.git
cd opengv && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install

# Rebuild rtabmap with OpenGV
cd ~/rtabmap/build
cmake .. -DWITH_OPENGV=ON
make -j$(nproc) && sudo make install

# Rebuild rtabmap_ros
cd ~/CA-MCW
colcon build --symlink-install
```

---

## 📄 License

Apache 2.0

---

## 👤 Maintainer

**Pravin Oli** — olipravin18@gmail.com · roboticsmaterials@gmail.com
Project: **CA-MCW** | Package: `vf_robot_slam`
