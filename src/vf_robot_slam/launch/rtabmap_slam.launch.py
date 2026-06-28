#!/usr/bin/env python3
#
# Copyright  EUROKNOWS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Pravin Oli
# Email: pravin.oli.08@gmail.com, olipravin18@gmail.com
# Company: EUROKNOWS CO., LTD.
# Website: https://www.euroknows.com/en/home/
#
# Erasmus Mundus Joint Masters in Intelligent Field Robotics System (IFROS)
# https://ifrosmaster.org/
#
# Universitat de Girona, Spain - https://www.udg.edu/en/
# Eötvös Loránd University, Hungary - https://www.elte.hu/
#

"""
rtabmap_slam — Mapping — ViroFighter UVC-1
══════════════════════════════════════════
Builds or extends a map. Output .db used by rtabmap_loc.launch.py.

PIPELINE
  [rgbd_sync] → rtabmap (slam mode) → .db + /map + map→odom TF

INCLUDES
  include/rgbd_sync.launch.py  (dual only)

ARGS (defaults marked *)
  camera       : *dual | d435i | d455
  map_name     : *default_map | <name> | /absolute/path
  maps_dir     : *~/CA-MCW/maps | <path>   (ignored if map_name is absolute)
  rviz         : *true | false
  new_map      : *true | false   (true=delete existing .db, false=continue)
  use_sim_time : *true | false

MAP PATH RESOLUTION
  Relative → maps_dir/map_name/map_name.db
  Absolute → map_name/map_name.db  (maps_dir ignored)

QUICK REFERENCE — copy-paste commands
  # [DEFAULT] Gazebo, dual, new map
  ros2 launch vf_robot_slam rtabmap_slam.launch.py map_name:=house_my1_map

  # Gazebo, dual, CONTINUE existing map
  ros2 launch vf_robot_slam rtabmap_slam.launch.py map_name:=house_my1_map new_map:=false

  # Gazebo, single D455
  ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=d455 map_name:=house_my1_map

  # Gazebo, single D435i
  ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=d435i map_name:=house_my1_map

  # Real robot, dual
  ros2 launch vf_robot_slam rtabmap_slam.launch.py map_name:=house_my1_map use_sim_time:=false

  # Real robot, single D455
  ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=d455 map_name:=house_my1_map use_sim_time:=false

  # Real robot, single D435i
  ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=d435i map_name:=house_my1_map use_sim_time:=false

  # Custom maps dir
  ros2 launch vf_robot_slam rtabmap_slam.launch.py map_name:=house_my1_map maps_dir:=~/my_maps

  # No RViz
  ros2 launch vf_robot_slam rtabmap_slam.launch.py map_name:=house_my1_map rviz:=false

  # Show args
  ros2 launch vf_robot_slam rtabmap_slam.launch.py --show-args

SAVE 2D MAP FOR NAV2 (while slam is running)
  ros2 run nav2_map_server map_saver_cli -f ~/CA-MCW/maps/house_my1_map/house_my1_map

KEY BEHAVIORS
  • Reg/Strategy=1 (ICP) — continuous odom gives good initial pose for ICP
    (loc uses visual instead — cold-start odom=0,0,0 breaks ICP)
  • Vis/EstimationType=0 (3D-3D not PnP) — avoids OpenGV dependency
  • Map folder auto-created if missing
  • OptimizeMaxError=1.0 — tighter than loc (3.0) to reject bad loop closures
  • parameters=[rtabmap_params, sub_params] — split dicts prevent silent YAML
    short-circuit on slash-keyed params (Mem/IncrementalMemory etc.)

CRITICAL LESSONS
  1. use_sim_time=true for Gazebo — sim stamps ~1000s vs wall ~1.77B,
     mismatch silently drops every RGBD frame
  2. frame_id=base_footprint — Gazebo publishes odom→base_footprint not base_link
  3. depth/camera_info MUST be remapped in rgbd_sync — see rgbd_sync.launch.py
  4. approx_sync_max_interval=0.05 not 0.0 — see rgbd_sync.launch.py
"""

import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _get_rtabmap_slam_params(database_path, delete_db, use_sim_time):
    """All RTAB-Map parameters for SLAM mode — inline, no YAML files."""
    return {
        # ── Time (CRITICAL for Gazebo) ──
        "use_sim_time": use_sim_time,
        # ── Frames ──
        "frame_id": "base_footprint",
        "odom_frame_id": "odom",
        "map_frame_id": "map",
        "publish_tf": True,
        # ── Sync ──
        "queue_size": 30,
        "approx_sync": True,
        # ── Database ──
        "database_path": database_path,
        "delete_db_on_start": delete_db,
        # ── SLAM mode ──
        "Mem/IncrementalMemory": "true",
        "Mem/InitWMWithAllNodes": "false",
        # ── Registration — ICP avoids needing OpenGV for multi-camera ──
        "Reg/Strategy": "1",
        "Vis/EstimationType": "0",
        "Reg/Force3DoF": "true",
        # ── Optimizer ──
        "Optimizer/Strategy": "1",
        "Optimizer/Iterations": "20",
        "Optimizer/Slam2D": "true",
        # ── Visual features ──
        "Vis/MinInliers": "15",
        "Vis/InlierDistance": "0.1",
        "Vis/MaxFeatures": "500",
        "Vis/FeatureType": "8",
        # ── Loop closure ──
        "Rtabmap/DetectionRate": "1.0",
        "Rtabmap/TimeThr": "0.0",
        "Rtabmap/LoopThr": "0.11",
        "RGBD/LoopClosureReextractFeatures": "true",
        "RGBD/OptimizeMaxError": "1.0",
        # ── Mapping thresholds ──
        "RGBD/LinearUpdate": "0.1",
        "RGBD/AngularUpdate": "0.1",
        "RGBD/CreateOccupancyGrid": "true",
        # ── Memory ──
        "Mem/ImageKept": "true",
        "Mem/STMSize": "30",
        # ── Grid map ──
        "Grid/FromDepth": "true",
        "Grid/RayTracing": "true",
        "Grid/3D": "false",
        "Grid/CellSize": "0.05",
        "Grid/RangeMin": "0.0",
        "Grid/RangeMax": "6.0",
        "Grid/MaxGroundHeight": "0.05",
        "Grid/MaxObstacleHeight": "2.0",
    }


def _resolve_map_folder(map_name, maps_dir):
    expanded = os.path.expanduser(map_name)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(os.path.expanduser(maps_dir), map_name)


def launch_setup(context, *args, **kwargs):
    camera = LaunchConfiguration("camera").perform(context)
    map_name = LaunchConfiguration("map_name").perform(context)
    maps_dir = LaunchConfiguration("maps_dir").perform(context)
    rviz = LaunchConfiguration("rviz").perform(context)
    new_map = LaunchConfiguration("new_map").perform(context)
    sim_time = LaunchConfiguration("use_sim_time").perform(context)

    use_sim_time = sim_time.lower() == "true"

    map_folder = _resolve_map_folder(map_name, maps_dir)
    map_base = os.path.basename(map_folder)
    database_path = os.path.join(map_folder, f"{map_base}.db")

    Path(map_folder).mkdir(parents=True, exist_ok=True)

    pkg_share = get_package_share_directory("vf_robot_slam")

    delete_db = new_map.lower() == "true"
    mode_str = "NEW MAP (deleting existing)" if delete_db else "CONTINUING existing map"

    actions = []

    actions.append(
        LogInfo(
            msg=[
                "\n",
                "=" * 70,
                "\n",
                "RTAB-Map SLAM Mode\n",
                "=" * 70,
                "\n",
                f"Camera:        {camera}\n",
                f"Map name:      {map_name}\n",
                f"Map folder:    {map_folder}\n",
                f"Mode:          {mode_str}\n",
                f"Sim time:      {use_sim_time}\n",
                f"Frame ID:      base_footprint\n",
                "\n",
                "Save 2D map while running:\n",
                f"  ros2 run nav2_map_server map_saver_cli -f {map_folder}/{map_base}\n",
                "=" * 70,
                "\n",
            ]
        )
    )

    # ── Include rgbd_sync (dual only) ────────────────────────────────────────
    if camera == "dual":
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_share, "launch", "include", "rgbd_sync.launch.py")
                ),
                launch_arguments={
                    "camera": camera,
                    "use_sim_time": sim_time,
                }.items(),
            )
        )

    # ── RTAB-Map node ────────────────────────────────────────────────────────
    rtabmap_params = _get_rtabmap_slam_params(database_path, delete_db, use_sim_time)

    if camera == "dual":
        # Split dicts — prevents silent YAML short-circuit on slash-keyed params
        sub_params = {
            "subscribe_depth": False,
            "subscribe_rgb": False,
            "subscribe_rgbd": True,
            "rgbd_cameras": 2,
        }
        remappings = [
            ("rgbd_image0", "/rgbd_image/d435i"),
            ("rgbd_image1", "/rgbd_image/d455"),
            ("odom", "/odom"),
            ("map", "/map"),
        ]
        node_parameters = [rtabmap_params, sub_params]
    else:
        # Single camera — subscribe directly to depth+rgb streams
        sub_params = {
            "subscribe_depth": True,
            "subscribe_rgb": True,
            "subscribe_rgbd": False,
        }
        if camera == "d435i":
            remappings = [
                ("rgb/image", "/d435i/rgb/d435i_rgb/image_raw"),
                ("rgb/camera_info", "/d435i/rgb/d435i_rgb/camera_info"),
                ("depth/image", "/d435i/depth/d435i_depth/depth/image_raw"),
                ("depth/camera_info", "/d435i/depth/d435i_depth/depth/camera_info"),
                ("odom", "/odom"),
                ("map", "/map"),
            ]
        else:  # d455
            remappings = [
                ("rgb/image", "/d455/rgb/d455_rgb/image_raw"),
                ("rgb/camera_info", "/d455/rgb/d455_rgb/camera_info"),
                ("depth/image", "/d455/depth/d455_depth/depth/image_raw"),
                ("depth/camera_info", "/d455/depth/d455_depth/depth/camera_info"),
                ("odom", "/odom"),
                ("map", "/map"),
            ]
        node_parameters = [rtabmap_params, sub_params]

    actions.append(
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=node_parameters,
            remappings=remappings,
        )
    )

    # ── RViz ─────────────────────────────────────────────────────────────────
    if rviz.lower() == "true":
        rviz_config = os.path.join(pkg_share, "rviz", "rtabmap_slam.rviz")
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera",
                default_value="dual",
                choices=["d435i", "d455", "dual"],
                description="Camera configuration.",
            ),
            DeclareLaunchArgument(
                "map_name",
                default_value="default_map",
                description=(
                    "Map folder name (resolved to maps_dir/map_name). "
                    "Pass an absolute path (e.g. /tmp/my_map) to save outside maps_dir."
                ),
            ),
            DeclareLaunchArgument(
                "maps_dir",
                default_value="~/CA-MCW/maps",
                description="Base directory where map folders are created (ignored when map_name is absolute).",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                choices=["true", "false"],
                description="Launch RViz.",
            ),
            DeclareLaunchArgument(
                "new_map",
                default_value="true",
                choices=["true", "false"],
                description="Start fresh map (true) or continue existing (false).",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                choices=["true", "false"],
                description="true for Gazebo, false for real robot.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
