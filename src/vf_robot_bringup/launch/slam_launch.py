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
SLAM Toolbox (Laser SLAM) — Include-Only — vf_robot_bringup
═══════════════════════════════════════════════════════════

Purpose
-------
Runs 2D laser-based SLAM using slam_toolbox (online_async mode).
Builds a map from /scan and provides the map→odom transform.

This file is NOT standalone. It is included by:
  • bringup_launch.py (localization:=slam_toolbox)

Arguments
---------
  params_file    : Nav2 parameters YAML (required)
  use_sim_time   : true | false (default: true)
  map_name       : map folder name → maps_dir/<name>
                   (or absolute path)
  maps_dir       : base directory for map folders

Map Resolution
--------------
  map_name is resolved as:
    • absolute path → used directly
    • otherwise     → maps_dir/map_name

  The map is stored using:
    <folder>/<name>.{posegraph,data}

Nodes Started
-------------
  slam_toolbox
    Subscribes to /scan → builds map
    Publishes /map and map→odom TF

  lifecycle_manager_slam
    Autostarts and manages slam_toolbox

Inputs Required
---------------
  • /scan topic (from depth_to_scan.launch.py)
  • /odom and TF: odom → base_footprint (sim or real robot)

Map Saving
----------
  Maps are NOT saved automatically.

  Save manually via:
    • ROS service:
        /slam_toolbox/serialize_map
    • RViz "Save Map" button (slam_toolbox panel)

  Output files:
    <map_name>.posegraph
    <map_name>.data

Behavior Notes
--------------
  • online_async mode allows continuous mapping while navigating
  • map_file_name defines the serialization target (no extension)
  • map directory is created automatically if missing

Failure Modes
-------------
  • Missing /scan or TF → mapping will not function
  • Invalid params_file → node startup failure
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_map_folder(map_name, maps_dir):
    """Return absolute map folder path.

    If map_name is already absolute (starts with / or ~) it is used directly
    as the map folder so the user can point outside ~/CA-MCW/maps.
    Otherwise the folder is maps_dir/map_name.
    The map file basename is always os.path.basename() of the resolved folder.
    """
    expanded = os.path.expanduser(map_name)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(os.path.expanduser(maps_dir), map_name)


def launch_setup(context, *args, **kwargs):
    map_name = LaunchConfiguration("map_name").perform(context)
    maps_dir = LaunchConfiguration("maps_dir").perform(context)
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    map_folder = _resolve_map_folder(map_name, maps_dir)
    map_base = os.path.basename(map_folder)
    # stem path used by slam_toolbox for serialize/deserialize (no extension)
    map_file_name = os.path.join(map_folder, map_base)

    Path(map_folder).mkdir(parents=True, exist_ok=True)

    print(f"[slam_launch] Map folder: {map_folder}")
    print(f"[slam_launch] Serialize target: {map_file_name}.{{posegraph,data}}")

    # ── SLAM Toolbox (online async) ──
    # Builds a 2D occupancy grid from /scan.
    # Publishes /map and map→odom TF.
    # map_file_name tells slam_toolbox where to serialize on service call.

    slam_toolbox_node = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "map_file_name": map_file_name,
            },
        ],
    )

    # ── Lifecycle Manager ──

    lifecycle_manager_slam = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_slam",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": [
                    "slam_toolbox",
                ],
            }
        ],
    )

    return [slam_toolbox_node, lifecycle_manager_slam]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", description="Nav2 params YAML"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "map_name",
                default_value="default_map",
                description=(
                    "Map folder name (resolved to maps_dir/map_name). "
                    "Pass an absolute path to save outside maps_dir."
                ),
            ),
            DeclareLaunchArgument(
                "maps_dir",
                default_value=os.path.join(
                    os.path.expanduser("~"), "CA-MCW", "maps"
                ),
                description="Base directory for map folders (ignored when map_name is absolute).",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
