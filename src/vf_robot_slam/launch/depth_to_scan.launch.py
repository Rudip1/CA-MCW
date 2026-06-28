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
Depth → LaserScan — Router Launch File — ViroFighter UVC-1
══════════════════════════════════════════════════════════

Single entry point for all depth-to-laserscan conversion.
Routes to the correct sub-launch based on the `method` argument.
Callers (e.g. vf_robot_bringup) never need to know which method is active.

────────────────────────────────────────────────────────────
ARGUMENTS
────────────────────────────────────────────────────────────
  method        : dimg | pc2scan         (default: pc2scan)
  camera        : d435i | d455 | dual    (default: dual)
  merge_scans   : true | false           (default: true)
  use_sim_time  : true | false           (default: true)

────────────────────────────────────────────────────────────
METHODS
────────────────────────────────────────────────────────────
  dimg
    Sub-launch : include/depth_to_scan_dimg.launch.py
    Converts a 2D depth image row → /scan.
    Fast in Gazebo (no PointCloud2 overhead). Recommended for sim.
    D435i blind zone < 1.1 m due to 60° tilt floor intersection.

  pc2scan
    Sub-launch : include/depth_to_scan_pc2scan.launch.py
    Converts a 3D PointCloud2 → /scan via world-space Z height filter.
    No floor blind zone for D435i. Recommended for real robot.
    May run slower in Gazebo (~2 Hz) due to pointcloud overhead.

────────────────────────────────────────────────────────────
OUTPUT TOPICS
────────────────────────────────────────────────────────────
  camera=d435i                          →  /scan
  camera=d455                           →  /scan
  camera=dual, merge_scans=true         →  /scan_d435i + /scan_d455 + /scan
  camera=dual, merge_scans=false        →  /scan_d435i + /scan_d455 only
                                            (no /scan — use Nav2 multi-source costmap)

────────────────────────────────────────────────────────────
ALL VALID COMBINATIONS
────────────────────────────────────────────────────────────
  Simulation (use_sim_time=true, default):
    method:=dimg   camera:=dual                              ← recommended
    method:=dimg   camera:=dual   merge_scans:=false
    method:=dimg   camera:=d455
    method:=dimg   camera:=d435i
    method:=pc2scan camera:=dual                             ← slow in sim
    method:=pc2scan camera:=dual  merge_scans:=false
    method:=pc2scan camera:=d455
    method:=pc2scan camera:=d435i

  Real robot (use_sim_time:=false):
    method:=pc2scan camera:=dual  use_sim_time:=false        ← recommended
    method:=pc2scan camera:=dual  use_sim_time:=false  merge_scans:=false
    method:=pc2scan camera:=d455  use_sim_time:=false
    method:=pc2scan camera:=d435i use_sim_time:=false
    method:=dimg   camera:=dual   use_sim_time:=false        ← D435i blind zone
    method:=dimg   camera:=d455   use_sim_time:=false
    method:=dimg   camera:=d435i  use_sim_time:=false

────────────────────────────────────────────────────────────
USAGE EXAMPLES
────────────────────────────────────────────────────────────
  # Gazebo — recommended
  ros2 launch vf_robot_slam depth_to_scan.launch.py method:=dimg camera:=dual

  # Gazebo — no merged scan (Nav2 reads each source separately)
  ros2 launch vf_robot_slam depth_to_scan.launch.py method:=dimg camera:=dual merge_scans:=false

  # Real robot — recommended
  ros2 launch vf_robot_slam depth_to_scan.launch.py method:=pc2scan camera:=dual use_sim_time:=false

  # Real robot — single D455
  ros2 launch vf_robot_slam depth_to_scan.launch.py method:=pc2scan camera:=d455 use_sim_time:=false

  # Show all arguments
  ros2 launch vf_robot_slam depth_to_scan.launch.py --show-args

────────────────────────────────────────────────────────────
INTEGRATION — vf_robot_bringup
────────────────────────────────────────────────────────────
  Bringup only calls THIS file. It does not reference the sub-launches.
  Convention: sim_bringup passes method:=dimg, robot_bringup passes method:=pc2scan.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    method = LaunchConfiguration("method").perform(context)
    camera = LaunchConfiguration("camera").perform(context)
    merge_scans = LaunchConfiguration("merge_scans").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)

    pkg_share = get_package_share_directory("vf_robot_slam")

    if method == "dimg":
        include_file = os.path.join(
            pkg_share, "launch", "include", "depth_to_scan_dimg.launch.py"
        )
    else:  # pc2scan
        include_file = os.path.join(
            pkg_share, "launch", "include", "depth_to_scan_pc2scan.launch.py"
        )

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(include_file),
            launch_arguments={
                "camera": camera,
                "merge_scans": merge_scans,
                "use_sim_time": use_sim_time,
            }.items(),
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "method",
                default_value="pc2scan",
                choices=["dimg", "pc2scan"],
                description=(
                    "Conversion method. "
                    "dimg: depthimage_to_laserscan (fast in Gazebo, D435i blind zone < 1.1 m). "
                    "pc2scan: pointcloud_to_laserscan (accurate, no blind zone, slow in Gazebo)."
                ),
            ),
            DeclareLaunchArgument(
                "camera",
                default_value="dual",
                choices=["d435i", "d455", "dual"],
                description="Camera configuration.",
            ),
            DeclareLaunchArgument(
                "merge_scans",
                default_value="true",
                choices=["true", "false"],
                description=(
                    "Merge dual scans into /scan. "
                    "Uses ira_laser_tools if installed, otherwise relays D455 → /scan."
                ),
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
