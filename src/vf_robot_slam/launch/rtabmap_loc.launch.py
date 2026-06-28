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
RTAB-Map Localization — ViroFighter UVC-1
══════════════════════════════════════════

Localizes within a previously built map using RTAB-Map visual localization.
Does NOT add new nodes to the map. Requires a .db file from a prior SLAM session.

────────────────────────────────────────────────────────────
ARGUMENTS
────────────────────────────────────────────────────────────
  camera        : d435i | d455 | dual    (default: dual)
  map_name      : folder name or absolute path  (default: default_map)
  maps_dir      : base directory for maps (default: ~/CA-MCW/maps)
  rviz          : true | false           (default: true)
  use_sim_time  : true | false           (default: true)

  map_name resolution:
    Relative → maps_dir/map_name/map_name.db
    Absolute → map_name/map_name.db  (maps_dir is ignored)

────────────────────────────────────────────────────────────
PREREQUISITES
────────────────────────────────────────────────────────────
  Map database must exist at:
      ~/CA-MCW/maps/<map_name>/<map_name>.db

  Build a map first with:
      ros2 launch vf_robot_slam rtabmap_slam.launch.py camera:=dual map_name:=house_my1_map

────────────────────────────────────────────────────────────
USAGE EXAMPLES
────────────────────────────────────────────────────────────
  # Gazebo — dual cameras (recommended)
  ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=dual map_name:=house_my1_map

  # Gazebo — single camera
  ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=d455 map_name:=house_my1_map

  # Real robot
  ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=dual map_name:=house_my1_map use_sim_time:=false

  # Custom maps directory
  ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=dual map_name:=house_my1_map maps_dir:=~/my_maps

  # Absolute map path
  ros2 launch vf_robot_slam rtabmap_loc.launch.py camera:=dual map_name:=/tmp/my_maps/office

# Show all arguments
  ros2 launch vf_robot_slam rtabmap_loc.launch.py --show-args

────────────────────────────────────────────────────────────
INCLUDES
────────────────────────────────────────────────────────────
  include/rgbd_sync.launch.py — spawned when camera:=dual only

────────────────────────────────────────────────────────────
KEY BEHAVIORS
────────────────────────────────────────────────────────────
  • /map published immediately on startup via service call after 3 s delay,
    so Nav2 can activate without waiting for the first visual loop closure.

  • Visual loop closure only (Reg/Strategy=0, NOT ICP).
    ICP requires a good initial pose — on every fresh Gazebo session odom
    starts at (0,0,0), making the initial map→odom guess wrong by an
    arbitrary amount. ICP diverges. Visual closure matches feature
    descriptors globally with no initial pose assumption.

  • Map is frozen: Mem/IncrementalMemory=false, RGBD/LinearUpdate=0.0,
    RGBD/AngularUpdate=0.0.

  • Map origin anchored to first node (RGBD/OptimizeFromGraphEnd=false)
    to prevent map rotation or jitter on loop closure.

  • parameters=[rtabmap_params, sub_params] — two separate dicts.
    Subscription params (subscribe_rgbd, rgbd_cameras) are isolated in
    a clean second dict to prevent silent short-circuit from YAML
    validation on slash-keyed rtabmap params (e.g. Mem/IncrementalMemory).

────────────────────────────────────────────────────────────
CRITICAL LESSONS (FROM DEBUGGING)
────────────────────────────────────────────────────────────
  1. Reg/Strategy MUST be 0 (visual) for cold-start relocalization.
     ICP was correct during SLAM (continuous odom = good initial guess)
     but breaks localization from a cold start in Gazebo.

  2. RGBD/OptimizeFromGraphEnd=false — was true, caused map to
     rotate/jitter on the first loop closure after startup.

  3. RGBD/OptimizeMaxError=3.0 — was 1.0, too tight, was rejecting
     valid first-lock loop closures where odom had drifted at startup.

  4. Rtabmap/LoopThr=0.11 — was 0.15 (too strict, missed first lock).
     0.09 caused false positives. 0.11 is the verified safe middle.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    LogInfo,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
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


def _get_rtabmap_loc_params(database_path, use_sim_time):
    """All RTAB-Map parameters for Localization mode — inline, no YAML files."""
    return {
        # ── Time (CRITICAL for Gazebo) ──
        "use_sim_time": use_sim_time,
        # ── Frames ──
        "frame_id": "base_footprint",  # must match odom child_frame_id
        "odom_frame_id": "odom",
        "map_frame_id": "map",
        "publish_tf": True,
        # ── Sync ──
        "queue_size": 30,
        "approx_sync": True,
        # ── Database ──
        "database_path": database_path,
        # ── Localization mode — do not add new nodes to the map ──
        "Mem/IncrementalMemory": "true",
        "Mem/InitWMWithAllNodes": "false",
        # ── Optimizer ──
        "Optimizer/Strategy": "1",
        "Optimizer/Iterations": "20",
        "Optimizer/Slam2D": "true",
        # ── Registration ──────────────────────────────────────────────────────
        # MUST be visual (0) for localization, NOT ICP (1).
        #
        # ICP needs a good initial pose to converge. On every fresh Gazebo
        # session the robot's odom starts at (0,0,0), making the initial
        # map→odom guess (= last saved transform) wrong by an arbitrary amount.
        # ICP diverges, loop closure is rejected, map→odom never corrects.
        #
        # Visual loop closure matches feature descriptors globally — no initial
        # pose needed. It reliably relocates the robot within 1–3 seconds of
        # startup. ICP was appropriate during SLAM (continuous odometry = good
        # initial guess), but it breaks localization from a cold start.
        "Reg/Strategy": "0",  # 0=visual (REQUIRED for cold-start relocalization)
        "Vis/EstimationType": "1",  # 1=PnP (2D→3D) — more robust than 3D-3D for reloc
        "Reg/Force3DoF": "true",
        # ── Visual features ──
        "Vis/MinInliers": "12",  # was 15 — slightly easier to get first lock
        "Vis/InlierDistance": "0.1",
        "Vis/MaxFeatures": "500",
        "Vis/FeatureType": "8",
        # ── Loop closure ──
        "Rtabmap/DetectionRate": "2.0",
        "Rtabmap/LoopThr": "0.09",  # was 0.15 (too strict → misses first lock)
        # 0.09 caused false positives; 0.11 is the safe middle
        "RGBD/LoopClosureReextractFeatures": "true",
        "RGBD/OptimizeMaxError": "3.0",  # was 1.0 — too tight, was rejecting valid first-lock
        # loop closures where odom had drifted during startup
        # ── Map is static — do not update ──
        "RGBD/LinearUpdate": "0.0",
        "RGBD/AngularUpdate": "0.0",
        # ── Memory — keep working memory small ──
        "Mem/ImageKept": "false",
        "Mem/STMSize": "10",
        # ── Anchor map origin to first node — prevents map rotation on loop closure ──
        "RGBD/OptimizeFromGraphEnd": "false",  # was true — caused map to rotate/jitter
    }


def launch_setup(context, *args, **kwargs):
    camera = LaunchConfiguration("camera").perform(context)
    map_name = LaunchConfiguration("map_name").perform(context)
    maps_dir = LaunchConfiguration("maps_dir").perform(context)
    rviz = LaunchConfiguration("rviz").perform(context)
    sim_time = LaunchConfiguration("use_sim_time").perform(context)

    use_sim_time = sim_time.lower() == "true"

    map_folder = _resolve_map_folder(map_name, maps_dir)
    map_base = os.path.basename(map_folder)
    database_path = os.path.join(map_folder, f"{map_base}.db")

    pkg_share = get_package_share_directory("vf_robot_slam")

    actions = []

    # ── Guard: check database exists ─────────────────────────────────────────
    if not os.path.exists(database_path):
        actions.append(
            LogInfo(
                msg=[
                    "\n",
                    "=" * 70,
                    "\n",
                    "ERROR: Map database not found!\n",
                    "=" * 70,
                    "\n",
                    f"Expected:  {database_path}\n",
                    "\nBuild the map first:\n",
                    f"  ros2 launch vf_robot_slam rtabmap_slam.launch.py "
                    f"camera:={camera} map_name:={map_name}\n",
                    "=" * 70,
                    "\n",
                ]
            )
        )
        return actions

    actions.append(
        LogInfo(
            msg=[
                "\n",
                "=" * 70,
                "\n",
                "RTAB-Map Localization Mode\n",
                "=" * 70,
                "\n",
                f"Camera:        {camera}\n",
                f"Map name:      {map_name}\n",
                f"Database:      {database_path}\n",
                f"Sim time:      {use_sim_time}\n",
                f"Frame ID:      base_footprint\n",
                "=" * 70,
                "\n",
            ]
        )
    )

    # ── Include rgbd_sync (shared module) ────────────────────────────────────
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
    rtabmap_params = _get_rtabmap_loc_params(database_path, use_sim_time)

    if camera == "dual":
        # Split into two dicts: rtabmap string params first, then ROS2-native
        # subscription params second.  ROS2 applies the list in order and later
        # dicts override earlier ones, so subscribe_rgbd/rgbd_cameras are
        # guaranteed to be set last regardless of what the string-param dict
        # contains.  Mixing them in one dict risks silent short-circuit when a
        # key with '/' triggers YAML validation on some launch_ros versions.
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

    # ── Publish map immediately after DB load ─────────────────────────────────
    # In localization mode rtabmap won't publish /map (or the map→odom TF that
    # Nav2 waits for) until the first visual loop closure.  Calling
    # publish_map after a short delay forces it to emit the stored graph as
    # an occupancy grid right away so Nav2 can activate.  3 s is enough for
    # the node to start; the DB may still be loading for large maps but
    # rtabmap will publish whatever is in working memory and update it later.
    actions.append(
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2",
                        "service",
                        "call",
                        "/rtabmap/publish_map",
                        "rtabmap_msgs/srv/PublishMap",
                        "{}",
                    ],
                    output="screen",
                )
            ],
        )
    )

    # ── RViz ─────────────────────────────────────────────────────────────────
    if rviz.lower() == "true":
        rviz_config = os.path.join(pkg_share, "rviz", "rtabmap_loc.rviz")
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
                    "Pass an absolute path (e.g. /tmp/my_map) to load from outside maps_dir."
                ),
            ),
            DeclareLaunchArgument(
                "maps_dir",
                default_value="~/CA-MCW/maps",
                description="Base directory containing map folders (ignored when map_name is absolute).",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                choices=["true", "false"],
                description="Launch RViz.",
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
