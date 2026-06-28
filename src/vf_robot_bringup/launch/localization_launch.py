#!/usr/bin/env python3

"""
AMCL Localization — Include-Only — vf_robot_bringup
════════════════════════════════════════════════════

Purpose
-------
Runs 2D localization using Nav2 AMCL with a pre-built occupancy grid map.
Provides the map→odom transform required by Nav2.

This file is NOT standalone. It is included by:
  • bringup_launch.py (localization:=amcl)

Arguments
---------
  map            : full path to .yaml map file (takes priority)
  map_name       : map folder name → maps_dir/<name>/<name>.yaml
  maps_dir       : base directory for map folders
  params_file    : composed Nav2 parameters file (required)
  use_sim_time   : true | false (default: true)

Map Resolution
--------------
  Resolution priority:
    1. map:=<full_path>        → used directly
    2. map_name:=<name>        → resolved via maps_dir

  If neither produces a valid file, launch fails immediately.

Nodes Started
-------------
  map_server
    Loads occupancy grid (.pgm/.yaml) → publishes /map

  amcl
    Subscribes to /scan + /map → publishes map→odom TF

  lifecycle_manager_localization
    Autostarts and manages both nodes (map_server → amcl order)

Inputs Required
---------------
  • /scan topic (from depth_to_scan.launch.py)
  • /odom and TF: odom → base_footprint (sim or real robot)
  • Valid map file (.yaml + .pgm)

Behavior Notes
--------------
  • map_server must become active before AMCL initializes
  • Initial pose must be set manually in RViz ("2D Pose Estimate")
  • params_file supplies all Nav2 tuning (AMCL + map_server)

Failure Modes
-------------
  • Missing map file → immediate RuntimeError
  • Missing /scan or TF → AMCL will not converge
"""

import os

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
    map_arg = LaunchConfiguration("map").perform(context)
    map_name = LaunchConfiguration("map_name").perform(context)
    maps_dir = LaunchConfiguration("maps_dir").perform(context)
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Resolve the yaml path: explicit map path wins; fall back to map_name.
    if map_arg:
        map_yaml = map_arg
    elif map_name:
        map_folder = _resolve_map_folder(map_name, maps_dir)
        map_base = os.path.basename(map_folder)
        map_yaml = os.path.join(map_folder, f"{map_base}.yaml")
    else:
        raise RuntimeError(
            "[localization_launch] No map specified. "
            "Provide map_name:=<name> or map:=<full_path_to_yaml>."
        )

    if not os.path.exists(map_yaml):
        raise RuntimeError(
            f"[localization_launch] Map file not found: {map_yaml}\n"
            "Build or export the map first, then re-launch."
        )

    print(f"[localization_launch] Loading map: {map_yaml}")

    # ── map_server ──
    # Loads the 2D occupancy grid from .pgm/.yaml and publishes on /map.
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "yaml_filename": map_yaml,
            },
        ],
    )

    # ── AMCL ──
    # Adaptive Monte Carlo Localization.
    # Subscribes to /scan and /map, publishes map→odom TF.
    # User provides initial pose via RViz "2D Pose Estimate" button.
    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            params_file,
            {"use_sim_time": use_sim_time},
        ],
    )

    # ── Lifecycle Manager ──
    # map_server must be active before AMCL (AMCL needs /map to initialize).
    lifecycle_manager_localization = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": [
                    "map_server",
                    "amcl",
                ],
            }
        ],
    )

    return [map_server, amcl, lifecycle_manager_localization]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value="",
                description=(
                    "Full path to map .yaml file. "
                    "Takes priority over map_name when set."
                ),
            ),
            DeclareLaunchArgument(
                "map_name",
                default_value="default_map",
                description=(
                    "Map folder name (resolved to maps_dir/map_name/map_name.yaml). "
                    "Pass an absolute path to load from outside maps_dir. "
                    "Ignored when map is set."
                ),
            ),
            DeclareLaunchArgument(
                "maps_dir",
                default_value=os.path.join(
                    os.path.expanduser("~"), "CA-MCW", "maps"
                ),
                description="Base directory containing map folders (ignored when map_name is absolute or map is set).",
            ),
            DeclareLaunchArgument("params_file", description="Nav2 params YAML"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            OpaqueFunction(function=launch_setup),
        ]
    )
