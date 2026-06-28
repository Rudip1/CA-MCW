#!/usr/bin/env python3
"""
RViz2 — vf_robot_bringup
═════════════════════════

Purpose
-------
Launches RViz2 with the vf_bringup.rviz config which includes Nav2 panels
(2D Pose Estimate, Nav2 Goal, costmap visualisations).

Can be launched standalone for debugging, or is included automatically by
bringup_launch.py when rviz:=true (the default).

Arguments
---------
  rviz_config    : full path to .rviz config file
                   (default: vf_robot_bringup/rviz/vf_bringup.rviz)
  use_sim_time   : true | false (default: true)

Nodes Started
-------------
  rviz2          : RViz2 visualisation node

Usage
-----
  # Standalone
  ros2 launch vf_robot_bringup rviz_launch.py

  # With custom config
  ros2 launch vf_robot_bringup rviz_launch.py \
      rviz_config:=/path/to/my_config.rviz

  # Real robot
  ros2 launch vf_robot_bringup rviz_launch.py use_sim_time:=false

RViz Panel Notes
----------------
  2D Pose Estimate : publishes /initialpose — required for AMCL initial pose
  Nav2 Goal        : publishes /goal_pose — click on map to send navigation goals
  Fixed Frame      : set to "map" once a map is available; use "odom" before that
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_bringup = get_package_share_directory("vf_robot_bringup")

    default_rviz = os.path.join(pkg_bringup, "rviz", "vf_bringup.rviz")

    declare_rviz_config = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz,
        description="Full path to RViz config file",
    )

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    return LaunchDescription(
        [
            declare_rviz_config,
            declare_use_sim_time,
            rviz2,
        ]
    )
