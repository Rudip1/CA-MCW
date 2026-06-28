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
# Display robot in RViz using xacro model — processed at launch time.
# Changes to any included .xacro file are reflected on every new launch
# without a full colcon rebuild (install with --symlink-install).
#
# Node graph:
#   robot_state_publisher     → processes xacro → publishes /robot_description + TF tree
#   joint_state_publisher_gui → GUI sliders for all non-fixed joints
#   rviz2                     → visualizes robot model and TF frames
#
# NOTE: joint_state_publisher_gui can be replaced with headless
#   joint_state_publisher by passing use_gui:=false at launch time.
#   This is useful when joint states come from a real robot or Gazebo.
#
# Run:  ros2 launch vf_robot_description display_robot_xacro.launch.py
#       ros2 launch vf_robot_description display_robot_xacro.launch.py \
#           use_gui:=false use_rviz:=true

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_share = get_package_share_directory("vf_robot_description")

    # ── launch arguments ──────────────────────────────────────────────────────
    declare_xacro_file = DeclareLaunchArgument(
        "xacro_file",
        default_value="uvc1_virofighter.xacro",
        description="Xacro filename inside urdf/xacro/ directory",
    )

    declare_use_gui = DeclareLaunchArgument(
        "use_gui",
        default_value="true",
        description="Launch joint_state_publisher_gui instead of headless",
    )

    declare_use_rviz = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz2",
    )

    declare_rviz_config = DeclareLaunchArgument(
        "rviz_config",
        default_value=os.path.join(pkg_share, "rviz", "model.rviz"),
        description="Full path to the RViz config file",
    )

    # ── resolved paths ────────────────────────────────────────────────────────
    xacro_file_path = PathJoinSubstitution(
        [
            FindPackageShare("vf_robot_description"),
            "urdf",
            "xacro",
            LaunchConfiguration("xacro_file"),
        ]
    )

    # Process xacro at launch time → produces the URDF XML string
    # IMPORTANT: wrap in ParameterValue(value_type=str) to prevent ROS2 Humble
    # from trying to parse the XML as YAML — which causes a launch crash.
    robot_description_content = ParameterValue(
        Command(
            [
                FindExecutable(name="xacro"),
                " ",
                xacro_file_path,
            ]
        ),
        value_type=str,
    )

    robot_description_param = {
        "robot_description": robot_description_content,
    }

    # ── nodes ─────────────────────────────────────────────────────────────────

    # 1. robot_state_publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description_param,
            {"use_sim_time": False},
        ],
    )

    # 2a. joint_state_publisher_gui (interactive sliders)
    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_gui")),
    )

    # 2b. joint_state_publisher (headless, zero positions)
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("use_gui")),
    )

    # 3. rviz2
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    return LaunchDescription(
        [
            declare_xacro_file,
            declare_use_gui,
            declare_use_rviz,
            declare_rviz_config,
            robot_state_publisher_node,
            joint_state_publisher_gui_node,
            joint_state_publisher_node,
            rviz_node,
        ]
    )
