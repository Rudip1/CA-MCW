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
# Publishes /robot_description from the xacro single source of truth in
# vf_robot_description. Both SDF-mode and xacro-mode Gazebo launches include
# this file — it provides the TF tree (base_link and below) that RViz needs.
#
# Why xacro and not a static URDF:
#   uvc1_virofighter.xacro is the single source of truth (includes
#   sensors.xacro + common_properties.xacro). Processing it here keeps
#   vf_robot_gazebo in sync with vf_robot_description automatically — no
#   manual conversion step is needed for TF tree publishing.
#
# Why both pipelines need this:
#   - xacro pipeline: spawn_entity reads /robot_description from this RSP.
#   - sdf  pipeline: the spawned model.sdf only carries the diff_drive
#                    plugin (odom → base_footprint). Everything below
#                    base_link still comes from this RSP.
#
# Included by every per-env launcher under:
#   launch/<env>_launch/<env>_world_{sdf,xacro}.launch.py
# (where <env> is empty_world, hospital_euroknows, hospital_my1, hospital_my2,
#  hospital_Tommaso_hospital, or house_my1).
#
# Standalone (rare, for debugging only):
#   ros2 launch vf_robot_gazebo vf_robot_state_publisher.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    Command,
    FindExecutable,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # xacro path — urdf/xacro/ subfolder layout
    xacro_path = PathJoinSubstitution(
        [
            FindPackageShare("vf_robot_description"),
            "urdf",
            "xacro",
            "uvc1_virofighter.xacro",
        ]
    )

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # ParameterValue with value_type=str prevents ROS2 Humble from trying
    # to parse the URDF XML output as YAML (which causes a launch crash).
    robot_description_content = ParameterValue(
        Command(
            [
                FindExecutable(name="xacro"),
                " ",
                xacro_path,
            ]
        ),
        value_type=str,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation (Gazebo) clock if true",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "robot_description": robot_description_content,
                    }
                ],
            ),
        ]
    )
