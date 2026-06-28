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
# Hospital my1 world (SDF pipeline) — robot loaded from model.sdf in vf_robot_gazebo/models/uvc1_virofighter
#
# Node graph:
#   gzserver + gzclient       → Gazebo simulation
#   vf_spawn_sdf              → spawns robot from model.sdf
#   vf_robot_state_publisher  → publishes TF tree from xacro (needed by RViz)
#   rqt_robot_steering        → teleop GUI
#
# World file:    worlds/hospital_my1_world/hospital_my1.world
# Env models:    models/hospital_my1_models
# Robot models:  models/uvc1_common, models/uvc1_virofighter (added via base models/ path)
#
# Run:  ros2 launch vf_robot_gazebo hospital_my1_world_sdf.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_vf_gazebo = get_package_share_directory("vf_robot_gazebo")
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    launch_dir = os.path.join(pkg_vf_gazebo, "launch")

    # ── Launch arguments ───────────────────────────────────────────────────
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock",
    )
    declare_x_pose = DeclareLaunchArgument(
        "x_pose",
        default_value="0.0",
        description="X spawn position",
    )
    declare_y_pose = DeclareLaunchArgument(
        "y_pose",
        default_value="0.0",
        description="Y spawn position",
    )
    declare_theta = DeclareLaunchArgument(
        "theta",
        default_value="0.0",
        description="Yaw spawn angle (radians)",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    theta = LaunchConfiguration("theta")

    world = os.path.join(
        pkg_vf_gazebo, "worlds", "hospital_my1_world", "hospital_vf.world"
    )

    # ── Environment variables — PREPEND, never replace ────────────────────
    # SDF uses model:// URIs which need GAZEBO_MODEL_PATH set correctly.
    gazebo_model_path = SetEnvironmentVariable(
        name="GAZEBO_MODEL_PATH",
        value=os.pathsep.join(
            [
                os.path.join(
                    pkg_vf_gazebo, "models", "hospital_my1_models"
                ),  # env-specific assets
                os.path.join(
                    pkg_vf_gazebo, "models"
                ),  # uvc1_common, uvc1_virofighter, …
                "/usr/share/gazebo-11/models",  # Gazebo built-ins
            ]
        ),
    )
    gazebo_plugin_path = SetEnvironmentVariable(
        name="GAZEBO_PLUGIN_PATH",
        value=os.pathsep.join(
            [
                "/opt/ros/humble/lib",
                "/usr/lib/x86_64-linux-gnu/gazebo-11/plugins",
            ]
        ),
    )
    gazebo_resource_path = SetEnvironmentVariable(
        name="GAZEBO_RESOURCE_PATH",
        value=os.pathsep.join(
            [
                "/usr/share/gazebo-11",
                "/opt/ros/humble/share",
            ]
        ),
    )

    # ── Gazebo server + client ─────────────────────────────────────────────
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world}.items(),
    )
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        )
    )

    # ── Spawn robot from SDF ───────────────────────────────────────────────
    spawn_robot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, "vf_spawn_sdf.launch.py")
        ),
        launch_arguments={
            "x_pose": x_pose,
            "y_pose": y_pose,
            "theta": theta,
        }.items(),
    )

    # ── Robot state publisher — publishes TF tree from xacro ───────────────
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, "vf_robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # ── Teleop GUI ─────────────────────────────────────────────────────────
    gui_teleop_node = Node(
        package="rqt_robot_steering",
        executable="rqt_robot_steering",
        name="rqt_robot_steering",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    ld = LaunchDescription()

    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_x_pose)
    ld.add_action(declare_y_pose)
    ld.add_action(declare_theta)

    ld.add_action(gazebo_model_path)
    ld.add_action(gazebo_plugin_path)
    ld.add_action(gazebo_resource_path)

    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_robot_cmd)
    ld.add_action(gui_teleop_node)

    return ld
