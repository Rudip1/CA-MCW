"""
evaluation_goalposes_collect.launch.py
--------------------------------------
Record start + waypoints for EVALUATION batch runs. No robot movement,
no planner — just RViz + pose_recorder.

Each Ctrl-C appends a new row (run_id auto-increments) to:
  <maps_dir>/<map_name>/evaluation_goalposes_collect.csv

CSV format (wide, one tour per row):
  run_id, notes, start_x, start_y, start_yaw, g1_x, g1_y, g1_yaw, g2_x, …

Use the row's run_id when invoking the evaluation batch launches.

Starts:
  1. map_server        – map for visual reference only
  2. lifecycle_manager – brings map_server active
  3. rviz2             – standard vf_bringup.rviz
  4. pose_recorder     – writes CSV on Ctrl-C

Usage:
  ros2 launch vf_robot_utils evaluation_goalposes_collect.launch.py map_name:=house_my1_map
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from vf_robot_utils.constants import MAPS_ROOT


def generate_launch_description():

    map_name_arg = DeclareLaunchArgument(
        'map_name',
        description='Map subfolder name, e.g. house_my1_map, my_map, test_map')
    maps_dir_arg = DeclareLaunchArgument(
        'maps_dir', default_value=str(MAPS_ROOT),
        description='Root maps directory')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false')

    map_name     = LaunchConfiguration('map_name')
    maps_dir     = LaunchConfiguration('maps_dir')
    use_sim_time = LaunchConfiguration('use_sim_time')

    map_yaml = PathJoinSubstitution(
        [maps_dir, map_name, PythonExpression(["'", map_name, "' + '.yaml'"])])

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'yaml_filename': map_yaml}],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server'],
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution(
            [FindPackageShare('vf_robot_bringup'), 'rviz', 'vf_bringup.rviz'])],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    pose_recorder_node = Node(
        package='vf_robot_utils',
        executable='pose_recorder',
        name='pose_recorder',
        output='screen',
        parameters=[{
            'use_sim_time':  use_sim_time,
            'map_name':      map_name,
            'output_dir':    maps_dir,
            'csv_filename':  'evaluation_goalposes_collect.csv',
        }],
    )

    return LaunchDescription([
        map_name_arg, maps_dir_arg, use_sim_time_arg,
        LogInfo(msg=['[evaluation_goalposes_collect] map: ', map_yaml,
                     '  →  evaluation_goalposes_collect.csv']),
        map_server_node,
        lifecycle_manager,
        rviz_node,
        pose_recorder_node,
    ])
