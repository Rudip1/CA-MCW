"""
imitation_inference_launch.py — Python sidecar for vf_imitationwt mode.

Launches imitation_inference_node.py which subscribes to /vf/features,
runs the trained imitation model via onnxruntime, and publishes /cmd_vel_nav.
VFController returns zero-twist in imitationwt mode; the sidecar drives
the robot directly.

This file is the canonical source of default imitation weights.
Other launch files read defaults from model_defaults.py.

  imitation_weights : run folder name inside imitation_wt/, e.g. "manual_v1"
  onnx_path         : escape hatch — when non-empty, bypasses folder resolution

Usage — default weights:
  ros2 launch vf_robot_controller imitation_inference_launch.py

Usage — switch run:
  ros2 launch vf_robot_controller imitation_inference_launch.py \\
      imitation_weights:=run2_2026_05_12

Usage — fully custom path:
  ros2 launch vf_robot_controller imitation_inference_launch.py \\
      onnx_path:=/abs/path/to/imitation.onnx

Verify:
  ros2 topic hz /cmd_vel_nav        # ~20 Hz
  ros2 topic echo /cmd_vel_nav --once
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from vf_controller.model_defaults import DEFAULT_IMITATION_WEIGHTS


def _resolve_paths(context, *args, **kwargs):
    imitation_weights = LaunchConfiguration("imitation_weights").perform(context)
    onnx_override     = LaunchConfiguration("onnx_path").perform(context)
    use_sim_time      = LaunchConfiguration("use_sim_time").perform(context)
    publish_topic     = LaunchConfiguration("publish_topic").perform(context)

    if onnx_override:
        resolved_onnx = onnx_override
    else:
        pkg = get_package_share_directory("vf_robot_controller")
        run_dir = os.path.join(pkg, "models", "imitation_wt", imitation_weights)
        resolved_onnx = os.path.join(run_dir, "imitation.onnx")

    node = Node(
        package="vf_robot_controller",
        executable="imitation_inference_node.py",
        name="imitation_inference_node",
        parameters=[{
            "onnx_path":     resolved_onnx,
            "use_sim_time":  use_sim_time == "true",
            "publish_topic": publish_topic,
        }],
        output="screen",
    )
    return [node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "imitation_weights", default_value=DEFAULT_IMITATION_WEIGHTS,
            description="Run folder name inside models/imitation_wt/, e.g. manual_v1",
        ),
        DeclareLaunchArgument(
            "onnx_path", default_value="",
            description="Escape hatch: absolute path to imitation.onnx; "
                        "bypasses imitation_weights.",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            choices=["true", "false"],
        ),
        DeclareLaunchArgument(
            "publish_topic", default_value="/cmd_vel_nav",
            description="Topic on which twist commands are published.",
        ),
        OpaqueFunction(function=_resolve_paths),
    ])
