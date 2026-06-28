"""
metacritic_inference_launch.py — Python sidecar for vf_inferencewt mode.

Launches metacritic_inference_node.py which subscribes to /vf/features,
runs the trained meta-critic via onnxruntime, and publishes
/vf_controller/meta_weights at ~20 Hz.

This sidecar is for diagnostic / standalone use. In a normal
vf_inferencewt run the C++ OnnxWeightProvider runs the same ONNX
in-process and drives MPPI directly — the sidecar is not in the critical
path. All launch files (bringup_launch.py, collect/inferencewt_data.launch.py)
read defaults from vf_controller/model_defaults.py.

  inference_model_type  : "raw" | "oracle" — selects metacritic_raw_wt/ or
                          metacritic_oracle_wt/ as the model family root.
  inference_weights     : run folder name inside that family root, e.g.
                          "raw_manual_v1" or "oracle_manual_v1".
  onnx_path             : escape hatch — when non-empty, skips folder
                          resolution entirely and uses this literal path.

Usage — default weights (model_defaults.py picks the folder):
  ros2 launch vf_robot_controller metacritic_inference_launch.py

Usage — switch model type:
  ros2 launch vf_robot_controller metacritic_inference_launch.py \\
      inference_model_type:=oracle inference_weights:=oracle_manual_v1

Usage — fully custom path:
  ros2 launch vf_robot_controller metacritic_inference_launch.py \\
      onnx_path:=/abs/path/to/meta_critic.onnx

Verify it is working (in another terminal):
  ros2 topic hz /vf_controller/meta_weights        # ~20 Hz
  ros2 topic echo /vf_controller/meta_weights --once
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from vf_controller.model_defaults import (
    DEFAULT_INFERENCE_TYPE,
    DEFAULT_INFERENCE_WEIGHTS,
)


def _resolve_paths(context, *args, **kwargs):
    inference_model_type = LaunchConfiguration("inference_model_type").perform(context)
    inference_weights    = LaunchConfiguration("inference_weights").perform(context)
    onnx_override        = LaunchConfiguration("onnx_path").perform(context)
    norm_override        = LaunchConfiguration("norm_path").perform(context)

    if onnx_override:
        resolved_onnx = onnx_override
        resolved_norm = norm_override
    else:
        pkg = get_package_share_directory("vf_robot_controller")
        family = f"metacritic_{inference_model_type}_wt"
        run_dir = os.path.join(pkg, "models", family, inference_weights)
        resolved_onnx = os.path.join(run_dir, "meta_critic.onnx")
        resolved_norm = os.path.join(run_dir, "feature_norm.json")

    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    n_critics    = LaunchConfiguration("n_critics").perform(context)

    node = Node(
        package="vf_robot_controller",
        executable="metacritic_inference_node.py",
        name="metacritic_inference_node",
        parameters=[{
            "onnx_path":        resolved_onnx,
            "norm_path":        resolved_norm,
            "n_critics":        int(n_critics),
            "expected_in_dim":  126,
            "use_sim_time":     use_sim_time == "true",
            "features_topic":   "/vf/features",
            "publish_topic":    "/vf_controller/meta_weights",
            "publish_on_features": True,
        }],
        output="screen",
    )
    return [node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "inference_model_type", default_value=DEFAULT_INFERENCE_TYPE,
            description='"raw" → metacritic_raw_wt/   "oracle" → metacritic_oracle_wt/',
            choices=["raw", "oracle"],
        ),
        DeclareLaunchArgument(
            "inference_weights", default_value=DEFAULT_INFERENCE_WEIGHTS,
            description="Run folder name inside the family root, e.g. raw_manual_v1",
        ),
        DeclareLaunchArgument(
            "onnx_path", default_value="",
            description="Escape hatch: absolute path to meta_critic.onnx; "
                        "bypasses inference_model_type + inference_weights.",
        ),
        DeclareLaunchArgument(
            "norm_path", default_value="",
            description="Absolute path to feature_norm.json — only used "
                        "when onnx_path is also set.",
        ),
        DeclareLaunchArgument(
            "n_critics", default_value="10",
            description="Number of critics the model outputs.",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            choices=["true", "false"],
        ),
        OpaqueFunction(function=_resolve_paths),
    ])
