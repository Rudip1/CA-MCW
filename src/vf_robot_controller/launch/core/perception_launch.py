"""
perception_launch.py — Phase 4 + Phase 5.

Starts the decoupled perception nodes alongside the controller stack.
Pulled in by vf_robot_bringup/launch/bringup_launch.py for the three vf_*
controllers (vf_fixedwt, vf_inferencewt, vf_imitationwt). The
vf_imitationwt sidecar consumes /vf/features even though MPPI never
runs, so perception is required for it too. Stock Nav2 baselines
(mppi/dwb/rpp/graceful) skip this launch.

Phase 4 nodes (always):
  gcf_node @ 5 Hz publishes
    /vf/gcf_state                    (std_msgs/Float32, [0,1])
    /vf/voxel_filtered_pointcloud    (sensor_msgs/PointCloud2)

Phase 5 nodes (always):
  context_node @ 10 Hz publishes
    /vf/context_state                (std_msgs/Int8, NavigationContext id)
  feature_extractor_node @ 20 Hz publishes
    /vf/features                     (std_msgs/Float32MultiArray)

Channel set is selected by `channel_config` (default channels_v1, six
non-SLAM channels, 126 dims). Pass `channel_config:=channels_v2` to
include reynolds (130 dims) or `channel_config:=channels_v3` for the
full SLAM-persistent set (170 dims).

CorridorCritic / VolumetricCritic / DynamicObstacleCritic consume the
Phase 4 topics with their own staleness checks; killing this launch does
not crash the controller. Feature consumers (data_collector_node,
metacritic_inference_node) likewise tolerate /vf/features dropping out.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("vf_robot_controller")
    default_params = os.path.join(pkg_share, "config", "perception", "perception.yaml")
    default_backend = os.path.join(pkg_share, "config", "perception", "backend_selection.yaml")

    params_arg = DeclareLaunchArgument(
        "perception_params_file",
        default_value=default_params,
        description="YAML params file shared by gcf_node, context_node, feature_extractor_node.",
    )
    channel_config_arg = DeclareLaunchArgument(
        "channel_config",
        default_value="channels_v1",
        choices=["channels_v1", "channels_v2", "channels_v3"],
        description="Which channel set the feature_extractor_node enables.",
    )
    backend_arg = DeclareLaunchArgument(
        "backend_selection_file",
        default_value=default_backend,
        description=(
            "YAML params file selecting the IMapBackend (rtabmap/static/cuvslam) "
            "and providing rtabmap_db_path / static_map_yaml. Read by both "
            "map_backend_node and feature_extractor_node when channels_v3 is "
            "active."
        ),
    )
    start_map_backend_arg = DeclareLaunchArgument(
        "start_map_backend_node",
        default_value="auto",
        choices=["auto", "true", "false"],
        description=(
            "Whether to launch the diagnostic map_backend_node alongside "
            "feature_extractor_node. 'auto' starts it only when "
            "channel_config:=channels_v3."
        ),
    )
    rtabmap_db_path_arg = DeclareLaunchArgument(
        "rtabmap_db_path",
        default_value="",
        description=(
            "Optional override for the RTAB-Map .db path. When non-empty, "
            "applied as a parameter override on feature_extractor_node "
            "and map_backend_node, taking precedence over "
            "backend_selection.yaml."
        ),
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true", choices=["true", "false"],
    )

    channel_yaml = PathJoinSubstitution([
        pkg_share, "config", "perception",
        [LaunchConfiguration("channel_config"), ".yaml"],
    ])

    gcf_node = Node(
        package="vf_robot_controller",
        executable="gcf_node",
        name="gcf_node",
        output="screen",
        parameters=[
            LaunchConfiguration("perception_params_file"),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    context_node = Node(
        package="vf_robot_controller",
        executable="context_node",
        name="context_node",
        output="screen",
        parameters=[
            LaunchConfiguration("perception_params_file"),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    feature_extractor_node = Node(
        package="vf_robot_controller",
        executable="feature_extractor_node",
        name="feature_extractor_node",
        output="screen",
        parameters=[
            LaunchConfiguration("perception_params_file"),
            channel_yaml,
            # backend_selection.yaml maps both map_backend_node and
            # feature_extractor_node — both nodes share these keys.
            LaunchConfiguration("backend_selection_file"),
            # Per-launch override of rtabmap_db_path; later entries win in
            # ROS 2 param resolution, so an explicit non-empty value here
            # supersedes whatever ships in backend_selection.yaml.
            {"rtabmap_db_path": LaunchConfiguration("rtabmap_db_path")},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    # Phase 6: diagnostic map_backend_node. Starts only when
    # channel_config:=channels_v3 (auto), or always/never on explicit override.
    from launch.conditions import IfCondition
    from launch.substitutions import PythonExpression

    auto_start_expr = PythonExpression([
        "'", LaunchConfiguration("start_map_backend_node"), "' == 'true' or "
        "('", LaunchConfiguration("start_map_backend_node"), "' == 'auto' and '",
        LaunchConfiguration("channel_config"), "' == 'channels_v3')",
    ])

    map_backend_node = Node(
        package="vf_robot_controller",
        executable="map_backend_node",
        name="map_backend_node",
        output="screen",
        parameters=[
            LaunchConfiguration("backend_selection_file"),
            {"rtabmap_db_path": LaunchConfiguration("rtabmap_db_path")},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        condition=IfCondition(auto_start_expr),
    )

    return LaunchDescription([
        params_arg,
        channel_config_arg,
        backend_arg,
        start_map_backend_arg,
        rtabmap_db_path_arg,
        use_sim_time_arg,
        gcf_node,
        context_node,
        feature_extractor_node,
        map_backend_node,
    ])
