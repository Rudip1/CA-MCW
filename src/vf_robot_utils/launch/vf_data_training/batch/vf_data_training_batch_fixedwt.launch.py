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
ViroFighter Robot Controller — vf_robot_utils/launch/vf_data_training/batch/vf_data_training_batch_fixedwt.launch.py
══════════════════════════════════════════════════════════════════════════════
vf_data_training/batch/vf_data_training_batch_fixedwt.launch.py
    — training-data collection with fixed weights, BATCH (CSV-driven) goals.

Same stack as the manual variant
(vf_robot_controller/launch/vf_data_training/manual/vf_data_training_manual_fixedwt.launch.py),
but instead of typing goals into RViz, this launch reads one row of a
goalposes_collect CSV (collected via training_goalposes_collect.launch.py)
and drives those goals sequentially via NavigateToPose. data_collector_node
listens exactly the same way and writes one HDF5 per goal.

Output path:
  vf_data/vf_data_training/batch/<map>/<goal_xy>/<Planner>/vf_fixedwt/run_*.h5

Required args:
  map:=         Map name (folder under MAPS_ROOT). Required.
  planner:=     Global planner key. Required. {NavFn, SmacPlanner2D,
                SmacPlannerHybrid, SmacLattice, ThetaStar}
  run_id:=      Integer row to replay from training_goalposes_collect.csv.

CSV resolution:
  <MAPS_ROOT>/<map>/training_goalposes_collect.csv

  CSV format (wide, written by pose_recorder):
    run_id, notes, start_x, start_y, start_yaw,
    g1_x, g1_y, g1_yaw, g2_x, g2_y, g2_yaw, ...

  The selected row's goals (g1, g2, ...) are sent in order via
  NavigateToPose. The start pose is used as the initial reposition
  (set reposition_first:=false to skip).

VolumetricCritic is disabled (column 9 of critic_costs is all zeros) —
same caveat as the manual variant; see vf_fixedwt.yaml.

channels_v3 is forced — the HDF5 always stores the full 170-dim feature
vector so trainers can slice down to v1 / v2 / v3 from one corpus.

RTAB-Map .db is auto-resolved to <MAPS_ROOT>/<map>/<map>.db unless
new_map:=true or rtabmap_db_path:=... is passed (same as manual variant).

Usage:
  ros2 launch vf_robot_utils vf_data_training_batch_fixedwt.launch.py \\
      map:=house_my1_map planner:=NavFn run_id:=0

  # Skip the start-pose reposition (robot already at start):
  ros2 launch vf_robot_utils vf_data_training_batch_fixedwt.launch.py \\
      map:=house_my1_map planner:=NavFn run_id:=0 reposition_first:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from vf_robot_utils.constants import MAPS_ROOT, TRAINING_ROOT

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetLaunchConfiguration,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Same critic order as vf_fixedwt.yaml (must match for HDF5 schema).
_CRITIC_NAMES = [
    "WeightedConstraintCritic",
    "WeightedCostCritic",
    "WeightedGoalCritic",
    "WeightedGoalAngleCritic",
    "WeightedPathAlignCritic",
    "WeightedPathFollowCritic",
    "WeightedPathAngleCritic",
    "WeightedPreferForwardCritic",
    "CorridorCritic",
    "VolumetricCritic",     # disabled in YAML; column stays as zeros
    "DynamicObstacleCritic",
]


def _resolve_paths(context):
    """Resolve map_name, RTAB-Map .db path, and CSV path.

    Fails loud if the goalposes CSV is missing — silently launching with
    an empty CSV would leave the user wondering why the robot never moves.
    """
    map_arg = LaunchConfiguration("map").perform(context)
    explicit_db = LaunchConfiguration("rtabmap_db_path").perform(context).strip()
    new_map = LaunchConfiguration("new_map").perform(context).strip().lower() == "true"

    if os.path.isabs(map_arg):
        map_dir = map_arg
        map_name = os.path.basename(map_arg.rstrip("/"))
    else:
        map_dir = os.path.join(str(MAPS_ROOT), map_arg)
        map_name = map_arg

    # ── RTAB-Map .db ──────────────────────────────────────────────────────
    if explicit_db:
        resolved_db = explicit_db
        if not os.path.isfile(resolved_db):
            raise RuntimeError(
                f"[vf_data_training_batch_fixedwt] rtabmap_db_path={resolved_db!r} "
                f"does not exist."
            )
    elif new_map:
        resolved_db = ""
    else:
        resolved_db = os.path.join(map_dir, f"{map_name}.db")
        if not os.path.isfile(resolved_db):
            raise RuntimeError(
                f"[vf_data_training_batch_fixedwt] expected RTAB-Map database\n"
                f"  {resolved_db}\n"
                f"to exist (auto-resolved from map:={map_arg!r}). Either map this "
                f"world first (new_map:=true), pass an explicit rtabmap_db_path:=, "
                f"or place the .db at the expected location."
            )

    # ── Goalposes CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(map_dir, "training_goalposes_collect.csv")
    if not os.path.isfile(csv_path):
        raise RuntimeError(
            f"[vf_data_training_batch_fixedwt] training_goalposes_collect.csv "
            f"not found at\n  {csv_path}\n"
            f"Collect goals first:\n"
            f"  ros2 launch vf_robot_utils training_goalposes_collect.launch.py "
            f"map_name:={map_name}"
        )

    return [
        SetLaunchConfiguration("data_map_name", map_name),
        SetLaunchConfiguration("resolved_rtabmap_db_path", resolved_db),
        SetLaunchConfiguration("resolved_csv_path", csv_path),
    ]


def generate_launch_description():
    pkg_bringup = get_package_share_directory("vf_robot_bringup")

    # ── Required args ─────────────────────────────────────────────────────
    declare_map = DeclareLaunchArgument(
        "map",
        description=(
            "Map to load (REQUIRED). Bare name (e.g. house_my1_map) → "
            "<MAPS_ROOT>/<map>/<map>.{yaml,db} and "
            "<MAPS_ROOT>/<map>/training_goalposes_collect.csv."
        ),
    )
    declare_planner = DeclareLaunchArgument(
        "planner",
        choices=[
            "NavFn",
            "SmacPlanner2D",
            "SmacPlannerHybrid",
            "SmacLattice",
            "ThetaStar",
        ],
        description=(
            "Global planner (REQUIRED). Burned into the data path and HDF5 "
            "attrs — must match the planner actually loaded by Nav2."
        ),
    )
    declare_run_id = DeclareLaunchArgument(
        "run_id",
        description=(
            "Integer row to replay from training_goalposes_collect.csv "
            "(REQUIRED). Each Ctrl-C of the goalposes_collect tool appends "
            "a new row; pick the one you want to drive."
        ),
    )

    # ── Bringup args (forwarded; defaults match the manual variant) ───────
    declare_localization = DeclareLaunchArgument(
        "localization", default_value="rtabmap_loc"
    )
    declare_camera = DeclareLaunchArgument("camera", default_value="dual")
    declare_scan_method = DeclareLaunchArgument("scan_method", default_value="pc2scan")
    declare_merge_scans = DeclareLaunchArgument("merge_scans", default_value="true")
    declare_new_map = DeclareLaunchArgument("new_map", default_value="false")
    declare_use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="true")
    declare_rviz = DeclareLaunchArgument("rviz", default_value="true")
    declare_headless = DeclareLaunchArgument("headless", default_value="false")
    declare_rtabmap_db_path = DeclareLaunchArgument(
        "rtabmap_db_path",
        default_value="",
        description=(
            "Override RTAB-Map .db path. Default empty → auto-resolved to "
            "<MAPS_ROOT>/<map>/<map>.db."
        ),
    )

    # ── Data collection args ──────────────────────────────────────────────
    declare_training_root = DeclareLaunchArgument(
        "training_root",
        default_value=str(TRAINING_ROOT),
        description="Override TRAINING_ROOT (vf_data/vf_data_training).",
    )
    declare_controller_label = DeclareLaunchArgument(
        "controller_label",
        default_value="vf_fixedwt",
    )
    declare_scenario_id = DeclareLaunchArgument(
        "scenario_id", default_value="batch_run"
    )
    declare_seed = DeclareLaunchArgument("seed", default_value="0")
    declare_episode_timeout = DeclareLaunchArgument(
        "episode_timeout_s", default_value="180.0"
    )
    declare_goal_radius = DeclareLaunchArgument(
        "goal_radius_m", default_value="0.10",
        description=(
            "Tolerance-based-close FALLBACK threshold (m). Mirrors Nav2's "
            "xy_goal_tolerance. The primary close path subscribes to "
            "/navigate_to_pose/_action/status; this radius only fires if "
            "no terminal Nav2 status arrives (e.g. RViz manual sessions)."
        ),
    )
    declare_nav2_close_settle_s = DeclareLaunchArgument(
        "nav2_close_settle_s", default_value="3.0",
        description=(
            "Seconds to keep recording after Nav2 reports a terminal "
            "status (SUCCEEDED/CANCELED/ABORTED) before closing the "
            "HDF5. Lets odom settle into its final pose so robot_pose[-1] "
            "is the actual stop pose, not mid-deceleration."
        ),
    )
    declare_nav2_status_close_enabled = DeclareLaunchArgument(
        "nav2_status_close_enabled", default_value="true",
        description=(
            "If true (default), the collector closes episodes on Nav2 "
            "action terminal status. If false, falls back to the "
            "goal_radius_m tolerance check only."
        ),
    )
    declare_goal_debounce = DeclareLaunchArgument(
        "goal_debounce_s", default_value="0.5"
    )
    declare_goal_cooldown = DeclareLaunchArgument(
        "goal_cooldown_s", default_value="2.0"
    )
    declare_goal_dedup_radius = DeclareLaunchArgument(
        "goal_dedup_radius_m", default_value="0.5"
    )
    declare_goal_yaw_eps = DeclareLaunchArgument(
        "goal_yaw_eps_rad", default_value="0.35"
    )

    # ── Tour-runner args ──────────────────────────────────────────────────
    # In batch_mode the lifecycle is: NavigateToPose → terminal status →
    # sleep settle_s (lets odom settle) → publish CLOSE → sleep
    # inter_leg_pause_s (collector flush) → publish OPEN for next goal.
    declare_settle_s = DeclareLaunchArgument(
        "settle_s", default_value="3.0",
        description=(
            "Sleep AFTER NavigateToPose terminal status, BEFORE the "
            "batch_control CLOSE is published (s). Lets /odom decelerate "
            "into the actual stop pose so the manifest's goal_x/y/yaw is "
            "the stabilised pose."
        ),
    )
    declare_inter_leg_pause_s = DeclareLaunchArgument(
        "inter_leg_pause_s", default_value="1.0",
        description=(
            "Sleep AFTER the batch_control CLOSE, BEFORE the next OPEN (s). "
            "Buffer so the collector finishes flushing the previous HDF5 "
            "before the next episode opens."
        ),
    )
    declare_post_reposition_stabilize_s = DeclareLaunchArgument(
        "post_reposition_stabilize_s", default_value="5.0",
        description=(
            "Pause after the reposition leg, before the first goal is sent. "
            "Longer than settle_s so localization and the controller settle "
            "before per-goal recording starts."
        ),
    )
    declare_per_goal_timeout_s = DeclareLaunchArgument(
        "per_goal_timeout_s", default_value="180.0",
        description="Hard timeout per NavigateToPose goal (s).",
    )
    declare_nav2_ready_timeout_s = DeclareLaunchArgument(
        "nav2_ready_timeout_s", default_value="120.0",
        description=(
            "Max wait (s) for Nav2 readiness before the tour starts: "
            "action server up + first /global_costmap/costmap update + "
            "first /odom message. Prevents the first goal from racing "
            "against Nav2 startup and being rejected."
        ),
    )
    declare_reposition_first = DeclareLaunchArgument(
        "reposition_first", default_value="true",
        description=(
            "If true, drive to the row's start pose before sending goals. "
            "Set false if the robot is already at the start pose."
        ),
    )
    declare_reposition_xy_tol_m = DeclareLaunchArgument(
        "reposition_xy_tol_m", default_value="0.5",
        description=(
            "XY tolerance (m) used to verify the reposition leg actually "
            "reached (start_x, start_y) before the per-goal loop begins."
        ),
    )
    declare_reposition_yaw_tol_rad = DeclareLaunchArgument(
        "reposition_yaw_tol_rad", default_value="0.5",
        description=(
            "Yaw tolerance (rad) used to verify the reposition leg "
            "(default ~28 degrees)."
        ),
    )
    declare_reposition_max_attempts = DeclareLaunchArgument(
        "reposition_max_attempts", default_value="3",
        description=(
            "Max NavigateToPose attempts to reach the start pose. Each "
            "retry first clears /global_costmap and /local_costmap to "
            "flush stale lethal cells from a previous run."
        ),
    )

    resolve_paths_op = OpaqueFunction(function=_resolve_paths)

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "controller": "vf_fixedwt",
            "planner": LaunchConfiguration("planner"),
            "map": LaunchConfiguration("map"),
            "localization": LaunchConfiguration("localization"),
            "camera": LaunchConfiguration("camera"),
            "scan_method": LaunchConfiguration("scan_method"),
            "merge_scans": LaunchConfiguration("merge_scans"),
            "new_map": LaunchConfiguration("new_map"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "rviz": LaunchConfiguration("rviz"),
            "headless": LaunchConfiguration("headless"),
            "channel_config": "channels_v3",  # forced — collect always 170-dim
            "rtabmap_db_path": LaunchConfiguration("resolved_rtabmap_db_path"),
        }.items(),
    )

    collector = Node(
        package="vf_robot_controller",
        executable="data_collector_node.py",
        name="data_collector_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "training_root": LaunchConfiguration("training_root"),
                "planner": LaunchConfiguration("planner"),
                "controller": LaunchConfiguration("controller_label"),
                "session_kind": "batch",
                "map_name": LaunchConfiguration("data_map_name"),
                "scenario_id": LaunchConfiguration("scenario_id"),
                "seed": LaunchConfiguration("seed"),
                "controller_mode": "fixedwt",
                "weight_provider": "fixed",
                "channels_config": "channels_v3",
                "channel_names": [
                    "robot_state",
                    "context",
                    "path_geometry",
                    "gcf_rosette",
                    "critic_history",
                    "obstacle_dynamics",
                    "reynolds",
                    "slam_persistent",
                ],
                "channel_dims": [9, 9, 14, 48, 30, 16, 4, 40],
                "critic_names": _CRITIC_NAMES,
                "episode_timeout_s": LaunchConfiguration("episode_timeout_s"),
                "flush_period_s": 1.0,
                "write_period_s": 0.05,
                "goal_radius_m": LaunchConfiguration("goal_radius_m"),
                "nav2_close_settle_s": LaunchConfiguration("nav2_close_settle_s"),
                "nav2_status_close_enabled":
                    LaunchConfiguration("nav2_status_close_enabled"),
                # Start with recording GATED OFF — tour_runner enables it
                # after the spawn→start reposition. Without this default,
                # the reposition leg would publish /goal_pose before
                # tour_runner has a chance to call set_recording(False).
                "recording_enabled_default": False,
                # Batch mode: tour_runner drives episode open/close
                # explicitly via /vf/batch/episode_control. The collector
                # ignores /goal_pose, /plan, the Nav2 action-status closer
                # and the goal_radius_m tolerance closer in this mode, so
                # exactly one HDF5 per goal is produced regardless of /plan
                # timing or replan churn.
                "batch_mode": True,
                "goal_reached_consecutive": 5,
                "max_obstacles": 0,
                "goal_debounce_s": LaunchConfiguration("goal_debounce_s"),
                "goal_cooldown_s": LaunchConfiguration("goal_cooldown_s"),
                "goal_dedup_radius_m": LaunchConfiguration("goal_dedup_radius_m"),
                "goal_yaw_eps_rad": LaunchConfiguration("goal_yaw_eps_rad"),
            }
        ],
    )

    # tour_runner drives the row's goals sequentially via NavigateToPose.
    # When it exits (last goal done or aborted), the launch shuts down.
    tour_runner = Node(
        package="vf_robot_utils",
        executable="tour_runner",
        name="tour_runner",
        output="screen",
        arguments=[
            "--csv", LaunchConfiguration("resolved_csv_path"),
            "--run-id", LaunchConfiguration("run_id"),
            "--settle-s", LaunchConfiguration("settle_s"),
            "--inter-leg-pause-s", LaunchConfiguration("inter_leg_pause_s"),
            "--post-reposition-stabilize-s",
            LaunchConfiguration("post_reposition_stabilize_s"),
            "--per-goal-timeout-s", LaunchConfiguration("per_goal_timeout_s"),
            "--nav2-ready-timeout-s",
            LaunchConfiguration("nav2_ready_timeout_s"),
            "--reposition-first", LaunchConfiguration("reposition_first"),
            "--reposition-xy-tol-m",
            LaunchConfiguration("reposition_xy_tol_m"),
            "--reposition-yaw-tol-rad",
            LaunchConfiguration("reposition_yaw_tol_rad"),
            "--reposition-max-attempts",
            LaunchConfiguration("reposition_max_attempts"),
        ],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    shutdown_on_tour_done = RegisterEventHandler(
        OnProcessExit(
            target_action=tour_runner,
            on_exit=[
                LogInfo(msg="[batch] tour_runner exited — shutting down launch."),
                EmitEvent(event=Shutdown(reason="tour complete")),
            ],
        )
    )

    return LaunchDescription(
        [
            declare_map,
            declare_planner,
            declare_run_id,
            declare_localization,
            declare_camera,
            declare_scan_method,
            declare_merge_scans,
            declare_new_map,
            declare_use_sim_time,
            declare_rviz,
            declare_headless,
            declare_rtabmap_db_path,
            declare_training_root,
            declare_controller_label,
            declare_scenario_id,
            declare_seed,
            declare_episode_timeout,
            declare_goal_radius,
            declare_nav2_close_settle_s,
            declare_nav2_status_close_enabled,
            declare_goal_debounce,
            declare_goal_cooldown,
            declare_goal_dedup_radius,
            declare_goal_yaw_eps,
            declare_settle_s,
            declare_inter_leg_pause_s,
            declare_post_reposition_stabilize_s,
            declare_per_goal_timeout_s,
            declare_nav2_ready_timeout_s,
            declare_reposition_first,
            declare_reposition_xy_tol_m,
            declare_reposition_yaw_tol_rad,
            declare_reposition_max_attempts,
            resolve_paths_op,
            bringup,
            collector,
            tour_runner,
            shutdown_on_tour_done,
        ]
    )
