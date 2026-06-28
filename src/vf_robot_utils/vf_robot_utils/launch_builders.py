#!/usr/bin/env python3
"""
launch_builders.py — shared LaunchDescription builders for the eval batch
launches. Mirrors vf_data_training_batch_fixedwt.launch.py's stack
(bringup + data_collector_node + tour_runner), but switches the bringup
controller, the weights folder (when applicable), and the output root
per family.

Used by every wrapper under
launch/vf_data_evaluation/batch/vf_data_evaluation_batch_<family>.launch.py.

Two family classes:

  * trained_wt :  imitationwt, rawwt, oraclewt
        Required CLI: map planner run_id hp ch
        Bringup loads vf_imitationwt / vf_inferencewt + ONNX sidecar.

  * baseline :    fixedwt, mppi, dwb, rpp, graceful
        Required CLI: map planner run_id
        Bringup loads the stock controller; no ONNX sidecar.

Output:
  vf_data/vf_data_evaluation/batch/<map>/<goal_xy>/<Planner>/<variant>/run_*.h5
    where <variant> = "<family>"                                 (baseline)
                    = "<family>_<hp>_<ch>" e.g. imitationwt_hardreg_v3  (trained_wt)
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

from ament_index_python.packages import get_package_share_directory

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

from vf_robot_utils.constants import EVALUATION_ROOT, MAPS_ROOT


# Critic order must match vf_fixedwt.yaml / vf_inferencewt.yaml / vf_imitationwt.yaml
# so the HDF5 critic_costs columns line up with the variant being recorded.
# Stock nav2 baselines (mppi/dwb/rpp/graceful) don't publish /vf/per_critic_costs
# so the corresponding HDF5 columns stay NaN; the names still ride along for
# schema uniformity across the corpus.
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
    "VolumetricCritic",          # disabled in YAML; column stays zeros
    "DynamicObstacleCritic",
]


# channels_<ch> → (channel_names, channel_dims).
# v1 = 126 dims, v2 = 130 dims (+reynolds), v3 = 170 dims (+slam_persistent).
_CHANNEL_LAYOUT: Dict[str, Tuple[List[str], List[int]]] = {
    "v1": (
        ["robot_state", "context", "path_geometry",
         "gcf_rosette", "critic_history", "obstacle_dynamics"],
        [9, 9, 14, 48, 30, 16],
    ),
    "v2": (
        ["robot_state", "context", "path_geometry",
         "gcf_rosette", "critic_history", "obstacle_dynamics",
         "reynolds"],
        [9, 9, 14, 48, 30, 16, 4],
    ),
    "v3": (
        ["robot_state", "context", "path_geometry",
         "gcf_rosette", "critic_history", "obstacle_dynamics",
         "reynolds", "slam_persistent"],
        [9, 9, 14, 48, 30, 16, 4, 40],
    ),
}


# Family-specific dispatch.
#   class               trained_wt | baseline
#   bringup_controller  controller arg passed to bringup_launch.py
#   sidecar_type        raw | oracle | imitation | none  (sidecar selector)
#   weights_arg         bringup arg pointing at the weights folder
#                       (imitation_weights | inference_weights | "")
#   controller_mode     HDF5 attr written by data_collector_node
#   weight_provider     HDF5 attr written by data_collector_node
_FAMILY: Dict[str, Dict[str, str]] = {
    # ── Trained-weight families ────────────────────────────────────────
    "imitationwt": {
        "class":                "trained_wt",
        "bringup_controller":   "vf_imitationwt",
        "sidecar_type":         "imitation",
        "weights_arg":          "imitation_weights",
        "controller_mode":      "imitationwt",
        "weight_provider":      "imitation",
    },
    "rawwt": {
        "class":                "trained_wt",
        "bringup_controller":   "vf_inferencewt",
        "sidecar_type":         "raw",
        "weights_arg":          "inference_weights",
        "controller_mode":      "inferencewt",
        "weight_provider":      "raw",
    },
    "oraclewt": {
        "class":                "trained_wt",
        "bringup_controller":   "vf_inferencewt",
        "sidecar_type":         "oracle",
        "weights_arg":          "inference_weights",
        "controller_mode":      "inferencewt",
        "weight_provider":      "oracle",
    },
    # ── Baseline families (no ONNX, no sidecar) ────────────────────────
    "fixedwt": {
        "class":                "baseline",
        "bringup_controller":   "vf_fixedwt",
        "sidecar_type":         "none",
        "weights_arg":          "",
        "controller_mode":      "fixedwt",
        "weight_provider":      "fixed",
    },
    "mppi": {
        "class":                "baseline",
        "bringup_controller":   "mppi",
        "sidecar_type":         "none",
        "weights_arg":          "",
        "controller_mode":      "baseline",
        "weight_provider":      "none",
    },
    "dwb": {
        "class":                "baseline",
        "bringup_controller":   "dwb",
        "sidecar_type":         "none",
        "weights_arg":          "",
        "controller_mode":      "baseline",
        "weight_provider":      "none",
    },
    "rpp": {
        "class":                "baseline",
        "bringup_controller":   "rpp",
        "sidecar_type":         "none",
        "weights_arg":          "",
        "controller_mode":      "baseline",
        "weight_provider":      "none",
    },
    "graceful": {
        "class":                "baseline",
        "bringup_controller":   "graceful",
        "sidecar_type":         "none",
        "weights_arg":          "",
        "controller_mode":      "baseline",
        "weight_provider":      "none",
    },
}


def _resolve_paths(family: str, context):
    """Resolve map dir, RTAB-Map .db, eval CSV path, and the variant tag.

    For trained_wt families the variant tag carries (hp, ch); for
    baseline families it's just the family name.
    """
    fam = _FAMILY[family]
    cls = fam["class"]

    map_arg = LaunchConfiguration("map").perform(context).strip()
    explicit_db = LaunchConfiguration("rtabmap_db_path").perform(context).strip()
    new_map = LaunchConfiguration("new_map").perform(context).strip().lower() == "true"

    # Baseline families don't take hp/ch; default channel layout = v3 so
    # any cross-controller HDF5 comparison reads the same feature schema.
    if cls == "trained_wt":
        hp = LaunchConfiguration("hp").perform(context).strip()
        ch = LaunchConfiguration("ch").perform(context).strip()
        if hp not in ("normal", "tuned", "hardreg"):
            raise RuntimeError(
                f"[vf_data_evaluation_batch_{family}] hp={hp!r} not in "
                f"(normal, tuned, hardreg)"
            )
        if ch not in ("v1", "v2", "v3"):
            raise RuntimeError(
                f"[vf_data_evaluation_batch_{family}] ch={ch!r} not in "
                f"(v1, v2, v3)"
            )
        weights_folder = f"{ch}_{hp}"
        variant = f"{family}_{hp}_{ch}"
        channel_key = ch
    else:
        weights_folder = ""
        variant = family
        channel_key = "v3"
    channel_config = f"channels_{channel_key}"

    if os.path.isabs(map_arg):
        map_dir = map_arg
        map_name = os.path.basename(map_arg.rstrip("/"))
    else:
        map_dir = os.path.join(str(MAPS_ROOT), map_arg)
        map_name = map_arg

    # ── RTAB-Map .db ─────────────────────────────────────────────────────
    if explicit_db:
        resolved_db = explicit_db
        if not os.path.isfile(resolved_db):
            raise RuntimeError(
                f"[vf_data_evaluation_batch_{family}] "
                f"rtabmap_db_path={resolved_db!r} does not exist."
            )
    elif new_map:
        resolved_db = ""
    else:
        resolved_db = os.path.join(map_dir, f"{map_name}.db")
        if not os.path.isfile(resolved_db):
            raise RuntimeError(
                f"[vf_data_evaluation_batch_{family}] expected RTAB-Map "
                f"database\n  {resolved_db}\nto exist (auto-resolved from "
                f"map:={map_arg!r}). Map the world first, pass an explicit "
                f"rtabmap_db_path:=, or place the .db at the expected "
                f"location."
            )

    # ── Evaluation goalposes CSV ─────────────────────────────────────────
    csv_path = None
    for fname in ("evaluation_goalposes_collect.csv", "evaluate_goals.csv"):
        candidate = os.path.join(map_dir, fname)
        if os.path.isfile(candidate):
            csv_path = candidate
            break
    if csv_path is None:
        raise RuntimeError(
            f"[vf_data_evaluation_batch_{family}] no eval CSV for "
            f"map={map_name!r}.\n  looked under: {map_dir}\n"
            f"  expected one of: evaluation_goalposes_collect.csv, "
            f"evaluate_goals.csv\nCollect goals first:\n"
            f"  ros2 launch vf_robot_utils "
            f"evaluation_goalposes_collect.launch.py map_name:={map_name}"
        )

    return [
        SetLaunchConfiguration("data_map_name", map_name),
        SetLaunchConfiguration("resolved_rtabmap_db_path", resolved_db),
        SetLaunchConfiguration("resolved_csv_path", csv_path),
        SetLaunchConfiguration("resolved_weights", weights_folder),
        SetLaunchConfiguration("resolved_variant", variant),
        SetLaunchConfiguration("resolved_channel_config", channel_config),
        SetLaunchConfiguration("resolved_channel_key", channel_key),
        SetLaunchConfiguration("resolved_bringup_controller",
                               fam["bringup_controller"]),
        LogInfo(msg=[
            f"[vf_data_evaluation_batch_{family}] variant={variant}  "
            f"channel={channel_config}  csv={csv_path}"
        ]),
    ]


def build_eval_batch_launch_description(family: str) -> LaunchDescription:
    """Build the LaunchDescription for one eval family.

    family ∈  imitationwt, rawwt, oraclewt    (trained_wt)
             fixedwt, mppi, dwb, rpp, graceful (baseline)
    """
    if family not in _FAMILY:
        raise ValueError(
            f"build_eval_batch_launch_description: unknown family={family!r}, "
            f"expected one of {sorted(_FAMILY)}"
        )
    fam = _FAMILY[family]
    cls = fam["class"]

    pkg_bringup = get_package_share_directory("vf_robot_bringup")

    # ── Required args (always) ─────────────────────────────────────────
    args: List = [
        DeclareLaunchArgument(
            "map",
            description=(
                "Map to load (REQUIRED). Bare name (e.g. house_my1_map) → "
                "<MAPS_ROOT>/<map>/<map>.{yaml,db} and "
                "<MAPS_ROOT>/<map>/evaluation_goalposes_collect.csv."
            ),
        ),
        DeclareLaunchArgument(
            "planner",
            choices=["NavFn", "SmacPlanner2D", "SmacPlannerHybrid",
                     "SmacLattice", "ThetaStar"],
            description=(
                "Global planner (REQUIRED). Burned into the data path and "
                "HDF5 attrs."
            ),
        ),
        DeclareLaunchArgument(
            "run_id",
            description=(
                "Integer row to replay from evaluation_goalposes_collect.csv."
            ),
        ),
    ]

    # ── Trained-wt-only args ───────────────────────────────────────────
    if cls == "trained_wt":
        args += [
            DeclareLaunchArgument(
                "hp", choices=["normal", "tuned", "hardreg"],
                description="HP group of the trained weights (REQUIRED)."),
            DeclareLaunchArgument(
                "ch", choices=["v1", "v2", "v3"],
                description="Channel set the weights were trained on (REQUIRED)."),
        ]

    # ── Bringup pass-through ───────────────────────────────────────────
    args += [
        DeclareLaunchArgument("localization", default_value="rtabmap_loc"),
        DeclareLaunchArgument("camera", default_value="dual"),
        DeclareLaunchArgument("scan_method", default_value="pc2scan"),
        DeclareLaunchArgument("merge_scans", default_value="true"),
        DeclareLaunchArgument("new_map", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument(
            "rtabmap_db_path", default_value="",
            description=(
                "Override RTAB-Map .db. Default empty → auto-resolve to "
                "<MAPS_ROOT>/<map>/<map>.db."
            ),
        ),
    ]

    # ── Data-collection args ───────────────────────────────────────────
    args += [
        DeclareLaunchArgument(
            "evaluation_root", default_value=str(EVALUATION_ROOT),
            description="Override EVALUATION_ROOT (vf_data/vf_data_evaluation)."),
        DeclareLaunchArgument("scenario_id", default_value="batch_run"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("episode_timeout_s", default_value="180.0"),
        DeclareLaunchArgument("goal_radius_m", default_value="0.10"),
        DeclareLaunchArgument("nav2_close_settle_s", default_value="3.0"),
        DeclareLaunchArgument("nav2_status_close_enabled", default_value="true"),
        DeclareLaunchArgument("goal_debounce_s", default_value="0.5"),
        DeclareLaunchArgument("goal_cooldown_s", default_value="2.0"),
        DeclareLaunchArgument("goal_dedup_radius_m", default_value="0.5"),
        DeclareLaunchArgument("goal_yaw_eps_rad", default_value="0.35"),
    ]

    # ── Tour-runner args ───────────────────────────────────────────────
    args += [
        DeclareLaunchArgument("settle_s", default_value="3.0"),
        DeclareLaunchArgument("inter_leg_pause_s", default_value="1.0"),
        DeclareLaunchArgument("post_reposition_stabilize_s", default_value="5.0"),
        DeclareLaunchArgument("per_goal_timeout_s", default_value="180.0"),
        DeclareLaunchArgument("nav2_ready_timeout_s", default_value="120.0"),
        DeclareLaunchArgument("reposition_first", default_value="true"),
        DeclareLaunchArgument("reposition_xy_tol_m", default_value="0.5"),
        DeclareLaunchArgument("reposition_yaw_tol_rad", default_value="0.5"),
        DeclareLaunchArgument("reposition_max_attempts", default_value="3"),
    ]

    resolve_paths_op = OpaqueFunction(
        function=lambda ctx: _resolve_paths(family, ctx)
    )

    # ── Bringup invocation ─────────────────────────────────────────────
    bringup_args = {
        "controller":      LaunchConfiguration("resolved_bringup_controller"),
        "planner":         LaunchConfiguration("planner"),
        "map":             LaunchConfiguration("map"),
        "localization":    LaunchConfiguration("localization"),
        "camera":          LaunchConfiguration("camera"),
        "scan_method":     LaunchConfiguration("scan_method"),
        "merge_scans":     LaunchConfiguration("merge_scans"),
        "new_map":         LaunchConfiguration("new_map"),
        "use_sim_time":    LaunchConfiguration("use_sim_time"),
        "rviz":            LaunchConfiguration("rviz"),
        "headless":        LaunchConfiguration("headless"),
        "channel_config":  LaunchConfiguration("resolved_channel_config"),
        "rtabmap_db_path": LaunchConfiguration("resolved_rtabmap_db_path"),
    }
    if cls == "trained_wt":
        bringup_args.update({
            "autostart_sidecar":   "true",
            "inference_model_type": fam["sidecar_type"]
                if fam["sidecar_type"] in ("raw", "oracle") else "raw",
            fam["weights_arg"]:     LaunchConfiguration("resolved_weights"),
        })
    else:
        bringup_args["autostart_sidecar"] = "false"

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "bringup_launch.py")
        ),
        launch_arguments=bringup_args.items(),
    )

    # ── Collector node (channel layout depends on ch arg at runtime) ───
    def _spawn_collector(context):
        channel_key = LaunchConfiguration("resolved_channel_key").perform(context)
        names, dims = _CHANNEL_LAYOUT[channel_key]

        return [Node(
            package="vf_robot_controller",
            executable="data_collector_node.py",
            name="data_collector_node",
            output="screen",
            parameters=[{
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                # Output root: data_collector_node uses `training_root` as
                # its on-disk root regardless of training/eval — we just
                # repoint it at EVALUATION_ROOT here.
                "training_root": LaunchConfiguration("evaluation_root"),
                "planner": LaunchConfiguration("planner"),
                "controller": LaunchConfiguration("resolved_variant"),
                "session_kind": "batch",
                "map_name": LaunchConfiguration("data_map_name"),
                "scenario_id": LaunchConfiguration("scenario_id"),
                "seed": LaunchConfiguration("seed"),
                "controller_mode": fam["controller_mode"],
                "weight_provider": fam["weight_provider"],
                "channels_config": f"channels_{channel_key}",
                "channel_names": names,
                "channel_dims": dims,
                "critic_names": _CRITIC_NAMES,
                "episode_timeout_s": LaunchConfiguration("episode_timeout_s"),
                "flush_period_s": 1.0,
                "write_period_s": 0.05,
                "goal_radius_m": LaunchConfiguration("goal_radius_m"),
                "nav2_close_settle_s": LaunchConfiguration("nav2_close_settle_s"),
                "nav2_status_close_enabled":
                    LaunchConfiguration("nav2_status_close_enabled"),
                "recording_enabled_default": False,
                "batch_mode": True,
                "goal_reached_consecutive": 5,
                "max_obstacles": 0,
                "goal_debounce_s": LaunchConfiguration("goal_debounce_s"),
                "goal_cooldown_s": LaunchConfiguration("goal_cooldown_s"),
                "goal_dedup_radius_m": LaunchConfiguration("goal_dedup_radius_m"),
                "goal_yaw_eps_rad": LaunchConfiguration("goal_yaw_eps_rad"),
            }],
        )]

    collector_op = OpaqueFunction(function=_spawn_collector)

    # ── tour_runner: drives NavigateToPose through the CSV row ─────────
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
                LogInfo(msg=f"[{family}] tour_runner exited — shutting down launch."),
                EmitEvent(event=Shutdown(reason="tour complete")),
            ],
        )
    )

    return LaunchDescription(args + [
        resolve_paths_op,
        bringup,
        collector_op,
        tour_runner,
        shutdown_on_tour_done,
    ])
