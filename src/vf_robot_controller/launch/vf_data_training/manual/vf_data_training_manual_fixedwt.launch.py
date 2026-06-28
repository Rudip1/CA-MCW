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
ViroFighter Robot Controller — vf_robot_controller/launch/vf_data_training/manual/vf_data_training_manual_fixedwt.launch.py
══════════════════════════════════════════════════════════════════════════════
vf_data_training/manual/vf_data_training_manual_fixedwt.launch.py
    — training-data collection with fixed weights, manual (RViz-driven) goals.

Brings up the full navigation stack with controller:=vf_fixedwt and
channel_config:=channels_v3 (both FORCED — this launch is
data-collection-only). Post-M10 /vf/per_critic_costs is published
unconditionally on every MPPI cycle, so data_collector_node always
writes HDF5 episodes under vf_data/vf_data_training/manual/. There is
no save_data toggle: if you want to drive vf_fixedwt without saving,
use vf_robot_bringup/bringup_launch.py with controller:=vf_fixedwt
instead.

Scope: this is the ONLY collect path inside vf_robot_controller and
covers fixedwt + manual driving only. The batch (CSV-driven) and
inspection (autonomous-exploration) training-collection paths live in
vf_robot_utils under launch/vf_data_training/{batch,inspection}/.

VolumetricCritic is disabled
----------------------------
As of 2026-05-09, VolumetricCritic (critic index 9) is set to
`enabled: false` in vf_fixedwt.yaml — it over-penalised corridor
traversal and produced too few positive samples for oracle weight
recovery. The critic name is still in _CRITIC_NAMES below and the
HDF5 schema still has 11 critic_costs columns, but column 9 will be
identically zero across all runs. Trainers should either pin the
critic's weight to the YAML default or drop the column at fit time.
Re-enabling: set `enabled: true` in vf_fixedwt.yaml.

Why channels_v3 is forced
-------------------------
The HDF5 always stores the full 170-dim feature vector (channels_v3 =
robot_state + context + path_geometry + gcf_rosette + critic_history
+ obstacle_dynamics + reynolds + slam_persistent). Trainers slice this
down at training time:
  v1 trainer → features[:, :126]
  v2 trainer → features[:, :130]
  v3 trainer → features[:, :170]
One training corpus, all three channel-set models.

If the SLAM backend (RtabmapBackend / StaticMapBackend) is not
configured when collecting, the slam_persistent slice (40 dims)
zero-fills — that's deterministic, not stale data, and matches how
channel_critic_history zero-fills in vf_imitationwt.

Goals are sent interactively via RViz "Nav2 Goal" or
``ros2 topic pub /goal_pose ...`` — not from launch args.
data_collector_node creates one HDF5 file per goal automatically.

The training HDF5 corpus is then used to train all three model families:
  train_raw_critics.py  → models/metacritic_raw_wt/<channels>_<runtag>/
  train_inference.py    → models/metacritic_oracle_wt/<channels>_<runtag>/
  train_imitation.py    → models/imitation_wt/<channels>_<runtag>/
where <channels> ∈ {v1, v2, v3} matches the trainer's --channels flag.

Data path written:
  vf_data/vf_data_training/manual/<map>/<goal_xy>/<Planner>/vf_fixedwt/run_*.h5

map and planner are REQUIRED (no defaults). This is deliberate — the
training corpus is labelled by these values, and a silent mislabel
(e.g. NavFn data saved under SmacPlanner2D/) would poison training
weeks downstream. Launch fails fast if either is missing.

RTAB-Map .db is auto-resolved
-----------------------------
By default, ``new_map:=false`` and ``rtabmap_db_path`` is auto-resolved
to ``<MAPS_ROOT>/<map>/<map>.db`` so RTAB-Map runs in localization mode
and the slam_persistent slice (last 40 dims of channels_v3) is
populated. The launch fails at parse time if that .db is missing.

To map a fresh world before any .db exists, pass ``new_map:=true``
explicitly — but be aware the slam_persistent slice will be zero-filled
on that run (use it to build the .db, then re-collect for full v3).

Usage:
  ros2 launch vf_robot_controller vf_data_training_manual_fixedwt.launch.py \\
      map:=house_my1_map planner:=NavFn

  ros2 launch vf_robot_controller vf_data_training_manual_fixedwt.launch.py \\
      map:=my_hospital planner:=SmacPlanner2D

  # First-time mapping run (slam_persistent will be zero):
  ros2 launch vf_robot_controller vf_data_training_manual_fixedwt.launch.py \\
      map:=new_world planner:=NavFn new_map:=true

  # Custom .db location:
  ros2 launch vf_robot_controller vf_data_training_manual_fixedwt.launch.py \\
      map:=house_my1_map planner:=NavFn \\
      rtabmap_db_path:=/some/other/place/house_my1_map.db
"""

import os

from ament_index_python.packages import get_package_share_directory
from vf_robot_utils.constants import MAPS_ROOT, TRAINING_ROOT

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Critic names for vf_fixedwt (11 critics — must match vf_fixedwt.yaml order).
# VolumetricCritic (index 9) is DISABLED in vf_fixedwt.yaml — kept in this
# list so the HDF5 schema column count stays 11 and trainers / loaders /
# eval scripts don't need to special-case the change. Its critic_costs
# column will be all zeros across every collected run; oracle QP must
# either pin its weight to the YAML default or drop the column at fit
# time. See vf_fixedwt.yaml VolumetricCritic block for the rationale.
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
    "VolumetricCritic",     # DISABLED in YAML; column stays in HDF5 as zeros
    "DynamicObstacleCritic",
]


def _resolve_map(context):
    """Resolve map_name and rtabmap .db path.

    Auto-resolves the .db to ``<MAPS_ROOT>/<map>/<map>.db`` (or, if map is an
    absolute folder path, ``<map>/<basename(map)>.db``). Skips resolution when:
      - the user passed an explicit ``rtabmap_db_path:=...`` (any non-empty value)
      - ``new_map:=true`` is set (RTAB-Map will start mapping; no .db needed)

    Otherwise we require the .db to exist on disk and fail loud if it doesn't —
    silently mapping-from-scratch when the user expected localization would
    leave the slam_persistent feature slice (40 dims) zero-filled and produce
    incomplete training data.
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

    if explicit_db:
        # User override — respect it, but verify it exists.
        resolved_db = explicit_db
        if not os.path.isfile(resolved_db):
            raise RuntimeError(
                f"[vf_data_training_manual_fixedwt] rtabmap_db_path={resolved_db!r} "
                f"does not exist."
            )
    elif new_map:
        # Mapping mode — no .db needed, RTAB-Map writes a fresh one.
        resolved_db = ""
    else:
        # Default: localize against <MAPS_ROOT>/<map>/<map>.db.
        resolved_db = os.path.join(map_dir, f"{map_name}.db")
        if not os.path.isfile(resolved_db):
            raise RuntimeError(
                f"[vf_data_training_manual_fixedwt] expected RTAB-Map database\n"
                f"  {resolved_db}\n"
                f"to exist (auto-resolved from map:={map_arg!r}). Options:\n"
                f"  1. Map this world first, then collect:  new_map:=true\n"
                f"     (warning: slam_persistent features will be zero on that run).\n"
                f"  2. Pass an explicit path:                rtabmap_db_path:=/abs/path/to.db\n"
                f"  3. Place the .db at the expected location and re-launch."
            )

    return [
        SetLaunchConfiguration("data_map_name", map_name),
        SetLaunchConfiguration("resolved_rtabmap_db_path", resolved_db),
    ]


def generate_launch_description():
    pkg_bringup = get_package_share_directory("vf_robot_bringup")

    # ── Bringup args (forwarded) ───────────────────────────────────────────
    # map: REQUIRED. No default — training-collection runs must declare the
    # map explicitly so the leaf path
    # vf_data/vf_data_training/manual/<map>/... is never mislabelled.
    declare_map = DeclareLaunchArgument(
        "map",
        description=(
            "Map to load (REQUIRED). Bare name (e.g. house_my1_map) → "
            "<MAPS_ROOT>/<map>/<map>.{yaml,db}, or absolute folder path."
        ),
    )
    declare_localization = DeclareLaunchArgument(
        "localization", default_value="rtabmap_loc"
    )
    declare_camera = DeclareLaunchArgument("camera", default_value="dual")
    declare_scan_method = DeclareLaunchArgument("scan_method", default_value="pc2scan")
    declare_merge_scans = DeclareLaunchArgument("merge_scans", default_value="true")
    # new_map defaults to false: training collection requires a pre-built map
    # so RTAB-Map runs in localization mode and the slam_persistent feature
    # slice (40 dims of channels_v3) is populated. Pass new_map:=true only if
    # you really do want to map fresh on this run (slam_persistent will be
    # zero-filled for that collection).
    declare_new_map = DeclareLaunchArgument("new_map", default_value="false")
    declare_use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="true")
    declare_rviz = DeclareLaunchArgument("rviz", default_value="true")
    declare_headless = DeclareLaunchArgument("headless", default_value="false")
    # channel_config is FORCED to channels_v3 here (no override). Rationale:
    # training datasets must always store the full 170-dim feature vector so
    # later training can slice down to v1 / v2 / v3 from one corpus.
    # rtabmap_db_path: optional override. Empty (default) → auto-resolved
    # from `map` arg by _resolve_map() to <MAPS_ROOT>/<map>/<map>.db. Only set
    # this if your .db lives outside the canonical maps/<map>/ layout.
    declare_rtabmap_db_path = DeclareLaunchArgument(
        "rtabmap_db_path",
        default_value="",
        description=(
            "Override RTAB-Map .db path. Default empty → auto-resolved to "
            "<MAPS_ROOT>/<map>/<map>.db."
        ),
    )

    # ── Planner metadata (forwarded to both bringup and data_collector) ────
    # planner: REQUIRED. No default — same reasoning as map. The leaf path
    # vf_data/vf_data_training/manual/<map>/<goal>/<Planner>/vf_fixedwt/ uses this
    # value verbatim, so a silent default would mislabel the corpus.
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

    # ── Data collection args ───────────────────────────────────────────────
    declare_training_root = DeclareLaunchArgument(
        "training_root",
        default_value=str(TRAINING_ROOT),
        description="Override TRAINING_ROOT (vf_data/vf_data_training).",
    )
    declare_controller_label = DeclareLaunchArgument(
        "controller_label",
        default_value="vf_fixedwt",
        description="Folder label in the data path. Distinct from the bringup controller name.",
    )
    declare_scenario_id = DeclareLaunchArgument(
        "scenario_id", default_value="manual_run"
    )
    declare_seed = DeclareLaunchArgument("seed", default_value="0")
    declare_episode_timeout = DeclareLaunchArgument(
        "episode_timeout_s", default_value="180.0"
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

    resolve_map_op = OpaqueFunction(function=_resolve_map)

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
                "session_kind": "manual",
                "map_name": LaunchConfiguration("data_map_name"),
                "scenario_id": LaunchConfiguration("scenario_id"),
                "seed": LaunchConfiguration("seed"),
                "controller_mode": "fixedwt",
                "weight_provider": "fixed",
                "channels_config": "channels_v3",  # forced — matches bringup
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
                # Mirror Nav2's xy_goal_tolerance from nav2_base.yaml (0.10 m).
                # When Nav2 reports succeeded the robot is by definition
                # within this distance, so the collector's prev_reached
                # check fires success=True / close_reason=goal_reached.
                "goal_radius_m": 0.10,
                "goal_reached_consecutive": 5,
                "max_obstacles": 0,
                "goal_debounce_s": LaunchConfiguration("goal_debounce_s"),
                "goal_cooldown_s": LaunchConfiguration("goal_cooldown_s"),
                "goal_dedup_radius_m": LaunchConfiguration("goal_dedup_radius_m"),
                "goal_yaw_eps_rad": LaunchConfiguration("goal_yaw_eps_rad"),
            }
        ],
    )

    return LaunchDescription(
        [
            declare_map,
            declare_localization,
            declare_camera,
            declare_scan_method,
            declare_merge_scans,
            declare_new_map,
            declare_use_sim_time,
            declare_rviz,
            declare_headless,
            declare_rtabmap_db_path,
            declare_planner,
            declare_training_root,
            declare_controller_label,
            declare_scenario_id,
            declare_seed,
            declare_episode_timeout,
            declare_goal_debounce,
            declare_goal_cooldown,
            declare_goal_dedup_radius,
            declare_goal_yaw_eps,
            resolve_map_op,
            bringup,
            collector,
        ]
    )
