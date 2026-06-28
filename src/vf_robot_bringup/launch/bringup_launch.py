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
ViroFighter Robot Bringup — vf_robot_bringup
══════════════════════════════════════════════════════════════════════════════

Single entry point for all ViroFighter (and TurtleBot3) navigation sessions.
Four orthogonal axes — robot, controller, localization, planner — can be
combined freely. Any valid combination launches a complete navigation stack
(Nav2 + selected localization backend + selected global planner).

────────────────────────────────────────────────────────────────────────────
ARGUMENTS
────────────────────────────────────────────────────────────────────────────
  robot         : virofighter | turtlebot3_waffle        (default: virofighter)
  controller    : vf_fixedwt | vf_inferencewt | vf_imitationwt
                  | mppi | dwb | rpp | graceful          (default: vf_inferencewt)
  localization  : rtabmap_slam | rtabmap_loc
                  | amcl | slam_toolbox                  (default: rtabmap_loc)
  planner       : NavFn | SmacPlanner2D | SmacPlannerHybrid
                  | SmacLattice | ThetaStar              (default: NavFn)
                  Selects config/nav2/planners/<Planner>.yaml and merges it
                  into the composed Nav2 params (planner_server section).

  camera        : d435i | d455 | dual                    (default: dual)
  scan_method   : dimg | pc2scan                         (default: pc2scan)
  merge_scans   : true | false                           (default: true)

  map           : bare map name OR absolute folder path  (default: house_my1_map)
                  Bare name (e.g. house_my1_map):
                    → <VF_MAPS_ROOT>/house_my1_map/house_my1_map.{db,yaml}
                  Absolute path (e.g. /data/maps/hospital):
                    → /data/maps/hospital/hospital.{db,yaml}
  new_map       : true | false — delete .db, start fresh (default: true)

  use_sim_time  : true | false                           (default: true)
  rviz          : true | false                           (default: true)
  rviz_config   : full path to .rviz config              (default: vf_bringup.rviz)
  headless      : true | false — suppress bringup-owned GUIs (default: false)
  autostart_sidecar     : true | false                      (default: true)
  inference_model_type  : raw | oracle                       (default: raw)
                          raw    → models/metacritic_raw_wt/<inference_weights>/
                          oracle → models/metacritic_oracle_wt/<inference_weights>/
  inference_weights     : run folder name                    (default: raw_manual_v1)
  imitation_weights     : run folder under models/imitation_wt/ (default: manual_v1)
  onnx_path             : escape hatch — absolute .onnx path bypasses folder resolution

────────────────────────────────────────────────────────────────────────────
USAGE EXAMPLES
────────────────────────────────────────────────────────────────────────────
  # Thesis demo — ViroFighter + meta-critic inference + RTAB-Map localization
  ros2 launch vf_robot_bringup bringup_launch.py \
      robot:=virofighter controller:=vf_inferencewt localization:=rtabmap_loc \
      camera:=dual map:=hospital use_sim_time:=true rviz:=true

  # Primary baseline — stock Nav2 MPPI + RTAB-Map localization
  ros2 launch vf_robot_bringup bringup_launch.py \
      robot:=virofighter controller:=mppi localization:=rtabmap_loc \
      camera:=dual map:=house_my1_map use_sim_time:=true rviz:=true

  # Ablation baseline — vf_fixedwt plugin + AMCL
  ros2 launch vf_robot_bringup bringup_launch.py \
      robot:=virofighter controller:=vf_fixedwt localization:=amcl \
      map:=house_my1_map

  # Build a new map — RTAB-Map SLAM mode
  ros2 launch vf_robot_bringup bringup_launch.py \
      robot:=virofighter controller:=mppi localization:=rtabmap_slam \
      map:=house_my1_map new_map:=true

  # Imitation baseline (sidecar owns /cmd_vel, Nav2 BT uses zero-twist slot)
  ros2 launch vf_robot_bringup bringup_launch.py \
      robot:=virofighter controller:=vf_imitationwt localization:=rtabmap_loc \
      map:=hospital autostart_sidecar:=true imitation_weights:=manual_v1

  # Inference with auto-started sidecar (raw critics weights)
  ros2 launch vf_robot_bringup bringup_launch.py \
      robot:=virofighter controller:=vf_inferencewt localization:=rtabmap_loc \
      map:=hospital autostart_sidecar:=true inference_weights:=raw_manual_v1

  # Inference with oracle weights — change type and weights folder
  ros2 launch vf_robot_bringup bringup_launch.py \
      robot:=virofighter controller:=vf_inferencewt localization:=rtabmap_loc \
      map:=hospital autostart_sidecar:=true \
      inference_model_type:=oracle inference_weights:=oracle_manual_v1

  # Geometric baseline — Regulated Pure Pursuit
  ros2 launch vf_robot_bringup bringup_launch.py \
      robot:=virofighter controller:=rpp localization:=rtabmap_loc \
      camera:=dual map:=house_my1_map use_sim_time:=true

  # Smooth-trajectory baseline — Graceful Motion Controller
  ros2 launch vf_robot_bringup bringup_launch.py \
      robot:=virofighter controller:=graceful localization:=rtabmap_loc \
      camera:=dual map:=house_my1_map use_sim_time:=true

  # Show all arguments
  ros2 launch vf_robot_bringup bringup_launch.py --show-args

────────────────────────────────────────────────────────────────────────────
PREREQUISITES
────────────────────────────────────────────────────────────────────────────
  Gazebo (sim) or real robot driver must be running before this launch.
  For simulation:
      ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py

  For rtabmap_loc, a map database must already exist at:
      <VF_MAPS_ROOT>/<map>/<map>.db
  Build one first with localization:=rtabmap_slam.

  Environment:
      source ~/CA-MCW/install/setup.bash
      source ~/rtabmap/install/setup.bash

  ONNX Runtime (for vf_inferencewt / vf_imitationwt sidecars):
      pip3 install onnxruntime    # system Python, no conda needed for inference

────────────────────────────────────────────────────────────────────────────
CONTROLLER FAMILIES AND SIDECARS
────────────────────────────────────────────────────────────────────────────
  vf_fixedwt    — Fixed 1.0 critic weights; MPPI + 3 custom critics. No sidecar.
  vf_inferencewt — Meta-critic NN weights from Python sidecar via
                  /vf_controller/meta_weights.
                  Sidecar: metacritic_inference_node.py
  vf_imitationwt — VFController returns zero-twist (VFMode::PASSIVE); Python
                  imitation sidecar drives via /cmd_vel_nav.
                  Sidecar: imitation_inference_node.py

  Data collection runs through fixedwt_data.launch.py in
  vf_robot_controller/launch/collect/ — not from bringup. The vf_fixedwt
  and vf_inferencewt controllers always publish /vf/per_critic_costs;
  the C++ collect-mode gate was removed in M10. vf_collectwt as a
  separate controller mode no longer exists.

  For vf_inferencewt and vf_imitationwt:
    autostart_sidecar:=true  (default) — bringup spawns the sidecar Node
      with the resolved ONNX path (from model_defaults.py or CLI overrides
      inference_model_type:= / inference_weights:= / imitation_weights:= /
      onnx_path:=).
    autostart_sidecar:=false — prints the manual ros2 run command instead.

  mppi, dwb, rpp, graceful — stock Nav2 plugins. No sidecar.

────────────────────────────────────────────────────────────────────────────
HOW IT WORKS INTERNALLY
────────────────────────────────────────────────────────────────────────────
  1. PARAM COMPOSITION (OpaqueFunction — runs before any node spawns)
     compose_params.compose() reads:
       - config/nav2/nav2_base.yaml              (base Nav2 params)
       - config/nav2/controllers/<ctrl>.yaml     (controller fragment)
       - config/nav2/planners/<planner>.yaml     (planner fragment)
       - config/nav2/localization/<loc>.yaml     (localization fragment, if any)
     ...then applies the robot profile's rewrites from config/robots/<robot>.yaml.
     The merged result is written to /tmp/nav2_<robot>_<ctrl>_<planner>_<loc>.yaml
     and stored in the composed_params_file LaunchConfiguration.
     Strict mode is enabled — any rewrite key missing from the merged params
     raises immediately with a "did you mean ...?" suggestion.

     Also resolves the `map` arg into resolved_map_name, resolved_maps_dir,
     and resolved_amcl_yaml for downstream branches.

  2. DEPTH-TO-SCAN (always active)
     depth_to_scan.launch.py converts depth camera output to /scan.

  3. LOCALIZATION BRANCH (exactly one activates)
     rtabmap_slam  → vf_robot_slam/rtabmap_slam.launch.py
     rtabmap_loc   → vf_robot_slam/rtabmap_loc.launch.py
     amcl          → vf_robot_bringup/localization_launch.py
     slam_toolbox  → vf_robot_bringup/slam_launch.py

  4. NAV2 STACK (always active)
     navigation_launch.py with composed_params_file.

  5. GCF PERCEPTION (vf_* controllers — perception_launch.py from core/)
     gcf_node publishes /vf/gcf_state and /vf/voxel_filtered_pointcloud for
     the Phase-3 custom critics.

  6. RVIZ (optional, rviz:=true)

  7. SIDECARS (optional, autostart_sidecar:=true)
     vf_inferencewt → metacritic_inference_node.py Node()
     vf_imitationwt → imitation_inference_node.py Node()

────────────────────────────────────────────────────────────────────────────
INCLUDES
────────────────────────────────────────────────────────────────────────────
  vf_robot_slam/depth_to_scan.launch.py        (always)
  vf_robot_slam/rtabmap_slam.launch.py         (localization:=rtabmap_slam)
  vf_robot_slam/rtabmap_loc.launch.py          (localization:=rtabmap_loc)
  vf_robot_bringup/localization_launch.py      (localization:=amcl)
  vf_robot_bringup/slam_launch.py              (localization:=slam_toolbox)
  vf_robot_bringup/navigation_launch.py        (always)
  vf_robot_bringup/rviz_launch.py              (rviz:=true)
  vf_robot_controller/launch/core/perception_launch.py  (vf_* controllers)
  metacritic_inference_node / imitation_inference_node  (autostart_sidecar:=true)
"""

import os

from ament_index_python.packages import get_package_share_directory
from vf_robot_utils.constants import MAPS_ROOT
from vf_controller.model_defaults import (
    DEFAULT_INFERENCE_TYPE,
    DEFAULT_INFERENCE_WEIGHTS,
    DEFAULT_IMITATION_WEIGHTS,
)

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

# ============================================================================
# Argument value lists — keep in sync with config/ contents
# ============================================================================

VALID_ROBOTS = [
    "virofighter",
    "turtlebot3_waffle",
]

VALID_CONTROLLERS = [
    # VF Robot Controller — thesis modes
    "vf_fixedwt",       # fixed weights; primary ablation baseline + training data collection
    "vf_inferencewt",   # meta-critic NN weights via inference sidecar
    "vf_imitationwt",   # zero-twist plugin; imitation sidecar drives /cmd_vel_nav
    # Stock Nav2 baselines
    "mppi",
    "dwb",
    "rpp",
    "graceful",
]

VALID_LOCALIZATIONS = [
    "rtabmap_slam",
    "rtabmap_loc",
    "amcl",
    "slam_toolbox",
]

VALID_PLANNERS = [
    "NavFn",
    "SmacPlanner2D",
    "SmacPlannerHybrid",
    "SmacLattice",
    "ThetaStar",
]

# Maps controller name → robot-profile family for controller_overrides lookup.
# vf_imitationwt uses a dedicated empty family because its YAML has no MPPI
# params — applying the "vf" velocity rewrites would fail strict composition.
CONTROLLER_FAMILY = {
    "vf_fixedwt":     "vf",
    "vf_inferencewt": "vf",
    "vf_imitationwt": "imitationwt",
    "mppi":           "mppi",
    "dwb":            "dwb",
    "rpp":            "rpp",
    "graceful":       "graceful",
}

# Maps planner name → YAML filename under config/nav2/planners/
PLANNER_YAML = {
    "NavFn":             "NavFn.yaml",
    "SmacPlanner2D":     "SmacPlanner2D.yaml",
    "SmacPlannerHybrid": "SmacPlannerHybrid.yaml",
    "SmacLattice":       "SmacLattice.yaml",
    "ThetaStar":         "ThetaStar.yaml",
}

# Controllers that require a Python sidecar node.
# Value: executable name registered in vf_robot_controller setup.py.
SIDECAR_CONTROLLERS = {
    "vf_inferencewt": "metacritic_inference_node.py",
    "vf_imitationwt": "imitation_inference_node.py",
}


# ============================================================================
# OpaqueFunction: compose params + resolve map arg at launch time
# ============================================================================


def _compose_action(context, *args, **kwargs):
    """
    Called by ROS launch after arguments are resolved but before nodes spawn.

    Responsibilities:
      1. Validate robot / controller / localization arguments.
      2. Resolve the `map` arg into resolved_map_name, resolved_maps_dir,
         resolved_amcl_yaml and stash them as LaunchConfigurations.
      3. Compose the Nav2 params file and stash as composed_params_file.
      4. Print a sidecar reminder if autostart_sidecar is false.
    """
    robot = LaunchConfiguration("robot").perform(context)
    controller = LaunchConfiguration("controller").perform(context)
    localization = LaunchConfiguration("localization").perform(context)
    planner = LaunchConfiguration("planner").perform(context)
    autostart_str = LaunchConfiguration("autostart_sidecar").perform(context)
    map_arg = LaunchConfiguration("map").perform(context)
    inference_model_type = LaunchConfiguration("inference_model_type").perform(context)
    inference_weights = LaunchConfiguration("inference_weights").perform(context)
    imitation_weights = LaunchConfiguration("imitation_weights").perform(context)
    onnx_override = LaunchConfiguration("onnx_path").perform(context)
    norm_override = LaunchConfiguration("norm_path").perform(context)

    if robot not in VALID_ROBOTS:
        raise RuntimeError(
            f"[vf_bringup] Invalid robot '{robot}'. Valid: {VALID_ROBOTS}"
        )
    if controller not in VALID_CONTROLLERS:
        raise RuntimeError(
            f"[vf_bringup] Invalid controller '{controller}'. "
            f"Valid: {VALID_CONTROLLERS}"
        )
    if localization not in VALID_LOCALIZATIONS:
        raise RuntimeError(
            f"[vf_bringup] Invalid localization '{localization}'. "
            f"Valid: {VALID_LOCALIZATIONS}"
        )
    if planner not in VALID_PLANNERS:
        raise RuntimeError(
            f"[vf_bringup] Invalid planner '{planner}'. Valid: {VALID_PLANNERS}"
        )

    # ── Map resolution ────────────────────────────────────────────────────────
    # Bare name:      house_my1_map  → <MAPS_ROOT>/house_my1_map/house_my1_map.{db,yaml}
    # Absolute path:  /data/maps/hospital → /data/maps/hospital/hospital.{db,yaml}
    if os.path.isabs(map_arg):
        resolved_maps_dir = os.path.dirname(map_arg)
        resolved_map_name = os.path.basename(map_arg)
    else:
        resolved_maps_dir = str(MAPS_ROOT)
        resolved_map_name = map_arg
    resolved_amcl_yaml = os.path.join(
        resolved_maps_dir, resolved_map_name, f"{resolved_map_name}.yaml"
    )

    # ── Compose Nav2 params ───────────────────────────────────────────────────
    from vf_robot_bringup.launch_utils.compose_params import (
        compose,
        load_robot_profile,
    )

    pkg_bringup = get_package_share_directory("vf_robot_bringup")
    config_root = os.path.join(pkg_bringup, "config")

    base_path = os.path.join(config_root, "nav2", "nav2_base.yaml")
    controller_path = os.path.join(
        config_root, "nav2", "controllers", f"{controller}.yaml"
    )
    planner_path = os.path.join(
        config_root, "nav2", "planners", PLANNER_YAML[planner]
    )
    robot_path = os.path.join(config_root, "robots", f"{robot}.yaml")

    if localization == "amcl":
        loc_path = os.path.join(config_root, "nav2", "localization", "amcl.yaml")
    elif localization == "slam_toolbox":
        loc_path = os.path.join(
            config_root, "nav2", "localization", "slam_toolbox.yaml"
        )
    else:
        loc_path = None

    family = CONTROLLER_FAMILY[controller]
    robot_rewrites = load_robot_profile(
        robot_path,
        controller_family=family,
        planner_family=planner,
    )

    # Inject the resolved absolute path to our custom Nav2 BT XML.
    # nav2_base.yaml ships an empty placeholder for this key; rewriting it
    # here keeps the YAML free of build-tree-specific paths while still
    # passing strict-mode key validation.
    bt_xml_path = os.path.join(
        config_root, "nav2", "bt_xml",
        "navigate_to_pose_w_replanning_and_recovery_vf.xml",
    )
    robot_rewrites[
        "bt_navigator.ros__parameters.default_nav_to_pose_bt_xml"
    ] = bt_xml_path

    composed_path = compose(
        base_path=base_path,
        controller_path=controller_path,
        planner_path=planner_path,
        localization_path=loc_path,
        robot_rewrites=robot_rewrites,
        label=f"{robot}_{controller}_{planner}_{localization}",
        strict=True,
    )

    # ── Resolve sidecar model paths ──────────────────────────────────────────
    # Priority: onnx_path override → folder-based resolution.
    pkg_vfctrl = get_package_share_directory("vf_robot_controller")
    models_root = os.path.join(pkg_vfctrl, "models")
    if onnx_override:
        resolved_onnx = onnx_override
        resolved_norm = norm_override
    elif controller == "vf_inferencewt":
        family = f"metacritic_{inference_model_type}_wt"
        resolved_onnx = os.path.join(models_root, family, inference_weights, "meta_critic.onnx")
        resolved_norm = os.path.join(models_root, family, inference_weights, "feature_norm.json")
    elif controller == "vf_imitationwt":
        resolved_onnx = os.path.join(models_root, "imitation_wt", imitation_weights, "imitation.onnx")
        resolved_norm = os.path.join(models_root, "imitation_wt", imitation_weights, "feature_norm.json")
    else:
        resolved_onnx = ""
        resolved_norm = ""

    print(f"[vf_bringup] Composed Nav2 params: {composed_path}")
    print(f"[vf_bringup]   robot:        {robot}")
    print(f"[vf_bringup]   controller:   {controller} (family: {family})")
    print(f"[vf_bringup]   localization: {localization}")
    print(f"[vf_bringup]   planner:      {planner} ({PLANNER_YAML[planner]})")
    print(f"[vf_bringup]   map:          {resolved_map_name} (in {resolved_maps_dir})")
    print(f"[vf_bringup]   bt_xml:       {bt_xml_path}")
    print(f"[vf_bringup]   rewrites:     {len(robot_rewrites)}")
    if resolved_onnx:
        print(f"[vf_bringup]   onnx:         {resolved_onnx}")

    # ── Sidecar reminder ──────────────────────────────────────────────────────
    autostart = autostart_str.lower() in ("true", "1", "yes")
    if controller in SIDECAR_CONTROLLERS:
        executable = SIDECAR_CONTROLLERS[controller]
        weights_arg = (
            f"inference_model_type:={inference_model_type} inference_weights:={inference_weights}"
            if controller == "vf_inferencewt"
            else f"imitation_weights:={imitation_weights}"
        )
        if autostart:
            print(f"[vf_bringup] autostart_sidecar=true — spawning {executable}")
        else:
            print("[vf_bringup] " + "=" * 70)
            print(f"[vf_bringup] Controller '{controller}' requires a Python sidecar.")
            print("[vf_bringup] In a SEPARATE terminal run:")
            print("[vf_bringup]")
            print("[vf_bringup]   source ~/CA-MCW/install/setup.bash")
            print(f"[vf_bringup]   ros2 run vf_robot_controller {executable} --ros-args \\")
            print(f"[vf_bringup]       -p onnx_path:={resolved_onnx} \\")
            print(f"[vf_bringup]       -p norm_path:={resolved_norm}")
            print("[vf_bringup]")
            print(f"[vf_bringup] (Or pass autostart_sidecar:=true {weights_arg})")
            print("[vf_bringup] " + "=" * 70)

    return [
        SetLaunchConfiguration("composed_params_file", composed_path),
        SetLaunchConfiguration("resolved_map_name", resolved_map_name),
        SetLaunchConfiguration("resolved_maps_dir", resolved_maps_dir),
        SetLaunchConfiguration("resolved_amcl_yaml", resolved_amcl_yaml),
        SetLaunchConfiguration("resolved_onnx_path", resolved_onnx),
        SetLaunchConfiguration("resolved_norm_path", resolved_norm),
    ]


# ============================================================================
# generate_launch_description
# ============================================================================


def generate_launch_description():

    pkg_bringup = get_package_share_directory("vf_robot_bringup")
    pkg_slam = get_package_share_directory("vf_robot_slam")
    pkg_vfctrl = get_package_share_directory("vf_robot_controller")

    # ── Argument declarations ────────────────────────────────────────────────

    declare_robot = DeclareLaunchArgument(
        "robot",
        default_value="virofighter",
        choices=VALID_ROBOTS,
        description=f"Robot profile. One of: {VALID_ROBOTS}",
    )

    declare_controller = DeclareLaunchArgument(
        "controller",
        default_value="vf_inferencewt",
        choices=VALID_CONTROLLERS,
        description=f"Controller plugin. One of: {VALID_CONTROLLERS}",
    )

    declare_localization = DeclareLaunchArgument(
        "localization",
        default_value="rtabmap_loc",
        choices=VALID_LOCALIZATIONS,
        description=f"SLAM/localization mode. One of: {VALID_LOCALIZATIONS}",
    )

    declare_planner = DeclareLaunchArgument(
        "planner",
        default_value="NavFn",
        choices=VALID_PLANNERS,
        description=(
            f"Global planner plugin. One of: {VALID_PLANNERS}. "
            "Selects the matching YAML from config/nav2/planners/ and merges "
            "it into the composed Nav2 params."
        ),
    )

    declare_camera = DeclareLaunchArgument(
        "camera",
        default_value="dual",
        choices=["d435i", "d455", "dual"],
        description="Camera configuration for depth_to_scan and RTAB-Map",
    )

    declare_scan_method = DeclareLaunchArgument(
        "scan_method",
        default_value="pc2scan",
        choices=["dimg", "pc2scan"],
        description="Depth-to-scan conversion method",
    )

    declare_merge_scans = DeclareLaunchArgument(
        "merge_scans",
        default_value="true",
        choices=["true", "false"],
        description="Merge dual camera scans into single /scan topic",
    )

    declare_map = DeclareLaunchArgument(
        "map",
        default_value="house_my1_map",
        description=(
            "Bare map name (e.g. house_my1_map) or absolute folder path "
            "(e.g. /data/maps/hospital). Bare name is resolved under "
            "$VF_MAPS_ROOT; absolute path uses dirname/basename split."
        ),
    )

    declare_new_map = DeclareLaunchArgument(
        "new_map",
        default_value="true",
        choices=["true", "false"],
        description="RTAB-Map SLAM: delete existing .db and start fresh",
    )

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        choices=["true", "false"],
        description="Use Gazebo clock (true) or wall clock (false)",
    )

    declare_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        choices=["true", "false"],
        description="Launch RViz with Nav2 panels (forced false when headless:=true)",
    )

    declare_headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        choices=["true", "false"],
        description=(
            "Suppress all GUI surfaces owned by bringup. When true, RViz "
            "is not started regardless of the `rviz:=` value."
        ),
    )

    declare_rviz_config = DeclareLaunchArgument(
        "rviz_config",
        default_value=os.path.join(pkg_bringup, "rviz", "vf_bringup.rviz"),
        description="Full path to RViz config file",
    )

    declare_autostart_sidecar = DeclareLaunchArgument(
        "autostart_sidecar",
        default_value="true",
        choices=["true", "false"],
        description=(
            "Spawn the Python sidecar Node directly from bringup when "
            "controller is vf_inferencewt or vf_imitationwt. Default true: "
            "the resolved ONNX (from model_defaults.py or CLI overrides) is "
            "loaded automatically. Pass false to start the sidecar manually "
            "in another terminal — bringup will print the ros2 run command."
        ),
    )

    declare_inference_model_type = DeclareLaunchArgument(
        "inference_model_type",
        default_value=DEFAULT_INFERENCE_TYPE,
        choices=["raw", "oracle"],
        description=(
            '"raw" → models/metacritic_raw_wt/   '
            '"oracle" → models/metacritic_oracle_wt/'
        ),
    )

    declare_inference_weights = DeclareLaunchArgument(
        "inference_weights",
        default_value=DEFAULT_INFERENCE_WEIGHTS,
        description=(
            "Run folder inside metacritic_{type}_wt/ to use for vf_inferencewt. "
            "e.g. raw_manual_v1, oracle_manual_v1"
        ),
    )

    declare_imitation_weights = DeclareLaunchArgument(
        "imitation_weights",
        default_value=DEFAULT_IMITATION_WEIGHTS,
        description=(
            "Run folder inside models/imitation_wt/ to use for vf_imitationwt. "
            "e.g. manual_v1, run2_2026_05_12"
        ),
    )

    declare_onnx_path = DeclareLaunchArgument(
        "onnx_path",
        default_value="",
        description=(
            "Escape hatch: absolute path to a .onnx model; bypasses "
            "inference_model_type + inference_weights / imitation_weights."
        ),
    )

    declare_norm_path = DeclareLaunchArgument(
        "norm_path",
        default_value="",
        description="Absolute path to feature_norm.json — only used when onnx_path is set.",
    )

    declare_channel_config = DeclareLaunchArgument(
        "channel_config",
        default_value="channels_v1",
        choices=["channels_v1", "channels_v2", "channels_v3"],
        description=(
            "Phase 5 / Phase 6 perception channel set. channels_v1 = 6 "
            "non-SLAM channels (126 dims), channels_v2 adds reynolds (130 "
            "dims), channels_v3 adds slam_persistent (170 dims)."
        ),
    )

    declare_rtabmap_db_path = DeclareLaunchArgument(
        "rtabmap_db_path",
        default_value="",
        description=(
            "Phase 6: full path to the RTAB-Map .db consumed by RtabmapBackend. "
            "Empty (default) lets backend_selection.yaml decide."
        ),
    )

    # ── Composer step (must run before any node that needs the params) ────────

    compose_op = OpaqueFunction(function=_compose_action)

    # ── depth_to_scan — always runs (all 4 modes need /scan) ─────────────────

    depth_to_scan = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_slam, "launch", "depth_to_scan.launch.py")
        ),
        launch_arguments={
            "method": LaunchConfiguration("scan_method"),
            "camera": LaunchConfiguration("camera"),
            "merge_scans": LaunchConfiguration("merge_scans"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    # ── Mode 1: RTAB-Map SLAM ─────────────────────────────────────────────────

    rtabmap_slam_branch = GroupAction(
        condition=LaunchConfigurationEquals("localization", "rtabmap_slam"),
        actions=[
            LogInfo(msg="[vf_bringup] localization: rtabmap_slam (build new map)"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_slam, "launch", "rtabmap_slam.launch.py")
                ),
                launch_arguments={
                    "camera": LaunchConfiguration("camera"),
                    "map_name": LaunchConfiguration("resolved_map_name"),
                    "maps_dir": LaunchConfiguration("resolved_maps_dir"),
                    "new_map": LaunchConfiguration("new_map"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "rviz": "false",
                }.items(),
            ),
        ],
    )

    # ── Mode 2: RTAB-Map Localization ─────────────────────────────────────────

    rtabmap_loc_branch = GroupAction(
        condition=LaunchConfigurationEquals("localization", "rtabmap_loc"),
        actions=[
            LogInfo(msg="[vf_bringup] localization: rtabmap_loc (load existing map)"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_slam, "launch", "rtabmap_loc.launch.py")
                ),
                launch_arguments={
                    "camera": LaunchConfiguration("camera"),
                    "map_name": LaunchConfiguration("resolved_map_name"),
                    "maps_dir": LaunchConfiguration("resolved_maps_dir"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "rviz": "false",
                }.items(),
            ),
        ],
    )

    # ── Mode 3: AMCL + map_server ─────────────────────────────────────────────

    amcl_branch = GroupAction(
        condition=LaunchConfigurationEquals("localization", "amcl"),
        actions=[
            LogInfo(msg="[vf_bringup] localization: amcl (load 2D occupancy map)"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_bringup, "launch", "localization_launch.py")
                ),
                launch_arguments={
                    "map": LaunchConfiguration("resolved_amcl_yaml"),
                    "params_file": LaunchConfiguration("composed_params_file"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                }.items(),
            ),
        ],
    )

    # ── Mode 4: SLAM Toolbox ──────────────────────────────────────────────────

    slam_toolbox_branch = GroupAction(
        condition=LaunchConfigurationEquals("localization", "slam_toolbox"),
        actions=[
            LogInfo(msg="[vf_bringup] localization: slam_toolbox (laser-based SLAM)"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_bringup, "launch", "slam_launch.py")
                ),
                launch_arguments={
                    "params_file": LaunchConfiguration("composed_params_file"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                }.items(),
            ),
        ],
    )

    # ── Nav2 stack — always runs ──────────────────────────────────────────────

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "navigation_launch.py")
        ),
        launch_arguments={
            "params_file": LaunchConfiguration("composed_params_file"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "controller": LaunchConfiguration("controller"),
        }.items(),
    )

    # ── GCF perception — all three VF controllers need gcf_node ─────────────
    # gcf_node publishes /vf/gcf_state and /vf/voxel_filtered_pointcloud for
    # the three custom critics (CorridorCritic, VolumetricCritic,
    # DynamicObstacleCritic) and /vf/features for the inference/imitation
    # sidecars.
    gcf_perception_branches = [
        GroupAction(
            condition=LaunchConfigurationEquals("controller", ctrl),
            actions=[
                LogInfo(msg=f"[vf_bringup] starting gcf_node for {ctrl}"),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(
                            pkg_vfctrl, "launch", "core", "perception_launch.py"
                        )
                    ),
                    launch_arguments={
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "channel_config": LaunchConfiguration("channel_config"),
                        "rtabmap_db_path": LaunchConfiguration("rtabmap_db_path"),
                    }.items(),
                ),
            ],
        )
        for ctrl in ("vf_fixedwt", "vf_inferencewt", "vf_imitationwt")
    ]

    # ── Sidecar nodes — spawned directly when autostart_sidecar:=true ─────────
    # onnxruntime is installed in system Python (pip3 install onnxruntime);
    # no conda needed. Pass onnx_path/norm_path so the node finds its model.

    _inference_cond = PythonExpression([
        "'", LaunchConfiguration("controller"), "' == 'vf_inferencewt'"
        " and '", LaunchConfiguration("autostart_sidecar"), "' == 'true'",
    ])
    inference_sidecar_branch = GroupAction(
        condition=IfCondition(_inference_cond),
        actions=[
            LogInfo(msg="[vf_bringup] autostart_sidecar: starting metacritic_inference_node"),
            Node(
                package="vf_robot_controller",
                executable="metacritic_inference_node.py",
                name="metacritic_inference_node",
                output="screen",
                parameters=[{
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "onnx_path": LaunchConfiguration("resolved_onnx_path"),
                    "norm_path": LaunchConfiguration("resolved_norm_path"),
                }],
            ),
        ],
    )

    _imitation_cond = PythonExpression([
        "'", LaunchConfiguration("controller"), "' == 'vf_imitationwt'"
        " and '", LaunchConfiguration("autostart_sidecar"), "' == 'true'",
    ])
    imitation_sidecar_branch = GroupAction(
        condition=IfCondition(_imitation_cond),
        actions=[
            LogInfo(msg="[vf_bringup] autostart_sidecar: starting imitation_inference_node"),
            Node(
                package="vf_robot_controller",
                executable="imitation_inference_node.py",
                name="imitation_inference_node",
                output="screen",
                parameters=[{
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "onnx_path": LaunchConfiguration("resolved_onnx_path"),
                    "norm_path": LaunchConfiguration("resolved_norm_path"),
                }],
            ),
        ],
    )

    # ── RViz — optional ───────────────────────────────────────────────────────

    rviz_enable_expr = PythonExpression([
        "'", LaunchConfiguration("rviz"), "' == 'true' and '",
        LaunchConfiguration("headless"), "' != 'true'",
    ])
    rviz_branch = GroupAction(
        condition=IfCondition(rviz_enable_expr),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_bringup, "launch", "rviz_launch.py")
                ),
                launch_arguments={
                    "rviz_config": LaunchConfiguration("rviz_config"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                }.items(),
            ),
        ],
    )

    # ── Assemble the launch description ───────────────────────────────────────

    return LaunchDescription(
        [
            # Arguments
            declare_robot,
            declare_controller,
            declare_localization,
            declare_planner,
            declare_camera,
            declare_scan_method,
            declare_merge_scans,
            declare_map,
            declare_new_map,
            declare_use_sim_time,
            declare_rviz,
            declare_rviz_config,
            declare_headless,
            declare_autostart_sidecar,
            declare_inference_model_type,
            declare_inference_weights,
            declare_imitation_weights,
            declare_onnx_path,
            declare_norm_path,
            declare_channel_config,
            declare_rtabmap_db_path,
            # Compose params + resolve map — produces composed_params_file
            # and resolved_map_name / resolved_maps_dir / resolved_amcl_yaml
            compose_op,
            # depth_to_scan — all modes need /scan
            depth_to_scan,
            # Exactly ONE of these branches activates based on localization:=
            rtabmap_slam_branch,
            rtabmap_loc_branch,
            amcl_branch,
            slam_toolbox_branch,
            # Nav2 stack — always runs, reads composed_params_file
            navigation,
            # GCF perception (all vf_* controllers)
            *gcf_perception_branches,
            # Sidecars — spawn as Nodes when autostart_sidecar:=true
            inference_sidecar_branch,
            imitation_sidecar_branch,
            # RViz — optional
            rviz_branch,
        ]
    )
