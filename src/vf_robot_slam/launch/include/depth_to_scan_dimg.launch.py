#!/usr/bin/env python3
"""
Depth → LaserScan via depthimage_to_laserscan — ViroFighter UVC-1
══════════════════════════════════════════════════════════════════

DO NOT launch this file directly. It is included by:
    ros2 launch vf_robot_slam depth_to_scan.launch.py method:=dimg

────────────────────────────────────────────────────────────
ARGUMENTS
────────────────────────────────────────────────────────────
  camera        : d435i | d455 | dual    (default: dual)
  merge_scans   : true | false           (default: true)
  use_sim_time  : true | false           (default: true)

────────────────────────────────────────────────────────────
WHEN TO USE
────────────────────────────────────────────────────────────
  Simulation (Gazebo) — RECOMMENDED:
    Processes a 2D depth image row, not a heavy PointCloud2.
    Runs at full 15–30 Hz even under Gazebo CPU load.

  Real robot — D455 only:
    D455 is horizontal at 0.429 m height — scan plane sits at
    correct obstacle height. Works correctly with no caveats.

  Real robot — D435i:
    D435i at 1.773 m with 60° downward tilt. The scan plane
    intersects the floor at ~1.02 m. range_min=1.1 m clips
    floor returns but also creates a blind zone for anything
    closer than 1.1 m. Use method:=pc2scan for D435i on real
    robot if close-obstacle detection is required.

────────────────────────────────────────────────────────────
OUTPUT TOPICS
────────────────────────────────────────────────────────────
  camera=d435i                          →  /scan
  camera=d455                           →  /scan
  camera=dual, merge_scans=true         →  /scan_d435i + /scan_d455 + /scan
  camera=dual, merge_scans=false        →  /scan_d435i + /scan_d455 only

────────────────────────────────────────────────────────────
NODES SPAWNED
────────────────────────────────────────────────────────────
  scan_merger        —  vf_robot_slam/scan_merger.py       (dual + merge only)
  depth_to_scan_d455  —  depthimage_to_laserscan_node
  depth_to_scan_d435i —  depthimage_to_laserscan_node

────────────────────────────────────────────────────────────
CAMERA PARAMETERS
────────────────────────────────────────────────────────────
  D455  — output_frame=camera_d455_link, range_min=0.6 m, range_max=6.0 m
           output_frame is camera_d455_link (NOT base_footprint) because
           depthimage_to_laserscan only stamps the header — it does not
           rotate angle values. D455 faces rear; angle=0 draws backward ✓

  D435i — output_frame=base_footprint, range_min=1.1 m, range_max=6.0 m
           Faces forward; angle=0 = +X_base = forward ✓
           range_min=1.1 m clips floor at ≤1.024 m; blind zone < 1.1 m

────────────────────────────────────────────────────────────
CRITICAL: STARTUP ORDER FOR DUAL MODE (Bug fix 2026-04-05)
────────────────────────────────────────────────────────────
  depthimage_to_laserscan uses LAZY SUBSCRIPTION — it will NOT
  subscribe to its depth image input until at least one node
  subscribes to its /scan output.

  FIX: scan_merger MUST start BEFORE the converter nodes.
  When converters check for subscribers, the merger is already
  listening → converters activate immediately.

  If reversed: converters start with zero subscribers, go idle,
  and never recover until restarted. This is why a second launch
  used to "fix" it — the new merger triggered the lazy check.

  Order enforced in this file: merger → d435i converter → d455 converter.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ── D455 parameters — horizontal, 0.429 m height ────────────────────────────
D455_PARAMS = {
    "range_min": 0.6,
    "range_max": 6.0,
    "scan_height": 1,
    "scan_time": 0.033,
    "inf_is_valid": False,
    "output_frame": "camera_d455_link",
}

# ── D435i parameters — 60° tilt, 1.773 m height ─────────────────────────────
D435I_PARAMS = {
    "range_min": 1.1,  # clips floor at ≤1.024 m; blind zone < 1.1 m
    "range_max": 6.0,
    "scan_height": 1,
    "scan_time": 0.033,
    "inf_is_valid": False,
    "output_frame": "base_footprint",
}


def _make_d455_node(name, scan_topic, use_sim_time):
    """Create a depthimage_to_laserscan node for the D455 camera."""
    return Node(
        package="depthimage_to_laserscan",
        executable="depthimage_to_laserscan_node",
        name=name,
        output="screen",
        parameters=[{**D455_PARAMS, "use_sim_time": use_sim_time}],
        remappings=[
            ("depth", "/d455/depth/d455_depth/depth/image_raw"),
            ("depth_camera_info", "/d455/depth/d455_depth/depth/camera_info"),
            ("scan", scan_topic),
        ],
    )


def _make_d435i_node(name, scan_topic, use_sim_time):
    """Create a depthimage_to_laserscan node for the D435i camera."""
    return Node(
        package="depthimage_to_laserscan",
        executable="depthimage_to_laserscan_node",
        name=name,
        output="screen",
        parameters=[{**D435I_PARAMS, "use_sim_time": use_sim_time}],
        remappings=[
            ("depth", "/d435i/depth/d435i_depth/depth/image_raw"),
            ("depth_camera_info", "/d435i/depth/d435i_depth/depth/camera_info"),
            ("scan", scan_topic),
        ],
    )


def launch_setup(context, *args, **kwargs):
    camera = LaunchConfiguration("camera").perform(context)
    merge_scans = LaunchConfiguration("merge_scans").perform(context)
    sim_time = LaunchConfiguration("use_sim_time").perform(context)
    use_sim_time = sim_time.lower() == "true"

    nodes = []

    # ── Single camera: d435i ─────────────────────────────────────────────────
    if camera == "d435i":
        nodes.append(
            LogInfo(
                msg=[
                    "\n",
                    "=" * 70,
                    "\n",
                    "depth_to_scan [dimg]: D435i → /scan\n",
                    "  range_min=1.1 m (floor clipped), blind zone < 1.1 m\n",
                    "=" * 70,
                    "\n",
                ]
            )
        )
        nodes.append(_make_d435i_node("depth_to_scan_d435i", "/scan", use_sim_time))

    # ── Single camera: d455 ──────────────────────────────────────────────────
    elif camera == "d455":
        nodes.append(
            LogInfo(
                msg=[
                    "\n",
                    "=" * 70,
                    "\n",
                    "depth_to_scan [dimg]: D455 → /scan\n",
                    "  output_frame=camera_d455_link, range_min=0.6 m\n",
                    "=" * 70,
                    "\n",
                ]
            )
        )
        nodes.append(_make_d455_node("depth_to_scan_d455", "/scan", use_sim_time))

    # ── Dual camera ──────────────────────────────────────────────────────────
    #
    # CRITICAL STARTUP ORDER:
    #   1. scan_merger FIRST  — subscribes to /scan_d435i and /scan_d455
    #   2. depth_to_scan_d435i — checks for subscribers on /scan_d435i,
    #                            finds scan_merger → activates depth input
    #   3. depth_to_scan_d455  — same for /scan_d455
    #
    # If this order is reversed, the converter nodes start with zero
    # subscribers and their lazy subscription keeps them idle forever.
    # ─────────────────────────────────────────────────────────────────────────
    else:
        # ── Step 1: Start scan_merger FIRST (if merging) ─────────────────────
        if merge_scans.lower() == "true":
            nodes.append(
                LogInfo(
                    msg=[
                        "\n",
                        "=" * 70,
                        "\n",
                        "depth_to_scan [dimg]: dual — scan_merger ACTIVE\n",
                        "  /scan_d435i + /scan_d455 → /scan (merged)\n",
                        "=" * 70,
                        "\n",
                    ]
                )
            )
            nodes.append(
                Node(
                    package="vf_robot_slam",
                    executable="scan_merger.py",
                    name="scan_merger",
                    output="screen",
                    parameters=[
                        {
                            "use_sim_time": use_sim_time,
                            "scan_topics": "/scan_d435i /scan_d455",
                            "output_topic": "/scan",
                            "output_frame": "base_footprint",
                            "angle_min": -3.14159,
                            "angle_max": 3.14159,
                            "range_min": 0.1,
                            "range_max": 6.0,
                        }
                    ],
                )
            )

        # ── Step 2: Start converter nodes AFTER merger is subscribed ─────────
        nodes.append(
            _make_d435i_node("depth_to_scan_d435i", "/scan_d435i", use_sim_time)
        )
        nodes.append(_make_d455_node("depth_to_scan_d455", "/scan_d455", use_sim_time))

        # ── No merge: warn the user ──────────────────────────────────────────
        if merge_scans.lower() != "true":
            nodes.append(
                LogInfo(
                    msg=[
                        "\n",
                        "=" * 70,
                        "\n",
                        "depth_to_scan [dimg]: dual — merge_scans:=false\n",
                        "  /scan_d435i + /scan_d455 published (no /scan)\n",
                        "  AMCL will not work — use Nav2 multi-source costmap\n",
                        "=" * 70,
                        "\n",
                    ]
                )
            )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera", default_value="dual", choices=["d435i", "d455", "dual"]
            ),
            DeclareLaunchArgument(
                "merge_scans", default_value="true", choices=["true", "false"]
            ),
            DeclareLaunchArgument(
                "use_sim_time", default_value="true", choices=["true", "false"]
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
