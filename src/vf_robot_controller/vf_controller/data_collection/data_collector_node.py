#!/usr/bin/env python3
"""
data_collector_node — Phase 7 sidecar.

Listens to the COLLECT-mode topic surface (post-M10 published unconditionally
by vf_fixedwt and vf_inferencewt) and writes one HDF5 file per navigation
episode (schema in vf_robot_controller/PACKAGE.txt).

Session layout
--------------
A *session* is one launch of this node. Episodes land under:

  <VF_DATA_ROOT>/vf_data_training/<session_kind>/<map>_<YYYYMMDD_HHMMSS>/
      session.json
      manifest.csv
      ep_000_<scenario>_<stamp>.h5
      ep_001_<scenario>_<stamp>.h5
      ...

session_kind:= manual    (launch/vf_data_training/manual/vf_data_training_manual_fixedwt.launch.py + RViz drives)
session_kind:= batch (csv_runner in collect mode passes session_dir:=)

The csv_runner overrides session_dir directly; manual launches pass map_name
and let this node compute the session folder.

Goal debounce + cooldown (fixes "many files per single goal" bug)
----------------------------------------------------------------
Goals can arrive on /goal_pose (RViz "2D Goal Pose") *or* /plan (final pose
of the planned path; this is the path used by RViz "Nav2 Goal", action
goals, episode_runner.py, ...). /plan fires every replan, so naive accept
opens a fresh file per replan. The fix:

  1. Debounce: a new candidate goal must remain stable (within
     goal_dedup_radius_m XY and goal_yaw_eps_rad yaw) for goal_debounce_s
     before we open an episode.
  2. Cooldown: after closing an episode, we latch out any candidate within
     goal_dedup_radius_m of the just-closed goal for goal_cooldown_s. This
     prevents the very next /plan from re-opening the same goal we just
     finished.

Result: strictly one HDF5 per goal, even when /goal_pose and /plan both
arrive and the planner replans every cycle.

Live status
-----------
Every flush tick we publish std_msgs/Float32MultiArray on
``/vf/collector_status`` with [n_episodes_closed, total_cycles, total_bytes,
current_episode_cycles, current_episode_bytes]. ``vf_session_status`` reads
session.json + manifest.csv for offline-only viewing too.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool, Float32MultiArray

try:
    from action_msgs.msg import GoalStatusArray

    HAS_GOAL_STATUS_MSG = True
except ImportError:  # pragma: no cover
    GoalStatusArray = None  # type: ignore
    HAS_GOAL_STATUS_MSG = False

# action_msgs/GoalStatus terminal codes — mirror the message constants
# so we don't take a build-time dep on action_msgs for these ints.
_NAV2_STATUS_SUCCEEDED = 4
_NAV2_STATUS_CANCELED = 5
_NAV2_STATUS_ABORTED = 6
_NAV2_STATUS_TERMINAL = (
    _NAV2_STATUS_SUCCEEDED,
    _NAV2_STATUS_CANCELED,
    _NAV2_STATUS_ABORTED,
)
_NAV2_STATUS_REASONS = {
    _NAV2_STATUS_SUCCEEDED: "nav2_succeeded",
    _NAV2_STATUS_CANCELED: "nav2_canceled",
    _NAV2_STATUS_ABORTED: "nav2_aborted",
}

try:
    from vf_robot_messages.msg import MppiCriticsStats

    HAS_CRITIC_MSG = True
except ImportError:  # pragma: no cover
    MppiCriticsStats = None  # type: ignore
    HAS_CRITIC_MSG = False

from vf_controller.data_collection.episode_writer import (
    CycleRow,
    EpisodeMetadata,
    EpisodeOutcome,
    EpisodeWriter,
    _git_commit,
    _isoformat_now,
)
from vf_controller.data_collection.goal_debouncer import (
    DebouncerConfig,
    Decision,
    GoalDebouncer,
)
from vf_controller.data_collection.session import (
    SESSION_KIND_BATCH,
    SESSION_KIND_MANUAL,
    SessionInfo,
    append_manifest_row,
    run_filename,
    session_dir_for,
    vfdata_leaf_for,
    write_session_json,
)


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _default_training_root() -> str:
    """Return the default vf_data/vf_data_training/ root."""
    try:
        from vf_robot_utils.constants import TRAINING_ROOT
        return str(TRAINING_ROOT)
    except ImportError:  # pragma: no cover
        env = os.environ.get("VF_DATA_ROOT")
        if env:
            return str(os.path.join(env, "vf_data_training"))
        return os.path.expanduser("~/CA-MCW/vf_data/vf_data_training")


class DataCollectorNode(Node):
    DEFAULT_FEATURE_DIM = 126  # channels_v1
    DEFAULT_CRITIC_COUNT = 11  # 8 wrappers + 3 custom (vf_fixedwt.yaml)

    # --------------------------------------------------------------- ctor
    def __init__(self) -> None:
        super().__init__("data_collector_node")

        # Parameters ────────────────────────────────────────────────────
        # vf_data path resolution:
        #   training_root / session_kind / map_name / goal_folder / planner / controller /
        # The goal_folder is computed per-episode from the actual goal pose.
        # Pass session_dir explicitly to bypass goal-based path derivation
        # (override for external scripts / csv_runner).
        self.declare_parameter("training_root", _default_training_root())
        self.declare_parameter("planner", "")        # PascalCase, e.g. "NavFn"
        self.declare_parameter("controller", "vf_fixedwt")
        self.declare_parameter("session_kind", SESSION_KIND_MANUAL)
        self.declare_parameter("map_name", "unknown_map")
        self.declare_parameter("session_dir", "")
        self.declare_parameter("session_suffix", "")

        self.declare_parameter("scenario_id", "manual_run")
        self.declare_parameter("seed", 0)
        self.declare_parameter("controller_mode", "collect")
        self.declare_parameter("weight_provider", "fixed")
        self.declare_parameter("channels_config", "channels_v1")
        # Empty default: filled from feature_extractor parameters or layout.
        self.declare_parameter("channel_names", [""])
        self.declare_parameter("channel_dims", [0])
        self.declare_parameter("critic_names", [""])
        self.declare_parameter("flush_period_s", 1.0)
        self.declare_parameter("write_period_s", 0.05)  # 20 Hz cycle clock
        self.declare_parameter("goal_radius_m", 0.4)
        self.declare_parameter("goal_reached_consecutive", 5)
        self.declare_parameter("episode_timeout_s", 180.0)
        self.declare_parameter("max_obstacles", 0)

        # Goal debounce / cooldown — fixes the "many files per goal" bug.
        self.declare_parameter("goal_debounce_s", 0.5)
        self.declare_parameter("goal_cooldown_s", 2.0)
        self.declare_parameter("goal_dedup_radius_m", 0.5)
        self.declare_parameter("goal_yaw_eps_rad", 0.35)  # ~20 deg

        self.declare_parameter("dynamic_obstacles_topic", "/vf/dynamic_obstacles_gt")
        self.declare_parameter("applied_weights_topic", "/vf/applied_weights")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("features_topic", "/vf/features")
        self.declare_parameter("critic_costs_topic", "/vf/per_critic_costs")
        self.declare_parameter("plan_topic", "/plan")
        self.declare_parameter("status_topic", "/vf/collector_status")
        self.declare_parameter("close_on_new_goal", True)

        # Nav2 action-status driven close (preferred over goal_radius_m
        # tolerance check). When tour_runner sends NavigateToPose goals,
        # the Nav2 action server publishes terminal status codes
        # (SUCCEEDED=4, CANCELED=5, ABORTED=6) on this topic. We listen
        # and schedule the episode close after `nav2_close_settle_s`
        # seconds so robot odom can settle into its final pose. The
        # tolerance-based close (goal_radius_m + goal_reached_consecutive)
        # remains as a fallback for manual / RViz sessions where no
        # NavigateToPose action is fired.
        self.declare_parameter(
            "nav2_action_status_topic", "/navigate_to_pose/_action/status"
        )
        self.declare_parameter("nav2_close_settle_s", 3.0)
        self.declare_parameter("nav2_status_close_enabled", True)

        # External recording gate. When `recording_enabled` is False, the
        # collector ignores incoming /goal_pose and /plan messages — no
        # episode opens, no HDF5 written. tour_runner toggles this
        # (False during the reposition leg, True for the actual tour) so
        # the spawn→start drive is not saved as data.
        self.declare_parameter(
            "recording_enabled_topic", "/vf/recording_enabled"
        )
        self.declare_parameter("recording_enabled_default", True)

        # Batch-mode explicit episode control. When batch_mode=True the
        # collector ignores /goal_pose, /plan, the Nav2 action status
        # closer and the goal_radius_m tolerance closer; episode open and
        # close are driven solely by tour_runner via batch_control_topic.
        # Format: std_msgs/Float32MultiArray data = [cmd, gx, gy, gyaw,
        # success] where cmd=1.0 opens with configured goal coords and
        # cmd=0.0 closes with success in {0.0, 1.0}. Folder name uses the
        # configured goal coords (what was asked); the manifest's
        # goal_x/y/yaw is overwritten with the stabilized odometry pose
        # at close time so it reflects where the robot actually stopped.
        self.declare_parameter("batch_mode", False)
        self.declare_parameter(
            "batch_control_topic", "/vf/batch/episode_control"
        )

        gp = self.get_parameter
        self.training_root = str(gp("training_root").value)
        self.planner = str(gp("planner").value).strip() or "unknown_planner"
        self.controller = str(gp("controller").value).strip() or "vf_fixedwt"
        self.session_kind = str(gp("session_kind").value).lower().strip()
        if self.session_kind not in (
            SESSION_KIND_MANUAL, SESSION_KIND_BATCH
        ):
            self.get_logger().warn(
                "Unknown session_kind=%r; defaulting to %r"
                % (self.session_kind, SESSION_KIND_MANUAL)
            )
            self.session_kind = SESSION_KIND_MANUAL
        self.map_name = str(gp("map_name").value) or "unknown_map"
        explicit_session_dir = str(gp("session_dir").value).strip()
        session_suffix = str(gp("session_suffix").value).strip()

        self.scenario_id = str(gp("scenario_id").value)
        self.seed = int(gp("seed").value)
        self.controller_mode = str(gp("controller_mode").value)
        self.weight_provider = str(gp("weight_provider").value)
        self.channels_config = str(gp("channels_config").value)
        self.channel_names = [s for s in gp("channel_names").value if s]
        self.channel_dims = [int(d) for d in gp("channel_dims").value if int(d) > 0]
        self.critic_names = [s for s in gp("critic_names").value if s]
        self.flush_period_s = float(gp("flush_period_s").value)
        self.write_period_s = float(gp("write_period_s").value)
        self.goal_radius_m = float(gp("goal_radius_m").value)
        self.goal_reached_consecutive = int(gp("goal_reached_consecutive").value)
        self.episode_timeout_s = float(gp("episode_timeout_s").value)
        self.max_obstacles = int(gp("max_obstacles").value)
        self.goal_debounce_s = float(gp("goal_debounce_s").value)
        self.goal_cooldown_s = float(gp("goal_cooldown_s").value)
        self.goal_dedup_radius_m = float(gp("goal_dedup_radius_m").value)
        self.goal_yaw_eps_rad = float(gp("goal_yaw_eps_rad").value)
        self.close_on_new_goal = bool(gp("close_on_new_goal").value)
        self.nav2_close_settle_s = float(gp("nav2_close_settle_s").value)
        self.nav2_status_close_enabled = bool(
            gp("nav2_status_close_enabled").value
        )
        self._recording_enabled = bool(gp("recording_enabled_default").value)
        self.batch_mode = bool(gp("batch_mode").value)

        # Resolve path mode.
        # If session_dir is given explicitly, use it (legacy/override mode).
        # Otherwise use per-goal vf_data leaf paths computed at episode open.
        self._explicit_session_dir: str = explicit_session_dir
        if explicit_session_dir:
            # Legacy override: all episodes go to this single folder.
            self.session_dir = explicit_session_dir
            os.makedirs(self.session_dir, exist_ok=True)
            self._use_vfdata_paths = False
        else:
            # New mode: each episode lands in its own goal-specific leaf.
            # self.session_dir is set per-episode in _open_episode().
            self.session_dir = ""
            self._use_vfdata_paths = True
        self._session_json_written = False
        self._next_episode_index = 0

        # Cached topic snapshots (latest-value-wins) ─────────────────────
        self._features: Optional[np.ndarray] = None
        self._features_dim: int = 0
        self._critic_costs: Optional[np.ndarray] = None
        self._critic_count: int = 0
        self._weights: Optional[np.ndarray] = None
        self._cmd_vel: Optional[np.ndarray] = None  # [vx, vy, wz]
        self._odom_pose: Optional[np.ndarray] = None  # [x, y, theta]
        self._goal_pose: Optional[np.ndarray] = None  # [x, y, theta]
        self._dyn_obstacles: Optional[np.ndarray] = None  # (M, 5) or None

        # Episode bookkeeping ────────────────────────────────────────────
        self._writer: Optional[EpisodeWriter] = None
        self._writer_path: str = ""
        self._writer_index: int = -1
        self._episode_start_time: Optional[float] = None
        self._episode_start_iso: str = ""
        self._episode_start_pose: Optional[np.ndarray] = None
        self._episode_goal_pose: Optional[np.ndarray] = None
        self._last_pose_for_path: Optional[np.ndarray] = None
        self._path_length_m: float = 0.0
        self._consec_in_radius: int = 0
        self._step_index: int = 0

        # Goal state machine — pure-Python, unit-tested without rclpy.
        self._debouncer = GoalDebouncer(DebouncerConfig(
            debounce_s=self.goal_debounce_s,
            cooldown_s=self.goal_cooldown_s,
            dedup_radius_m=self.goal_dedup_radius_m,
            yaw_eps_rad=self.goal_yaw_eps_rad,
        ))

        # Aggregate session stats published on /vf/collector_status.
        self._session_episodes_closed: int = 0
        self._session_total_steps: int = 0
        self._session_total_bytes: int = 0

        # Nav2 action-status close state. Set when a terminal status
        # arrives; the tick timer fires the close once sim-time reaches
        # `_pending_nav2_close_at`. New goal arrival resets these (the
        # new_goal close path takes over).
        self._pending_nav2_close_at: Optional[float] = None
        self._pending_nav2_close_reason: str = ""
        self._pending_nav2_close_success: bool = False
        self._last_nav2_terminal_goal_id: Optional[bytes] = None

        # Per-episode /plan history for the tier-6 XTE metric. Every /plan
        # that arrives while a writer is open is appended as
        # (sim_time_seconds, (N,3) xy-yaw polyline). Reset on episode open;
        # serialised under HDF5 group ``global_path_plans/`` on close.
        # Independent of batch_mode / recording_enabled goal gates — those
        # govern episode lifecycle, not metric collection.
        self._plan_history: list[tuple[float, "np.ndarray"]] = []

        self._warned: dict = {}

        # Subscriptions ──────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # latched QoS reserved for future use; kept for symmetry with
        # other VF nodes that subscribe to /map etc.
        _ = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Float32MultiArray, str(gp("features_topic").value),
            self._on_features, 10,
        )
        self.create_subscription(
            Twist, str(gp("cmd_vel_topic").value),
            self._on_cmd_vel, 10,
        )
        self.create_subscription(
            Odometry, str(gp("odom_topic").value),
            self._on_odom, sensor_qos,
        )
        self.create_subscription(
            PoseStamped, str(gp("goal_topic").value),
            self._on_goal, 10,
        )
        # /plan covers RViz Nav2 Goal, action-goals, episode_runner.py.
        self.create_subscription(
            Path, str(gp("plan_topic").value),
            self._on_plan, 10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(gp("applied_weights_topic").value),
            self._on_weights, 10,
        )
        if HAS_CRITIC_MSG:
            self.create_subscription(
                MppiCriticsStats,  # type: ignore[arg-type]
                str(gp("critic_costs_topic").value),
                self._on_critic_costs, 20,
            )

        # External recording gate (latched). tour_runner uses this to
        # silence the spawn→start reposition leg. When False, _on_goal /
        # _on_plan early-return without proposing anything to the
        # debouncer. The default value is set via launch param so manual
        # RViz sessions (which don't publish to this topic) keep working.
        gate_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Bool, str(gp("recording_enabled_topic").value),
            self._on_recording_enabled, gate_qos,
        )

        # Batch-mode explicit episode lifecycle control. Always subscribed
        # (the subscription is cheap); the callback is a no-op when
        # batch_mode is False.
        self.create_subscription(
            Float32MultiArray,
            str(gp("batch_control_topic").value),
            self._on_batch_control, 10,
        )

        # Nav2 action status — listen for SUCCEEDED / CANCELED / ABORTED.
        # Action status topics use RELIABLE + TRANSIENT_LOCAL durability so
        # late-joining subscribers still see the most recent goal's terminal
        # state.
        if HAS_GOAL_STATUS_MSG and self.nav2_status_close_enabled:
            nav2_status_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
            )
            self.create_subscription(
                GoalStatusArray,  # type: ignore[arg-type]
                str(gp("nav2_action_status_topic").value),
                self._on_nav2_status, nav2_status_qos,
            )
        elif not HAS_GOAL_STATUS_MSG:
            self._warn_once(
                "action_msgs",
                "action_msgs.msg.GoalStatusArray not importable; "
                "falling back to goal_radius_m tolerance close only.",
            )
        else:
            self._warn_once(
                "vf_robot_messages",
                "vf_robot_messages.msg.MppiCriticsStats not importable; "
                "critic_costs will be NaN.",
            )
        try:
            from visualization_msgs.msg import MarkerArray  # noqa: WPS433

            self.create_subscription(
                MarkerArray,
                str(gp("dynamic_obstacles_topic").value),
                self._on_obstacles_marker, 10,
            )
        except Exception:
            self._warn_once(
                "dyn_obstacles_msg",
                "MarkerArray not available; dynamic-obstacles GT "
                "subscription disabled.",
            )

        # Status publisher ───────────────────────────────────────────────
        self._status_pub = self.create_publisher(
            Float32MultiArray, str(gp("status_topic").value), 5,
        )

        # Timers ─────────────────────────────────────────────────────────
        self._tick_timer = self.create_timer(self.write_period_s, self._tick)
        self._flush_timer = self.create_timer(self.flush_period_s, self._flush)
        # Debounce / cooldown clock runs faster than flush so the open is
        # snappy once the goal is stable.
        self._goal_clock = self.create_timer(0.1, self._poll_pending_goal)

        path_info = (
            self.session_dir if not self._use_vfdata_paths
            else f"vf_data/vf_data_training/{self.session_kind}/{self.map_name}/<goal>/{self.planner}/{self.controller}/"
        )
        self.get_logger().info(
            "data_collector_node ready: path_mode=%s map=%s "
            "planner=%s controller=%s channels=%s critics=%d"
            % (
                "vfdata" if self._use_vfdata_paths else "legacy",
                self.map_name,
                self.planner,
                self.controller,
                self.channels_config,
                len(self.critic_names),
            )
        )
        self.get_logger().info("Output root: %s" % path_info)

    # ------------------------------------------------------------- helpers
    def _warn_once(self, key: str, msg: str) -> None:
        if not self._warned.get(key):
            self.get_logger().warn(msg)
            self._warned[key] = True

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # --------------------------------------------------------- callbacks
    def _on_features(self, msg: Float32MultiArray) -> None:
        if not msg.data:
            return
        arr = np.asarray(msg.data, dtype=np.float32)
        self._features = arr
        if self._features_dim == 0:
            self._features_dim = int(arr.size)
            if not self.channel_dims:
                self.channel_dims = [self._features_dim]
                if not self.channel_names:
                    self.channel_names = ["features_flat"]

    def _on_critic_costs(self, msg) -> None:
        names = list(msg.critics)
        costs = np.asarray(msg.costs_sum, dtype=np.float32)
        self._critic_costs = costs
        if self._critic_count == 0 and costs.size > 0:
            self._critic_count = int(costs.size)
        if not self.critic_names and names:
            self.critic_names = names

    def _on_nav2_status(self, msg) -> None:
        """Schedule episode close when Nav2 reports a terminal goal status.

        Nav2 publishes the action's status_list as goals transition through
        ACCEPTED → EXECUTING → SUCCEEDED/CANCELED/ABORTED. We pick the most
        recent terminal entry and arm a delayed close (sim-time-based) so
        the robot's odom can settle into its final pose before the HDF5
        episode is finalised. The close itself runs from `_tick`.
        """
        if self.batch_mode:
            return  # batch_mode: tour_runner publishes the close explicitly
        if self._writer is None:
            return  # nothing to close
        if not self.nav2_status_close_enabled:
            return
        latest_terminal = None
        latest_stamp = -1.0
        for s in msg.status_list:
            if s.status not in _NAV2_STATUS_TERMINAL:
                continue
            stamp = (
                s.goal_info.stamp.sec
                + s.goal_info.stamp.nanosec * 1e-9
            )
            if stamp >= latest_stamp:
                latest_stamp = stamp
                latest_terminal = s
        if latest_terminal is None:
            return

        goal_id = bytes(latest_terminal.goal_info.goal_id.uuid)
        if goal_id == self._last_nav2_terminal_goal_id:
            return  # already armed for this goal
        self._last_nav2_terminal_goal_id = goal_id

        self._pending_nav2_close_at = (
            self._now() + max(0.0, self.nav2_close_settle_s)
        )
        self._pending_nav2_close_success = (
            latest_terminal.status == _NAV2_STATUS_SUCCEEDED
        )
        self._pending_nav2_close_reason = _NAV2_STATUS_REASONS.get(
            latest_terminal.status, "nav2_terminal"
        )
        self.get_logger().info(
            "Nav2 status=%d (%s) — closing episode in %.1fs after odom settle"
            % (
                latest_terminal.status,
                self._pending_nav2_close_reason,
                self.nav2_close_settle_s,
            )
        )

    def _clear_pending_nav2_close(self) -> None:
        """Reset the armed Nav2-close state. Call on episode close/open."""
        self._pending_nav2_close_at = None
        self._pending_nav2_close_reason = ""
        self._pending_nav2_close_success = False

    def _on_weights(self, msg: Float32MultiArray) -> None:
        self._weights = np.asarray(msg.data, dtype=np.float32)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._cmd_vel = np.array(
            [msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32
        )

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self._odom_pose = np.array(
            [p.position.x, p.position.y, _yaw_from_quaternion(p.orientation)],
            dtype=np.float32,
        )

    def _on_goal(self, msg: PoseStamped) -> None:
        if self.batch_mode:
            return  # batch_mode: episode lifecycle driven by batch_control_topic
        if not self._recording_enabled:
            return  # gate closed (tour_runner is repositioning)
        goal = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                _yaw_from_quaternion(msg.pose.orientation),
            ],
            dtype=np.float32,
        )
        # /goal_pose is explicit. Push through debounce too — the planner
        # will still follow up with /plan, and we want a single open
        # decision. The debounce is short enough not to delay the start
        # appreciably.
        self._propose_goal(goal, source="goal_pose")

    def _on_plan(self, msg: Path) -> None:
        # Cache the plan for tier-6 XTE BEFORE the lifecycle gates — we want
        # the plan history under both batch_mode (tour_runner) and
        # interactive RViz mode whenever a writer is open.
        self._append_plan_to_history(msg)

        if self.batch_mode:
            return  # batch_mode: episode lifecycle driven by batch_control_topic
        if not self._recording_enabled:
            return  # gate closed
        if not msg.poses:
            return
        final = msg.poses[-1].pose
        goal = np.array(
            [
                final.position.x,
                final.position.y,
                _yaw_from_quaternion(final.orientation),
            ],
            dtype=np.float32,
        )
        self._propose_goal(goal, source="plan")

    def _append_plan_to_history(self, msg: Path) -> None:
        """Append one /plan to the in-memory history for tier-6 XTE.

        No-op if no writer is open (we only record plans that belong to
        the currently-open episode). Time is taken from msg.header.stamp
        (sim time with use_sim_time=true), to match the robot_pose
        sim_time used by the metric.
        """
        if self._writer is None or not msg.poses:
            return
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        poses = np.empty((len(msg.poses), 3), dtype=np.float64)
        for i, p in enumerate(msg.poses):
            poses[i, 0] = p.pose.position.x
            poses[i, 1] = p.pose.position.y
            poses[i, 2] = _yaw_from_quaternion(p.pose.orientation)
        self._plan_history.append((t, poses))

    def _on_recording_enabled(self, msg: Bool) -> None:
        prev = self._recording_enabled
        self._recording_enabled = bool(msg.data)
        if prev == self._recording_enabled:
            return
        if self._recording_enabled:
            self.get_logger().info("recording ENABLED — accepting new goals")
        else:
            self.get_logger().info(
                "recording DISABLED — incoming goals will be ignored "
                "(reposition / tour-pause)"
            )
            # If a writer is open when the gate closes, leave it alone —
            # the current episode finishes via Nav2 status / tolerance /
            # timeout; we just stop opening NEW episodes.

    def _on_batch_control(self, msg: Float32MultiArray) -> None:
        """Explicit episode lifecycle for batch_mode (driven by tour_runner).

        Format: [cmd, gx, gy, gyaw, success]
          cmd >= 0.5 → OPEN: snapshot current odom as start, set the
                       configured goal coords (used for folder name and
                       per-cycle goal field), open a fresh writer.
          cmd <  0.5 → CLOSE: overwrite _episode_goal_pose with the
                       robot's current odom (the stabilized stop pose),
                       then close the writer with success in {False, True}
                       and reason="batch_succeeded" / "batch_failed".

        Outside batch_mode this is a no-op; manual sessions and the legacy
        debouncer-driven path are unaffected.
        """
        if not self.batch_mode:
            return
        if len(msg.data) < 5:
            self.get_logger().warn(
                "batch_control message too short (need 5 floats, got %d) — "
                "ignored" % len(msg.data)
            )
            return
        cmd = float(msg.data[0])
        if cmd >= 0.5:
            gx = float(msg.data[1])
            gy = float(msg.data[2])
            gyaw = float(msg.data[3])
            if self._writer is not None:
                # Defensive: a stale episode is open. Close it before
                # opening a fresh one so we never overlap.
                self.get_logger().warn(
                    "batch_control OPEN received with writer already "
                    "open — force-closing previous episode."
                )
                self._close_episode(success=False, reason="batch_force_close")
            self._goal_pose = np.array([gx, gy, gyaw], dtype=np.float32)
            self.get_logger().info(
                "[batch_control] OPEN configured_goal=(%.3f, %.3f, %.3f)"
                % (gx, gy, gyaw)
            )
            self._open_episode()
            return

        # CLOSE
        if self._writer is None:
            self.get_logger().info(
                "[batch_control] CLOSE received with no writer open — ignored."
            )
            return
        success = float(msg.data[4]) >= 0.5
        # Overwrite the episode's goal pose with the actual stabilized stop
        # pose (per spec: the configured CSV goal may not be reached when the
        # action aborts/cancels, so the manifest records where the robot
        # actually ended up).
        if self._odom_pose is not None:
            self._episode_goal_pose = self._odom_pose.copy()
        reason = "batch_succeeded" if success else "batch_failed"
        self.get_logger().info(
            "[batch_control] CLOSE success=%s reason=%s" % (success, reason)
        )
        self._close_episode(success=success, reason=reason)

    def _on_obstacles_marker(self, msg) -> None:
        rows = []
        for m in msg.markers:
            rows.append(
                [
                    float(m.id),
                    float(m.pose.position.x),
                    float(m.pose.position.y),
                    float("nan"),
                    float("nan"),
                ]
            )
        self._dyn_obstacles = np.asarray(rows, dtype=np.float32) if rows else None

    # --------------------------------------------------- goal debounce
    def _propose_goal(self, goal: np.ndarray, *, source: str) -> None:
        now = self._now()
        decision = self._debouncer.propose(goal, now=now)
        if decision is Decision.REFRESH:
            # Replan of the active goal — refresh the latest snapshot.
            self._goal_pose = goal
            return
        if decision is Decision.IGNORE:
            return
        if decision is Decision.PENDING:
            return
        # Decision.COMMIT
        self._commit_goal(goal, source=source)

    def _poll_pending_goal(self) -> None:
        """Periodically check whether the pending goal has matured.

        Needed because /plan and /goal_pose may not fire faster than
        goal_debounce_s on their own — we still need to open eventually.
        """
        now = self._now()
        if self._debouncer.poll(now=now) is not Decision.COMMIT:
            return
        pending = self._debouncer.pending_goal
        if pending is None:
            return
        self._commit_goal(pending, source="debounce_timer")

    def _commit_goal(self, goal: np.ndarray, *, source: str) -> None:
        """Open a new episode for this goal.

        Pre: writer must be None (we never overlap episodes). If a writer
        is open and a *different* goal lands here, close it first
        (controlled by ``close_on_new_goal``).
        """
        if self._writer is not None:
            if self.close_on_new_goal:
                # Race-condition fix: if the user clicks the NEXT goal
                # before the consec-in-radius counter (default 5 ticks
                # = 250 ms) catches up, naively closing as
                # success=False/new_goal mis-labels what was actually
                # a successful run. Treat the previous episode as
                # success=True if the robot is currently inside the
                # previous goal's radius — Nav2 has already declared
                # "reached" by the time the user clicks the next goal.
                prev_reached = (
                    self._odom_pose is not None
                    and self._goal_pose is not None
                    and float(np.linalg.norm(
                        self._odom_pose[:2] - self._goal_pose[:2]
                    )) < self.goal_radius_m
                )
                if prev_reached:
                    self.get_logger().info(
                        "New goal received (source=%s); previous goal was "
                        "within %.2fm — closing previous as goal_reached."
                        % (source, self.goal_radius_m)
                    )
                    self._close_episode(success=True, reason="goal_reached")
                else:
                    self.get_logger().info(
                        "New goal received (source=%s) — closing current "
                        "episode (robot not yet at previous goal)." % source
                    )
                    self._close_episode(success=False, reason="new_goal")
            else:
                return

        self._goal_pose = goal
        self._debouncer.mark_committed(goal)

        self.get_logger().info(
            "Episode opened (goal source: %s)" % source
        )
        self._open_episode()

    # --------------------------------------------------------- lifecycle
    def _open_episode(self) -> None:
        D = self._features_dim if self._features_dim > 0 else self.DEFAULT_FEATURE_DIM
        K = (
            self._critic_count
            if self._critic_count > 0
            else (len(self.critic_names) or self.DEFAULT_CRITIC_COUNT)
        )

        meta = EpisodeMetadata(
            scenario_id=self.scenario_id,
            seed=self.seed,
            controller_mode=self.controller_mode,
            weight_provider=self.weight_provider,
            channels_config=self.channels_config,
            channel_names=self.channel_names or ["features_flat"],
            channel_dims=self.channel_dims or [D],
            critic_names=self.critic_names or ["critic_%d" % i for i in range(K)],
        )

        self._next_episode_index += 1
        index = self._next_episode_index - 1

        if self._use_vfdata_paths:
            # New vf_data layout: compute leaf from current goal coordinates.
            from vf_robot_utils.io.vf_data_paths import goal_folder_name, run_ts
            if self._goal_pose is not None:
                gx, gy, gt = (float(v) for v in self._goal_pose)
                goal_folder = goal_folder_name(gx, gy, gt)
            else:
                goal_folder = "goal_unknown"
            leaf = vfdata_leaf_for(
                self.training_root,
                self.session_kind,
                self.map_name,
                goal_folder,
                self.planner,
                self.controller,
            )
            os.makedirs(str(leaf), exist_ok=True)
            self.session_dir = str(leaf)
            h5_name = run_ts() + ".h5"
            path = os.path.join(self.session_dir, h5_name)
            # session.json is per-leaf; write once per leaf (not per session).
            # Reset the flag when the leaf changes (new goal).
            leaf_marker = str(leaf)
            if getattr(self, "_current_leaf", None) != leaf_marker:
                self._current_leaf = leaf_marker
                self._session_json_written = False
        else:
            # Legacy mode: flat session folder, ep_NNN_scenario_ts.h5 naming.
            from vf_controller.data_collection.session import episode_filename
            h5_name = episode_filename(index, self.scenario_id)
            path = os.path.join(self.session_dir, h5_name)

        # Lazily write session.json on the first episode for this leaf.
        # Delay until here so channels_config and critic counts filled from
        # upstream messages are already correct.
        if not self._session_json_written:
            info = SessionInfo(
                session_kind=self.session_kind,
                map_name=self.map_name,
                started_at_iso=_isoformat_now(),
                controller_mode=self.controller_mode,
                weight_provider=self.weight_provider,
                channels_config=self.channels_config,
                scenario_id=self.scenario_id,
                seed=self.seed,
                episode_timeout_s=self.episode_timeout_s,
                write_period_s=self.write_period_s,
                goal_radius_m=self.goal_radius_m,
                git_commit=_git_commit(),
                extra={
                    "feature_dim": int(D),
                    "critic_count": int(K),
                    "planner": self.planner,
                    "controller": self.controller,
                },
            )
            try:
                write_session_json(self.session_dir, info)
                self._session_json_written = True
            except Exception as e:  # pragma: no cover
                self.get_logger().warn("write_session_json failed: %s" % e)

        self._writer = EpisodeWriter(
            path=path,
            feature_dim=D,
            critic_count=K,
            meta=meta,
            max_obstacles=self.max_obstacles,
        )
        self._writer_path = path
        self._writer_index = index
        self._episode_start_time = self._now()
        self._episode_start_iso = _isoformat_now()
        self._episode_start_pose = (
            self._odom_pose.copy() if self._odom_pose is not None else None
        )
        self._episode_goal_pose = (
            self._goal_pose.copy() if self._goal_pose is not None else None
        )
        self._last_pose_for_path = None
        self._path_length_m = 0.0
        self._consec_in_radius = 0
        self._step_index = 0
        # Reset /plan history for the new episode (tier-6 XTE reference).
        self._plan_history = []
        # New episode supersedes any armed Nav2 close from the previous one.
        self._clear_pending_nav2_close()
        self.get_logger().info("Episode opened: %s" % path)

    def _close_episode(self, *, success: bool, reason: str) -> None:
        if self._writer is None:
            self._clear_pending_nav2_close()
            return
        # If we got here from any path (tolerance, new_goal, timeout,
        # shutdown) clear any armed Nav2 close — it's already happening.
        self._clear_pending_nav2_close()
        end_t = self._now()
        dt = (
            (end_t - self._episode_start_time)
            if self._episode_start_time is not None
            else float("nan")
        )
        outcome = EpisodeOutcome(
            success=success,
            collision_count=0,  # TODO: subscribe to /collision_monitor
            time_to_goal_s=dt,
            path_length_m=self._path_length_m,
            mean_clearance_m=float("nan"),
            goal_reached_at_step=self._step_index if success else -1,
        )
        n_steps = self._step_index
        ended_iso = _isoformat_now()
        # Persist tier-6 XTE reference. The full history goes under
        # global_path_plans/; the last plan is also stored as
        # global_path_poses for backward-compat with code that only knows
        # about the single-path field.
        if self._plan_history:
            try:
                self._writer.write_global_path_plans(self._plan_history)
                final_poses = self._plan_history[-1][1]
                self._writer.write_global_path(final_poses.tolist())
            except Exception as e:  # pragma: no cover
                self.get_logger().warn(
                    "write_global_path_plans failed: %s" % e
                )
        try:
            self._writer.close(outcome=outcome)
        except Exception as e:  # pragma: no cover
            self.get_logger().error("EpisodeWriter close failed: %s" % e)

        # File size after close.
        try:
            size_bytes = os.path.getsize(self._writer_path)
        except OSError:
            size_bytes = 0

        # Append to session manifest.
        sx = sy = syaw = float("nan")
        gx = gy = gyaw = float("nan")
        if self._episode_start_pose is not None:
            sx, sy, syaw = (float(v) for v in self._episode_start_pose)
        if self._episode_goal_pose is not None:
            gx, gy, gyaw = (float(v) for v in self._episode_goal_pose)

        try:
            append_manifest_row(self.session_dir, {
                "episode_index": self._writer_index,
                "h5_filename": os.path.basename(self._writer_path),
                "scenario_id": self.scenario_id,
                "seed": self.seed,
                "controller_mode": self.controller_mode,
                "channels_config": self.channels_config,
                "start_x": sx, "start_y": sy, "start_yaw": syaw,
                "goal_x": gx, "goal_y": gy, "goal_yaw": gyaw,
                "success": bool(success),
                "close_reason": reason,
                "n_steps": int(n_steps),
                "duration_s": float(dt) if not math.isnan(dt) else "",
                "path_length_m": float(self._path_length_m),
                "size_bytes": int(size_bytes),
                "started_at_iso": self._episode_start_iso,
                "ended_at_iso": ended_iso,
            })
        except Exception as e:  # pragma: no cover
            self.get_logger().warn("manifest append failed: %s" % e)

        # Update aggregate session stats.
        self._session_episodes_closed += 1
        self._session_total_steps += int(n_steps)
        self._session_total_bytes += int(size_bytes)

        self.get_logger().info(
            "Episode closed (reason=%s, success=%s, steps=%d, dt=%.2fs, "
            "size=%d B): %s"
            % (reason, success, n_steps, dt, size_bytes, self._writer_path)
        )

        # Hand the just-closed goal off to the debouncer so the next
        # /plan from the same target is filtered for goal_cooldown_s.
        self._debouncer.mark_closed(
            self._episode_goal_pose, now=end_t,
        )

        self._writer = None
        self._writer_path = ""
        self._writer_index = -1
        self._episode_start_time = None
        self._episode_start_pose = None
        self._episode_goal_pose = None

    # ----------------------------------------------------------- tick
    def _tick(self) -> None:
        # Fire any pending Nav2-status-driven close BEFORE the early returns,
        # so a settled close still runs even if cmd_vel / features dropped
        # out at the very end of the episode.
        if (
            self._writer is not None
            and self._pending_nav2_close_at is not None
            and self._now() >= self._pending_nav2_close_at
        ):
            reason = self._pending_nav2_close_reason or "nav2_terminal"
            success = self._pending_nav2_close_success
            self._clear_pending_nav2_close()
            self._close_episode(success=success, reason=reason)
            return  # writer is gone; nothing more to do this tick

        if self._writer is None:
            return
        # Baseline mode (stock Nav2 controllers: mppi/dwb/rpp/graceful/fixedwt)
        # does not publish /vf/features, /vf/per_critic_costs, or
        # /vf/applied_weights. We still write per-step rows so the Safety /
        # Efficiency / Motion-quality / Path-adherence metric columns
        # (t1_/t2_/t3_/t4_/t6_) can be computed from /odom + /cmd_vel; the
        # VF-only Adaptation columns (t5_) are NaN-padded.
        is_baseline = (self.controller_mode == "baseline")
        if not is_baseline and self._features is None:
            return
        if self._cmd_vel is None or self._odom_pose is None or self._goal_pose is None:
            return  # wait for the first batch of all required signals

        K = self._writer.critic_count
        D = self._writer.feature_dim

        features = (
            self._features
            if self._features is not None
            else np.full((D,), np.nan, dtype=np.float32)
        )

        critic_costs = (
            self._critic_costs
            if self._critic_costs is not None
            else np.full((K,), np.nan, dtype=np.float32)
        )
        if self._critic_costs is None and not is_baseline:
            self._warn_once(
                "no_critic_costs",
                "/vf/per_critic_costs not received; writing NaN. Did you "
                "set controller_mode=collect?",
            )

        weights = (
            self._weights
            if self._weights is not None
            else np.full((K,), np.nan, dtype=np.float32)
        )
        if self._weights is None and not is_baseline:
            self._warn_once(
                "no_weights",
                "/vf/applied_weights not published yet; writing NaN.",
            )

        row = CycleRow(
            features=features,
            critic_costs=critic_costs,
            critic_weights=weights,
            selected_action=self._cmd_vel,
            robot_pose=self._odom_pose,
            goal=self._goal_pose,
            dynamic_obstacles=self._dyn_obstacles,
            sim_time=self._now(),
        )
        self._writer.append(row)

        # Path length integral.
        if self._last_pose_for_path is not None:
            d = float(
                np.linalg.norm(self._odom_pose[:2] - self._last_pose_for_path[:2])
            )
            self._path_length_m += d
        self._last_pose_for_path = self._odom_pose.copy()

        self._step_index += 1

        # Goal-reached detection.
        # Disabled in batch_mode: tour_runner closes the episode explicitly
        # after the NavigateToPose action terminates and the robot settles,
        # so this tolerance path would race against the batch_control close.
        if not self.batch_mode:
            d_goal = float(
                np.linalg.norm(self._odom_pose[:2] - self._goal_pose[:2])
            )
            if d_goal < self.goal_radius_m:
                self._consec_in_radius += 1
            else:
                self._consec_in_radius = 0

            if self._consec_in_radius >= self.goal_reached_consecutive:
                self._close_episode(success=True, reason="goal_reached")
                return

        # Timeout (kept in batch_mode as a safety net in case tour_runner
        # crashes before publishing close).
        if self._episode_start_time is not None:
            if self._now() - self._episode_start_time > self.episode_timeout_s:
                self._close_episode(success=False, reason="timeout")

    def _flush(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
            except Exception as e:  # pragma: no cover
                self.get_logger().warn("flush failed: %s" % e)
        self._publish_status()

    def _publish_status(self) -> None:
        msg = Float32MultiArray()
        cur_steps = float(self._step_index) if self._writer is not None else 0.0
        cur_bytes = (
            float(self._writer.current_size_bytes())
            if self._writer is not None else 0.0
        )
        msg.data = [
            float(self._session_episodes_closed),
            float(self._session_total_steps),
            float(self._session_total_bytes),
            cur_steps,
            cur_bytes,
        ]
        try:
            self._status_pub.publish(msg)
        except Exception:
            pass

    # ---------------------------------------------------------- shutdown
    def destroy_node(self) -> bool:
        try:
            if self._writer is not None:
                self._close_episode(success=False, reason="shutdown")
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
