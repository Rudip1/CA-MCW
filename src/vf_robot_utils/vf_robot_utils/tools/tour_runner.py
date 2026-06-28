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
tour_runner.py — Drive one row of a goalposes_collect CSV via NavigateToPose.

Used by vf_data_training_batch_fixedwt.launch.py to replay a tour previously
recorded with pose_recorder.

CSV format (wide, written by pose_recorder):
  run_id, notes, start_x, start_y, start_yaw,
  g1_x, g1_y, g1_yaw, g2_x, g2_y, g2_yaw, ...

The number of goal triples per row is data-driven — the row is consumed
until the next g{i}_x/g{i}_y triple is empty/missing. So one row may
contain 3 goals and another 17.

Behaviour (batch_mode protocol with data_collector_node)
--------------------------------------------------------
tour_runner is the AUTHORITY on episode lifecycle in batch mode. It
publishes a Float32MultiArray on /vf/batch/episode_control whose data
field is [cmd, gx, gy, gyaw, success] where cmd=1.0 opens an episode
with the given configured goal coordinates and cmd=0.0 closes the open
episode with success in {0.0, 1.0}.

For each goal in the row:
  1. Sample is implicit — the collector snapshots /odom at OPEN.
  2. Publish OPEN with the configured (gx, gy, gyaw) → collector creates
     a fresh HDF5 in <map>/goal_x.._y.._t../<Planner>/<controller>/.
  3. Send NavigateToPose for the same (gx, gy, gyaw).
  4. Wait for the action's terminal status (SUCCEEDED/CANCELED/ABORTED)
     or per_goal_timeout_s (then we cancel and treat as failure).
  5. Sleep settle_s so /odom decelerates into the actual stop pose.
  6. Publish CLOSE with success=(status == "succeeded"). The collector
     overwrites the manifest's goal_x/y/yaw with the stabilised /odom
     pose at close time and finalises the HDF5.
  7. Sleep inter_leg_pause_s — buffer for the writer to flush.

For the optional reposition leg (start pose) no episode is opened, so
the spawn→start drive lands no HDF5 file regardless of the action result.

Goal cycling policy: one pass through the row, then exit. (No looping —
the user re-runs with a different run_id to drive a different tour.)
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from typing import List, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.qos import DurabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray

try:
    from nav2_msgs.srv import ClearEntireCostmap

    HAS_CLEAR_COSTMAP = True
except ImportError:  # pragma: no cover
    ClearEntireCostmap = None  # type: ignore
    HAS_CLEAR_COSTMAP = False


GoalT = Tuple[float, float, float]


def _yaw_to_quat(yaw: float) -> Tuple[float, float]:
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _read_row(csv_path: str, run_id: int) -> Tuple[Optional[GoalT], List[GoalT]]:
    """Find the row whose run_id column equals `run_id` and return
    (start, goals).

    The goal count is data-driven: we walk g1, g2, g3, ... until the next
    g{i}_x/g{i}_y triple is missing or empty. So the row may contain any
    number of goals.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rid = int(row.get("run_id", -1))
            except (ValueError, TypeError):
                continue
            if rid != run_id:
                continue

            start: Optional[GoalT] = None
            sx = (row.get("start_x") or "").strip()
            sy = (row.get("start_y") or "").strip()
            sy_yaw = (row.get("start_yaw") or "").strip()
            if sx and sy and sy_yaw:
                start = (float(sx), float(sy), float(sy_yaw))

            goals: List[GoalT] = []
            i = 1
            while True:
                gx = (row.get(f"g{i}_x") or "").strip()
                gy = (row.get(f"g{i}_y") or "").strip()
                gyaw = (row.get(f"g{i}_yaw") or "").strip()
                if not (gx and gy):
                    break
                goals.append((float(gx), float(gy), float(gyaw or 0.0)))
                i += 1
            return start, goals
    raise SystemExit(
        f"[tour_runner] run_id={run_id} not found in {csv_path}. "
        f"Available rows visible with: csvtool col 1 {csv_path}"
    )


class TourRunner(Node):
    def __init__(self, args):
        super().__init__("tour_runner")
        self.set_parameters(
            [rclpy.parameter.Parameter(
                "use_sim_time", rclpy.Parameter.Type.BOOL, True)]
        )
        self.args = args
        self._ac = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # Latched batch-mode episode control. data_collector_node listens
        # here when batch_mode=True. TRANSIENT_LOCAL so a late-joining
        # collector still receives the most recent command.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self._batch_pub = self.create_publisher(
            Float32MultiArray, "/vf/batch/episode_control", latched_qos,
        )

        # Legacy /vf/recording_enabled gate. Kept so a non-batch_mode
        # collector running alongside this node also stays silent during
        # the reposition leg. With batch_mode=True the collector ignores
        # this topic entirely.
        self._gate_pub = self.create_publisher(
            Bool, "/vf/recording_enabled", latched_qos,
        )

        # data_collector_node publishes /vf/collector_status every flush
        # tick (~1 Hz) once it has fully subscribed. We wait for the first
        # message before firing the reposition, otherwise the reposition
        # leg races against the collector's subscriptions coming up.
        self._collector_alive = False
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.create_subscription(
            Float32MultiArray, "/vf/collector_status",
            self._on_collector_status, qos_be,
        )

        # Nav2 readiness probes. Without these the very first goal may
        # fire before the action server, costmaps, or localization are
        # actually live, causing an instant rejection that leaves an
        # empty 0-cycle HDF5 stub.
        self._global_costmap_seen = False
        self._local_costmap_seen = False
        self._odom_seen = False
        self._latest_odom: Optional[Tuple[float, float, float]] = None

        # Costmap-clear service clients (used between reposition retries
        # to flush stale lethal cells left over from the previous run).
        # Some Nav2 builds may not expose these; if so we just skip.
        if HAS_CLEAR_COSTMAP:
            self._clear_global_cli = self.create_client(
                ClearEntireCostmap,
                "/global_costmap/clear_entirely_global_costmap",
            )
            self._clear_local_cli = self.create_client(
                ClearEntireCostmap,
                "/local_costmap/clear_entirely_local_costmap",
            )
        else:
            self._clear_global_cli = None
            self._clear_local_cli = None
        cm_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.create_subscription(
            OccupancyGrid, "/global_costmap/costmap",
            self._on_global_costmap, cm_qos,
        )
        self.create_subscription(
            OccupancyGrid, "/local_costmap/costmap",
            self._on_local_costmap, cm_qos,
        )
        self.create_subscription(
            Odometry, "/odom", self._on_odom_seen, qos_be,
        )

    def _on_collector_status(self, _msg) -> None:
        self._collector_alive = True

    def _on_global_costmap(self, _msg) -> None:
        self._global_costmap_seen = True

    def _on_local_costmap(self, _msg) -> None:
        self._local_costmap_seen = True

    def _on_odom_seen(self, msg) -> None:
        self._odom_seen = True
        p = msg.pose.pose
        # yaw from quaternion
        siny_cosp = 2.0 * (p.orientation.w * p.orientation.z
                           + p.orientation.x * p.orientation.y)
        cosy_cosp = 1.0 - 2.0 * (p.orientation.y * p.orientation.y
                                 + p.orientation.z * p.orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self._latest_odom = (
            float(p.position.x), float(p.position.y), float(yaw),
        )

    # -------------------------------------------------- gate / batch ctrl
    def set_recording(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = bool(enabled)
        self._gate_pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().info(
            f"/vf/recording_enabled := {bool(enabled)}"
        )

    def open_batch_episode(self, gx: float, gy: float, gyaw: float) -> None:
        """Tell the collector to open a fresh episode with these
        configured goal coords (folder name is derived from them)."""
        msg = Float32MultiArray()
        msg.data = [1.0, float(gx), float(gy), float(gyaw), 0.0]
        self._batch_pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().info(
            f"[batch_control] OPEN goal=({gx:.3f}, {gy:.3f}, {gyaw:.3f})"
        )

    def close_batch_episode(self, success: bool) -> None:
        """Tell the collector to close the open episode. The collector
        snapshots /odom at this instant and writes it as the manifest's
        goal_x/y/yaw (stabilised stop pose)."""
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0, 1.0 if success else 0.0]
        self._batch_pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().info(
            f"[batch_control] CLOSE success={success}"
        )

    def wait_for_collector(self, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not self._collector_alive:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() > deadline:
                self.get_logger().warn(
                    f"data_collector_node did not publish "
                    f"/vf/collector_status within {timeout_s:.0f}s — "
                    f"continuing anyway (first leg may not be recorded)."
                )
                return False
        return True

    def wait_for_nav2(self, timeout_s: float = 120.0) -> bool:
        """Block until the Nav2 stack is actually ready to accept goals.

        Readiness = navigate_to_pose action server up + at least one
        global_costmap update + at least one /odom message. The local
        costmap is a soft signal (logged if missing) but not required —
        some launches publish it lazily.

        Returns False on timeout (caller may proceed anyway with a warn).
        """
        if not self._ac.wait_for_server(timeout_sec=timeout_s):
            self.get_logger().error(
                f"navigate_to_pose action server not available within "
                f"{timeout_s:.0f}s"
            )
            return False
        deadline = time.monotonic() + timeout_s
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._global_costmap_seen and self._odom_seen:
                if not self._local_costmap_seen:
                    self.get_logger().warn(
                        "Proceeding without /local_costmap/costmap — "
                        "first goal may need an extra second to plan."
                    )
                self.get_logger().info(
                    "Nav2 ready: action_server=YES, global_costmap=YES, "
                    f"local_costmap={'YES' if self._local_costmap_seen else 'NO'}, "
                    "odom=YES"
                )
                return True
            if time.monotonic() > deadline:
                self.get_logger().warn(
                    f"Nav2 readiness probes timed out after {timeout_s:.0f}s "
                    f"(global_costmap={self._global_costmap_seen}, "
                    f"odom={self._odom_seen}) — continuing anyway."
                )
                return False
        return False

    # ----------------------------------------------------------- nav
    def _build_goal(self, x: float, y: float, yaw: float):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        qz, qw = _yaw_to_quat(yaw)
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw
        return goal

    def send_goal(self, x: float, y: float, yaw: float, label: str):
        """Send a NavigateToPose goal and block until the action server
        accepts (or rejects) it. Returns the goal handle on accept,
        otherwise (None, reason_str). The episode SHOULD only be opened
        after a successful accept — that way a rejection leaves no
        empty HDF5 stub on disk.
        """
        if not self._ac.wait_for_server(timeout_sec=60.0):
            self.get_logger().error(
                "navigate_to_pose action server not available within 60s"
            )
            return None, "no_server"

        goal = self._build_goal(x, y, yaw)
        self.get_logger().info(
            f"[{label}] Sending goal x={x:.2f} y={y:.2f} yaw={yaw:.2f}"
        )
        send_fut = self._ac.send_goal_async(goal)
        deadline = time.monotonic() + self.args.per_goal_timeout_s
        while rclpy.ok() and not send_fut.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() > deadline:
                return None, "send_timeout"

        handle = send_fut.result()
        if handle is None or not handle.accepted:
            return None, "rejected"
        return handle, "accepted"

    def wait_for_result(self, handle, label: str) -> str:
        """Block until the action terminates, returning a status string."""
        result_fut = handle.get_result_async()
        deadline = time.monotonic() + self.args.per_goal_timeout_s
        while rclpy.ok() and not result_fut.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() > deadline:
                handle.cancel_goal_async()
                for _ in range(10):
                    rclpy.spin_once(self, timeout_sec=0.1)
                return "timeout"

        status = result_fut.result().status
        return {4: "succeeded", 5: "canceled", 6: "aborted"}.get(
            status, f"status_{status}"
        )

    # ------------------------------------------- costmap clear / reposition
    def _clear_costmaps(self) -> None:
        """Clear global + local costmaps via Nav2 services. No-op if the
        services aren't available — prints a warning."""
        if not HAS_CLEAR_COSTMAP:
            self.get_logger().warn(
                "nav2_msgs/srv/ClearEntireCostmap not importable — "
                "skipping costmap clear."
            )
            return
        for cli, name in (
            (self._clear_global_cli, "global"),
            (self._clear_local_cli, "local"),
        ):
            if cli is None:
                continue
            if not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn(
                    f"clear_{name}_costmap service not up — skipping"
                )
                continue
            future = cli.call_async(ClearEntireCostmap.Request())
            deadline = time.monotonic() + 3.0
            while rclpy.ok() and not future.done():
                rclpy.spin_once(self, timeout_sec=0.1)
                if time.monotonic() > deadline:
                    break
            if future.done():
                self.get_logger().info(f"cleared {name} costmap")
            else:
                self.get_logger().warn(
                    f"clear_{name}_costmap call timed out"
                )

    @staticmethod
    def _xy_yaw_close(
        target: Tuple[float, float, float],
        current: Tuple[float, float, float],
        xy_tol: float,
        yaw_tol: float,
    ) -> Tuple[bool, float, float]:
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        dist = math.hypot(dx, dy)
        dyaw = abs(
            (target[2] - current[2] + math.pi) % (2.0 * math.pi) - math.pi
        )
        return (dist <= xy_tol and dyaw <= yaw_tol), dist, dyaw

    def reposition_to(
        self,
        sx: float, sy: float, syaw: float,
        xy_tol: float, yaw_tol: float,
        max_attempts: int,
        verify_settle_s: float = 2.0,
        retry_pause_s: float = 2.0,
    ) -> bool:
        """Drive to (sx, sy, syaw) with retry + pose verification.

        Each attempt: send NavigateToPose → wait for terminal → settle →
        check /odom is within (xy_tol, yaw_tol) of the target. On
        failure we clear costmaps, pause, and retry. Returns True iff
        the robot is verified at the target by some attempt.
        """
        target = (float(sx), float(sy), float(syaw))
        for attempt in range(1, max_attempts + 1):
            label = f"reposition[{attempt}/{max_attempts}]"
            handle, accept = self.send_goal(sx, sy, syaw, label)
            if handle is None:
                self.get_logger().warn(f"[{label}] action {accept}")
            else:
                status = self.wait_for_result(handle, label)
                self.get_logger().info(f"[{label}] action -> {status}")
            # Let /odom decelerate and the new pose latch.
            self.sleep_spinning(verify_settle_s)
            cur = self._latest_odom
            if cur is None:
                self.get_logger().warn(
                    f"[{label}] no /odom yet — cannot verify"
                )
            else:
                ok, dist, dyaw = self._xy_yaw_close(
                    target, cur, xy_tol, yaw_tol,
                )
                if ok:
                    self.get_logger().info(
                        f"[{label}] VERIFIED at "
                        f"({cur[0]:.2f}, {cur[1]:.2f}, {cur[2]:.2f}) — "
                        f"dist={dist:.2f}m, dyaw={dyaw:.2f}rad"
                    )
                    return True
                self.get_logger().warn(
                    f"[{label}] not at start: "
                    f"current=({cur[0]:.2f}, {cur[1]:.2f}, {cur[2]:.2f}) "
                    f"target=({target[0]:.2f}, {target[1]:.2f}, "
                    f"{target[2]:.2f}) "
                    f"dist={dist:.2f}m (tol={xy_tol}m), "
                    f"dyaw={dyaw:.2f}rad (tol={yaw_tol}rad)"
                )
            if attempt < max_attempts:
                self.get_logger().info(
                    f"[{label}] clearing costmaps and retrying..."
                )
                self._clear_costmaps()
                self.sleep_spinning(retry_pause_s)
        return False

    def sleep_spinning(self, seconds: float) -> None:
        """time.sleep equivalent that keeps spinning ROS callbacks so
        latched publishers actually deliver and clocks tick."""
        if seconds <= 0.0:
            return
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def main():
    ap = argparse.ArgumentParser(
        description="Replay a row of a goalposes_collect CSV via "
                    "NavigateToPose with explicit batch episode control."
    )
    ap.add_argument("--csv", required=True,
                    help="Path to <map>/training_goalposes_collect.csv (or "
                         "<map>/evaluation_goalposes_collect.csv).")
    ap.add_argument("--run-id", type=int, required=True,
                    help="run_id of the row to replay.")
    ap.add_argument("--settle-s", type=float, default=3.0,
                    help="Sleep AFTER NavigateToPose terminal status, BEFORE "
                         "publishing the batch_control CLOSE. Lets /odom "
                         "settle into the final stop pose so the manifest's "
                         "goal_x/y/yaw is the actual stabilised pose. "
                         "Default 3.0s.")
    ap.add_argument("--inter-leg-pause-s", type=float, default=1.0,
                    help="Sleep AFTER batch_control CLOSE, BEFORE the next "
                         "OPEN. Buffer so the collector can flush the HDF5 "
                         "before the next episode opens. Default 1.0s.")
    ap.add_argument("--post-reposition-stabilize-s", type=float, default=5.0,
                    help="Pause after the reposition leg, before the first "
                         "goal is sent. Longer than settle_s so localization "
                         "and the controller settle before recording starts.")
    ap.add_argument("--per-goal-timeout-s", type=float, default=180.0,
                    help="Hard timeout per NavigateToPose goal (s). Treated "
                         "as failure (action canceled) on expiry.")
    ap.add_argument("--nav2-ready-timeout-s", type=float, default=120.0,
                    help="Max wait (s) for Nav2 to be ready before the tour "
                         "starts. Readiness = action server up + first "
                         "global_costmap update + first /odom message. "
                         "Default 120.0s.")
    ap.add_argument("--reposition-first", type=_str2bool, default=True,
                    help="Drive to row's start pose before sending goals.")
    ap.add_argument("--reposition-xy-tol-m", type=float, default=0.5,
                    help="XY tolerance (m) used to verify reposition "
                         "actually reached (start_x, start_y). Default 0.5.")
    ap.add_argument("--reposition-yaw-tol-rad", type=float, default=0.5,
                    help="Yaw tolerance (rad) used to verify reposition. "
                         "Default 0.5 (~28 degrees).")
    ap.add_argument("--reposition-max-attempts", type=int, default=3,
                    help="Max NavigateToPose attempts to reach the start "
                         "pose. Each retry clears costmaps first. Default 3.")

    argv = sys.argv[1:]
    if "--ros-args" in argv:
        argv = argv[: argv.index("--ros-args")]
    args = ap.parse_args(argv)

    start, goals = _read_row(args.csv, args.run_id)
    if not goals:
        print(
            f"[tour_runner] run_id={args.run_id} has no goals in {args.csv}; "
            f"nothing to do.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(
        f"[tour_runner] run_id={args.run_id}  start={start}  "
        f"n_goals={len(goals)}",
        flush=True,
    )

    rclpy.init()
    node = TourRunner(args)

    # Give Nav2 + sim clock a moment to come up.
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    # Wait until data_collector_node is actually subscribed before firing
    # the reposition. Without this, latched messages we publish below may
    # land before the collector's subscriptions are ready.
    if not node.wait_for_collector(timeout_s=30.0):
        node.get_logger().warn(
            "Proceeding without confirmed collector — first leg may be lost."
        )

    # Belt-and-braces: also close the legacy gate. With batch_mode=True the
    # collector ignores this topic, but a stray non-batch_mode collector
    # would otherwise record the reposition leg via /plan.
    node.set_recording(False)

    # Pre-flight: don't send a goal until Nav2 is actually ready. Without
    # this, the very first action gets rejected (action server not up
    # yet, or costmaps empty) and the user sees a stub HDF5 with no
    # robot motion in RViz.
    nav2_ready = node.wait_for_nav2(timeout_s=args.nav2_ready_timeout_s)
    if not nav2_ready:
        node.get_logger().warn(
            "Continuing without confirmed Nav2 readiness — early goals "
            "may be rejected."
        )

    failures = 0
    skipped = 0
    try:
        # ── Reposition leg ───────────────────────────────────────────────
        if args.reposition_first and start is not None:
            sx, sy, syaw = start
            node.get_logger().info(
                f"Reposition leg (NOT recorded — no episode opened). "
                f"Target=({sx:.2f}, {sy:.2f}, {syaw:.2f})  "
                f"max_attempts={args.reposition_max_attempts}"
            )
            reached = node.reposition_to(
                sx, sy, syaw,
                xy_tol=args.reposition_xy_tol_m,
                yaw_tol=args.reposition_yaw_tol_rad,
                max_attempts=args.reposition_max_attempts,
            )
            if not reached:
                # Per user policy: log loud and proceed anyway. The data
                # collected from a wrong start is still useful for some
                # analyses; the manifest's start_x/y/yaw will record the
                # actual (non-nominal) leg-start pose so it's never
                # silently misattributed.
                node.get_logger().error(
                    "REPOSITION FAILED after %d attempts — proceeding "
                    "from CURRENT pose. The recorded start_x/y/yaw in "
                    "the manifest will be the actual robot pose, NOT "
                    "the configured CSV start." % args.reposition_max_attempts
                )
            node.get_logger().info(
                f"stabilizing {args.post_reposition_stabilize_s:.1f}s "
                f"before first goal..."
            )
            node.sleep_spinning(args.post_reposition_stabilize_s)

        # Legacy gate True is harmless under batch_mode=True; left for
        # compatibility with any non-batch_mode collector listening here.
        node.set_recording(True)

        # ── Per-goal data-collection loop ────────────────────────────────
        for i, (gx, gy, gyaw) in enumerate(goals, 1):
            label = f"g{i}/{len(goals)}"
            # 1. Send NavigateToPose and wait for ACCEPT (synchronous).
            handle, accept = node.send_goal(gx, gy, gyaw, label)
            if handle is None:
                # Action rejected before any motion. DO NOT open an
                # episode — that would leave a 0-cycle HDF5 stub. Skip
                # this leg entirely.
                node.get_logger().warn(
                    f"[{label}] action {accept} — skipping (no episode written)"
                )
                skipped += 1
                failures += 1
                continue
            # 2. Open episode now that Nav2 has accepted the goal. The
            #    collector snapshots /odom as the leg's start pose and
            #    creates a fresh run_*.h5.
            node.open_batch_episode(gx, gy, gyaw)
            # 3. Block until the action terminates.
            status = node.wait_for_result(handle, label)
            success = (status == "succeeded")
            node.get_logger().info(f"[{label}] -> {status}")
            if not success:
                failures += 1
            # 4. Settle so /odom decelerates into the final stop pose
            #    before the collector samples it.
            node.get_logger().info(
                f"[{label}] settling {args.settle_s:.1f}s before close"
            )
            node.sleep_spinning(args.settle_s)
            # 5. Close episode. Collector overwrites manifest goal_x/y/yaw
            #    with the stabilised /odom and finalises the HDF5.
            node.close_batch_episode(success)
            # 6. Inter-leg buffer so the writer can flush before the next
            #    OPEN arrives.
            node.sleep_spinning(args.inter_leg_pause_s)

        completed = len(goals) - failures
        node.get_logger().info(
            f"tour complete: {completed}/{len(goals)} succeeded "
            f"(skipped={skipped} due to action reject before motion)"
        )
    finally:
        # Defensive: if we crashed mid-leg, send a CLOSE so the collector
        # doesn't hold an open writer forever.
        try:
            node.close_batch_episode(False)
        except Exception:
            pass
        try:
            node.set_recording(False)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
