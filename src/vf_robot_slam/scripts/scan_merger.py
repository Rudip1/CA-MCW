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
LaserScan Merger Node — ViroFighter UVC-1
═════════════════════════════════════════

Merges /scan_d435i and /scan_d455 into a single /scan topic.
Replaces ira_laser_tools — zero external dependencies.

How it works:
    1. Subscribes to two LaserScan topics
    2. If input scans share the output frame AND scan geometry (angle_min,
       angle_max, angle_increment), merges directly by taking per-bin
       minimum — no coordinate conversion needed.
    3. If frames differ, transforms each scan's rays into the output frame
       using TF2 (vectorized numpy).
    4. Publishes combined LaserScan on /scan

Performance:
    Same-frame path: single numpy minimum() call, < 0.1 ms.
    Cross-frame path: vectorized numpy projection, < 1 ms.
    TF lookups are cached after first query (static URDF frames).

Usage:
    ros2 run vf_robot_slam scan_merger.py

Parameters:
    scan_topics:        space-separated input topics (default: "/scan_d435i /scan_d455")
    output_topic:       merged output topic (default: "/scan")
    output_frame:       frame_id for merged scan (default: "base_footprint")
    angle_min:          minimum angle in radians (default: -pi)
    angle_max:          maximum angle in radians (default: +pi)
    angle_increment:    angular resolution (default: 0.00581 ~ 0.33°)
    range_min:          minimum range in meters (default: 0.1)
    range_max:          maximum range in meters (default: 6.0)
"""

import math
import numpy as np
import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


class ScanMerger(Node):
    def __init__(self):
        super().__init__("scan_merger")

        # Declare parameters
        self.declare_parameter("scan_topics", "/scan_d435i /scan_d455")
        self.declare_parameter("output_topic", "/scan")
        self.declare_parameter("output_frame", "base_footprint")
        self.declare_parameter("angle_min", -math.pi)
        self.declare_parameter("angle_max", math.pi)
        self.declare_parameter("angle_increment", 0.00581)  # ~0.33°
        self.declare_parameter("range_min", 0.1)
        self.declare_parameter("range_max", 6.0)

        # Read parameters
        topics_str = self.get_parameter("scan_topics").value
        self.scan_topics = topics_str.split()
        self.output_topic = self.get_parameter("output_topic").value
        self.output_frame = self.get_parameter("output_frame").value
        self.angle_min = self.get_parameter("angle_min").value
        self.angle_max = self.get_parameter("angle_max").value
        self.angle_increment = self.get_parameter("angle_increment").value
        self.range_min = self.get_parameter("range_min").value
        self.range_max = self.get_parameter("range_max").value

        # Number of bins in merged scan
        self.num_bins = int((self.angle_max - self.angle_min) / self.angle_increment)

        # TF — only needed if input frames differ from output_frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # TF cache — keyed by frame_id → (tx, ty, yaw)
        self._tf_cache = {}

        # QoS — match LaserScan publishers (best effort)
        scan_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribe to each input scan
        self.latest_scans = {}
        for topic in self.scan_topics:
            self.latest_scans[topic] = None
            self.create_subscription(
                LaserScan,
                topic,
                lambda msg, t=topic: self._scan_callback(t, msg),
                scan_qos,
            )
            self.get_logger().info(f"Subscribed to {topic}")

        # Publisher
        self.pub = self.create_publisher(LaserScan, self.output_topic, scan_qos)

        # Merge timer — 30 Hz
        self.create_timer(0.033, self._merge_and_publish)

        self.get_logger().info(
            f"ScanMerger: {self.scan_topics} → {self.output_topic} "
            f"in {self.output_frame} ({self.num_bins} bins)"
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _scan_callback(self, topic: str, msg: LaserScan):
        self.latest_scans[topic] = msg

    # ── TF cache ─────────────────────────────────────────────────────────────

    def _get_tf_2d(self, frame_id: str):
        """
        Get cached 2D transform (tx, ty, yaw) from frame_id → output_frame.
        Returns None if TF is unavailable.
        """
        if frame_id in self._tf_cache:
            return self._tf_cache[frame_id]

        try:
            tf = self.tf_buffer.lookup_transform(
                self.output_frame,
                frame_id,
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except Exception:
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)

        result = (t.x, t.y, yaw)
        self._tf_cache[frame_id] = result

        self.get_logger().info(
            f"TF cached: {frame_id} → {self.output_frame} "
            f"(tx={t.x:.3f}, ty={t.y:.3f}, yaw={math.degrees(yaw):.1f}°)"
        )
        return result

    # ── Merge logic ──────────────────────────────────────────────────────────

    def _is_same_geometry(self, scan: LaserScan) -> bool:
        """Check if a scan has the same frame and angular layout as output."""
        return (
            scan.header.frame_id == self.output_frame
            and len(scan.ranges) == self.num_bins
            and abs(scan.angle_min - self.angle_min) < 1e-4
            and abs(scan.angle_increment - self.angle_increment) < 1e-5
        )

    def _merge_same_frame(self, scan: LaserScan, merged: np.ndarray):
        """
        Fast path: input scan already in output frame with same geometry.
        Just take per-bin minimum — no coordinate conversion needed.
        """
        ranges = np.array(scan.ranges, dtype=np.float64)

        # Invalidate out-of-range readings (set to inf so they lose)
        invalid = (
            (ranges < scan.range_min) | (ranges > scan.range_max) | ~np.isfinite(ranges)
        )
        ranges[invalid] = float("inf")

        np.minimum(merged, ranges, out=merged)

    def _merge_cross_frame(self, scan: LaserScan, merged: np.ndarray):
        """
        Slow path: input scan in a different frame or different geometry.
        Project rays through Cartesian coordinates into output frame.
        """
        tf_2d = self._get_tf_2d(scan.header.frame_id)
        if tf_2d is None:
            return

        tx, ty, yaw = tf_2d

        ranges = np.array(scan.ranges, dtype=np.float64)
        n = len(ranges)
        if n == 0:
            return

        ray_angles = scan.angle_min + np.arange(n) * scan.angle_increment

        valid = (
            (ranges >= scan.range_min)
            & (ranges <= scan.range_max)
            & np.isfinite(ranges)
        )
        ranges = ranges[valid]
        ray_angles = ray_angles[valid]

        if ranges.size == 0:
            return

        x_local = ranges * np.cos(ray_angles)
        y_local = ranges * np.sin(ray_angles)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x_out = tx + x_local * cos_yaw - y_local * sin_yaw
        y_out = ty + x_local * sin_yaw + y_local * cos_yaw

        range_out = np.sqrt(x_out * x_out + y_out * y_out)
        angle_out = np.arctan2(y_out, x_out)

        range_valid = (range_out >= self.range_min) & (range_out <= self.range_max)
        range_out = range_out[range_valid]
        angle_out = angle_out[range_valid]

        if range_out.size == 0:
            return

        bin_idx = ((angle_out - self.angle_min) / self.angle_increment).astype(np.int32)
        bin_valid = (bin_idx >= 0) & (bin_idx < self.num_bins)
        bin_idx = bin_idx[bin_valid]
        range_out = range_out[bin_valid]

        np.minimum.at(merged, bin_idx, range_out)

    def _merge_and_publish(self):
        # ── Collect available scans ──────────────────────────────────────────
        # Use whatever is latest from each topic — no staleness rejection.
        #
        # WHY NO STALENESS CHECK:
        # The old merger checked (now - scan.stamp) < 0.5s. With Gazebo sim
        # time, the merger's clock and the scan timestamps can drift due to
        # clock propagation jitter. Result: most scans got rejected, output
        # dropped to ~2 Hz instead of 30 Hz.
        #
        # This is safe because: each scan represents the latest sensor reading
        # from that camera. If a camera stops publishing, its last scan stays
        # in the merge — which is correct (the obstacles are still there).
        # When the camera publishes again, the callback replaces it.
        scans = [s for s in self.latest_scans.values() if s is not None]

        if not scans:
            return

        # Initialize merged ranges to inf (no reading)
        merged = np.full(self.num_bins, float("inf"))

        # Merge each input scan
        for scan in scans:
            if self._is_same_geometry(scan):
                self._merge_same_frame(scan, merged)
            else:
                self._merge_cross_frame(scan, merged)

        # ── Build output message ─────────────────────────────────────────────
        # Use the newest input timestamp so downstream nodes (Nav2, AMCL)
        # get the most accurate time reference.
        newest_stamp = max(rclpy.time.Time.from_msg(s.header.stamp) for s in scans)

        msg = LaserScan()
        msg.header.stamp = newest_stamp.to_msg()
        msg.header.frame_id = self.output_frame
        msg.angle_min = self.angle_min
        msg.angle_max = self.angle_max
        msg.angle_increment = self.angle_increment
        msg.time_increment = 0.0
        msg.scan_time = 0.033
        msg.range_min = self.range_min
        msg.range_max = self.range_max

        # inf = no reading (correct for Nav2/AMCL)
        # DO NOT replace inf with 0.0 — that means "obstacle at zero distance"
        msg.ranges = merged.tolist()

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
