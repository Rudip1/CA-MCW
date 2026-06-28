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
PointCloud2 → LaserScan Converter — ViroFighter UVC-1
═════════════════════════════════════════════════════

Replaces pointcloud_to_laserscan — zero lazy subscription, zero message_filters.

Why this exists:
    The ROS 2 pointcloud_to_laserscan node uses message_filters::Subscriber with
    a lazy subscription thread. In Humble + CycloneDDS, subscriptions created by
    the background thread after spin() starts are never processed by the executor.
    The node appears subscribed (ros2 node info confirms it), receives data at
    the DDS layer, but cloudCallback never fires. This custom node uses a normal
    rclpy subscription — no lazy pattern, no message_filters, deterministic.

How it works:
    1. Subscribes to PointCloud2 (normal rclpy subscription, BEST_EFFORT QoS)
    2. On each message, looks up TF from pointcloud frame → target_frame
    3. Transforms all points using the full 4×4 rotation+translation matrix
    4. Filters by world-space Z height (removes floor, ceiling)
    5. Projects remaining points to 2D (x,y) → range + angle
    6. Bins into angular slots, keeps closest range per bin
    7. Publishes LaserScan in target_frame

Usage:
    ros2 run vf_robot_slam pc_to_scan.py

Parameters:
    target_frame:       output frame for scan (default: "base_footprint")
    min_height:         minimum world Z to keep (default: 0.02 m)
    max_height:         maximum world Z to keep (default: 2.0 m)
    angle_min:          scan start angle in radians (default: -pi)
    angle_max:          scan end angle in radians (default: +pi)
    angle_increment:    angular resolution (default: 0.00581 ~ 0.33°)
    range_min:          minimum 2D range (default: 0.1 m)
    range_max:          maximum 2D range (default: 6.0 m)
    transform_tolerance: max TF wait time in seconds (default: 0.1)
"""

import math
import struct

import numpy as np
import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from tf2_ros import Buffer, TransformListener


def _build_rotation_matrix(q):
    """Build a 3×3 rotation matrix from a geometry_msgs Quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _extract_xyz(msg: PointCloud2):
    """
    Extract x, y, z arrays from a PointCloud2 message using struct.

    Returns (N, 3) float32 numpy array. NaN points are excluded.
    Works with any field layout (handles offset + datatype correctly).
    """
    # Find x, y, z field offsets
    offsets = {}
    for field in msg.fields:
        if field.name in ("x", "y", "z"):
            offsets[field.name] = field.offset

    if len(offsets) != 3:
        return np.empty((0, 3), dtype=np.float32)

    ox, oy, oz = offsets["x"], offsets["y"], offsets["z"]
    point_step = msg.point_step
    data = bytes(msg.data)
    n_points = msg.width * msg.height

    # Pre-allocate and extract using struct
    points = np.empty((n_points, 3), dtype=np.float32)
    fmt = "<f"  # little-endian float32

    for i in range(n_points):
        base = i * point_step
        points[i, 0] = struct.unpack_from(fmt, data, base + ox)[0]
        points[i, 1] = struct.unpack_from(fmt, data, base + oy)[0]
        points[i, 2] = struct.unpack_from(fmt, data, base + oz)[0]

    # Remove NaN points
    valid = ~np.isnan(points).any(axis=1)
    return points[valid]


class PC2ToScan(Node):
    def __init__(self):
        super().__init__("pc_to_scan")

        # ── Declare parameters ───────────────────────────────────────────────
        self.declare_parameter("target_frame", "base_footprint")
        self.declare_parameter("min_height", 0.02)
        self.declare_parameter("max_height", 2.0)
        self.declare_parameter("angle_min", -math.pi)
        self.declare_parameter("angle_max", math.pi)
        self.declare_parameter("angle_increment", 0.00581)
        self.declare_parameter("range_min", 0.1)
        self.declare_parameter("range_max", 6.0)
        self.declare_parameter("transform_tolerance", 0.1)

        # ── Read parameters ──────────────────────────────────────────────────
        self.target_frame = self.get_parameter("target_frame").value
        self.min_height = self.get_parameter("min_height").value
        self.max_height = self.get_parameter("max_height").value
        self.angle_min_val = self.get_parameter("angle_min").value
        self.angle_max_val = self.get_parameter("angle_max").value
        self.angle_increment = self.get_parameter("angle_increment").value
        self.range_min = self.get_parameter("range_min").value
        self.range_max = self.get_parameter("range_max").value
        self.tf_tolerance = self.get_parameter("transform_tolerance").value

        self.num_bins = int(
            (self.angle_max_val - self.angle_min_val) / self.angle_increment
        )

        # ── TF2 ──────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── QoS — match Gazebo / RealSense pointcloud publishers ─────────────
        # Use RELIABLE to match Gazebo's default; also compatible with
        # RealSense which typically publishes RELIABLE.
        cloud_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        scan_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Subscriber — normal rclpy, NO message_filters, NO lazy sub ───────
        # Topic is remapped via launch file, default "cloud_in" for CLI usage
        self.create_subscription(
            PointCloud2, "cloud_in", self._cloud_callback, cloud_qos
        )

        # ── Publisher ────────────────────────────────────────────────────────
        self.scan_pub = self.create_publisher(LaserScan, "scan", scan_qos)

        # ── Cache for static TF (optical frame → base_footprint) ─────────────
        self._cached_frame = None
        self._cached_rot = None
        self._cached_trans = None

        self.get_logger().info(
            f"PC2ToScan: cloud_in → scan "
            f"(target_frame={self.target_frame}, "
            f"height={self.min_height}–{self.max_height} m, "
            f"range={self.range_min}–{self.range_max} m)"
        )

    def _get_transform(self, source_frame: str):
        """
        Look up and cache the transform from source_frame → target_frame.

        For camera frames attached to the URDF, this is a static transform
        (published at time 0.0). We cache it after the first successful
        lookup to avoid repeated TF queries.
        """
        if self._cached_frame == source_frame and self._cached_rot is not None:
            return self._cached_rot, self._cached_trans

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(),  # latest available (time 0 for static)
                timeout=rclpy.duration.Duration(seconds=self.tf_tolerance),
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed: {source_frame} → {self.target_frame}: {e}",
                throttle_duration_sec=2.0,
            )
            return None, None

        rot = _build_rotation_matrix(tf.transform.rotation)
        trans = np.array(
            [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ]
        )

        # Cache it — URDF transforms are static
        self._cached_frame = source_frame
        self._cached_rot = rot
        self._cached_trans = trans

        self.get_logger().info(f"TF cached: {source_frame} → {self.target_frame}")
        return rot, trans

    def _cloud_callback(self, msg: PointCloud2):
        # ── Get transform ────────────────────────────────────────────────────
        rot, trans = self._get_transform(msg.header.frame_id)
        if rot is None:
            return

        # ── Extract XYZ points ───────────────────────────────────────────────
        points = _extract_xyz(msg)
        if points.shape[0] == 0:
            return

        # ── Transform to target_frame ────────────────────────────────────────
        # points_tf = (R @ points.T).T + trans  →  (N,3)
        points_tf = (rot @ points.T).T + trans

        # ── Filter by world-space Z height ───────────────────────────────────
        z = points_tf[:, 2]
        mask = (z >= self.min_height) & (z <= self.max_height)
        points_2d = points_tf[mask, :2]  # only x, y needed now

        if points_2d.shape[0] == 0:
            # Publish empty scan (all inf) so downstream knows we're alive
            self._publish_empty_scan(msg.header.stamp)
            return

        # ── Project to 2D polar ──────────────────────────────────────────────
        x = points_2d[:, 0]
        y = points_2d[:, 1]
        ranges = np.sqrt(x * x + y * y)
        angles = np.arctan2(y, x)

        # ── Filter by range ──────────────────────────────────────────────────
        range_mask = (ranges >= self.range_min) & (ranges <= self.range_max)
        ranges = ranges[range_mask]
        angles = angles[range_mask]

        if ranges.shape[0] == 0:
            self._publish_empty_scan(msg.header.stamp)
            return

        # ── Bin into angular slots (keep closest per bin) ────────────────────
        bin_indices = ((angles - self.angle_min_val) / self.angle_increment).astype(
            np.int32
        )
        valid_bin = (bin_indices >= 0) & (bin_indices < self.num_bins)
        bin_indices = bin_indices[valid_bin]
        ranges = ranges[valid_bin]

        # Initialize all bins to inf
        scan_ranges = np.full(self.num_bins, float("inf"), dtype=np.float32)

        # Keep minimum range per bin — numpy ufunc at method
        np.minimum.at(scan_ranges, bin_indices, ranges.astype(np.float32))

        # ── Publish ──────────────────────────────────────────────────────────
        scan = LaserScan()
        scan.header.stamp = msg.header.stamp
        scan.header.frame_id = self.target_frame
        scan.angle_min = float(self.angle_min_val)
        scan.angle_max = float(self.angle_max_val)
        scan.angle_increment = float(self.angle_increment)
        scan.time_increment = 0.0
        scan.scan_time = 0.033
        scan.range_min = float(self.range_min)
        scan.range_max = float(self.range_max)
        scan.ranges = scan_ranges.tolist()

        self.scan_pub.publish(scan)

    def _publish_empty_scan(self, stamp):
        """Publish an all-inf scan so downstream nodes know we're alive."""
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = self.target_frame
        scan.angle_min = float(self.angle_min_val)
        scan.angle_max = float(self.angle_max_val)
        scan.angle_increment = float(self.angle_increment)
        scan.time_increment = 0.0
        scan.scan_time = 0.033
        scan.range_min = float(self.range_min)
        scan.range_max = float(self.range_max)
        scan.ranges = [float("inf")] * self.num_bins

        self.scan_pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = PC2ToScan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
