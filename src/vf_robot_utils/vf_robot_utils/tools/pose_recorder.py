#!/usr/bin/env python3
"""
pose_recorder.py
----------------
Listens to two RViz topics:
  - /initialpose   (geometry_msgs/PoseWithCovarianceStamped)  → "2D Pose Estimate"
  - /goal_pose     (geometry_msgs/PoseStamped)                → "2D Nav Goal" / waypoints

On SIGINT (Ctrl-C or launch shutdown) the session row is appended to:
  <output_dir>/<map_name>/<map_name>_waypoints.csv

CSV format (comma-delimited, compatible with csv_schema.py):
  run_id,notes,start_x,start_y,start_yaw,g1_x,g1_y,g1_yaw,g2_x,g2_y,g2_yaw,...
"""

import csv
import math
import os
import signal
import sys
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from vf_robot_utils.constants import MAPS_ROOT


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _r3(v: float) -> float:
    return round(v, 3)


class PoseRecorder(Node):

    def __init__(self):
        super().__init__('pose_recorder')

        self.declare_parameter('map_name',    'house_my1_map')
        self.declare_parameter('output_dir',  str(MAPS_ROOT))
        self.declare_parameter('csv_filename', '')  # empty → <map_name>_waypoints.csv

        self._map_name    = self.get_parameter('map_name').value
        self._output_dir  = self.get_parameter('output_dir').value
        _csv_fn           = self.get_parameter('csv_filename').value
        self._csv_filename = _csv_fn if _csv_fn else f'{self._map_name}_waypoints.csv'

        self._lock    = threading.Lock()
        self._start: dict | None = None
        self._goals:  list[dict] = []
        self._saved   = False

        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self._cb_start, 10)
        self.create_subscription(
            PoseStamped, '/goal_pose', self._cb_goal, 10)

        signal.signal(signal.SIGINT,  self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        self.get_logger().info(
            f'[PoseRecorder] Ready — map: {self._map_name}\n'
            f'  CSV → {self._output_dir}/{self._map_name}/{self._csv_filename}\n'
            f'  Click "2D Pose Estimate": 1st click = START, every click after = GOAL waypoint.\n'
            f'  Ctrl-C to save and exit.'
        )

    # ── callbacks ─────────────────────────────────────────────────────────

    def _cb_start(self, msg: PoseWithCovarianceStamped):
        q = msg.pose.pose.orientation
        x, y, yaw = _r3(msg.pose.pose.position.x), _r3(msg.pose.pose.position.y), _r3(_yaw_from_quaternion(q))
        with self._lock:
            if self._start is None:
                self._start = {'x': x, 'y': y, 'yaw': yaw}
                self.get_logger().info(f'[PoseRecorder] START  x={x}  y={y}  yaw={yaw:.3f}')
            else:
                idx = len(self._goals) + 1
                self._goals.append({'x': x, 'y': y, 'yaw': yaw})
                self.get_logger().info(f'[PoseRecorder] GOAL g{idx}  x={x}  y={y}  yaw={yaw:.3f}')

    def _cb_goal(self, msg: PoseStamped):
        q = msg.pose.orientation
        x, y, yaw = _r3(msg.pose.position.x), _r3(msg.pose.position.y), _r3(_yaw_from_quaternion(q))
        with self._lock:
            idx = len(self._goals) + 1
            self._goals.append({'x': x, 'y': y, 'yaw': yaw})
        self.get_logger().info(f'[PoseRecorder] GOAL g{idx}  x={x}  y={y}  yaw={yaw:.3f}')

    # ── CSV ───────────────────────────────────────────────────────────────

    def _csv_path(self) -> str:
        folder = os.path.join(self._output_dir, self._map_name)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, self._csv_filename)

    def _next_run_id(self, csv_path: str) -> int:
        if not os.path.isfile(csv_path):
            return 0
        max_id = -1
        try:
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    try:
                        max_id = max(max_id, int(row.get('run_id', -1)))
                    except ValueError:
                        pass
        except Exception:
            pass
        return max_id + 1

    def _save(self):
        with self._lock:
            if self._saved:
                return
            start = self._start
            goals = list(self._goals)

        if start is None and not goals:
            self.get_logger().warn('[PoseRecorder] Nothing recorded — CSV not updated.')
            return

        csv_path = self._csv_path()
        run_id   = self._next_run_id(csv_path)

        new_row: dict = {'run_id': str(run_id), 'notes': ''}
        if start:
            new_row.update({'start_x': str(start['x']), 'start_y': str(start['y']),
                            'start_yaw': str(start['yaw'])})
        else:
            new_row.update({'start_x': '', 'start_y': '', 'start_yaw': ''})
        for i, g in enumerate(goals, 1):
            new_row[f'g{i}_x'] = str(g['x'])
            new_row[f'g{i}_y'] = str(g['y'])
            new_row[f'g{i}_yaw'] = str(g['yaw'])

        # Read existing rows + build merged header
        existing_rows, existing_header = [], []
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, newline='') as f:
                    reader = csv.DictReader(f)
                    existing_header = list(reader.fieldnames or [])
                    existing_rows   = [dict(r) for r in reader]
            except Exception as e:
                self.get_logger().error(f'[PoseRecorder] Could not read CSV: {e}')

        goal_cols = [f'g{i}_{ax}' for i in range(1, len(goals) + 1) for ax in ('x', 'y', 'yaw')]
        merged = list(existing_header)
        for col in ['run_id', 'notes', 'start_x', 'start_y', 'start_yaw'] + goal_cols:
            if col not in merged:
                merged.append(col)

        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=merged, extrasaction='ignore')
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({col: row.get(col, '') for col in merged})
                writer.writerow({col: new_row.get(col, '') for col in merged})
            self._saved = True
            self.get_logger().info(
                f'[PoseRecorder] Saved run_id={run_id} ({len(goals)} goal(s)) → {csv_path}')
        except Exception as e:
            self.get_logger().error(f'[PoseRecorder] Write failed: {e}')

    # ── shutdown ──────────────────────────────────────────────────────────

    def _on_signal(self, signum, frame):
        self.get_logger().info('[PoseRecorder] Saving session …')
        self._save()
        rclpy.shutdown()
        sys.exit(0)

    def on_shutdown(self):
        self._save()


def main(args=None):
    rclpy.init(args=args)
    node = PoseRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
