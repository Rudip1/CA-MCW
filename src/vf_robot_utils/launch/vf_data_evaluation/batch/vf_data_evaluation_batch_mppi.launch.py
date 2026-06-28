#!/usr/bin/env python3
"""
vf_data_evaluation_batch_mppi.launch.py
---------------------------------------
Batch (CSV-driven) evaluation of the stock Nav2 MPPI baseline.
No ONNX, no sidecar; bringup loads controller:=mppi.

Output:
  vf_data/vf_data_evaluation/batch/<map>/<goal>/<Planner>/mppi/run_*.h5

Usage:
  ros2 launch vf_robot_utils vf_data_evaluation_batch_mppi.launch.py \\
      map:=house_my1_map planner:=NavFn run_id:=0
"""
from vf_robot_utils.launch_builders import build_eval_batch_launch_description


def generate_launch_description():
    return build_eval_batch_launch_description("mppi")
