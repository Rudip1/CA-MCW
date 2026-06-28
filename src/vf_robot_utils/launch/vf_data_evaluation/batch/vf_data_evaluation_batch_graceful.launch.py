#!/usr/bin/env python3
"""
vf_data_evaluation_batch_graceful.launch.py
-------------------------------------------
Batch (CSV-driven) evaluation of the stock Nav2 Graceful baseline.
No ONNX, no sidecar; bringup loads controller:=graceful.

Output:
  vf_data/vf_data_evaluation/batch/<map>/<goal>/<Planner>/graceful/run_*.h5

Usage:
  ros2 launch vf_robot_utils vf_data_evaluation_batch_graceful.launch.py \\
      map:=house_my1_map planner:=NavFn run_id:=0
"""
from vf_robot_utils.launch_builders import build_eval_batch_launch_description


def generate_launch_description():
    return build_eval_batch_launch_description("graceful")
