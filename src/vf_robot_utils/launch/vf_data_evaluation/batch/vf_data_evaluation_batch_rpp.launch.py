#!/usr/bin/env python3
"""
vf_data_evaluation_batch_rpp.launch.py
--------------------------------------
Batch (CSV-driven) evaluation of the stock Nav2 Regulated Pure Pursuit
baseline. No ONNX, no sidecar; bringup loads controller:=rpp.

Output:
  vf_data/vf_data_evaluation/batch/<map>/<goal>/<Planner>/rpp/run_*.h5

Usage:
  ros2 launch vf_robot_utils vf_data_evaluation_batch_rpp.launch.py \\
      map:=house_my1_map planner:=NavFn run_id:=0
"""
from vf_robot_utils.launch_builders import build_eval_batch_launch_description


def generate_launch_description():
    return build_eval_batch_launch_description("rpp")
