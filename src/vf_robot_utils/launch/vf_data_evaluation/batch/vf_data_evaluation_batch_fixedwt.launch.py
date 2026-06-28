#!/usr/bin/env python3
"""
vf_data_evaluation_batch_fixedwt.launch.py
------------------------------------------
Batch (CSV-driven) evaluation with the fixed-weight controller (the
ablation baseline that produced the training corpus). No ONNX, no
sidecar — bringup loads vf_fixedwt and MPPI runs with the YAML weights.

Output:
  vf_data/vf_data_evaluation/batch/<map>/<goal>/<Planner>/fixedwt/run_*.h5

Usage:
  ros2 launch vf_robot_utils vf_data_evaluation_batch_fixedwt.launch.py \\
      map:=house_my1_map planner:=NavFn run_id:=0
"""
from vf_robot_utils.launch_builders import build_eval_batch_launch_description


def generate_launch_description():
    return build_eval_batch_launch_description("fixedwt")
