#!/usr/bin/env python3
"""
vf_data_evaluation_batch_rawwt.launch.py
----------------------------------------
Batch (CSV-driven) evaluation of one trained raw-meta-critic variant.
Mirrors the stack used by the working
vf_data_training_batch_fixedwt.launch.py (bringup + data_collector_node +
tour_runner), but:

  - bringup loads controller:=vf_inferencewt with inference_model_type:=raw
    and inference_weights:=<ch>_<hp>. The sidecar metacritic_inference_node
    publishes /vf/weights at 20 Hz to drive MPPI critic weights at runtime;
  - the collector's `training_root` parameter is repointed at
    EVALUATION_ROOT so HDF5s land under vf_data/vf_data_evaluation/.

Output path:
  vf_data/vf_data_evaluation/batch/<map>/<goal_xy>/<Planner>/rawwt_<hp>_<ch>/run_*.h5

Required args:
  map:=         Map name (folder under MAPS_ROOT).
  planner:=     Global planner key. {NavFn, SmacPlanner2D, SmacPlannerHybrid,
                SmacLattice, ThetaStar}
  run_id:=      Integer row to replay from evaluation_goalposes_collect.csv.
  hp:=          Hyperparam group: normal | tuned | hardreg.
  ch:=          Channel set:      v1 | v2 | v3.

Usage:
  ros2 launch vf_robot_utils vf_data_evaluation_batch_rawwt.launch.py \\
      map:=house_my1_map planner:=NavFn run_id:=0 hp:=hardreg ch:=v3

  ros2 launch vf_robot_utils vf_data_evaluation_batch_rawwt.launch.py \\
      map:=house_my1_map planner:=NavFn run_id:=0 hp:=normal ch:=v1

See launch_builders.build_eval_batch_launch_description for the full arg
surface (the optional knobs match vf_data_training_batch_fixedwt.launch.py
1:1 — settle_s, post_reposition_stabilize_s, reposition_first, etc.).
"""
from vf_robot_utils.launch_builders import build_eval_batch_launch_description


def generate_launch_description():
    return build_eval_batch_launch_description("rawwt")
