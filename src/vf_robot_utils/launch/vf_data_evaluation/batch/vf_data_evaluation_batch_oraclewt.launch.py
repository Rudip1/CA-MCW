#!/usr/bin/env python3
"""
vf_data_evaluation_batch_oraclewt.launch.py
-------------------------------------------
Batch (CSV-driven) evaluation of one trained oracle-meta-critic variant.
Mirrors the stack used by the working
vf_data_training_batch_fixedwt.launch.py (bringup + data_collector_node +
tour_runner), but:

  - bringup loads controller:=vf_inferencewt with inference_model_type:=oracle
    and inference_weights:=<ch>_<hp>. The sidecar metacritic_inference_node
    loads models/metacritic_oracle_wt/<ch>_<hp>/meta_critic.onnx (the
    QP-labelled variant) and publishes /vf/weights to drive MPPI;
  - the collector's `training_root` parameter is repointed at
    EVALUATION_ROOT so HDF5s land under vf_data/vf_data_evaluation/.

Note: on the current 85-episode corpus the oracle models did not converge
(train/val ~1.21 flat across all 9 oracle variants) — closed-loop behaviour
will likely be close to vf_fixedwt with uniform weights. Run anyway so the
thesis comparison has a numeric baseline rather than an asterisk.

Output path:
  vf_data/vf_data_evaluation/batch/<map>/<goal_xy>/<Planner>/oraclewt_<hp>_<ch>/run_*.h5

Required args:
  map:=         Map name (folder under MAPS_ROOT).
  planner:=     Global planner key. {NavFn, SmacPlanner2D, SmacPlannerHybrid,
                SmacLattice, ThetaStar}
  run_id:=      Integer row to replay from evaluation_goalposes_collect.csv.
  hp:=          Hyperparam group: normal | tuned | hardreg.
  ch:=          Channel set:      v1 | v2 | v3.

Usage:
  ros2 launch vf_robot_utils vf_data_evaluation_batch_oraclewt.launch.py \\
      map:=house_my1_map planner:=NavFn run_id:=0 hp:=hardreg ch:=v3

  ros2 launch vf_robot_utils vf_data_evaluation_batch_oraclewt.launch.py \\
      map:=house_my1_map planner:=NavFn run_id:=0 hp:=normal ch:=v1

See launch_builders.build_eval_batch_launch_description for the full arg
surface (the optional knobs match vf_data_training_batch_fixedwt.launch.py
1:1 — settle_s, post_reposition_stabilize_s, reposition_first, etc.).
"""
from vf_robot_utils.launch_builders import build_eval_batch_launch_description


def generate_launch_description():
    return build_eval_batch_launch_description("oraclewt")
