"""
model_defaults.py — single source of truth for default model run folders.

All launch files import from here so that switching the active weights
requires editing exactly one file. To switch, comment the current ACTIVE
block and uncomment the one you want.

Naming convention for run folders
---------------------------------
    <channels>_<runtag>
    where <channels> ∈ {v1, v2, v3} matches the channel set the model
    was trained on (channels_v1 = 126 dims, channels_v2 = 130 dims,
    channels_v3 = 170 dims). The runtag is free-form (manual_v1,
    seed0, run2_2026_05_20, ...).

Variable contracts:
  DEFAULT_INFERENCE_TYPE    : "raw" | "oracle"
                              Selects metacritic_raw_wt/  vs  metacritic_oracle_wt/
  DEFAULT_INFERENCE_WEIGHTS : run folder name inside that family directory
  DEFAULT_IMITATION_WEIGHTS : run folder name inside imitation_wt/

Resolved on-disk paths:
  metacritic ONNX → models/metacritic_<DEFAULT_INFERENCE_TYPE>_wt/<DEFAULT_INFERENCE_WEIGHTS>/meta_critic.onnx
  imitation ONNX  → models/imitation_wt/<DEFAULT_IMITATION_WEIGHTS>/imitation.onnx

Override at launch time without editing this file:
  ros2 launch vf_robot_bringup bringup_launch.py \\
      inference_model_type:=raw  inference_weights:=v1_manual_v1
  ros2 launch vf_robot_bringup bringup_launch.py \\
      imitation_weights:=v3_run2
  ros2 launch vf_robot_bringup bringup_launch.py \\
      onnx_path:=/abs/custom/path.onnx   # escape hatch — bypasses both folder args
"""

# =============================================================================
# Inference (meta-critic) — pick ONE block.
# =============================================================================

# ── ACTIVE: ORACLE labels, channels_v1 (126 dims) ────────────────────────────
DEFAULT_INFERENCE_TYPE    = "oracle"
DEFAULT_INFERENCE_WEIGHTS = "v1_manual_v1"
# Resolved: models/metacritic_oracle_wt/v1_manual_v1/meta_critic.onnx

# ── ORACLE labels, channels_v2 (130 dims) ────────────────────────────────────
# DEFAULT_INFERENCE_TYPE    = "oracle"
# DEFAULT_INFERENCE_WEIGHTS = "v2_manual_v1"
# Resolved: models/metacritic_oracle_wt/v2_manual_v1/meta_critic.onnx

# ── ORACLE labels, channels_v3 (170 dims) ────────────────────────────────────
# DEFAULT_INFERENCE_TYPE    = "oracle"
# DEFAULT_INFERENCE_WEIGHTS = "v3_manual_v1"
# Resolved: models/metacritic_oracle_wt/v3_manual_v1/meta_critic.onnx

# ── RAW labels, channels_v1 (126 dims) ───────────────────────────────────────
# DEFAULT_INFERENCE_TYPE    = "raw"
# DEFAULT_INFERENCE_WEIGHTS = "v1_manual_v1"
# Resolved: models/metacritic_raw_wt/v1_manual_v1/meta_critic.onnx

# ── RAW labels, channels_v2 (130 dims) ───────────────────────────────────────
# DEFAULT_INFERENCE_TYPE    = "raw"
# DEFAULT_INFERENCE_WEIGHTS = "v2_manual_v1"
# Resolved: models/metacritic_raw_wt/v2_manual_v1/meta_critic.onnx

# ── RAW labels, channels_v3 (170 dims) ───────────────────────────────────────
# DEFAULT_INFERENCE_TYPE    = "raw"
# DEFAULT_INFERENCE_WEIGHTS = "v3_manual_v1"
# Resolved: models/metacritic_raw_wt/v3_manual_v1/meta_critic.onnx


# =============================================================================
# Imitation (behaviour cloning, vx + wz direct output) — pick ONE block.
# =============================================================================

# ── ACTIVE: imitation, channels_v1 (126 dims) ────────────────────────────────
DEFAULT_IMITATION_WEIGHTS = "v1_manual_v1"
# Resolved: models/imitation_wt/v1_manual_v1/imitation.onnx

# ── imitation, channels_v2 (130 dims) ────────────────────────────────────────
# DEFAULT_IMITATION_WEIGHTS = "v2_manual_v1"
# Resolved: models/imitation_wt/v2_manual_v1/imitation.onnx

# ── imitation, channels_v3 (170 dims) ────────────────────────────────────────
# DEFAULT_IMITATION_WEIGHTS = "v3_manual_v1"
# Resolved: models/imitation_wt/v3_manual_v1/imitation.onnx
