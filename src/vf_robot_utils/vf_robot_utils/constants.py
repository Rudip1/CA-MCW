"""
constants.py — single source of truth for workspace-rooted paths and
model directories shared by vf_robot_utils and vf_robot_controller.

Import example:
    from vf_robot_utils.constants import TRAINING_ROOT, EVALUATION_ROOT, MAPS_ROOT

Path precedence (each override wins independently):
    VF_DATA_ROOT       → DATA_ROOT       (default: <workspace>/vf_data)
    VF_MAPS_ROOT       → MAPS_ROOT       (default: <workspace>/maps)
    VF_MODELS_ROOT     → MODELS_ROOT     (default: <workspace>/src/vf_robot_controller/models)
    VF_WORKSPACE_ROOT  → WORKSPACE_ROOT  (default: nearest dir containing src/+install/)
"""

import os
from pathlib import Path


def _find_workspace_root() -> Path:
    """Walk up from this file looking for the colcon workspace marker
    (a directory that has both src/ and install/ as siblings). Falls
    back to ~/CA-MCW then $HOME so imports never fail at module
    load."""
    here = Path(__file__).resolve().parent
    for cur in [here, *here.parents]:
        if (cur / "src").is_dir() and (cur / "install").is_dir():
            return cur
    candidate = Path.home() / "CA-MCW"
    return candidate if candidate.is_dir() else Path.home()


WORKSPACE_ROOT = Path(
    os.environ.get("VF_WORKSPACE_ROOT", str(_find_workspace_root()))
)

# ── Data store ─────────────────────────────────────────────────────────
# DATA_ROOT/vf_data_training/{manual,batch}/<map>/<goal>/<Planner>/<controller>/run_*.h5
# DATA_ROOT/vf_data_evaluation/batch/<map>/<goal>/<Planner>/<variant>/run_*.h5
DATA_ROOT = Path(
    os.environ.get("VF_DATA_ROOT", str(WORKSPACE_ROOT / "vf_data"))
)
TRAINING_ROOT   = DATA_ROOT / "vf_data_training"
EVALUATION_ROOT = DATA_ROOT / "vf_data_evaluation"

# 2-D maps + per-map sidecars (training_goalposes_collect.csv,
# evaluation_goalposes_collect.csv, <map>.yaml/.pgm/.db).
MAPS_ROOT = Path(
    os.environ.get("VF_MAPS_ROOT", str(WORKSPACE_ROOT / "maps"))
)

# Trained ONNX/PT artefacts, organised by family/<ch>_<hp>/.
MODELS_ROOT = Path(
    os.environ.get(
        "VF_MODELS_ROOT",
        str(WORKSPACE_ROOT / "src" / "vf_robot_controller" / "models"),
    )
)
MODELS_METACRITIC_RAW_ROOT    = MODELS_ROOT / "metacritic_raw_wt"
MODELS_METACRITIC_ORACLE_ROOT = MODELS_ROOT / "metacritic_oracle_wt"
MODELS_IMITATION_ROOT         = MODELS_ROOT / "imitation_wt"
