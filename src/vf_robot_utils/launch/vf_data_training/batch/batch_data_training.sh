#!/bin/bash
# =============================================================================
# batch_data_training.sh — Drive one row of training_goalposes_collect.csv
#                          and collect HDF5 episodes under
#                          vf_data/vf_data_training/batch/.
#
# Wraps:  vf_data_training_batch_fixedwt.launch.py
#
# Usage:
#   ./batch_data_training.sh <map_name> <planner> <run_id> [extra launch args...]
#
# Examples:
#   ./batch_data_training.sh house_my1_map NavFn 0
#   ./batch_data_training.sh house_my1_map SmacPlanner2D 1 reposition_first:=false
#   ./batch_data_training.sh my_hospital ThetaStar 2 settle_s:=3.0
#
# Prerequisites:
#   1. Map exists at maps/<map_name>/{<map_name>.yaml, <map_name>.db}
#   2. Goals already recorded:
#        ros2 launch vf_robot_utils training_goalposes_collect.launch.py \
#            map_name:=<map_name>
#      Each Ctrl-C of that step appends one row (run_id auto-increments).
# =============================================================================

set -euo pipefail

if [ "$#" -lt 3 ]; then
    cat <<'EOF'
Usage: batch_data_training.sh <map_name> <planner> <run_id> [extra args...]

Args:
  map_name   Folder under maps/ (e.g. house_my1_map, my_hospital)
  planner    NavFn | SmacPlanner2D | SmacPlannerHybrid | SmacLattice | ThetaStar
  run_id     Integer row in maps/<map_name>/training_goalposes_collect.csv
EOF
    exit 1
fi

MAP_NAME="$1"
PLANNER="$2"
RUN_ID="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_FILE="${SCRIPT_DIR}/vf_data_training_batch_fixedwt.launch.py"

if [ ! -f "${LAUNCH_FILE}" ]; then
    echo "ERROR: launch file not found: ${LAUNCH_FILE}" >&2
    exit 2
fi

echo "============================================="
echo "  Batch training run"
echo "  Map:        ${MAP_NAME}"
echo "  Planner:    ${PLANNER}"
echo "  run_id:     ${RUN_ID}"
echo "  Launch:     ${LAUNCH_FILE}"
echo "============================================="

ros2 launch "${LAUNCH_FILE}" \
    "map:=${MAP_NAME}" \
    "planner:=${PLANNER}" \
    "run_id:=${RUN_ID}" \
    "$@"
