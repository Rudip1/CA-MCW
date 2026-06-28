#!/usr/bin/env bash
# sweep_classical.sh — drive the 4 classical Nav2 baselines through one
# evaluation tour row (default: run_id=0 of evaluation_goalposes_collect.csv).
#
# Prereqs:
#   1. Gazebo must already be running in another terminal:
#        ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py
#   2. data_collector_node patch applied so baseline mode writes per-step
#      rows (see EVALUATION_PLAN.md § 3 and the matching commit).
#
# Usage:
#   bash sweep_classical.sh                         # default map+planner+run
#   MAP=house_my1_map PLANNER=NavFn RUN=0 bash sweep_classical.sh
#   HEADLESS=true bash sweep_classical.sh
#
# Output:
#   vf_data/vf_data_evaluation/batch/<MAP>/<goal>/<PLANNER>/<ctrl>/run_*.h5
#   — one new HDF5 per goal per controller (3 goals * 4 ctrl = 12 new files).

set -euo pipefail

MAP="${MAP:-house_my1_map}"
PLANNER="${PLANNER:-NavFn}"
RUN="${RUN:-0}"
HEADLESS="${HEADLESS:-false}"
RVIZ="${RVIZ:-false}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"

CONTROLLERS=( mppi dwb rpp graceful )

echo "[sweep] map=$MAP planner=$PLANNER run_id=$RUN headless=$HEADLESS"
echo "[sweep] controllers: ${CONTROLLERS[*]}"
echo

for CTRL in "${CONTROLLERS[@]}"; do
    echo "============================================================"
    echo "[sweep] ${CTRL}  (run_id=${RUN})"
    echo "============================================================"
    ros2 launch vf_robot_utils \
        "vf_data_evaluation_batch_${CTRL}.launch.py" \
        map:="${MAP}" \
        planner:="${PLANNER}" \
        run_id:="${RUN}" \
        headless:="${HEADLESS}" \
        rviz:="${RVIZ}"
    echo "[sweep] ${CTRL} done; cooling down ${SLEEP_BETWEEN}s"
    sleep "${SLEEP_BETWEEN}"
done

echo
echo "[sweep] all 4 classical baselines done."
echo "[sweep] verify per-step rows with:"
echo "    python3 -c \"import h5py, glob; \\"
echo "      [print(f, h5py.File(f,'r')['sim_time'].shape) for f in glob.glob('vf_data/vf_data_evaluation/batch/${MAP}/goal_*/${PLANNER}/{mppi,dwb,rpp,graceful}/run_*.h5')]\""
