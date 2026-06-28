#!/usr/bin/env bash
# run_comparison_all.sh — comparison figures for EVERY controller.
# [ORCHESTRATOR] self-contained: aggregate HDF5 -> comparison figures.
#   Calls python modules directly (no other .sh).
#
# Same figure set as run_comparison.sh, but over every controller in
# the dataset (all trained + classical) instead of the curated
# 6-controller headline set. This tree goes in the thesis appendix.
#
# Output: _aggregate/<STEM>/comparison_all/
#   tiers/         tier_1_outcome.pdf ... tier_6_adaptation.pdf
#   topsis_ranking.pdf
#   xte/           xte_overview / violin / profile / envelope
#   trajectories/  trajectories_<goal>.pdf
#   comparison_summary.csv
#
# Env (defaults):  MAP=house_my1_map   STEM=thesis_eval
#
# Usage:
#   bash src/vf_robot_utils/tools/run_comparison_all.sh

set -eo pipefail

MAP="${MAP:-house_my1_map}"
STEM="${STEM:-thesis_eval}"
ROOT="vf_data/vf_data_evaluation/batch/${MAP}"
AGG="${ROOT}/_aggregate/${STEM}"

cd "$(dirname "$0")/../../.."
set +u; source install/setup.bash; set -u

echo "==========================================================="
echo "[comparison_all] map=${MAP}  stem=${STEM}"
echo "==========================================================="

echo
echo "----------------- aggregate HDF5 -> results.csv ----------------"
python3 -m vf_robot_utils.analysis.csv_pipeline.aggregate_csv \
    --root "${ROOT}" --csv-stem "${STEM}"

RES="${AGG}/results.csv"

echo
echo "----------------- comparison figures (all controllers) --------"
python3 -m vf_robot_utils.analysis.figures.comparison_figures \
    --results "${RES}" --agg-dir "${AGG}" --root "${ROOT}" --scope full

echo
echo "==========================================================="
echo "[comparison_all] done. Folder: ${AGG}/comparison_all/"
echo "==========================================================="
