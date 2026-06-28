#!/usr/bin/env bash
# run_comparison.sh — full cross-controller comparison suite (one command).
# [ORCHESTRATOR] self-contained: aggregate HDF5 -> TOPSIS -> comparison
#   figures. Calls python modules directly (no other .sh).
#
# Compares a fixed curated set from the start — no screen->top3->top1
# funnel. The set:
#     mppi, dwb, rpp           classical baselines
#     fixedwt                  custom fixed-weight baseline
#     imitationwt_normal_v1    imitation
#     <TOPSIS rank 1>          top trained adaptive controller
#
# Produces, for that set, every comparison-style figure (Stage-A renderer
# styles): per-tier bar panels, trajectory overlays, XTE violin / profile
# / envelope.
#
# Output: _aggregate/<STEM>/comparison/
#   tiers/         tier_1_outcome.pdf ... tier_6_adaptation.pdf
#   trajectories/  trajectories_<goal>.pdf
#   xte/           xte_violin.pdf, xte_profile_<goal>.pdf,
#                  xte_envelope_<controller>.pdf
#   comparison_summary.csv
#
# For every controller (all trained + classical) use run_comparison_all.sh.
#
# Env (defaults):  MAP=house_my1_map   STEM=thesis_eval
#
# Usage:
#   bash src/vf_robot_utils/tools/run_comparison.sh

set -eo pipefail

MAP="${MAP:-house_my1_map}"
STEM="${STEM:-thesis_eval}"
ROOT="vf_data/vf_data_evaluation/batch/${MAP}"
AGG="${ROOT}/_aggregate/${STEM}"

cd "$(dirname "$0")/../../.."
set +u; source install/setup.bash; set -u

echo "==========================================================="
echo "[comparison] map=${MAP}  stem=${STEM}"
echo "==========================================================="

echo
echo "----------------- aggregate HDF5 -> results.csv ----------------"
python3 -m vf_robot_utils.analysis.csv_pipeline.aggregate_csv \
    --root "${ROOT}" --csv-stem "${STEM}"

RES="${AGG}/results.csv"

echo
echo "----------------- TOPSIS selection (picks top-1) ---------------"
python3 -m vf_robot_utils.analysis.csv_pipeline.controller_selection \
    --results "${RES}"

echo
echo "----------------- comparison figures (headline 6-set) ----------"
python3 -m vf_robot_utils.analysis.figures.comparison_figures \
    --results "${RES}" --agg-dir "${AGG}" --root "${ROOT}" --scope headline

echo
echo "==========================================================="
echo "[comparison] done. Folder: ${AGG}/comparison/"
echo "==========================================================="
