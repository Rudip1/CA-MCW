#!/usr/bin/env bash
# sweep_eval.sh -- collect evaluation HDF5 logs for one controller cohort.
# [LEAF] data collection -- the only script that runs ROS/Gazebo and produces
#        raw HDF5. Run this first; the analysis scripts consume its output.
#
# Each entry is one `ros2 launch vf_data_evaluation_batch_<family>` call,
# which replays one tour row (the 3 evaluation goals) and shuts itself
# down when tour_runner exits.
#
#   Gazebo is NOT started here. Start it yourself in a separate terminal:
#     ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py
#
#   This script NEVER deletes existing HDF5. Re-running a variant appends
#   another run_*.h5 -- the pre-flight table shows the current per-variant
#   count so you can prune by hand first if you want a controlled n.
#
# Cohorts are never mixed: one invocation does trained OR baseline OR a
# single named variant.
#
# Usage:
#   bash sweep_eval.sh trained                       # 19 trained variants
#   bash sweep_eval.sh baselines                     # 5 baseline controllers
#   bash sweep_eval.sh rawwt_hardreg_v3              # one trained variant
#   bash sweep_eval.sh mppi                          # one baseline
#   bash sweep_eval.sh trained --list                # preview, launch nothing
#   bash sweep_eval.sh baselines --no-timeout        # let BT recoveries run
#   bash sweep_eval.sh baselines --per-goal-timeout 600 --episode-timeout 600
#
# Flags (any order, after the cohort/variant):
#   --list                       Show the pre-flight table and exit.
#   --no-timeout                 Shortcut for
#                                  --per-goal-timeout 99999
#                                  --episode-timeout 99999
#                                Lets Nav2's BT keep replanning + running
#                                recovery behaviors until the goal is
#                                achieved (or the BT itself gives up).
#   --per-goal-timeout <s>       Forward as per_goal_timeout_s:= to the
#                                ros2 launch (tour_runner per-NavigateToPose
#                                cap; default 180).
#   --episode-timeout <s>        Forward as episode_timeout_s:= to the
#                                ros2 launch (collector-side episode cap;
#                                default 180).
#
# Env (defaults):
#   MAP=house_my1_map   PLANNER=NavFn   RUN=0
#   HEADLESS=false      RVIZ=true       SLEEP_BETWEEN=5
#   LAUNCH_TIMEOUT=1800   per-variant outer cap in seconds (script side);
#                         set 0 to disable. Independent of the Nav2-side
#                         timeouts above.

set -uo pipefail

MAP="${MAP:-house_my1_map}"
PLANNER="${PLANNER:-NavFn}"
RUN="${RUN:-0}"
HEADLESS="${HEADLESS:-false}"
RVIZ="${RVIZ:-true}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"
LAUNCH_TIMEOUT="${LAUNCH_TIMEOUT:-1800}"

# ---- workspace root + ROS env ------------------------------------------
cd "$(dirname "$0")/../../.."
set +u; source install/setup.bash; set -u

# ---- cohorts -----------------------------------------------------------
# 18 metacritic variants + 1 imitation variant = 19 trained.
TRAINED=()
for fam in rawwt oraclewt; do
  for hp in normal tuned hardreg; do
    for ch in v1 v2 v3; do
      TRAINED+=("${fam}_${hp}_${ch}")
    done
  done
done
TRAINED+=("imitationwt_normal_v1")
BASELINES=(fixedwt mppi dwb rpp graceful)

is_baseline() { case "$1" in fixedwt|mppi|dwb|rpp|graceful) return 0;; *) return 1;; esac; }
is_trained()  { case "$1" in rawwt_*|oraclewt_*|imitationwt_*) return 0;; *) return 1;; esac; }

usage() {
  cat >&2 <<EOF
usage: sweep_eval.sh {trained|baselines|<variant>} [flags]

flags:
  --list                       preview the per-variant HDF5 count and exit
  --no-timeout                 set per-goal + episode timeouts to 99999 s
  --per-goal-timeout <s>       forwarded as per_goal_timeout_s:= to launch
  --episode-timeout <s>        forwarded as episode_timeout_s:= to launch
EOF
}

# ---- arg parse ---------------------------------------------------------
MODE="${1:-}"
shift || true
LIST_ONLY=false
PER_GOAL_TIMEOUT=""
EPISODE_TIMEOUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      LIST_ONLY=true; shift ;;
    --no-timeout)
      PER_GOAL_TIMEOUT=99999; EPISODE_TIMEOUT=99999; shift ;;
    --per-goal-timeout)
      [[ $# -ge 2 ]] || { echo "[sweep] --per-goal-timeout needs a value" >&2; usage; exit 64; }
      PER_GOAL_TIMEOUT="$2"; shift 2 ;;
    --episode-timeout)
      [[ $# -ge 2 ]] || { echo "[sweep] --episode-timeout needs a value" >&2; usage; exit 64; }
      EPISODE_TIMEOUT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[sweep] unknown flag: '$1'" >&2; usage; exit 64 ;;
  esac
done

VARIANTS=()
COHORT=""
case "$MODE" in
  trained)   VARIANTS=("${TRAINED[@]}");   COHORT="trained"  ;;
  baselines) VARIANTS=("${BASELINES[@]}"); COHORT="baseline" ;;
  "")
    usage; exit 64 ;;
  *)
    if is_trained "$MODE"; then VARIANTS=("$MODE"); COHORT="trained"
    elif is_baseline "$MODE"; then VARIANTS=("$MODE"); COHORT="baseline"
    else
      echo "[sweep] unknown mode/variant: '$MODE'" >&2
      usage
      exit 64
    fi ;;
esac

EVAL_DIR="vf_data/vf_data_evaluation/batch/${MAP}"
LOG_DIR="${EVAL_DIR}/_sweep_logs"

count_h5() {  # count_h5 <variant>  -> number of run_*.h5 across all goals
  ls "${EVAL_DIR}"/*/"${PLANNER}/$1"/*.h5 2>/dev/null | wc -l | tr -d ' '
}

# Render the launch-arg row in the pre-flight banner. Empty string means
# "use the launch file's own default".
timeout_summary() {
  local pg="${PER_GOAL_TIMEOUT:-<default 180>}"
  local ep="${EPISODE_TIMEOUT:-<default 180>}"
  echo "[sweep] per_goal_timeout_s=${pg}  episode_timeout_s=${ep}"
}

# ---- pre-flight --------------------------------------------------------
echo "============================================================"
echo "[sweep] cohort=${COHORT}  map=${MAP}  planner=${PLANNER}  run_id=${RUN}"
echo "[sweep] headless=${HEADLESS}  rviz=${RVIZ}  variants=${#VARIANTS[@]}"
timeout_summary
echo "[sweep] NEVER deletes HDF5 -- re-runs append. Current counts:"
echo "------------------------------------------------------------"
printf "  %-26s %s\n" "VARIANT" "existing run_*.h5"
for v in "${VARIANTS[@]}"; do
  printf "  %-26s %s\n" "$v" "$(count_h5 "$v")"
done
echo "------------------------------------------------------------"

if $LIST_ONLY; then
  echo "[sweep] --list: nothing launched."
  exit 0
fi

# ---- Gazebo health check ----------------------------------------------
if ! pgrep -x gzserver >/dev/null 2>&1; then
  echo "[sweep] ERROR: Gazebo (gzserver) is not running." >&2
  echo "        Start it first, in a separate terminal:" >&2
  echo "          ros2 launch vf_robot_gazebo house_my1_world_xacro.launch.py" >&2
  exit 1
fi
echo "[sweep] Gazebo OK (gzserver running)."
mkdir -p "${LOG_DIR}"

# Build the optional timeout args once. Empty string -> arg not passed,
# the launch file's own default applies.
EXTRA_ARGS=()
[[ -n "${PER_GOAL_TIMEOUT}" ]] && EXTRA_ARGS+=("per_goal_timeout_s:=${PER_GOAL_TIMEOUT}")
[[ -n "${EPISODE_TIMEOUT}"  ]] && EXTRA_ARGS+=("episode_timeout_s:=${EPISODE_TIMEOUT}")

launch_variant() {  # launch_variant <variant>
  local v="$1" fam rest hp ch
  local tcmd=()
  [[ "${LAUNCH_TIMEOUT}" != "0" ]] && tcmd=(timeout --signal=SIGINT "${LAUNCH_TIMEOUT}")
  if is_trained "$v"; then
    fam="${v%%_*}"; rest="${v#*_}"; hp="${rest%_*}"; ch="${rest##*_}"
    "${tcmd[@]}" ros2 launch vf_robot_utils \
        "vf_data_evaluation_batch_${fam}.launch.py" \
        map:="${MAP}" planner:="${PLANNER}" run_id:="${RUN}" \
        hp:="${hp}" ch:="${ch}" headless:="${HEADLESS}" rviz:="${RVIZ}" \
        "${EXTRA_ARGS[@]}"
  else
    "${tcmd[@]}" ros2 launch vf_robot_utils \
        "vf_data_evaluation_batch_${v}.launch.py" \
        map:="${MAP}" planner:="${PLANNER}" run_id:="${RUN}" \
        headless:="${HEADLESS}" rviz:="${RVIZ}" \
        "${EXTRA_ARGS[@]}"
  fi
}

# ---- sweep -------------------------------------------------------------
SUMMARY=()
i=0
for v in "${VARIANTS[@]}"; do
  i=$((i + 1))
  before="$(count_h5 "$v")"
  echo
  echo "############################################################"
  echo "[sweep] (${i}/${#VARIANTS[@]}) ${v}   (had ${before} h5)"
  echo "############################################################"
  log="${LOG_DIR}/${v}.log"
  if launch_variant "$v" 2>&1 | tee "${log}"; then rc=0; else rc=$?; fi
  after="$(count_h5 "$v")"
  gained=$((after - before))
  if   [[ $gained -ge 1 ]]; then status="OK (+${gained} h5)"
  elif [[ $rc -eq 124   ]]; then status="TIMEOUT (no new h5)"
  else                          status="FAIL rc=${rc} (no new h5)"
  fi
  SUMMARY+=("$(printf '  %-26s %-12s %s' "$v" "${before}->${after}" "$status")")
  echo "[sweep] ${v}: ${status}"
  [[ $i -lt ${#VARIANTS[@]} ]] && sleep "${SLEEP_BETWEEN}"
done

# ---- summary -----------------------------------------------------------
echo
echo "============================================================"
echo "[sweep] cohort=${COHORT} done. Per-variant result:"
echo "------------------------------------------------------------"
printf "  %-26s %-12s %s\n" "VARIANT" "h5 n" "STATUS"
for line in "${SUMMARY[@]}"; do echo "$line"; done
echo "------------------------------------------------------------"
echo "[sweep] logs: ${LOG_DIR}/"
echo "[sweep] next: bash src/vf_robot_utils/tools/run_thesis_all.sh"
