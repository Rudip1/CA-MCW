#!/usr/bin/env bash
# sync_to_thesis.sh -- copy evaluation artefacts into the thesis tree.
# [PUBLISH] run last, after run_comparison.sh / run_comparison_all.sh.
# Copy-only: never deletes thesis content, never edits LaTeX.
#
#   Figures : _aggregate/<stem>/comparison{,_all}/  (PDF tree, recursive)
#             ->  figures/ch5/comparison{,_all}/
#   Context : results.csv + controller_selection.csv + the comparison
#             summary CSVs + evaluation_plots.txt
#             ->  references/docs/  -- the bundle read when writing the
#             Results chapter.
#   Manifest: references/docs/SNAPSHOT.md  (sync date, git SHA,
#             per-controller episode count).
#
# Env (defaults):
#   MAP=house_my1_map  STEM=thesis_eval
#   THESIS_ROOT=<workspace>/ELTE_IK_IFRoS_Thesis_Pravin-Oli

set -eo pipefail

MAP="${MAP:-house_my1_map}"
STEM="${STEM:-thesis_eval}"

cd "$(dirname "$0")/../../.."
WS="$(pwd)"
THESIS_ROOT="${THESIS_ROOT:-${WS}/ELTE_IK_IFRoS_Thesis_Pravin-Oli}"
AGG="vf_data/vf_data_evaluation/batch/${MAP}/_aggregate/${STEM}"

[[ -d "${AGG}" ]] || { echo "[sync] no ${AGG} -- run run_comparison.sh first" >&2; exit 1; }
[[ -d "${THESIS_ROOT}" ]] || { echo "[sync] no thesis at ${THESIS_ROOT}" >&2; exit 1; }

DOCS="${THESIS_ROOT}/references/docs"
mkdir -p "${DOCS}"

# ---- figures: comparison{,_all} PDF trees -> figures/ch5/ --------------
nfig=0
for scope in comparison comparison_all; do
  [[ -d "${AGG}/${scope}" ]] || continue
  dst="${THESIS_ROOT}/figures/ch5/${scope}"
  while IFS= read -r -d '' pdf; do
    rel="${pdf#${AGG}/${scope}/}"
    mkdir -p "${dst}/$(dirname "${rel}")"
    cp -f "${pdf}" "${dst}/${rel}"
    nfig=$((nfig + 1))
    echo "  fig  ch5/${scope}/${rel}"
  done < <(find "${AGG}/${scope}" -name '*.pdf' -print0)
done

# ---- context bundle -> references/docs ---------------------------------
ncsv=0
for f in "${AGG}/results.csv" "${AGG}/controller_selection.csv" \
         "${AGG}/comparison/comparison_summary.csv" \
         "${AGG}/comparison_all/comparison_summary.csv"; do
  [[ -e "$f" ]] || continue
  # Prefix the scope so the two comparison_summary.csv files don't clash.
  case "$f" in
    */comparison/*)     cp -f "$f" "${DOCS}/comparison_summary.csv" ;;
    */comparison_all/*) cp -f "$f" "${DOCS}/comparison_all_summary.csv" ;;
    *)                  cp -f "$f" "${DOCS}/" ;;
  esac
  ncsv=$((ncsv + 1))
done
# the figure catalog -- single source of truth is the package root
cp -f src/vf_robot_utils/evaluation_plots.txt "${DOCS}/" 2>/dev/null || true

# ---- snapshot manifest -------------------------------------------------
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
{
  echo "# Evaluation snapshot"
  echo
  echo "- Synced: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "- Workspace git SHA: ${SHA}"
  echo "- Map: ${MAP}    Stem: ${STEM}"
  echo "- Source: ${AGG}"
  echo "- Figures copied: ${nfig}    CSV copied: ${ncsv}"
  echo
  echo "This directory is the canonical context bundle for writing the"
  echo "thesis Results chapter. CSVs hold the numbers, evaluation_plots.txt"
  echo "is the per-figure catalog."
  echo
  echo "## Per-controller episode count (results.csv)"
  echo
  if [[ -f "${AGG}/results.csv" ]]; then
    python3 - "${AGG}/results.csv" <<'PY'
import csv, sys, collections
n = collections.Counter()
with open(sys.argv[1], newline="") as fh:
    for r in csv.DictReader(fh):
        n[r.get("controller", "?")] += 1
for c in sorted(n):
    print(f"- {c}: {n[c]}")
PY
  else
    echo "- (results.csv not found)"
  fi
} > "${DOCS}/SNAPSHOT.md"

echo
echo "[sync] ${nfig} figures -> figures/ch5/comparison{,_all}/"
echo "[sync] ${ncsv} CSV     -> references/docs/"
echo "[sync] manifest        -> references/docs/SNAPSHOT.md"
echo "[sync] done. Context bundle for thesis writing: ${DOCS}/"
