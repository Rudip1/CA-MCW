#!/usr/bin/env bash
#
# Copyright  EUROKNOWS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Pravin Oli
# https://www.euroknows.com/en/home/
#
# xacro_to_sdf.sh
# ────────────────
# Resolves xacro → URDF, then converts to SDF for Gazebo simulation.
#
# Run from inside the urdf/xacro/ folder:
#   ./xacro_to_sdf.sh uvc1_virofighter.xacro
#
# Folder depth (NEW layout):
#   urdf/xacro/ → urdf/ → vf_robot_description/ → src/ → CA-MCW/
#   4 levels up to workspace root
#
# Pipeline:
#   Step 1 — Xacro resolution  (ALWAYS — this script only accepts .xacro)
#   Step 2 — URDF validation   (check_urdf on the resolved output)
#   Step 3 — URDF → raw SDF   (gz sdf -p)
#   Step 4 — Post-processing:
#             4a. Fix mesh URIs  (model://anything → model://uvc1_common)
#             4b. Auto-extract ALL <gazebo> plugins from resolved URDF
#                 NOTE: plugins are extracted from the RESOLVED file so
#                 that xacro:macro expansions (fisheye, ultrasonic) are
#                 fully present — not just the raw .xacro source.
#             4c. Inject plugins before </model>
#             4d. Add <?xml version="1.0"?> declaration
#   Step 5 — Backup + save → vf_robot_gazebo/models/uvc1_virofighter/model.sdf
#   Step 6 — Diff old vs new
#
# Single source of truth:
#   xacro is the master — sensors.xacro + common_properties.xacro are
#   included automatically. Edit any xacro file → re-run → SDF in sync.
#   Never edit model.sdf manually.

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
ask()   { echo -e "${CYAN}[ASK]${NC}   $*"; }

usage() {
    echo ""
    echo -e "  ${CYAN}Usage:${NC}   ./xacro_to_sdf.sh <xacro_filename>"
    echo -e "  ${CYAN}Example:${NC} ./xacro_to_sdf.sh uvc1_virofighter.xacro"
    echo ""
    echo "  Run from inside the urdf/xacro/ directory."
    echo "  Only .xacro files are accepted — use urdf_to_sdf.sh for plain URDF."
    echo ""
    exit 1
}

[ -z "$1" ] && { warn "No xacro filename provided."; usage; }

# Guard: only accept .xacro files
[[ "$1" == *.xacro ]] || error "Input must be a .xacro file. Got: $1
  Use urdf/urdf/urdf_to_sdf.sh for plain URDF files."

# ── paths ─────────────────────────────────────────────────────────────────────
# urdf/xacro/xacro_to_sdf.sh
# depth: urdf/xacro → urdf → vf_robot_description → src → CA-MCW
#                    ^^^^   ^^^^^^^^^^^^^^^^^^^^   ^^^   ^^^^^^^^^^^^
#                    +1          +2                +3       +4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XACRO_SRC="${SCRIPT_DIR}/$1"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"    # ← 4 levels up
SDF_DEST_DIR="${WORKSPACE_ROOT}/src/vf_robot_gazebo/models/uvc1_virofighter"
SDF_DEST="${SDF_DEST_DIR}/model.sdf"
MODEL_NAME="uvc1_common"

echo ""
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info " Xacro → SDF Converter"
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info " Script dir : ${SCRIPT_DIR}"
info " Workspace  : ${WORKSPACE_ROOT}"
info " Source     : ${XACRO_SRC}"
info " Destination: ${SDF_DEST}"
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_and_install() {
    local TOOL="$1" PKG="$2" LABEL="$3"
    if ! command -v "${TOOL}" &>/dev/null; then
        warn "${LABEL} ('${TOOL}') is NOT installed."
        ask "Install ${PKG} now? [y/N]: "
        read -r REPLY; echo ""
        if [[ "${REPLY}" =~ ^[Yy]$ ]]; then
            sudo apt-get update -qq && sudo apt-get install -y "${PKG}"
            info "${LABEL} installed ✓"
        else
            warn "Skipping ${LABEL}."; return 1
        fi
    else
        info "${LABEL} found ✓  ($(command -v ${TOOL}))"
    fi
    return 0
}

info "── Checking dependencies ──────────────────────────"
check_and_install "xacro"      "ros-humble-xacro" "xacro"      || error "xacro is required for this script."
check_and_install "check_urdf" "liburdfdom-tools"  "check_urdf" || CHECKURDF_MISSING=1
check_and_install "python3"    "python3"           "python3"    || error "python3 is required."

GZ_CMD=""
if   command -v gz  &>/dev/null; then GZ_CMD="gz";  info "Gazebo CLI found ✓  (gz)"
elif command -v ign &>/dev/null; then GZ_CMD="ign"; info "Ignition CLI found ✓  (ign)"
else
    warn "Neither 'gz' nor 'ign' found."
    echo "  [1] Gazebo Garden/Harmonic  →  gz-tools2"
    echo "  [2] Ignition Fortress       →  ignition-fortress"
    echo "  [3] Abort"
    ask "Choice [1/2/3]: "; read -r GZ_CHOICE; echo ""
    case "${GZ_CHOICE}" in
        1) sudo apt-get update -qq && sudo apt-get install -y gz-tools2;        GZ_CMD="gz"  ;;
        2) sudo apt-get update -qq && sudo apt-get install -y ignition-fortress; GZ_CMD="ign" ;;
        *) error "Gazebo CLI required. Aborting." ;;
    esac
    info "Gazebo CLI installed ✓  (${GZ_CMD})"
fi
echo ""

[ -f "${XACRO_SRC}" ]    || error "Xacro file not found: ${XACRO_SRC}"
[ -d "${SDF_DEST_DIR}" ] || error "Gazebo model dir not found: ${SDF_DEST_DIR}"

ROS_SETUP="/opt/ros/humble/setup.bash"
[ -f "${ROS_SETUP}" ] && source "${ROS_SETUP}" || warn "ROS Humble setup.bash not found."
WS_SETUP="${WORKSPACE_ROOT}/install/setup.bash"
[ -f "${WS_SETUP}" ]  && source "${WS_SETUP}"  || warn "Workspace install/setup.bash not sourced."

# ── Step 1 — Xacro resolution (ALWAYS for this script) ───────────────────────
RESOLVED_URDF="/tmp/vf_xacro_resolved_$$.urdf"
info "── Step 1 : Xacro resolution ───────────────────────"
info "Resolving: ${XACRO_SRC}"
info "Includes:  sensors.xacro + common_properties.xacro (auto via xacro:include)"
xacro "${XACRO_SRC}" -o "${RESOLVED_URDF}"
info "Xacro resolved ✓  →  ${RESOLVED_URDF}"
# The resolved file is used for BOTH gz sdf conversion AND plugin extraction.
# This is critical: xacro:macro blocks (fisheye cameras, ultrasonics) are
# only visible in the resolved output, not in the raw .xacro source.
echo ""

# ── Step 2 — Validate resolved URDF ──────────────────────────────────────────
info "── Step 2 : URDF validation ────────────────────────"
if [ -z "${CHECKURDF_MISSING}" ]; then
    check_urdf "${RESOLVED_URDF}" && info "URDF is valid ✓" || warn "Warnings above — continuing"
else
    warn "Skipping (check_urdf not installed)"
fi
echo ""

# ── Step 3 — URDF → raw SDF ──────────────────────────────────────────────────
TEMP_SDF="/tmp/vf_xacro_raw_$$.sdf"
info "── Step 3 : Converting resolved URDF → SDF ────────"
${GZ_CMD} sdf -p "${RESOLVED_URDF}" > "${TEMP_SDF}" || error "SDF conversion failed."
[ -s "${TEMP_SDF}" ] || error "Converted SDF is empty."
info "Raw SDF generated ✓"
echo ""

# ── Step 4 — Post-processing ──────────────────────────────────────────────────
PATCHED_SDF="/tmp/vf_xacro_patched_$$.sdf"
info "── Step 4 : Post-processing SDF ───────────────────"
info "  4a. Fix mesh URIs        → model://${MODEL_NAME}/meshes/..."
info "  4b. Extract plugins from resolved URDF <gazebo> tags"
info "      (resolved = all xacro:macro expansions fully present)"
info "  4c. Inject plugins before </model>"
info "  4d. Add XML declaration"

python3 - "${TEMP_SDF}" "${PATCHED_SDF}" "${MODEL_NAME}" "${RESOLVED_URDF}" << 'PYEOF'
import sys, re
import xml.etree.ElementTree as ET

raw_sdf    = sys.argv[1]
out_sdf    = sys.argv[2]
model_name = sys.argv[3]
urdf_path  = sys.argv[4]   # resolved URDF — macros already expanded

with open(raw_sdf) as f:
    content = f.read()

# 4a — fix mesh URIs
content = re.sub(r'model://[^/]+/meshes/', f'model://{model_name}/meshes/', content)
print("[INFO]  Mesh URIs fixed ✓")

# 4b — extract plugins from resolved URDF
# Using resolved URDF (not raw .xacro) so xacro:macro-generated <gazebo>
# tags (fisheye cameras, ultrasonics) are all present and extractable.
try:
    with open(urdf_path) as pf:
        urdf_xml = pf.read().replace('\r\n', '\n').replace('\r', '\n')
    root = ET.fromstring(urdf_xml)

    plugin_lines = [
        "\n    <!-- ==========================================",
        "         AUTO-EXTRACTED FROM XACRO (resolved)",
        "         Source: urdf/xacro/xacro_to_sdf.sh",
        "         Do not edit model.sdf manually.",
        "         Edit xacro files and re-run xacro_to_sdf.sh",
        "         ========================================== -->\n"
    ]
    extracted = 0
    for gz in root.findall('gazebo'):
        ref = gz.get('reference', '')
        for child in gz:
            if ref:
                plugin_lines.append(f'    <!-- ref: {ref} -->')
            plugin_lines.append(f'    {ET.tostring(child, encoding="unicode")}')
            extracted += 1

    plugin_block = '\n'.join(plugin_lines) + '\n'
    print(f"[INFO]  Extracted {extracted} block(s) from <gazebo> tags ✓")

except ET.ParseError as e:
    print(f"[ERROR] Failed to parse resolved URDF: {e}", file=sys.stderr); sys.exit(1)

# 4c — inject plugins
if 'libgazebo_ros_diff_drive.so' not in content:
    pos = content.rfind('</model>')
    if pos == -1:
        print("[ERROR] </model> not found in SDF", file=sys.stderr); sys.exit(1)
    content = content[:pos] + plugin_block + content[pos:]
    print("[INFO]  Plugins injected ✓")
else:
    print("[INFO]  Plugins already present — skipping (idempotent) ✓")

# 4d — XML declaration
if not content.strip().startswith('<?xml'):
    content = '<?xml version="1.0"?>\n' + content
    print("[INFO]  XML declaration added ✓")

with open(out_sdf, 'w') as f:
    f.write(content)
print("[INFO]  Post-processing complete ✓")
PYEOF

[ -s "${PATCHED_SDF}" ] || error "Post-processed SDF is empty."
echo ""

# ── Step 5 — Backup + save ────────────────────────────────────────────────────
info "── Step 5 : Saving to Gazebo model folder ──────────"
[ -f "${SDF_DEST}" ] && { cp "${SDF_DEST}" "${SDF_DEST}.bak"; info "Backed up → model.sdf.bak"; }
cp "${PATCHED_SDF}" "${SDF_DEST}"
info "model.sdf saved ✓  →  ${SDF_DEST}"
echo ""

# ── Step 6 — Diff ─────────────────────────────────────────────────────────────
if [ -f "${SDF_DEST}.bak" ]; then
    info "── Step 6 : Diff (old vs new) ──────────────────────"
    DIFF_LINES=$(diff "${SDF_DEST}.bak" "${SDF_DEST}" 2>/dev/null | wc -l || echo 0)
    if [ "${DIFF_LINES}" -eq 0 ]; then
        info "No changes — SDF identical to previous."
    else
        info "${DIFF_LINES} lines changed:"
        diff --color=always "${SDF_DEST}.bak" "${SDF_DEST}" || true
    fi
    echo ""
fi

rm -f "${RESOLVED_URDF}" "${TEMP_SDF}" "${PATCHED_SDF}"

info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info " ✓  All done!"
info "   Output → ${SDF_DEST}"
info ""
info "   Workflow:"
info "   1. Edit any xacro file in urdf/xacro/"
info "   2. cd urdf/xacro && ./xacro_to_sdf.sh uvc1_virofighter.xacro"
info "   3. Never edit model.sdf manually"
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
