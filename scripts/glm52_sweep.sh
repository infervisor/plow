#!/usr/bin/env bash
# glm52_sweep.sh — GLM-5.2-FP8 plow DECODE sweep (TP x ctx) -> perf-data/glm52-plow-decode.json.
#
# Loads the emitter's FULL-MODEL .pkt + the host-prepped weight dir (scripts/glm52_prep_full.py),
# runs 1-token decode at each TP degree across the ctx list, takes the median ms/tok (TPOT), and
# writes the JSON in the SAME schema as perf-data/glm52-vllm-decode.json (engine=plow) via
# glm52_sweep_json.py so both slot into consolidate_perf.py.
#
# Mirrors scripts/tp_decode_sweep.sh: ONE decode process per TP degree (binds that many GPUs once,
# then sweeps every ctx via the harness's --sweep), clean pinned env under `sg render`.
#
#   scripts/glm52_sweep.sh <build-dir> <model.pkt> <prep-dir> [tp-list] [ctx-list]
# defaults: tp-list "4 8"   ctx-list "1k,4k,8k,16k,32k,64k,128k"
#
# The decode harness binary is the emitter's full-model GLM decoder (env GLM_DECODE, default
# glm52_decode) built into <build-dir> alongside interp_decode.elf. Until it lands, test the parse +
# JSON path with a canned SWEEP block:  echo '<sweep text>' | GLM_DECODE=cat scripts/glm52_sweep.sh ...
# or run scripts/glm52_sweep_selftest.sh.
#
# GPU COORDINATION: TP8 = the whole node. Do NOT launch the TP8 point without messaging main first
# (the node is a single shared resource — bring-up/sweep and any baseline run SEQUENTIALLY).
set -euo pipefail

DIR="${1:?build dir with the GLM decode harness + interp_decode.elf}"
PKT="${2:?full-model model.pkt}"
PREP="${3:?prepped weight dir (GLM-5.2-FP8-plow)}"
TPS="${4:-4 8}"
CTXS="${5:-1k,4k,8k,16k,32k,64k,128k}"
STEPS="${STEPS:-21}"
GLM_DECODE="${GLM_DECODE:-glm52_decode}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTJSON="${OUTJSON:-$REPO/perf-data/glm52-plow-decode.json}"
VERSION="${VERSION:-plow @ $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo glm52-prep) $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '')}"

echo "GLM-5.2 plow decode sweep — TP: $TPS  ctx: $CTXS  (median of $STEPS)"
echo "  harness=$GLM_DECODE  pkt=$PKT  prep=$PREP  -> $OUTJSON"

ROWS="$(mktemp)"; trap 'rm -f "$ROWS"' EXIT
for tp in $TPS; do
  echo "============================================================"
  echo "TP=$tp$([ "$tp" = 8 ] && echo '   [whole-node — coordinate with main before launching]')"
  # Capture the harness SWEEP block; parse "  <ctx> <ms/tok> <tok/s>" data rows into "tp ctx tpot".
  sg render -c "cd $DIR && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME \
    LD_LIBRARY_PATH=/opt/rocm/lib ${DEVS:+HIP_VISIBLE_DEVICES=$DEVS} ./$GLM_DECODE $PKT $PREP --tp $tp \
    --sweep $CTXS --steps $STEPS" 2>&1 | tee "/dev/stderr" \
    | awk -v tp="$tp" '
        /^  *[0-9]+ +[0-9.]+ +[0-9.]+/ { print tp, $1, $2 }
      ' >> "$ROWS"
done

echo "============================================================"
if [ -s "$ROWS" ]; then
  VERSION="$VERSION" python3 "$REPO/scripts/glm52_sweep_json.py" --out "$OUTJSON" --version "$VERSION" < "$ROWS"
else
  echo "no SWEEP rows parsed — harness produced no data (scaffold stand-in? check $GLM_DECODE output)"; exit 1
fi
