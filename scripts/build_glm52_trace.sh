#!/usr/bin/env bash
# build_glm52_trace.sh — emit a 5-layer SUBSET GLM decode pkt (L0-2 dense, L3-4 MoE) sharded for
# a given TP, build the block-fp8 decode interp + the trace-instrumented glm52_decode host, so a
# ctx x TP trace sweep can dump per-op PlowTraceRec.  READ-ONLY tracing: no kernel changes.
#
#   nix develop -c ./scripts/build_glm52_trace.sh <TP> [CTX] [OUT]
# defaults: CTX=131072 (128k)  OUT=/home/lava/models/glm52_trace2
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
TP="${1:-1}"; CTX="${2:-131072}"; OUT="${3:-/home/lava/models/glm52_trace2}"
PREP="${PLOW_PREP_DIR:-/home/lava/models/GLM-5.2-FP8-plow}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-/opt/rocm-7.0.2/lib/llvm/bin/clang-offload-bundler}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"; cd "$REPO"

cargo build --quiet -p plowc --bin plowc
env GLM_FULL=1 PLOW_FP8=1 GLM_NLAYERS=5 \
    "$REPO/target/debug/plowc" --hf-dir "$PREP" --emit devblob --max-ctx "$CTX" --n-cu 256 \
    --num-gpus "$TP" --out "$OUT/glm52_sub_tp${TP}.pkt"

cd "$OUT"
if [ ! -f interp_decode.elf ]; then
  hipcc --offload-arch="$ARCH" -O3 -w -DPLOW_BUCKET_DECODE=1 --genco "$R/amd/interp.hip" -o i_decode.co $INC
  "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input=i_decode.co --output=interp_decode.elf
fi
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o glm52_decode \
    "$R/tests/glm52_decode.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d glm52_decode | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }
ls -l --time-style=+%H:%M:%S "glm52_sub_tp${TP}.pkt" interp_decode.elf glm52_decode | awk '{print "   ",$NF,$5"B",$6}'
echo "OK — subset TP${TP} pkt (ctx=${CTX}) + interp + harness in $OUT"
