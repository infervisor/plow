#!/usr/bin/env bash
# build_glm52_decode.sh — build the GLM-5.2-FP8 multi-GPU TP decode harness (glm52_decode.c) +
# the block-fp8 decode interpreter, and emit the full-model sharded .pkt for a given TP degree.
#
#   nix develop -c ./scripts/build_glm52_decode.sh <OUT-dir> [TP] [NLAYERS] [CTX]
# defaults: OUT=/home/lava/models/glm52_tp  TP=4  NLAYERS=all(78)  CTX=4096
#
# Emits <OUT>/glm52_tp<TP>.pkt (GLM_FULL=1, PLOW_FP8=1, --tp TP), builds interp_decode.elf
# (block-fp8 expert arms 45/46 + XREDUCE 24) and the glm52_decode host binary.
#
#   run (TP4, GPUs 4-7):
#   sg render -c 'cd <OUT> && /usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME LD_LIBRARY_PATH=/opt/rocm/lib \
#       HIP_VISIBLE_DEVICES=4,5,6,7 ./glm52_decode glm52_tp4.pkt /home/lava/models/GLM-5.2-FP8-plow \
#       --tp 4 --gen 24'
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-/home/lava/models/glm52_tp}"
TP="${2:-4}"
NLAYERS="${3:-}"                      # empty = all 78
CTX="${4:-4096}"
PREP="${PLOW_PREP_DIR:-/home/lava/models/GLM-5.2-FP8-plow}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
# Discover the bundler from the INSTALLED ROCm instead of pinning a version. This
# was pinned to 7.0.2 and the path does not exist on a 7.2.4 box, so the build died
# on the first machine that had the GPU. $PLOW_BUNDLER still overrides.
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"; cd "$REPO"

# 1) emit the sharded full-model .pkt (block-fp8) from the serving emitter.
cargo build --quiet -p plowc --bin plowc
NLENV=(); [ -n "$NLAYERS" ] && NLENV=(GLM_NLAYERS="$NLAYERS")
env GLM_FULL=1 PLOW_FP8=1 "${NLENV[@]}" \
    "$REPO/target/debug/plowc" --hf-dir "$PREP" --emit devblob --max-ctx "$CTX" --n-cu 256 \
    --num-gpus "$TP" --out "$OUT/glm52_tp${TP}.pkt"

cd "$OUT"
rm -f i_decode.co interp_decode.elf glm52_decode

# 2) decode interpreter (block-fp8 expert arms 45/46 + XREDUCE 24, unconditional).
hipcc --offload-arch="$ARCH" -O3 -w -DPLOW_BUCKET_DECODE=1 --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input=i_decode.co --output=interp_decode.elf

# 3) host harness (system gcc, clean env — nix gcc RUNPATH aborts at load).
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o glm52_decode \
    "$R/tests/glm52_decode.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d glm52_decode | grep -qi runpath && { echo "FAIL: RUNPATH leaked into host binary"; exit 1; }

ls -l --time-style=+%H:%M:%S "glm52_tp${TP}.pkt" interp_decode.elf glm52_decode \
  | awk '{print "   ", $NF, $5"B", $6}'
echo "OK — TP${TP} pkt + interp + harness in $OUT"
