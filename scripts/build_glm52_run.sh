#!/usr/bin/env bash
# build_glm52_run.sh — MILESTONE-1 CLOSE build: emit the GLM-5.2 (GlmMoeDsa) single-layer .pkt from
# the serving emitter, build the decode interpreter (bf16 MLA/MoE + block-fp8 experts 45/46) and the
# ms1 loader glm52_run.c. Drive the EMITTED pkt against the HF fixture with the host-prepped weights.
#
#   nix develop -c ./scripts/build_glm52_run.sh /home/lava/models/glm52_run
#   # weights (once): nix develop -c python3 scripts/glm52_prep.py --out /home/lava/models/glm52_prep
#   # fixture (once): built by glm52_real_oracle.py (reused: /home/lava/models/glm52b4/glm52_real_fixture.bin)
#   sg render -c 'cd <OUT> && /usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME LD_LIBRARY_PATH=/opt/rocm/lib \
#       HIP_VISIBLE_DEVICES=0 ./glm52_run glm52_bf16.pkt /home/lava/models/glm52_prep <fixture> 3'
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-/home/lava/models/glm52_run}"
PREP="${PLOW_PREP_DIR:-/home/lava/models/glm52_prep}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
# Discover the bundler from the INSTALLED ROCm instead of pinning a version. This
# was pinned to 7.0.2 and the path does not exist on a 7.2.4 box, so the build died
# on the first machine that had the GPU. $PLOW_BUNDLER still overrides.
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
CTX="${GLM_CTX:-512}"
mkdir -p "$OUT"; cd "$REPO"

# 1) emit the single-layer .pkt (bf16 + block-fp8) from the serving emitter (glm_main dispatch).
cargo build --quiet -p plowc --bin plowc
"$REPO/target/debug/plowc" --hf-dir "$PREP" --emit devblob --max-ctx "$CTX" --n-cu 256 --out "$OUT/glm52_bf16.pkt"
PLOW_FP8=1 "$REPO/target/debug/plowc" --hf-dir "$PREP" --emit devblob --max-ctx "$CTX" --n-cu 256 --out "$OUT/glm52_fp8.pkt"

cd "$OUT"
rm -f i_decode.co interp_decode.elf glm52_run

# 2) decode interpreter (block-fp8 expert arms 45/46 dispatched unconditionally).
hipcc --offload-arch="$ARCH" -O3 -w -DPLOW_BUCKET_DECODE=1 --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input=i_decode.co --output=interp_decode.elf

# 3) ms1 loader (system gcc, clean env — nix gcc RUNPATH aborts at load).
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o glm52_run \
    "$R/tests/glm52_run.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d glm52_run | grep -qi runpath && { echo "FAIL: RUNPATH leaked into host binary"; exit 1; }

ls -l --time-style=+%H:%M:%S glm52_bf16.pkt glm52_fp8.pkt interp_decode.elf glm52_run \
  | awk '{print "   ", $NF, $5"B", $6}'
echo "OK — run (bf16): sg render -c 'cd $OUT && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=0 ./glm52_run glm52_bf16.pkt $PREP <fixture> 3'"
