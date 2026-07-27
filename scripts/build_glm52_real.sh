#!/usr/bin/env bash
# Build the GLM-5.2 (GlmMoeDsa) REAL-WEIGHT single-layer B4 de-risk: interp_decode.elf (bf16 MLA+MoE
# ops + block-fp8 experts 45/46) + the real-weight host harness (glm52_real_block_gfx950_test.c).
# The fixture (glm52_real_oracle.py) is ~10 GB and built separately (needs the real GLM-5.2-FP8
# weights). Run under `sg render` on ONE GPU (needs ~29 GB device RAM for 256 fp8+bf16 experts). [B4]
#
#   nix develop --command bash -c './scripts/build_glm52_real.sh /home/lava/models/glm52b4'
#   # fixture (once):  nix develop -c python3 runtime/tests/glm52_real_oracle.py <OUT>/glm52_real_fixture.bin
#   sg render -c 'cd <OUT> && /usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME \
#       LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=1 ./glm52_real_test interp_decode.elf glm52_real_fixture.bin'
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/home/lava/models/glm52b4}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
# Discover the bundler from the INSTALLED ROCm instead of pinning a version. This
# was pinned to 7.0.2 and the path does not exist on a 7.2.4 box, so the build died
# on the first machine that had the GPU. $PLOW_BUNDLER still overrides.
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"; cd "$OUT"

rm -f i_decode.co interp_decode.elf glm52_real_test

# Decode interpreter (bf16 bucket + block-fp8 expert arms 45/46, dispatched unconditionally).
hipcc --offload-arch="$ARCH" -O3 -w -DPLOW_BUCKET_DECODE=1 --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input=i_decode.co --output=interp_decode.elf

# Host harness with SYSTEM gcc in a CLEAN env (nix gcc RUNPATH aborts at load).
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o glm52_real_test \
    "$R/tests/glm52_real_block_gfx950_test.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d glm52_real_test | grep -qi runpath && { echo "FAIL: RUNPATH leaked into host binary"; exit 1; }

ls -l --time-style=+%H:%M:%S interp_decode.elf glm52_real_test glm52_real_fixture.bin 2>/dev/null \
  | awk '{print "   ", $NF, $5"B", $6}'
echo "OK — run: sg render -c 'cd $OUT && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=1 ./glm52_real_test interp_decode.elf glm52_real_fixture.bin'"
