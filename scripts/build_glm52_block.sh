#!/usr/bin/env bash
# Build the GLM-5.2 (GlmMoeDsa) single-block B1 de-risk: interp_decode.elf (bf16 MLA+MoE ops +
# block-fp8 experts 45/46), the HF-transformers ORACLE fixture (glm52_oracle.py), and the host
# harness (glm52_block_gfx950_test.c). Run under `sg render` on ONE GPU.   [GLM52-B1]
#
#   nix develop --command bash -c './scripts/build_glm52_block.sh /tmp/glm52b'
#   sg render -c 'cd /tmp/glm52b && /usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME \
#       LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=6 ./glm52_test interp_decode.elf glm52_fixture.bin'
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/glm52b}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-/opt/rocm-7.0.2/lib/llvm/bin/clang-offload-bundler}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"; cd "$OUT"

rm -f i_decode.co interp_decode.elf glm52_test

# Decode interpreter (bf16 bucket): carries GEMV/GEMV_GLU/RMSNORM/RESIDUAL, the MLA read path
# (FLASH_MLA_DECODE/FLASH_MERGE/O_UV_FOLD), MoE router+experts+combine, AND the block-fp8 expert
# arms (45/46, dispatched unconditionally — no PLOW_FP8 needed for those).
hipcc --offload-arch="$ARCH" -O3 -w -DPLOW_BUCKET_DECODE=1 --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input=i_decode.co --output=interp_decode.elf

# HF oracle fixture (synthetic seeded bf16 weights, real dims; RAM-forced E=16, L<=2048 dense).
if [ "${GLM52_SKIP_ORACLE:-0}" != 1 ]; then
  python3 "$R/tests/glm52_oracle.py" glm52_fixture.bin
fi

# Host harness with the SYSTEM gcc in a CLEAN env (nix gcc bakes a RUNPATH that aborts at load).
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o glm52_test \
    "$R/tests/glm52_block_gfx950_test.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d glm52_test | grep -qi runpath && { echo "FAIL: RUNPATH leaked into host binary"; exit 1; }

ls -l --time-style=+%H:%M:%S interp_decode.elf glm52_test glm52_fixture.bin 2>/dev/null \
  | awk '{print "   ", $NF, $5"B", $6}'
echo "OK — run: sg render -c 'cd $OUT && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=6 ./glm52_test interp_decode.elf glm52_fixture.bin'"
