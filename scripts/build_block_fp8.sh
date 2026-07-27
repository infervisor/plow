#!/usr/bin/env bash
# Build the block-fp8 (DeepSeek/GLM weight_block_size [128,128]) decode-GEMV test:
# test_kernels.elf (with the gemv_fp8_blk wrapper) + the host harness. Run under `sg render`.
#
#   nix develop --command bash -c './scripts/build_block_fp8.sh /tmp/blkfp8'
#   sg render -c 'cd /tmp/blkfp8 && /usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME \
#       LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=7 ./block_fp8_test test_kernels.elf'
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/blkfp8}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
# Discover the bundler from the INSTALLED ROCm instead of pinning a version. This
# was pinned to 7.0.2 and the path does not exist on a 7.2.4 box, so the build died
# on the first machine that had the GPU. $PLOW_BUNDLER still overrides.
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"; cd "$OUT"

rm -f tk.co test_kernels.elf block_fp8_test

# Device: the golden wrappers (share op_gemm.h with the interpreter).
hipcc --offload-arch="$ARCH" -O3 -w --genco "$R/amd/test_kernels.hip" -o tk.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input=tk.co --output=test_kernels.elf

# Register-usage report for the new block-fp8 GEMV (must stay on the decode budget, occ>=2).
U=$(hipcc --offload-arch="$ARCH" -O3 -w -Rpass-analysis=kernel-resource-usage \
      --genco "$R/amd/test_kernels.hip" -o /dev/null $INC 2>&1 | grep -A6 'gemv_fp8_blk' || true)
echo "--- gemv_fp8_blk resource usage ---"
echo "$U" | grep -E 'VGPRs|AGPRs|Occupancy|Spill|SGPRs' || echo "  (usage lines not captured)"
echo "-----------------------------------"

# Host harness with the SYSTEM gcc in a CLEAN env (nix gcc bakes a RUNPATH that aborts at load).
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o block_fp8_test \
    "$R/tests/block_fp8_gfx950_test.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d block_fp8_test | grep -qi runpath && { echo "FAIL: RUNPATH leaked into host binary"; exit 1; }

ls -l --time-style=+%H:%M:%S test_kernels.elf block_fp8_test | awk '{print "   ", $NF, $5"B", $6}'
echo "OK — run: sg render -c 'cd $OUT && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=7 ./block_fp8_test test_kernels.elf'"
