#!/usr/bin/env bash
# Build the single-block DSA sparse-attention lever bench (runtime/bench/dsa_gather_bench.c):
# test_kernels.elf (indexer score + G4 radix select + gather/dense MLA) + the host harness.  [DSA/G5]
#
#   nix develop --command bash -c './scripts/build_dsa_bench.sh /home/lava/models/dsa_bench'
#   sg render -c 'cd <OUT> && /usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME \
#       LD_LIBRARY_PATH=/opt/rocm/lib ROCR_VISIBLE_DEVICES=0 ./dsa_bench test_kernels.elf'
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/home/lava/models/dsa_bench}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
# Discover the bundler from the INSTALLED ROCm instead of pinning a version. This
# was pinned to 7.0.2 and the path does not exist on a 7.2.4 box, so the build died
# on the first machine that had the GPU. $PLOW_BUNDLER still overrides.
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"; cd "$OUT"
rm -f tk.co test_kernels.elf dsa_bench

hipcc --offload-arch="$ARCH" -O3 -w --genco "$R/amd/test_kernels.hip" -o tk.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input=tk.co --output=test_kernels.elf

echo "--- new-kernel resource usage ---"
hipcc --offload-arch="$ARCH" -O3 -w -Rpass-analysis=kernel-resource-usage --genco "$R/amd/test_kernels.hip" \
    -o /dev/null $INC 2>&1 | grep -E "Function Name: (index_select|index_score_mfma_128|index_score_fast_128|index_score_128)|VGPRs|Spill" \
    | grep -A2 -E "index_" || true
echo "---------------------------------"

/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o dsa_bench \
    "$R/bench/dsa_gather_bench.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d dsa_bench | grep -qi runpath && { echo "FAIL: RUNPATH leaked into host binary"; exit 1; }
ls -l --time-style=+%H:%M:%S test_kernels.elf dsa_bench | awk '{print "   ",$NF,$5"B",$6}'
echo "OK — run: sg render -c 'cd $OUT && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME LD_LIBRARY_PATH=/opt/rocm/lib ROCR_VISIBLE_DEVICES=0 ./dsa_bench test_kernels.elf'"
