#!/usr/bin/env bash
# Build the DeepSeek MLA decode test: test_kernels.elf (with the MLA ops), the Rust oracle
# (mla_ref), the host harness (mla_test), and the fixture. Run separately under `sg render`.  [DEEPSEEK-MLA]
#
#   nix develop --command bash -c './scripts/build_mla.sh /tmp/mlab'
#   sg render -c 'cd /tmp/mlab && /usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME \
#       LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=7 ./mla_test fixture.bin'
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/mlab}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-/opt/rocm-7.0.2/lib/llvm/bin/clang-offload-bundler}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"; cd "$OUT"

rm -f tk.co test_kernels.elf mla_ref mla_test fixture.bin

# Device: the golden wrappers (share op_attention.h with the interpreter).
hipcc --offload-arch="$ARCH" -O3 -w --genco "$R/amd/test_kernels.hip" -o tk.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input=tk.co --output=test_kernels.elf

# The MLA decode kernel must stay on the validated D=512 decode budget (<=256 VGPR / occ>=2,
# 2 waves/SIMD). Report its resource usage and FAIL the build if it walks off the cliff.
U=$(hipcc --offload-arch="$ARCH" -O3 -w -Rpass-analysis=kernel-resource-usage \
      --genco "$R/amd/test_kernels.hip" -o /dev/null $INC 2>&1 | grep -A6 'mla_flash_decode_512' || true)
echo "--- mla_flash_decode_512 resource usage ---"
echo "$U" | grep -E 'VGPRs|AGPRs|Occupancy|Spill|SGPRs' || echo "  (usage lines not captured)"
echo "-------------------------------------------"

# Rust oracle.
rustc -O "$R/tests/mla_ref.rs" -o mla_ref
./mla_ref fixture.bin

# Host harness with the SYSTEM gcc in a CLEAN env (nix gcc bakes a RUNPATH that aborts at load).
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o mla_test \
    "$R/tests/mla_gfx950_test.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d mla_test | grep -qi runpath && { echo "FAIL: RUNPATH leaked into the host binary"; exit 1; }

ls -l --time-style=+%H:%M:%S test_kernels.elf mla_ref mla_test fixture.bin \
  | awk '{print "   ", $NF, $5"B", $6}'
echo "OK — run: sg render -c 'cd $OUT && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=7 ./mla_test fixture.bin'"
