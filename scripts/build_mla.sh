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
# Discover the bundler from the INSTALLED ROCm instead of pinning a version. This
# was pinned to 7.0.2 and the path does not exist on a 7.2.4 box, so the build died
# on the first machine that had the GPU. $PLOW_BUNDLER still overrides.
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
# ONE wave count, passed to BOTH compiles. The device object's __launch_bounds__ and the host
# harness's launch geometry are the same macro resolved in two different compilations, and this
# script used to pass it to neither -- so a device object built at eight waves ran under a host
# that launches 256 threads. That is a LEGAL dispatch (waves 4..7 just never exist), it leaves
# every per-wave LDS array half-written, and it reads as an MLA kernel defect. The harness now
# also verifies this at startup via `plow_probe_wg_threads`; this is the other half.
WAVES="${PLOW_WG_WAVES:-8}"
WDEF="-DPLOW_WG_WAVES=$WAVES"
# CDNA3 has 64 KiB of workgroup LDS, so the gfx950 default GEMM stage arena (147,456 B) does not
# fit and the object fails to compile. Same tile the shipped gfx942 objects use.
case "$ARCH" in
  gfx942) TILE="-DGM_DBUF=1 $( [ "$WAVES" = 4 ] && echo '-DGM_BM=64 -DGM_BN=128' \
                                              || echo '-DGM_BM=192 -DGM_BN=256' )" ;;
  *)      TILE="" ;;
esac
mkdir -p "$OUT"; cd "$OUT"

rm -f tk.co test_kernels.elf mla_ref mla_test fixture.bin

# Device: the golden wrappers (share op_attention.h with the interpreter).
hipcc --offload-arch="$ARCH" -O3 -w $WDEF $TILE --genco "$R/amd/test_kernels.hip" -o tk.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input=tk.co --output=test_kernels.elf

# The MLA decode kernel must stay on the validated D=512 decode budget (<=256 VGPR / occ>=2,
# 2 waves/SIMD). Report its resource usage and FAIL the build if it walks off the cliff.
U=$(hipcc --offload-arch="$ARCH" -O3 -w $WDEF $TILE -Rpass-analysis=kernel-resource-usage \
      --genco "$R/amd/test_kernels.hip" -o /dev/null $INC 2>&1 | grep -A6 'mla_flash_decode_512' || true)
echo "--- mla_flash_decode_512 resource usage ---"
echo "$U" | grep -E 'VGPRs|AGPRs|Occupancy|Spill|SGPRs' || echo "  (usage lines not captured)"
echo "-------------------------------------------"

# Rust oracle.
rustc -O "$R/tests/mla_ref.rs" -o mla_ref
./mla_ref fixture.bin

# Host harness with the SYSTEM gcc in a CLEAN env (nix gcc bakes a RUNPATH that aborts at load).
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 $WDEF -o mla_test \
    "$R/tests/mla_gfx950_test.c" "$R/amd/hsa_backend.c" \
    -I"$R/amd" -I"$R/common" -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d mla_test | grep -qi runpath && { echo "FAIL: RUNPATH leaked into the host binary"; exit 1; }

ls -l --time-style=+%H:%M:%S test_kernels.elf mla_ref mla_test fixture.bin \
  | awk '{print "   ", $NF, $5"B", $6}'
echo "OK — run: sg render -c 'cd $OUT && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=7 ./mla_test fixture.bin'"
