#!/usr/bin/env bash
# Build the TILE-TRIGGERED COLLECTIVE microbench (coarse 1-gate vs tiled 14-gate all-reduce,
# plus the real decode row-GEMV workgroup-skew instrument). See runtime/tests/tp_tilegate_bench.c.
# Produces, into the output dir (default /tmp/tgate, or $1):
#   tp_tilegate_kernels.elf   the gfx950 device kernels
#   tp_tilegate_bench         the host harness (links hsa_backend.c + libhsa-runtime64)
#
# Same toolchain contract as build_tp_allreduce.sh: hipcc for the device code, the SYSTEM gcc
# in a CLEAN env for the host.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/tgate}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-/opt/rocm/lib/llvm/bin/clang-offload-bundler}"
mkdir -p "$OUT"; cd "$OUT"

rm -f tp_tilegate.co tp_tilegate_kernels.elf tp_tilegate_bench

hipcc --offload-arch="$ARCH" -O3 -w --genco "$R/tests/tp_tilegate_kernels.hip" -o tp_tilegate.co
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=tp_tilegate.co --output=tp_tilegate_kernels.elf

/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/gcc -O2 -std=gnu11 -o tp_tilegate_bench \
    "$R/tests/tp_tilegate_bench.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm

ls -l --time-style=+%H:%M:%S tp_tilegate_kernels.elf tp_tilegate_bench \
  | awk '{print "   ", $NF, $5"B", $6}'
echo "run:  (cd $OUT && sudo env LD_LIBRARY_PATH=/opt/rocm/lib ./tp_tilegate_bench 0 1 2 3)"
echo "      rank count = number of device ids; env TG_HIDDEN TG_ITERS TG_GEMV_BLK TG_ELF"
