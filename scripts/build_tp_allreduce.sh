#!/usr/bin/env bash
# Build the N-RANK, BATCH-SWEPT one-shot all-reduce microbench (validates op_collective.h's
# d_xreduce_oneshot bit-exact + latency across N gfx950 GPUs over XGMI).
# Produces, into the output dir (default /tmp/tpar, or $1):
#   tp_allreduce_kernels.elf   the gfx950 device kernels (fill + one-shot all-reduce)
#   tp_allreduce_bench         one-shot host harness
#   tp_allreduce_prefill_bench configurable transformer-prefill two-shot host harness
#
# Same toolchain contract as build_tp_p2p.sh: nix-shell hipcc for the device code,
# the SYSTEM gcc in a CLEAN env for the host.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/tpar}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-/opt/rocm/lib/llvm/bin/clang-offload-bundler}"
mkdir -p "$OUT"; cd "$OUT"

rm -f tp_allreduce.co tp_allreduce_kernels.elf tp_allreduce_bench tp_allreduce_prefill_bench

# Device code object -> unbundled raw ELF (the form hsa_backend.c loads).
XR_DEF=""
[ "${PLOW_XR_AGG:-0}" = 1 ] && XR_DEF="-DPLOW_XR_AGG=1"
hipcc --offload-arch="$ARCH" -O3 -w $XR_DEF --genco \
    "$R/tests/tp_allreduce_kernels.hip" -o tp_allreduce.co
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=tp_allreduce.co --output=tp_allreduce_kernels.elf

# Host harness with system gcc, clean env.
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o tp_allreduce_bench \
    "$R/tests/tp_allreduce_bench.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 \
    -o tp_allreduce_prefill_bench \
    "$R/tests/tp_allreduce_prefill_bench.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d tp_allreduce_bench | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }
readelf -d tp_allreduce_prefill_bench | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }

ls -l --time-style=+%H:%M:%S tp_allreduce_kernels.elf tp_allreduce_bench \
    tp_allreduce_prefill_bench \
  | awk '{print "   ", $NF, $5"B", $6}'
echo "run:  (cd $OUT && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME LD_LIBRARY_PATH=/opt/rocm/lib ./tp_allreduce_bench 1 2 3)"
echo "      rank count = number of device ids; env TP_HIDDEN (default 7168) TP_ITERS TP_ELF"
echo "prefill: (cd $OUT && ./tp_allreduce_prefill_bench 0 1 2 3 4 5 6 7)  PLOW_XR_AGG=${PLOW_XR_AGG:-0}"
echo "         env TP_ROWS TP_HIDDEN TP_NWG (default 256) TP_GATHER TP_ONESHOT"
echo "config:  (cd $OUT && TP_ROWS=8192 TP_HIDDEN=7168 TP_NWG=80 ./tp_allreduce_prefill_bench --check-config)"
echo "sweep:   scripts/run_tp_allreduce_prefill_sweep.sh $OUT gpu0 gpu1 gpu2 gpu3 gpu4 gpu5 gpu6 gpu7"
