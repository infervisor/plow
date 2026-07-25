#!/usr/bin/env bash
# Build the 2-GPU one-shot all-reduce microbench (validates op_collective.h's
# d_xreduce_oneshot bit-exact + latency across two gfx950 GPUs over XGMI).
# Produces, into the output dir (default /tmp/tpar, or $1):
#   tp_allreduce_kernels.elf   the gfx950 device kernels (fill + one-shot all-reduce)
#   tp_allreduce_bench         the host harness (links hsa_backend.c + libhsa-runtime64)
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

rm -f tp_allreduce.co tp_allreduce_kernels.elf tp_allreduce_bench

# Device code object -> unbundled raw ELF (the form hsa_backend.c loads).
hipcc --offload-arch="$ARCH" -O3 -w --genco "$R/tests/tp_allreduce_kernels.hip" -o tp_allreduce.co
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=tp_allreduce.co --output=tp_allreduce_kernels.elf

# Host harness with system gcc, clean env.
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o tp_allreduce_bench \
    "$R/tests/tp_allreduce_bench.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d tp_allreduce_bench | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }

ls -l --time-style=+%H:%M:%S tp_allreduce_kernels.elf tp_allreduce_bench \
  | awk '{print "   ", $NF, $5"B", $6}'
echo "run:  (cd $OUT && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME LD_LIBRARY_PATH=/opt/rocm/lib ./tp_allreduce_bench 0 1)"
