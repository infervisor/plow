#!/usr/bin/env bash
# Build the cross-GPU (XGMI) transport de-risk microbench for tensor-parallel
# decode. Produces, into the output dir (default /tmp/tpb, or $1):
#   tp_p2p_kernels.elf   the gfx950 device kernels (peer r/w, ping-pong, reduce)
#   tp_p2p_bench         the host harness (links hsa_backend.c + libhsa-runtime64)
#
# Same toolchain contract as build_gfx950.sh: nix-shell hipcc for the device
# code, the SYSTEM gcc in a CLEAN env for the host (a nix RUNPATH into glibc 2.42
# vs the system 2.35 ELF interpreter aborts with a bogus "stack smashing").
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/tpb}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-/opt/rocm/lib/llvm/bin/clang-offload-bundler}"
mkdir -p "$OUT"; cd "$OUT"

rm -f tp_p2p.co tp_p2p_kernels.elf tp_p2p_bench

# Device code object -> unbundled raw ELF (the form hsa_backend.c loads).
hipcc --offload-arch="$ARCH" -O3 -w --genco "$R/tests/tp_p2p_kernels.hip" -o tp_p2p.co
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=tp_p2p.co --output=tp_p2p_kernels.elf

# Host harness with system gcc, clean env.
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o tp_p2p_bench \
    "$R/tests/tp_p2p_bench.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d tp_p2p_bench | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }

ls -l --time-style=+%H:%M:%S tp_p2p_kernels.elf tp_p2p_bench \
  | awk '{print "   ", $NF, $5"B", $6}'
echo "run:  (cd $OUT && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME LD_LIBRARY_PATH=/opt/rocm/lib ./tp_p2p_bench 0 1)"
