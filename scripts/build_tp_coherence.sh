#!/usr/bin/env bash
# Build the adversarial cross-GPU (XGMI) coherence stress test. Produces, into the
# output dir (default /tmp/tpcv, or $1):
#   tp_coherence_kernels.elf   the gfx950 producer/consumer device kernels
#   tp_coherence_bench         the host harness (links hsa_backend.c)
#
# Decides whether the flat one-shot all-reduce's system-scope xctr handshake
# (op_collective.h) is a real happens-before on gfx950 XGMI or a hardware accident.
# Run:  ./tp_coherence_bench [iters] [pairsCSV] [ndata_words]
#   e.g. ./tp_coherence_bench 1000000 0-1,0-4,4-0,3-7 16
#
# Same toolchain contract as build_tp_p2p.sh: nix-shell hipcc for the device code,
# the SYSTEM gcc in a CLEAN env for the host.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/tpcv}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
# Discover the bundler from the INSTALLED ROCm instead of pinning a version. This
# was pinned to 7.0.2 and the path does not exist on a 7.2.4 box, so the build died
# on the first machine that had the GPU. $PLOW_BUNDLER still overrides.
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
mkdir -p "$OUT"; cd "$OUT"

rm -f coh.co tp_coherence_kernels.elf tp_coherence_bench

hipcc --offload-arch="$ARCH" -O3 -w --genco "$R/tests/tp_coherence_kernels.hip" -o coh.co
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=coh.co --output=tp_coherence_kernels.elf

/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o tp_coherence_bench \
    "$R/tests/tp_coherence_bench.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d tp_coherence_bench | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }

ls -l --time-style=+%H:%M:%S tp_coherence_kernels.elf tp_coherence_bench \
  | awk '{print "   ", $NF, $5"B", $6}'
