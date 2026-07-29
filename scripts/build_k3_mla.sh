#!/usr/bin/env bash
# build_k3_mla.sh — the Kimi-K3 GATED MLA BLOCK numeric gate (rung 3).            [K3-MLA-GATE]
#
# Same environment traps as build_k3_moe.sh / build_k3_real.sh (plans/knob-contract.md §0a):
# the .co needs system ROCm, which `nix develop` BREAKS (GLIBC_2.38); the host binary needs SYSTEM
# gcc in a scrubbed env or it aborts as "stack smashing detected".
#
# -DPLOW_MXFP4=1 is NOT optional — the routed experts are mxfp4 and the LatentMoE half of this block
# is rung 2's graph verbatim. (The routed-expert arms 45/46 are not themselves behind that flag, but
# the object must match the one rung 2 was validated on.)
#
#   ./scripts/build_k3_mla.sh [outdir]                       # build, OUTSIDE nix
#   PYTHONNOUSERSITE=1 nix develop .#quantize --command \
#       python3 runtime/tests/k3_mla_oracle.py <outdir>/k3_mla_fixture.bin
#   perf-data/harness/gpulease -n 1 k3mla sg render -c \
#       'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; cd <outdir> && \
#        LD_LIBRARY_PATH=/opt/rocm/lib ./k3_mla_test interp_decode.elf k3_mla_fixture.bin'
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/home/lava/models/k3mla}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"
cd "$OUT"
# A failed build must leave NOTHING behind, or the next run silently tests a stale object.
rm -f i_decode.co interp_decode.elf k3_mla_test

# DECODE bucket, one token. The MLA decode path (FLASH_MLA_DECODE + MLA_MERGE_FOLD) and the decode
# MoE path both carry T=1; T>1 is the separate prefill graph, which this gate does not touch.
echo "[1/2] device code object ($ARCH, decode bucket, PLOW_GEMV_MM=1, PLOW_MXFP4=1, PLOW_K3=1)"
hipcc --offload-arch="$ARCH" -O3 -w -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_MXFP4=1 -DPLOW_K3=1 \
      --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=i_decode.co --output=interp_decode.elf

echo "[2/2] host harness (system gcc, scrubbed env)"
/usr/bin/env -i PATH=/usr/bin:/bin HOME="${HOME:-/tmp}" /usr/bin/gcc -O2 -std=gnu11 -o k3_mla_test \
    "$R/tests/k3_mla_block_gfx950_test.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d k3_mla_test | grep -qi runpath && { echo "FAIL: RUNPATH leaked into k3_mla_test"; exit 1; }

ls -l --time-style=+%H:%M:%S interp_decode.elf k3_mla_test
echo "built in $OUT"
