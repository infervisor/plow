#!/usr/bin/env bash
# build_k3_moe.sh — the Kimi-K3 MoE-BLOCK numeric gate (rung 2) and the MXFP4 nibble probe.
#                                                                    [K3-MOE-GATE] [K3-MXFP4]
# Same environment traps as build_k3_real.sh / build_kda_real.sh:
# the .co needs system ROCm, which `nix develop` BREAKS (GLIBC_2.38); the host binaries need SYSTEM
# gcc in a scrubbed env or they abort as "stack smashing detected".
#
# ONE FLAG THAT IS NOT OPTIONAL: -DPLOW_MXFP4=1. `case PLOW_DOP_GEMV_MXFP4` sits inside
# `#if PLOW_MXFP4` (interp.hip), so without it the mxfp4 GEMV arm is compiled out and the packet
# falls to the dispatch `default:` — a SILENT NOP that leaves the output buffer untouched. That is
# how the nibble probe's first run came back matching NEITHER reading at exactly 1.000e+00, and it
# is why that probe reports INCONCLUSIVE instead of picking the smaller of two identical numbers.
# The routed-expert arms (45/46) are NOT behind that flag; only the standalone mxfp4 GEMV/GLU are.
#
#   ./scripts/build_k3_moe.sh [outdir]                      # build, OUTSIDE nix
#   PYTHONNOUSERSITE=1 nix develop .#quantize --command \
#       python3 runtime/tests/k3_moe_oracle.py <outdir>/k3_moe_fixture.bin
#   PYTHONNOUSERSITE=1 nix develop .#quantize --command \
#       python3 runtime/tests/k3_mxfp4_nibble_oracle.py <outdir>/mxnib_fixture.bin
#   perf-data/harness/gpulease -n 1 k3moe sg render -c \
#       'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; cd <outdir> && \
#        LD_LIBRARY_PATH=/opt/rocm/lib ./k3_moe_test interp_decode.elf k3_moe_fixture.bin'
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/home/lava/models/k3moe}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"
cd "$OUT"
# A failed build must leave NOTHING behind, or the next run silently tests a stale object.
rm -f i_decode.co interp_decode.elf k3_moe_test mxnib_test

# This gate is the DECODE MoE path, which carries ONE token: MoeRouterTopk's table is [k], not
# [T,k], and d_moe_expert_glu_fp8_blk reads a single x row. So PLOW_GEMV_MM=1 is correct here, not
# a compromise — unlike the rung-1 block gate, where it must be >= T.
echo "[1/2] device code object ($ARCH, decode bucket, PLOW_GEMV_MM=1, PLOW_MXFP4=1, PLOW_K3=1)"
hipcc --offload-arch="$ARCH" -O3 -w -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_MXFP4=1 -DPLOW_K3=1 \
      --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=i_decode.co --output=interp_decode.elf

echo "[2/2] host harnesses (system gcc, scrubbed env)"
for t in k3_moe_block_gfx950_test:k3_moe_test k3_mxfp4_nibble_test:mxnib_test; do
    src="${t%%:*}"; bin="${t##*:}"
    /usr/bin/env -i PATH=/usr/bin:/bin HOME="${HOME:-/tmp}" /usr/bin/gcc -O2 -std=gnu11 -o "$bin" \
        "$R/tests/$src.c" "$R/amd/hsa_backend.c" \
        -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
    readelf -d "$bin" | grep -qi runpath && { echo "FAIL: RUNPATH leaked into $bin"; exit 1; }
done

ls -l --time-style=+%H:%M:%S interp_decode.elf k3_moe_test mxnib_test
echo "built in $OUT"
