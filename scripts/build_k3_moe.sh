#!/usr/bin/env bash
# build_k3_moe.sh — the Kimi-K3 MoE-BLOCK numeric gate (rung 2) and the MXFP4 nibble probe.
#                                                                    [K3-MOE-GATE] [K3-MXFP4]
# This gate is built only with the flake-pinned ROCm 7.14 toolchain.
#
# ONE FLAG THAT IS NOT OPTIONAL: -DPLOW_MXFP4=1. `case PLOW_DOP_GEMV_MXFP4` sits inside
# `#if PLOW_MXFP4` (interp.hip), so without it the mxfp4 GEMV arm is compiled out and the packet
# falls to the dispatch `default:` — a SILENT NOP that leaves the output buffer untouched. That is
# how the nibble probe's first run came back matching NEITHER reading at exactly 1.000e+00, and it
# is why that probe reports INCONCLUSIVE instead of picking the smaller of two identical numbers.
# The routed-expert arms (45/46) are NOT behind that flag; only the standalone mxfp4 GEMV/GLU are.
#
#   nix develop --command ./scripts/build_k3_moe.sh [outdir]
#   nix develop .#quantize --command env PYTHONNOUSERSITE=1 \
#       python3 runtime/tests/k3_moe_oracle.py <outdir>/k3_moe_fixture.bin
#   nix develop .#quantize --command env PYTHONNOUSERSITE=1 \
#       python3 runtime/tests/k3_mxfp4_nibble_oracle.py <outdir>/mxnib_fixture.bin
#   nix develop --command perf-data/harness/gpulease -n 1 k3moe \
#       <outdir>/k3_moe_test <outdir>/interp_decode_k3.elf <outdir>/k3_moe_fixture.bin
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/home/lava/models/k3moe}}"
source "$REPO/scripts/nix_rocm_714.sh"
plow_init_rocm_714
ARCH="$PLOW_K3_ARCH"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"
cd "$OUT"
# A failed build must leave NOTHING behind, or the next run silently tests a stale object.
rm -f i_decode.co interp_decode_k3.elf k3_moe_test mxnib_test

# This gate is the DECODE MoE path, which carries ONE token: MoeRouterTopk's table is [k], not
# [T,k], and d_moe_expert_glu_fp8_blk reads a single x row. So PLOW_GEMV_MM=1 is correct here, not
# a compromise — unlike the rung-1 block gate, where it must be >= T.
echo "[1/2] device code object ($ARCH, decode bucket, PLOW_GEMV_MM=1, PLOW_MXFP4=1, PLOW_K3=1)"
"$PLOW_K3_HIPCC" --offload-arch="$ARCH" -O3 -w -DPLOW_ARCH_SUFFIX="$ARCH" \
      -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_MXFP4=1 -DPLOW_K3=1 \
      --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$PLOW_K3_BUNDLER" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=i_decode.co --output=interp_decode_k3.elf
plow_audit_gfx942_decode_object interp_decode_k3.elf "$REPO/scripts/asm_expect_gfx942.json"

echo "[2/2] host harnesses (Nix toolchain)"
for t in k3_moe_block_gfx950_test:k3_moe_test k3_mxfp4_nibble_test:mxnib_test; do
    src="${t%%:*}"; bin="${t##*:}"
    "$PLOW_K3_HOST_CC" -O2 -std=gnu11 -DPLOW_TEST_ARCH_SUFFIX="$ARCH" -o "$bin" \
        "$R/tests/$src.c" "$R/amd/hsa_backend.c" \
        -I"$PLOW_K3_ROCM/include" -L"$PLOW_K3_ROCM/lib" \
        -Wl,-rpath,"$PLOW_K3_ROCM/lib" -lhsa-runtime64 -lm
    plow_assert_nix_binary "$bin"
done

ls -l --time-style=+%H:%M:%S interp_decode_k3.elf k3_moe_test mxnib_test
echo "built in $OUT"
