#!/usr/bin/env bash
# build_k3_mla.sh — the Kimi-K3 GATED MLA BLOCK numeric gate (rung 3).            [K3-MLA-GATE]
#
# This gate is built only with the flake-pinned ROCm 7.14 toolchain.
#
# -DPLOW_MXFP4=1 is NOT optional — the routed experts are mxfp4 and the LatentMoE half of this block
# is rung 2's graph verbatim. (The routed-expert arms 45/46 are not themselves behind that flag, but
# the object must match the one rung 2 was validated on.)
#
#   nix develop --command ./scripts/build_k3_mla.sh [outdir]
#   nix develop .#quantize --command env PYTHONNOUSERSITE=1 \
#       python3 runtime/tests/k3_mla_oracle.py <outdir>/k3_mla_fixture.bin
#   nix develop --command perf-data/harness/gpulease -n 1 k3mla \
#       <outdir>/k3_mla_test <outdir>/interp_decode_k3.elf <outdir>/k3_mla_fixture.bin
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/home/lava/models/k3mla}}"
source "$REPO/scripts/nix_rocm_714.sh"
plow_init_rocm_714
ARCH="$PLOW_K3_ARCH"
INC="-I$R/amd -I$R/common"
# Extra -D flags for an AXIS A/B against the same source. The fp8-latent-KV gate is
#   nix develop --command env PLOW_EXTRA_DEFS=-DPLOW_FP8_KV=1 \
#       ./scripts/build_k3_mla.sh <outdir-fp8>
# built into its OWN outdir, because PLOW_FP8_KV is a SWAP: that object has no bf16
# FLASH_MLA_DECODE arm, so pointing the bf16 harness at it would hit the silent `default:`.
EXTRA="${PLOW_EXTRA_DEFS:-}"
case " $EXTRA " in
    *" -DPLOW_FP8_KV=1 "*) OBJ=interp_decode_fp8kv_k3.elf ;;
    *) OBJ=interp_decode_k3.elf ;;
esac
mkdir -p "$OUT"
cd "$OUT"
# A failed build must leave NOTHING behind, or the next run silently tests a stale object.
rm -f i_decode.co interp_decode_k3.elf interp_decode_fp8kv_k3.elf k3_mla_test

# DECODE bucket, one token. The MLA decode path (FLASH_MLA_DECODE + MLA_MERGE_FOLD) and the decode
# MoE path both carry T=1; T>1 is the separate prefill graph, which this gate does not touch.
echo "[1/2] device code object ($ARCH, decode bucket, PLOW_GEMV_MM=1, PLOW_MXFP4=1, PLOW_K3=1 $EXTRA)"
"$PLOW_K3_HIPCC" --offload-arch="$ARCH" -O3 -w -DPLOW_ARCH_SUFFIX="$ARCH" \
      -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_MXFP4=1 -DPLOW_K3=1 $EXTRA \
      --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$PLOW_K3_BUNDLER" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=i_decode.co --output="$OBJ"
plow_audit_gfx942_decode_object "$OBJ" "$REPO/scripts/asm_expect_gfx942.json"

echo "[2/2] host harness (Nix toolchain)"
"$PLOW_K3_HOST_CC" -O2 -std=gnu11 -DPLOW_TEST_ARCH_SUFFIX="$ARCH" -o k3_mla_test \
    "$R/tests/k3_mla_block_gfx950_test.c" "$R/amd/hsa_backend.c" \
    -I"$PLOW_K3_ROCM/include" -L"$PLOW_K3_ROCM/lib" \
    -Wl,-rpath,"$PLOW_K3_ROCM/lib" -lhsa-runtime64 -lm
plow_assert_nix_binary k3_mla_test

ls -l --time-style=+%H:%M:%S "$OBJ" k3_mla_test
echo "built in $OUT"
