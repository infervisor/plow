#!/usr/bin/env bash
# build_k3_real.sh — the Kimi-K3 COMPLETE-BLOCK numeric gate.            [K3-BLOCK-GATE]
#
# Modelled on scripts/build_kda_real.sh, which is modelled on scripts/build_glm52_real.sh,
# This gate is built only with the flake-pinned ROCm 7.14 toolchain.
#
#   nix develop --command ./scripts/build_k3_real.sh [outdir]
#   nix develop .#quantize --command env PYTHONNOUSERSITE=1 \
#       python3 runtime/tests/k3_real_oracle.py <outdir>/k3_fixture.bin    # fixture, ONCE
#   nix develop --command perf-data/harness/gpulease -n 1 k3-real \
#       <outdir>/k3_block_test <outdir>/interp_decode_k3.elf <outdir>/k3_fixture.bin
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/home/lava/models/k3block}}"
source "$REPO/scripts/nix_rocm_714.sh"
plow_init_rocm_714
ARCH="$PLOW_K3_ARCH"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"
cd "$OUT"
# A failed build must leave NOTHING behind, or the next run silently tests a stale object.
rm -f i_decode.co interp_decode_k3.elf k3_block_test

# PLOW_GEMV_MM MUST BE >= T. `gemv_rows` accumulates `float acc[MM]` with MM = PLOW_GEMV_MM, a
# COMPILE-TIME constant defaulting to 1, and writes C[m*N+n] only for m < MM. At T>1 with the
# default, every GEMV writes ROW 0 and leaves rows 1..T-1 UNTOUCHED — not a crash, three quarters
# of the tokens carrying stale buffer contents, with the signature sqrt((T-1)/T) = 0.866 on every
# post-GEMV stage. Recorded in the design notes §6.3 and build_gfx950.sh:51-64.
GVMM="${PLOW_GEMV_MM:-${K3_T:-4}}"
echo "[1/2] device code object ($ARCH, decode bucket, PLOW_GEMV_MM=$GVMM, PLOW_K3=1)"
"$PLOW_K3_HIPCC" --offload-arch="$ARCH" -O3 -w -DPLOW_ARCH_SUFFIX="$ARCH" \
      -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM="$GVMM" -DPLOW_K3=1 \
      --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$PLOW_K3_BUNDLER" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=i_decode.co --output=interp_decode_k3.elf
plow_audit_gfx942_decode_object interp_decode_k3.elf "$REPO/scripts/asm_expect_gfx942.json"

echo "[2/2] host harness (Nix toolchain)"
"$PLOW_K3_HOST_CC" -O2 -std=gnu11 -DPLOW_TEST_ARCH_SUFFIX="$ARCH" -o k3_block_test \
    "$R/tests/k3_block_gfx950_test.c" "$R/amd/hsa_backend.c" \
    -I"$PLOW_K3_ROCM/include" -L"$PLOW_K3_ROCM/lib" \
    -Wl,-rpath,"$PLOW_K3_ROCM/lib" -lhsa-runtime64 -lm
plow_assert_nix_binary k3_block_test

ls -l --time-style=+%H:%M:%S interp_decode_k3.elf k3_block_test
echo "built in $OUT"
