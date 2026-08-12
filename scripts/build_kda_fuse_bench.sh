#!/usr/bin/env bash
# build_kda_fuse_bench.sh — the KDA packet-count instrument.            [K3-KDA-FUSE]
#
# Uses only the ROCm 7.14 toolchain exported by the repository Nix flake.
#
#   nix develop --command ./scripts/build_kda_fuse_bench.sh [outdir]
#   nix develop --command perf-data/tools/gpulease -n 1 kda-fuse \
#       <outdir>/kda_fuse_bench <outdir>/interp_decode_kda.elf 12 8 69 50
#
# ROCR_VISIBLE_DEVICES, not HIP_VISIBLE_DEVICES: this is a bare ROCr binary and HIP's variable does
# nothing to it. ROCr enumerates in KFD-node order, which is NOT rocm-smi's order — re-derive the
# map and confirm the device is idle before trusting a number.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
source "$REPO/scripts/nix_rocm_714.sh"
plow_init_rocm_714
OUT="${1:-${PLOW_BUILD_DIR:-$REPO/build-amd/kda-fuse/gfx942}}"
ARCH="$PLOW_K3_ARCH"
HIPCC="$PLOW_K3_HIPCC"
BUN="$PLOW_K3_BUNDLER"
HOST_CC="$PLOW_K3_HOST_CC"
ROCM="$PLOW_K3_ROCM"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"
cd "$OUT"
rm -f i_decode.co interp_decode_k3.elf kda_fuse_bench

echo "[1/2] device code object ($ARCH, decode bucket, PLOW_GEMV_MM=1, PLOW_K3=1)"
"$HIPCC" --offload-arch="$ARCH" -O3 -w -DPLOW_ARCH_SUFFIX="$ARCH" \
      -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_K3=1 \
      --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=i_decode.co --output=interp_decode_k3.elf
plow_audit_gfx942_decode_object interp_decode_k3.elf "$REPO/scripts/asm_expect_gfx942.json"

echo "[2/2] host harness (Nix compiler + ROCm 7.14)"
"$HOST_CC" -O2 -std=gnu11 -DPLOW_TEST_ARCH_SUFFIX="$ARCH" -o kda_fuse_bench \
    "$R/tests/kda_fuse_bench_gfx950.c" "$R/amd/hsa_backend.c" \
    -I"$ROCM/include" -L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" -lhsa-runtime64 -lm
plow_assert_nix_binary kda_fuse_bench

ls -l --time-style=+%H:%M:%S interp_decode_k3.elf kda_fuse_bench
echo "built in $OUT"
