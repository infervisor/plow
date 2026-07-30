#!/usr/bin/env bash
# build_kda_fuse_bench.sh — the KDA packet-count instrument.            [K3-KDA-FUSE]
#
# Same two environment traps as scripts/build_k3_real.sh (§0a of the knob contract): the .co needs
# SYSTEM ROCm (`nix develop` breaks it with GLIBC_2.38), and the host binary needs SYSTEM gcc in a
# scrubbed env or nix's RUNPATH aborts it as "stack smashing detected".
#
#   ./scripts/build_kda_fuse_bench.sh [outdir]                  # build, OUTSIDE nix
#   sg render -c 'cd <outdir> && env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/opt/rocm/lib \
#       ROCR_VISIBLE_DEVICES=<n> ./kda_fuse_bench interp_decode.elf 12 8 69 50'
#
# ROCR_VISIBLE_DEVICES, not HIP_VISIBLE_DEVICES: this is a bare ROCr binary and HIP's variable does
# nothing to it. ROCr enumerates in KFD-node order, which is NOT rocm-smi's order — re-derive the
# map and confirm the device is idle before trusting a number.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/home/lava/models/k3kdafuse}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"
cd "$OUT"
rm -f i_decode.co interp_decode.elf kda_fuse_bench

echo "[1/2] device code object ($ARCH, decode bucket, PLOW_GEMV_MM=1, PLOW_K3=1)"
hipcc --offload-arch="$ARCH" -O3 -w -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_K3=1 \
      --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=i_decode.co --output=interp_decode.elf

echo "[2/2] host harness (system gcc, scrubbed env)"
/usr/bin/env -i PATH=/usr/bin:/bin HOME="${HOME:-/tmp}" /usr/bin/gcc -O2 -std=gnu11 -o kda_fuse_bench \
    "$R/tests/kda_fuse_bench_gfx950.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d kda_fuse_bench | grep -qi runpath && { echo "FAIL: RUNPATH leaked into host binary"; exit 1; }

ls -l --time-style=+%H:%M:%S interp_decode.elf kda_fuse_bench
echo "built in $OUT"
