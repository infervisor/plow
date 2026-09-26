#!/usr/bin/env bash
# Build the DeepSeek-V4.1 sm_90a kernel cubin: runtime/nvidia/dsv41/dsv41_all.cu -> $OUT.
# Uses PLOW_NVCC (set by `nix develop`) or the nix-store CUDA 12.9 toolkit, and a host gcc CUDA accepts.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$ROOT/target/dsv41/dsv41_sm90a.cubin}
NVCC=${PLOW_NVCC:-$(ls -d /nix/store/*-cuda-merged-12.9/bin/nvcc 2>/dev/null | head -1)}
CCBIN=${PLOW_NVCC_CCBIN:-$(ls -d /nix/store/*-gcc-wrapper-14.*/bin 2>/dev/null | head -1)}
[ -x "$NVCC" ] || { echo "nvcc not found; set PLOW_NVCC" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
# nvcc resolves through its symlink to cuda_nvcc, whose include/ lacks cuda_runtime.h (see flake.nix).
CUDA_INC=${PLOW_CUDA_INCLUDE:-$(dirname "$NVCC")/../include}
"$NVCC" ${CCBIN:+-ccbin "$CCBIN"} -I "$CUDA_INC" -arch=sm_90a -O3 -cubin -std=c++17 -lineinfo \
  -Xptxas -v ${DSV41_NVCC_FLAGS:-} -o "$OUT" "$ROOT/runtime/nvidia/dsv41/dsv41_all.cu" 2> "${OUT%.cubin}.ptxas.log" \
  || { cat "${OUT%.cubin}.ptxas.log" >&2; exit 1; }
grep -E 'spill|error' "${OUT%.cubin}.ptxas.log" | grep -v ' 0 bytes spill' >&2 || true
echo "$OUT"
