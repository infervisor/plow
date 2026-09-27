#!/usr/bin/env bash
# Build the DeepSeek-V4.1 sm_90a kernel cubin: runtime/nvidia/dsv41/dsv41_all.cu -> $OUT.
# Uses PLOW_NVCC (set by `nix develop`) or the nix-store CUDA 12.9 toolkit, and a host gcc CUDA accepts.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$ROOT/target/dsv41/dsv41_sm90a.cubin}
# Outside `nix develop`: the newest CUDA 12.x merged toolkit and gcc 14 wrapper in the store, chosen
# by version rather than glob order, so repeated builds pick the same toolchain.
newest() { ls -d $1 2>/dev/null | sed -E "s|^(.*-$2-([0-9.]+))(.*)$|\2\t\1\3|" | sort -V | tail -1 | cut -f2; }
NVCC=${PLOW_NVCC:-$(newest '/nix/store/*-cuda-merged-12.*/bin/nvcc' cuda-merged)}
CCBIN=${PLOW_NVCC_CCBIN:-$(newest '/nix/store/*-gcc-wrapper-14.*/bin' gcc-wrapper)}
[ -x "$NVCC" ] || { echo "nvcc not found; set PLOW_NVCC" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
# nvcc resolves through its symlink to cuda_nvcc, whose include/ lacks cuda_runtime.h (see flake.nix).
CUDA_INC=${PLOW_CUDA_INCLUDE:-$(dirname "$NVCC")/../include}
"$NVCC" ${CCBIN:+-ccbin "$CCBIN"} -I "$CUDA_INC" -arch=sm_90a -O3 -cubin -std=c++17 -lineinfo \
  -Xptxas -v ${DSV41_NVCC_FLAGS:-} -o "$OUT" "$ROOT/runtime/nvidia/dsv41/dsv41_all.cu" 2> "${OUT%.cubin}.ptxas.log" \
  || { cat "${OUT%.cubin}.ptxas.log" >&2; exit 1; }
grep -E 'spill|error' "${OUT%.cubin}.ptxas.log" | grep -v ' 0 bytes spill' >&2 || true
echo "$OUT"
