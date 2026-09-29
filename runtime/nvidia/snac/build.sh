#!/usr/bin/env bash
# Build libplow_snac.so (native SNAC-24kHz decoder).  Usage: build.sh <out.so>
# Uses $PLOW_NVCC (else nvcc on PATH).
set -euo pipefail
out=${1:?usage: build.sh <out.so>}
here=$(cd "$(dirname "$0")" && pwd)
nvcc=${PLOW_NVCC:-nvcc}
mkdir -p "$(dirname "$out")"
"$nvcc" -O3 -std=c++17 -cudart static -gencode arch=compute_90a,code=sm_90a -shared -Xcompiler -fPIC \
  -o "$out" "$here/snac.cu"
echo "built $out"
