#!/usr/bin/env bash
# Build libplow_s3gen.so (native Chatterbox S3Gen: speech tokens -> 24 kHz PCM).
# Usage: build.sh <out.so>     Uses $PLOW_NVCC (else nvcc on PATH).
set -euo pipefail
out=${1:?usage: build.sh <out.so>}
here=$(cd "$(dirname "$0")" && pwd)
nvcc=${PLOW_NVCC:-nvcc}
mkdir -p "$(dirname "$out")"
"$nvcc" -O3 -std=c++17 -cudart static -gencode arch=compute_90a,code=sm_90a -shared -Xcompiler -fPIC \
  ${PLOW_S3GEN_NVCC_FLAGS:-} -o "$out" "$here/s3gen.cu"
echo "built $out"
