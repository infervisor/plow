#!/usr/bin/env bash
# Build runtime/tests/dsv41_blockfp8_gfx942_test.hip — DeepSeek-V4.1's [32,32] ue8m0 block-fp8
# GEMM (d_gemm_fp8_mx -> d_gemm_t<WFP8MX>) against a host reference.
#
# That arm gates 39.8% of a V4.1 8k prefill — every projection in the model — because plow's other
# block-fp8 kernels all assume [128,128] with f32 scales. See
# docs/amd/deepseek-v41-flash-mi300x.md section 5.2 item 5b.
#
# PLOW_WG_WAVES=8 is REQUIRED, not a default: the 128x128 tile's wave decomposition assumes it,
# and the arch header's tile constants come from plow_config.h, so -I must reach one.
#
# usage:  scripts/build_dsv41_blockfp8.sh OUT_DIR [PLOW_CONFIG_INCLUDE_DIR]
#   then: perf-data/tools/gpulease -n 1 dsv41-blkfp8 OUT_DIR/dsv41_blockfp8_gfx942_test
set -euo pipefail

out=${1:?usage: $0 OUT_DIR [PLOW_CONFIG_INCLUDE_DIR]}
root=$(cd "$(dirname "$0")/.." && pwd)
cfg=${2:-}
mkdir -p "$out"

HIPCC=${PLOW_HIPCC:-${ROCM_PATH:-/opt/rocm}/bin/hipcc}
inc=(-I"$root/runtime/amd" -I"$root/runtime/common")
def=(-DPLOW_WG_WAVES=8 -DGM_DBUF=1 -DGM_BM=192 -DGM_BN=256
     -DPLOW_BUCKET_PREFILL=1 -DPLOW_BUCKET_DECODE=0 -DPLOW_FP8=1)
if [[ -n $cfg ]]; then
    inc+=(-I"$cfg")
    def+=(-DPLOW_CONFIG='"plow_config.h"')
fi

"$HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 "${inc[@]}" "${def[@]}" \
    -c "$root/runtime/tests/dsv41_blockfp8_gfx942_test.hip" -o "$out/dsv41_blockfp8.o"

c++ "$out/dsv41_blockfp8.o" \
    -L"${ROCM_PATH:-/opt/rocm}/lib" -Wl,-rpath,"${ROCM_PATH:-/opt/rocm}/lib" -lamdhip64 \
    -o "$out/dsv41_blockfp8_gfx942_test"

echo "built $out/dsv41_blockfp8_gfx942_test"
echo "run:  $root/perf-data/tools/gpulease -n 1 dsv41-blkfp8 $out/dsv41_blockfp8_gfx942_test"
