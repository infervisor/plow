#!/usr/bin/env bash
# Build and run the Gemma-4 per-kernel ROUTE MATRIX on sm_90a: for every dense projection shape
# of the 12B and the 26B-A4B, plow's shipped ws384 GEMM body vs cuBLASLt, in BOTH precisions.
#
#   bf16 arm : k_ws384 <PROD,false>  vs  cuBLASLt bf16          (the shipped BF16 route pair)
#   w8a8 arm : k_ws384_fp8 <PROD,true> vs cuBLASLt e4m3 with
#              CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F        (the FP8 route plow cannot reach)
#
# Both arms come from ONE source file (runtime/bench/nvidia/bf16_gemm_vs_cublas_bench.cu) with the
# SAME protocol -- rotated cold weights, alternating arm order, 6 rounds, median of the middle two,
# bounded-error correctness gate, output guard bytes -- so the two precisions are comparable.
#
# The nvcc flags below are the ones scripts/build_sm90a_gemma4_segments.sh gives the shipped
# interp_sm90a_pfgemm.cubin, so the body measured here is the body that serves.
#
# NOT run inside `nix develop`: nvcc's CPATH collides with the CUDA math headers (same reason
# build_sm90a_gemma4_segments.sh uses `env -i`). plowrt resolves libcublasLt.so.13 even inside
# nix, so /usr/local/cuda is also the right cuBLASLt to measure against.
#
# usage: scripts/bench/gemma4_route_matrix.sh <outdir> [rows]
#   ROUTE_STAGE=build  compile only (CPU, no lease) -- use this while a campaign holds the card
#   ROUTE_STAGE=run    run only, from binaries a previous build stage left in <outdir>
#   unset              both, in order
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:?usage: gemma4_route_matrix.sh <outdir> [rows]}
rows=${2:-}
mkdir -p "$out"
stage=${ROUTE_STAGE:-all}

src=$root/runtime/bench/nvidia/bf16_gemm_vs_cublas_bench.cu
common=(
  # -gencode, NOT -arch=sm_90a. For an EXECUTABLE (unlike the -cubin segment builds) -arch
  # makes nvcc run ptxas TWICE: once as -arch=compute_90 to assemble the embedded
  # forward-compat PTX, and compute_90 has no sm_90a features, so every wgmma in ws384 is
  # rejected there before the real sm_90a pass ever runs. -gencode emits SASS only.
  -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 -Xptxas=-v
  -I "$root/runtime/common" -I "$root/runtime/nvidia"
  -DPLOW_BENCH_WS384=1 -DPLOW_BENCH_GEMMA4_ALL=1 -DPLOW_BENCH_GEMMA4_26B=1
  -DPGM90_TMA_STAGES=3 -DPGM90_WS384_PREFETCH=1 -DPGM90_WS384_ISSUE_CURSOR=1
  -DPGM90_WS384_SMEPI=0
  -DPLOW_NV_GEMM_ONLY=1
)
if [ "$stage" != run ]; then
for arm in bf16 w8a8; do
  extra=()
  [ "$arm" = w8a8 ] && extra=(-DPLOW_BENCH_W8A8=1 -DPGM90_FP8_PROMOTE=1)
  echo "== building $arm"
  # The compile takes the CPU-quiet lock SHARED. Without it an nvcc build lands inside whatever
  # latency measurement is running -- that is how a certificate arm got contaminated on 2026-09-23,
  # and a build is exactly the perturbation the lock exists to exclude.
  "$root/scripts/bench/quiets.sh" /tmp/plow-cpu-quiet.lock \
    env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    "${common[@]}" "${extra[@]}" "$src" -o "$out/route_$arm" -lcublasLt -lcuda \
    >"$out/build_$arm.log" 2>&1 || { tail -40 "$out/build_$arm.log"; exit 1; }
done
fi
if [ "$stage" = build ]; then echo "built: $out/route_bf16 $out/route_w8a8"; exit 0; fi

lease=${PLOW_GPULEASE_BIN:-$root/../../../perf-data/tools/gpulease}
[ -x "$lease" ] || lease=/home/lava/plow/perf-data/tools/gpulease
for arm in bf16 w8a8; do
  echo "== running $arm"
  GPU_LEASE_TIMEOUT=${GPU_LEASE_TIMEOUT:-43200} "$lease" -n 1 "route-$arm" \
    env LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    "$root/scripts/bench/quietx.sh" /tmp/plow-cpu-quiet.lock \
    "$out/route_$arm" $rows 2>&1 | tee "$out/route_$arm.out"
done
echo "results: $out/route_bf16.out $out/route_w8a8.out"
