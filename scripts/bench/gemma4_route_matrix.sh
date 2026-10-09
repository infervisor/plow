#!/usr/bin/env bash
# Build and run the Gemma-4 per-kernel ROUTE MATRIX on sm_90a: for every dense projection shape
# of the 12B, the 26B-A4B and the 31B, plow's shipped ws384 GEMM body vs cuBLASLt, in BOTH precisions.
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
# Run inside nix develop. Compilation uses its isolated nvcc environment; GPU arms enter gpuq.
#
# usage: scripts/bench/gemma4_route_matrix.sh <outdir> [rows]
#   ROUTE_STAGE=build  compile only (CPU, no lease) -- use this while a campaign holds the card
#   ROUTE_STAGE=run    run only, from binaries a previous build stage left in <outdir>
#   unset              both, in order
#   ROUTE_TC64=1       W8A8 only, default M64; M128 probes two TC64 passes vs quantization + Lt
#   ROUTE_ARM=bf16|w8a8|both  compile/run one or both precisions (default both)
#   PLOW_BENCH_SHAPE=name   queue just one named shape (for example g12_lmhead)
set -euo pipefail
: "${ROCM_PATH:?run inside nix develop}" "${CUDA_PATH:?CUDA toolkit missing}"
: "${PLOW_NVCC:?Nix nvcc missing}" "${PLOW_NVCC_PATH:?Nix nvcc PATH missing}"
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:?usage: gemma4_route_matrix.sh <outdir> [rows]}
rows=${2:-}
mkdir -p "$out"
stage=${ROUTE_STAGE:-all}
arms=(bf16 w8a8)
case "${ROUTE_ARM:-both}" in
  bf16) arms=(bf16) ;;
  w8a8) arms=(w8a8) ;;
  both) ;;
  *) echo "ROUTE_ARM must be bf16, w8a8, or both" >&2; exit 2 ;;
esac
if [ "${ROUTE_TC64:-0}" = 1 ]; then
  [ "${ROUTE_ARM:-both}" != bf16 ] || { echo "ROUTE_TC64 requires w8a8" >&2; exit 2; }
  arms=(w8a8)
  rows=${rows:-64}
fi
if [ "${PLOW_BENCH_SHAPE:-}" = g12_lmhead ] && [ "${ROUTE_ARM:-both}" != bf16 ]; then
  echo "g12_lmhead requires ROUTE_ARM=bf16" >&2; exit 2
fi

src=$root/runtime/bench/nvidia/bf16_gemm_vs_cublas_bench.cu
common=(
  # -gencode, NOT -arch=sm_90a. For an EXECUTABLE (unlike the -cubin segment builds) -arch
  # makes nvcc run ptxas TWICE: once as -arch=compute_90 to assemble the embedded
  # forward-compat PTX, and compute_90 has no sm_90a features, so every wgmma in ws384 is
  # rejected there before the real sm_90a pass ever runs. -gencode emits SASS only.
  -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 -Xptxas=-v
  -I "$root/runtime/common" -I "$root/runtime/nvidia"
  -DPLOW_BENCH_WS384=1 -DPLOW_BENCH_GEMMA4_ALL=1 -DPLOW_BENCH_GEMMA4_26B=1 -DPLOW_BENCH_GEMMA4_31B=1
  -DPGM90_TMA_STAGES=3 -DPGM90_WS384_PREFETCH=1 -DPGM90_WS384_ISSUE_CURSOR=1
  -DPGM90_WS384_SMEPI=0
  -DPLOW_NV_GEMM_ONLY=1
)
if [ "${PLOW_BENCH_SHAPE:-}" = g12_lmhead ]; then
  common+=(-DPLOW_BENCH_GEMMA4_HEAD=1)
fi
if [ "$stage" != run ]; then
for arm in "${arms[@]}"; do
  extra=()
  [ "$arm" = w8a8 ] && extra=(-DPLOW_BENCH_W8A8=1 -DPGM90_FP8_PROMOTE=1)
  [ "${ROUTE_TC64:-0}" = 1 ] && extra+=(-DPLOW_NV_FP8_DECODE_TC64=1 -DPLOW_NV_QUANT_FP8_VLLM=1)
  echo "== building $arm"
  # The compile takes the CPU-quiet lock SHARED. Without it an nvcc build lands inside whatever
  # latency measurement is running -- that is how a certificate arm got contaminated on 2026-09-23,
  # and a build is exactly the perturbation the lock exists to exclude.
  printf '%q ' env "NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}" "NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-}" "$PLOW_NVCC" "${common[@]}" "${extra[@]}" "$src" -o "$out/route_$arm" -lcublasLt -lcuda -L "$CUDA_PATH/lib/stubs" > "$out/compile_$arm.txt"
  "$root/scripts/bench/quiets.sh" /tmp/plow-cpu-quiet.lock \
    env -i PATH="$PLOW_NVCC_PATH" NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" \
    NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}" "$PLOW_NVCC" \
    "${common[@]}" "${extra[@]}" "$src" -o "$out/route_$arm" -lcublasLt -lcuda \
    -L "$CUDA_PATH/lib/stubs" \
    >"$out/build_$arm.log" 2>&1 || { tail -40 "$out/build_$arm.log"; exit 1; }
done
fi
if [ "$stage" = build ]; then
  for arm in "${arms[@]}"; do echo "built: $out/route_$arm"; done
  exit 0
fi

driver=$(realpath "${PLOW_LIBCUDA:-/usr/lib/x86_64-linux-gnu/libcuda.so.1}")
driver_dir=$(realpath -m "$out/cuda-driver")
mkdir -p "$driver_dir"
ln -sfn "$driver" "$driver_dir/libcuda.so"
ln -sfn "$driver" "$driver_dir/libcuda.so.1"
for arm in "${arms[@]}"; do
  echo "== queueing $arm"
  shape_env=()
  [ -z "${PLOW_BENCH_SHAPE:-}" ] || shape_env=("PLOW_BENCH_SHAPE=$PLOW_BENCH_SHAPE")
  python3 "$root/scripts/bench/gpuq.py" submit "route-$arm" 1 \
    timeout --kill-after=30s 1800 nix develop --command bash -c 'out=$1; shift; exec "$@" > "$out" 2>&1' \
    route "$out/route_$arm.out" env LD_LIBRARY_PATH="$driver_dir:$CUDA_PATH/lib:${LD_LIBRARY_PATH:-}" "${shape_env[@]}" \
    "$root/scripts/bench/quietx.sh" /tmp/plow-cpu-quiet.lock "$out/route_$arm" $rows
done
for arm in "${arms[@]}"; do echo "queued result: $out/route_$arm.out"; done
