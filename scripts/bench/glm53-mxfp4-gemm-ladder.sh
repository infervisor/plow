#!/usr/bin/env bash
# GLM-5.3 MXFP4 TP8 per-rank dense GEMMs at the vLLM 0.29 dtype contract, prefill + decode rungs,
# on ONE leased GPU, through gemm_tile_sweep (every compiled tile, f64 spot-check, JSONL samples).
#   build (CPU):  glm53-mxfp4-gemm-ladder.sh build OBJ
#   run (queue):  gpuq.py submit glm53-gemm-ladder 1 glm53-mxfp4-gemm-ladder.sh run OBJ OUT
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mode=${1:?build|run}
obj=$(realpath -m -- "${2:?OBJ}")
case "$mode" in
build)
    : "${ROCM_PATH:?run build inside nix develop}"
    mkdir -p "$obj"
    hipcc --offload-arch=gfx950 -O3 -w --genco "$repo/runtime/amd/test_kernels.hip" -o "$obj/tk.co" \
        -I"$repo/runtime/amd" -I"$repo/runtime/common"
    bun=$(command -v clang-offload-bundler || echo "$ROCM_PATH/llvm/bin/clang-offload-bundler")
    "$bun" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
        --input="$obj/tk.co" --output="$obj/test_kernels.elf"
    # System cc and system ROCm, cleared env: nix's CPATH shadows the glibc/HSA this links.
    env -i PATH=/usr/bin:/bin HOME="$HOME" cc -O2 -std=gnu11 -o "$obj/gemm_tile_sweep" \
        "$repo/runtime/bench/gemm/gemm_tile_sweep.c" "$repo/runtime/amd/hsa_backend.c" \
        -I"$repo/runtime/bench" -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
    sha256sum "$obj/test_kernels.elf" "$obj/gemm_tile_sweep" > "$obj/build.sha256"
    ;;
run)
    out=$(realpath -m -- "${3:?OUT}")
    mkdir -p "$out"
    cd "$obj"
    export LD_LIBRARY_PATH=/opt/rocm/lib
    export PLOW_GEMM_JSONL="$out/samples.jsonl"
    rows=${ROWS:-"1 8 16 32 64 128 1024 2048 4096 8192 16384"}
    # label N K quant: fused q_a+kv_a (vLLM N=2624), q_b, o_proj, indexer wq_b, shared expert,
    # dense layers, router (BF16, FP32 out in production; timed here as BF16 GEMM).
    shapes=(
        "qkv_a 2624 6144 BlockFp8" "q_b 2048 2048 BlockFp8" "o_proj 6144 2048 BlockFp8"
        "idx_wq_b 4096 2048 BlockFp8"
        "shared_gate_up 512 6144 Mxfp4" "shared_down 6144 256 Mxfp4"
        "dense_gate_up 3072 6144 Mxfp4" "dense_down 6144 1536 Mxfp4"
        "router 256 6144 None" "idx_wk_w 160 6144 None"
    )
    for m in $rows; do
        for s in "${shapes[@]}"; do
            read -r label n k q <<< "$s"
            ./gemm_tile_sweep "$m" "$n" "$k" "$label" "$q" > "$out/$label-M$m.log" 2>&1 \
                || echo "FAIL $label M$m rc=$?" >> "$out/failures.txt"
        done
    done
    test ! -e "$out/failures.txt" || cat "$out/failures.txt"
    ;;
esac
