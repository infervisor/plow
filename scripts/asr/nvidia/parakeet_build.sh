#!/usr/bin/env bash
# Build Parakeet TDT (parakeet-tdt-0.6b-v3) for sm_89: the rnnt.greedy.v1 packet (full-context
# FastConformer, folded batch norm, TDT joint) and the speech object beside it. Run inside
# `nix develop`. <n_cu>: 58 (L4, default) or 142 (L40S).
#
#   scripts/asr/nvidia/parakeet_build.sh <parakeet-tdt-0.6b-v3.q8_0.gguf> <fresh dir> [n_cu]
set -euo pipefail
gguf=$(realpath "$1")
out=$2
n_cu=${3:-58}
repo=$(cd "$(dirname "$0")/../../.." && pwd)
buckets=200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000
[ -e "$out" ] && [ -n "$(ls -A "$out")" ] && { echo "$out exists and is not empty" >&2; exit 2; }
mkdir -p "$out"
out=$(realpath "$out")

cd "$repo"
cargo build --release -p plowrt --features cuda,gguf,dist --example asr_parakeet_pipeline_compile
target=${CARGO_TARGET_DIR:-$repo/target}
"$target/release/examples/asr_parakeet_pipeline_compile" "$gguf" "$buckets" "$out/parakeet.pkt" 16 "$n_cu"

# Speech ops 163..178, CopyColsF32 (196, the TDT joint split) and Conv1dF32 (197, the centred
# depthwise convolutions).
cmake -S runtime -B "$out/cmake" "-DPLOW_CUBIN_NVCC=$(command -v nvcc)" -DPLOW_SM89_CUBIN=ON -DPLOW_CUBIN_SPEECH=ON -DPLOW_CUBIN_ARCH=sm_89 \
  -DPLOW_CUBIN_GEMMA=OFF "-DPLOW_EXTRA_DEFINES=-DPLOW_SPEECH_OPS=0x60000ffffull" > "$out/cmake.log"
cmake --build "$out/cmake" --target nv_cubins >> "$out/cmake.log"
cp "$out/cmake/cubin/interp_sm89_speech.cubin" "$out/"
cp "$gguf" "$out/tokenizer.q8_0.gguf"
git rev-parse HEAD > "$out/commit"
sha256sum "$out"/parakeet.pkt "$out"/interp_sm89_speech.cubin > "$out/sha256"
