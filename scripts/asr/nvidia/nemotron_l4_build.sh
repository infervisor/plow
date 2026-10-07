#!/usr/bin/env bash
# Build recipes/infervisor/nemotron-3.5-asr/sm89-{l4,l40s}-tp1.toml: the RNNT packet pipeline
# (offline buckets + cache-aware stream programs) and the sm_89 speech object beside it. Run inside
# `nix develop`. <n_cu>: 58 (L4, default) or 142 (L40S).
#
#   scripts/asr/nvidia/nemotron_l4_build.sh <nemotron-3.5-asr-streaming-0.6b.q8_0.gguf> <fresh dir> [n_cu]
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
cargo build --release -p plowrt --features cuda,gguf,dist --example asr_nemotron_pipeline_compile
target=${CARGO_TARGET_DIR:-$repo/target}
"$target/release/examples/asr_nemotron_pipeline_compile" "$gguf" "$buckets" "$out/nemotron.pkt" 16 "$n_cu"

# Speech ops 163..178 plus 196 (CopyColsF32: the stream programs' [cache|new] windows); a packet
# with stream programs fails on an object without bit 33.
cmake -S runtime -B "$out/cmake" "-DPLOW_CUBIN_NVCC=$(command -v nvcc)" -DPLOW_SM89_CUBIN=ON -DPLOW_CUBIN_SPEECH=ON -DPLOW_CUBIN_ARCH=sm_89 \
  -DPLOW_CUBIN_GEMMA=OFF "-DPLOW_EXTRA_DEFINES=-DPLOW_SPEECH_OPS=0x20000ffffull" > "$out/cmake.log"
cmake --build "$out/cmake" --target nv_cubins >> "$out/cmake.log"
cp "$out/cmake/cubin/interp_sm89_speech.cubin" "$out/"
cp "$gguf" "$out/tokenizer.q8_0.gguf"
git rev-parse HEAD > "$out/commit"
sha256sum "$out"/nemotron.pkt "$out"/interp_sm89_speech.cubin > "$out/sha256"
