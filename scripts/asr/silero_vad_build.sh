#!/usr/bin/env bash
# Build the Silero VAD v5 packet (`--asr-vad-packet`; runs on the CPU, no device object). Run
# inside `nix develop` with a Python that has torch, silero-vad and safetensors.
#
#   scripts/asr/silero_vad_build.sh <python> <fresh dir>
set -euo pipefail
python=$1
out=$2
repo=$(cd "$(dirname "$0")/../.." && pwd)
[ -e "$out" ] && [ -n "$(ls -A "$out")" ] && { echo "$out exists and is not empty" >&2; exit 2; }
mkdir -p "$out"
out=$(realpath "$out")

cd "$repo"
"$python" -I scripts/asr/silero_export.py "$out"
cargo build --release -p plowrt --example asr_silero_vad_compile
target=${CARGO_TARGET_DIR:-$repo/target}
"$target/release/examples/asr_silero_vad_compile" "$out/model.safetensors" "$out/silero_vad.pkt"
git rev-parse HEAD > "$out/commit"
sha256sum "$out/silero_vad.pkt" > "$out/sha256"
