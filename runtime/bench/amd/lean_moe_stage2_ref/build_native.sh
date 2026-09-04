#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HERE="$ROOT/runtime/bench/amd/lean_moe_stage2_ref"
OUT=${1:-/tmp/plow-moe2-native}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")

HIPCC=$(command -v hipcc)
TOOLROOT=$(cd "$(dirname "$HIPCC")/.." && pwd)
BUNDLER="$TOOLROOT/lib/llvm/bin/clang-offload-bundler"
READELF="$TOOLROOT/lib/llvm/bin/llvm-readelf"
OBJCOPY="$TOOLROOT/lib/llvm/bin/llvm-objcopy"

hipcc --genco --offload-arch=gfx950 -O3 -std=c++17 "$HERE/native_kernel.hip" -o "$OUT/kernel.bundle"
"$BUNDLER" --type=o --unbundle --input="$OUT/kernel.bundle" --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output="$OUT/kernel.co"
python3 "$HERE/native_manifest.py" "$OUT/kernel.co" "$OUT/manifest.json" "$READELF" "$OBJCOPY"
