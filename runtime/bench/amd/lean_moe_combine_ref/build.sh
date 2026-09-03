#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

root=$(cd "$(dirname "$0")/../../../.." && pwd)
out=$1
mkdir -p "$out"

hipcc=${HIPCC:-$(command -v hipcc)}
toolroot=$(cd "$(dirname "$hipcc")/.." && pwd)
bundler="$toolroot/lib/llvm/bin/clang-offload-bundler"
"$hipcc" -O3 --offload-arch=gfx950 -std=c++17 -x hip --genco \
  -I"$root/runtime/amd" \
  "$root/runtime/bench/amd/lean_moe_combine_ref/kernel.hip" \
  -o "$out/kernel.bundle"
"$bundler" --type=o --unbundle --input="$out/kernel.bundle" \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output="$out/kernel.co"
"$hipcc" -O3 --offload-arch=gfx950 -std=c++17 -x hip --genco \
  -I"$root/runtime/amd" \
  "$root/runtime/bench/amd/lean_moe_combine_ref/control_kernel.hip" \
  -o "$out/control.bundle"
"$bundler" --type=o --unbundle --input="$out/control.bundle" \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output="$out/control.co"

python3 "$root/runtime/bench/amd/lean_moe_combine_ref/manifest.py" \
  "$out/kernel.co" "$out/control.co" "$out/manifest.json"
