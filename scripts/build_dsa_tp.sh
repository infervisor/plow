#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
    echo "usage: $0 OBJECT_DIR (inside nix develop)" >&2
    exit 2
fi
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$1"
stem=$(mktemp "$1/.dsa-tp.XXXXXX")
trap 'rm -f "$stem" "$stem.co" "$stem.elf"' EXIT
"${PLOW_HIPCC:?run inside nix develop}" --genco --offload-arch=gfx942 -O3 -w \
    -std=c++17 -I"$root/runtime/amd" -I"$root/runtime/common" \
    "$root/runtime/amd/dsa_tp_adapter.hip" -o "$stem.co"
"${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --input="$stem.co" --output="$stem.elf"
mv "$stem.elf" "$1/dsa_tp_adapter_gfx942.elf"
