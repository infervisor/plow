#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "usage: $0 OBJECT_DIR AITER_MOE_CODE_OBJECT (inside nix develop)" >&2
    exit 2
fi
out=$1
object=$2
expected=65b4c0a0b290dd83039047c18e0bb86f4253790e926dce324ddb6b45a7b28650
actual=$(sha256sum "$object")
if [[ ${actual%% *} != "$expected" ]]; then
    echo "AITER object does not match the qualified gfx942 MoE ABI" >&2
    exit 1
fi
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
stem=$(mktemp "$out/.moe-aiter.XXXXXX")
trap 'rm -f "$stem" "$stem.co" "$stem.elf"' EXIT
"${PLOW_HIPCC:?run inside nix develop}" --genco --offload-arch=gfx942 -O3 -w \
    -std=c++17 -I"$root/runtime/amd" -I"$root/runtime/common" \
    "$root/runtime/amd/moe_aiter_adapter.hip" -o "$stem.co"
"${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --input="$stem.co" --output="$stem.elf"
mv "$stem.elf" "$out/moe_aiter_adapter_gfx942.elf"
target="$out/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co"
if ! cmp -s "$object" "$target"; then
    cp "$object" "$target"
fi
