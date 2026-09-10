#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 && $# != 3 ]]; then
    echo "usage: $0 OBJECT_DIR AITER_MOE_CODE_OBJECT [AITER_FLAT_CODE_OBJECT] (inside nix develop)" >&2
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
flat_object=${3:-}
if [[ -n "$flat_object" ]]; then
    actual=$(sha256sum "$flat_object")
    if [[ ${actual%% *} != be7052284094e7cedeb266afb24d4d6723bdf4234ac391b2e5b29473d8ee8f06 ]]; then
        echo "AITER flat object does not match the qualified gfx942 MoE ABI" >&2
        exit 1
    fi
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

if [[ -n "$flat_object" ]]; then
    target="$out/fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3.co"
    if ! cmp -s "$flat_object" "$target"; then
        cp "$flat_object" "$target"
    fi
fi
