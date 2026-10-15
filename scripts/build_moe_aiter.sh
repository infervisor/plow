#!/usr/bin/env bash
set -euo pipefail

# gfx950 / GLM-5.3-Flash (H4096): `--gfx950 OBJECT_DIR AITER_FMOE_SILU_DIR`, the directory holding
# vLLM ROCm AITER's aiter_meta/hsa/gfx950/fmoe/silu objects. Copies the pinned per-row-bucket
# tiles and builds the H4096/OCP adapter and align.
if [[ ${1:-} == --gfx950 ]]; then
    if [[ $# -ne 3 ]]; then
        echo "usage: $0 --gfx950 OBJECT_DIR AITER_FMOE_SILU_DIR (inside nix develop)" >&2
        exit 2
    fi
    out=$2
    src=$3
    root=$(cd "$(dirname "$0")/.." && pwd)
    mkdir -p "$out"
    while read -r hash name; do
        actual=$(sha256sum "$src/$name")
        if [[ ${actual%% *} != "$hash" ]]; then
            echo "AITER gfx950 object $name does not match the pinned hash" >&2
            exit 1
        fi
        cmp -s "$src/$name" "$out/$name" || cp "$src/$name" "$out/$name"
    done <<'PINS'
07df0af964cf26dde7635b6f524fd84701a15100f5b88fc9366599e20edeea31 fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_pf2_16x128.co
2b2c1852d84d47f4a03ad5814087be4b4b21e0acd9ee0efdd3f7792a6afe509a fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x128.co
6a583441c6bb139c6af20c5198614b7b02b74fdba370580a5a5bbe164bd7bb34 fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co
e442ee49fd88b5d165b8f8313d790c3ec7a7978140fa2306876612191429e413 fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_64x256.co
PINS
    stem=$(mktemp "$out/.moe-aiter.XXXXXX")
    trap 'rm -f "$stem" "$stem.co" "$stem.elf"' EXIT
    for kind in aiter align; do
        "${PLOW_HIPCC:?run inside nix develop}" --genco --offload-arch=gfx950 -O3 -w \
            -std=c++17 -DMOE_AITER_H=4096 -I"$root/runtime/amd" -I"$root/runtime/common" \
            "$root/runtime/amd/moe_${kind}_adapter.hip" -o "$stem.co"
        "${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
            --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --input="$stem.co" --output="$stem.elf"
        mv "$stem.elf" "$out/moe_${kind}_adapter_gfx950.elf"
    done
    exit 0
fi

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 OBJECT_DIR AITER_MOE_CODE_OBJECT [AITER_FLAT_CODE_OBJECT [AITER_TILE64_CODE_OBJECT]] (inside nix develop)" >&2
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
tile64_object=${4:-}
if [[ -n "$tile64_object" ]]; then
    actual=$(sha256sum "$tile64_object")
    if [[ ${actual%% *} != f8efb79a4ecfd80c7c4d6e20c797c7bdcdac26f1c22c825c640b239e07e88779 ]]; then
        echo "AITER 64-row object does not match the qualified gfx942 MoE ABI" >&2
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
# The native MoE align (PLOW_GLM_MOE_NATIVE_ALIGN): its own object, so the adapter above is unchanged.
"${PLOW_HIPCC:?run inside nix develop}" --genco --offload-arch=gfx942 -O3 -w \
    -std=c++17 -I"$root/runtime/amd" -I"$root/runtime/common" \
    "$root/runtime/amd/moe_align_adapter.hip" -o "$stem.co"
"${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --input="$stem.co" --output="$stem.elf"
mv "$stem.elf" "$out/moe_align_adapter_gfx942.elf"
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

if [[ -n "$tile64_object" ]]; then
    target="$out/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_psx_64x256.co"
    if ! cmp -s "$tile64_object" "$target"; then
        cp "$tile64_object" "$target"
    fi
fi
