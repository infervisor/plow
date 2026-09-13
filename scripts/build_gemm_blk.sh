#!/usr/bin/env bash
# Build the PLOW_GLM_GEMM_BLK objects into OBJECT_DIR: the activation-quant adapter from the tree,
# plus AITER's six gfx942 block-scale FP8 GEMM objects, each checked against the hash
# `crates/plowrt/src/exec/amd_gemm_blk.rs` pins.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 OBJECT_DIR AITER_FP8GEMM_BLOCKSCALE_DIR (inside nix develop)" >&2
    echo "  e.g. $0 out /workspace/aiter/hsa/gfx942/fp8gemm_blockscale" >&2
    exit 2
fi
out=$1
src=$2
declare -A pinned=(
    [32]=3d1acfd1e5bf6f16816334d8aede37f93af0cbb3f0d6030dddb1deacf7e38d8f
    [48]=f9a85f3cca7df1d2fea8b73f717a71674595b650bb4d9bafbb45b1298f62d841
    [64]=b3b0814cdfc6be1cc838ba7e9065aa763dc2dcd72eb0c26afda31efab2884736
    [80]=9f79c4151eab0d216595010507d671945ef8c1abc5e8271f026d21b155a1c5f1
    [96]=6cda76fcdafd257f73d9cb9cbcf22b4730077b8d41c5bc9c6b28942799c9e5af
    [128]=de905e406509c62f78d0760a36e762f7de89ded5d39fd3f98062eb58a8078b99
)
for tile in "${!pinned[@]}"; do
    object="$src/fp8gemm_bf16_blockscale_BpreShuffle_${tile}x128.co"
    actual=$(sha256sum "$object")
    if [[ ${actual%% *} != "${pinned[$tile]}" ]]; then
        echo "$object does not match the qualified gfx942 block-scale ABI" >&2
        exit 1
    fi
done
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
stem=$(mktemp "$out/.gemm-blk.XXXXXX")
trap 'rm -f "$stem" "$stem.co" "$stem.elf"' EXIT
"${PLOW_HIPCC:?run inside nix develop}" --genco --offload-arch=gfx942 -O3 -w \
    -std=c++17 -I"$root/runtime/amd" -I"$root/runtime/common" \
    "$root/runtime/amd/gemm_blk_adapter.hip" -o "$stem.co"
"${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --input="$stem.co" --output="$stem.elf"
mv "$stem.elf" "$out/gemm_blk_adapter_gfx942.elf"
for tile in "${!pinned[@]}"; do
    name="fp8gemm_bf16_blockscale_BpreShuffle_${tile}x128.co"
    if ! cmp -s "$src/$name" "$out/$name"; then
        cp "$src/$name" "$out/$name"
    fi
done
