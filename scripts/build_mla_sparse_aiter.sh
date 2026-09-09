#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "usage: $0 OBJECT_DIR AITER_QH8_CODE_OBJECT (inside nix develop)" >&2
    exit 2
fi
out=$1
object=$2
expected=cd8fa62e18abada15beeeedd49357bcc1e9e2eee7353d038533f30cac93c3607
actual=$(sha256sum "$object")
if [[ ${actual%% *} != "$expected" ]]; then
    echo "AITER object does not match the qualified gfx942 QH8 ABI" >&2
    exit 1
fi
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
stem=$(mktemp "$out/.mla-sparse.XXXXXX")
trap 'rm -f "$stem" "$stem.co" "$stem.elf"' EXIT
"${PLOW_HIPCC:?run inside nix develop}" --genco --offload-arch=gfx942 -O3 -w \
    -std=c++17 -I"$root/runtime/amd" -I"$root/runtime/common" \
    "$root/runtime/amd/mla_sparse_adapter.hip" \
    -o "$stem.co"
"${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --input="$stem.co" --output="$stem.elf"
mv "$stem.elf" "$out/mla_sparse_adapter_gfx942.elf"
target="$out/mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co"
if ! cmp -s "$object" "$target"; then
    cp "$object" "$target"
fi
