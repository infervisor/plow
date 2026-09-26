#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 && $# != 3 ]] || [[ $# == 3 && ${3:-} != --single-pass && ${3:-} != --bf16-gfx950 ]]; then
    echo "usage: $0 OBJECT_DIR PINNED_AITER_CODE_OBJECT [--single-pass|--bf16-gfx950] (inside nix develop)" >&2
    exit 2
fi
out=$1
object=$2
flags=()
arch=gfx942
object_name=mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co
expected=cd8fa62e18abada15beeeedd49357bcc1e9e2eee7353d038533f30cac93c3607
if [[ ${3:-} == --single-pass ]]; then
    flags=(-DPLOW_MLA_SPARSE_SINGLE_PASS=1)
elif [[ ${3:-} == --bf16-gfx950 ]]; then
    arch=gfx950
    object_name=mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps.co
    expected=b6d4181c3ed19750b22a02dc0d290727272ed53091678c5e75c39f45d9832cfd
fi
actual=$(sha256sum "$object")
if [[ ${actual%% *} != "$expected" ]]; then
    echo "AITER object does not match the pinned $arch ABI" >&2
    exit 1
fi
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
stem=$(mktemp "$out/.mla-sparse.XXXXXX")
trap 'rm -f "$stem" "$stem.co" "$stem.elf"' EXIT
"${PLOW_HIPCC:?run inside nix develop}" --genco --offload-arch="$arch" -O3 -w \
    "${flags[@]}" \
    -std=c++17 -I"$root/runtime/amd" -I"$root/runtime/common" \
    "$root/runtime/amd/mla_sparse_adapter.hip" \
    -o "$stem.co"
"${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
    --targets="hipv4-amdgcn-amd-amdhsa--$arch" --input="$stem.co" --output="$stem.elf"
mv "$stem.elf" "$out/mla_sparse_adapter_$arch.elf"
target="$out/$object_name"
if ! cmp -s "$object" "$target"; then
    cp "$object" "$target"
fi
