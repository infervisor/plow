#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "usage: $0 OBJECT_DIR HIPBLASLT_BF16_BIAS_OBJECT HIPBLASLT_BF16_NOBIAS_OBJECT (inside nix develop)" >&2
    exit 2
fi

out=$1
bias=$2
nobias=$3
[[ $(sha256sum "$bias") == 31409ab7a2a665257dbd33b7de5283b44150ada24e86377498e01e9b3fce9412* ]] || {
    echo "hipBLASLt bias object does not match the qualified gfx942 ABI" >&2
    exit 1
}
[[ $(sha256sum "$nobias") == aaddd9d254a4d2bd6a3f013eb81b6cedf04438511958aae8417d63e0f54224bb* ]] || {
    echo "hipBLASLt no-bias object does not match the qualified gfx942 ABI" >&2
    exit 1
}

mkdir -p "$out"
tmp=$(mktemp -d "$out/.gemma-lt.XXXXXX")
trap 'rm -r -- "$tmp"' EXIT
"${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --input="$bias" \
    --output="$tmp/glm_lt_gfx942.elf"
"$PLOW_BUNDLER" --unbundle --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --input="$nobias" \
    --output="$tmp/gemma_lt_nobias_gfx942.elf"

[[ $(sha256sum "$tmp/glm_lt_gfx942.elf") == efa5b0365bedc2effa52265c85eded14fb63febd9c067bab37138d99db607db5* ]]
[[ $(sha256sum "$tmp/gemma_lt_nobias_gfx942.elf") == 0ab860e928c070fa3ee22f5f91725539caaa0a12223296060b8a2bda818ace0d* ]]
mv "$tmp/glm_lt_gfx942.elf" "$out/glm_lt_gfx942.elf"
mv "$tmp/gemma_lt_nobias_gfx942.elf" "$out/gemma_lt_nobias_gfx942.elf"
