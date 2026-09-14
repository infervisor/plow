#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "usage: $0 OBJECT_DIR HIPBLASLT_BF16_CODE_OBJECT (inside nix develop)" >&2
    exit 2
fi
expected_compressed=31409ab7a2a665257dbd33b7de5283b44150ada24e86377498e01e9b3fce9412
actual=$(sha256sum "$2")
if [[ ${actual%% *} != "$expected_compressed" ]]; then
    echo "hipBLASLt object does not match the qualified gfx942 projection ABI" >&2
    exit 1
fi
mkdir -p "$1"
stem=$(mktemp "$1/.glm-lt.XXXXXX")
trap 'rm -f "$stem" "$stem.elf"' EXIT
"${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --input="$2" --output="$stem.elf"
actual=$(sha256sum "$stem.elf")
if [[ ${actual%% *} != efa5b0365bedc2effa52265c85eded14fb63febd9c067bab37138d99db607db5 ]]; then
    echo "unbundled hipBLASLt object does not match qualified image" >&2
    exit 1
fi
mv "$stem.elf" "$1/glm_lt_gfx942.elf"
