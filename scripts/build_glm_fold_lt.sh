#!/usr/bin/env bash
set -euo pipefail
# The second argument is the bundled hipBLASLt code object or, as a serving set holds it, the
# already-unbundled image; both are hash-pinned.
if [[ $# != 2 ]]; then
    echo "usage: $0 OBJECT_DIR HIPBLASLT_FP32_CODE_OBJECT|glm_fold_lt_gfx942.elf (inside nix develop)" >&2
    exit 2
fi
bundled=7ef5a5bcb69a9eb6df58ba91af82ae0a2e54c2005ebc7bdc534b9b29113a24e7
image=209f4165a46672d5289816f0df6b41cf5a3f44d78452ff2a28c8cd7a0e1b77d1
actual=$(sha256sum "$2")
actual=${actual%% *}
[[ $actual == "$bundled" || $actual == "$image" ]] || { echo "unqualified FP32 GEMM object" >&2; exit 1; }
mkdir -p "$1"
tmp=$(mktemp -d "$1/.glm-fold.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
if [[ $actual == "$image" ]]; then
    cp "$2" "$tmp/gemm.elf"
else
    "${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
        --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --input="$2" --output="$tmp/gemm.elf"
fi
actual=$(sha256sum "$tmp/gemm.elf")
[[ ${actual%% *} == "$image" ]] || exit 1
"${PLOW_HIPCC:?run inside nix develop}" --genco --offload-arch=gfx942 -O3 -std=c++17 \
    -Iruntime/amd -Iruntime/common runtime/amd/glm_fold_adapter.hip -o "$tmp/adapter.co"
"${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
    --input="$tmp/adapter.co" --output="$tmp/adapter.elf"
mv "$tmp/gemm.elf" "$1/glm_fold_lt_gfx942.elf"
mv "$tmp/adapter.elf" "$1/glm_fold_adapter.elf"
