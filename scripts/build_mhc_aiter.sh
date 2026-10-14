#!/usr/bin/env bash
# build_mhc_aiter.sh <objdir> <aiter_meta/csrc> — PLOW_GLM_HC_AITER object (mhc_aiter_gfx950.elf). nix develop.
set -euo pipefail
OUT=${1:?objdir}; C=${2:?aiter_meta/csrc}
HERE=$(cd "$(dirname "$0")/.." && pwd)
stem=$(mktemp "$OUT/.mhc-aiter.XXXXXX")
trap 'rm -f "$stem" "$stem.co"' EXIT
"${PLOW_HIPCC:?run inside nix develop}" --genco --offload-arch=gfx950 -O3 -w -std=c++20 \
  -I"$C" -I"$C/include" -I"$C/include/opus" -o "$stem.co" "$HERE/runtime/amd/mhc_aiter.hip"
"${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --input="$stem.co" --output="$OUT/mhc_aiter_gfx950.elf"
sha256sum "$OUT/mhc_aiter_gfx950.elf"
