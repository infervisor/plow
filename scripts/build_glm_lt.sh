#!/usr/bin/env bash
set -euo pipefail
# gfx942: build_glm_lt.sh OBJECT_DIR HIPBLASLT_BF16_CODE_OBJECT
# gfx950: build_glm_lt.sh --gfx950 OBJECT_DIR [HIPBLASLT_BF16_CODE_OBJECT] (default: the host ROCm 10
#         TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk object)
arch=gfx942
expected_compressed=31409ab7a2a665257dbd33b7de5283b44150ada24e86377498e01e9b3fce9412
expected_elf=efa5b0365bedc2effa52265c85eded14fb63febd9c067bab37138d99db607db5
if [[ ${1:-} == --gfx950 ]]; then
    shift
    arch=gfx950
    expected_compressed=1a7eed7856434d880fcb00d0754f2c1561b8c61a6d5eb89735a5c06e1ae4440e
    expected_elf=379704897bfc6227f36260ac81cde6cfe9d608bd373623dd974bd4c852b1f5fe
    set -- "${1:-}" "${2:-/opt/rocm/lib/hipblaslt/library/gfx950/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_ID75a0_gfx950.co}"
fi
if [[ $# != 2 || -z $1 ]]; then
    echo "usage: $0 [--gfx950] OBJECT_DIR HIPBLASLT_BF16_CODE_OBJECT (inside nix develop)" >&2
    exit 2
fi
actual=$(sha256sum "$2")
if [[ ${actual%% *} != "$expected_compressed" ]]; then
    echo "hipBLASLt object does not match the qualified $arch projection ABI" >&2
    exit 1
fi
mkdir -p "$1"
stem=$(mktemp "$1/.glm-lt.XXXXXX")
trap 'rm -f "$stem" "$stem.elf"' EXIT
"${PLOW_BUNDLER:?run inside nix develop}" --unbundle --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--$arch --input="$2" --output="$stem.elf"
actual=$(sha256sum "$stem.elf")
if [[ ${actual%% *} != "$expected_elf" ]]; then
    echo "unbundled hipBLASLt object does not match qualified image" >&2
    exit 1
fi
if [[ $arch == gfx950 ]]; then
    # Only the kernels runtime/amd/glm_lt_gfx950.json names (6.8 MB, not the 503 MB library).
    here=$(cd "$(dirname "$0")" && pwd)
    sub=$(mktemp -d "$1/.glm-lt-sub.XXXXXX")
    trap 'rm -rf "$stem" "$stem.elf" "$sub"' EXIT
    python3 -c 'import json, sys
seen = []
for s in json.load(open(sys.argv[1])):
    if s["name"] not in seen:
        seen.append(s["name"])
print("\n".join(seen))' "$here/../runtime/amd/glm_lt_gfx950.json" > "$sub/names.txt"
    "$ROCM_PATH/llvm/bin/llvm-readelf" --notes "$stem.elf" > "$sub/notes.txt"
    python3 "$here/glm_lt_subset.py" "$stem.elf" "$sub/notes.txt" "$sub/names.txt" "$sub" > /dev/null
    "$ROCM_PATH/llvm/bin/llvm-mc" -triple=amdgcn-amd-amdhsa -mcpu=gfx950 -filetype=obj "$sub/subset.s" -o "$sub/subset.o"
    "$ROCM_PATH/llvm/bin/ld.lld" -shared "$sub/subset.o" -o "$stem.elf"
    actual=$(sha256sum "$stem.elf")
    if [[ ${actual%% *} != 88995bce9413c67da6df1a2d7e41aa95b0ba73f4ee3626de4a26e025580f1b0f ]]; then
        echo "gfx950 hipBLASLt subset does not match the qualified image" >&2
        exit 1
    fi
fi
mv "$stem.elf" "$1/glm_lt_$arch.elf"
