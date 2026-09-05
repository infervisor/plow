#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: build_gdn_prefill_sm90.sh <output-directory>}"
HERE="${PLOW_ROOT:-.}"
PYTHON="${PLOW_GDN_PYTHON:-python3}"
CUDA="${PLOW_CUDA_HOME:-/usr/local/cuda}"
DSL_LIB="${PLOW_GDN_DSL_LIB_DIR:?set to the installed CuTe DSL runtime library directory}"
CXX="${PLOW_GDN_CXX:-/usr/bin/g++}"
CXX_LIB="$(dirname "$(readlink -f "$("$CXX" -print-file-name=libstdc++.so)")")"
mkdir -p "$OUT"
if [ "${PLOW_GDN_SKIP_EXPORT:-0}" != "1" ]; then
    CUDA_VISIBLE_DEVICES='' LD_PRELOAD="$DSL_LIB/libcute_dsl_runtime.so" \
        "$PYTHON" "$HERE/scripts/export_gdn_prefill_sm90.py" "$OUT"
fi
env -i PATH="$CUDA/bin:/usr/bin:/bin" \
    "$CXX" -std=c++17 -O3 -shared -fPIC -Wl,-z,defs \
    "$HERE/runtime/nvidia/gdn_prefill.cpp" "$OUT/gdn_sm90.o" \
    -I "$OUT" -I "$CUDA/include" -L "$CUDA/lib64" -lcudart \
    -L "$DSL_LIB" -lcute_dsl_runtime -Wl,-rpath,"$CUDA/lib64" -Wl,-rpath,"$DSL_LIB" -Wl,-rpath,"$CXX_LIB" \
    -o "$OUT/libplow_gdn_prefill.so"
