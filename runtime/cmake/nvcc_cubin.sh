#!/usr/bin/env bash
# nvcc_cubin.sh — compile ONE .cubin and gate it on its kernel symbol.
#
#   nvcc_cubin.sh <nvcc> <out.cubin> <kernel-symbol> <optional|required> <nvcc args...>
#
# Runs nvcc with a CLEAN environment: under `nix develop`, CPATH points nvcc's
# host pass at nix glibc headers that conflict with the CUDA math headers, and
# the compile fails in a way that looks like a CUDA installation problem. Same
# `env -i` scripts/build_sm120_cubin.sh has always used.
#
# The symbol gate is not decoration: `plowrt serve` resolves the kernel by NAME
# (exec::gpu's decode_symbol/prefill_symbol), so a cubin that built fine but
# whose mangled name moved is a runtime failure, not a build one.
#
# `optional` downgrades every failure to a warning (the device sampler: serving
# falls back to host sampling). `required` is fatal.
set -euo pipefail

NVCC="${1:?nvcc path}"
OUT="${2:?out.cubin}"
SYM="${3:?kernel symbol}"
REQ="${4:?optional|required}"
shift 4

CLEAN=(env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin)
CUOBJDUMP="$(dirname "$NVCC")/cuobjdump"

fail() {
    if [ "$REQ" = optional ]; then
        echo "WARN: $*" >&2
        exit 0
    fi
    echo "FATAL: $*" >&2
    exit 1
}

mkdir -p "$(dirname "$OUT")"
"${CLEAN[@]}" "$NVCC" "$@" -o "$OUT" || fail "nvcc failed for $OUT"
"${CLEAN[@]}" "$CUOBJDUMP" -symbols "$OUT" | grep -q "$SYM" ||
    fail "$SYM not found in $OUT — kernel name/signature changed; update exec::gpu's
       kernel-name constant (or set PLOW_NV_KERNEL / PLOW_NV_KERNEL_PF)."
echo "built $OUT ($(stat -c%s "$OUT") B), kernel $SYM present"
