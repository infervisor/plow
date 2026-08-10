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

# PLOW_NVCC_PATH replaces the clean PATH when the toolchain does not live in
# /usr (the nix sandbox has no /usr/bin at all — the host gcc must come from
# the caller). NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS pass through for the same
# reason: nix's nvcc gets its -ccbin that way, and `env -i` would strip it.
CLEAN=(env -i PATH="${PLOW_NVCC_PATH:-/usr/local/cuda/bin:/usr/bin:/bin}")
[ -n "${NVCC_PREPEND_FLAGS:-}" ] && CLEAN+=(NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS")
[ -n "${NVCC_APPEND_FLAGS:-}" ] && CLEAN+=(NVCC_APPEND_FLAGS="$NVCC_APPEND_FLAGS")
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
# Read the symbol table into a variable rather than piping it into `grep -q`.
# `grep -q` exits at the FIRST match, which closes the pipe while cuobjdump is
# still writing; cuobjdump then dies of SIGPIPE (141) and `set -o pipefail`
# promotes that to a pipeline failure — so the gate reports "$SYM not found"
# about an object whose symbol grep had just successfully matched.
#
# Whether it fires is pure scheduling luck: it needs the reader to exit between
# two of the writer's write() calls. cuobjdump emits ~1 KB here, one write, so it
# has never been seen on this side. The identical construct on the AMD side
# (~5 KB, two writes) failed 4-5 of 17 objects per run under `-j16`, every one of
# them with PIPESTATUS `141 0` — writer killed, grep exit 0, symbol present.
SYMS="$("${CLEAN[@]}" "$CUOBJDUMP" -symbols "$OUT" || true)"
grep -q "$SYM" <<<"$SYMS" ||
    fail "$SYM not found in $OUT — kernel name/signature changed; update exec::gpu's
       kernel-name constant (or set PLOW_NV_KERNEL / PLOW_NV_KERNEL_PF)."
echo "built $OUT ($(stat -c%s "$OUT") B), kernel $SYM present"
