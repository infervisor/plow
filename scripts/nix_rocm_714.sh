#!/usr/bin/env bash

plow_rocm_fail() {
    echo "FAIL: $*" >&2
    return 2
}

plow_require_nix_tool() {
    local name="$1" path="$2" target
    target="$(readlink -f "$path")"
    case "$target" in
        /nix/store/*) ;;
        *) plow_rocm_fail "$name must resolve into /nix/store, got ${target:-missing}" ;;
    esac
    [ -x "$target" ] || plow_rocm_fail "$name is not executable at $target"
}

plow_init_rocm_714() {
    [ -n "${IN_NIX_SHELL:-}" ] || plow_rocm_fail "run through nix develop"
    [ "${PLOW_TOOLCHAIN_LABEL:-}" = "rocm-7.14.0-nix" ] ||
        plow_rocm_fail "expected ROCm 7.14.0 from the flake, got ${PLOW_TOOLCHAIN_LABEL:-unset}"
    : "${ROCM_PATH:?nix develop did not set ROCM_PATH}"
    : "${PLOW_HIPCC:?nix develop did not set PLOW_HIPCC}"
    : "${PLOW_BUNDLER:?nix develop did not set PLOW_BUNDLER}"
    : "${PLOW_READELF:?nix develop did not set PLOW_READELF}"
    : "${PLOW_HOST_CC:?nix develop did not set PLOW_HOST_CC}"

    case "$(readlink -f "$ROCM_PATH")" in
        /nix/store/*) ;;
        *) plow_rocm_fail "ROCM_PATH must resolve into /nix/store, got $ROCM_PATH" ;;
    esac

    PLOW_K3_HIPCC="$PLOW_HIPCC"
    PLOW_K3_BUNDLER="$PLOW_BUNDLER"
    PLOW_K3_READELF="$PLOW_READELF"
    PLOW_K3_HOST_CC="$(command -v "$PLOW_HOST_CC")"
    PLOW_K3_ROCM="$(readlink -f "$ROCM_PATH")"
    PLOW_K3_ARCH="${PLOW_HIP_ARCH:-gfx942}"
    [ "$PLOW_K3_ARCH" = gfx942 ] ||
        plow_rocm_fail "K3 MI325X gates require gfx942, got $PLOW_K3_ARCH"

    plow_require_nix_tool hipcc "$PLOW_K3_HIPCC"
    plow_require_nix_tool clang-offload-bundler "$PLOW_K3_BUNDLER"
    plow_require_nix_tool llvm-readelf "$PLOW_K3_READELF"
    plow_require_nix_tool host-cc "$PLOW_K3_HOST_CC"
    case "$("$PLOW_K3_HIPCC" --version)" in
        *"HIP version: 7.14."*) ;;
        *) plow_rocm_fail "expected HIP 7.14 from the flake" ;;
    esac

    export PLOW_K3_HIPCC PLOW_K3_BUNDLER PLOW_K3_READELF PLOW_K3_HOST_CC
    export PLOW_K3_ROCM PLOW_K3_ARCH
}

plow_assert_nix_binary() {
    local binary="$1" dynamic
    dynamic="$("$PLOW_K3_READELF" -d "$binary")"
    case "$dynamic" in
        *"/opt/"*|*"/usr/"*) plow_rocm_fail "$binary contains a non-Nix dynamic path" ;;
    esac
}

plow_audit_gfx942_decode_object() {
    local object="$1" expect="$2" headers symbols notes vgpr lds
    headers="$("$PLOW_K3_READELF" -h "$object")"
    case "$headers" in
        *gfx942*) ;;
        *) plow_rocm_fail "$object is not a gfx942 code object" ;;
    esac
    symbols="$("$PLOW_K3_READELF" -sW "$object")"
    grep -qE "FUNC .* plow_interp_dec_gfx942$" <<<"$symbols" ||
        plow_rocm_fail "$object does not export plow_interp_dec_gfx942"
    notes="$("$PLOW_K3_READELF" --notes "$object")"
    vgpr="$(sed -n "s/.*\\.vgpr_count: *//p" <<<"$notes" | head -1)"
    lds="$(sed -n "s/.*\\.group_segment_fixed_size: *//p" <<<"$notes" | head -1)"
    [ -n "$vgpr" ] && [ "$vgpr" -le 256 ] ||
        plow_rocm_fail "$object is over the gfx942 decode register cliff (${vgpr:-missing} VGPR)"
    [ -n "$lds" ] && [ "$lds" -le 65536 ] ||
        plow_rocm_fail "$object is over the gfx942 LDS ceiling (${lds:-missing} bytes)"
    python3 "$(dirname "$expect")/asm_audit.py" --expect "$expect" "$object"
}
