#!/usr/bin/env bash
# build_sm120_cubin.sh — the prebuilt interpreter modules `plowrt serve` loads.
#
#   scripts/build_sm120_cubin.sh <out.cubin> [extra nvcc/-D flags...]
#
# DEPRECATED — SCHEDULED FOR REMOVAL. The build lives in runtime/CMakeLists.txt
# (`-DPLOW_SM120_CUBIN=ON`, target `sm120_cubins`). Do NOT add new callers; drive
# CMake directly. This wrapper exists only so in-flight callers keep working
# while they are migrated. See plans/interp-build-system.md for the checklist.
#
# THIN WRAPPER. The build itself now lives in runtime/CMakeLists.txt
# (`-DPLOW_SM120_CUBIN=ON`, target `sm120_cubins`) and this script configures and
# drives it, then copies the artifacts to the paths its callers expect. It used
# to carry its own hand-written copy of every object's define set, independently
# of CMake's — and the two drifted. That drift WAS the bug class:
#   * -DPLOW_NV_FA_GF_FULL=4 reached the plain decode object here but not the
#     _fp8kv sibling that an fp8-KV asset actually loads;
#   * the fp8-KV prefill object hardcoded -DPLOW_NV_FA_PIPE=0, so
#     PLOW_FP8_KV_FASTPF was unreachable through this script.
# Both are structurally impossible now: see the cubin section of
# runtime/CMakeLists.txt, where every set is derived from one base + two axes.
#
# The CLI is unchanged, so every existing caller and doc keeps working:
#   <out.cubin>            decode object; <out>_pf.cubin is the prefill object,
#                          sample_sm120.cubin lands in the same directory
#   PLOW_ROOT              source root (a worktree builds its OWN source)
#   PLOW_EXTRA_DEFINES     raw extra nvcc flags (the decode tuner's sweep knob)
#   PLOW_BUILD_FP8KV=1     also emit <out>_fp8kv.cubin / <out>_pf_fp8kv.cubin
# Extra arguments are forwarded to cmake, so the README's
#   build_sm120_cubin.sh <out> -DPLOW_NV_W8A8=ON -DPLOW_FP8_KV=ON
# does what it reads like.
#
# nvcc still runs under a clean environment (runtime/cmake/nvcc_cubin.sh): under
# `nix develop`, CPATH points nvcc's host pass at nix glibc headers that conflict
# with the CUDA math headers.
set -euo pipefail

OUT="${1:?usage: build_sm120_cubin.sh <out.cubin> [cmake -D flags...]}"
shift
HERE="${PLOW_ROOT:-/root/plow}"
CMAKE="${CMAKE:-cmake}"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

FP8KV="${PLOW_BUILD_FP8KV:-0}"

"$CMAKE" -S "$HERE/runtime" -B "$BUILD_DIR" \
    -DPLOW_SM120_CUBIN=ON \
    -DPLOW_SM120_CUBIN_FP8KV="$([ "$FP8KV" = 1 ] && echo ON || echo OFF)" \
    -DPLOW_CUBIN_DIR="$BUILD_DIR/cubin" \
    -DPLOW_EXTRA_DEFINES="${PLOW_EXTRA_DEFINES:-}" \
    "$@" >/dev/null
"$CMAKE" --build "$BUILD_DIR" --target sm120_cubins

# Place the artifacts where the callers of this script have always found them.
OUT_DIR="$(dirname "$OUT")"
mkdir -p "$OUT_DIR"
cp "$BUILD_DIR/cubin/interp_sm120.cubin" "$OUT"
cp "$BUILD_DIR/cubin/interp_sm120_pf.cubin" "${OUT%.cubin}_pf.cubin"
[ -f "$BUILD_DIR/cubin/sample_sm120.cubin" ] &&
    cp "$BUILD_DIR/cubin/sample_sm120.cubin" "$OUT_DIR/sample_sm120.cubin"
if [ "$FP8KV" = 1 ]; then
    cp "$BUILD_DIR/cubin/interp_sm120_fp8kv.cubin" "${OUT%.cubin}_fp8kv.cubin"
    cp "$BUILD_DIR/cubin/interp_sm120_pf_fp8kv.cubin" "${OUT%.cubin}_pf_fp8kv.cubin"
fi
exit 0
