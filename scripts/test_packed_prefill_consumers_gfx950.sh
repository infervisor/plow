#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
HSACO=${1:?usage: test_packed_prefill_consumers_gfx950.sh <packed-family-hsaco-dir>}
HIPCC=${PLOW_HIPCC:-hipcc}
ROCM=$(cd "$(dirname "$(command -v "$HIPCC")")/.." && pwd)
READELF=${PLOW_LLVM_READELF:-$ROCM/lib/llvm/bin/llvm-readelf}
OUT=${TMPDIR:-/tmp}/plow-packed-prefill-consumers

check_object() {
  local object=$1
  shift
  [[ -f "$object" ]] || { echo "FATAL: missing $object" >&2; exit 1; }
  local symbols
  symbols=$("$READELF" -sW "$object")
  local marker
  for marker in plow_packed_prefill_abi_1 "$@"; do
    grep -qE "OBJECT .* ${marker}\$" <<<"$symbols" || {
      echo "FATAL: $object is missing required marker $marker" >&2
      exit 1
    }
  done
}

check_object "$HSACO/interp_packed_mla_norm.elf" \
  plow_packed_prefill_mla_norm_segments_1
check_object "$HSACO/interp_packed_mla_flash.elf" \
  plow_packed_prefill_mla_flash_segments_1
check_object "$HSACO/interp_packed_kda.elf" \
  plow_packed_prefill_kda_serial_segments_1 plow_packed_prefill_kda_chunk_segments_1 \
  plow_packed_prefill_kda_consumers_1 plow_kda_chunk_bt64_arm_1

"$HIPCC" --offload-arch=gfx950 -O3 -w -DPACKED_MLA_DEVICE --genco \
  "$ROOT/runtime/tests/packed_mla_gfx950_test.hip" -o "$OUT.co" \
  -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common"
"$HIPCC" -O2 -w "$ROOT/runtime/tests/packed_mla_gfx950_test.hip" \
  -o "$OUT" -lamdhip64

[[ ${PLOW_CONSUMER_SUITE_COMPILE_ONLY:-0} = 1 ]] && exit 0
exec "$ROOT/perf-data/tools/gpulease" -n 1 packed-prefill-consumers "$OUT" "$OUT.co"
