#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
HIPCC=${PLOW_HIPCC:-hipcc}
ROCM=${ROCM_PATH:-/opt/rocm}
OUT=${KDA_BENCH_OUT:-${TMPDIR:-/tmp}/plow-kda-chunk-bench}
WAVES=${KDA_CHUNK_WAVES:-8}
case "$WAVES" in
  4|8) ;;
  *) echo "KDA_CHUNK_WAVES must be 4 or 8" >&2; exit 2 ;;
esac
THREADS=$((WAVES * 64))

"$HIPCC" --offload-arch=gfx950 -O3 -w -DKDA3_DEVICE -DPLOW_WG_WAVES="$WAVES" --genco \
  "$ROOT/runtime/tests/kda_step_cdna3_test.hip" -o "$OUT.co" \
  -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common"

HIPROOT=$(cd "$(dirname "$(readlink -f "$(command -v "$HIPCC")")")/.." && pwd)
BUNDLER=${PLOW_BUNDLER:-$HIPROOT/lib/llvm/bin/clang-offload-bundler}
READELF=$HIPROOT/lib/llvm/bin/llvm-readelf
[ -x "$BUNDLER" ] && [ -x "$READELF" ] || {
  echo "missing ROCm bundler/readelf beside $HIPCC" >&2
  exit 2
}
"$BUNDLER" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
  --input="$OUT.co" --output="$OUT.elf"
read -r VGPR SGPR VSPILL SSPILL PRIVATE WAVE LDS <<<"$("$READELF" -n "$OUT.elf" | awk '
  function emit() { if (name == "k_intra_cached") print v, s, vs, ss, p, w, l }
  $1 == "-" && $2 == ".agpr_count:" {
    if (started) emit()
    started = 1; name = ""; v = s = vs = ss = p = w = l = 0
  }
  started && $1 == ".name:" { name = $2 }
  started && $1 == ".vgpr_count:" { v = $2 }
  started && $1 == ".sgpr_count:" { s = $2 }
  started && $1 == ".vgpr_spill_count:" { vs = $2 }
  started && $1 == ".sgpr_spill_count:" { ss = $2 }
  started && $1 == ".private_segment_fixed_size:" { p = $2 }
  started && $1 == ".wavefront_size:" { w = $2 }
  started && $1 == ".group_segment_fixed_size:" { l = $2 }
  END { emit() }
')"
[ "$WAVE" = 64 ] && [ "$VSPILL" = 0 ] && [ "$SSPILL" = 0 ] && [ "$PRIVATE" = 0 ] || {
  echo "k_intra_cached resource gate failed: wave=$WAVE vgpr_spill=$VSPILL sgpr_spill=$SSPILL private=$PRIVATE" >&2
  exit 2
}
[ "$VGPR" -le 96 ] && [ "$LDS" -le 163840 ] || {
  echo "k_intra_cached resource ceiling failed: vgpr=$VGPR lds=$LDS" >&2
  exit 2
}
echo "k_intra_cached resources: wave=$WAVE vgpr=$VGPR sgpr=$SGPR lds=$LDS private=$PRIVATE spills=$VSPILL/$SSPILL"

"$HIPCC" -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -DKDA_BENCH_THREADS="$THREADS" -I"$ROCM/include" \
  "$ROOT/runtime/tests/kda_step_cdna3_test.hip" -o "$OUT" \
  -L"$ROCM/lib" -lamdhip64

exec "$ROOT/perf-data/tools/gpulease" -n 1 kda-chunk-gfx950 \
  env KDA3_BENCH=1 "$OUT" "$OUT.co"
