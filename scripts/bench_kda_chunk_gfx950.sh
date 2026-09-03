#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
HIPCC=${PLOW_HIPCC:-hipcc}
ROCM=${ROCM_PATH:-/opt/rocm}
OUT=${TMPDIR:-/tmp}/plow-kda-chunk-bench

"$HIPCC" --offload-arch=gfx950 -O3 -w -DKDA3_DEVICE --genco \
  "$ROOT/runtime/tests/kda_step_cdna3_test.hip" -o "$OUT.co" \
  -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common"
"$HIPCC" -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I"$ROCM/include" \
  "$ROOT/runtime/tests/kda_step_cdna3_test.hip" -o "$OUT" \
  -L"$ROCM/lib" -lamdhip64

exec "$ROOT/perf-data/tools/gpulease" -n 1 kda-chunk-gfx950 \
  env KDA3_BENCH=1 "$OUT" "$OUT.co"
