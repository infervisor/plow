#!/usr/bin/env bash
# Sequence-split KDA carry vs the shipping key-fed carry: f64-reference accuracy and timing.
# The kernels are one --genco object built with the shipping lean-object defines.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:-/tmp/plow-kda-carry-split}
HIPCC=${HIPCC:-$(command -v hipcc)}
mkdir -p "$OUT"

"$HIPCC" --offload-arch=gfx950 -O3 -w --genco -Rpass-analysis=kernel-resource-usage \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_KDA_CARRY_PIPE=1 -DPLOW_KDA_CARRY_KTR16=1 \
    -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" \
    "$ROOT/runtime/bench/amd/kda_carry_split/kernel.hip" -o "$OUT/kernel.co" 2>"$OUT/resources"
"$HIPCC" --offload-arch=gfx950 -O3 -w --genco -Rpass-analysis=kernel-resource-usage \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_KDA_CARRY_PIPE=1 -DPLOW_KDA_CARRY_KTR16=1 \
    -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" \
    "$ROOT/runtime/amd/kda_chunk_carry_split.hip" -o "$OUT/split.co" 2>>"$OUT/resources"
"$HIPCC" -O2 -w "$ROOT/runtime/bench/amd/kda_carry_split/bench.cpp" -o "$OUT/bench"
grep -E "Function Name|VGPRs:|AGPRs:|ScratchSize|Occupancy" "$OUT/resources" | sed 's/^.*remark: *//; s/ \[-R.*$//' | paste - - - - -

[ "${COMPILE_ONLY:-0}" = 1 ] && exit 0
"$OUT/bench" "${T:-8192}" "${H:-8}" "${SAMPLES:-11}" "${REF:-1}" "$OUT/kernel.co" "$OUT/split.co" | tee "$OUT/results.txt"
