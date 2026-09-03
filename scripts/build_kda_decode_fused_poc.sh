#!/usr/bin/env bash
# Compile-only gate for the standalone gfx950 KDA decode fusion proof. This script never opens a
# GPU. Use run_kda_decode_fused_poc.sh for the separately leased runtime oracle.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-${PLOW_BUILD_DIR:-/tmp/plow-kda-decode-fused-poc}}
ROCM=${ROCM_PATH:-/opt/rocm}
HIPCC=${PLOW_HIPCC:-$ROCM/bin/hipcc}
SRC=$ROOT/runtime/tests/kda_decode_fused_poc_gfx950.hip
INC=(-I"$ROOT/runtime/amd" -I"$ROOT/runtime/common")
[ -x "$HIPCC" ] || { echo "FAIL: no hipcc at $HIPCC" >&2; exit 2; }
mkdir -p "$OUT"
rm -f "$OUT/kda_decode_fused_poc.co" "$OUT/kda_decode_fused_poc" "$OUT/resources.txt"

echo "[1/3] gfx950 device object"
"$HIPCC" --offload-arch=gfx950 -O3 -w -DKDA_FUSED_DEVICE --genco "$SRC" \
    -o "$OUT/kda_decode_fused_poc.co" "${INC[@]}"

echo "[2/3] resource census"
usage=$("$HIPCC" --offload-arch=gfx950 -O3 -w -DKDA_FUSED_DEVICE \
    -Rpass-analysis=kernel-resource-usage --genco "$SRC" -o /dev/null "${INC[@]}" 2>&1)
printf '%s\n' "$usage" | awk '
  /Function Name: kda_decode_(fused_poc|control_conv|control_step|control_norm)/ {show=1; n=0}
  show {print; n++} n==13 {show=0}
' | tee "$OUT/resources.txt"
grep -q 'Function Name: kda_decode_fused_poc' "$OUT/resources.txt" || {
    echo "FAIL: fused-kernel resource report missing" >&2; exit 2; }
spill=$(sed -n '/Function Name: kda_decode_fused_poc/,/Function Name:/{s/.*VGPRs Spill: \([0-9][0-9]*\).*/\1/p}' "$OUT/resources.txt" | head -1)
sgspill=$(sed -n '/Function Name: kda_decode_fused_poc/,/Function Name:/{s/.*SGPRs Spill: \([0-9][0-9]*\).*/\1/p}' "$OUT/resources.txt" | head -1)
scratch=$(sed -n '/Function Name: kda_decode_fused_poc/,/Function Name:/{s/.*ScratchSize \[bytes\/lane\]: \([0-9][0-9]*\).*/\1/p}' "$OUT/resources.txt" | head -1)
vgpr=$(sed -n '/Function Name: kda_decode_fused_poc/,/Function Name:/{s/.*remark:     VGPRs: \([0-9][0-9]*\).*/\1/p}' "$OUT/resources.txt" | head -1)
occ=$(sed -n '/Function Name: kda_decode_fused_poc/,/Function Name:/{s/.*Occupancy \[waves\/SIMD\]: \([0-9][0-9]*\).*/\1/p}' "$OUT/resources.txt" | head -1)
[ -n "$spill" ] && [ -n "$sgspill" ] && [ -n "$scratch" ] && [ -n "$vgpr" ] && [ -n "$occ" ] || { echo "FAIL: incomplete resource census" >&2; exit 2; }
[ "$spill" -eq 0 ] || { echo "FAIL: fused POC spills $spill VGPRs" >&2; exit 2; }
[ "$sgspill" -eq 0 ] || { echo "FAIL: fused POC spills $sgspill SGPRs" >&2; exit 2; }
[ "$scratch" -eq 0 ] || { echo "FAIL: fused POC uses $scratch bytes/lane scratch" >&2; exit 2; }
[ "$occ" -ge 1 ] || { echo "FAIL: fused POC has zero occupancy" >&2; exit 2; }

echo "[3/3] host oracle (compiled, not run)"
"$HIPCC" -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 "$SRC" -o "$OUT/kda_decode_fused_poc" \
    -I"$ROCM/include" -L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" -lamdhip64
test -x "$OUT/kda_decode_fused_poc"
echo "OK: compile/static gates passed; VGPR=$vgpr spill=$spill scratch=$scratch occupancy=$occ"
echo "Deferred runtime: $ROOT/scripts/run_kda_decode_fused_poc.sh $OUT"
