#!/usr/bin/env bash
# Ad-hoc register-cliff probe: mirrors `check()` in scripts/build_gfx950.sh exactly
# (same hipcc invocation, same -Rpass-analysis parse) but only REPORTS, never deletes.
# Used to price PLOW_MLA_PREFILL / PLOW_MOE_PREFILL against the 256/occ-2 prefill cliff.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
INC="-I$R/amd -I$R/common"
ARCH="${PLOW_HIP_ARCH:-gfx950}"

run() {
  local name="$1"; shift
  local U V A O S
  U=$(hipcc --offload-arch="$ARCH" -O3 -w "$@" --genco \
        -Rpass-analysis=kernel-resource-usage "$R/amd/interp.hip" -o /dev/null $INC 2>&1)
  V=$(echo "$U" | grep -oP 'VGPRs: \K\d+' | head -1)
  A=$(echo "$U" | grep -oP 'AGPRs: \K\d+' | head -1)
  O=$(echo "$U" | grep -oP 'Occupancy \[waves/SIMD\]: \K\d+' | head -1)
  S=$(echo "$U" | grep -oP 'VGPRs Spill: \K\d+' | head -1)
  if [ -z "$V" ]; then
    printf "   %-26s COMPILE FAILED\n" "$name"
    echo "$U" | grep -iE 'error' | head -8
    return
  fi
  printf "   %-26s VGPR=%-3s AGPR=%-3s total=%-3s occ=%s spill=%s\n" \
         "$name" "$V" "$A" "$((V + A))" "$O" "$S"
}

run "prefill baseline"      -DPLOW_BUCKET_DECODE=0
run "prefill +MLA"          -DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1
run "prefill +MLA +MOE"     -DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1
run "prefill +MOE only"     -DPLOW_BUCKET_DECODE=0 -DPLOW_MOE_PREFILL=1
