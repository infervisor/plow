#!/bin/bash
# Register-cliff check for every bucket that inlines d_mla_merge_fold, the same gate
# scripts/build_gfx950.sh:229 applies. ROCm tooling must run OUTSIDE `nix develop` (§0a).
R="$(cd "$(dirname "$0")/.." && pwd)/runtime"
export PATH=/opt/rocm/bin:/usr/bin:/bin
unset LD_LIBRARY_PATH
INC="-I$R/amd -I$R/common"
rc=0
check() { # <name> <defs> <max-total> <min-occ>
  local U V A O S
  U=$(hipcc --offload-arch=gfx950 -O3 -w $2 --genco \
        -Rpass-analysis=kernel-resource-usage "$R/amd/interp.hip" -o /dev/null $INC 2>&1)
  V=$(echo "$U" | grep -oP 'VGPRs: \K\d+' | head -1)
  A=$(echo "$U" | grep -oP 'AGPRs: \K\d+' | head -1)
  O=$(echo "$U" | grep -oP 'Occupancy \[waves/SIMD\]: \K\d+' | head -1)
  S=$(echo "$U" | grep -oP 'VGPRs Spill: \K\d+' | head -1)
  L=$(echo "$U" | grep -oP 'LDS Size \[bytes/block\]: \K\d+' | head -1)
  printf "   %-14s VGPR=%-3s AGPR=%-3s total=%-3s occ=%s spill=%s lds=%s\n" "$1" "$V" "$A" "$((V + A))" "$O" "$S" "$L"
  if [ "$((V + A))" -gt "$3" ] || [ "$O" -lt "$4" ]; then
    echo "   ^^ OVER THE CLIFF ($1)"; rc=1
  fi
}
check decode      "-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1" 256 2
check decode_fp8  "-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_FP8=1" 256 2
check prefill     "-DPLOW_BUCKET_DECODE=0" 256 2
check prefill_fp8 "-DPLOW_BUCKET_DECODE=0 -DPLOW_FP8=1" 256 2
check prefill_mla "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1" 256 2
exit $rc
