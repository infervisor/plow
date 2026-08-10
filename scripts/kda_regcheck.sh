#!/usr/bin/env bash
# kda_regcheck.sh — the register-cliff report for the interpreter objects that carry the KDA arms.
#
# Same shape as build_gfx950.sh's check(): the 8-wave interpreters must stay <= 256 total
# (2 waves/SIMD) or occupancy halves. Run OUTSIDE `nix develop` — the nix shell's libstdc++/glibc
# shadow the system ones and every system ROCm binary dies with `GLIBC_2.38 not found`
# (the design notes §0a).
#
#   ./scripts/kda_regcheck.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
INC="-I$R/amd -I$R/common"

check() { # <name> <defs...>
    local name=$1; shift
    local U V A O S SS L
    U=$(hipcc --offload-arch="$ARCH" -O3 -w "$@" --genco \
          -Rpass-analysis=kernel-resource-usage "$R/amd/interp.hip" -o /dev/null $INC 2>&1)
    if echo "$U" | grep -qE "error:"; then
        echo "   $name COMPILE FAILED"; echo "$U" | grep -E "error:" | head -8; return 1
    fi
    V=$(echo "$U" | grep -oP 'VGPRs: \K\d+' | head -1)
    A=$(echo "$U" | grep -oP 'AGPRs: \K\d+' | head -1)
    O=$(echo "$U" | grep -oP 'Occupancy \[waves/SIMD\]: \K\d+' | head -1)
    S=$(echo "$U" | grep -oP 'VGPRs Spill: \K\d+' | head -1)
    SS=$(echo "$U" | grep -oP 'SGPRs Spill: \K\d+' | head -1)
    L=$(echo "$U" | grep -oP 'LDS Size \[bytes/block\]: \K\d+' | head -1)
    printf "   %-16s VGPR=%-4s AGPR=%-3s total=%-4s occ=%-2s vspill=%-3s sspill=%-3s lds=%s\n" \
           "$name" "$V" "$A" "$((V + A))" "$O" "$S" "$SS" "$L"
}

echo "interpreter objects, $ARCH:"
check decode  -DPLOW_BUCKET_DECODE=1
check prefill -DPLOW_BUCKET_DECODE=0
check prefill_mla_moe -DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1
