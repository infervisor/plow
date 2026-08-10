#!/usr/bin/env bash
# k3_rung3_regcheck.sh — the register table for rung 3's kernel changes, A/B against pristine HEAD.
#
# Two changes touch device code in this rung:
#   * `d_mla_out_gate` (op 106, op_k3.h) — a new streaming elementwise arm.
#   * `exec_mla_merge_fold`'s VT dispatch (interp.hip) — a THIRD template instantiation,
#     `d_mla_merge_fold<512, 128>`, for Kimi-K3's v_head_dim = 128.
#
# The second is the one to watch: register allocation in the megakernel is the worst case over every
# INLINED arm, and prefill already sits AT the 256-VGPR / occ-2 cliff with 2 VGPR spills
# (the design notes §6g-WAVE4). A new instantiation of an existing template is exactly the
# kind of change that is "free" until it is not.
#
# Run OUTSIDE `nix develop` — the nix shell's libstdc++/glibc shadow the system ones and every
# system ROCm binary dies with GLIBC_2.38 (the design notes §0a). Compile-only; no GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
R="$PWD/runtime"
INC="-I$R/amd -I$R/common"
ARCH="${PLOW_HIP_ARCH:-gfx950}"

row() { # <label> <defs>
  local U V A O S L
  U=$(hipcc --offload-arch="$ARCH" -O3 -w $2 --genco \
        -Rpass-analysis=kernel-resource-usage "$R/amd/interp.hip" -o /dev/null $INC 2>&1)
  V=$(echo "$U" | grep -oP 'VGPRs: \K\d+' | head -1)
  A=$(echo "$U" | grep -oP 'AGPRs: \K\d+' | head -1)
  O=$(echo "$U" | grep -oP 'Occupancy \[waves/SIMD\]: \K\d+' | head -1)
  S=$(echo "$U" | grep -oP 'VGPRs Spill: \K\d+' | head -1)
  local SS
  SS=$(echo "$U" | grep -oP 'SGPRs Spill: \K\d+' | head -1)
  L=$(echo "$U" | grep -oP 'LDS Size \[bytes/block\]: \K\d+' | head -1)
  printf "  %-22s VGPR=%-4s AGPR=%-3s total=%-4s occ=%-2s vspill=%-3s sspill=%-3s LDS=%s\n" \
         "$1" "$V" "$A" "$((V + A))" "$O" "$S" "$SS" "$L"
}

echo "gfx950 register table (${ARCH}) — rung 3"
row "decode"           "-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1"
row "decode+MXFP4"     "-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_MXFP4=1"
row "prefill"          "-DPLOW_BUCKET_DECODE=0"
row "prefill_mla_moe"  "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1"
row "flash"            "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1"
