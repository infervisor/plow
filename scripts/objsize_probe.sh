#!/usr/bin/env bash
# Per-bucket code-object SIZE probe. `scripts/build_gfx950.sh` gates the REGISTER cliff and nothing
# gates object growth — but a register-neutral arm that grew the decode object 15.6% measured a
# +32% decode regression inside the persistent megakernel (knob-contract; `mla.rs` GF=8), so
# "it fits the register budget" is necessary and not sufficient. This builds the four buckets that
# matter for GLM-5.2 and prints their bytes, so a change can be priced per bucket instead of in
# aggregate.
#
# Run OUTSIDE `nix develop` (the nix libstdc++/glibc shadow the system ones and hipcc dies with
# GLIBC_2.38 not found) and NOT under `gpulease` (it exports HIP_VISIBLE_DEVICES and hipcc then
# reports no device). No GPU is used.
#   scripts/objsize_probe.sh <outdir>
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?outdir}"
mkdir -p "$OUT"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
INC="-I$R/runtime/amd -I$R/runtime/common"
HIPCC="${ROCM_PATH:-/opt/rocm}/bin/hipcc"

gen() { # <name> <defs>
    $HIPCC --offload-arch="$ARCH" -O3 -w $2 --genco "$R/runtime/amd/interp.hip" \
        -o "$OUT/i_$1.co" $INC
    printf "%-24s %10d\n" "i_$1.co" "$(stat -c%s "$OUT/i_$1.co")"
}

# The three buckets the runtime selects between at load time, plus the whole-layer MLA+MoE prefill
# object GLM-5.2's stacked blob actually loads.
gen decode          "-DPLOW_BUCKET_DECODE=1"
gen prefill         "-DPLOW_BUCKET_DECODE=0"
gen flash           "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1"
gen prefill_mla_moe "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1"
