#!/usr/bin/env bash
# Rebuild ONLY the interpreter objects whose collective bodies changed, with the
# register-cliff gate from scripts/build_gfx950.sh applied to each.
#
# Scoped deliberately: the fix in runtime/amd/op_collective.h touches
# d_xreduce_twoshot_mega, which lives behind `#if !defined(PLOW_BUCKET_FLASH)` and
# is inlined into the PREFILL and DECODE interpreters. The flash object excludes
# the collectives entirely, so it is not rebuilt and its 4-wave/512-reg budget is
# not disturbed.
#
# hipcc must run OUTSIDE `nix develop`: the nix libstdc++/glibc shadow the system
# ones and hipcc dies with "GLIBC_2.38 not found". Same contract as the C harness
# builds. Do NOT run this under gpulease (knob-contract §0) — leases are for
# running, and gpulease exports HIP_VISIBLE_DEVICES which makes hipcc report
# "no ROCm-capable device is detected".
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:?output dir}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
GQB="${PLOW_GQ_BATCH:-1}"
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"

# name : defs : max-total-regs : min-occupancy
SPECS=(
  "prefill:-DPLOW_BUCKET_DECODE=0:256:2"
  "prefill_gq:-DPLOW_BUCKET_DECODE=0 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB:256:2"
  "decode:-DPLOW_BUCKET_DECODE=1:256:2"
  "decode_gq:-DPLOW_BUCKET_DECODE=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB:256:2"
)

for spec in "${SPECS[@]}"; do
  N="${spec%%:*}"; rest="${spec#*:}"
  D="${rest%%:*}"; rest="${rest#*:}"
  MAXR="${rest%%:*}"; MINO="${rest#*:}"

  # THE CLIFF GATE. Over budget is HSA_STATUS_ERROR_INVALID_ISA at *runtime*;
  # this build error is the cheap failure. Never raise a cap to make it pass.
  U=$(hipcc --offload-arch="$ARCH" -O3 -w $D --genco \
        -Rpass-analysis=kernel-resource-usage "$R/amd/interp.hip" -o /dev/null $INC 2>&1)
  V=$(echo "$U" | grep -oP 'VGPRs: \K\d+' | head -1)
  A=$(echo "$U" | grep -oP 'AGPRs: \K\d+' | head -1)
  O=$(echo "$U" | grep -oP 'Occupancy \[waves/SIMD\]: \K\d+' | head -1)
  S=$(echo "$U" | grep -oP 'VGPRs Spill: \K\d+' | head -1)
  if [ -z "$V" ] || [ -z "$O" ]; then
    echo "BUILD FAILED: $N — could not read resource usage from hipcc" >&2
    echo "$U" | tail -20 >&2; exit 1
  fi
  printf "   %-11s VGPR=%-3s AGPR=%-3s total=%-3s occ=%s spill=%s\n" "$N" "$V" "$A" "$((V+A))" "$O" "$S"
  if [ "$((V+A))" -gt "$MAXR" ] || [ "$O" -lt "$MINO" ]; then
    echo "BUILD FAILED: $N over the register cliff (total $((V+A)) > $MAXR, or occ $O < $MINO)." >&2
    exit 1
  fi

  hipcc --offload-arch="$ARCH" -O3 -w $D --genco "$R/amd/interp.hip" -o "$OUT/i_$N.co" $INC
  "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
      --input="$OUT/i_$N.co" --output="$OUT/interp_$N.elf"
  rm -f "$OUT/i_$N.co"
done

ls -l --time-style=+%H:%M:%S "$OUT"/interp_*.elf | awk '{print "   ", $NF, $5"B", $6}'
