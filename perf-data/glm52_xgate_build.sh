#!/usr/bin/env bash
# Build the cross-GPU-gate CEILING objects for GLM-5.2 TP4, DECODE and PREFILL.
#
# Runs OUTSIDE `nix develop` (knob-contract §0a: the nix glibc shadows the system one and
# every ROCm binary then dies with GLIBC_2.38 not found). No GPU needed.
#
# Source tree is a private COPY of `runtime/` (./rt) so that editing op_collective.h can
# never perturb the worktree build the armed final benchmark is compiling from.
#
# Arms (all keep the SIGNAL, the fabric traffic, the packet count, the gate graph, the
# workgroup count and the acquire fence identical; only the WAIT is removed):
#   base       shipping
#   nowait     -DPLOW_XR_NOWAIT=1     decode one-shot rendezvous deleted
#                                     prefill two-shot BOTH rendezvous deleted
#   nowaitrs   -DPLOW_XR_NOWAIT_RS=1  prefill two-shot: ONLY gate_rs deleted (the half a
#                                     producer-side tile watermark could ever replace)
set -euo pipefail
D=/home/lava/models/glm52_xrpf
R="$D/rt"
INC="-I$R/amd -I$R/common"
ARCH=gfx950
BUN="$(ls -1 /opt/rocm/lib/llvm/bin/clang-offload-bundler /opt/rocm/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)"
cd "$D"; mkdir -p obj

genco () { # <defs> <out>
  hipcc --offload-arch=$ARCH -O3 -w $1 --genco "$R/amd/interp.hip" -o "obj/$2.co" $INC
  "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
      --input="obj/$2.co" --output="obj/$2.elf"
}

DEC="-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1"
PF="-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1"
GQ="-DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=1"

# MARGINAL-ACQUIRE arms (decode only): k system acquires where the protocol takes 1.
# Numerically CORRECT, so these must reproduce the control token exactly.
for k in 2 4; do
  genco "$DEC -DPLOW_XR_ACQ_N=$k"     "d_acq$k"
  genco "$PF  -DPLOW_XR_ACQ_N=$k"     "interp_prefill_mla_moe_acq$k"
  genco "$PF $GQ -DPLOW_XR_ACQ_N=$k"  "interp_prefill_mla_moe_gq_acq$k"
done

for arm in "base:" "nowait:-DPLOW_XR_NOWAIT=1" "nowaitrs:-DPLOW_XR_NOWAIT_RS=1" "nosig:-DPLOW_XR_NOSIG=1" "shuffle:-DPLOW_XR_SHUFFLE=1"; do
  tag=${arm%%:*}; def=${arm#*:}
  # decode: the C harness loads this by PLOW_INTERP, no _gq twin needed
  [ "$tag" = nowaitrs ] || genco "$DEC $def"        "d_$tag"
  # prefill MLA+MoE, both schedulers (plowrt picks the _gq one unless PLOW_NO_GQ=1)
  genco "$PF $def"        "interp_prefill_mla_moe_$tag"
  genco "$PF $GQ $def"    "interp_prefill_mla_moe_gq_$tag"
done

# host harness for the decode arm: SYSTEM gcc, clean env (a nix-gcc RUNPATH aborts at HSA load)
/usr/bin/gcc -O2 -std=gnu11 -o "$D/glm52_decode" "$R/tests/glm52_decode.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d "$D/glm52_decode" | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }

ls -l --time-style=+%H:%M:%S obj/*.elf | awk '{print "   ",$NF,$5"B",$6}'
