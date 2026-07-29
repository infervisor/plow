#!/usr/bin/env bash
# Build ONLY the two MLA-prefill interpreter objects into an existing object dir.
#
# Why this exists: `PrefillArm::detect` (`crates/plowrt/src/exec/amd.rs:281`) scans EVERY program
# for `MlaMergeFold`, and GLM's DECODE program uses it — so a decode-only GLM packet is classified
# `PrefillArm::Mla` and the loader REFUSES to start without `interp_prefill_mla{,_gq}.elf`, even
# though no prefill program exists to dispatch them. `build-amd/hsaco-abi144` does not carry them.
#
# ROCm tooling must run OUTSIDE `nix develop` (knob-contract §0a: the nix shell's glibc shadows the
# system one and every ROCm binary dies with `GLIBC_2.38 not found`).
#   scripts/build_glm52_mla_pf_obj.sh <objdir>
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
O="${1:?objdir}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
GQB="${PLOW_GQ_BATCH:-1}"
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
mkdir -p "$O"

for v in "" "_gq"; do
  defs="-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1"
  [ -n "$v" ] && defs="$defs -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB"
  rm -f "$O/i_prefill_mla$v.co" "$O/interp_prefill_mla$v.elf"
  echo "== interp_prefill_mla$v.elf"
  hipcc --offload-arch="$ARCH" -O3 -w $defs --genco "$R/amd/interp.hip" -o "$O/i_prefill_mla$v.co" $INC
  "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
         --input="$O/i_prefill_mla$v.co" --output="$O/interp_prefill_mla$v.elf"
done
ls -la "$O"/interp_prefill_mla*.elf
