#!/usr/bin/env bash
# Build the shim toolchain vLLM 0.28 needs to run above TP1 on this host.
#
# THE PROBLEM, once, so nobody rediscovers it three times as this campaign did.
# The vLLM 0.28 ROCm wheel needs glibc >= 2.39 and the host has 2.35, so its interpreter runs
# with the NIX glibc first on LD_LIBRARY_PATH. Its multiproc workers inherit that, and every
# SYSTEM binary they fork then dies:
#   * rocminfo   -> "Get GPU arch from rocminfo failed ... exit status 127"  (AITER arch probe)
#   * gcc        -> "CalledProcessError ... hip_utils.c"                     (Triton host ext)
#   * lld        -> "MLIRError ... lld invocation failed"                    (AITER flydsl JIT,
#                                                                             at the FIRST real
#                                                                             forward, after the
#                                                                             server is healthy)
# At TP1 none of these are reached, which is why the Gemma-4-31B runs never saw them.
#
# THREE THINGS ARE LOAD-BEARING ABOUT THE SHIM AND ALL THREE WERE LEARNED THE HARD WAY:
#   1. It must be on PATH. `aiter/jit/utils/cpp_extension.py` tries `shutil.which(tool)` FIRST
#      and only then <rocm_home>/bin/<tool>, so setting ROCM_PATH alone does not redirect it.
#   2. Its shebang must be a NIX binary. A `#!/bin/sh` shim cannot even load under the poisoned
#      LD_LIBRARY_PATH.
#   3. It must clear the variable with a SHELL BUILTIN. `env -u LD_LIBRARY_PATH ...` fails too —
#      coreutils `env` is itself a system binary and dies with
#      "undefined symbol: __tunable_is_initialized, version GLIBC_PRIVATE".
#
# THE SHIM IS A `PATH` OVERLAY, NOT A ROCM ROOT. Point ROCM_PATH at the COMPLETE
# tree (/opt/rocm-7.2.4). An earlier attempt pointed ROCM_PATH at this directory,
# whose lib/include are symlinks but which has no `amdgcn/bitcode` and no
# `llvm/`; AITER's flydsl MLIR pipeline needs both to link a GPU module and died
# with a bare "lld invocation failed" that reads like a missing binary.
#
#   scripts/glm53_vllm_shim.sh [outdir]        # default build-glm53/rocm-shim
#   PATH=<outdir>/bin:$PATH CC=<outdir>/bin/vllm-cc ROCM_PATH=/opt/rocm-7.2.4 vllm serve ...
#
# AND NOTE: `VLLM_ROCM_USE_AITER=0` is NOT an escape hatch for GLM-5.3. vLLM
# refuses outright — "Sparse attention indexer ROCm path is only supported on
# AITER" — because the DSA indexer has no non-AITER ROCm kernel. AITER is
# mandatory for this model, so these shims are mandatory with it.
set -euo pipefail
OUT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build-glm53/rocm-shim}"
ROCM="${SHIM_ROCM:-/opt/rocm-7.2.4}"
[ -x "$ROCM/bin/rocminfo" ] || { echo "FAIL: no rocminfo under $ROCM" >&2; exit 2; }
NIXBASH="${SHIM_BASH:-$(ls -d /nix/store/*bash*/bin/bash 2>/dev/null | head -1)}"
[ -x "$NIXBASH" ] || { echo "FAIL: no nix bash found; set SHIM_BASH" >&2; exit 2; }
mkdir -p "$OUT/bin" "$OUT/.info"
# Version file: workers read <root>/.info/version at init_device and /opt/rocm has none.
tr -d '\n' < "$ROCM/.info/version" > "$OUT/.info/version"; echo >> "$OUT/.info/version"
ln -sfn "$ROCM/lib" "$OUT/lib"; ln -sfn "$ROCM/include" "$OUT/include"
emit () {  # emit <name> <target>
  printf '#!%s\nunset LD_LIBRARY_PATH\nexec %s "$@"\n' "$NIXBASH" "$2" > "$OUT/bin/$1"
  chmod +x "$OUT/bin/$1"
}
for t in rocminfo lld ld.lld clang clang++ hipcc llvm-objcopy clang-offload-bundler amdgpu-arch; do
  for c in "$ROCM/lib/llvm/bin/$t" "$ROCM/llvm/bin/$t" "$ROCM/bin/$t"; do
    [ -x "$c" ] && { emit "$t" "$c"; break; }
  done
done
emit vllm-cc /usr/bin/gcc          # Triton's host C-extension compiler
echo "shim: $OUT  ($(ls "$OUT/bin" | wc -l) tools, rocm $(cat "$OUT/.info/version"))"
# Prove it survives the poisoned environment rather than assuming it. Try every nix glibc in
# the store: which one vLLM's interpreter actually carries is not knowable from here, and the
# shim has to work under any of them.
ok=0
for g in $(ls -d /nix/store/*glibc*/lib 2>/dev/null); do
  # NOT `| grep -q`: with `set -o pipefail` grep exits on the first match, rocminfo takes
  # SIGPIPE, and the pipeline reports 141 — a passing shim would read as a failure.
  out="$(LD_LIBRARY_PATH="$g:/lib/x86_64-linux-gnu" "$OUT/bin/rocminfo" 2>/dev/null || true)"
  case "$out" in *gfx*) ok=$((ok + 1)) ;; esac
done
if [ "$ok" -gt 0 ]; then
  echo "verified: rocminfo runs under $ok nix-glibc LD_LIBRARY_PATH variant(s)"
else
  echo "WARN: shim did not verify — check $NIXBASH and $ROCM" >&2
fi
