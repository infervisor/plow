#!/usr/bin/env bash
# build_kda_real.sh — the Kimi-K3 KDA single-layer numeric gate.       [K3-KDA-GATE]
#
# Modelled on scripts/build_glm52_real.sh, including its two environment traps
# (the design notes §0a):
#   - the .co needs system ROCm, which `nix develop` BREAKS (GLIBC_2.38);
#   - the host binary needs SYSTEM gcc in a scrubbed env, because nix's gcc bakes a RUNPATH to nix
#     glibc while the ELF interpreter is the system one, and the result aborts as
#     "stack smashing detected".
#
#   ./scripts/build_kda_real.sh [outdir]                     # build, OUTSIDE nix
#   PYTHONNOUSERSITE=1 nix develop .#quantize --command \
#       python3 runtime/tests/kda_real_oracle.py <outdir>/kda_fixture.bin   # fixture, ONCE
#   sg render -c 'cd <outdir> && env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/opt/rocm/lib \
#       ./kda_block_test interp_decode.elf kda_fixture.bin'                 # run, one GPU
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/home/lava/models/k3kda}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"
cd "$OUT"
# A failed build must leave NOTHING behind, or the next run silently tests a stale object.
rm -f i_decode.co interp_decode.elf kda_block_test

# PLOW_GEMV_MM MUST BE >= T, and this is the trap that cost the first run of this gate.
#
# op_gemm.h's `gemv_rows` accumulates `float acc[MM]` with MM = PLOW_GEMV_MM, a COMPILE-TIME
# constant defaulting to 1, and writes `C[m*N+n]` only for `m < MM`. At T>1 with the default, every
# GEMV in the block writes ROW 0 and leaves rows 1..T-1 UNTOUCHED — which is not a crash, it is
# three quarters of the tokens carrying whatever was in the buffer. build_gfx950.sh:51-64 records
# the same bug for batched decode ("every AMD decode object compiled at op_gemm.h's default of 1
# and wrote row 0 only"). It is the contract's §4 shape exactly: the arm exists, is correct, is
# register-gated, and nothing routes to it.
#
# It shows up in the residual table as ~sqrt((T-1)/T) on every post-GEMV stage — 0.79-0.87 at T=4 —
# which looks like a broken kernel and is not.
GVMM="${PLOW_GEMV_MM:-${KDA_T:-4}}"
echo "[1/2] device code object ($ARCH, decode bucket, PLOW_GEMV_MM=$GVMM, PLOW_K3=1)"
hipcc --offload-arch="$ARCH" -O3 -w -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM="$GVMM" -DPLOW_K3=1 \
      --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=i_decode.co --output=interp_decode.elf

echo "[2/2] host harness (system gcc, scrubbed env)"
/usr/bin/env -i PATH=/usr/bin:/bin HOME="${HOME:-/tmp}" /usr/bin/gcc -O2 -std=gnu11 -o kda_block_test \
    "$R/tests/kda_block_gfx950_test.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d kda_block_test | grep -qi runpath && { echo "FAIL: RUNPATH leaked into host binary"; exit 1; }

ls -l --time-style=+%H:%M:%S interp_decode.elf kda_block_test
echo "built in $OUT"
