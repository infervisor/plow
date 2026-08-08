#!/usr/bin/env bash
# build_k3_real.sh — the Kimi-K3 COMPLETE-BLOCK numeric gate.            [K3-BLOCK-GATE]
#
# Modelled on scripts/build_kda_real.sh, which is modelled on scripts/build_glm52_real.sh,
# including the environment traps of the design notes:
#   - the .co needs system ROCm, which `nix develop` BREAKS (GLIBC_2.38);
#   - the host binary needs SYSTEM gcc in a scrubbed env, because nix's gcc bakes a RUNPATH to nix
#     glibc while the ELF interpreter is the system one, and the result aborts as
#     "stack smashing detected".
#
#   ./scripts/build_k3_real.sh [outdir]                      # build, OUTSIDE nix
#   PYTHONNOUSERSITE=1 nix develop .#quantize --command \
#       python3 runtime/tests/k3_real_oracle.py <outdir>/k3_fixture.bin    # fixture, ONCE
#   sg render -c 'cd <outdir> && env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/opt/rocm/lib \
#       ./k3_block_test interp_decode.elf k3_fixture.bin'                  # run, ONE GPU
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/home/lava/models/k3block}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"
cd "$OUT"
# A failed build must leave NOTHING behind, or the next run silently tests a stale object.
rm -f i_decode.co interp_decode.elf k3_block_test

# PLOW_GEMV_MM MUST BE >= T. `gemv_rows` accumulates `float acc[MM]` with MM = PLOW_GEMV_MM, a
# COMPILE-TIME constant defaulting to 1, and writes C[m*N+n] only for m < MM. At T>1 with the
# default, every GEMV writes ROW 0 and leaves rows 1..T-1 UNTOUCHED — not a crash, three quarters
# of the tokens carrying stale buffer contents, with the signature sqrt((T-1)/T) = 0.866 on every
# post-GEMV stage. Recorded in the design notes and build_gfx950.sh:51-64.
GVMM="${PLOW_GEMV_MM:-${K3_T:-4}}"
echo "[1/2] device code object ($ARCH, decode bucket, PLOW_GEMV_MM=$GVMM, PLOW_K3=1)"
hipcc --offload-arch="$ARCH" -O3 -w -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM="$GVMM" -DPLOW_K3=1 \
      --genco "$R/amd/interp.hip" -o i_decode.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input=i_decode.co --output=interp_decode.elf

echo "[2/2] host harness (system gcc, scrubbed env)"
/usr/bin/env -i PATH=/usr/bin:/bin HOME="${HOME:-/tmp}" /usr/bin/gcc -O2 -std=gnu11 -o k3_block_test \
    "$R/tests/k3_block_gfx950_test.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d k3_block_test | grep -qi runpath && { echo "FAIL: RUNPATH leaked into host binary"; exit 1; }

ls -l --time-style=+%H:%M:%S interp_decode.elf k3_block_test
echo "built in $OUT"
