#!/usr/bin/env bash
# Build the external-Tensile-kernel bench: unbundle a gfx950 Tensile code object out of the
# INSTALLED hipBLASLt, build plow's own test kernels, and build the host driver that loads
# BOTH into one HSA context and races them.  No HIP runtime, no libhipblaslt.
#
#   nix develop --command bash -c './scripts/build_tensile_ext.sh /tmp/tsext'
#   perf-data/tools/gpulease -n 1 tsext sg render -c 'cd /tmp/tsext && \
#       /usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME LD_LIBRARY_PATH=/opt/rocm/lib \
#       ./gemm_tensile_ext tensile.co <sym> test_kernels.elf gemm_c5 4096 8192 5376'
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/tsext}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
ROCM="${ROCM_PATH:-/opt/rocm}"
BUN="${PLOW_BUNDLER:-$(ls -1 "$ROCM"/lib/llvm/bin/clang-offload-bundler \
        "$ROCM"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
HIPCC="${PLOW_HIPCC:-$ROCM/bin/hipcc}"
# NOTE (2026-07-27): do NOT run this under `nix develop`. The nix shell puts its own
# libstdc++/glibc on LD_LIBRARY_PATH, and every SYSTEM ROCm binary (hipcc,
# clang-offload-bundler, llvm-readobj) then dies with `GLIBC_2.38 not found`. Run it from
# a plain shell; the only toolchain this script needs is the installed ROCm.

# The bf16 TN "Custom" library: 2 hand-written assembly kernels, MT256x256x64_MI16x16x1,
# the winner GEMM_MFMA_SHAPE_VERDICT.md measured at 1619 TF/s (98% of sustained peak) on
# q_proj.  22 KB on disk, 256 KB unbundled -- small enough to ship, unlike the general
# TensileLibrary objects (17 MB compressed / 587 MB unbundled, 4158 kernels).
SRC_CO="${PLOW_TENSILE_CO:-$ROCM/lib/hipblaslt/library/TensileLibrary_BB_BB_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_${ARCH}.co}"

mkdir -p "$OUT"; cd "$OUT"
rm -f tk.co test_kernels.elf tensile.co gemm_tensile_ext

[ -f "$SRC_CO" ] || { echo "no Tensile object at $SRC_CO"; exit 1; }
# The ROCm bundler is a SYSTEM binary: nix's LD_LIBRARY_PATH puts a newer libstdc++ in
# front of it and it dies on a missing GLIBC symbol. Run it (and readobj) with a clean env.
CLEAN=(/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME")
"${CLEAN[@]}" "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input="$SRC_CO" --output="$OUT/tensile.co"
echo "--- tensile kernels ---"
"${CLEAN[@]}" "$ROCM"/llvm/bin/llvm-readobj --elf-output-style=GNU --symbols "$OUT/tensile.co" \
    2>/dev/null | awk '$4=="FUNC"{print "   "$8}' | sort -u

"$HIPCC" --offload-arch="$ARCH" -O3 -w --genco "$R/amd/test_kernels.hip" -o tk.co $INC
"${CLEAN[@]}" "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
       --input="$OUT/tk.co" --output="$OUT/test_kernels.elf"

# Host driver with the SYSTEM gcc in a clean env (nix gcc bakes a RUNPATH that aborts at load).
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 \
    -o gemm_tensile_ext "$R/ubench/gemm_tensile_ext.c" "$R/amd/hsa_backend.c" \
    -I"$ROCM/include" -L"$ROCM/lib" -lhsa-runtime64 -lm
readelf -d gemm_tensile_ext | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }

ls -l --time-style=+%H:%M:%S tensile.co test_kernels.elf gemm_tensile_ext \
    | awk '{print "   ", $NF, $5"B", $6}'
echo "OK -> $OUT"
