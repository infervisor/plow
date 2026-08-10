#!/usr/bin/env bash
# Build the DSA sparse-PREFILL indexer bench (runtime/bench/interp/dsa_pf_indexer_bench.c):
# test_kernels.elf (ops 117/118 shipped + rebuilt arms) + the host harness.  [GLM52-DSA-PF-IDX]
#
#   bash ./scripts/build_dsa_pf_indexer_bench.sh /root/.claude/jobs/b09a4bcc/tmp/dsa_pf_idx
#   cd <OUT> && ROCR_VISIBLE_DEVICES=0 ./dsa_pf_bench test_kernels.elf
#
# Run OUTSIDE nix (hipcc needs system glibc), like scripts/build_gfx942.sh.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/dsa_pf_idx}}"
ARCH="${PLOW_HIP_ARCH:-gfx942}"
ROCM="${ROCM_PATH:-/opt/rocm}"
HIPCC="${PLOW_HIPCC:-$ROCM/bin/hipcc}"
# `ls` over a multi-path probe exits 2 when ANY path is absent, and `pipefail` then kills the
# script before it prints a thing. Swallow that; the executability check below is the real gate.
BUN="${PLOW_BUNDLER:-$(ls -1 "$ROCM"/lib/llvm/bin/clang-offload-bundler \
        "$ROCM"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1 || true)}"
[ -x "$BUN" ]   || { echo "FAIL: no clang-offload-bundler found (set PLOW_BUNDLER)"; exit 1; }
[ -x "$HIPCC" ] || { echo "FAIL: no hipcc at $HIPCC (set PLOW_HIPCC)"; exit 1; }
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"; cd "$OUT"
rm -f tk.co test_kernels.elf dsa_pf_bench

"$HIPCC" --offload-arch="$ARCH" -O3 -w --genco "$R/amd/test_kernels.hip" -o tk.co $INC
"$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input=tk.co \
       --output=test_kernels.elf

# The two score arms' register/LDS/spill profile is the whole point of arm B — print it.
echo "--- indexer kernel resource usage ---"
"$HIPCC" --offload-arch="$ARCH" -O3 -w -Rpass-analysis=kernel-resource-usage --genco \
    "$R/amd/test_kernels.hip" -o /dev/null $INC 2>&1 \
  | grep -E "Function Name: (index_score_pf_128|index_score_pf_row_128|index_select_pf_k|index_select_pf_fast_k)|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|LDS|Spill" \
  | awk '/Function Name/{p=($0 ~ /index_score_pf|index_select_pf/)} p' || true
echo "-------------------------------------"

gcc -O2 -std=gnu11 -o dsa_pf_bench "$R/bench/interp/dsa_pf_indexer_bench.c" "$R/amd/hsa_backend.c" \
    -I"$ROCM/include" -L"$ROCM/lib" -lhsa-runtime64 -lm
ls -l --time-style=+%H:%M:%S test_kernels.elf dsa_pf_bench | awk '{print "   ",$NF,$5"B",$6}'
echo "OK — run: cd $OUT && ROCR_VISIBLE_DEVICES=0 ./dsa_pf_bench test_kernels.elf"
