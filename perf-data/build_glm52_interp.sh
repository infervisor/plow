#!/bin/sh
# Build ONLY the GLM decode/prefill interpreter objects (no pkt re-emit) + report the register
# cliff. ROCm tooling must run OUTSIDE `nix develop` (contract §0a).
#   build_glm52_interp.sh <out-elf-prefix>
set -e
R="$(cd "$(dirname "$0")/.." && pwd)/runtime"
OUT="${1:-/tmp/mf}"
PATH=/opt/rocm/bin:/usr/bin:/bin
export PATH
unset LD_LIBRARY_PATH
BUN=$(ls -1 /opt/rocm/lib/llvm/bin/clang-offload-bundler /opt/rocm/llvm/bin/clang-offload-bundler \
      /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)
INC="-I$R/amd -I$R/common"

echo "== decode bucket =="
hipcc --offload-arch=gfx950 -O3 -w -DPLOW_BUCKET_DECODE=1 --genco "$R/amd/interp.hip" \
      -o "$OUT"_dec.co $INC -Rpass-analysis=kernel-resource-usage 2>&1 |
  grep -E "Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|Spill|LDS" | sed 's/^.*remark: //'
"$BUN" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
       --input="$OUT"_dec.co --output="$OUT"_dec.elf
ls -l "$OUT"_dec.elf
