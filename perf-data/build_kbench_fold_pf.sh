#!/bin/sh
# Build the PREFILL-shape MLA_MERGE_FOLD microbench. ROCm tooling runs OUTSIDE `nix develop`.
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"
ROCM=${ROCM_PATH:-/opt/rocm}
PATH=$ROCM/bin:/usr/bin:/bin
export PATH
unset LD_LIBRARY_PATH
hipcc --offload-arch=gfx942 -O3 -w -std=c++17 --genco \
    "$R/runtime/bench/amd/glm52_kbench_fold_pf.hip" -o /tmp/glm_kfold_pf.co \
    -I"$R/runtime/amd" -I"$R/runtime/common"
g++ -O2 -std=c++17 "$R/runtime/bench/amd/glm52_kbench_fold_pf.cpp" -o /tmp/kb_fold_pf \
    -I$ROCM/include -D__HIP_PLATFORM_AMD__ -L$ROCM/lib -lamdhip64
echo "built /tmp/glm_kfold_pf.co /tmp/kb_fold_pf"
