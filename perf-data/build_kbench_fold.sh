#!/bin/sh
# Build the MLA_MERGE_FOLD microbench. ROCm tooling must run OUTSIDE `nix develop` (contract §0a).
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"
PATH=/opt/rocm/bin:/usr/bin:/bin
export PATH
unset LD_LIBRARY_PATH
hipcc --offload-arch=gfx950 -O3 -w -DPLOW_BUCKET_DECODE=1 -std=c++17 --genco \
    "$R/perf-data/glm52_kbench_fold.hip" -o /tmp/glm_kfold.co \
    -I"$R/runtime/amd" -I"$R/runtime/common"
g++ -O2 -std=c++17 "$R/perf-data/glm52_kbench_fold.cpp" -o /tmp/kb_fold \
    -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ -L/opt/rocm/lib -lamdhip64
echo "built /tmp/glm_kfold.co /tmp/kb_fold"
