#!/usr/bin/env bash
# Decode AttnRes single-workgroup arm vs the banded multi-workgroup arm (PLOW_ATTNRES_DECODE_MWG).
#   GPU_LEASE_TIMEOUT=7200 perf-data/tools/gpulease -n 1 attnres-mwg \
#     nix develop -c runtime/bench/amd/attnres_decode_mwg/run.sh [out-dir] [iters]
set -euo pipefail
out=${1:-/tmp/plow-attnres-decode-mwg}
iters=${2:-2000}
mkdir -p "$out"
hipcc --offload-arch=gfx950 -O3 -w -DPLOW_K3=1 -DPLOW_ATTNRES_DECODE_MWG=1 \
  -I runtime/amd -I runtime/common \
  runtime/bench/amd/attnres_decode_mwg/bench.hip -o "$out/bench"
hipcc --genco --offload-arch=gfx950 -O3 -w -DPLOW_K3=1 -DPLOW_ATTNRES_DECODE_MWG=1 \
  -I runtime/amd -I runtime/common \
  runtime/bench/amd/attnres_decode_mwg/bench.hip -o "$out/bundle.co"
"${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler" -unbundle -type=o \
  -input="$out/bundle.co" -targets=hipv4-amdgcn-amd-amdhsa--gfx950 -output="$out/kernel.elf"
"${ROCM_PATH}/lib/llvm/bin/llvm-readobj" --notes "$out/kernel.elf" > "$out/resources.txt"
awk '/\.name:/ {name=$2} /\.(vgpr_count|sgpr_count|vgpr_spill_count|sgpr_spill_count|private_segment_fixed_size|group_segment_fixed_size):/ {print name, $1, $2}' \
  "$out/resources.txt" | grep -E "candidate|control" | sort -u
"$out/bench" "$iters" | tee "$out/result.txt"
