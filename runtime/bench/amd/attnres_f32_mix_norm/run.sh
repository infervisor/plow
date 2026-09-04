#!/usr/bin/env bash
set -euo pipefail
out=${1:-/tmp/plow-attnres-f32-mix-norm}
mkdir -p "$out"
hipcc --offload-arch=gfx950 -O3 -w -DPLOW_K3=1 \
  -I runtime/amd -I runtime/common \
  runtime/bench/amd/attnres_f32_mix_norm/bench.hip -o "$out/bench"
hipcc --genco --offload-arch=gfx950 -O3 -w -DPLOW_K3=1 \
  -I runtime/amd -I runtime/common \
  runtime/bench/amd/attnres_f32_mix_norm/bench.hip -o "$out/bundle.co"
"${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler" -unbundle -type=o \
  -input="$out/bundle.co" -targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
  -output="$out/kernel.elf"
"${ROCM_PATH}/lib/llvm/bin/llvm-readobj" --notes "$out/kernel.elf" > "$out/resources.txt"
hipcc --offload-arch=gfx950 -O3 -w -DPLOW_K3=1 -DPLOW_ATTNRES_TOKENS=8192 \
  -I runtime/amd -I runtime/common \
  runtime/bench/amd/attnres_f32_mix_norm/bench.hip -o "$out/prefill-bench"
hipcc --genco --offload-arch=gfx950 -O3 -w -DPLOW_K3=1 -DPLOW_ATTNRES_TOKENS=8192 \
  -I runtime/amd -I runtime/common \
  runtime/bench/amd/attnres_f32_mix_norm/bench.hip -o "$out/prefill-bundle.co"
"${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler" -unbundle -type=o \
  -input="$out/prefill-bundle.co" -targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
  -output="$out/prefill-kernel.elf"
"${ROCM_PATH}/lib/llvm/bin/llvm-readobj" --notes "$out/prefill-kernel.elf" \
  > "$out/prefill-resources.txt"
set +e
"$out/bench" | tee "$out/result.txt"
decode_rc=${PIPESTATUS[0]}
"$out/prefill-bench" | tee "$out/prefill-result.txt"
prefill_rc=${PIPESTATUS[0]}
set -e
resource_rc=0
for metadata in "$out/resources.txt" "$out/prefill-resources.txt"; do
  awk '
    /\.name:.*candidate/ { candidate=1; next }
    /\.name:/ { candidate=0 }
    candidate && /\.(private_segment_fixed_size|sgpr_spill_count|vgpr_spill_count):/ && $2 != 0 {
      bad=1
    }
    END { exit bad ? 1 : 0 }
  ' "$metadata" || resource_rc=1
done
(( decode_rc == 0 && prefill_rc == 0 && resource_rc == 0 ))
