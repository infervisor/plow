#!/usr/bin/env bash
# Build and run the Phase 2 dense-GQA token-batch gates (runtime/tests/token_batch_dense_gfx942_test.hip).
#
# Must run inside `nix develop` — the shipped objects are built with the flake's hipcc, and a
# system /opt/rocm compiler produces objects that are not comparable. Takes a GPU lease: these
# are exact bit comparisons, but a contended card still perturbs nothing here, and the lease is
# how the box stays honest about who is using what.
#
#   nix develop /app/plow --command scripts/token_batch_dense_test.sh
set -euo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${PLOW_TB_ARCH:-gfx942}"
OUT="${PLOW_TB_OUT:-/tmp/tb_dense}"
SRC="$W/runtime/tests/token_batch_dense_gfx942_test.hip"

"$PLOW_HIPCC" --offload-arch="$ARCH" -O3 -DTB_DEVICE_TU -I"$W/runtime/amd" -I"$W/runtime/common" \
  -c "$SRC" -o "$OUT.dev.o"
c++ -O2 -std=c++17 -x c++ -D__HIP_PLATFORM_AMD__ -I"$ROCM_PATH/include" -I"$W/runtime/common" \
  -c "$SRC" -o "$OUT.host.o"
c++ "$OUT.host.o" "$OUT.dev.o" -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 \
  -o "$OUT"

GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-21600}" \
  "$W/perf-data/tools/gpulease" -n 1 tb-dense-gates "$OUT"
