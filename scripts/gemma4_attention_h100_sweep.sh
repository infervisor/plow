#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
build=${PLOW_FA_SWEEP_BUILD:-/tmp/gemma4-attention-h100-build}
output=${PLOW_FA_SWEEP_OUTPUT:-/tmp/gemma4-attention-h100.jsonl}
nvcc=${PLOW_FA_NVCC:-/usr/local/cuda-13.0/bin/nvcc}
nvenv=(env -u NVCC_PREPEND_FLAGS -u LIBRARY_PATH -u NIX_LDFLAGS)

case $build in /tmp/*) ;; *) echo "build directory must be under /tmp" >&2; exit 2;; esac
case $output in /tmp/*) ;; *) echo "raw output must be under /tmp" >&2; exit 2;; esac

common=(-ccbin /usr/bin/g++-11 -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 -Xptxas=-v
  -I "$repo/runtime/common" -I "$repo/runtime/nvidia")

prefill() {
  local name=$1 hd=$2 bkv=$3
  shift 3
  "${nvenv[@]}" "$nvcc" "${common[@]}" -DPLOW_FA_SWEEP_HD="$hd" -DPLOW_FA_SWEEP_BKV="$bkv" \
    "$@" "$repo/runtime/nvidia/experiments/gemma4_fa_prefill_h100_sweep.cu" \
    -lcuda -o "$build/$name"
}

build_all() {
  mkdir -p "$build"
  prefill prefill_hd256_pair32 256 32 \
    -DPLOW_NV_FA_WGITEM=1 -DPLOW_NV_FA_GQA2_PAIR=1 \
    -DPLOW_NV_PACKED_REQUEST=1 -DPLOW_NV_PACKED_FA_WGMMA=1 \
    -DPLOW_NV_PACKED_FA_TMA=1
  prefill prefill_hd256_wgitem32 256 32 -DPLOW_NV_FA_WGITEM=1
  prefill prefill_hd256_onewg32 256 32 -DPLOW_NV_FA_WGITEM=1 \
    -DPLOW_NV_FA_WGITEM_ONE=1 -DPLOW_FA_SWEEP_THREADS=128
  prefill prefill_hd256_hdsplit32 256 32
  prefill prefill_hd256_hdsplit64 256 64
  prefill prefill_hd512_bkv16_s2 512 16 -DPLOW_NV_FA512_WG=1
  prefill prefill_hd512_bkv32_s2 512 32 -DPLOW_NV_FA512_WG=1
  prefill prefill_hd512_bkv64_s1 512 64 \
    -DPLOW_NV_FA512_WG=1 -DPLOW_NV_FA512_KV64=1

  "${nvenv[@]}" "$nvcc" "${common[@]}" -include cstdint -DPLOW_FA_BENCH_HD=512 \
    -DPLOW_FA_BENCH_NH=16 -DPLOW_FA_BENCH_KVH=1 \
    -DPLOW_FA_BENCH_SHORT_BURST=1 \
    "$repo/runtime/nvidia/experiments/fa_gf_full_h100_ab.cu" \
    -o "$build/decode_hd512_global"
  "${nvenv[@]}" "$nvcc" "${common[@]}" -include cstdint -DPLOW_FA_BENCH_HD=256 \
    -DPLOW_FA_BENCH_NH=16 -DPLOW_FA_BENCH_KVH=8 \
    -DPLOW_FA_BENCH_WINDOW=1024 -DPLOW_FA_BENCH_RING=16384 \
    -DPLOW_FA_BENCH_SHORT_BURST=1 \
    "$repo/runtime/nvidia/experiments/fa_gf_full_h100_ab.cu" \
    -o "$build/decode_hd256_local"
  "${nvenv[@]}" "$nvcc" "${common[@]}" -DPLOW_TEST_FA_ROWS=512 \
    -DPLOW_TEST_FA_HD256_ONLY=1 -DPLOW_NV_FA_WGITEM=1 \
    -DPLOW_NV_FA_GQA2_PAIR=1 \
    "$repo/runtime/tests/packed_flash_sm90_correct.cu" -lcuda \
    -o "$build/control_hd256_gqa2"
  "${nvenv[@]}" "$nvcc" "${common[@]}" -DPLOW_TEST_FA_ROWS=512 \
    "$repo/runtime/tests/packed_flash_sm90_correct.cu" -lcuda \
    -o "$build/control_hd512_role"
  rm -f "$build/sha256.txt"
  find "$build" -maxdepth 1 -type f -perm -u+x -print0 | sort -z | \
    xargs -0 sha256sum > "$build/sha256.txt"
}

run_all() {
  : > "$output"
  local clock_output=${output%.jsonl}-clocks.csv
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw \
    --format=csv,noheader,nounits --loop-ms=100 > "$clock_output" &
  local clock_pid=$!
  trap 'kill "$clock_pid" 2>/dev/null || true; wait "$clock_pid" 2>/dev/null || true' EXIT
  local bin rows kv ns
  local control_objects=${PLOW_FA_CONTROL_OBJECTS:-/tmp/gqa2-allrung-objects}
  test -f "$control_objects/interp_sm90a_pfpackedfa256_gqa2.cubin"
  test -f "$control_objects/interp_sm90a_pfattn_hd512.cubin"
  for rows in 128 256 512; do
    for kv in 1024 4096 16384; do
      "$build/control_hd256_gqa2" --campaign-timing --rows "$rows" \
        --kv-length "$kv" --interpreter \
        "$control_objects/interp_sm90a_pfpackedfa256_gqa2.cubin" >> "$output"
    done
    for kv in 1024 4096 8192 16384; do
      "$build/control_hd512_role" --campaign-timing --rows "$rows" \
        --kv-length "$kv" --lean-hd512 --interpreter \
        "$control_objects/interp_sm90a_pfattn_hd512.cubin" >> "$output"
    done
  done
  for ns in 1 2 3 4; do
    "$build/control_hd256_gqa2" --campaign-timing --seed "$ns" --rows 512 \
      --kv-length 16384 --interpreter \
      "$control_objects/interp_sm90a_pfpackedfa256_gqa2.cubin" >> "$output"
    "$build/control_hd512_role" --campaign-timing --seed "$ns" --rows 512 \
      --kv-length 16384 --lean-hd512 --interpreter \
      "$control_objects/interp_sm90a_pfattn_hd512.cubin" >> "$output"
  done
  # Exact warp/tile comparisons. Prefill packets use nsplit=1; add long-history
  # split probes only where global attention can otherwise underfill H100.
  for bin in prefill_hd256_pair32 prefill_hd256_wgitem32 prefill_hd256_onewg32 \
             prefill_hd256_hdsplit32 prefill_hd256_hdsplit64; do
    for rows in 128 256 512; do
      for kv in 1024 4096 16384; do
        "$build/$bin" "$rows" "$kv" 1 1 >> "$output"
      done
    done
  done
  for bin in prefill_hd512_bkv16_s2 prefill_hd512_bkv32_s2 \
             prefill_hd512_bkv64_s1; do
    for rows in 128 256 512; do
      for kv in 1024 4096 8192 16384; do
        "$build/$bin" "$rows" "$kv" 1 1 >> "$output"
      done
    done
  done
  for bin in prefill_hd256_pair32 prefill_hd256_wgitem32 prefill_hd256_onewg32 \
             prefill_hd256_hdsplit32 prefill_hd256_hdsplit64 \
             prefill_hd512_bkv16_s2 prefill_hd512_bkv32_s2 prefill_hd512_bkv64_s1; do
    for ns in 1 2 3 4; do
      "$build/$bin" 512 16384 1 1 "$ns" >> "$output"
    done
  done
  for rows in 128 512; do
    for ns in 1 2 4 8; do
      "$build/prefill_hd512_bkv32_s2" "$rows" 16384 "$ns" 1 >> "$output"
    done
  done

  # Decode nsplit is measured separately because merge cost and SM-grid
  # alignment differ from prefill. The current packet cap is 17.
  for kv in 1024 4096 8192 16384; do
    for ns in 8 12 16 17 18 24 32; do
      "$build/decode_hd256_local" "$kv" 2 "$ns" 1 "/tmp/fa-decode-hd256-$kv-$ns.bin" \
        >> "$output"
    done
    for ns in 12 16 17 18 24 32 33; do
      "$build/decode_hd512_global" "$kv" 2 "$ns" 1 "/tmp/fa-decode-hd512-$kv-gf2-$ns.bin" \
        >> "$output"
    done
    for ns in 16 17 18 24 32 33 34; do
      "$build/decode_hd512_global" "$kv" 4 "$ns" 1 "/tmp/fa-decode-hd512-$kv-gf4-$ns.bin" \
        >> "$output"
    done
    for ns in 32 33 48 64 66; do
      "$build/decode_hd512_global" "$kv" 8 "$ns" 1 "/tmp/fa-decode-hd512-$kv-gf8-$ns.bin" \
        >> "$output"
    done
  done
  kill "$clock_pid" 2>/dev/null || true
  wait "$clock_pid" 2>/dev/null || true
  trap - EXIT
}

case ${1:-} in
  build)
    build_all
    ;;
  run-leased)
    perf="$repo/perf-data/tools/gpulease"
    "$perf" -n 1 gemma4-attention-h100-sweep \
      env PLOW_FA_SWEEP_LEASED=1 \
      LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 \
      "$0" run
    ;;
  run)
    test "${PLOW_FA_SWEEP_LEASED:-0}" = 1 || {
      echo "run through: $0 run-leased" >&2
      exit 2
    }
    run_all
    ;;
  *)
    echo "usage: $0 build|run-leased" >&2
    exit 2
    ;;
esac
