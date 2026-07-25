#!/usr/bin/env bash
# collect_occ1.sh — produce the committed PoC result set: per-(config,shape) MfmaUtil + TF/s, raw
# rocprof CSVs, and a summary CSV. Run under `sg render`.
set -uo pipefail
cd "$(dirname "$0")/build_occ1"
DEV=${DEV:-0}
RES=occ1_results
mkdir -p "$RES"
SUM="$RES/summary.csv"
echo "config,tile,waves,occ,shape,TFs,pct_peak,MfmaUtil_pct,MemUnitStalled_pct,LdsBankConflict" > "$SUM"
declare -A SN=([0]=q_proj [3]=gate/up [4]=down)

# config  elf  kernel  threads  tile  waves  occ
CONFIGS=(
  "occ1_128x256_d2 gemm_occ1_poc.elf occ1_128x256_d2 256 128x256 4 1"
  "occ1_128x256_d3 gemm_occ1_poc.elf occ1_128x256_d3 256 128x256 4 1"
  "occ1_128x256_d4 gemm_occ1_poc.elf occ1_128x256_d4 256 128x256 4 1"
  "occ1_192x256_d3 gemm_occ1_poc.elf occ1_192x256_d3 256 192x256 4 1"
  "occ1_256x256_d2 gemm_occ1_poc.elf occ1_256x256_d2 256 256x256 4 1"
  "occ1_256x256_d3 gemm_occ1_poc.elf occ1_256x256_d3 256 256x256 4 1"
  "occ1_128x128_bk128_d4 gemm_occ1_poc.elf occ1_128x128_bk128_d4 256 128x128 4 1"
  "bf16_gemm_c5 test_kernels.elf gemm_c5 512 192x256 8 2"
  "bf16_gemm_c0 test_kernels.elf gemm_c0 512 256x256 8 2"
  "bf16_gemm_c3 test_kernels.elf gemm_c3 512 128x128 8 2"
)

for row in "${CONFIGS[@]}"; do
  read -r name elf kern thr tile waves occ <<<"$row"
  # throughput (no prof): capture TF/s + pct per shape
  declare -A TFS PCT
  while read -r sh tf pct; do TFS[$sh]=$tf; PCT[$sh]=$pct; done < <(
    env ROCR_VISIBLE_DEVICES=$DEV LD_LIBRARY_PATH=/opt/rocm/lib \
      ./gemm_occ1_bench "$elf" "$kern" "$thr" 4096 2>/dev/null \
    | awk '/q_proj|gate\/up|down/{gsub(/%/,"",$8); print $1, $6, $8}')
  # MfmaUtil per shape
  for S in 0 3 4; do
    OUT="$RES/${name}_s${S}"; rm -rf "$OUT"
    env PROF=1 ROCR_VISIBLE_DEVICES=$DEV LD_LIBRARY_PATH=/opt/rocm/lib \
      rocprofv3 --pmc MfmaUtil MemUnitStalled LdsBankConflict --output-format csv -d "$OUT" \
      -- ./gemm_occ1_bench "$elf" "$kern" "$thr" 4096 qwen "$S" >/dev/null 2>&1
    CSV=$(find "$OUT" -name "*counter_collection.csv" | head -1)
    read -r mu ms lc < <(awk -F, '$2>8{gsub(/"/,"",$16);
      if($16=="MfmaUtil"){mu+=$17;n++} if($16=="MemUnitStalled"){ms+=$17;m++} if($16=="LdsBankConflict"){lc+=$17;l++}}
      END{printf "%.1f %.2f %.0f", (n?mu/n:0),(m?ms/m:0),(l?lc/l:0)}' "$CSV")
    shp=${SN[$S]}
    echo "$name,$tile,$waves,$occ,$shp,${TFS[$shp]:-NA},${PCT[$shp]:-NA},$mu,$ms,$lc" >> "$SUM"
  done
done
echo "=== summary ==="
column -t -s, "$SUM"
