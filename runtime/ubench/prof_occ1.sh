#!/usr/bin/env bash
# prof_occ1.sh — rocprofv3 MfmaUtil sweep for the occ-1 PoC. For each (elf, kernel, threads, shape)
# it runs the bench under rocprof (PROF=1: 8 warm + 6 timed dispatches, single shape), then averages
# MfmaUtil / MemUnitStalled / LdsBankConflict over the timed dispatches (Dispatch_Id > 8).
#
#   usage: prof_occ1.sh <elf> <kernel> <threads>   (sweeps shapes q/gate/down)
set -uo pipefail
cd "$(dirname "$0")/build_occ1"
ELF=$1 K=$2 THR=$3
DEV=${DEV:-0}
declare -A SNAME=([0]=q_proj [3]=gate/up [4]=down)
for S in 0 3 4; do
  OUT="p_${K}_s${S}"
  rm -rf "$OUT"
  env PROF=1 ROCR_VISIBLE_DEVICES=$DEV LD_LIBRARY_PATH=/opt/rocm/lib \
    rocprofv3 --pmc MfmaUtil MemUnitStalled LdsBankConflict --output-format csv -d "$OUT" \
    -- ./gemm_occ1_bench "$ELF" "$K" "$THR" 4096 qwen "$S" >/dev/null 2>&1
  CSV=$(find "$OUT" -name "*counter_collection.csv" | head -1)
  awk -F, -v k="$K" -v s="${SNAME[$S]}" '
    $2>8 {
      gsub(/"/,"",$16);
      if($16=="MfmaUtil"){mu+=$17; nm++}
      if($16=="MemUnitStalled"){ms+=$17; ns++}
      if($16=="LdsBankConflict"){lc+=$17; nl++}
    }
    END{ printf "  %-24s %-8s  MfmaUtil=%5.1f%%  MemUnitStalled=%5.1f%%  LdsBankConflict=%.0f\n",
        k, s, (nm?mu/nm:0), (ns?ms/ns:0), (nl?lc/nl:0) }' "$CSV"
done
