#!/usr/bin/env bash
# p0_bw.sh <outdir>: P0.3 tiers on the isolated worker set (L2R_CPUS, default the 90 inference cores).
#   L2: 1.0 / 1.3 / 1.5 MiB per core, L3: 3 / 4 MiB per core, DRAM: 64 MiB per core;
#   modes 0 (AVX-512 read), 1 (AMX-BF16 stream), 2 (AVX-512 BF16 GEMV); 4 KiB pages and THP; 3 reps each.
#   Sustained frequency: modes 3 / 4 (compute only) under turbostat; DRAM runs under uncore IMC CAS counters.
set -u
O=${1:?outdir}; mkdir -p "$O"
B=${L2R_BW:-/tmp/g4c/l2r/l2r_bw}
SECS=${SECS:-2}; REPS=${REPS:-3}
for huge in 0 1; do
  for sz in 1048576 1363148 1572864 3145728 4194304 67108864; do
    for mode in 0 1 2; do
      if [ $huge = 1 ]; then L2R_HUGE=1 $B $sz $mode $SECS $REPS; else $B $sz $mode $SECS $REPS; fi
    done
  done
done > "$O/tiers.jsonl" 2> "$O/tiers.err"
# IMC CAS counters while the DRAM tier runs (64 B per CAS).
sudo perf stat -a -x, -e 'uncore_imc/cas_count_read_sch0/,uncore_imc/cas_count_read_sch1/,uncore_imc/cas_count_write_sch0/,uncore_imc/cas_count_write_sch1/' \
  -o "$O/imc.csv" -- $B 67108864 0 $SECS 1 > "$O/imc_run.jsonl" 2>> "$O/tiers.err"
# Sustained frequency per mode: 10 s compute (and streams) with turbostat sampling every second.
for mode in 3 4 1 2; do
  sz=4096; [ $mode -le 2 ] && sz=1048576
  sudo turbostat --quiet --interval 1 --num_iterations 10 --show CPU,Busy%,Bzy_MHz,Avg_MHz,CoreTmp,PkgWatt,RAMWatt \
    -o "$O/turbostat.mode$mode.txt" > /dev/null 2>&1 &
  tp=$!
  sleep 0.5
  $B $sz $mode 10 1 >> "$O/freq_runs.jsonl"
  wait $tp
done
echo done > "$O/DONE"
