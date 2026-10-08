#!/usr/bin/env bash
# p1_matrix.sh <outdir> [ways...]: P1.2 capacity x interference matrix with l2r_lock.
#   For each L2 lock way count (default "12 14"): reload pseudo_lock_sram with l2_ways=<w> l3_ways=0 (only
#   when no region is live), then for weight slices 0.5 / 1.0 / 1.25 MiB and the way count's lockable limit:
#   normal warm cache vs driver lock, scenarios A-D, 10 s x 3 reps, AMX kernel; AVX kernel on A and C.
#   Restores the module parameters it found. Long runs (scenario E) are a separate invocation: P1_LONG=1.
set -u
O=${1:?outdir}; shift; WAYS=${*:-12 14}
B=${L2R_LOCK_BIN:-/tmp/g4c/l2r/l2r_lock}
KO=$(dirname "$0")/../../driver/pseudo_lock_sram.ko
mkdir -p "$O"
orig_l2=$(cat /sys/module/pseudo_lock_sram/parameters/l2_ways 2>/dev/null)
orig_l3=$(cat /sys/module/pseudo_lock_sram/parameters/l3_ways 2>/dev/null)
orig_io=$(cat /sys/module/pseudo_lock_sram/parameters/l3_io_ways 2>/dev/null)
reload() {
  if [ -n "$(grep -v '^#' /sys/class/misc/pseudo_lock/regions 2>/dev/null | grep -v '^$')" ]; then
    echo "live regions present; not reloading" >&2; return 1
  fi
  sudo rmmod pseudo_lock_sram && sudo insmod "$KO" "$@" && sudo chgrp "$(id -g)" /dev/pseudo_lock && sudo chmod 0660 /dev/pseudo_lock
  cat /sys/class/misc/pseudo_lock/caps
}
for w in $WAYS; do
  reload l2_ways=$w l3_ways=0 > "$O/caps.w$w.txt" || exit 1
  lockable=$(awk '{for(i=1;i<=NF;i++) if($i=="B/core,") print $(i-1)}' "$O/caps.w$w.txt")
  for sz in 524288 1048576 1310720 $((lockable / 4096 * 4096)); do
    for lock in 0 1; do
      for sc in A B C D; do
        for rep in 1 2 3; do
          L2R_LOCK=$lock L2R_KERNEL=amx $B $sz 10 $sc > "$O/w$w.s$sz.l$lock.$sc.amx.r$rep.jsonl" 2>> "$O/err.log"
        done
      done
      for sc in A C; do
        L2R_LOCK=$lock L2R_KERNEL=avx $B $sz 10 $sc > "$O/w$w.s$sz.l$lock.$sc.avx.r1.jsonl" 2>> "$O/err.log"
      done
    done
  done
  if [ "${P1_LONG:-0}" = 1 ]; then
    for secs in 600 1800; do
      L2R_LOCK=1 L2R_KERNEL=amx $B $((lockable / 4096 * 4096)) $secs E > "$O/w$w.long$secs.l1.E.amx.jsonl" 2>> "$O/err.log"
    done
  fi
done
reload l2_ways=$orig_l2 l3_ways=$orig_l3 l3_io_ways=$orig_io > "$O/caps.restored.txt"
echo done > "$O/DONE"
