#!/usr/bin/env bash
# p4_lock.sh <tile_kib> <path-C env...>: E2B L4 (0.92 MiB/worker, fully lockable) plain vs pseudo-lock at 12 and 14 L2
# ways, KV paths B / A / C at 32K and 128K, 3 reps, plus one counter run per cell. Restores the module parameters.
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
O=/tmp/g4c/l2r/results/p4_lock
R=/tmp/g4c/l2r/ref
T=$1; shift
CENV=("$@")
KO=$S/../../driver/pseudo_lock_sram.ko
mkdir -p $O
orig_l2=$(cat /sys/module/pseudo_lock_sram/parameters/l2_ways)
orig_l3=$(cat /sys/module/pseudo_lock_sram/parameters/l3_ways)
orig_io=$(cat /sys/module/pseudo_lock_sram/parameters/l3_io_ways)
reload() {
  if [ -n "$(grep -v '^#' /sys/class/misc/pseudo_lock/regions 2>/dev/null | grep -v '^$')" ]; then
    echo "live regions present; not reloading" >&2; return 1
  fi
  sudo rmmod pseudo_lock_sram && sudo insmod "$KO" "$@" && sudo chgrp "$(id -g)" /dev/pseudo_lock && sudo chmod 0660 /dev/pseudo_lock
  cat /sys/class/misc/pseudo_lock/caps
}
cells() { # mode-tag lock-env...
  local mt=$1; shift
  for ref in e2b.L4.c32768 e2b.L4.c131072; do
    for p in B A C; do
      if [ $p = C ]; then pa=A; ex=("${CENV[@]}"); else pa=$p; ex=(); fi
      for r in 1 2 3; do
        bash $S/p4_run.sh $O $ref.$mt.$p.r$r $R/$ref $pa 1 L2R_KV_TILE_KIB=$T "${ex[@]}" "$@"
      done
      L2R_BIN=/tmp/g4c/l2r/l2r_layer bash $S/p4_ctr.sh $O/ctr $ref.$mt.$p $R/$ref $pa 1 L2R_KV_TILE_KIB=$T "${ex[@]}" "$@" | tail -1
    done
  done
}
cells plain
for w in 12 14; do
  reload l2_ways=$w l3_ways=0 > $O/caps.w$w.txt || exit 1
  cells lock$w L2R_LOCK=1 L2R_RESIDENT_KIB=4096
done
reload l2_ways=$orig_l2 l3_ways=$orig_l3 l3_io_ways=$orig_io > $O/caps.restored.txt
python3 $S/p2_gate.py $O/*.json > $O/gate.txt; echo "gate rc=$?"
python3 $S/p4_sum.py $O > $O/summary.md
echo P4_LOCK_DONE
