#!/usr/bin/env bash
# m_res.sh: multi-model P1 / P4 / stability, after m_sweep.sh (l2r_layer.m2, AVX where a lock must cover the largest slice):
#   residency plain vs lock under counters, B 1 / 16: 12 ways (12B L0 TP4, 26B L0 head, 26B L0 expert group),
#     then 14 ways (31B L0 TP8, 12B L5 TP4)
#   P4: 128K context by tiling the 16K dumps 8x (L2R_CTX_REPEAT; timing only), B=1, cold (path A) and warm (path B)
#   stability: 12B L0 TP4 and the 26B expert group (12 ways, lock), 31B L0 TP8 (14 ways, lock), B=16, 10 min each
#     (runs of ~60 s, each gated and counted), turbostat throughout. Driver back at 12 ways at the end.
set -u
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
KO=$S/../../driver/pseudo_lock_sram.ko
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/m_res; mkdir -p $O
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.m2
until grep -q M_SWEEP_DONE /tmp/g4c/l2r/m_chain.log; do sleep 30; done
ways() { # reload pseudo_lock_sram with l2_ways=$1 when no region is live
  grep -q "$2 B/core" /sys/class/misc/pseudo_lock/caps && return 0
  [ -n "$(grep -v '^#' /sys/class/misc/pseudo_lock/regions | grep -v '^$')" ] && { echo "regions live, cannot reload"; return 1; }
  sudo rmmod pseudo_lock_sram && sudo insmod "$KO" l2_ways=$1 l3_ways=0 && sudo chgrp "$(id -g)" /dev/pseudo_lock && sudo chmod 0660 /dev/pseudo_lock
  grep -q "$2 B/core" /sys/class/misc/pseudo_lock/caps || { echo "driver not at $1 ways"; return 1; }
}
rows() { local s=$R/$1$2; for i in $(seq 1 $(( $3 - 1 ))); do s=$s,$R/$1.o$i$2; done; echo $s; }
eg0=$(python3 -c "
import json, glob
print(max(glob.glob('$R/26b.L0.c2048.ex13g*/meta.json'), key=lambda f: json.load(open(f))['pairs']).split('/')[-2].split('.')[-1])")
res() { # <base> <suffix> <stage> <kib> <lockenv>
  for arm in plain lock; do
    for b in 1 16; do
      ex=""; [ $arm = lock ] && ex="$5"
      tag=$3.avx.b$b.$arm
      bash $S/p4_ctr.sh $O $tag $R/$1$2 B $b L2R_GEMV=avx L2R_ROWS=$(rows $1 $2 $b) $ex | tail -1
      echo "$tag $(python3 $S/p2_gate.py $O/$tag.json 2>&1 | head -1 | cut -c1-120) $(grep -o '"lock":{[^}]*}' $O/$tag.json)"
    done
  done
}
ways 12 1376256 || exit 1
res 12b.L0.c2048 .tp4r0 12b.L0.c2048.tp4r0 1344 "L2R_RESIDENT_KIB=1344 L2R_LOCK=1"
res 26b.L0.c2048 .head 26b.L0.c2048.head 1344 "L2R_RESIDENT_KIB=1344 L2R_LOCK=1"
res 26b.L0.c2048 .$eg0 26b.L0.c2048.$eg0 1344 "L2R_LOCK=1 L2R_EXPERTS_LOCK=1"
# P4: 128K by tiling (no numerics), B=1
for st in 12b.L5.c16384.tp4r0 31b.L5.c16384.tp8r0 26b.L5.c16384.head; do
  for p in A B; do
    tag=${st/c16384/c131072}.amx.b1.$p
    bash $S/p4_ctr.sh $O $tag $R/$st $p 1 L2R_GEMV=amx L2R_CTX_REPEAT=8 L2R_KV_TILE_KIB=256 L2R_KV_PFD=8 | tail -1
  done
done
stab() { # <name> <base> <suffix> <envs...>: 10 runs of ~60 s
  local name=$1 base=$2 suf=$3; shift 3
  local D=$O/stab.$name; rm -rf $D; mkdir -p $D
  sudo turbostat --quiet --interval 10 --show Time_Of_Day_Seconds,Busy%,Bzy_MHz,PkgWatt,RAMWatt -o $D/turbostat.txt &
  local TS=$! t0=$(date +%s)
  local p50=$(python3 -c "import json; print(json.loads(open('$O/$name.avx.b16.lock.json').read().strip().splitlines()[-1])['step_us']['p50'])" 2>/dev/null || echo 500)
  local steps=$(python3 -c "print(int(60e6 / $p50))")
  for i in $(seq -w 1 10); do
    P4_STEPS=$steps bash $S/p4_ctr.sh $D stab.$i $R/$base$suf B 16 L2R_GEMV=avx L2R_ROWS=$(rows $base $suf 16) "$@" > /dev/null 2>&1
    echo "$name stab.$i t=$(( $(date +%s) - t0 ))s $(python3 $S/p2_gate.py $D/stab.$i.json 2>&1 | head -1 | cut -c1-110) $(grep -o '"lock":{[^}]*}' $D/stab.$i.json)"
  done
  sudo pkill -f "turbostat --quiet --interval 10"; wait $TS 2>/dev/null
  python3 $S/p5_stab_sum.py $D
}
stab 12b.L0.c2048.tp4r0 12b.L0.c2048 .tp4r0 L2R_RESIDENT_KIB=1344 L2R_LOCK=1
stab 26b.L0.c2048.$eg0 26b.L0.c2048 .$eg0 L2R_LOCK=1 L2R_EXPERTS_LOCK=1
ways 14 1605632 || exit 1
res 31b.L0.c2048 .tp8r0 31b.L0.c2048.tp8r0 1568 "L2R_RESIDENT_KIB=1568 L2R_LOCK=1"
res 12b.L5.c2048 .tp4r0 12b.L5.c2048.tp4r0 1568 "L2R_RESIDENT_KIB=1568 L2R_LOCK=1"
stab 31b.L0.c2048.tp8r0 31b.L0.c2048 .tp8r0 L2R_RESIDENT_KIB=1568 L2R_LOCK=1
ways 12 1376256
cat /sys/class/misc/pseudo_lock/caps
echo M_RES_DONE
