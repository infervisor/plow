#!/usr/bin/env bash
# m_cold.sh: after m_e4b.sh -> 16K full-attention stages with cold KV (P4 path A: KV rotated through >= 1.5 GiB, every
# step from DRAM), AMX, B 1 / 16 (26B head: B 1, no 16K row dumps), as the pipeline sees KV with ~50 microbatches in flight
until grep -q M_E4B_DONE /tmp/g4c/l2r/m_e4b.log; do sleep 30; done
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/m_cold; mkdir -p $O
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.m2
rows() { local s=$R/$1$2; for i in $(seq 1 $(( $3 - 1 ))); do s=$s,$R/$1.o$i$2; done; echo $s; }
KV16="L2R_KV_TILE_KIB=256 L2R_KV_PFD=8"
one() { # <base> <suffix> <B>
  tag=$1$2.amx.b$3.A
  bash $S/p4_run.sh $O $tag $R/$1$2 A $3 L2R_GEMV=amx L2R_ROWS=$(rows $1 $2 $3) $KV16
  python3 $S/p2_gate.py $O/$tag.json 2>&1 | head -1 | sed "s/^/$tag /" | cut -c1-170
}
for b in 1 16; do
  one 12b.L5.c16384 .tp4r0 $b
  one 31b.L5.c16384 .tp8r0 $b
done
one 26b.L5.c16384 .head 1
echo M_COLD_DONE
