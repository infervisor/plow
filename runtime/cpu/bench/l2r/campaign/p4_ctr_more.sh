#!/usr/bin/env bash
# p4_ctr_more.sh: counter runs for paths B/A/C on E2B L4 2K/8K/64K and E4B L5 2K/32K/128K (plain, tile 256).
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
O=/tmp/g4c/l2r/results/p4_lock/ctr
R=/tmp/g4c/l2r/ref
for ref in e2b.L4.c2048 e2b.L4.c8192 e2b.L4.c65536 e4b.L5.c2048 e4b.L5.c32768 e4b.L5.c131072; do
  bash $S/p4_ctr.sh $O $ref.plain.B $R/$ref B 1 L2R_KV_TILE_KIB=256 | tail -1
  bash $S/p4_ctr.sh $O $ref.plain.A $R/$ref A 1 L2R_KV_TILE_KIB=256 | tail -1
  bash $S/p4_ctr.sh $O $ref.plain.C $R/$ref A 1 L2R_KV_TILE_KIB=256 L2R_KV_PFD=8 L2R_KV_PFH=t0 | tail -1
done
