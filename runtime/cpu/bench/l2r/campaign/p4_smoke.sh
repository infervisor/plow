#!/usr/bin/env bash
# p4_smoke.sh: numerics of the P4 knobs (not timed).
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
O=/tmp/g4c/l2r/results/p4_smoke; mkdir -p $O
B=/tmp/g4c/l2r/l2r_layer.p4
R=/tmp/g4c/l2r/ref
run() { tag=$1 ref=$2; shift 2; env L2R_BCAST=repnt "$@" $B $R/$ref 200 > $O/$tag.json 2> $O/$tag.err; echo "$tag rc=$?"; }
run base e2b.L4.c16384
run tile16 e2b.L4.c16384 L2R_KV_TILE_KIB=16
run tile256 e2b.L4.c16384 L2R_KV_TILE_KIB=256
run seq4 e2b.L4.c16384 L2R_SEQS=4 L2R_KV_TILE_KIB=64
run cp4pf e2b.L4.c16384 L2R_KV_COPIES=4 L2R_KV_PFD=8 L2R_KV_PFH=t0 L2R_KV_EARLY_KIB=64
run e4bl5 e4b.L5.c16384 L2R_KV_TILE_KIB=64 L2R_SEQS=2
run e2bl0 e2b.L0.c2048 L2R_KV_TILE_KIB=16
python3 $S/p2_gate.py $O/*.json
grep -ho '"attn_seqs_worst_rel_rms":[^,]*' $O/*.json
