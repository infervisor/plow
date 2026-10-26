#!/usr/bin/env bash
# m_moe23.sh: 26B-A4B L0 expert groups at 2 and 3 sockets per layer (64 / 43 experts, 760 / 507 MB per socket: L2 + L3),
# rows o0..o15; then every group at B 1 / 4 / 8 / 16 (AMX) and 1 / 16 (AVX), 2 reps, with L2R_EXPERTS_ROTATE (each step
# reads a different subset of the socket's experts, as a pipeline does) and without (same experts every step)
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
S=$PWD
for G in 2 3; do
  for r in $(seq 0 15); do
    d=$R/26bg$G.L0.c2048; [ $r -gt 0 ] && d=$R/26bg$G.L0.c2048.o$r
    [ -f $d.ex${G}g0/meta.json ] && continue
    /tmp/g4c/bin/vllm-py ref_layer.py /tmp/models/google/gemma-4-26B-A4B-it 0 2048 $d 8192 $((r * 997)) moe:$G > $d.log 2>&1
    echo "G$G o$r rc=$? $(tail -1 $d.log | cut -c1-120)"
  done
done
echo REFS_DONE
O=/tmp/g4c/l2r/results/m_moe23; mkdir -p $O
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.m3
rows() { local s=$R/$1$2; for i in $(seq 1 $(( $3 - 1 ))); do s=$s,$R/$1.o$i$2; done; echo $s; }
for G in 2 3; do
  for g in $(seq 0 $((G - 1))); do
    for v in rot same; do
      ex=""; [ $v = rot ] && ex="L2R_EXPERTS_ROTATE=37"
      for gm in amx avx; do
        for b in 1 4 8 16; do
          [ $gm = avx ] && [ $b != 1 ] && [ $b != 16 ] && continue
          for rep in 1 2; do
            tag=26bg$G.L0.c2048.ex${G}g$g.$gm.b$b.$v.r$rep
            bash $S/p4_run.sh $O $tag $R/26bg$G.L0.c2048.ex${G}g$g B $b L2R_GEMV=$gm L2R_ROWS=$(rows 26bg$G.L0.c2048 .ex${G}g$g $b) $ex > /dev/null
            python3 $S/p2_gate.py $O/$tag.json 2>&1 | head -1 | sed "s/^/$tag /" | cut -c1-170
          done
        done
      done
    done
  done
done
echo M_MOE23_DONE
