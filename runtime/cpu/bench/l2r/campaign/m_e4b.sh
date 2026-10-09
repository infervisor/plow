#!/usr/bin/env bash
# m_e4b.sh: after m_res.sh -> E4B TP=2 socket slices (the 12-way tp plan's E4B stage): L0 / L5 at 2K, rows o0..o15
# (o0 every rank), then the same sweep as m_sweep.sh (AMX B 1/4/8/16, AVX B 1/16, 2 reps)
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
until grep -q M_RES_DONE /tmp/g4c/l2r/m_res.log; do sleep 30; done
R=/tmp/g4c/l2r/ref
S=$PWD
for L in 0 5; do
  for r in $(seq 0 15); do
    d=$R/e4b.L$L.c2048; [ $r -gt 0 ] && d=$d.o$r
    [ -f $d.tp2r0/meta.json ] && continue
    tp=2:0; [ $r = 0 ] && tp=2
    /tmp/g4c/bin/vllm-py ref_layer.py /tmp/models/google/gemma-4-E4B-it $L 2048 $d.tp 8192 $((r * 997)) $tp > $d.tp.log 2>&1
    rc=$?
    for x in $d.tp.tp2r*; do mv $x ${x/.tp.tp2/.tp2}; done
    echo "e4b L$L o$r tp $tp rc=$rc $(tail -1 $d.tp.log | cut -c1-160)"
  done
done
O=/tmp/g4c/l2r/results/m_sweep
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.m2
rows() { local s=$R/$1$2; for i in $(seq 1 $(( $3 - 1 ))); do s=$s,$R/$1.o$i$2; done; echo $s; }
for L in 0 5; do
  for g in amx avx; do
    for b in 1 4 8 16; do
      [ $g = avx ] && [ $b != 1 ] && [ $b != 16 ] && continue
      for rep in 1 2; do
        tag=e4b.L$L.c2048.tp2r0.$g.b$b.r$rep
        bash $S/p4_run.sh $O $tag $R/e4b.L$L.c2048.tp2r0 B $b L2R_GEMV=$g L2R_ROWS=$(rows e4b.L$L.c2048 .tp2r0 $b) > /dev/null
        python3 $S/p2_gate.py $O/$tag.json 2>&1 | head -1 | sed "s/^/$tag /" | cut -c1-170
      done
    done
  done
done
echo M_E4B_DONE
