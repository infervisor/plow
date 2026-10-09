#!/usr/bin/env bash
# comb_prof.sh: decode-step profile, PLOW_CPU_COMBINE = 0 / 4 / 8 / 16, twice each.
for rep in 1 2; do
  for g in 0 4 8 16; do
    taskset -c 2-31,34-63,66-95,98-127,130-159,162-191 env PLOW_CPU_WEIGHT_AFFINE=1 PLOW_CPU_COMBINE=$g PROF_DUMP=/tmp/g4c/l2r/results/p5_comb.g$g.r$rep.csv \
      /tmp/plow-target-l2r/release/examples/cpu_profile /tmp/g4c/l2r/rtx90-2k/gemma-4-E2B-it/model.pkt /tmp/models/google/gemma-4-E2B-it --threads 90 --prompt-tokens 1000 2>&1 \
      | grep -E "decode step|tokens:" | tr '\n' ' ' | sed "s/^/g$g r$rep: /"; echo
    python3 /tmp/g4c/l2r/prof_bump.py /tmp/g4c/l2r/results/p5_comb.g$g.r$rep.csv.decode_step | grep -E "\| GEMV \||\| GEMV_GLU|consecutive"
  done
done
