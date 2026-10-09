#!/usr/bin/env bash
# gen_refs.sh: FP32 layer references for the P2 stage (E2B L0 sliding / L4 full; E4B L0 sliding / L5 full).
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
for spec in "E2B 4 2048" "E4B 0 2048" "E4B 5 2048" "E2B 4 16384" "E4B 5 16384"; do
  set -- $spec
  m=$(echo $1 | tr A-Z a-z)
  /tmp/g4c/bin/vllm-py ref_layer.py /tmp/models/google/gemma-4-$1-it $2 $3 $R/$m.L$2.c$3 > $R/$m.L$2.c$3.log 2>&1
  echo "$spec rc=$? $(tail -1 $R/$m.L$2.c$3.log)"
done
echo REFS_DONE
