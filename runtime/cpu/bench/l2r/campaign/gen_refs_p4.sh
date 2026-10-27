#!/usr/bin/env bash
# gen_refs_p4.sh: long-context FP32 layer references for P4 (full-attention layers 8K-128K, sliding layers at 128K),
# plus a chunked re-run of e2b.L4.c16384 to check chunked prefill against the P2 single-shot dump.
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
for spec in "E2B 4 16384 8192 chk" "E2B 4 8192 8192" "E4B 5 8192 8192" "E2B 4 32768 4096" "E4B 5 32768 4096" \
            "E2B 4 65536 2048" "E4B 5 65536 2048" "E2B 4 131072 2048" "E4B 5 131072 2048" \
            "E2B 0 131072 2048" "E4B 0 131072 2048"; do
  set -- $spec
  m=$(echo $1 | tr A-Z a-z)
  d=$R/$m.L$2.c$3${5:+.$5}
  [ -f $d/meta.json ] && { echo "$spec exists"; continue; }
  s=$(date +%s)
  /tmp/g4c/bin/vllm-py ref_layer.py /tmp/models/google/gemma-4-$1-it $2 $3 $d $4 > $d.log 2>&1
  echo "$spec rc=$? $(( $(date +%s) - s ))s $(tail -1 $d.log)"
done
echo REFS_DONE
