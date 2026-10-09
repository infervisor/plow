#!/usr/bin/env bash
# p5_ident.sh: greedy outputs with PLOW_CPU_COMBINE=0 vs 16 must be identical (E2B, E4B; 8 prompts, 96 tokens,
# plus 4 concurrent requests so batched decode rungs run too).
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
source $WT/scripts/bench/plowbench.sh
O=/tmp/g4c/l2r/results/p5_ident; mkdir -p $O
cat > $O/prompts.py <<'EOF'
import json, sys, urllib.request, concurrent.futures as cf
port, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
P = ["Explain how a mill race powers a waterwheel.", "Write a haiku about granite.", "List three uses of a lathe.",
     "Summarize the causes of the French Revolution in two sentences.", "What is 17 times 23? Show the steps.",
     "Translate 'the river is cold' into French and German.", "Describe the smell of rain to someone who cannot smell.",
     "Give a Python function that reverses a linked list."]
def ask(p):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": 96, "temperature": 0}).encode()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600)
    return json.load(r)["choices"][0]["message"]["content"]
res = [ask(p) for p in P]
with cf.ThreadPoolExecutor(4) as ex:
    res += list(ex.map(ask, P[:4]))
json.dump(res, open(out, "w"), indent=1)
EOF
for M in gemma-4-E2B-it gemma-4-E4B-it; do
  A=/tmp/g4c/l2r/rtx90-2k/$M; [ $M = gemma-4-E4B-it ] && A=/tmp/g4c/l2r/pk90/$M
  for g in 0 16; do
    port=$(pb_free_port)
    PLOW_CPU_WEIGHT_AFFINE=1 PLOW_CPU_COMBINE=$g PLOW_CPU_THREADS=90 pb_serve_start /tmp/g4c/l2r/bin/plowrt-comb-iso $A $A $port $O/$M.g$g.serve.log
    pb_serve_wait 900 || { pb_serve_stop; continue; }
    python3 $O/prompts.py $port "$(pb_model_id)" $O/$M.g$g.json
    pb_serve_stop
  done
  python3 -c "
import json; a=json.load(open('$O/$M.g0.json')); b=json.load(open('$O/$M.g16.json'))
print('$M identical %d/%d' % (sum(x==y for x,y in zip(a,b)), len(a)))"
done
