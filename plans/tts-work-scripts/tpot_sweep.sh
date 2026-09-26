#!/usr/bin/env bash
# tpot_sweep.sh <plowrt> <assets> <out.jsonl> — greedy decode TPOT at conc 1,2,4,8,16 (one load each).
RT=$1; A=$2; OUT=$3; : > $OUT
for c in 1 2 4 8 16; do
  $RT bench --assets $A --random-input-len 64 --output-len 512 --concurrency $c \
     --requests $((c*2)) --warmup-requests $c --engine-diagnostics 2>/dev/null | tail -n 400 > /tmp/tpot_$$.json
  /root/tts-work/venv-ref/bin/python - $c /tmp/tpot_$$.json $OUT <<'EOF'
import json, sys
c, f, out = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(f).read(); i = txt.find("{")
d = json.loads(txt[i:])
keep = {k: v for k, v in d.items() if isinstance(v, (int, float)) and any(s in k for s in ("tpot", "ttft", "tok", "throughput", "decode_rung", "step"))}
keep["conc"] = int(c)
print(json.dumps(keep)); open(out, "a").write(json.dumps(keep) + "\n")
EOF
done
