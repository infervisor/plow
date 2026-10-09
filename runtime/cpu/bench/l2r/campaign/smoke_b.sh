#!/usr/bin/env bash
# smoke_b.sh: batched l2r_layer numerics: B=1 old vs new binary (same refdir), then B=4 and 16 with distinct rows, AMX + AVX
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/smoke_b; mkdir -p $O
ref=$R/e2b.L0.c2048
rows=$ref; for i in $(seq 1 15); do rows=$rows,$ref.o$i; done
for g in avx amx; do
  L2R_GEMV=$g L2R_BCAST=repnt /tmp/g4c/l2r/l2r_layer $ref 300 > $O/old.$g.json 2> $O/old.$g.err
  python3 $S/p2_gate.py $O/old.$g.json | head -1 | sed "s/^/old b1 /"
  for b in 1 4 16; do
    L2R_GEMV=$g L2R_BCAST=repnt L2R_BATCH=$b L2R_ROWS=$rows /tmp/g4c/l2r/l2r_layer.b $ref 300 > $O/new.$g.b$b.json 2> $O/new.$g.b$b.err
    python3 $S/p2_gate.py $O/new.$g.b$b.json | sed "s/^/new b$b /" | grep -v "^new b$b     q:" | head -4
  done
  L2R_GEMV=$g L2R_BCAST=direct L2R_BATCH=4 L2R_ROWS=$rows /tmp/g4c/l2r/l2r_layer.b $ref 300 > $O/new.$g.direct.b4.json 2> $O/new.$g.direct.b4.err
  python3 $S/p2_gate.py $O/new.$g.direct.b4.json | head -1 | sed "s/^/direct b4 /"
done
python3 - <<'EOF'
import json
for g in ("avx", "amx"):
    a = json.loads(open(f"/tmp/g4c/l2r/results/smoke_b/old.{g}.json").read().splitlines()[-1])["err"]
    b = json.loads(open(f"/tmp/g4c/l2r/results/smoke_b/new.{g}.b1.json").read().splitlines()[-1])["err"]
    print(g, "b1 err identical to old binary:", all(a[k] == b[k] for k in a))
EOF
