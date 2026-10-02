#!/usr/bin/env bash
# step_grid.sh <assets> <outdir> — plow's kernel-only decode step at matched context (the plow
# half of llm_grid.sh's d<B>x<ctx> cells; served decode ticks land within ~1% of it), plus
# per-instruction sweeps for scripts/bench/op_roof.py. Run inside a lease:
#
#   $GL -n 1 steps timeout 1800 scripts/bench/step_grid.sh <assets> <out>
#   scripts/bench/op_roof.py <out>/disasm.txt --ctx 1024 --sweep 64=<out>/sweep.b64.c1024.jsonl
#
# Env (defaults): STEP_BENCH (<CARGO_TARGET_DIR or repo target>/release/examples/step_bench),
# PLOWRT (same dir's plowrt, for disasm), SLOTS ("1 64 128"), CTXS ("512 1024 2048"), STEPS (48),
# SWEEP ("1x1024 64x1024": B x ctx rungs to sweep; empty = none), STEP_TIMEOUT (600 s per run).
# Output: <out>/steps.tsv (slots ctx mean_ms), step.b<B>.c<ctx>.log (including token digest),
# sweep.b<B>.c<ctx>.jsonl, disasm.txt.
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -ge 2 ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
A=$1 O=$2
T=${CARGO_TARGET_DIR:-$HERE/target}/release
SB=${STEP_BENCH:-$T/examples/step_bench} RT=${PLOWRT:-$T/plowrt}
[ -x "$SB" ] || { echo "no step_bench at $SB (cargo build --release -p plowrt --features cuda --example step_bench)" >&2; exit 2; }
mkdir -p "$O"
cp "$SB" "$O/step_bench"
[ -x "$RT" ] || { echo "no plowrt disassembler at $RT" >&2; exit 2; }
"$RT" disasm "$A/model.pkt" > "$O/disasm.txt" || exit 1
[ -s "$O/disasm.txt" ] || { echo "empty disassembly at $O/disasm.txt" >&2; exit 1; }
sb() { PLOW_HSACO=$A PLOW_CHECKPOINT=${PLOW_CHECKPOINT:-$A/checkpoint} timeout "${STEP_TIMEOUT:-600}" "$O/step_bench" "$A" "$@"; }
printf 'slots\tctx\tmean_ms\n' > "$O/steps.tsv"
failed=0
for c in ${CTXS:-512 1024 2048}; do
    for s in ${SLOTS:-1 64 128}; do
        log="$O/step.b$s.c$c.log"
        if sb "$s" "$c" "${STEPS:-48}" --warmup 4 > "$log" 2>&1; then
            ms=$(grep -o 'mean_ms=[0-9.]*' "$log" | tail -1 | cut -d= -f2)
            [ -n "$ms" ] || failed=1
        else
            ms=""
            failed=1
        fi
        printf '%s\t%s\t%s\n' "$s" "$c" "${ms:-FAIL}" | tee -a "$O/steps.tsv"
    done
done
[ "$failed" -eq 0 ] || { echo "one or more step runs failed; see $O/steps.tsv" >&2; exit 1; }
n=$(grep -c '^#' "$O/disasm.txt" 2>/dev/null || echo 0)
for bc in ${SWEEP-1x1024 64x1024}; do
    b=${bc%x*} c=${bc#*x}
    # The decode program's instruction count bounds the sweep.
    last=$(PYTHONPATH="$HERE/scripts/bench${PYTHONPATH:+:$PYTHONPATH}" python3 - "$O/disasm.txt" "$b" <<'PY'
import sys
from op_roof import parse

programs = [p for p in parse(open(sys.argv[1]).read())
            if p[0] == "decode" and p[1] == int(sys.argv[2])]
if len(programs) != 1:
    raise SystemExit(f"expected one decode T={sys.argv[2]} in disassembly, found {len(programs)}")
indices = [inst[0] for inst in programs[0][2]]
if indices != list(range(len(indices))):
    raise SystemExit("decode instruction indices are not contiguous from zero")
print(len(indices))
PY
    ) || exit 1
    [ "$last" -gt 1 ] || { echo "empty decode T=$b in disassembly" >&2; exit 1; }
    sweep_log="$O/sweep.b$b.c$c.log"
    if ! sb "$b" "$c" 10 --warmup 4 --sweep "0..$last" > "$sweep_log" 2>&1; then
        tail -n 3 "$sweep_log" >&2
        exit 1
    fi
    grep '^{"cap"' "$sweep_log" > "$O/sweep.b$b.c$c.jsonl"
    PYTHONPATH="$HERE/scripts/bench${PYTHONPATH:+:$PYTHONPATH}" python3 - "$O/sweep.b$b.c$c.jsonl" "$last" <<'PY' || exit 1
import sys
from op_roof import sweep_deltas

_, deltas, _ = sweep_deltas(sys.argv[1])
if len(deltas) != int(sys.argv[2]):
    raise SystemExit(f"sweep has {len(deltas)} instruction deltas, expected {sys.argv[2]}")
PY
    echo "sweep B=$b ctx=$c: $(wc -l < "$O/sweep.b$b.c$c.jsonl") caps"
done
