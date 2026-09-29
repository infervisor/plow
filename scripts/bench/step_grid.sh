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
# Output: <out>/steps.tsv (slots ctx mean_ms), sweep.b<B>.c<ctx>.jsonl, disasm.txt.
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -ge 2 ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
A=$1 O=$2
T=${CARGO_TARGET_DIR:-$HERE/target}/release
SB=${STEP_BENCH:-$T/examples/step_bench} RT=${PLOWRT:-$T/plowrt}
[ -x "$SB" ] || { echo "no step_bench at $SB (cargo build --release -p plowrt --features cuda --example step_bench)" >&2; exit 2; }
mkdir -p "$O"
cp "$SB" "$O/step_bench"
[ -x "$RT" ] && "$RT" disasm "$A/model.pkt" > "$O/disasm.txt" 2>/dev/null
sb() { PLOW_HSACO=$A PLOW_CHECKPOINT=${PLOW_CHECKPOINT:-$A/checkpoint} timeout "${STEP_TIMEOUT:-600}" "$O/step_bench" "$A" "$@"; }
printf 'slots\tctx\tmean_ms\n' > "$O/steps.tsv"
for c in ${CTXS:-512 1024 2048}; do
    for s in ${SLOTS:-1 64 128}; do
        ms=$(sb "$s" "$c" "${STEPS:-48}" --warmup 4 2>&1 | grep -o 'mean_ms=[0-9.]*' | tail -1 | cut -d= -f2)
        printf '%s\t%s\t%s\n' "$s" "$c" "${ms:-FAIL}" | tee -a "$O/steps.tsv"
    done
done
n=$(grep -c '^#' "$O/disasm.txt" 2>/dev/null || echo 0)
for bc in ${SWEEP-1x1024 64x1024}; do
    b=${bc%x*} c=${bc#*x}
    # The decode program's instruction count bounds the sweep; without a disasm sweep far enough.
    last=$(awk -v t="$b" '/^===== program T=/{p=($3=="T="t)} p&&/^#/{n=$1} END{print substr(n,2)+1}' "$O/disasm.txt" 2>/dev/null)
    [ -n "$last" ] && [ "$last" -gt 1 ] || last=${n:-2000}
    sb "$b" "$c" 10 --warmup 4 --sweep "0..$last" 2>&1 | grep '^{"cap"' > "$O/sweep.b$b.c$c.jsonl"
    echo "sweep B=$b ctx=$c: $(wc -l < "$O/sweep.b$b.c$c.jsonl") caps"
done
