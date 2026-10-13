#!/usr/bin/env bash
# step_boundary.sh <assets> <outdir> — kernel-only decode steps at the KV block boundaries, with
# CUDA_LAUNCH_BLOCKING so a fault names its launch. Live KV commits KV in 4096-row blocks; a
# kernel that stages rows past kv_len reads the unmapped next block only when the context sits
# just below a boundary, which the round contexts of step_grid.sh never exercise (the packed
# prefill fault of 2026-10-11 reproduced at ctx 4094/4095/8191 and passed at 4080). Run through
# the queue, inside a lease:
#
#   scripts/bench/gpuq.py submit boundary 1 timeout 1800 scripts/bench/step_boundary.sh <assets> <out>
#
# Env (defaults): STEP_BENCH (<CARGO_TARGET_DIR or repo target>/release/examples/step_bench),
# SLOTS ("1 8"), CTXS ("4094 4095 4096 8191 8192"), STEPS (12), STEP_TIMEOUT (600 s per run),
# PLOW_VMM_LIVE (1). Output: <out>/boundary.tsv (slots ctx mean_ms|FAIL|SKIP when ctx + STEPS exceeds max_ctx),
# step.b<B>.c<ctx>.log.
# Exit 1 on any failed step.
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -ge 2 ] || { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
A=$1 O=$2
T=${CARGO_TARGET_DIR:-$HERE/target}/release
SB=${STEP_BENCH:-$T/examples/step_bench}
[ -x "$SB" ] || { echo "no step_bench at $SB (cargo build --release -p plowrt --features cuda --example step_bench)" >&2; exit 2; }
mkdir -p "$O"
cp "$SB" "$O/step_bench"
export CUDA_LAUNCH_BLOCKING=1 PLOW_VMM_LIVE=${PLOW_VMM_LIVE:-1}
sb() { PLOW_HSACO=$A PLOW_CHECKPOINT=${PLOW_CHECKPOINT:-$A/checkpoint} timeout "${STEP_TIMEOUT:-600}" "$O/step_bench" "$A" "$@"; }
printf 'slots\tctx\tmean_ms\n' > "$O/boundary.tsv"
failed=0
for c in ${CTXS:-4094 4095 4096 8191 8192}; do
    for s in ${SLOTS:-1 8}; do
        log="$O/step.b$s.c$c.log"
        if sb "$s" "$c" "${STEPS:-12}" --warmup 2 > "$log" 2>&1; then
            ms=$(grep -o 'mean_ms=[0-9.]*' "$log" | tail -1 | cut -d= -f2)
            [ -n "$ms" ] || failed=1
        elif grep -q "exceeds the compiled context" "$log"; then
            # ctx + STEPS past the packet's max_ctx: not a boundary the packet can reach.
            ms=SKIP
        else
            ms=""
            failed=1
            grep -m1 -i "error\|fault\|illegal" "$log" | cut -c1-160 >&2
        fi
        printf '%s\t%s\t%s\n' "$s" "$c" "${ms:-FAIL}" | tee -a "$O/boundary.tsv"
    done
done
[ "$failed" -eq 0 ] || { echo "one or more boundary steps failed; see $O/boundary.tsv" >&2; exit 1; }
