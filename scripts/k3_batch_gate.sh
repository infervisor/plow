#!/usr/bin/env bash
# k3_batch_gate.sh — the correctness gate batched K3 decode must pass BEFORE its refusals are
# lifted.                                                                    [K3-BATCH-GATE]
#
# WRITE THE GATE FIRST. `perf-data/k3-batched-decode-design.md` §5/§6.2 puts this ahead of the
# remaining wiring on purpose: the KDA recurrence is the model's core, a wrong one is FLUENT
# rather than broken, and the two refusals it would replace
# (`exec/amd.rs:3264`, `serve/engine.rs:187`) are today the only thing standing between a
# half-wired batch path and silent cross-sequence corruption. A gate that exists after the
# refusals are lifted is a gate that arrived too late.
#
# THE TWO CHECKS, and they catch different bugs:
#
#   A. IDENTICAL PROMPTS. B copies of one prompt must produce B IDENTICAL streams. Catches a
#      shared carried state directly: if slot 1's token threads into slot 0's KDA state the
#      streams diverge, and because every slot ran the same prompt any difference at all is a
#      bug rather than a legitimate difference.
#
#   B. DIFFERENT PROMPTS, RAGGED LENGTHS. B different prompts must each produce what that prompt
#      produces ALONE at B=1. Catches per-slot position and kvlen handling, which check A cannot
#      see because identical prompts share their positions. Lengths are deliberately unequal —
#      `perf-data/batched-decode-amd-status.md:19-31` is the precedent, where exactly this shape
#      (prompts of length 3/5/7/4) caught ragged-position bugs on the dense path.
#
# Check B is the one that matters most and the one most likely to be skipped, because it needs a
# B=1 reference run per prompt and is therefore B+1 model loads rather than one.
#
#   ./scripts/k3_batch_gate.sh <blob-dir> <hsaco-dir> <checkpoint> [B]
#
#   STEPS  decode steps per arm (default 24)
#   CTX    --ctx (default 5; see kimi-k3-README.md §5 — at ctx > prompt length the run decodes
#          over KV nobody prefilled, so a correctness A/B must use ctx == prompt length)
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOB="${1:?blob dir}"; HSACO="${2:?hsaco dir}"; CKPT="${3:?checkpoint}"; B="${4:-4}"
STEPS="${STEPS:-24}"; CTX="${CTX:-5}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
LEASE="$WT/perf-data/harness/gpulease"

# Four prompts of DELIBERATELY UNEQUAL length. Token ids, not text: `--prompt` takes ids.
P1="1008,10484,318,15383,387"          # 5 — the README's known-good "capital of France"
P2="1008,10484,318"                    # 3
P3="1008,10484,318,15383,387,13,646"   # 7
P4="1008,10484,318,15383"              # 4
PROMPTS=("$P1" "$P2" "$P3" "$P4")

run() { # <label> <prompt-spec> <extra-args...>
  local lbl="$1"; shift; local pr="$1"; shift
  local log="/tmp/k3bg_$lbl.log"
  GPU_LEASE_TIMEOUT=7200 "$LEASE" -n 8 "k3bg-$lbl" sg render -c \
    "PLOW_L2_PLACE_DISPATCH=1 nix develop $WT --command $BIN amd-bench --blob $BLOB/model.pkt \
     --hsaco $HSACO --checkpoint $CKPT --tp 8 --steps $STEPS --ctx $CTX --prompt '$pr' $*" \
    > "$log" 2>&1
  # the generated id list, one line per sequence slot
  grep -oE "^  \[[0-9, ]+\]" "$log"
}

echo "=== CHECK A: $B copies of one prompt must give $B identical streams ==="
same=""
for _ in $(seq 1 "$B"); do same="${same:+$same;}$P1"; done
mapfile -t A < <(run "identical" "$same" --batched)
if [ "${#A[@]}" -eq 0 ]; then
  echo "  NO STREAMS PARSED — see /tmp/k3bg_identical.log"
  grep -viE "^\[2m|dev shell|lean tool|amd gpu" /tmp/k3bg_identical.log | tail -4
  echo ">>> CHECK A: INCONCLUSIVE"; A_OK=2
elif [ "${#A[@]}" -ne "$B" ]; then
  echo "  expected $B streams, got ${#A[@]} — the engine is not running $B slots"
  echo ">>> CHECK A: FAIL"; A_OK=1
else
  uniq_n=$(printf '%s\n' "${A[@]}" | sort -u | wc -l)
  printf '  %s\n' "${A[@]}"
  [ "$uniq_n" -eq 1 ] && { echo ">>> CHECK A: PASS ($B identical)"; A_OK=0; } \
                      || { echo ">>> CHECK A: FAIL — $uniq_n distinct streams; a slot is reading another's state"; A_OK=1; }
fi

echo
echo "=== CHECK B: $B different prompts must each match their B=1 solo run ==="
declare -a SOLO
for i in $(seq 0 $((B-1))); do
  p="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
  mapfile -t one < <(run "solo$i" "$p")
  SOLO[$i]="${one[0]:-MISSING}"
  echo "  solo[$i] len=$(($(tr -cd ',' <<<"$p" | wc -c)+1)) -> ${SOLO[$i]:0:60}..."
done
spec=""; for i in $(seq 0 $((B-1))); do spec="${spec:+$spec;}${PROMPTS[$((i % ${#PROMPTS[@]}))]}"; done
mapfile -t BATCHED < <(run "ragged" "$spec" --batched)
B_OK=0
if [ "${#BATCHED[@]}" -ne "$B" ]; then
  echo "  expected $B streams, got ${#BATCHED[@]}"; B_OK=1
else
  for i in $(seq 0 $((B-1))); do
    if [ "${BATCHED[$i]}" = "${SOLO[$i]}" ]; then echo "  slot $i: MATCHES solo"
    else echo "  slot $i: DIFFERS from solo"; echo "     batched ${BATCHED[$i]:0:70}"; echo "     solo    ${SOLO[$i]:0:70}"; B_OK=1; fi
  done
fi
[ "$B_OK" -eq 0 ] && echo ">>> CHECK B: PASS" || echo ">>> CHECK B: FAIL — per-slot position/kvlen handling"

echo
# A ONE-SLOT BATCH PROVES NOTHING, and saying otherwise is worse than saying nothing. Both checks
# are about slots interfering with each other; with one slot there is nothing to interfere with,
# so they pass by construction on a build that has no batch support at all. B=1 is useful only as
# a smoke test that this script parses real output and compares it correctly.
if [ "$B" -lt 2 ]; then
  echo "BATCH GATE: VACUOUS at B=$B — both checks are about cross-slot interference and there is"
  echo "  only one slot. This run tested the HARNESS, not the engine. Re-run at B >= 2 (and"
  echo "  ideally >= 4, so check B exercises all four ragged lengths) before believing anything."
  exit 2
fi
if [ "${A_OK:-1}" -eq 0 ] && [ "$B_OK" -eq 0 ]; then
  echo "BATCH GATE: PASS at B=$B — the refusals may be lifted"
  exit 0
fi
echo "BATCH GATE: NOT PASSED — leave exec/amd.rs:3264 and serve/engine.rs:187 in place"
exit 1
