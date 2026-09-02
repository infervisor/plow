#!/usr/bin/env bash
# k3_batch_gate.sh — the correctness gate batched K3 decode must pass BEFORE its refusals are
# lifted.                                                                    [K3-BATCH-GATE]
#
# WRITE THE GATE FIRST. `perf-data/archive/k3/k3-batched-decode-design.md` §5/§6.2 puts this ahead of the
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
#   B. DIFFERENT PROMPTS, RAGGED LENGTHS, COMPARED ACROSS TWO BATCH WIDTHS. The same B different
#      prompts are run at width B and at a SECOND width, and slot s must agree between them.
#      Catches per-slot position and kvlen handling, which check A cannot see because identical
#      prompts share their positions. Lengths are deliberately unequal —
#      `perf-data/batched-decode-amd-status.md:19-31` is the precedent, where exactly this shape
#      (prompts of length 3/5/7/4) caught ragged-position bugs on the dense path.
#
#      IT COMPARES TWO BATCHED RUNS, NOT A BATCHED RUN AGAINST A SOLO ONE, and that is a
#      correction rather than a convenience. A batched decode routes MoE through the GROUPED
#      expert kernel and a B=1 decode through the per-slot one; they accumulate in different
#      orders, and greedy decoding turns any tie-break into a different token a few steps later.
#      Measured: B=1 continues "The population is approximately 67 million people", B=4/8/16 all
#      continue "The capital of Germany is Berlin" — both fluent, both right, neither a defect.
#      Token-identity across those two paths is a criterion no correct implementation can meet,
#      so demanding it made the gate report FAIL on a working batch. Two batched widths DO share
#      a kernel, so between them token-identity is exactly the right bar — and it still tests
#      what check B is for, because the per-slot strides, positions and kvlens differ with width.
#
# Check B is the one that matters most and the one most likely to be skipped, because it needs a
# B=1 reference run per prompt and is therefore B+1 model loads rather than one.
#
#   ./scripts/k3_batch_gate.sh <blob-dir> <hsaco-dir> <checkpoint> [B] [alt-blob] [alt-hsaco]
#
# `alt-blob`/`alt-hsaco` are a build at a DIFFERENT batch width (both must move together — the
# hsaco carries PLOW_DECODE_BATCH too, since it sizes PLOW_GEMV_MM). Check B is skipped, loudly,
# without them.
#
# THE ALT BUILD MUST BE A DIFFERENT WIDTH, and reusing $BLOB for it compares the batch path
# against ITSELF at the same width — which passes whenever the batch path is self-consistently
# wrong.
#
#   STEPS  decode steps per arm (default 24)
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOB="${1:?blob dir}"; HSACO="${2:?hsaco dir}"; CKPT="${3:?checkpoint}"; B="${4:-4}"
ALT_BLOB="${5:-}"; ALT_HSACO="${6:-}"
STEPS="${STEPS:-24}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
LEASE="$WT/perf-data/tools/gpulease"

# Four prompts of DELIBERATELY UNEQUAL length. Token ids, not text: `--prompt` takes ids.
P1="1008,10484,318,15383,387"          # 5 — the README's known-good "capital of France"
P2="1008,10484,318"                    # 3
P3="1008,10484,318,15383,387,13,646"   # 7
P4="1008,10484,318,15383"              # 4
PROMPTS=("$P1" "$P2" "$P3" "$P4")

run() { # <label> <blob-dir> <hsaco-dir> <prompt-spec> <extra-args...>
  local lbl="$1"; shift; local bl="$1"; shift; local hs="$1"; shift; local pr="$1"; shift
  local log="/tmp/k3bg_$lbl.log"
  GPU_LEASE_TIMEOUT=7200 "$LEASE" -n 8 "k3bg-$lbl" sg render -c \
    "PLOW_L2_PLACE_DISPATCH=1 nix develop $WT --command $BIN amd-bench --blob $bl/model.pkt \
     --hsaco $hs --checkpoint $CKPT --tp 8 --steps $STEPS --prompt '$pr' $*" \
    > "$log" 2>&1
  # the generated id list, one line per sequence slot
  grep -oE "^  \[[0-9, ]+\]" "$log"
}

echo "=== CHECK A: $B copies of one prompt must give $B identical streams ==="
same=""
for _ in $(seq 1 "$B"); do same="${same:+$same;}$P1"; done
mapfile -t A < <(run "identical" "$BLOB" "$HSACO" "$same" --batched)
if grep -qE '^Error:| ERROR |>>> .*FAIL' /tmp/k3bg_identical.log; then
  echo "  decode command failed — see /tmp/k3bg_identical.log"
  echo ">>> CHECK A: FAIL"; A_OK=1
elif [ "${#A[@]}" -eq 0 ]; then
  echo "  NO STREAMS PARSED — see /tmp/k3bg_identical.log"
  grep -viE "^\[2m|dev shell|lean tool|amd gpu" /tmp/k3bg_identical.log | tail -4
  echo ">>> CHECK A: INCONCLUSIVE"; A_OK=2
elif [ "${#A[@]}" -ne "$B" ]; then
  echo "  expected $B streams, got ${#A[@]} — the engine is not running $B slots"
  echo ">>> CHECK A: FAIL"; A_OK=1
else
  uniq_n=$(printf '%s\n' "${A[@]}" | sort -u | wc -l)
  printf '  %s\n' "${A[@]}"
  if ! printf '%s\n' "${A[@]}" | grep -Eq '[1-9]'; then
    echo ">>> CHECK A: FAIL — every generated token is zero; identical dead streams are not correctness"
    A_OK=1
  elif [ "$uniq_n" -eq 1 ]; then
    echo ">>> CHECK A: PASS ($B identical)"; A_OK=0
  else
    echo ">>> CHECK A: FAIL — $uniq_n distinct streams; a slot is reading another's state"; A_OK=1
  fi
fi

echo
echo "=== CHECK B: $B ragged prompts must give the same streams at a second width ==="
if [ -z "$ALT_BLOB" ] || [ -z "$ALT_HSACO" ]; then
  # Silently reusing $BLOB here would compare the batch path against itself and print PASS.
  echo "  NO ALT BUILD GIVEN (arguments 5 and 6). Check B needs a build at a DIFFERENT batch"
  echo "  width; without one it would compare the batched blob to itself and pass vacuously."
  echo ">>> CHECK B: NOT RUN"
  B_OK=1
else
spec=""; for i in $(seq 0 $((B-1))); do spec="${spec:+$spec;}${PROMPTS[$((i % ${#PROMPTS[@]}))]}"; done
mapfile -t BATCHED < <(run "ragged" "$BLOB" "$HSACO" "$spec" --batched)
# The alt build runs the SAME prompt list. Its width comes from its own blob, so it may produce
# more or fewer streams; only the slots both runs have are comparable, and the prompts line up
# because the list is positional.
mapfile -t ALT < <(run "ragged_alt" "$ALT_BLOB" "$ALT_HSACO" "$spec" --batched)
B_OK=0
n=$(( ${#BATCHED[@]} < ${#ALT[@]} ? ${#BATCHED[@]} : ${#ALT[@]} ))
if grep -qE '^Error:| ERROR |>>> .*FAIL' /tmp/k3bg_ragged.log /tmp/k3bg_ragged_alt.log; then
  echo "  a decode command failed — see /tmp/k3bg_ragged{,_alt}.log"; B_OK=1
elif [ "${#BATCHED[@]}" -ne "$B" ]; then
  echo "  expected $B streams from the primary build, got ${#BATCHED[@]}"; B_OK=1
elif [ "$n" -lt 2 ]; then
  echo "  alt build produced $n comparable streams — nothing to compare"; B_OK=1
elif ! printf '%s\n' "${BATCHED[@]}" "${ALT[@]}" | grep -Eq '[1-9]'; then
  echo "  every generated token is zero — cross-width agreement is vacuous"; B_OK=1
else
  echo "  comparing $n slots ($B-wide vs ${#ALT[@]}-wide)"
  for i in $(seq 0 $((n-1))); do
    if [ "${BATCHED[$i]}" = "${ALT[$i]}" ]; then echo "  slot $i: MATCHES the other width"
    else echo "  slot $i: DIFFERS across widths"; echo "     w=$B   ${BATCHED[$i]:0:70}"; echo "     w=${#ALT[@]} ${ALT[$i]:0:70}"; B_OK=1; fi
  done
fi
fi
[ "$B_OK" -eq 0 ] && echo ">>> CHECK B: PASS" || echo ">>> CHECK B: FAIL — per-slot position/kvlen handling differs with batch width"

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
