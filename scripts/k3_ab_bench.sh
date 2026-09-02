#!/usr/bin/env bash
# INTERLEAVED, ORDER-REVERSED production-scheduler A/B.                        [MEASUREMENT-GATE]
#
#   ./scripts/k3_ab_bench.sh <A-label> <A-assets> <B-label> <B-assets> [reps] [batch-prompts]
#
# WHY THIS EXISTS, and it is not a convenience.
#
# Every B=16 comparison in this campaign was ONE run against ONE baseline, and that resolves
# nothing. MEASURED on this box, same blob, same object, same command, four consecutive runs:
#
#     177.207   178.579   182.085   260.533   ms/step
#
# Excluding the outlier: mean 179.29, sd 2.52 -- so per-run noise is ~+/-1.4% (1 sd), roughly ONE
# RUN IN FOUR is a +45% outlier, and the SAME code measured 167-169 in one session and 177-182 in
# the next (a 6-8% inter-session drift). A single-run A/B at that noise level cannot see anything
# under ~10%, which is larger than almost every optimisation worth making.
#
# It also produced a WRONG CONCLUSION, which is the real argument for this file. Two MoE
# align-scan variants measured 192.5 and 211.8 ms and were confidently attributed to barrier cost
# -- until reverting the change entirely measured 182.2 against the 167.2 the identical code had
# produced earlier. The experiment could not distinguish its own effect from the drift, and an
# explanation was invented to fit a number that was not real.
#
# WHAT THIS DOES ABOUT IT
#
#   * INTERLEAVES the two arms (A B B A A B B A ...) rather than running all of A then all of B,
#     so a monotonic drift lands on both arms equally instead of on whichever ran second.
#   * REVERSES the order every pair, which is what cancels a first-in-pair bias.
#   * Reports MEDIAN, not mean: one +45% outlier moves a 4-run mean by 11% and the median by 0.
#   * Reports the observed spread of EACH arm and refuses a verdict when the arms overlap.
#
# `<assets>` is a directory with `model.pkt` and `hsaco/`. The two arms may differ in either --
# a rebuilt object, a re-emitted blob, or both -- which is what makes this usable for a kernel
# change, an emitter change, or a knob.
#
#   REPS     pairs per arm (default 4, so 8 runs per arm)
#   STEPS    decode steps per run (default 64; below ~32 the per-step average is itself noisy)
#   CKPT     optional production checkpoint override (otherwise assets/checkpoint)
#   BATCHED  1 for concurrent serving (default 1); CONCURRENCY defaults to 16
#   SETTLE_C wait for the hottest GPU to fall to this junction temperature before each run
#            (default 0 = off). SETTLE_MAX caps the wait in seconds (default 180).
#
# THE THERMAL CARRYOVER, which is why SETTLE_C exists.
#
# `gpulease` gives EXCLUSIVITY, not a consistent starting state. This box is shared, and the lease
# log shows jobs running back to back:
#
#   11:14:41  slabab  ACQUIRED ... 11:18:17 slabab RELEASED  held=216s
#   11:18:18  k3rep   ACQUIRED                              <- 1 second later
#
# `k3rep` is the run that produced 177 / 179 / 182 / 260 ms, and it began on GPUs that had just
# been driven hard for 216 s by somebody else's job. Junction temperature and power were MEASURED
# at 43-51 C and 311-322 W while another session's job held the lease, so "idle" on this box is
# not idle. A measurement that does not report its starting temperature cannot be compared with
# one taken at a different time, which is exactly the inter-session drift 8.6a documents.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AL="${1:?A label}"; AA="${2:?A assets}"; BL="${3:?B label}"; BA="${4:?B assets}"
REPS="${5:-4}"
STEPS="${STEPS:-64}"; CKPT="${CKPT:-}"; BATCHED="${BATCHED:-1}"
CONCURRENCY="${CONCURRENCY:-16}"
REPORT_DIR="${REPORT_DIR:-/tmp/k3-ab-reports}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
P="1008,10484,318,15383,387"

[ -x "$BIN" ] || { echo "no plowrt at $BIN — cargo build --release -p plowrt --features hsa"; exit 1; }

[ "$BATCHED" = "1" ] || CONCURRENCY=1
mkdir -p "$REPORT_DIR"

hot_c() { # hottest junction temperature across all GPUs, or empty if rocm-smi is unavailable
  rocm-smi --showtemp 2>/dev/null \
    | grep -oE "junction\) \(C\): [0-9.]+" | grep -oE "[0-9.]+$" \
    | sort -g | tail -1
}

settle() { # wait for the box to cool, if asked
  [ "${SETTLE_C:-0}" = "0" ] && return 0
  local t0=$SECONDS
  while :; do
    local c; c=$(hot_c)
    [ -z "$c" ] && return 0
    awk -v a="$c" -v b="$SETTLE_C" 'BEGIN{exit !(a<=b)}' && return 0
    [ $((SECONDS - t0)) -ge "${SETTLE_MAX:-180}" ] && {
      echo "    (settle gave up at ${c}C after $((SECONDS-t0))s)"; return 0; }
    sleep 5
  done
}

run_one() { # <assets> -> production p50 TPOT on stdout
  local a="$1"
  local out
  local env_args=(PLOW_L2_PLACE_DISPATCH=1)
  [ -n "$CKPT" ] && env_args+=("PLOW_CHECKPOINT=$CKPT")
  out="$(mktemp "$REPORT_DIR/bench.XXXXXX.json")"
  nix develop "$WT" --command env "${env_args[@]}" "$BIN" bench \
    --assets "$a" --prompt-ids "$P" \
    --concurrency "$CONCURRENCY" --requests "$CONCURRENCY" \
    --warmup-requests "$CONCURRENCY" --output-len "$STEPS" \
    --max-hold-ms 0 --slo-ms "${SLO_MS:-60000}" >"$out" || return 1
  python3 - "$out" "$CONCURRENCY" "$STEPS" <<'PY'
import json, sys
p, concurrency, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(p) as f:
    r = json.load(f)
assert r["schema"] == "plowrt.bench.v1"
assert r["completed"] == concurrency and r["failed"] == 0
assert r["output_tokens"] == concurrency * steps
assert r["scheduler"]["rejected"] == 0 and r["scheduler"]["admit_shed"] == 0
print(r["tpot_ms"]["p50"])
PY
}

echo "A = $AL  ($AA)"
echo "B = $BL  ($BA)"
echo "reps=$REPS steps=$STEPS concurrency=$CONCURRENCY  -> $((REPS*2)) runs per arm, interleaved and order-reversed"
echo

AV=(); BV=()
for r in $(seq 1 "$REPS"); do
  # ORDER REVERSAL: odd pairs run A then B, even pairs run B then A.
  if [ $((r % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
  for w in $order; do
    settle
    tc=$(hot_c)
    if [ "$w" = "A" ]; then v=$(run_one "$AA"); AV+=("$v"); else v=$(run_one "$BA"); BV+=("$v"); fi
    printf "  pair %d  %s  %s ms p50 TPOT   (start %sC)\n" "$r" "$w" "${v:-FAIL}" "${tc:-?}"
  done
done

echo
python3 - "$AL" "$BL" "${AV[*]}" "${BV[*]}" <<'PY'
import statistics, sys
al, bl, a_s, b_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
A = [float(x) for x in a_s.split() if x]
B = [float(x) for x in b_s.split() if x]
if len(A) < 2 or len(B) < 2:
    print("NOT ENOUGH SAMPLES — a run failed; see the lines above"); raise SystemExit(1)

def stat(v):
    med = statistics.median(v)
    return med, min(v), max(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

am, alo, ahi, asd = stat(A)
bm, blo, bhi, bsd = stat(B)
print(f"{al:>24s}  median {am:8.3f}  min {alo:8.3f}  max {ahi:8.3f}  sd {asd:6.3f}  n={len(A)}")
print(f"{bl:>24s}  median {bm:8.3f}  min {blo:8.3f}  max {bhi:8.3f}  sd {bsd:6.3f}  n={len(B)}")
d = 100.0 * (bm - am) / am
print(f"\n  B vs A on the median: {d:+.2f}%")

# A verdict only where the arms do not overlap. Ranges rather than a t-test on purpose: the
# outlier distribution here is not normal (one run in four at +45%), so a parametric interval
# would be confidently wrong.
if bhi < alo:
    print(f"  VERDICT: {bl} is FASTER — its worst run beats {al}'s best.")
elif blo > ahi:
    print(f"  VERDICT: {bl} is SLOWER — its best run loses to {al}'s worst.")
else:
    ov = min(ahi, bhi) - max(alo, blo)
    print(f"  VERDICT: NOT RESOLVED — the ranges overlap by {ov:.3f} ms.")
    print(f"           At this spread the experiment cannot see a {abs(d):.1f}% effect.")
    print(f"           Raise REPS, or the effect is smaller than the instrument.")
PY
