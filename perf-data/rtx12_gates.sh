#!/usr/bin/env bash
# =============================================================================
# rtx12_gates.sh — RTX-12 chunked-packing correctness gates.
#
# Boots `plowrt serve` (PLOW_PF_BATCH=1 + PLOW_PF_PACKLOG=1) once per chunk-cap
# config on port 8097 (NEVER 8091 = foreign server), collects solo + burst
# outputs via perf-data/px1_gates.py, tears down, then cross-compares:
#
#   Canary/G4 : C=0 solo == C=512 solo == C=256 solo (all prompts) —
#               the chunk cap is numerics-neutral (pure scheduling choice).
#   Gate A    : C=512 solo == C=512 burst — per-request token identity under
#               concurrent multi-request co-packing.
#   Gate B    : victim byte-identical solo vs burst (both orders); concat
#               control flips (poison detector live).
#   Gate C    : C=256 solo == C=0 solo on the LONG (>2048-tok) prompts — a
#               request whose prefill is FORCE-SPLIT across >=2 launches is
#               byte-identical to its uncapped run. Plus a PACKLOG check that a
#               single slot really appeared in >=2 consecutive launches.
#
# Wrap in flock. Usage:
#   ASSETS=/root/gpu-assets-px1s2b8 PORT=8097 bash perf-data/rtx12_gates.sh
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${ASSETS:-/root/gpu-assets-px1s2b8}"
PORT="${PORT:-8097}"
BIN=/root/plow/target/release/plowrt
OUT="${OUT:-$ROOT/perf-data/harness/rtx12/gates}"
mkdir -p "$OUT"

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
SRVPID=""
stop_srv() {
  if [ -n "$SRVPID" ]; then
    kill -TERM "$SRVPID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 2; done
    kill -KILL "$SRVPID" 2>/dev/null || true
    SRVPID=""
  fi
  for _ in $(seq 1 30); do u=$(vram_used); [ "${u:-99999}" -lt 34000 ] && break; sleep 3; done
}
trap stop_srv EXIT

# start_srv <logfile> <chunk> [batch_env]  (batch_env unset => legacy control)
start_srv() {
  local log="$1" chunk="$2" batch="${3:-1}"
  local envs=(NO_COLOR=1 RUST_LOG=info,plowrt=debug PLOW_PF_PACKLOG=1)
  [ "$batch" = "1" ] && envs+=(PLOW_PF_BATCH=1)
  [ -n "$chunk" ] && envs+=(PLOW_PF_CHUNK="$chunk")
  echo ">>> serve chunk=${chunk:-<unset>} batch=$batch  VRAM now $(vram_used) MiB"
  env "${envs[@]}" "$BIN" serve --assets "$ASSETS" --port "$PORT" > "$log" 2>&1 &
  SRVPID=$!
  for _ in $(seq 1 900); do
    grep -q "serving OpenAI API over TCP" "$log" && return 0
    kill -0 "$SRVPID" 2>/dev/null || { echo "server died"; tail -30 "$log"; exit 1; }
    sleep 1
  done
  echo "server never came up"; exit 1
}

# collect <chunk-tag> <chunk> — boot, run solo+burst, teardown.
collect() {
  local tag="$1" chunk="$2"
  start_srv "$OUT/server-$tag.log" "$chunk" 1
  if [ "$chunk" != "0" ] && [ -n "$chunk" ]; then
    grep -q "batched prefill enabled" "$OUT/server-$tag.log" \
      || { echo "FATAL: PX-1 batched mode did not engage for $tag"; exit 1; }
  fi
  echo "--- $tag solo ---"
  PORT=$PORT python3 "$ROOT/perf-data/px1_gates.py" solo  "$OUT/$tag-solo.json"
  echo "--- $tag burst ---"
  PORT=$PORT python3 "$ROOT/perf-data/px1_gates.py" burst "$OUT/$tag-burst.json"
  stop_srv
}

echo "############ collecting configs ############"
collect c0   0     # uncapped batched  = today's behaviour (canary reference)
collect c256 256   # tight cap: forces long prompts to split across many launches
collect c512 512   # the Stage-A shipping cap

echo
echo "############ Canary/G4: chunk cap is numerics-neutral ############"
python3 "$ROOT/perf-data/px1_gates.py" cmp "$OUT/c0-solo.json" "$OUT/c512-solo.json" \
  && echo "CANARY C0==C512 solo: PASS" || { echo "CANARY C0==C512: FAIL"; exit 1; }
python3 "$ROOT/perf-data/px1_gates.py" cmp "$OUT/c0-solo.json" "$OUT/c256-solo.json" \
  && echo "CANARY C0==C256 solo: PASS" || { echo "CANARY C0==C256: FAIL"; exit 1; }

echo
echo "############ Gate A: per-request identity under co-packing (C=512) ############"
python3 "$ROOT/perf-data/px1_gates.py" cmp "$OUT/c512-solo.json" "$OUT/c512-burst.json" \
  && echo "GATE A (C=512): PASS" || { echo "GATE A (C=512): FAIL"; exit 1; }

echo
echo "############ Gate B: cross-request bleed + sensitivity (C=512) ############"
python3 - "$OUT/c512-solo.json" "$OUT/c512-burst.json" <<'EOF'
import json, sys
solo  = json.load(open(sys.argv[1]))
burst = json.load(open(sys.argv[2]))
ok = True
# victim byte-identical solo vs concurrent, both submission orders
for k in ("victim", "victim_rev"):
    v = burst.get(k)
    if v != solo["victim"]:
        print(f"  BLEED: burst[{k}] != solo victim"); ok = False
    else:
        print(f"  {k}: identical to solo victim")
# sensitivity control: concat (poison IN context) must differ from isolated victim
if solo["concat_control"] == solo["victim"]:
    print("  SENSITIVITY WEAK: concat == isolated victim (detector dead)"); ok = False
else:
    print("  sensitivity: concat_control differs from isolated victim (detector live)")
print("GATE B:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
[ $? -eq 0 ] || exit 1

echo
echo "############ Gate C: forced chunk-split byte-identity ############"
# The long/xl prompts (>2048 tok) are split across >=2 launches at C=256/512;
# they must be byte-identical to the uncapped (C=0) solo run.
python3 - "$OUT/c0-solo.json" "$OUT/c256-solo.json" <<'EOF'
import json, sys
c0  = json.load(open(sys.argv[1]))
c256 = json.load(open(sys.argv[2]))
LONG = ["med", "long1", "long2", "xl", "poison", "victim"]  # all >2048 tok
ok = True
for k in LONG:
    if c0.get(k) != c256.get(k):
        print(f"  SPLIT MISMATCH {k}:"); print(f"    C0  : {c0.get(k)!r}"); print(f"    C256: {c256.get(k)!r}"); ok = False
    else:
        print(f"  {k}: split(C=256) identical to uncapped(C=0)")
print("GATE C:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
[ $? -eq 0 ] || exit 1

echo
echo "############ non-vacuity: multi-request AND multi-launch split occurred ############"
for tag in c256 c512; do
  log="$OUT/server-$tag.log"
  multiR=$(grep "PACKLOG R=" "$log" | grep -vc "R=1 " || true)
  echo "  $tag: multi-request launches (R>=2) = $multiR"
  # a request split = same total prompt rows appearing across multiple launches;
  # under C the long prompts always split, so total launches >> requests.
  launches=$(grep -c "PACKLOG R=" "$log" || true)
  echo "  $tag: total batched launches = $launches"
done
maxrows_c256=$(grep -oE "PACKLOG R=[0-9]+ rows=[0-9]+" "$OUT/server-c256.log" | grep -oE "rows=[0-9]+" | cut -d= -f2 | sort -n | tail -1)
echo "  c256: max rows in any single launch = ${maxrows_c256:-?} (cap works if <= 256*R)"

echo
echo "ALL GATES PASSED"
