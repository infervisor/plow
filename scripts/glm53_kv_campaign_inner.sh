#!/usr/bin/env bash
# INSIDE-nix, INSIDE-lease half of the decode-KV campaign. See glm53_kv_campaign.sh.
#
# One arm = serve a bundle, wait for readiness, bench it, take its greedy stream,
# stop it. A bundle that CANNOT load (the bf16 131072 arm is expected not to fit)
# is recorded as a load failure with its refusal text and the sequence continues —
# that refusal is a result, not an error.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="${PLOW_HSACO_DIR:-/app/plow/build-glm53/hsaco}"
BIN="$WT/target-glm53/release"
RES="$WT/build-glm53/kvbench"
mkdir -p "$RES"
export LD_LIBRARY_PATH="${ROCM_PATH:?nix develop did not set ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PLOW_HSACO="$OBJ" PLOW_MLA_PF_V2=1 PLOW_L2_PLACE_DISPATCH=1 PLOW_TP_NO_AUDIT=1

PORT=19410
SRV=""

# `serve_arm <bundle> <extra env assignments...>` -> 0 ready, 1 failed to load.
serve_arm () {
  local bundle="$1"; shift
  PORT=$((PORT + 1))
  echo "=== serve $bundle on $PORT ${*:-}"
  env "$@" "$BIN/plowrt" serve --assets "$bundle" --port "$PORT" \
    > "$RES/serve-$(basename "$bundle").log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 "${READY_TIMEOUT:-600}"); do
    curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "!! server exited during load"; SRV=""; return 1; }
    sleep 2
  done
  echo "!! never became ready"
  return 1
}

stop_arm () {
  [ -n "$SRV" ] || return 0
  kill -TERM "$SRV" 2>/dev/null
  for _ in $(seq 1 90); do kill -0 "$SRV" 2>/dev/null || break; sleep 2; done
  kill -0 "$SRV" 2>/dev/null && kill -KILL "$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null
  SRV=""
  sleep 5
}

bench_arm () {  # <label> <inputs...> ; REPEATS/OUTLEN from env
  local label="$1"; shift
  python3 "$WT/scripts/bench_packed_serve.py" --url "http://127.0.0.1:$PORT" \
    --out "$RES/$label.jsonl" --label "$label" --inputs "$@" \
    --outputs "${OUTLEN:-128}" --concurrency 1 --repeats "${REPEATS:-5}" --warmups 1 \
    2>&1 | tail -3
}

greedy_arm () {  # <label> <lens csv>
  python3 "$WT/scripts/glm53_greedy_probe.py" --url "http://127.0.0.1:$PORT" \
    --arm "$1" --out "$RES/greedy-$1.json" --lens "$2" --max-tokens "${GREEDY_TOK:-128}" \
    2>&1 | tail -2
}

run_ctl ()  { serve_arm "$WT/build-glm53/tp4-ctl"  PLOW_MLA_NS_LIVE=0 || return
              bench_arm ctl 1024 4096 16384 30000; greedy_arm ctl 1024,4096,16384,30000; }
run_fp8 ()  { serve_arm "$WT/build-glm53/tp4-fp8"  PLOW_MLA_NS_LIVE=0 || return
              bench_arm fp8 1024 4096 16384 30000; greedy_arm fp8 1024,4096,16384,30000; }

# The capacity pair. bf16 at 131072 is expected to REFUSE — 78 layers x 1152 B
# per position is 10.97 GiB of KV against ~10.2 GiB free after a 181.75 GiB
# weight slab — and fp8 at the same max_ctx is expected to load at 6.13 GiB.
run_ctl131k () { READY_TIMEOUT=180 serve_arm "$WT/build-glm53/tp4-ctl131k" PLOW_MLA_NS_LIVE=0 \
                 && { echo "ctl131k LOADED (the capacity claim is wrong)"; REPEATS=3 bench_arm ctl131k 30000; } \
                 || { echo "ctl131k did NOT load:"; tail -6 "$RES/serve-tp4-ctl131k.log"; }; }
run_fp8131k () { serve_arm "$WT/build-glm53/tp4-fp8131k" PLOW_MLA_NS_LIVE=0 || return
                 REPEATS=3 bench_arm fp8131k 30000 65536 120000
                 greedy_arm fp8131k 30000,65536; }

# The DSA pair, both at max_ctx 66560 so the emitter arms the gather (its
# crossover is a strict `ctx > 65536`) and the dense control is the same blob
# with the gate off.
run_dense66k () { serve_arm "$WT/build-glm53/tp4-dense66k" PLOW_MLA_NS_LIVE=0 || return
                  REPEATS=3 bench_arm dense66k 4096 8192 16384 32768 60000
                  greedy_arm dense66k 4096,16384,60000; }
run_dsa ()      { serve_arm "$WT/build-glm53/tp4-dsa" PLOW_MLA_NS_LIVE=0 || return
                  REPEATS=3 bench_arm dsa 4096 8192 16384 32768 60000
                  greedy_arm dsa 4096,16384,60000; }

# The split-policy pair, on the SHIPPED long-context blob (max_ctx 32768, decode
# rungs 1 and 2, baked ns=64) so the only difference between the arms is one env
# var. The ON arm's log line is the proof the count actually moved: a patch that
# reaches the wrong decode rung is a silent no-op that reads as a clean null.
run_nslive0 () { serve_arm "/app/plow/build-glm53/tp4-long" PLOW_MLA_NS_LIVE=0 || return
                 bench_arm nslive0b 1024 4096 8192 16384 30000
                 greedy_arm nslive0b 1024,4096,8192,16384,30000; }
run_nslive1 () { serve_arm "/app/plow/build-glm53/tp4-long" PLOW_MLA_NS_LIVE=1 || return
                 bench_arm nslive1b 1024 4096 8192 16384 30000
                 greedy_arm nslive1b 1024,4096,8192,16384,30000
                 echo "split re-points: $(grep -c 're-pointed at the live' "$RES/serve-tp4-long.log")"
                 grep "re-pointed at the live" "$RES/serve-tp4-long.log" | head -8; }

trap 'stop_arm' EXIT
for arm in ${ARMS:-ctl fp8 ctl131k fp8131k nslive0 nslive1 dense66k dsa}; do
  echo "########## $arm  $(date -u +%H:%M:%S)"
  "run_$arm"
  stop_arm
done
echo "########## campaign done $(date -u +%H:%M:%S)"
