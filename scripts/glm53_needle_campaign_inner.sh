#!/usr/bin/env bash
# INSIDE-nix, INSIDE-lease half of the follow-up pair. See glm53_needle_campaign.sh.
#
# Two things the first campaign could not cover:
#
#  * NEEDLE retrieval for the bf16/fp8 latent pair. Greedy agreement says how far
#    two arms stayed token-identical; it cannot say whether the answer got worse,
#    and fp8 KV's documented failure mode is retrieval degrading with context.
#  * A bf16-vs-fp8 TPOT point ABOVE 32768. The 32768 control cannot reach it and
#    the 131072 bf16 blob does not load at all, so the comparison runs
#    `tp4-dense66k` (bf16, max_ctx 66560) against `tp4-fp8131k` (fp8, max_ctx
#    131072) at the SAME live 65536 — the two blobs differ in max_ctx, which the
#    first campaign showed does not move TPOT (fp8 at live 30000 measured 42.615
#    on the 32768 blob and 42.517 on the 131072 one, 0.2% apart).
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="${PLOW_HSACO_DIR:-/app/plow/build-glm53/hsaco}"
BIN="$WT/target-glm53/release"
RES="$WT/build-glm53/kvbench"
mkdir -p "$RES"
export LD_LIBRARY_PATH="${ROCM_PATH:?nix develop did not set ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PLOW_HSACO="$OBJ" PLOW_MLA_PF_V2=1 PLOW_L2_PLACE_DISPATCH=1 PLOW_TP_NO_AUDIT=1 \
       PLOW_MLA_NS_LIVE=0

PORT=19430
# bundle:needle-lens:long-bench-input  ("-" skips that half)
for spec in ${ARMS:-tp4-ctl:4096,16384,30000:- tp4-fp8:4096,16384,30000:- tp4-dense66k:-:65536 tp4-fp8131k:-:65536}; do
  bundle="${spec%%:*}"; rest="${spec#*:}"
  needle_lens="${rest%%:*}"; long_in="${rest#*:}"
  PORT=$((PORT + 1))
  echo "########## $bundle on $PORT  needle=$needle_lens long=$long_in  $(date -u +%H:%M:%S)"
  "$BIN/plowrt" serve --assets "$WT/build-glm53/$bundle" --port "$PORT" \
    > "$RES/needle-serve-$bundle.log" 2>&1 &
  srv=$!
  ready=0
  for _ in $(seq 1 300); do
    curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$srv" 2>/dev/null || break
    sleep 2
  done
  if [ "$ready" = 1 ]; then
    if [ "$needle_lens" != "-" ]; then
      python3 "$WT/scripts/glm53_needle_probe.py" --url "http://127.0.0.1:$PORT" \
        --arm "$bundle" --out "$RES/needle-$bundle.json" --lens "$needle_lens"
    fi
    if [ "$long_in" != "-" ]; then
      python3 "$WT/scripts/bench_packed_serve.py" --url "http://127.0.0.1:$PORT" \
        --out "$RES/long-$bundle.jsonl" --label "long-$bundle" --inputs "$long_in" \
        --outputs 128 --concurrency 1 --repeats 3 --warmups 1 2>&1 | tail -2
    fi
  else
    echo "!! $bundle never became ready"
    tail -4 "$RES/needle-serve-$bundle.log"
  fi
  kill -TERM "$srv" 2>/dev/null
  for _ in $(seq 1 90); do kill -0 "$srv" 2>/dev/null || break; sleep 2; done
  kill -0 "$srv" 2>/dev/null && kill -KILL "$srv" 2>/dev/null
  wait "$srv" 2>/dev/null
  sleep 5
done
echo "########## follow-up campaign done $(date -u +%H:%M:%S)"
