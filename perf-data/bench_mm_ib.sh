#!/usr/bin/env bash
# =============================================================================
# bench_mm_ib.sh — campaign MM1-ib: multi-model co-resident serving capacity.
# =============================================================================
# One `plowrt serve` process, TWO models resident (12B @132k b1 + 26B-A4B
# @132k b1 — the measured-to-fit pair), concurrent load on BOTH through the
# pinned inference-benchmarker (one ib process per model slug, same URL, same
# profile as the single-model B2 rows: 4k prompt / 128 out, greedy).
#
#   GPU_LEASE_TIMEOUT=7200 gpulease final-mm bash perf-data/bench_mm_ib.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/target/release/plowrt"
A12="${A12:-/root/gpu-assets-s6/b1}"
A26="${A26:-/root/gpu-assets-26b/b1}"
PORT="${PORT:-8098}"
VUS="${VUS:-1 2}"          # per model, simultaneously
DUR="${DUR:-120s}"
WARM="${WARM:-15s}"
PROMPT_TOKS="${PROMPT_TOKS:-4000}"
CAMPAIGN=MM1-ib

OUTBASE="$ROOT/perf-data/harness/b2-ib"
SRVLOG="$OUTBASE/mm-server.log"
mkdir -p "$OUTBASE/mm-12b/results" "$OUTBASE/mm-26b/results"

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
SRVPID=""
cleanup() {
  [ -n "$SRVPID" ] && { kill -TERM "$SRVPID" 2>/dev/null; sleep 5; kill -KILL "$SRVPID" 2>/dev/null; }
  for _ in $(seq 1 30); do u=$(vram_used); echo ">>> VRAM used: ${u} MiB"; [ "${u:-99999}" -lt 1000 ] && break; sleep 3; done
}
trap cleanup EXIT

NO_COLOR=1 RUST_LOG=info,plowrt=debug "$BIN" serve \
  --assets "$A12" --assets "$A26" --port "$PORT" > "$SRVLOG" 2>&1 &
SRVPID=$!
for i in $(seq 1 900); do
  grep -q "serving OpenAI API over TCP" "$SRVLOG" && break
  kill -0 "$SRVPID" 2>/dev/null || { echo "plowrt died"; tail -40 "$SRVLOG"; exit 1; }
  sleep 1
done
echo ">>> co-resident server up; VRAM $(vram_used) MiB"

for NAME in gemma-4-12b-it gemma-4-26b-a4b-it; do
  G=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0}")
  echo ">>> gate[$NAME]: $G"
  echo "$G" | grep -qi paris || { echo "GATE FAILED $NAME"; exit 1; }
done
echo ">>> VRAM with both resident: $(vram_used) MiB"

ENGINE_COMMIT=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)

run_ib() {  # run_ib <tag> <model-name> <tokenizer> <vu>
  local TAG=$1 NAME=$2 TOK=$3 N=$4
  ( cd "$OUTBASE/$TAG" && "$ROOT/perf-data/bench_ib.sh" \
      --tokenizer-name "$TOK" --model-name "$NAME" \
      --url "http://127.0.0.1:$PORT" --no-console \
      --warmup "$WARM" --duration "$DUR" \
      --dataset-file github_code.json \
      --prompt-options "num_tokens=$PROMPT_TOKS,min_tokens=$PROMPT_TOKS,max_tokens=$PROMPT_TOKS,variance=0" \
      --decode-options num_tokens=128,min_tokens=128,max_tokens=128,variance=0 \
      --benchmark-kind throughput --max-vus "$N" \
      --extra-meta "campaign=$CAMPAIGN,engine=plow,tag=$TAG,engine_commit=$ENGINE_COMMIT,point=vu$N,coresident=12b+26b" \
      --run-id "$CAMPAIGN-$TAG-vu$N" )
}

for N in $VUS; do
  echo ">>> MM point: vu$N per model, simultaneous"
  run_ib mm-12b gemma-4-12b-it google/gemma-4-12B-it "$N" &
  P1=$!
  run_ib mm-26b gemma-4-26b-a4b-it google/gemma-4-26B-A4B-it "$N" &
  P2=$!
  wait $P1; R1=$?; wait $P2; R2=$?
  echo ">>> MM vu$N done (12b=$R1 26b=$R2)"
done
echo ">>> MM done"
