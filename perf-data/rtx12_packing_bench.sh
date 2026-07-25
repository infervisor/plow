#!/usr/bin/env bash
# =============================================================================
# rtx12_packing_bench.sh — measure how the SHIPPED plow mux packs concurrent
# requests (RTX-12 packing baseline). ONE resident plow server (B=8, since the
# B=16 ctx8k blob does not co-fit with the foreign 33 GB server), PLOW_PF_BATCH=1
# + PLOW_PF_PACKLOG=1. For each workload cell it brackets the server log by line
# count so the PACKLOG lines and cumulative WALL lines emitted DURING that cell
# can be sliced out and turned into an R-per-launch histogram + prefill/decode
# wall-time split. Run the whole thing under ONE gpu bench lock hold (a resident
# VRAM server cannot release the lock between cells).
#
# Usage (wrap in flock): MODE=1|2 bash perf-data/rtx12_packing_bench.sh
#   MODE=1 → uniform-512 + uniform-4k closed-loop VU sweep
#   MODE=2 → mixed closed-loop VU sweep + bursty open-loop rate sweep
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN=/root/plow/target/release/plowrt          # built to the SHARED target dir
ASSETS="${ASSETS:-/root/gpu-assets-px1s2b8}"   # 12B ctx8k B=8, shipped varlen pf cubin
PORT="${PORT:-8097}"                           # NOT 8091 (foreign server)
MODEL_NAME=gemma-4-12b-it
TOKENIZER=google/gemma-4-12B-it
DUR="${DUR:-90s}"
WARM="${WARM:-15s}"
MODE="${MODE:-1}"

OUTDIR="$ROOT/perf-data/harness/rtx12/mode$MODE"
SRVLOG="$OUTDIR/server.log"
BRACKETS="$OUTDIR/brackets.tsv"                # cell -> log line range + meta
mkdir -p "$OUTDIR"
: > "$BRACKETS"
cd "$OUTDIR"

# ib tool + its libssl live in the SHARED checkout's target dir (built there).
TOOLS=/root/plow/target/tools
mkdir -p "$TOOLS/lib"
ln -sf /usr/lib/x86_64-linux-gnu/libssl.so.3 /usr/lib/x86_64-linux-gnu/libcrypto.so.3 \
   "$TOOLS/lib/" 2>/dev/null || true
IBBIN="$TOOLS/bin/inference-benchmarker"

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
SRVPID=""
cleanup() {
  if [ -n "$SRVPID" ]; then
    echo ">>> shutting down plow server (pid $SRVPID)"
    kill -TERM "$SRVPID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 2; done
    kill -KILL "$SRVPID" 2>/dev/null
  fi
  for _ in $(seq 1 30); do
    u=$(vram_used); echo ">>> VRAM used: ${u} MiB"
    [ "${u:-99999}" -lt 34000 ] && break   # back down to ~foreign-only
    sleep 3
  done
}
trap cleanup EXIT

# ---- serve (PLOW_PF_BATCH=1 + PLOW_PF_PACKLOG=1) ----------------------------
echo ">>> starting plow server on :$PORT (B=8, packlog on)"
NO_COLOR=1 RUST_LOG=info PLOW_PF_BATCH=1 PLOW_PF_PACKLOG=1 \
  "$BIN" serve --assets "$ASSETS" --port "$PORT" > "$SRVLOG" 2>&1 &
SRVPID=$!
for i in $(seq 1 900); do
  grep -q "serving OpenAI API over TCP" "$SRVLOG" && break
  kill -0 "$SRVPID" 2>/dev/null || { echo "plowrt died"; tail -40 "$SRVLOG"; exit 1; }
  sleep 1
done
echo ">>> plow serving on :$PORT (pid $SRVPID)"
echo ">>> VRAM after load: $(vram_used) MiB"

# Correctness gate before any measurement.
GATE=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0}")
echo ">>> gate: $GATE"
echo "$GATE" | grep -qi paris || { echo "GATE FAILED"; exit 1; }

run_cell() {
  # run_cell <label> <kind> <load-arg> <prompt-opts>
  local label="$1" kind="$2" loadarg="$3" popts="$4"
  local l0 l1
  l0=$(wc -l < "$SRVLOG")
  echo ">>> CELL $label  kind=$kind $loadarg  prompt=[$popts]"
  local args=( --tokenizer-name "$TOKENIZER" --model-name "$MODEL_NAME"
    --url "http://127.0.0.1:$PORT" --no-console --warmup "$WARM" --duration "$DUR"
    --dataset-file github_code.json
    --prompt-options "$popts"
    --decode-options "num_tokens=128,min_tokens=128,max_tokens=128,variance=0"
    --benchmark-kind "$kind"
    --extra-meta "campaign=RTX12-packing,tag=$label"
    --run-id "RTX12-$label" )
  # shellcheck disable=SC2206
  args+=( $loadarg )
  LD_LIBRARY_PATH="$TOOLS/lib" "$IBBIN" "${args[@]}" || echo ">>> $label FAILED"
  l1=$(wc -l < "$SRVLOG")
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$l0" "$l1" "$kind" "$popts" >> "$BRACKETS"
  echo ">>> CELL $label done (log lines $l0..$l1)"
}

U512="num_tokens=512,min_tokens=512,max_tokens=512,variance=0"
U4K="num_tokens=4000,min_tokens=4000,max_tokens=4000,variance=0"
# mixed: wide spread spanning short..long (realized lengths reported from PACKLOG
# chunks). Not a perfect 60/40 bimodal — an honest heterogeneous proxy.
MIX="num_tokens=1200,min_tokens=256,max_tokens=4096,variance=2200000"

if [ "$MODE" = "1" ]; then
  for vu in 1 4 8 16 32; do run_cell "u512-vu$vu" throughput "--max-vus $vu" "$U512"; done
  for vu in 1 4 8 16 32; do run_cell "u4k-vu$vu"  throughput "--max-vus $vu" "$U4K";  done
elif [ "$MODE" = "2" ]; then
  for vu in 1 4 8 16 32; do run_cell "mix-vu$vu"  throughput "--max-vus $vu" "$MIX"; done
  # bursty = open-loop ConstantArrivalRate (poisson arrivals -> bursts), cap B=8.
  for r in 2 4 8; do run_cell "burst4k-r$r"  rate "--rates $r --max-vus 8" "$U4K"; done
  for r in 4 8;   do run_cell "burstmix-r$r" rate "--rates $r --max-vus 8" "$MIX"; done
fi

echo ">>> MODE $MODE done; server.log=$SRVLOG brackets=$BRACKETS"
