#!/usr/bin/env bash
# =============================================================================
# rtx12_chunked_bench.sh — RTX-12 chunked-packing perf campaign (Stage A/B).
#
# For each (PLOW_PF_CHUNK C, PLOW_PF_INTERLEAVE Q) config it boots ONE resident
# plow server (B=8, PLOW_PF_BATCH=1 + PLOW_PF_PACKLOG=1) on port 8097 (NEVER
# 8091 = foreign server), runs the 4k-uniform VU sweep {1,4,8,16,32} (the
# target workload), and — for the OFF and primary ON config — the 512-uniform
# sweep too (regression check). One mode-dir per config holds its server.log +
# brackets.tsv + results/, so rtx12_analyze.py builds the R-histogram +
# tok/s/ITL/TTFT per cell. Run the WHOLE script under ONE flock hold (a resident
# VRAM server cannot release the lock between cells); server torn down per config.
#
# Wrap in flock. Usage: bash perf-data/rtx12_chunked_bench.sh
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN=/root/plow/target/release/plowrt
ASSETS="${ASSETS:-/root/gpu-assets-px1s2b8}"
PORT="${PORT:-8097}"
MODEL_NAME=gemma-4-12b-it
TOKENIZER=google/gemma-4-12B-it
DUR="${DUR:-90s}"
WARM="${WARM:-15s}"
IBBIN=/root/plow/target/tools/bin/inference-benchmarker
TOOLS=/root/plow/target/tools
BASEDIR="$ROOT/perf-data/harness/rtx12/chunked"
mkdir -p "$BASEDIR" "$TOOLS/lib"
ln -sf /usr/lib/x86_64-linux-gnu/libssl.so.3 /usr/lib/x86_64-linux-gnu/libcrypto.so.3 "$TOOLS/lib/" 2>/dev/null || true

U512="num_tokens=512,min_tokens=512,max_tokens=512,variance=0"
U4K="num_tokens=4000,min_tokens=4000,max_tokens=4000,variance=0"

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
SRVPID=""
SRVLOG=""
BRACKETS=""
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

# boot_srv <mode-dir> <C> <Q>
boot_srv() {
  local dir="$1" C="$2" Q="$3"
  SRVLOG="$dir/server.log"; BRACKETS="$dir/brackets.tsv"
  : > "$BRACKETS"
  echo ">>> boot C=$C Q=$Q  VRAM now $(vram_used) MiB"
  NO_COLOR=1 RUST_LOG=info PLOW_PF_BATCH=1 PLOW_PF_PACKLOG=1 PLOW_PF_CHUNK="$C" PLOW_PF_INTERLEAVE="$Q" \
    "$BIN" serve --assets "$ASSETS" --port "$PORT" > "$SRVLOG" 2>&1 &
  SRVPID=$!
  for _ in $(seq 1 900); do
    grep -q "serving OpenAI API over TCP" "$SRVLOG" && break
    kill -0 "$SRVPID" 2>/dev/null || { echo "plowrt died"; tail -40 "$SRVLOG"; exit 1; }
    sleep 1
  done
  echo ">>> serving :$PORT (pid $SRVPID), VRAM $(vram_used) MiB"
  local G
  G=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0}")
  echo "$G" | grep -qi paris || { echo "GATE FAILED: $G"; exit 1; }
}

# run_cell <mode-dir> <label> <popts>
run_cell() {
  local dir="$1" label="$2" popts="$3" vu="$4"
  local l0 l1
  ( cd "$dir"
    l0=$(wc -l < "$SRVLOG")
    echo ">>> CELL $label vu=$vu"
    LD_LIBRARY_PATH="$TOOLS/lib" "$IBBIN" \
      --tokenizer-name "$TOKENIZER" --model-name "$MODEL_NAME" \
      --url "http://127.0.0.1:$PORT" --no-console --warmup "$WARM" --duration "$DUR" \
      --dataset-file github_code.json --prompt-options "$popts" \
      --decode-options "num_tokens=128,min_tokens=128,max_tokens=128,variance=0" \
      --benchmark-kind throughput --max-vus "$vu" \
      --extra-meta "campaign=RTX12-chunked,tag=$label" --run-id "RTX12-$label" \
      || echo ">>> $label FAILED"
    l1=$(wc -l < "$SRVLOG")
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$l0" "$l1" throughput "$popts" >> "$BRACKETS"
    echo ">>> CELL $label done (lines $l0..$l1)"
  )
}

# config <tag> <C> <Q> <do_u512>
config() {
  local tag="$1" C="$2" Q="$3" do512="$4"
  local dir="$BASEDIR/$tag"
  mkdir -p "$dir/results"
  boot_srv "$dir" "$C" "$Q"
  for vu in 1 4 8 16 32; do run_cell "$dir" "$tag-u4k-vu$vu" "$U4K" "$vu"; done
  if [ "$do512" = "1" ]; then
    for vu in 1 4 8 16 32; do run_cell "$dir" "$tag-u512-vu$vu" "$U512" "$vu"; done
  fi
  stop_srv
  echo ">>> config $tag done"
}

# ---- the matrix ----
# OFF baseline (this binary; C=0 == today's behaviour) + short-prompt regression
config off        0    2048 1
# Stage A: per-request chunk cap at the default Q=2048 (R~4 at C=512, R~2 at C=1024)
config c512q2048  512  2048 1
config c1024q2048 1024 2048 0
# Stage B: couple C with a smaller interleave quantum Q (R held ~2, smaller stall)
config c512q1024  512  1024 0

echo ">>> ALL CONFIGS DONE"
echo ">>> analyzing..."
OUTBASE=rtx12-chunked-packing python3 "$ROOT/perf-data/rtx12_analyze.py" \
  "$BASEDIR/off" "$BASEDIR/c512q2048" "$BASEDIR/c1024q2048" "$BASEDIR/c512q1024"
echo ">>> DONE"
