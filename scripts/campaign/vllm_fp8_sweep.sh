#!/usr/bin/env bash
# vLLM 0.28 half of the FP8 ladder: ctx x concurrency x prefix, one leased server per
# (model, prefix, profile), each under the CPU-quiet lock INSIDE the lease. Run OUTSIDE nix
# (vLLM's engine core needs the system ninja). The hub checkpoints are llm-compressor
# FP8_DYNAMIC (per-channel weights, per-token activations, bf16 lm_head) — plow's PTPC numerics.
#
#   OUT      result root (default /opt/dlami/nvme/tmp/fp8-r3/vllm)
#   MODELS   subset of "12b 26b"   PREFIXES  subset of "0 50" (PREFIX_PCT; >0 serves WITH caching)
#   PROFILES subset of "rt hc"     IN_LENS   default "128 1024 4096 8192 15000"
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-/opt/dlami/nvme/tmp/fp8-r3/vllm}"
LEASE="${GPULEASE:-/home/lava/plow/perf-data/tools/gpulease}"
HUB=/opt/dlami/nvme/hf-cache/hub
export GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-43200}" PATH="/opt/pytorch/bin:$PATH" HF_HOME=/opt/dlami/nvme/hf-cache
export IN_LENS="${IN_LENS:-128 1024 4096 8192 15000}" OUTLEN=128
export GATE_PROMPT=$'<bos><start_of_turn>user\nWhat is the capital of France? Answer in one short sentence.<end_of_turn>\n<start_of_turn>model\n'
export BENCH_EXTRA_ARGS="--num-warmups 2 --seed 42"
for m in ${MODELS:-12b 26b}; do
  # The 26B FP8 hub dir ships no tokenizer.json (vLLM then tokenizes the gate prompt to 2 tokens
  # and generates nothing): FP8_26B is a symlink farm of it plus the BF16 26B tokenizer.json.
  case $m in 12b) dir=$HUB/gemma-4-12b-it-fp8 ;;
             26b) dir=${FP8_26B:-/opt/dlami/nvme/tmp/fp8-r3/models/gemma-4-26b-a4b-it-fp8} ;; esac
  for p in ${PREFIXES:-0 50}; do
    for prof in ${PROFILES:-rt hc}; do
      case $prof in rt) concs="1 4"; np=32 ;; hc) concs="16 32"; np=64 ;; esac
      o="$OUT/$m-p$p-$prof"; mkdir -p "$o"
      pc=""; [ "$p" != 0 ] && pc=1
      env CONCS="$concs" NPROMPT=$np PREFIX_PCT=$p PREFIX_CACHE=$pc OUTDIR="$o" \
        "$LEASE" -n 1 "vllm-fp8-$m-p$p-$prof" \
        "$WT/scripts/bench/quietx.sh" /tmp/plow-cpu-quiet.lock \
        "$WT/scripts/bench_vllm_h100.sh" "$dir" 8611 1800 > "$o/results.raw" 2> "$o/stderr.log"
      grep -E '^[0-9]+,' "$o/results.raw" > "$o/rows.csv" || true
      echo "$(date -u +%FT%TZ) $m p$p $prof rows=$(grep -c ',' "$o/rows.csv" 2>/dev/null || echo 0)"
    done
  done
done
