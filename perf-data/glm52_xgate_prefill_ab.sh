#!/usr/bin/env bash
# PREFILL cross-GPU rendezvous CEILING, GLM-5.2 TP4. One lease, controls interleaved.
#
# The question: does a tile-id / watermark cross-GPU gate beat the existing system-scope
# counter for TP PREFILL? This measures the HARD CEILING on any such redesign — no
# redesign can beat deleting the protocol outright.
#
# ARMS (every arm runs the SAME model.pkt, the SAME 2021 instructions, the SAME 256
# workgroups and the SAME 156 collectives per launch; ONLY the two prefill MLA+MoE code
# objects differ, and they differ only inside d_xreduce_twoshot_mega):
#   base      shipping
#   nowaitrs  gate_rs wait deleted   — the half a PRODUCER-SIDE tile watermark addresses
#   nowait    both rendezvous waits deleted
#   nosig     both waits AND all signalling deleted — the ABSOLUTE protocol ceiling.
#             Nothing that still publishes and observes progress can beat this arm.
# nowait/nowaitrs/nosig are NUMERICALLY WRONG on purpose (a rank may read a partial before
# its peer wrote it). They are measurement instruments, never results. `base` is the only
# arm whose output means anything, and its coherence is checked below.
#
# INSTRUMENT: `PLOW_TTFT_LOG=1` -> one `PF CHUNK ... drain=<ms>` line per prefill launch,
# the DEVICE WALL of that launch. `plowrt serve` amortises the 183 GiB/rank weight load
# over many requests, so each arm yields ~N samples instead of the 1 `amd-bench` gives.
#
# §0-BENCH: this is an EXPERIMENT instrument (plowrt, but a curl driver, not `vllm bench
# serve`). No number from it may be placed beside a vLLM number.
set -u
D=/home/lava/models/glm52_xrpf
PORT=8137
NREQ="${NREQ:-14}"
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
# THE instrument. Without it there is no `PF CHUNK` line and the run measures nothing.
export PLOW_TTFT_LOG=1
mkdir -p "$D/rt_logs"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"

# ~900 tokens of prompt: plan_chunks then covers it with the SINGLE T=1024 bucket, so every
# request is exactly ONE prefill launch and there is no padded tail chunk to average in.
PROMPT_FILE=$D/prompt.txt
[ -f "$PROMPT_FILE" ] || { echo "missing $PROMPT_FILE"; exit 1; }
PAYLOAD=$D/payload.json

arm_run () { # arm_run <tag>
  # NOT one `local` statement: bash expands the whole declaration in one pass, so
  # `local arm="$1" log=...$arm...` reads $arm before it is assigned -> unbound under set -u.
  local arm="$1"
  local log="$D/rt_logs/$arm.log"
  echo "########## ARM $arm"
  cd "$WT" || return 1   # `nix develop` resolves the flake from the CWD
  setsid nix develop --command "$WT/target/release/plowrt" serve \
      --assets "$D/assets_$arm" --port $PORT >"$log" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 900); do
    kill -0 $SRV 2>/dev/null || { echo "!! server died"; tail -30 "$log"; return 1; }
    curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
    sleep 1
  done
  [ $ok = 1 ] || { echo "!! never ready"; tail -30 "$log"; kill -KILL -$SRV 2>/dev/null; return 1; }
  # REFUSE TO MEASURE A CPU-BACKEND RUN. A plowrt built without `--features hsa` reports
  # hsa=false, silently selects the CPU reference backend AND the byte-fallback tokenizer,
  # answers in fluent-looking garbage, and emits no PF CHUNK at all — which is exactly how
  # the first attempt at this measurement produced six complete arms of nothing.
  if grep -q "falling back to CPU" "$log" || grep -q "hsa=false" "$log"; then
    echo "!! CPU reference backend — build plowrt with --features hsa"; tail -20 "$log"
    kill -KILL -$SRV 2>/dev/null; return 1
  fi
  echo "   ready"
  for i in $(seq 1 "$NREQ"); do
    curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' --data-binary @"$PAYLOAD" >"$D/rt_logs/$arm.resp$i.json"
  done
  # first response of the CONTROL is the coherence evidence; the wrong-by-design arms are
  # expected to be incoherent and that is not a failure of the measurement.
  echo "   sample: $(head -c 300 "$D/rt_logs/$arm.resp1.json")"
  kill -TERM -$SRV 2>/dev/null; sleep 3; kill -KILL -$SRV 2>/dev/null; sleep 2
  local nchunk; nchunk=$(grep -c 'PF CHUNK' "$log")
  echo "   PF CHUNK lines: $nchunk"
  [ "$nchunk" -gt 0 ] || echo "   !! NO PF CHUNK — PLOW_TTFT_LOG did not reach the server"
  echo "########## end $arm"
}

for a in base nosig base nowait nowaitrs base; do arm_run "$a" ; mv "$D/rt_logs/$a.log" "$D/rt_logs/$a.$(date +%s).log"; done
echo ALLDONE
touch "$D/prefill_ab.done"
