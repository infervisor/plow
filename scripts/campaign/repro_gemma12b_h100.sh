#!/usr/bin/env bash
# repro_gemma12b_h100.sh <step> — reproduce the Gemma-4-12B-it-FP8 / 1x H100 strict comparison
# (Infervisor vs vLLM 0.28, docs/bringup/results/gemma12b-fp8-20260930/comparison.md) from this checkout.
#
#   build                    CPU: plowc + plowrt + both packets (FP8-KV production, BF16-KV rq1k)
#   ref                      GPU: FP32 reference + vLLM peer capture, only when $REF is absent
#   gate <fp8|bf16>          GPU: serve the bundle, two FP32-reference captures, score (writes gates.json)
#   bench <arm> <workload>   GPU: one arm of one workload through llm_grid.sh
#                            arms: fp8 bf16 vllm vllm-bf16; workloads: st4k st15k agentic prod
#   report                   CPU: campaign.py report for every pair present + one combined markdown
#   all                      every step in order (GPU steps run in-process: launch it inside one lease)
#
# GPU steps go through the queue, one lease each (CLAUDE.md), e.g.
#   scripts/bench/gpuq.py submit g12-fp8-st4k 1 env OUT=... scripts/campaign/repro_gemma12b_h100.sh bench fp8 st4k
# The queue strips PLOW_* from the submitter's env: pass this script's inputs in the command.
#
# Inputs (env): OUT (required, scratch outside the repo); GEMMA12B_CHECKPOINT (re-keyed FP8 checkpoint
# the recipes build from); HF (HF gemma-4-12b-it-fp8 dir: tokenizer, vLLM model, FP32 reference);
# PYREF (python with torch for the gate; PB_VLLM is set from it for the client); VLLM_PY (python with
# vllm 0.28.0 for the baseline server, default $PYREF); REF / REF_VLLM (FP32 reference + vLLM capture,
# default $OUT/fp32ref/{ref,vllm}.json); CORPUS (dir with pg1342.txt pg944.txt repo-docs.md
# repo-code.rs, only for `ref` when $REF is absent); OBJECT_ENV (extra `--object-env`, e.g.
# "NVCC_APPEND_FLAGS=-ccbin=/usr/bin/g++-14" on a box without nix); RT_ENV (runtime env for plowrt,
# e.g. "PLOW_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1"); VLLM_ENV (env for the vLLM server).
#
# The qualified campaign (2026-10-05) used REF sha256 b2988c51…, prompts sha256 103024034d…, vLLM
# peer capture 74a7a572…; same files give the same gate. Packets serve from `plowrt serve --assets`
# alone (serve.json carries the recipe [serve.env]).
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${OUT:?set OUT to a scratch directory outside the repo}"
: "${PYREF:=python3}" "${VLLM_PY:=$PYREF}" "${REF:=$OUT/fp32ref/ref.json}" "${REF_VLLM:=$OUT/fp32ref/vllm.json}"
: "${OBJECT_ENV:=}" "${RT_ENV:=}" "${VLLM_ENV:=}"
TARGET=${CARGO_TARGET_DIR:-$REPO/target}
FP8_RECIPE=$REPO/recipes/infervisor/gemma-4-12b/sm90a-h100-tp1.toml
BF16_RECIPE=$REPO/scripts/campaign/recipes/sm90a-h100-tp1-fp8-16k-c64-rq1k.toml
# The baselines, exactly as qualified: matched FP8 KV (fp8_per_token_head runs only on TRITON_ATTN) and
# vLLM's fastest config overall (BF16 KV, FLASH_ATTN auto-selected). llm_grid.sh adds
# --gpu-memory-utilization $VLLM_MEM --max-num-seqs 256.
VLLM_FP8KV_ARGS="--dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details"
VLLM_BF16_ARGS="--dtype bfloat16 --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details"
PROD_RATES_Q="0.628 0.771 0.987"   # vLLM-calibrated mean in-flight 16 / 48 / 96 (agentic_turns.py --target-concurrency)
COMMON="HF=${HF:-} MODEL_NAME=checkpoint REPS=2 SAMPLED= PREFILL_CONCS= DECODE= PACKLOG=1 PB_BENCH_TIMEOUT=5400"

die() { echo "repro: $*" >&2; exit 2; }
need() { [ -n "${!1:-}" ] || die "set $1"; }
mkdir -p "$OUT"

build() {
  need GEMMA12B_CHECKPOINT
  (cd "$REPO" && cargo build --release -p plowc && cargo build --release -p plowrt --features cuda --bin plowrt --example step_bench) || exit
  mkdir -p "$OUT/bin" && cp "$TARGET/release/plowrt" "$OUT/bin/plowrt" && git -C "$REPO" rev-parse HEAD > "$OUT/bin/plowrt.sha"
  for arm in fp8 bf16; do
    local recipe=$FP8_RECIPE; [ $arm = bf16 ] && recipe=$BF16_RECIPE
    rm -rf "$OUT/pk/$arm"
    python3 "$REPO/scripts/campaign/campaign.py" build "$recipe" --out "$OUT/pk/$arm" --no-probe \
      ${OBJECT_ENV:+--object-env "$OBJECT_ENV"} > "$OUT/pk/$arm.log" 2>&1 || die "build $arm failed: $OUT/pk/$arm.log"
    echo "$arm packet $(sha256sum "$OUT/pk/$arm/assets/model.pkt" | cut -c1-12)"
  done
}

ref() {
  [ -s "$REF" ] && [ -s "$REF_VLLM" ] && { echo "reference present: $REF"; return; }
  need HF
  mkdir -p "$(dirname "$REF")"
  if [ ! -s "$REF" ]; then
    need CORPUS
    "$PYREF" "$REPO/scripts/llm/fp32_ref_gate.py" prompts --hf "$HF" --out "$(dirname "$REF")/prompts.json" \
      --corpus pride="$CORPUS/pg1342.txt" beagle="$CORPUS/pg944.txt" docs="$CORPUS/repo-docs.md" code="$CORPUS/repo-code.rs" || exit
    "$PYREF" "$REPO/scripts/llm/fp32_ref_gate.py" reference --hf "$HF" --prompts "$(dirname "$REF")/prompts.json" --out "$REF" || exit
  fi
  [ -s "$REF_VLLM" ] || serve_capture vllm "$(dirname "$REF_VLLM")" "$REF_VLLM"
}

# serve_capture <plow assets dir|vllm> <workdir> <capture json>: one server, the FP32-reference capture.
serve_capture() {
  local what=$1 dir=$2 cap=$3
  mkdir -p "$dir"
  (
    source "$REPO/scripts/bench/plowbench.sh"
    PB_SERVER_PORT=$(pb_free_port); PB_SERVER_LOG=$dir/server.log
    if [ "$what" = vllm ]; then
      env $VLLM_ENV "$VLLM_PY" -m vllm.entrypoints.cli.main serve "$HF" --port "$PB_SERVER_PORT" --served-model-name checkpoint \
        --gpu-memory-utilization 0.9 --max-num-seqs 256 $VLLM_FP8KV_ARGS > "$PB_SERVER_LOG" 2>&1 &
    else
      env $RT_ENV "$OUT/bin/plowrt" serve --assets "$what" --port "$PB_SERVER_PORT" > "$PB_SERVER_LOG" 2>&1 &
    fi
    PB_SERVER_PID=$!; trap pb_serve_stop EXIT
    pb_serve_wait 900 || exit 3
    "$PYREF" "$REPO/scripts/llm/fp32_ref_gate.py" capture --url "http://127.0.0.1:$PB_SERVER_PORT" \
      --ref "$REF" --arm "$(basename "$cap" .json)" --concurrency 16 --out "$cap" > "${cap%.json}.log" 2>&1
  )
}

gate() {
  local arm=$1 g=$OUT/gate/$1
  [ -s "$REF" ] && [ -s "$REF_VLLM" ] || die "no FP32 reference: run \`ref\` or set REF/REF_VLLM"
  rm -rf "$g"; mkdir -p "$g/llm_fp32_ref"
  sha256sum "$OUT/pk/$arm/assets/model.pkt" | cut -d' ' -f1 > "$g/packet.sha256"
  serve_capture "$OUT/pk/$arm/assets" "$g/llm_fp32_ref" "$g/llm_fp32_ref/plow.json" || die "gate $arm capture failed"
  printf '%s\n' 'schema = "plow.recipe.v1"' 'name = "repro-gate"' '[gates]' "python = \"$PYREF\"" \
    '[gates.llm_fp32_ref]' "reference = \"$REF\"" "vllm_capture = \"$REF_VLLM\"" > "$g/gate.toml"
  python3 "$REPO/scripts/campaign/campaign.py" gate "$g/gate.toml" --assets "$OUT/pk/$arm/assets" --out "$g" --score-only
}

bench() {
  local arm=$1 wl=$2 out=$OUT/res/$2/$1 wlenv mode=
  case $wl in
    st4k) wlenv="CONCS='32 128' ISL=4096 OSL=128" ;;
    st15k) wlenv="CONCS='32 128' ISL=15000 OSL=128" ;;
    agentic) wlenv="AGENTIC_CONCS='32 64 128'"; mode=--agentic ;;
    prod) wlenv="PROD_RATES='$PROD_RATES_Q' PROD_DURATION=300 PROD_WARMUP=75 PROD_COOLDOWN=25"; mode=--prod ;;
    *) die "workload $wl" ;;
  esac
  need HF
  rm -rf "$out"; mkdir -p "$out"
  case $arm in
    fp8|bf16)
      local kv=fp8_per_token_head; [ $arm = bf16 ] && kv=bfloat16
      eval env $COMMON PYREF=$PYREF $RT_ENV PLOWRT=$OUT/bin/plowrt PLOWRT_GIT_SHA=$(cat "$OUT/bin/plowrt.sha") \
        KV_DTYPE=$kv ASSETS=$OUT/pk/$arm/assets $wlenv bash "$REPO/scripts/bench/llm_grid.sh" plow "$out" $mode ;;
    vllm|vllm-bf16)
      local args=$VLLM_FP8KV_ARGS; [ $arm = vllm-bf16 ] && args=$VLLM_BF16_ARGS
      eval env $COMMON PYREF=$VLLM_PY $VLLM_ENV VLLM_MEM=0.9 "VLLM_ARGS='$args'" $wlenv \
        bash "$REPO/scripts/bench/llm_grid.sh" vllm "$out" $mode ;;
    *) die "arm $arm" ;;
  esac > "$out/run.log" 2>&1
}

report() {
  local pairs="st4k:vllm:fp8 st15k:vllm:fp8 agentic:vllm:fp8 prod:vllm:fp8 st4k:vllm-bf16:bf16 st15k:vllm-bf16:bf16 agentic:vllm-bf16:bf16"
  local all=$OUT/report/ALL-strict-comparison.md p wl base inf name
  mkdir -p "$OUT/report"
  printf '# Gemma-4-12B-it-FP8 on 1x H100: Infervisor vs vLLM 0.28 (strict)\n\nplowrt %s; every section is `campaign.py report` output.\n' \
    "$(cat "$OUT/bin/plowrt.sha" 2>/dev/null)" > "$all"
  for p in $pairs; do
    IFS=: read -r wl base inf <<< "$p"
    [ -d "$OUT/res/$wl/$base" ] && [ -d "$OUT/res/$wl/$inf" ] || continue
    name=$wl-$inf
    python3 "$REPO/scripts/campaign/campaign.py" report --baseline "$OUT/res/$wl/$base" --infervisor "$OUT/res/$wl/$inf" \
      --gate "$OUT/gate/$inf/gates.json" --out "$OUT/report/$name"
    echo "$name rc=$?"
    [ -s "$OUT/report/$name/comparison.md" ] || continue
    printf '\n## %s (%s vs %s)\n\n' "$wl" "$inf" "$base" >> "$all"
    sed '1d; s/^## /### /' "$OUT/report/$name/comparison.md" >> "$all"
  done
  echo "combined: $all"
}

case ${1:-} in
  build) build ;;
  ref) ref ;;
  gate) gate "${2:?fp8|bf16}" ;;
  bench) bench "${2:?arm}" "${3:?workload}" ;;
  report) report ;;
  all)
    build && ref && gate fp8 && gate bf16 || exit
    for wl in st4k st15k agentic; do for arm in vllm fp8 vllm-bf16 bf16; do bench $arm $wl; done; done
    bench vllm prod; bench fp8 prod
    report ;;
  *) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
