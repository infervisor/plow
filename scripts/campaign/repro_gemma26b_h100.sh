#!/usr/bin/env bash
# repro_gemma26b_h100.sh <step> — the Gemma-4-26B-A4B-it / 1x H100 strict comparison (Infervisor vs
# vLLM 0.28), BF16 and FP8, from this checkout. The 26B twin of repro_gemma12b_h100.sh.
#
#   build [bf16|fp8]         CPU: plowc + plowrt + step_bench + the packet(s)
#   ref <bf16|fp8>           GPU: FP32 reference + vLLM peer capture, only when absent
#   gate <bf16|fp8>          GPU: serve the packet, FP32-reference capture, score (writes gates.json)
#   bench <arm> <workload>   GPU: one arm of one workload through llm_grid.sh
#                            arms: bf16 fp8 vllm-bf16 vllm-fp8; workloads: st4k st15k agentic lat,
#                            lc32k lc128k (32768 / 130944 prompts, c1/c4/c16), prod (open-loop mix at RATES
#                            sessions/s), prodlong (the mix with a long-prompt tail to 131072)
#   report                   CPU: campaign.py report for every pair present + one combined markdown
#
# GPU steps go through the queue, one lease each (CLAUDE.md), e.g.
#   scripts/bench/gpuq.py submit g26-bf16-st4k 1 env OUT=... scripts/campaign/repro_gemma26b_h100.sh bench bf16 st4k
# The queue strips PLOW_* from the submitter's env: pass this script's inputs in the command.
#
# Inputs (env): OUT (required, scratch outside the repo); HF_BF16 (google/gemma-4-26B-A4B-it snapshot:
# tokenizer, vLLM model, plow BF16 checkpoint, FP32 reference); HF_FP8 (the hub compressed-tensors
# FP8 checkpoint, with a tokenizer.json: vLLM model and FP32 reference); PLOW_FP8_CKPT (HF_FP8 re-keyed
# by scripts/gemma4_fp8_hub_rekey.py: the plow FP8 checkpoint); BF16_RECIPE / FP8_RECIPE; PYREF (python
# with torch + vllm 0.28.0); REF_<BF16|FP8> / REF_VLLM_<BF16|FP8> (FP32 reference + vLLM capture,
# default $OUT/fp32ref/<arm>/{ref,vllm}.json); PROMPTS (fp32_ref_gate.py prompt set; built from CORPUS
# when absent); OBJECT_ENV (extra `--object-env`, e.g. "NVCC_APPEND_FLAGS=-ccbin=/usr/bin/g++-14");
# RT_ENV (runtime env for plowrt); VLLM_ENV (env for the vLLM server); MAX_MODEL_LEN (vLLM
# --max-model-len; default the recipe's max_ctx: FP8 131072, BF16 16384); RATES (prod/prodlong sessions/s).
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${OUT:?set OUT to a scratch directory outside the repo}"
: "${PYREF:=python3}" "${OBJECT_ENV:=}" "${RT_ENV:=}" "${VLLM_ENV:=}" "${PROMPTS:=$OUT/fp32ref/prompts.json}"
TARGET=${CARGO_TARGET_DIR:-$REPO/target}
: "${BF16_RECIPE:=$REPO/recipes/infervisor/gemma-4-26b-a4b/sm90a-h100-tp1-bf16.toml}"
: "${FP8_RECIPE:=$REPO/recipes/infervisor/gemma-4-26b-a4b/sm90a-h100-tp1-fp8.toml}"
# llm_grid.sh adds --gpu-memory-utilization $VLLM_MEM --max-num-seqs 256. fp8_per_token_head KV runs
# only on TRITON_ATTN.
VLLM_COMMON="--enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details"
VLLM_BF16_ARGS="--dtype bfloat16 --max-model-len ${MAX_MODEL_LEN:-16384} $VLLM_COMMON"
VLLM_FP8_ARGS="--dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len ${MAX_MODEL_LEN:-131072} $VLLM_COMMON"
COMMON="REPS=${REPS:-2} SAMPLED= PREFILL_CONCS= DECODE= PACKLOG=1 PB_BENCH_TIMEOUT=5400"

die() { echo "repro26: $*" >&2; exit 2; }
need() { [ -n "${!1:-}" ] || die "set $1"; }
hf_of() { case $1 in bf16|vllm-bf16) echo "${HF_BF16:-}" ;; *) echo "${HF_FP8:-}" ;; esac; }
prec_of() { case $1 in bf16|vllm-bf16) echo bf16 ;; *) echo fp8 ;; esac; }
ref_of() { local p; p=$(prec_of "$1"); local v="REF_${p^^}"; echo "${!v:-$OUT/fp32ref/$p/ref.json}"; }
refv_of() { local p; p=$(prec_of "$1"); local v="REF_VLLM_${p^^}"; echo "${!v:-$OUT/fp32ref/$p/vllm.json}"; }
vargs_of() { case $(prec_of "$1") in bf16) echo "$VLLM_BF16_ARGS" ;; *) echo "$VLLM_FP8_ARGS" ;; esac; }
mkdir -p "$OUT"

build() {
  (cd "$REPO" && cargo build --release -p plowc && cargo build --release -p plowrt --features cuda --bin plowrt --example step_bench) || exit
  mkdir -p "$OUT/bin" && cp "$TARGET/release/plowrt" "$TARGET/release/examples/step_bench" "$OUT/bin/" \
    && git -C "$REPO" rev-parse HEAD > "$OUT/bin/plowrt.sha"
  for arm in ${1:-bf16 fp8}; do
    local recipe=$BF16_RECIPE hf=${HF_BF16:-}
    [ "$arm" = fp8 ] && { recipe=$FP8_RECIPE; hf=${PLOW_FP8_CKPT:-}; }
    [ -n "$hf" ] || die "set HF_BF16 / PLOW_FP8_CKPT"
    rm -rf "$OUT/pk/$arm"; mkdir -p "$OUT/pk"
    python3 "$REPO/scripts/campaign/campaign.py" build "$recipe" --out "$OUT/pk/$arm" --no-probe --hf-dir "$hf" \
      ${OBJECT_ENV:+--object-env "$OBJECT_ENV"} > "$OUT/pk/$arm.log" 2>&1 || die "build $arm failed: $OUT/pk/$arm.log"
    echo "$arm packet $(sha256sum "$OUT/pk/$arm/assets/model.pkt" | cut -c1-12)"
  done
}

ref() {
  local p=$1 hf r rv; hf=$(hf_of "$p"); r=$(ref_of "$p"); rv=$(refv_of "$p")
  [ -s "$r" ] && [ -s "$rv" ] && { echo "reference present: $r"; return; }
  [ -n "$hf" ] || die "set HF_${p^^}"
  mkdir -p "$(dirname "$r")" "$(dirname "$PROMPTS")"
  if [ ! -s "$PROMPTS" ]; then
    need CORPUS
    "$PYREF" "$REPO/scripts/llm/fp32_ref_gate.py" prompts --hf "$hf" --out "$PROMPTS" \
      --corpus pride="$CORPUS/pg1342.txt" beagle="$CORPUS/pg944.txt" docs="$CORPUS/repo-docs.md" code="$CORPUS/repo-code.rs" || exit
  fi
  [ -s "$r" ] || "$PYREF" "$REPO/scripts/llm/fp32_ref_gate.py" reference --hf "$hf" --prompts "$PROMPTS" --out "$r" || exit
  [ -s "$rv" ] || serve_capture "vllm-$p" "$(dirname "$rv")" "$rv"
}

# serve_capture <arm> <workdir> <capture json>: one server (plow arm or vllm-<prec>), the FP32-reference capture.
serve_capture() {
  local arm=$1 dir=$2 cap=$3
  mkdir -p "$dir"
  (
    source "$REPO/scripts/bench/plowbench.sh"
    PB_SERVER_PORT=$(pb_free_port); PB_SERVER_LOG=$dir/server.log
    case $arm in
      vllm-*) env $VLLM_ENV "$PYREF" -m vllm.entrypoints.cli.main serve "$(hf_of "$arm")" --port "$PB_SERVER_PORT" \
                --served-model-name checkpoint --gpu-memory-utilization 0.9 --max-num-seqs 256 $(vargs_of "$arm") \
                > "$PB_SERVER_LOG" 2>&1 & ;;
      *) env $RT_ENV "$OUT/bin/plowrt" serve --assets "$OUT/pk/$arm/assets" --port "$PB_SERVER_PORT" > "$PB_SERVER_LOG" 2>&1 & ;;
    esac
    PB_SERVER_PID=$!; trap pb_serve_stop EXIT
    pb_serve_wait 900 || exit 3
    "$PYREF" "$REPO/scripts/llm/fp32_ref_gate.py" capture --url "http://127.0.0.1:$PB_SERVER_PORT" \
      --ref "$(ref_of "$arm")" --arm "$(basename "$cap" .json)" --concurrency "${GATE_CONC:-16}" --out "$cap" > "${cap%.json}.log" 2>&1
  )
}

# gate <arm>: the canonical recipe's [gates] (llm_fp32_ref, plus llm_fp32_ref_long when REF_LONG_<P> is
# set) on the packet: campaign.py gate --dry-run writes run.sh, this lease runs it, --score-only scores.
gate() {
  local arm=$1 g=$OUT/gate/$1 P r rv rl rvl recipe=$BF16_RECIPE only=llm_fp32_ref
  [ "$arm" = fp8 ] && recipe=$FP8_RECIPE
  P=$(prec_of "$arm" | tr a-z A-Z); r=$(ref_of "$arm"); rv=$(refv_of "$arm")
  local vl="REF_LONG_$P" vvl="REF_VLLM_LONG_$P"; rl=${!vl:-} rvl=${!vvl:-}
  [ -s "$r" ] && [ -s "$rv" ] || die "no FP32 reference for $arm: run \`ref $(prec_of "$arm")\`"
  if [ -n "$rl" ]; then
    [ -s "$rl" ] && [ -s "$rvl" ] || die "REF_LONG_$P set but $rl / $rvl missing"
    only=llm_fp32_ref,llm_fp32_ref_long
  fi
  export "G26_${P}_FP32_REF=$r" "G26_${P}_FP32_VLLM=$rv" "G26_${P}_FP32_REF_LONG=$rl" "G26_${P}_FP32_VLLM_LONG=$rvl"
  rm -rf "$g"
  local c=(python3 "$REPO/scripts/campaign/campaign.py" gate "$recipe" --assets "$OUT/pk/$arm/assets" --out "$g"
           --plowrt "$OUT/bin/plowrt" --only "${GATE_ONLY:-$only}")
  "${c[@]}" --dry-run > /dev/null || die "gate $arm: campaign.py gate --dry-run failed"
  env $RT_ENV bash "$g/run.sh" > "$g/run.log" 2>&1
  "${c[@]}" --score-only
}

bench() {
  local arm=$1 wl=$2 out=$OUT/res/$2/$1 wlenv mode= hf
  case $wl in
    st4k) wlenv="CONCS='32 128' ISL=4096 OSL=128" ;;
    st15k) wlenv="CONCS='32 128' ISL=15000 OSL=128" ;;
    agentic) wlenv="AGENTIC_CONCS='32 64 128'"; mode=--agentic ;;
    lat) wlenv="CONCS='1 4' ISL=1024 OSL=128" ;;
    lc32k) wlenv="CONCS='1 4 16' ISL=32768 OSL=128" ;;
    lc128k) wlenv="CONCS='1 4 16' ISL=130944 OSL=128" ;;
    prod) need RATES; wlenv="PROD_RATES='$RATES'"; mode=--prod ;;
    prodlong) need RATES; wlenv="PROD_RATES='$RATES' PROD_ARGS='--max-model-len 131072 --first-sigma 2.0'"; mode=--prod ;;
    *) die "workload $wl" ;;
  esac
  hf=$(hf_of "$arm"); [ -n "$hf" ] || die "set HF_$(prec_of "$arm" | tr a-z A-Z)"
  rm -rf "$out"; mkdir -p "$out"
  case $arm in
    bf16|fp8)
      local kv=bfloat16; [ "$arm" = fp8 ] && kv=fp8_per_token_head
      eval env $COMMON HF=$hf MODEL_NAME=checkpoint PYREF=$PYREF $RT_ENV PLOWRT=$OUT/bin/plowrt \
        PLOWRT_GIT_SHA=$(cat "$OUT/bin/plowrt.sha") KV_DTYPE=$kv ASSETS=$OUT/pk/$arm/assets $wlenv \
        bash "$REPO/scripts/bench/llm_grid.sh" plow "$out" $mode ;;
    vllm-bf16|vllm-fp8)
      eval env $COMMON HF=$hf MODEL_NAME=checkpoint PYREF=$PYREF $VLLM_ENV VLLM_MEM=0.9 "VLLM_ARGS='$(vargs_of "$arm")'" \
        $wlenv bash "$REPO/scripts/bench/llm_grid.sh" vllm "$out" $mode ;;
    *) die "arm $arm" ;;
  esac > "$out/run.log" 2>&1
}

report() {
  local all=$OUT/report/ALL-strict-comparison.md wl inf name
  mkdir -p "$OUT/report"
  printf '# Gemma-4-26B-A4B-it on 1x H100: Infervisor vs vLLM 0.28 (strict)\n\nplowrt %s; every section is `campaign.py report` output.\n' \
    "$(cat "$OUT/bin/plowrt.sha" 2>/dev/null)" > "$all"
  for inf in bf16 fp8; do
    for wl in lat st4k st15k agentic lc32k lc128k prod prodlong; do
      [ -d "$OUT/res/$wl/vllm-$inf" ] && [ -d "$OUT/res/$wl/$inf" ] || continue
      name=$wl-$inf
      python3 "$REPO/scripts/campaign/campaign.py" report --baseline "$OUT/res/$wl/vllm-$inf" \
        --infervisor "$OUT/res/$wl/$inf" --gate "$OUT/gate/$inf/gates.json" --out "$OUT/report/$name"
      echo "$name rc=$?"
      [ -s "$OUT/report/$name/comparison.md" ] || continue
      printf '\n## %s (%s vs vllm-%s)\n\n' "$wl" "$inf" "$inf" >> "$all"
      sed '1d; s/^## /### /' "$OUT/report/$name/comparison.md" >> "$all"
    done
  done
  echo "combined: $all"
}

case ${1:-} in
  build) build "${2:-}" ;;
  ref) ref "${2:?bf16|fp8}" ;;
  gate) gate "${2:?bf16|fp8}" ;;
  bench) bench "${2:?arm}" "${3:?workload}" ;;
  report) report ;;
  *) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
