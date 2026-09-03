#!/usr/bin/env bash
# One measured pass against an OpenAI-compatible raw-completions endpoint.
# Both showdown arms invoke this exact client and endpoint.
#
#   bringup_bench.sh <tag> <base_url> <served_model> <tokenizer_dir> [round]
#
# Workload maps are whitespace-separated KEY=VALUE entries, keyed by input length;
# `default=` is the fallback. INPUT_MAP/INPUT_LENS keep the legacy IN_LENS default.
#   INPUT_LENS="128 1024 4096"
#   CONCURRENCY_MAP="default=1 4096=4"
#   PROMPT_MAP="default=6 4096=12"
#   WARMUP_MAP="default=1 4096=2"
# OUTLEN_MAP follows the same form. VLLM_CLIENT_COMMAND_ARGV is the complete
# command (default `vllm`); BRINGUP_CLIENT_ARGV appends vllm-bench args.
set -euo pipefail

[ "$#" -ge 4 ] || {
  echo "usage: $0 <tag> <base_url> <served_model> <tokenizer_dir> [round]" >&2
  exit 2
}
TAG=$1 URL=$2 MODEL=$3 TOK=$4 ROUND=${5:-1}
INPUT_LENS=${INPUT_MAP:-${INPUT_LENS:-${IN_LENS:-128 1024 4096}}}
CONCURRENCY_MAP=${CONCURRENCY_MAP:-default=1}
PROMPT_MAP=${PROMPT_MAP:-default=${NPROMPT:-6}}
WARMUP_MAP=${WARMUP_MAP:-default=1}
OUTLEN_MAP=${OUTLEN_MAP:-default=${OUTLEN:-8}}
OUTDIR=${BRINGUP_OUT:-/tmp/bringup-$USER}
SEED=${BRINGUP_SEED:-42}
ARTIFACT_DIGEST=${BRINGUP_ARTIFACT_DIGEST:-unknown}
mkdir -p "$OUTDIR"

map_value() { # <map> <key>
  local map=$1 key=$2 ent fallback=
  for ent in $map; do
    case "$ent" in
      "$key"=*) printf '%s\n' "${ent#*=}"; return 0 ;;
      default=*) fallback=${ent#*=} ;;
      *=*) ;;
      *) echo "invalid map entry '$ent' (expected KEY=VALUE)" >&2; return 2 ;;
    esac
  done
  [ -n "$fallback" ] || { echo "map has no '$key=' or default=: $map" >&2; return 2; }
  printf '%s\n' "$fallback"
}

positive_int() {
  case "$2" in *[!0-9]*|0|'') echo "$1 must be a positive integer, got '$2'" >&2; exit 2;; esac
}
nonnegative_int() {
  case "$2" in *[!0-9]*|'') echo "$1 must be a non-negative integer, got '$2'" >&2; exit 2;; esac
}

validate_log() { # <log> <prompts> <input_len> <output_len> <emit-row:0|1> <conc>
  python3 - "$TAG" "$ROUND" "$1" "$2" "$3" "$4" "$5" "$6" "$ARTIFACT_DIGEST" <<'PY'
import math, re, sys
tag, rnd, log, nprompt, inlen, outlen, emit, conc, digest = sys.argv[1:]
txt = open(log, encoding="utf-8", errors="replace").read()
def count(name):
    m = re.search(rf"{re.escape(name)}:\s+([\d,]+)", txt)
    if not m:
        raise SystemExit(f"missing {name} in {log}")
    return int(m.group(1).replace(",", ""))
success = count("Successful requests")
input_tokens = count("Total input tokens")
generated = count("Total generated tokens")
failed = count("Failed requests")
expected_requests = int(nprompt)
expected_input = expected_requests * int(inlen)
expected_output = expected_requests * int(outlen)
if failed != 0:
    raise SystemExit(f"failed requests in {log}: {failed}")
if success != expected_requests:
    raise SystemExit(f"incomplete requests in {log}: {success} != {expected_requests}")
if input_tokens != expected_input:
    raise SystemExit(f"input-token mismatch in {log}: {input_tokens} != {expected_input}")
if generated != expected_output:
    raise SystemExit(f"partial output in {log}: {generated} != {expected_output}")
def metric(pattern, name):
    m = re.search(pattern, txt)
    if not m:
        raise SystemExit(f"missing {name} in {log}")
    value = float(m.group(1))
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"invalid {name} in {log}: {value}")
    return m.group(1)
if emit == "0":
    raise SystemExit(0)
def latency(name):
    mean = metric(rf"(?:Mean|Average) {name} \(ms\):\s+([\d.eE+-]+)", f"mean {name}")
    med = metric(rf"Median {name} \(ms\):\s+([\d.eE+-]+)", f"median {name}")
    p99 = metric(rf"P99 {name} \(ms\):\s+([\d.eE+-]+)", f"p99 {name}")
    return mean, med, p99
ttft = latency("TTFT")
tpot = latency("TPOT")
itl = latency("ITL")
e2el = latency("E2EL")
req_s = metric(r"Request throughput \(req/s\):\s+([\d.eE+-]+)", "request throughput")
out_s = metric(r"Output token throughput \(tok/s\):\s+([\d.eE+-]+)", "output throughput")
print("\t".join([
    tag, rnd, inlen, ttft[0], ttft[1], ttft[2], tpot[1], conc, nprompt,
    outlen, str(input_tokens), str(generated), req_s, out_s, digest,
    tpot[0], tpot[2], itl[0], itl[1], itl[2], e2el[0], e2el[1], e2el[2],
]))
PY
}

read -r -a CLIENT_COMMAND <<<"${VLLM_CLIENT_COMMAND_ARGV:-vllm}"
[ "${#CLIENT_COMMAND[@]}" -gt 0 ] || { echo "VLLM_CLIENT_COMMAND_ARGV is empty" >&2; exit 2; }
read -r -a CLIENT_EXTRA <<<"${BRINGUP_CLIENT_ARGV:-}"
run_client() { # <log> <inlen> <outlen> <prompts> <concurrency> <seed>
  "${CLIENT_COMMAND[@]}" bench serve --backend openai --base-url "$URL" --endpoint /v1/completions \
    --model "$MODEL" --tokenizer "$TOK" \
    --dataset-name random --random-input-len "$2" --random-output-len "$3" \
    --random-range-ratio 0 --request-rate inf --temperature 0 \
    --num-prompts "$4" --max-concurrency "$5" --ignore-eos --seed "$6" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    "${CLIENT_EXTRA[@]}" >"$1" 2>&1
}

for IL in $INPUT_LENS; do
  positive_int input-length "$IL"
  CONC=$(map_value "$CONCURRENCY_MAP" "$IL")
  NP=$(map_value "$PROMPT_MAP" "$IL")
  NW=$(map_value "$WARMUP_MAP" "$IL")
  OL=$(map_value "$OUTLEN_MAP" "$IL")
  positive_int concurrency "$CONC"
  positive_int prompts "$NP"
  nonnegative_int warmups "$NW"
  positive_int output-length "$OL"
  if [ "$NW" -gt 0 ]; then
    WLOG="$OUTDIR/$TAG-r$ROUND-in$IL-warmup.log"
    run_client "$WLOG" "$IL" "$OL" "$NW" "$CONC" "$((SEED + IL * 2))"
    validate_log "$WLOG" "$NW" "$IL" "$OL" 0 "$CONC"
  fi
  LOG="$OUTDIR/$TAG-r$ROUND-in$IL.log"
  run_client "$LOG" "$IL" "$OL" "$NP" "$CONC" "$((SEED + IL * 2 + 1))"
  ROW=$(validate_log "$LOG" "$NP" "$IL" "$OL" 1 "$CONC")
  printf '%s\n' "$ROW" >>"$OUTDIR/cells.tsv"
done
