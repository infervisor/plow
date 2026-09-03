#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d /tmp/bringup-bench-selftest.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

FAKE="$TMP/vllm"
cat >"$FAKE" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[ "$1" = bench ] && [ "$2" = serve ] || exit 70
shift 2
np= il= ol=
exact_range= exact_rate= exact_temp= exact_pct=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --num-prompts) np=$2; shift 2 ;;
    --random-input-len) il=$2; shift 2 ;;
    --random-output-len) ol=$2; shift 2 ;;
    --random-range-ratio) exact_range=$2; shift 2 ;;
    --request-rate) exact_rate=$2; shift 2 ;;
    --temperature) exact_temp=$2; shift 2 ;;
    --metric-percentiles) exact_pct=$2; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "$np" ] && [ -n "$il" ] && [ -n "$ol" ]
[ "$exact_range" = 0 ] && [ "$exact_rate" = inf ] && [ "$exact_temp" = 0 ]
[ "$exact_pct" = 50,90,99 ]
input=$((np * il)); output=$((np * ol))
[ "${FAKE_BAD_INPUT:-0}" = 0 ] || input=$((input - 1))
printf 'Successful requests: %s\nFailed requests: 0\n' "$np"
printf 'Total input tokens: %s\nTotal generated tokens: %s\n' "$input" "$output"
printf 'Request throughput (req/s): 1.0\nOutput token throughput (tok/s): 2.0\n'
for metric in TTFT TPOT ITL E2EL; do
  printf 'Mean %s (ms): 3.0\nMedian %s (ms): 2.0\nP99 %s (ms): 4.0\n' \
    "$metric" "$metric" "$metric"
done
SH
chmod +x "$FAKE"

OUT="$TMP/good"
INPUT_MAP="128 1024" CONCURRENCY_MAP="default=1 1024=3" \
  PROMPT_MAP="default=2 1024=4" WARMUP_MAP="default=1 1024=2" \
  OUTLEN_MAP="default=3" VLLM_CLIENT_COMMAND_ARGV="env $FAKE" BRINGUP_OUT="$OUT" \
  BRINGUP_ARTIFACT_DIGEST=test-digest \
  "$HERE/bringup_bench.sh" arm http://invalid model tokenizer r1

[ "$(wc -l <"$OUT/cells.tsv")" -eq 2 ]
awk -F '\t' '
  $3 == 128  { if ($8 != 1 || $9 != 2 || $15 != "test-digest" || NF != 23) exit 1; a=1 }
  $3 == 1024 { if ($8 != 3 || $9 != 4 || $15 != "test-digest" || NF != 23) exit 1; b=1 }
  END { exit !(a && b) }
' "$OUT/cells.tsv"
[ "$(find "$OUT" -name '*.log' | wc -l)" -eq 4 ]

if FAKE_BAD_INPUT=1 INPUT_MAP=128 WARMUP_MAP=default=0 PROMPT_MAP=default=2 \
  OUTLEN_MAP=default=3 VLLM_CLIENT_COMMAND_ARGV="$FAKE" BRINGUP_OUT="$TMP/bad" \
  "$HERE/bringup_bench.sh" arm http://invalid model tokenizer r1 >/dev/null 2>&1; then
  echo "count mismatch was accepted" >&2
  exit 1
fi

echo "bringup bench selftest: PASS"
