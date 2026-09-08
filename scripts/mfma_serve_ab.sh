#!/usr/bin/env bash
# Interleaved control/candidate served A/B for the MFMA batched-decode candidate.
# Runs INSIDE one gpulease and INSIDE `nix develop` (plowrt silently falls back to the CPU
# reference backend outside it, and every number would be meaningless).
#
#   GPU_LEASE_TIMEOUT=21600 perf-data/tools/gpulease -n 1 gemma31-mfma-serve \
#     nix develop /app/plow --command scripts/mfma_serve_ab.sh <ctl-obj> <cand-obj> <outdir> [rounds]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CTL="${1:?control objdir}"
CAND="${2:?candidate objdir}"
OUT="${3:?output dir}"
ROUNDS="${4:-2}"
ASSETS="${PLOW_ASSETS:-/app/plow/build-gemma31/assets-ctx131072-chunk8192}"
BIN="${PLOW_BIN_DIR:-/app/plow/target-glm53/release}"
mkdir -p "$OUT"

serve_up() { # $1 objdir  $2 port  $3 logfile
  PLOW_PF_BATCH=1 PLOW_PF_CHUNK=8192 PLOW_MULTISTEP=4 PLOW_TP_NO_AUDIT=1 \
    "$ROOT/scripts/glm53_serve_inner.sh" "$ASSETS" "$2" "$1" "$BIN" > "$3" 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$2/v1/models" | grep -q 200; then
      # THE BACKEND LINE IS THE GATE: outside nix, or with a bad object dir, plowrt serves
      # coherent answers from the CPU reference interpreter at fictional speed.
      if ! grep -qi "hsa" "$3"; then
        echo "REFUSING: no HSA backend line in $3"; kill $SRV 2>/dev/null; return 1
      fi
      echo "up: $1 on $2 after ${i}s"; return 0
    fi
    kill -0 $SRV 2>/dev/null || { echo "server died, see $3"; return 1; }
    sleep 1
  done
  echo "TIMEOUT waiting for $2"; kill $SRV 2>/dev/null; return 1
}

serve_down() {
  kill $SRV 2>/dev/null
  for i in $(seq 1 60); do kill -0 $SRV 2>/dev/null || return 0; sleep 1; done
  kill -9 $SRV 2>/dev/null
}

run_arm() { # $1 label  $2 objdir  $3 port  $4 round
  serve_up "$2" "$3" "$OUT/serve_$1_r$4.log" || return 1
  python3 "$ROOT/scripts/bench_packed_serve.py" --url "http://127.0.0.1:$3" \
    --out "$OUT/bench_$1_r$4.json" --label "$1-r$4" \
    --inputs 128 512 2048 8192 --outputs 64 --concurrency 1 4 \
    --repeats 3 --warmups 1 > "$OUT/bench_$1_r$4.txt" 2>&1
  rc=$?
  serve_down
  return $rc
}

for r in $(seq 1 "$ROUNDS"); do
  # palindromic: control first on odd rounds, candidate first on even
  if [ $((r % 2)) = 1 ]; then order="ctl cand"; else order="cand ctl"; fi
  for a in $order; do
    case "$a" in
      ctl)  run_arm ctl  "$CTL"  19900 "$r" || exit 1 ;;
      cand) run_arm cand "$CAND" 19901 "$r" || exit 1 ;;
    esac
  done
done
echo "=== done, results in $OUT"
