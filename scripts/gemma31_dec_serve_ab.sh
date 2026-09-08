#!/usr/bin/env bash
# Interleaved served A/B over BLOBS (not objects), for the decode-residue campaign.
# Runs INSIDE one gpulease and INSIDE `nix develop` — outside it plowrt silently falls back
# to the CPU reference interpreter and every number is fiction.
#
#   GPU_LEASE_TIMEOUT=21600 perf-data/tools/gpulease -n 1 gemma31-dec-serve \
#     nix develop /app/plow --command scripts/gemma31_dec_serve_ab.sh <outdir> <rounds> \
#        ctl=<assetsdir> cand=<assetsdir> ...
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?output dir}"; shift
ROUNDS="${1:?rounds}"; shift
OBJ="${PLOW_OBJ_DIR:-/app/plow/build-gemma31/hsaco-tiered}"
BIN="${PLOW_BIN_DIR:-/app/plow/build-gemma31/residue/target/release}"
PORT0="${PORT0:-20100}"
INPUTS="${INPUTS:-128 512 2048 8192}"
mkdir -p "$OUT"

serve_up() { # $1 assetsdir  $2 port  $3 logfile
  PLOW_PF_BATCH=1 PLOW_PF_CHUNK=8192 PLOW_MULTISTEP=4 PLOW_TP_NO_AUDIT=1 \
    "$ROOT/scripts/glm53_serve_inner.sh" "$1" "$2" "$OBJ" "$BIN" > "$3" 2>&1 &
  SRV=$!
  for i in $(seq 1 300); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$2/v1/models" | grep -q 200; then
      grep -qi "hsa" "$3" || { echo "REFUSING: no HSA backend line in $3"; kill $SRV; return 1; }
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

run_arm() { # $1 label  $2 assetsdir  $3 port  $4 round
  serve_up "$2" "$3" "$OUT/serve_$1_r$4.log" || return 1
  python3 "$ROOT/scripts/bench_packed_serve.py" --url "http://127.0.0.1:$3" \
    --out "$OUT/bench_$1_r$4.json" --label "$1-r$4" \
    --inputs $INPUTS --outputs 64 --concurrency 1 4 \
    --repeats 3 --warmups 1 > "$OUT/bench_$1_r$4.txt" 2>&1
  rc=$?
  serve_down
  return $rc
}

specs=("$@")
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) = 1 ]; then ord=("${specs[@]}"); else
    ord=(); for ((i=${#specs[@]}-1; i>=0; i--)); do ord+=("${specs[$i]}"); done; fi
  p=$PORT0
  for spec in "${ord[@]}"; do
    run_arm "${spec%%=*}" "${spec#*=}" "$p" "$r" || exit 1
    p=$((p + 1))
  done
done
echo "=== done, results in $OUT"
