#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/nix_rocm_714.sh"
plow_init_rocm_714

SHAPES="${PLOW_K3_SHAPES:-$ROOT/scripts/tune_shapes_k3.txt}"
OBJECT="${PLOW_K3_OBJECT:-/home/lava/models/k3_mi325x/hsaco/interp_prefill_fp8kv_k3_moe_a4w4.elf}"
BUILD_DIR="${PLOW_K3_BUILD_DIR:-$ROOT/build-amd/k3-mi325x-roof}"
HARNESS="${PLOW_K3_HARNESS:-$BUILD_DIR/bench/interp_mxfp4_bench}"
JSONL="${PLOW_GEMM_JSONL:?set PLOW_GEMM_JSONL to a new output path}"
CAMPAIGN="${PLOW_CAMPAIGN:-k3-mi325x-prod-interp-mxfp4}"
LEASE="${PLOW_LEASE_LABEL:-k3-mi325x-mxfp4-full}"
PLOWC="${PLOWC:-$ROOT/target/release/plowc}"

for path in "$SHAPES" "$OBJECT" "$PLOWC"; do
    [ -f "$path" ] || { echo "FAIL: missing $path" >&2; exit 2; }
done
[ -x "$HARNESS" ] || {
    cmake -S "$ROOT/runtime" -B "$BUILD_DIR" \
        -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --target interp_mxfp4_bench -j"${JOBS:-2}"
}
[ -x "$HARNESS" ] || { echo "FAIL: harness is not executable: $HARNESS" >&2; exit 2; }
[ -x "$PLOWC" ] || { echo "FAIL: plowc is not executable: $PLOWC" >&2; exit 2; }

if [ "${PLOW_K3_MXFP4_WORKER:-0}" != 1 ]; then
    [ ! -e "$JSONL" ] || {
        echo "FAIL: refusing to append to existing sample file $JSONL" >&2
        exit 2
    }
    mkdir -p "$(dirname "$JSONL")"
    "$ROOT/perf-data/tools/gpulease" -n 1 "$LEASE" \
        env PLOW_K3_MXFP4_WORKER=1 "$0"
    cd "$ROOT"
    "$PLOWC" tune ingest --gpu MI325X --db tuning --samples "$JSONL" --campaign "$CAMPAIGN"
    "$PLOWC" tune status --gpu MI325X --db tuning
    "$PLOWC" tune best --gpu MI325X --db tuning --quant Mxfp4
    exit 0
fi

object_sha="$(sha256sum "$OBJECT" | awk '{print $1}')"
export PLOW_STAGE4_CLEARED=1
export PLOW_GPU=MI325X
export PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix
export PLOW_BUILD_ID="gfx942-${object_sha:0:16}"
export PLOW_LEASE_LABEL="$LEASE"
export PLOW_GEMM_JSONL="$JSONL"

count=0
while read -r m n k label quant; do
    [ "$quant" = Mxfp4 ] || continue
    count=$((count + 1))
    echo "[$count/96] $label: $m x $n x $k"
    "$HARNESS" "$OBJECT" "$m" "$n" "$k" "$label"
done < <(awk '!/^#/ && NF >= 5 {print $1, $2, $3, $4, $5}' "$SHAPES")

[ "$count" -eq 96 ] || {
    echo "FAIL: expected 96 Mxfp4 shapes, measured $count" >&2
    exit 2
}
