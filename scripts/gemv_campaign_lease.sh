#!/usr/bin/env bash
# Build the current gfx950 harness outside one lease, sweep a live compiler census inside it,
# and publish only complete, correct samples for the physically detected GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NIX="${PLOW_NIX_BIN:-nix}"
OBJ_MM="${PLOW_GEMV_OBJ_MM:-16}"
TOOLCHAIN_LABEL="${PLOW_TOOLCHAIN_LABEL:-}"

capture_nix_toolchain() {
  local output
  output="$(env -u PLOW_TOOLCHAIN_LABEL "$NIX" develop "$ROOT" --command bash -c 'printf "__PLOW_TOOLCHAIN__=%s\n" "${PLOW_TOOLCHAIN_LABEL:?flake did not export PLOW_TOOLCHAIN_LABEL}"')"
  TOOLCHAIN_LABEL="$(awk -F= '/^__PLOW_TOOLCHAIN__=/ {print $2}' <<<"$output")"
  [[ -n "$TOOLCHAIN_LABEL" && "$(grep -c '^__PLOW_TOOLCHAIN__=' <<<"$output")" == 1 ]] || {
    echo "FAIL: could not capture one exact PLOW_TOOLCHAIN_LABEL from nix develop" >&2
    return 1
  }
}

build_harness() {
  local obj=$1 rocm hipcc bundler cc
  if [[ "${PLOW_GEMV_BUILD_DIRECT:-}" == 1 ]]; then
    [[ "${PLOW_GEMV_SELFTEST:-}" == 1 ]] || {
      echo "FAIL: PLOW_GEMV_BUILD_DIRECT is reserved for the CPU mock selftest" >&2
      return 1
    }
    rocm="${ROCM_PATH:-/opt/rocm}"
    hipcc="${PLOW_HIPCC:?direct mock build needs PLOW_HIPCC}"
    bundler="${PLOW_CLANG_OFFLOAD_BUNDLER:?direct mock build needs PLOW_CLANG_OFFLOAD_BUNDLER}"
    cc="${PLOW_CC:?direct mock build needs PLOW_CC}"
  else
    [[ "${PLOW_GEMV_BUILD_IN_NIX:-}" == 1 ]] || {
      echo "FAIL: production harness construction must run in nix develop" >&2
      return 1
    }
    rocm="${ROCM_PATH:?nix develop did not set ROCM_PATH}"
    hipcc="$(command -v hipcc)"
    for bundler in "$rocm/lib/llvm/bin/clang-offload-bundler" "$rocm/llvm/bin/clang-offload-bundler"; do
      [[ -x "$bundler" ]] && break
    done
    cc="${CC:-cc}"
  fi
  [[ -x "$hipcc" && -x "$bundler" ]] || {
    echo "FAIL: incomplete ROCm toolchain under $rocm" >&2
    return 1
  }
  echo "build hipcc : $hipcc"
  echo "toolchain   : $TOOLCHAIN_LABEL"
  "$hipcc" --offload-arch=gfx950 -O3 -w -DPLOW_GEMV_MM="$OBJ_MM" -DPLOW_GEMV_WALK=1 --genco "$ROOT/runtime/amd/test_kernels.hip" -o "$obj/test_kernels.co" -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common"
  "$bundler" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --input="$obj/test_kernels.co" --output="$obj/test_kernels.elf"
  "$cc" -O2 -std=gnu11 -o "$obj/gemv_row_sweep" "$ROOT/runtime/bench/gemm/gemv_row_sweep.c" "$ROOT/runtime/amd/hsa_backend.c" -I"$rocm/include" -L"$rocm/lib" -Wl,-rpath,"$rocm/lib" -lhsa-runtime64 -lm
}

if [[ "${1:-}" == --build-only ]]; then
  [[ $# == 2 ]] || { echo "usage: $0 --build-only OBJ" >&2; exit 2; }
  [[ -n "$TOOLCHAIN_LABEL" ]] || {
    echo "FAIL: internal Nix build did not receive the captured toolchain label" >&2
    exit 1
  }
  build_harness "$2"
  exit
fi

OBJ="${1:?usage: $0 OBJ JSONL CAMPAIGN -- EMIT_COMMAND...}"
JSONL="${2:?usage: $0 OBJ JSONL CAMPAIGN -- EMIT_COMMAND...}"
CAMPAIGN="${3:?usage: $0 OBJ JSONL CAMPAIGN -- EMIT_COMMAND...}"
[[ "${4:-}" == -- && $# -ge 5 ]] || {
  echo "usage: $0 OBJ JSONL CAMPAIGN -- EMIT_COMMAND..." >&2
  exit 2
}
shift 4
EMIT=("$@")

GPU="${TUNE_GPU:?set TUNE_GPU to MI350X or MI355X}"
[[ "$GPU" == MI350X || "$GPU" == MI355X ]] || {
  echo "FAIL: gfx950 campaign requires TUNE_GPU=MI350X or MI355X" >&2
  exit 2
}
[[ "$OBJ_MM" =~ ^(1|2|4|8|16)$ ]] || {
  echo "FAIL: PLOW_GEMV_OBJ_MM must be 1, 2, 4, 8, or 16" >&2
  exit 2
}
LEASE="${PLOW_GPULEASE_BIN:-$ROOT/perf-data/tools/gpulease}"
DB="${PLOW_TUNE_DB:-$ROOT/tuning}"
FILTER="${PLOW_GEMV_CENSUS_FILTER:-}"
if [[ "${PLOW_GEMV_BUILD_DIRECT:-}" == 1 ]]; then
  [[ -n "$TOOLCHAIN_LABEL" ]] || {
    echo "FAIL: mock build must set PLOW_TOOLCHAIN_LABEL explicitly" >&2
    exit 1
  }
else
  capture_nix_toolchain
fi
[[ ! -e "$OBJ" && ! -L "$OBJ" ]] || {
  echo "FAIL: OBJ must be a fresh private path: $OBJ" >&2
  exit 1
}
[[ ! -e "$JSONL" ]] || {
  echo "FAIL: refusing to overwrite samples: $JSONL" >&2
  exit 1
}
mkdir -p "$(dirname "$OBJ")" "$(dirname "$JSONL")"
mkdir "$OBJ"
OBJ="$(cd "$OBJ" && pwd)"

TMP="$(mktemp -d /tmp/plow-gemv-campaign.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
CENSUS="$TMP/census.log"
RAW="$TMP/raw.jsonl"
SWEEP_LOG="$TMP/sweep.log"

run_tunedb() {
  if [[ -n "${PLOW_TUNEDB_GEMV_BIN:-}" ]]; then
    "$PLOW_TUNEDB_GEMV_BIN" "$@"
  else
    "$NIX" develop "$ROOT" --command env PLOW_TOOLCHAIN_LABEL="$TOOLCHAIN_LABEL" cargo run --quiet --release -p tunedb --bin tunedb-gemv -- "$@"
  fi
}

identity() {
  local gpu=$1 output cell build toolchain oracle
  output="$(run_tunedb best --db "$DB" --gpu "$gpu")"
  cell="$(awk -F': ' '$1 ~ /^cell/ {print $2}' <<<"$output")"
  build="$(awk -F': ' '$1 ~ /^build/ {print $2}' <<<"$output")"
  toolchain="$(awk -F': ' '$1 ~ /^toolchain/ {print $2}' <<<"$output")"
  oracle="$(awk -F': ' '$1 ~ /^oracle/ {print $2}' <<<"$output")"
  [[ "$cell" == amd/gfx950/mi350x && -n "$build" && "$toolchain" == "$TOOLCHAIN_LABEL" && -n "$oracle" ]] || {
    echo "FAIL: expected amd/gfx950/mi350x and complete interpreter/toolchain/oracle identity" >&2
    return 1
  }
  printf '%s\t%s\t%s\t%s\n' "$cell" "$build" "$toolchain" "$oracle"
}

# Census and every build step happen before the lease.
PLOW_TUNE_DUMP=1 "${EMIT[@]}" >"$CENSUS" 2>&1
python3 "$ROOT/scripts/gemv_campaign_census.py" plan --census "$CENSUS" --filter "$FILTER" --obj-mm "$OBJ_MM" >/dev/null
if [[ "${PLOW_GEMV_BUILD_DIRECT:-}" == 1 ]]; then
  build_harness "$OBJ"
else
  "$NIX" develop "$ROOT" --command env PLOW_GEMV_BUILD_IN_NIX=1 PLOW_TOOLCHAIN_LABEL="$TOOLCHAIN_LABEL" PLOW_GEMV_OBJ_MM="$OBJ_MM" bash "$ROOT/scripts/gemv_campaign_lease.sh" --build-only "$OBJ"
fi
before="$(identity "$GPU")"

if env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-7200}" PLOW_GEMV_CENSUS_FILTER="$FILTER" PLOW_GEMV_OBJ_MM="$OBJ_MM" "$LEASE" -n 1 gemv-campaign bash "$ROOT/scripts/rebench_tune_gemv.sh" "$OBJ" "$RAW" "$CENSUS" >"$SWEEP_LOG" 2>&1; then
  :
else
  rc=$?
  cat "$SWEEP_LOG" >&2
  echo "FAIL: sweep/lease failed rc=$rc; samples will not be ingested" >&2
  exit "$rc"
fi
cat "$SWEEP_LOG"

mapfile -t reports < <(
  awk '/^AMD Instinct MI(350X|355X)[[:space:]]+[0-9]+ CUs$/ {print $3 "\t" $4}' "$SWEEP_LOG" | sort -u
)
all_reports="$(grep -Ec '^AMD Instinct .* [0-9]+ CUs$' "$SWEEP_LOG" || true)"
matched_reports="$(grep -Ec '^AMD Instinct MI(350X|355X)[[:space:]]+[0-9]+ CUs$' "$SWEEP_LOG" || true)"
[[ "${#reports[@]}" == 1 && "$all_reports" == "$matched_reports" ]] || {
  echo "FAIL: sweep must report exactly one unique MI350X/MI355X CU identity" >&2
  exit 1
}
IFS=$'\t' read -r detected_gpu detected_cus <<<"${reports[0]}"
[[ "$detected_cus" == 256 ]] || {
  echo "FAIL: $detected_gpu reported $detected_cus CUs, expected 256" >&2
  exit 1
}
echo "detected GPU: $detected_gpu $detected_cus CUs"

python3 "$ROOT/scripts/gemv_campaign_census.py" filter --census "$CENSUS" --filter "$FILTER" --obj-mm "$OBJ_MM" --raw "$RAW" --output "$JSONL"

after="$(identity "$detected_gpu")"
[[ "$before" == "$after" ]] || {
  echo "FAIL: tuning cell/interpreter identity changed during campaign" >&2
  exit 1
}
IFS=$'\t' read -r expected_cell expected_build expected_toolchain expected_oracle <<<"$before"
run_tunedb ingest --db "$DB" --gpu "$detected_gpu" --samples "$JSONL" --campaign "$CAMPAIGN" --expect-cell "$expected_cell" --expect-build "$expected_build" --expect-toolchain "$expected_toolchain" --expect-oracle "$expected_oracle"
