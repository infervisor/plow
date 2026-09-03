#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d /tmp/plow-gemv-campaign-selftest.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"/{bin,db,out}
export MOCK_LOG="$TMP/events.log"

cat >"$TMP/bin/emitter" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ -z "${PLOW_IN_LEASE:-}" && "${PLOW_TUNE_DUMP:-}" == 1 ]]
echo emit-outside >>"$MOCK_LOG"
echo 'TUNEDUMP_GEMV 1 64 32 None PLOW_DOP_GEMV MISS' >&2
echo 'TUNEDUMP_GEMV 1 96 32 None PLOW_DOP_GEMV_QKVG MISS' >&2
echo 'TUNEDUMP_GEMV 1 128 32 Mxfp4 PLOW_DOP_GEMV_MXFP4 MISS' >&2
SH
cat >"$TMP/bin/hipcc" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ -z "${PLOW_IN_LEASE:-}" ]]
echo hipcc-outside >>"$MOCK_LOG"
while [[ "$1" != -o ]]; do shift; done
touch "$2"
SH
cat >"$TMP/bin/bundler" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ -z "${PLOW_IN_LEASE:-}" ]]
echo bundler-outside >>"$MOCK_LOG"
for arg; do [[ "$arg" == --output=* ]] && touch "${arg#--output=}"; done
SH
cat >"$TMP/bin/sweep-source" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "${PLOW_IN_LEASE:-}" == 1 ]]
[[ "$PWD" == */obj-* && -f test_kernels.elf ]]
n=$1
k=$2
[[ "$k" == 32 && ( "$n" == 64 || "$n" == 96 || "$n" == 128 ) ]]
if [[ -n "${MOCK_WRONG_GPU:-}" ]]; then
  echo 'AMD Instinct MI325X  304 CUs'
else
  echo 'AMD Instinct MI355X  256 CUs'
fi
if [[ "$n" == 64 ]]; then
  stem=gemv
  quant=None
  [[ "$PLOW_GEMV_ARMS" == gemv ]]
elif [[ "$n" == 96 ]]; then
  stem=gemv_qkvg
  quant=None
  [[ "$PLOW_GEMV_ARMS" == gemv_qkvg ]]
else
  stem=gemv_mxfp4
  quant=Mxfp4
  [[ "$PLOW_GEMV_ARMS" == gemv_mxfp4 ]]
fi
for m in 1 2 4 8 16 32 64 128; do
  [[ "$stem" == gemv_mxfp4 && "$m" != 1 ]] && continue
  for mm in 1 2 4 8 16; do
    [[ "$stem" == gemv_mxfp4 && "$mm" != 16 ]] && continue
    [[ -n "${MOCK_MISSING_RUNG:-}" && "$stem" == gemv_qkvg && "$m" == 128 && "$mm" == 16 ]] && continue
    correct=true
    [[ -n "${MOCK_BAD_SAMPLE:-}" && "$stem" == gemv && "$m" == 1 && "$mm" == 1 ]] && correct=false
    printf '{"m":%s,"n":%s,"k":32,"quant":"%s","mm":%s,"sym":"%s_m%s","correct":%s,"samples_ns":[1,2,3,4,5]}\n' "$m" "$n" "$quant" "$mm" "$stem" "$mm" "$correct" >>"$PLOW_GEMV_JSONL"
  done
done
if [[ -n "${MOCK_DUPLICATE:-}" && "$stem" == gemv ]]; then
  tail -n 1 "$PLOW_GEMV_JSONL" >>"$PLOW_GEMV_JSONL"
fi
if [[ -n "${MOCK_UNEXPECTED:-}" && "$stem" == gemv ]]; then
  echo '{"m":1,"n":64,"k":32,"quant":"None","mm":1,"sym":"unknown_m1","correct":true,"samples_ns":[1,2,3,4,5]}' >>"$PLOW_GEMV_JSONL"
fi
SH
cat >"$TMP/bin/cc" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ -z "${PLOW_IN_LEASE:-}" ]]
echo cc-outside >>"$MOCK_LOG"
while [[ "$1" != -o ]]; do shift; done
cp "$MOCK_SWEEP_SOURCE" "$2"
chmod +x "$2"
SH
cat >"$TMP/bin/gpulease" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
echo lease >>"$MOCK_LOG"
shift 3
[[ -z "${MOCK_CONTENDED:-}" ]] || exit 76
exec env PLOW_IN_LEASE=1 "$@"
SH
cat >"$TMP/bin/tunedb" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ -z "${PLOW_IN_LEASE:-}" ]]
cmd=$1
gpu=
for ((i=1; i<=$#; i++)); do
  [[ "${!i}" == --gpu ]] || continue
  j=$((i + 1))
  gpu="${!j}"
done
if [[ "$cmd" == best ]]; then
  count=$(cat "$MOCK_IDENTITY_COUNT")
  count=$((count + 1))
  echo "$count" >"$MOCK_IDENTITY_COUNT"
  build=current-digest
  [[ -n "${MOCK_DIGEST_DRIFT:-}" && "$count" -gt 1 ]] && build=changed-digest
  printf 'cell        : amd/gfx950/mi350x\nbuild       : %s\ntoolchain   : rocm-7.14.0-nix\noracle      : gemv-test\n' "$build"
elif [[ "$cmd" == ingest ]]; then
  echo ingest >>"$MOCK_LOG"
  [[ "$gpu" == MI355X ]]
  samples= expect_cell= expect_build= expect_toolchain= expect_oracle=
  while (($#)); do
    case "$1" in
      --samples) samples=$2; shift 2 ;;
      --expect-cell) expect_cell=$2; shift 2 ;;
      --expect-build) expect_build=$2; shift 2 ;;
      --expect-toolchain) expect_toolchain=$2; shift 2 ;;
      --expect-oracle) expect_oracle=$2; shift 2 ;;
      *) shift ;;
    esac
  done
  [[ "$(wc -l <"$samples")" -eq 81 ]]
  [[ "$expect_cell" == amd/gfx950/mi350x && "$expect_build" == current-digest ]]
  [[ "$expect_toolchain" == rocm-7.14.0-nix && "$expect_oracle" == gemv-test ]]
else
  exit 2
fi
SH
chmod +x "$TMP/bin"/*
export MOCK_SWEEP_SOURCE="$TMP/bin/sweep-source"
export MOCK_IDENTITY_COUNT="$TMP/identity-count"

run_campaign() {
  local name=$1 output="$TMP/out/$1.jsonl"
  echo 0 >"$MOCK_IDENTITY_COUNT"
  TUNE_GPU=MI350X PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix PLOW_GEMV_BUILD_DIRECT=1 PLOW_GEMV_SELFTEST=1 PLOW_HIPCC="$TMP/bin/hipcc" PLOW_CLANG_OFFLOAD_BUNDLER="$TMP/bin/bundler" PLOW_CC="$TMP/bin/cc" PLOW_GPULEASE_BIN="$TMP/bin/gpulease" PLOW_TUNEDB_GEMV_BIN="$TMP/bin/tunedb" PLOW_TUNE_DB="$TMP/db" "$ROOT/scripts/gemv_campaign_lease.sh" "$TMP/obj-$name" "$output" current-census -- "$TMP/bin/emitter"
}

: >"$MOCK_LOG"
run_campaign pass >"$TMP/pass.log"
[[ "$(grep -c '^lease$' "$MOCK_LOG")" -eq 1 ]]
[[ "$(grep -c '^ingest$' "$MOCK_LOG")" -eq 1 ]]
for phase in emit hipcc bundler cc; do grep -q "^${phase}-outside$" "$MOCK_LOG"; done
grep -q 'detected GPU: MI355X 256 CUs' "$TMP/pass.log"
grep -q 'kept 81 passing rows covering 17 runtime census cases' "$TMP/pass.log"
for stem in gemv gemv_qkvg; do
  for mm in 1 2 4 8 16; do
    grep -q "\"sym\":\"${stem}_m${mm}\"" "$TMP/out/pass.jsonl"
  done
done
[[ "$(grep -Ec '"sym":"gemv_m(1|2|4|8|16)"' "$TMP/out/pass.jsonl")" -eq 40 ]]
[[ "$(grep -c '"sym":"gemv_qkvg_m' "$TMP/out/pass.jsonl")" -eq 40 ]]
grep -q '"sym":"gemv_mxfp4_m16"' "$TMP/out/pass.jsonl"

before=$(grep -c '^ingest$' "$MOCK_LOG")
if MOCK_MISSING_RUNG=1 run_campaign missing >"$TMP/missing.log" 2>&1; then
  echo "FAIL: missing QKVG rung was ingested" >&2
  exit 1
fi
grep -q 'sweep missed 1 demanded rung sample' "$TMP/missing.log"
if MOCK_BAD_SAMPLE=1 run_campaign bad >"$TMP/bad.log" 2>&1; then
  echo "FAIL: incorrect demanded sample was ingested" >&2
  exit 1
fi
grep -q 'demanded sample failed correctness' "$TMP/bad.log"
if MOCK_DUPLICATE=1 run_campaign duplicate >"$TMP/duplicate.log" 2>&1; then
  echo "FAIL: duplicate demanded sample was ingested" >&2
  exit 1
fi
grep -q 'duplicate sweep sample' "$TMP/duplicate.log"
if MOCK_UNEXPECTED=1 run_campaign unexpected >"$TMP/unexpected.log" 2>&1; then
  echo "FAIL: unexpected sweep row was ingested" >&2
  exit 1
fi
grep -q 'unexpected sweep symbol' "$TMP/unexpected.log"
if MOCK_WRONG_GPU=1 run_campaign wrong-gpu >"$TMP/wrong-gpu.log" 2>&1; then
  echo "FAIL: wrong physical GPU was accepted" >&2
  exit 1
fi
grep -q 'exactly one unique MI350X/MI355X' "$TMP/wrong-gpu.log"
if MOCK_CONTENDED=1 run_campaign contended >"$TMP/contended.log" 2>&1; then
  echo "FAIL: contended campaign passed" >&2
  exit 1
fi
grep -q 'sweep/lease failed rc=76' "$TMP/contended.log"
if MOCK_DIGEST_DRIFT=1 run_campaign drift >"$TMP/drift.log" 2>&1; then
  echo "FAIL: interpreter identity drift was ingested" >&2
  exit 1
fi
grep -q 'tuning cell/interpreter identity changed' "$TMP/drift.log"
[[ "$(grep -c '^ingest$' "$MOCK_LOG")" -eq "$before" ]]

mkdir "$TMP/obj-existing"
touch "$TMP/obj-existing/shared"
if TUNE_GPU=MI355X "$ROOT/scripts/gemv_campaign_lease.sh" "$TMP/obj-existing" "$TMP/out/existing.jsonl" x -- "$TMP/bin/emitter" >"$TMP/existing.log" 2>&1; then
  echo "FAIL: shared OBJ was accepted" >&2
  exit 1
fi
grep -q 'OBJ must be a fresh private path' "$TMP/existing.log"
echo "PASS: GEMV campaign derives complete current demand and fails closed before ingest"
