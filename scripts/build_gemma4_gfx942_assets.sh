#!/usr/bin/env bash
# Reproduce a packet-stamped Gemma-4 BF16 or FP8/W8A8 MI300X serving bundle.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hf_dir="${1:?usage: $0 <hf-dir> <output-dir> <bf16|fp8>}"
output="${2:?usage: $0 <hf-dir> <output-dir> <bf16|fp8>}"
precision="${3:?usage: $0 <hf-dir> <output-dir> <bf16|fp8>}"

[ -n "${IN_NIX_SHELL:-}" ] || {
  echo "FAIL: run through nix develop --command" >&2
  exit 2
}
[ -f "$hf_dir/config.json" ] || { echo "FAIL: $hf_dir/config.json is missing" >&2; exit 2; }
hf_dir="$(cd "$hf_dir" && pwd)"
case "$precision" in bf16|fp8) ;; *) echo "FAIL: precision must be bf16 or fp8" >&2; exit 2;; esac
[ ! -e "$output" ] || { echo "FAIL: output already exists: $output" >&2; exit 2; }
[ -z "$(git -C "$repo" status --porcelain)" ] || {
  echo "FAIL: reproducible assets require a clean source tree" >&2
  exit 2
}

cargo build -p plowc --release

mkdir -p "$(dirname "$output")"
stage="$(mktemp -d "$(dirname "$output")/.gemma4-gfx942.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/assets"

precision_env=()
if [ "$precision" = fp8 ]; then
  precision_env=(PLOW_FP8=1 PLOW_W8A8=1)
fi

env \
  PLOW_DECODE_BATCH=8 \
  PLOW_DECODE_BATCH_LADDER=1,2,4,8 \
  PLOW_DENSE_PF_NS=1 \
  PLOW_EMIT_PACKED_PREFILL=1 \
  PLOW_L2_PLACE_PREFILL=0 \
  PLOW_MAX_CHUNK=8192 \
  "${precision_env[@]}" \
  "$repo/target/release/plowc" \
    --hf-dir "$hf_dir" --gpu MI300X --arch gfx942 --n-cu 304 \
    --max-ctx 16512 --emit devblob --out "$stage/assets"

python3 "$repo/scripts/check_gemma4_gfx942_assets.py" \
  "$stage/assets/build.json" "$precision"

PLOW_HSACO_CONFIG="$stage/assets" \
  "$repo/scripts/build_gfx942.sh" "$stage/hsaco"

for rung in 1 2 4; do
  [ -f "$stage/hsaco/lowrung$rung/interp_decode.elf" ] || {
    echo "FAIL: decode tier object lowrung$rung/interp_decode.elf is missing" >&2
    exit 2
  }
done

git -C "$repo" rev-parse HEAD > "$stage/source-commit.txt"
( cd "$hf_dir" && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum ) \
  > "$stage/CHECKPOINT_SHA256SUMS"
( cd "$stage" && {
    find assets hsaco -type f -print0 | sort -z | xargs -0 sha256sum
    sha256sum source-commit.txt CHECKPOINT_SHA256SUMS
  } > SHA256SUMS )
mv "$stage" "$output"
trap - EXIT
echo ">>> reproducible bundle: $output"
