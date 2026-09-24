#!/usr/bin/env bash
# scripts/block_e2e.sh — end-to-end single-block comparison: plow vs vLLM.
#
# Compiles ONE block out of a real checkpoint, benches it on plow, benches the
# SAME block through vLLM's own decoder layer, and diffs the two sweeps.
#
#   ./scripts/block_e2e.sh <model-dir> <block-config.json> [layer] [max_ctx]
#
# e.g.
#   ./scripts/block_e2e.sh /workspace/models/gemma-4-26B-A4B-it \
#       perf-data/block-configs/gemma4-26b-a4b-moe.json 0 2048
#
# Every GPU step runs under `gpulease`; a contended run (rc=76) is reported, not
# silently accepted. Env:
#   PLOW_PY     python with a CUDA torch + vLLM (default: /opt/pytorch/bin/python)
#   CUBIN_DIR   sm90a interpreter cubins — an objdir built by
#               scripts/build_sm90a_gemma4_segments.sh, or any packet's objects/ dir.
#               No default: there is no canonical location, so it is checked, not guessed.
#   BATCH/CTX   sweep grid (default 1,4 and 128,1024)
#   OUT         work dir (default /dev/shm/block-e2e)
#
# PREREQUISITE — the cubins and the torch build must match the installed driver.
# The vLLM half needs ninja on PATH and must NOT run inside `nix develop` (the
# engine core dies at startup there), which is why only the build steps below
# use `nix develop -c`.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?usage: block_e2e.sh <model-dir> <block-config.json> [layer] [max_ctx]}"
BCFG="${2:?usage: block_e2e.sh <model-dir> <block-config.json> [layer] [max_ctx]}"
LAYER="${3:-0}"
MAXCTX="${4:-2048}"

PY="${PLOW_PY:-/opt/pytorch/bin/python}"
CUBIN_DIR="${CUBIN_DIR:-}"
GPULEASE="${GPULEASE:-$REPO/perf-data/tools/gpulease}"
BATCH="${BATCH:-1,4}"
CTX="${CTX:-128,1024}"
OUT="${OUT:-/dev/shm/block-e2e}"
export GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-5400}"

ASSET="$OUT/asset"
mkdir -p "$ASSET"
cd "$REPO"

say() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------- 1. build ---
say "1/5 build plowc + plowrt:block_run"
nix develop -c cargo build --release -p plowc --bin plowc || exit 1
nix develop -c cargo build --release -p plowrt --features cuda --example block_run || exit 1

# -------------------------------------------------------------- 2. compile ---
# `plowc --block L` emits the PLOWDEV blob AND a sibling block.json into --out,
# which is a DIRECTORY (the deprecated `gemma4` bin took a .pkt path instead).
say "2/5 compile block L$LAYER from $MODEL (max_ctx=$MAXCTX)"
./target/release/plowc --hf-dir "$MODEL" --emit devblob --arch sm_90a \
  --max-ctx "$MAXCTX" --block "$LAYER" --out "$ASSET" || exit 1

# ------------------------------------------------------------- 3. assemble ---
# GpuEngine::load finds the blob in the dir, reads block.json, and takes the
# checkpoint from <asset>/checkpoint (or $PLOW_CHECKPOINT). The interpreter and
# prefill cubins are picked up by their default filenames.
say "3/5 assemble asset dir"
[ -n "$CUBIN_DIR" ] || { echo "set CUBIN_DIR to an sm90a objdir (see header)" >&2; exit 2; }
for c in interp_sm90a.cubin interp_sm90a_pf.cubin; do
  [ -f "$CUBIN_DIR/$c" ] || { echo "CUBIN_DIR missing $c" >&2; exit 2; }
done
ln -sf "$CUBIN_DIR/interp_sm90a.cubin"    "$ASSET/interp_sm90a.cubin"
ln -sf "$CUBIN_DIR/interp_sm90a_pf.cubin" "$ASSET/interp_sm90a_pf.cubin"
ln -sfn "$MODEL" "$ASSET/checkpoint"
ls -l "$ASSET"

# ------------------------------------------------------------ 4. plow bench ---
say "4/5 plow: block_run bench"
"$GPULEASE" "e2e-plow-L$LAYER" \
  ./target/release/examples/block_run "$ASSET" bench \
    --batch "$BATCH" --ctx "$CTX" --iters 100 --warmup 20 --prefill-iters 10 \
    --out-dir "$OUT/bench"
rc=$?
[ "$rc" = 76 ] && echo "*** plow run CONTENDED (rc=76) — re-run before trusting ***" >&2
[ "$rc" != 0 ] && [ "$rc" != 76 ] && { echo "plow bench failed rc=$rc" >&2; exit "$rc"; }
PLOW_SWEEP="$OUT/bench/sweep.json"

# --------------------------------------------------------- 5. vLLM baseline ---
say "5/5 vLLM: same block through its own decoder layer"
"$GPULEASE" "e2e-vllm-L$LAYER" \
  env PATH=/opt/pytorch/bin:"$PATH" "$PY" "$REPO/scripts/block_layer_bench.py" "$BCFG" \
    --batch "$BATCH" --ctx "$CTX" --iters 100 --warmup 20 --prefill-iters 10 \
    --out "$OUT/vllm.json"
rc=$?
[ "$rc" = 76 ] && echo "*** vLLM run CONTENDED (rc=76) — re-run before trusting ***" >&2
[ "$rc" != 0 ] && [ "$rc" != 76 ] && { echo "vllm bench failed rc=$rc" >&2; exit "$rc"; }

# ------------------------------------------------------------------ compare ---
for phase in decode prefill; do
  say "compare ($phase)"
  python3 "$REPO/scripts/block_compare.py" --plow "$PLOW_SWEEP" \
    --baseline "$OUT/vllm.json" --phase "$phase" \
    --json "$OUT/compare-$phase.json"
done
echo
echo "artifacts in $OUT"
