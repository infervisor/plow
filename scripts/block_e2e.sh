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
#   CUBIN_DIR   OPTIONAL fallback objdir. `--emit devblob+cubin` builds this block's own
#               objects from its manifest, which is what the loader requires: interpreter
#               objects are HASH-PINNED to the packet and a cubin specialised for a
#               different one is refused at load. Only used for objects the emit did not
#               produce, and a cubin from another packet will be rejected.
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

export LD_LIBRARY_PATH="${PLOW_CUDA_LIBS:-/usr/local/cuda-13.2/targets/x86_64-linux/lib:/usr/local/cuda/lib64}:${LD_LIBRARY_PATH:-}"
export PLOW_LIBCUDA="${PLOW_LIBCUDA:-$(ls -1 /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)}"

# The DEFAULT segment-class policy faults this packet: the first prefill launch dies with
# CUDA_ERROR_LAUNCH_FAILED (719) at bucket_t=128. Measured PURE=1 runs (FA512 either way),
# PURE=0 and the default both fault. Every working 26B serve config in this campaign sets it.
export PLOW_PF_SEG_PURE="${PLOW_PF_SEG_PURE:-1}"

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
nix develop -c ./target/release/plowc --hf-dir "$MODEL" --emit devblob+cubin --arch sm_90a \
  --max-ctx "$MAXCTX" --block "$LAYER" --out "$ASSET" || exit 1

# ------------------------------------------------------------- 3. assemble ---
# GpuEngine::load finds the blob in the dir, reads block.json, and takes the
# checkpoint from <asset>/checkpoint (or $PLOW_CHECKPOINT). The interpreter and
# prefill cubins are picked up by their default filenames.
say "3/5 assemble asset dir"
# `--emit devblob+cubin` builds the base interpreter objects only. It does NOT build the Gemma-4
# ROLE objects (interp_sm90a_pfattn_*, _pffa, _pfpackedfa*, attn_softmax_sm90a) or the MoE Lt glue,
# even though this packet's `union` names FlashPrefill/hd256 -- so the prefill seg graph dispatches
# a flash segment with no object carrying that arm and the first launch dies with
# CUDA_ERROR_LAUNCH_FAILED (719). Those objects are HASH-PINNED (build.json `pairing`), so they
# cannot be borrowed from another packet's objdir: they must be built from THIS packet's
# plow_config.h, which is exactly what build_sm90a_gemma4_segments.sh does.
for c in interp_sm90a.cubin interp_sm90a_pf.cubin; do
  [ -e "$ASSET/$c" ] || { echo "emit produced no $c" >&2; exit 2; }
done
say "3b/5 build this packet's Gemma-4 role objects"
PLOW_CUBIN_CONFIG="$ASSET/plow_config.h" \
  bash "$REPO/scripts/build_sm90a_gemma4_segments.sh" "$ASSET" "$ASSET/objects" || exit 1
export PLOW_PF_SEG_DIR="$ASSET/objects"
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
