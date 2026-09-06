#!/usr/bin/env bash
# tune_block_sweep.sh — TIER 2 knob sweep on single-layer block assets.
#
#   scripts/tune_block_sweep.sh <matrix-file> [out.tsv]
#
# WHY THIS TIER EXISTS. Isolated kernel microbenches (runtime/tests/*.cu) do not
# predict the megakernel — scripts/tune_decode_sweep.sh documents one case
# (gemv_lab_h100 says row-blocking wins 1.4x, in context it loses) and a second
# was measured on sm_120a: the GEMV harness times one gemv_rows<16> as costing
# the SAME as two gemv_rows<8> (0.163 vs 0.163 ms) while in the megakernel
# -DGV_MM_MAX=16 is worth 41.17 -> 28.8 ms. A single-layer BLOCK asset runs the
# real megakernel, so it keeps that context, and it reproduced the full-model
# ratio to 1.4% (1.45x vs 1.43x) at ~1/15th the cost.
#
# LAYER KINDS ARE NOT INTERCHANGEABLE. Gemma-4 is 40 sliding (hd 256, kvh 8) +
# 8 full (hd 512, kvh 1). A sliding-only block cannot score PLOW_NS_FULL_ABS at
# all — the emitter filters it on `gemv_family && full`. So we time ONE block per
# kind and score the model as the kind-weighted sum:
#
#     score_us = N_SLIDE * L_slide + N_FULL * L_full
#
# where L is the MARGINAL per-layer cost. A block asset also declares the
# embedding/lm_head weights, so a 1-layer block carries a fixed overhead O;
# differencing a 2-layer against a 1-layer block gave O = 15.08 us against
# L_slide = 530.38 us on Gemma-4-12B — 2.8%, and constant across knobs, so it
# cancels in a RANKING. The absolute score also omits the lm_head GEMV the full
# model runs once per token (~1.6 ms bf16 at 262144 vocab); that too is constant.
# Treat the score as a comparator, not a predicted TPOT.
#
# COST MODEL. A block run is ~3.3 s but an nvcc build is ~90 s, so compilation
# dominates: builds go WIDE (parallel, bounded by JOBS) and runs go serial (one
# GPU). That inversion is the whole reason this script is shaped this way.
#
# MATRIX FILE: one config per line, `name<TAB>-Dfoo=1 -Dbar=2<TAB>ENV=1 ENV2=x`.
# Blank lines and `#` comments ignored. A `baseline` row is a good idea.
#
# TWO AXES, because not every knob is a -D. The prefill BUCKET ladder is picked
# at run time by GpuEngine::pick_prefill_bucket (a padded-row cost model with a
# per-launch penalty), and its policy knobs — PLOW_PF_COVER, PLOW_PF_CHUNK_COST —
# are environment, not compile-time. Column 3 sets those, and configs that differ
# only in env REUSE one cubin build instead of paying 90 s to rebuild it.
#
# NOT TUNABLE HERE: segmented dispatch (PLOW_NV_SEGMENTS / PLOW_NV_SEG_GEMM,
# the "switch interpreter / wave size" objects). The serve path requires a
# single coarse segment — `check_coarse_single_segment()` — and a segmented
# bucket DISABLES prefill outright (exec/gpu.rs: "Absent cubin, a segmented
# bucket, or a missing GQ appendix disables prefill"). PLOW_UNISEG=1 is
# mandatory on sm_120 for that reason, so those objects are unreachable from
# serve and scoring them here would measure a path nothing runs.
set -euo pipefail

ROOT="${PLOW_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORK="${WORK:-/dev/shm/block-sweep}"
MATRIX="${1:?usage: tune_block_sweep.sh <matrix-file> [out.tsv]}"
OUT="${2:-$WORK/results.tsv}"
JOBS="${JOBS:-6}"

# Block assets, one per layer kind. Emit them with the SAME decode batch you
# intend to serve: --batch selects active slots, NOT kernel width.
SLIDE_ASSET="${SLIDE_ASSET:-/root/plow-out/blk-slide}"
FULL_ASSET="${FULL_ASSET:-/root/plow-out/blk-full}"
N_SLIDE="${N_SLIDE:-40}"
N_FULL="${N_FULL:-8}"
BATCH="${BATCH:-16}"
CTX="${CTX:-1024}"
ITERS="${ITERS:-60}"

# Defines every arm carries (the committed sm_120 decode recipe). Sweep deltas
# are appended after these, so a later -D wins.
BASE_D="-DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_FA_GF_FULL=4 -DPLOW_NV_EMBED_SMEM=1 -DPLOW_FP8_KV=1 -DGV_MM_MAX=16"

NVCC=/usr/local/cuda/bin/nvcc
SRC="$ROOT/runtime/nvidia/interp_sm120.cu"
BLOCK_RUN="$ROOT/target/release/examples/block_run"
mkdir -p "$WORK" "$(dirname "$OUT")"

[ -x "$BLOCK_RUN" ] || { echo "FATAL: build it first: cargo build --release -p plowrt --features cuda --example block_run" >&2; exit 1; }
for a in "$SLIDE_ASSET" "$FULL_ASSET"; do
  [ -f "$a/model.pkt" ] || { echo "FATAL: no block asset at $a — emit with plowc --block" >&2; exit 1; }
done

names=(); defs=(); envs=()
# Split on tabs MANUALLY. `IFS=$'\t' read -r n d e` collapses consecutive tabs
# (tab is IFS whitespace), so a row with an empty defines column silently
# shifts the env column into it — nvcc then gets `FOO=1` as an input file.
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue;; esac
  n="${line%%$'\t'*}"
  rest="${line#*$'\t'}"; [ "$rest" = "$line" ] && rest=""
  d="${rest%%$'\t'*}"
  e="${rest#*$'\t'}"; [ "$e" = "$rest" ] && e=""
  names+=("$n"); defs+=("$d"); envs+=("$e")
done < "$MATRIX"
echo "sweep: ${#names[@]} configs, ${JOBS}-way build, block=${BATCH}x${CTX}"

# ---- phase 1: build every decode object, WIDE ----------------------------
declare -A BUILT
pids=()
for i in "${!names[@]}"; do
  key="D:${defs[$i]}"   # prefixed: bash assoc arrays reject an empty subscript
  # configs differing only in ENV share a cubin — do not rebuild for them
  if [ -n "${BUILT[$key]+x}" ]; then
    ln -sf "$WORK/${BUILT[$key]}.cubin" "$WORK/${names[$i]}.cubin"; continue
  fi
  BUILT[$key]="${names[$i]}"
  (
    env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin "$NVCC" -arch=sm_120a -O3 -cubin \
      -I "$ROOT/runtime/common" -I "$ROOT/runtime/nvidia" \
      $BASE_D ${defs[$i]} -o "$WORK/${names[$i]}.cubin" "$SRC" 2>"$WORK/${names[$i]}.log" \
      || echo "BUILD FAILED: ${names[$i]} (see $WORK/${names[$i]}.log)" >&2
  ) &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
done
wait
echo "built/linked $(for n in "${names[@]}"; do [ -e "$WORK/$n.cubin" ] && echo x; done | wc -l)/${#names[@]}"

# ---- phase 2: time each, SERIAL (one GPU) --------------------------------
run_one() { # <asset> <cubin> <env> -> "<decode_us> <prefill_ms>"
  cp "$2" "$1/interp_sm120.cubin"
  env PLOW_LIBCUDA="${PLOW_LIBCUDA:-/usr/lib/x86_64-linux-gnu/libcuda.so.1}" PLOW_UNISEG=1 ${3:-} \
    "$BLOCK_RUN" "$1" bench --batch "$BATCH" --ctx "$CTX" --iters "$ITERS" --warmup 10 2>/dev/null \
    | awk '/decode median=/{d=$0} END{
        match(d,/decode median= *[0-9.]+/); dv=substr(d,RSTART,RLENGTH); sub(/.*= */,"",dv);
        match(d,/prefill median= *[0-9.]+/); pv=substr(d,RSTART,RLENGTH); sub(/.*= */,"",pv);
        print dv, pv }'
}

printf 'name\tslide_us\tfull_us\tscore_us\tpf_slide_ms\tdefines\tenv\n' > "$OUT"
for i in "${!names[@]}"; do
  cb="$WORK/${names[$i]}.cubin"; [ -f "$cb" ] || continue
  read -r s ps <<<"$(run_one "$SLIDE_ASSET" "$cb" "${envs[$i]}")"
  read -r f pf <<<"$(run_one "$FULL_ASSET"  "$cb" "${envs[$i]}")"
  [ -n "$s" ] && [ -n "$f" ] || { echo "RUN FAILED: ${names[$i]}" >&2; continue; }
  sc=$(python3 -c "print(f'{$N_SLIDE*$s + $N_FULL*$f:.1f}')")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${names[$i]}" "$s" "$f" "$sc" "${ps:-na}" "${defs[$i]}" "${envs[$i]}" >> "$OUT"
  echo "  ${names[$i]}: slide=${s}us full=${f}us score=${sc}us prefill=${ps:-na}ms"
done

echo; echo "=== ranked (lower score is better) ==="
{ head -1 "$OUT"; tail -n +2 "$OUT" | sort -t$'\t' -k4 -g; } | awk -F'\t' '{printf "%-14s %-11s %-10s %-11s %-12s %s %s\n",$1,$2,$3,$4,$5,$6,$7}'
