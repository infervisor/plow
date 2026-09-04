#!/usr/bin/env bash
# k3_tp_equivalence.sh — THE SAME K3 ASSET AT tp=1 AND tp=8 MUST PRODUCE THE SAME LOGITS.
#
# WHAT THIS CATCHES THAT NOTHING ELSE IN THE TREE DOES.
#
# Tensor parallelism is the one axis every other K3 gate is blind to. The three real-weight rung
# gates (`k3_block`, `k3_moe_block`, `k3_mla_block`) are single-GPU and single-layer. The op gates
# (`k3_attn_res_oracle`, `k3_fuse_exact`, `gemv_qkvg`, `kda_block`) are single-GPU by construction.
# `plowrt/tests/multi_gpu.rs` runs on 8 GPUs and dispatches no kernel at all. So the entire
# collective path — who writes which peer slot, at which offset, in what order — had NO numeric
# check anywhere, and that is exactly where K3's bring-up bug lived:
#
#   `emit_k3_latent_moe`'s shared-expert `down_proj` wrote an ordinary arena tensor instead of the
#   peer slot `d_xreduce` sums out of. The reduce therefore returned whatever was at slot 0, which
#   is `act.og_tp` — the SAME LAYER's `o_proj` partial. Every MoE layer computed
#   `up_latent + attn` instead of `up_latent + shared_expert`, on 92 of 93 layers, and the shared
#   expert (2 x 3072 intermediate) was discarded entirely. No fault, no NaN, no missing weight:
#   stable plausible logits and a model that greedily emits ',' then ' ' forever.
#
# A peer-slot contract violation is invisible to every per-op gate, every per-layer fixture and
# every tp=1 run. This script is the control that sees it. Measured discrimination, real weights,
# `PLOW_LAYERS=2`, cosine of the full 163840-wide logit vector:
#
#            tp1 vs tp8 cos      argmax
#   BROKEN      0.946582         64052 vs 2336        (prefill)
#               0.453694         43414 vs 11379       (first decode step)
#   FIXED       0.999986         equal
#               0.999985         equal
#
# — four orders of magnitude in `1 - cos`. The default floor of 0.9999 sits between them with
# enormous margin in both directions.
#
# WHY `PLOW_LAYERS=2`, AND DO NOT "OPTIMISE" IT DOWN TO 1.
#
# Layer 0 is KDA + a DENSE MLP (`first_k_dense_replace = 1`) and has no shared expert and no
# routed experts, so it exercises no MoE collective. Layer 1 is the first latent MoE. That is not
# a detail, it is the whole gate: at N=1 the broken tree and the fixed tree BOTH score 0.999993,
# because there is nothing for the bug to corrupt. A future reader who shortens this to N=1 to
# save 15 seconds will leave a green test that cannot fail. N=1 is kept in the default list on
# purpose — as the NEGATIVE CONTROL that localises a failure to the mixer/dense half rather than
# the MoE half — but `2` is mandatory and this script refuses to run without it.
#
# WHY THE LOGIT VECTOR AND NOT THE ARGMAX. Argmax agreement is a weak signal in both directions: a
# near-tie flips on benign reassociation, and a broken model can still agree on a dominant token.
# The 5-layer prefill-vs-decode data from the same bring-up had `cos 0.939` with the argmax
# ALREADY differing — the token said "different" long before it said how much. So the gate is a
# cosine floor on the full vector, plus argmax equality, and it reports both.
#
# WHAT IT DELIBERATELY DOES NOT COVER. `PLOW_LAYERS=4` adds the first MLA layer (0-based 3) and its
# fp8 KV cache, whose quantisation differs between the tp=1 and tp=8 FlashMLA geometries
# (`nsplit`/`gf` are functions of the LOCAL head count). Measured after the fix: mostly 0.99997,
# but one step at 0.990 — real and benign, and a floor loose enough to admit it is too loose to be
# a gate. Add `4` to the layer list with `PLOW_K3_COS=0.985` if you want it; it is not the default.
#
# BOTH SIDES RUN THE DECODE PROGRAM. Both blobs are emitted `K3_PREFILL=0`, so a multi-token
# prompt is walked through the decode program one token at a time on both sides
# (`AmdEngine`/`AmdTpGroup`'s decode-only arm). The ONLY difference between the two runs is the
# degree. A prefill ladder on one side and not the other would confound TP with the GEMM/GEMV seam.
#
# COST AND PREREQUISITES. Real weights only — there is nothing to compare on unbound weights.
# `PLOW_LAYERS=2` is ~20 GiB of checkpoint: at tp=8 that is ~2.6 GiB a rank, at tp=1 it is ~20 GiB
# on GPU 0 alone, so the tp=1 side is what sets the free-VRAM requirement. About 60 s a side
# including emit, load and bind; ~3 minutes for the default two-depth sweep. Every prerequisite is
# a clean SKIP (exit 0) rather than a failure, so a CI wrapper may call this unconditionally on a
# box that has neither 8 GPUs nor 1.5 TB of Kimi-K3.
#
#   nix develop --command scripts/k3_tp_equivalence.sh
#   nix develop --command env PLOW_K3_EQ_SHARD_HEAD=1 scripts/k3_tp_equivalence.sh
#   nix develop --command env PLOW_LAYERS=1,2,4 PLOW_K3_COS=0.985 scripts/k3_tp_equivalence.sh
#
# Env: PLOW_K3_CKPT (default the k3_farm symlink farm, else the HF snapshot), PLOW_K3_HSACO
# (/home/lava/models/k3_mi325x/hsaco), PLOW_LAYERS (1,2), PLOW_K3_COS (0.9999), PLOW_K3_STEPS (3),
# PLOW_K3_PROMPT ("The capital of France is"), PLOW_K3_OUT (a tmpdir), PLOW_K3_MIN_FREE_GIB (28),
# PLOW_K3_ARCH (gfx942), PLOW_K3_GPU (MI325X), PLOW_K3_NCU (304), PLOW_K3_VOCAB (163840),
# PLOW_K3_EQ_SHARD_HEAD (0; when 1, shard only the tp=8 head and gate its global greedy ids against
# the tp=1 full-vocabulary control while comparing rank 0's local logits to control slice 0).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/nix_rocm_714.sh"
plow_init_rocm_714

skip() { echo "SKIP: $*"; exit 0; }
fail() { echo "FAIL: $*"; exit 1; }

CK="${PLOW_K3_CKPT:-/home/lava/models/k3_farm}"
HS="${PLOW_K3_HSACO:-/home/lava/models/k3_mi325x/hsaco}"
LEASE="${PLOW_GPULEASE_BIN:-$ROOT/perf-data/tools/gpulease}"
PLOWC="${PLOW_K3_PLOWC:-$ROOT/target/release/plowc}"
PLOWRT="${PLOW_K3_PLOWRT:-$ROOT/target/release/plowrt}"
STEPS="${PLOW_K3_STEPS:-3}"
COS="${PLOW_K3_COS:-0.9999}"
PROMPT="${PLOW_K3_PROMPT:-1008,10484,318,15383,387}"   # "The capital of France is"
OUT="${PLOW_K3_OUT:-/tmp/k3_tp_equiv}"
MIN_FREE_GIB="${PLOW_K3_MIN_FREE_GIB:-28}"
ARCH="${PLOW_K3_ARCH:-gfx942}"
GPU="${PLOW_K3_GPU:-MI325X}"
NCU="${PLOW_K3_NCU:-304}"
VOCAB="${PLOW_K3_VOCAB:-163840}"
SHARD_HEAD="${PLOW_K3_EQ_SHARD_HEAD:-0}"
case "$SHARD_HEAD" in 0|1) ;; *) fail "PLOW_K3_EQ_SHARD_HEAD must be 0 or 1" ;; esac
compare_extra=()
if [ "$SHARD_HEAD" = 1 ]; then compare_extra+=(--sharded8); fi
IFS=, read -ra NLAYERS <<< "${PLOW_LAYERS:-1,2}"

# --- prerequisites, every one a clean skip ---------------------------------------------------

# An EXPLICIT PLOW_K3_CKPT that does not exist is a skip, not a silent fallback: falling back
# would run the gate against a different checkpoint than the one asked for and report PASS.
if [ -n "${PLOW_K3_CKPT:-}" ]; then
  [ -d "$CK" ] || skip "PLOW_K3_CKPT=$CK does not exist"
else
  [ -d "$CK" ] || CK="$(ls -d "$HOME"/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/*/ 2>/dev/null | head -1 || true)"
  [ -n "$CK" ] && [ -d "$CK" ] || skip "no Kimi-K3 checkpoint (set PLOW_K3_CKPT); this gate is weight-dependent by nature"
fi
ls "$CK"/*.safetensors >/dev/null 2>&1 || skip "$CK has no safetensors shards"
[ -x "$PLOWC" ]  || skip "$PLOWC absent — cargo build --release -p plowc"
[ -x "$PLOWRT" ] || skip "$PLOWRT absent — cargo build --release -p plowrt --features hsa"
[ -d "$HS" ] || skip "$HS absent — see docs/BUILD.md for the AMD objects"
[ -x "$LEASE" ] || skip "$LEASE absent — repository GPU lease helper is required"
command -v rocm-smi >/dev/null 2>&1 || skip "rocm-smi absent — cannot count GPUs"

NGPU="$(rocm-smi --showid 2>/dev/null | grep -oE '^GPU\[[0-9]+\]' | sort -u | wc -l)"
[ "$NGPU" -ge 8 ] || skip "$NGPU GPU(s) visible, this gate needs 8 (the tp=8 side is the point)"

# The tp=1 side puts the whole truncated model on GPU 0, so GPU 0's headroom is the binding
# constraint. Reported rather than assumed: a box running something else is a SKIP, not a
# mysterious `hsa_amd_memory_pool_allocate` failure three minutes in.
FREE_GIB="$(rocm-smi --showmeminfo vram 2>/dev/null | awk '
  /GPU\[0\].*VRAM Total Memory/  { t=$NF }
  /GPU\[0\].*VRAM Total Used/    { u=$NF }
  END { if (t) printf "%d", (t-u)/1073741824 }')"
[ -n "$FREE_GIB" ] || skip "could not read GPU 0 VRAM from rocm-smi"
[ "$FREE_GIB" -ge "$MIN_FREE_GIB" ] || \
  skip "GPU 0 has ${FREE_GIB} GiB free, the tp=1 side needs ~${MIN_FREE_GIB} (something else is resident)"

# `2` is the depth that contains the first latent MoE and therefore the first shared expert. See
# the header: without it this script is green by construction.
printf '%s\n' "${NLAYERS[@]}" | grep -qx 2 || \
  fail "PLOW_LAYERS must include 2 — N=1 has no MoE layer and cannot fail. Read the header."

echo "checkpoint  $CK"
echo "objects     $HS"
echo "depths      ${NLAYERS[*]}   steps $STEPS   cos floor $COS"
echo "target      $ARCH / $GPU / $NCU CUs"
echo "GPU 0 free  ${FREE_GIB} GiB of ${NGPU} GPUs visible"
echo

rm -rf "$OUT"; mkdir -p "$OUT"
bad=0

for N in "${NLAYERS[@]}"; do
  for G in 1 8; do
    b="$OUT/blob_${N}_${G}"; l="$OUT/log_${N}_${G}"
    mkdir -p "$b" "$l"
    # K3_PREFILL=0 on BOTH sides: one program, walked a token at a time, so the degree is the
    # only difference. `--num-gpus` is what makes the emit sharded; there is no `--tp` on plowc.
    shard=0
    if [ "$SHARD_HEAD" = 1 ] && [ "$G" = 8 ]; then shard=1; fi
    K3_FULL=1 PLOW_LAYERS="$N" K3_PREFILL=0 PLOW_FP8_KV=1 PLOW_MXFP4=1 \
      PLOW_K3_SHARD_HEAD="$shard" \
      "$PLOWC" --hf-dir "$CK" --emit devblob --arch "$ARCH" --gpu "$GPU" \
      --num-gpus "$G" --parallel tp --max-ctx 4096 --n-cu "$NCU" --out "$b" 2>&1 \
      | grep -aE "^kimi_k3: emitted" \
      || fail "N=$N tp=$G: emit failed"
    cat >"$b/weights.json" <<JSON
{
  "network": "k3-tp-equiv-n${N}-tp${G}",
  "gpu": "$GPU",
  "num_gpus": $G,
  "parallel": "tp",
  "weight_shared": false,
  "weight": null,
  "kv": null,
  "fusion": null,
  "buckets": [],
  "static_tensors": [],
  "static_tensors_file_emitted": false,
  "weight_tiling": null
}
JSON
    report="$OUT/run_${N}_${G}.json"
    log="$OUT/run_${N}_${G}.txt"
    run_rc=0
    "$LEASE" -n "$G" "k3-tp-equiv-n${N}-tp${G}" \
      "$PLOWRT" --rt-checkpoint "$CK" --rt-hsaco "$HS" \
      --amd-tens-snap "$l" --amd-snap-tensors act.logits --amd-snap-slot 0 \
      --amd-tp-agree-every 1 --amd-tp-no-audit=false \
      bench --assets "$b" --prompt-ids "$PROMPT" --concurrency 1 --requests 1 \
      --warmup-requests 0 --output-len "$((STEPS + 1))" \
      --token-audit --engine-diagnostics >"$report" 2>"$log" || run_rc=$?
    [ "$run_rc" -eq 0 ] || { tail -40 "$log"; fail "N=$N tp=$G: leased production run exited $run_rc"; }
  done

  python3 "$ROOT/scripts/k3_tp_equivalence_compare.py" \
    --report1 "$OUT/run_${N}_1.json" --snap1 "$OUT/log_${N}_1" --asset1 "$OUT/blob_${N}_1" \
    --packet1 "$OUT/blob_${N}_1/model.pkt" \
    --report8 "$OUT/run_${N}_8.json" --snap8 "$OUT/log_${N}_8" --asset8 "$OUT/blob_${N}_8" \
    --packet8 "$OUT/blob_${N}_8/model.pkt" --objects "$HS" --checkpoint "$CK" \
    --prompt "$PROMPT" --steps "$STEPS" --cos "$COS" --layers "$N" --vocab "$VOCAB" \
    "${compare_extra[@]}" || bad=1
  echo
done

if [ "$bad" -ne 0 ]; then
  echo "FAIL — tp=1 and tp=8 do not compute the same model"
  exit 1
fi
echo "PASS — tp=1 and tp=8 agree on every dumped logit vector at depths ${NLAYERS[*]}"
