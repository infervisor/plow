#!/usr/bin/env bash
# glm52_tpctx_sweep.sh — GLM-5.2, APPLES-TO-APPLES context sweep to 128k, TP2/TP4/TP8,
# plow vs vLLM, through ONE client (`vllm bench serve`, §0-BENCH).
#
#   ./scripts/glm52_tpctx_sweep.sh emit                 # plowc only. NO GPU, NO lease.
#   ./scripts/glm52_tpctx_sweep.sh tune <objdir>        # gemm tile campaign + gate. 1 GPU.
#   perf-data/harness/gpulease -n 4 glm-ctx4 sg render -c \
#       './scripts/glm52_tpctx_sweep.sh run 4'          # coherence -> plow -> vLLM, TP4
#   perf-data/harness/gpulease -n 8 glm-ctx8 sg render -c \
#       './scripts/glm52_tpctx_sweep.sh run 8'          # ditto, TP8 (needs ALL EIGHT cards)
#   ./scripts/glm52_tpctx_sweep.sh emitns               # the 8-bundle GF x nsplit grid. NO GPU.
#   VARIANT=gf4ns64- perf-data/harness/gpulease -n 4 ns4 sg render -c \
#       './scripts/glm52_tpctx_sweep.sh nsrun 4'        # ONE nsplit arm: op-level + identity
#
# ============================================================================
# WHAT CHANGED FROM THE PREVIOUS VERSION OF THIS FILE, AND WHY
# ============================================================================
# The earlier `emit`/`shapes`/`run` harness had the TP2 non-fit and the DSA-off rule right,
# and both survive below. Four things made it unable to produce the table:
#
#  1. IT EMITTED ONE BLOB PER (TP, ctx) — 15 emits. `--max-ctx` sizes the KV ring and the
#     activation arena, and a shorter prompt runs on the same programs, so ONE blob per TP
#     at the top ctx serves the whole sweep. 3 emits, not 15.
#  2. ITS LADDER CARRIED A 16384 RUNG THAT CANNOT RUN. `MAX_CHUNK = 8192`
#     (`crates/plowrt/src/exec/amd.rs:718`) and `plan_chunks` filters every bucket above it,
#     so the rung cost `act.part` device memory at emit (T*top_k*hidden f32 = 3.2 GiB at
#     16384) and was then never dispatched.
#  3. IT HAD NO vLLM SIDE, so it could not produce an apples-to-apples row at all — only a
#     plow column. §0-BENCH needs the same client against both engines.
#  4. ITS BUNDLES HAD NO `weights.json`. `plowc --emit devblob` does not write one and
#     `plowrt serve` opens it unconditionally, so every `run` died during load with a bare
#     `Io { path: .../weights.json, NotFound }` that reads like a missing checkpoint.
#
# ============================================================================
# WHAT IS REACHABLE ON THIS BOX. Read before planning around the matrix.
# ============================================================================
#
# TP2 IS IMPOSSIBLE — CAPACITY, NOT DIVISIBILITY, AND NOT CONTEXT.
#   GLM-5.2 block-fp8 prepped checkpoint: 766,871,079,074 B = 714.2 GiB.
#   Card (MI355X, `rocm-smi --showmeminfo vram`): 309,220,868,096 B = 288.0 GiB.
#     TP1  714.2 GiB/rank   2.48x the card
#     TP2  357.1 GiB/rank   1.24x the card   <- 69 GiB over, WEIGHTS ALONE, at ctx 0
#     TP4  178.6 GiB/rank   fits (measured 183.08 GiB/rank live, incl. KV+activations)
#     TP8   89.3 GiB/rank   fits with room
#   Head divisibility is NOT the limit (64 attn heads -> 32/16/8 per rank at TP2/4/8, and
#   256 routed experts divide too — the `c.heads % tp == 0` guard at mla.rs:4009/:4115
#   passes at all three). There is no ctx at which TP2 fits, because the deficit is in the
#   WEIGHTS: dropping ctx to 0 removes KV, not the 69 GiB. TP2 needs a smaller checkpoint,
#   weight streaming, or EP-with-offload. None exist. TP2 is emitted here anyway (emit needs
#   no GPU and the blob is proof the compiler is not the blocker) but never run.
#
# CTX 131072 REQUIRES PLOW_GLM_DSA=0, AND THAT IS THE HEADLINE CAVEAT.
#   `GlmCfg::dsa` (crates/devgen/src/mla.rs:136) arms the DSA sparse indexer->select->gather
#   path at `ctx > CROSSOVER(65536)`. That path is recorded as producing DEGENERATE OUTPUT
#   and is unvalidated past ctx 2048 (knob-contract §6g-DSA; task #6 open). mla.rs:135 names
#   `PLOW_GLM_DSA=0` "the apples-to-apples decode baseline", so that is what this uses: DENSE
#   attention at every ctx, one dense curve with no discontinuity at 64k.
#
#   *** vLLM SERVES THE SAME CHECKPOINT WITH DSA ARMED. *** GLM-5.2's `index_topk` is 2048,
#   so above ~2k context vLLM's attention reads 2048 KV rows per layer per token and plow
#   reads ALL OF THEM — 64x more at 128k. plow is doing STRICTLY MORE WORK in every cell
#   above 2048. A plow win under that asymmetry is a real win; a plow loss is NOT evidence
#   about plow's kernels, and must never be reported without this paragraph attached.
#
# ONE BLOB PER TP, NOT ONE PER CELL.
#   `--max-ctx` sizes the KV ring and the activation arena; a prompt shorter than max_ctx
#   runs on the same programs. So a single ctx-131072 blob serves every point of the sweep
#   and the emit cost (139 ms fixed + 0.943 ms/row) is paid 3 times, not 15.
#
# THE PREFILL BUCKET LADDER IS CAPPED AT 8192 BY THE RUNTIME, NOT BY TASTE.
#   `MAX_CHUNK = 8192` (crates/plowrt/src/exec/amd.rs:718) and `plan_chunks` FILTERS OUT
#   every bucket above it, so a 16384 rung would cost device memory at emit
#   (`act.part` = T*top_k*hidden f32 = 1.6 GiB at T=8192, 3.2 at 16384) and then never be
#   dispatched. 8192 is therefore the widest USEFUL rung, and a 131072-token prompt is
#   16 launches whatever ladder is chosen. The lower rungs are near-free (the arena is sized
#   to the WIDEST bucket) and they buy tail coverage: `plan_chunks` charges every launch
#   LAUNCH_ROWS=416 rows, so covering a 4096-token prompt as 2x2048 (cost 4928) beats
#   1x8192 (cost 8608). Ladder: 128, 512, 2048, 8192.
# ============================================================================
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${GLM_SWEEP_DIR:-/home/lava/models/glm52_ctxsweep}"
CKPT="${PLOW_CKPT:-/home/lava/models/GLM-5.2-plow}"
OBJ="${PLOW_HSACO:-/home/lava/plow/build-amd/ctxsweep-objs}"
TOKZ_SRC="${TOKZ_SRC:-/home/lava/models/glm52_ttft/tokenizer.json}"
# 131072 IS THE PROMPT, NOT THE CONTEXT — and getting this wrong costs a whole run.
# The client asks for `--random-input-len 131072 --random-output-len 128`, so a request
# occupies 131,200 positions. At `--max-model-len 131072` vLLM rejects it outright ("you
# requested 131200 tokens") and plow's KV ring would be asked for rows past its end, which
# is the silent-corruption class this repo has hit three times. So BOTH engines are given
# 135168 (132k) and the sweep point stays a true 131072-token prompt on both sides.
MAXCTX="${MAXCTX:-135168}"
LADDER="${LADDER:-full:128,512,2048,8192}"
# The sweep points. 4096 anchors the short end (below it TTFT is launch-bound and the
# comparison is about the server, not the model); 65536 is the last ctx plow would run
# dense WITHOUT the DSA override, so it is the one point where the two engines' attention
# regimes are as close as this checkpoint allows.
CTXS="${CTXS:-4096 16384 32768 65536 131072}"
# PROMPT COUNT IS CTX-DEPENDENT, AND IT IS THE SAME NUMBER FOR BOTH ENGINES AT EACH CTX.
#
# plow's prefill cost is superlinear in prompt length — MEASURED at TP4 on this build:
# TTFT 4.77 s @ 4k, 26.1 s @ 16k, 72.3 s @ 32k, i.e. roughly n^1.5, which extrapolates to
# ~9 min per request at 131072. At a flat 8 prompts + 2 warm-ups the 128k point alone would
# run for an hour and a half and hold every card in the lease while it did.
#
# Cutting the sample count at long ctx costs almost nothing: TTFT at 32k already has a
# mean/median spread under 0.03% (72339.38 vs 72319.58 ms), so the estimator is not what is
# limiting these cells. What WOULD break the comparison is giving the two engines different
# counts, so `npr` is a pure function of ctx and both `bench_plowrt_serve.sh` and
# `bench_vllm_chat.sh` are handed the identical value.
npr () { case "$1" in 4096|16384) echo "${NPROMPT:-8}" ;; 32768) echo "${NPROMPT_MID:-4}" ;;
                      *) echo "${NPROMPT_LONG:-2}" ;; esac; }
nwarm () { case "$1" in 4096|16384) echo 2 ;; *) echo 1 ;; esac; }
OUTLEN="${OUTLEN:-128}"
CONC="${CONC:-1}"
PORT="${PORT:-8155}"
mkdir -p "$OUT"

# VARIANT names the bundle directory so the DSA-ON and DENSE arms of the same tree can sit
# side by side. Empty = the original `tp<N>` bundles, kept byte-identical so the published
# dense rows stay addressable. `dsa-` / `dense-` are the A/B pair emitted by `emitdsa`.
#
# THE TWO BUNDLES DIFFER IN ONE BIT AND NOTHING ELSE: `PLOW_GLM_DSA`. build.json does NOT
# record it (it is an emit-time env read inside `GlmCfg::dsa`), so the two directories are
# the only label — same trap as PLOW_GLM_GF in §6g-GF8. Do not rename them.
VARIANT="${VARIANT:-}"
bundle () { echo "$OUT/${VARIANT}tp$1"; }

emit_one () { # <tp>   ; DSA_ENV = "0" forces dense, "" arms the gate
  local tp="$1" b; b="$(bundle "$tp")"; mkdir -p "$b"
  echo "== emit TP$tp  max-ctx=$MAXCTX  ladder=$LADDER  PLOW_GLM_DSA=${DSA_ENV-0}" \
       "PLOW_GLM_GF=${GF_ENV:-<auto>} PLOW_GLM_NS=${NS_ENV:-<auto>}" \
       "GLM_NLAYERS=${NLAYERS:-<all 78>} objs=$OBJ"
  # NLAYERS truncates the emit to the first N layers (`glm_emit_full`, mla.rs) while keeping the
  # FULL serving structure and the TP sharding — it is a SEARCH vehicle, not a shipping blob.
  # The 4-minute 183 GiB/rank weight load is the entire cost of an arm (glm52_decode.c:224), and
  # an emit-time knob like PLOW_GLM_NS pays it once per arm; N=8 (3 dense + 5 MoE, since
  # `first_k_dense_replace=3`) cuts it ~10x and still exercises both layer kinds. Do NOT confuse
  # it with the single-layer bring-up path (GLM_FULL unset), which asserts tp == 1.
  env GLM_FULL=1 PLOW_FP8=1 GLM_SHARD_HEAD=1 GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 \
      PLOW_MLA_PREFILL="$LADDER" PLOW_GLM_DSA="${DSA_ENV-0}" \
      ${NLAYERS:+GLM_NLAYERS="$NLAYERS"} \
      ${GF_ENV:+PLOW_GLM_GF="$GF_ENV"} ${NS_ENV:+PLOW_GLM_NS="$NS_ENV"} \
    nix develop --command "$WT/target/release/plowc" --hf-dir "$CKPT" --emit devblob \
      ${NOROPEGEN:+--no-rope-gen} \
      --max-ctx "$MAXCTX" --n-cu 256 --num-gpus "$tp" --out "$b/model.pkt"
  ln -sfn "$CKPT"    "$b/checkpoint"
  ln -sfn "$OBJ"     "$b/hsaco"
  ln -sfn "$(readlink -f "$TOKZ_SRC")" "$b/tokenizer.json"
  # NOROPEGEN=1 adds `--no-rope-gen`, which BAKES the RoPE tables into the init section and keeps
  # the container at v5/v6. `plowrt` reads v7 and needs none of this; the C harnesses
  # (`glm52_decode`, `chat`) have NO table generator and REFUSE a v7 blob outright rather than
  # serve cos=sin=0 silently. So any bundle destined for an op-level microbench needs it, and any
  # bundle destined for `runarm`/`run` must NOT have it (it would no longer be the shipping form).
  #
  # `plowc --emit devblob` writes model.pkt + build.json but NOT weights.json, and
  # `plowrt serve` opens weights.json unconditionally — without it the server dies during
  # load with a bare `Io { path: .../weights.json, NotFound }` that reads like a missing
  # checkpoint. `num_gpus` here must agree with the packet's own tp.
  cat > "$b/weights.json" <<JSON
{
  "network": "glm-5.2",
  "gpu": "mi355x",
  "num_gpus": $tp,
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
  ls -l --time-style=+%H:%M "$b/model.pkt" | awk '{printf "   %s  %.1f MB\n", $NF, $5/1048576}'
}

case "${1:-}" in

emit)
  cd "$WT"
  nix develop --command cargo build --release -p plowc --bin plowc
  for tp in ${TPS:-4 8 2}; do emit_one "$tp"; done
  ;;

# ------------------------------------------------------------- DSA-ON / DENSE A/B PAIR
# Emits BOTH arms from ONE tree so the only difference between them is the DSA gate.
#
# WHY A FRESH DENSE ARM AND NOT THE PUBLISHED `tp<N>` BUNDLES: those were emitted at 20:32
# on 2026-07-28, i.e. BEFORE 3699ff1 changed `mla.rs` (the dispatch-width narrowing, worth
# -0.446 ms/token). Comparing a HEAD DSA bundle against them would fold that -0.446 into
# whatever DSA does. The pair below shares a tree, a plowc, and an object directory.
#
# WHAT DSA CAN AND CANNOT MOVE — read before reading any TTFT delta:
#   TTFT CANNOT MOVE. The MLA prefill emit is `FlashMlaPrefill` at EVERY ctx *including with
#   the DSA gate armed* (mla.rs:2335 and `_flash_gather_prefill_contract`): `FlashGatherPrefill`
#   has no emit site because nothing produces its `idx[b][t][top_k]` per-QUERY operand —
#   `IndexScore` scores ONE query and `IndexSelect` emits ONE `iidx[top_k]`. So both arms run
#   the identical dense, ctx-linear prefill and any TTFT difference between them is noise.
#   DSA is a DECODE-ONLY change in this tree, and TPOT is the only cell it can touch.
emitdsa)
  cd "$WT"
  nix develop --command cargo build --release -p plowc --bin plowc
  # NOT `VAR=x emit_one ...`: in bash a prefix assignment on a FUNCTION call persists after
  # the call returns, so the second arm would silently inherit the first's DSA_ENV.
  for tp in ${TPS:-4 8}; do
    DSA_ENV="";  VARIANT="dsa-";   emit_one "$tp"
    DSA_ENV="0"; VARIANT="dense-"; emit_one "$tp"
  done
  ;;

# ------------------------------------------------------------- §6g-GF8 PHASE B, A/B PAIR
# `PLOW_GLM_GF` IS AN EMIT-TIME KNOB (`glm_gf` reads it in devgen and bakes it into `i[7]` AND
# into the packet's workgroup list), so the A/B is two bundles, not two env settings on one.
# `build.json` records no GF at all — the DIRECTORY NAME is the only label. Do not rename.
#
# Why this is worth an A/B at all: `glm_gf` has returned 8 for every ctx>4096 GLM blob forever,
# but the interpreter had only `if (gf==2) <2> else <4>` until 9dc27bb (2026-07-28 22:45), so
# i[7]=8 ran GF=4 silently. Every long-ctx GLM row recorded before that object build is a GF=4
# row. THIS pair is the first time the two arms can actually differ on gfx950.
#
# THE THIRD ARM IS NOT OPTIONAL IF THE FIRST TWO TIE. `glm_nsplit`'s chip-fill cap is written
# for GF=4, so GF=8 halves the latent traffic AND halves the work items `(nh_l/GF)*nsplit` —
# to first order those cancel. `gf8ns` pins nsplit to 2x so the work-item count is matched and
# only the traffic differs. `GF8_NS` must be set to twice whatever `glm_nsplit` picks.
emitgf)
  cd "$WT"
  nix develop --command cargo build --release -p plowc --bin plowc
  for tp in ${TPS:-8}; do
    DSA_ENV="0"; GF_ENV="4"; NS_ENV="";           VARIANT="gf4-";   emit_one "$tp"
    DSA_ENV="0"; GF_ENV="8"; NS_ENV="";           VARIANT="gf8-";   emit_one "$tp"
    [ -n "${GF8_NS:-}" ] && {
      DSA_ENV="0"; GF_ENV="8"; NS_ENV="$GF8_NS";  VARIANT="gf8ns-"; emit_one "$tp"; }
  done
  ;;

# ------------------------------------------------- THE 2D (GF x nsplit) OCCUPANCY SWEEP
# `VARIANT=gf4ns64- ./glm52_tpctx_sweep.sh nsrun 4` runs one arm; `emitns` emits all eight.
#
# WHY THIS SWEEP EXISTS. `glm_nsplit`'s recorded optimum (ns=16 for ctx<=8k) was fitted against
# a `MlaMergeFold` that no longer exists — its body was rewritten (~6.5 ms) and its dispatch was
# then narrowed 256 -> 128 workgroups in 3699ff1. The docstring's whole argument is a BALANCE
# ("the O(nsplit) merge growth eats the decode saving"), so making the merge cheaper moves the
# balance point toward MORE splits. §6b-STALE is the precedent: the same byte-identical blob
# reversed sign three times across three merge implementations.
#
# THE ARITHMETIC BEING TESTED (GLM-5.2 TP4: nh_l = 64/4 = 16, GLM_MLA_GF = 4):
#   n_grp = nh_l/GLM_MLA_GF = 4, fill = ceil(256/4) = 64, ns = clamp(ctx/512, 16, min(fill,kv_tiles))
#   ctx 4k-8k -> ns 16 ->  64 work items -> 25% of the 256 CUs
#   ctx 16k   -> ns 32 -> 128 items      -> 50%
#   ctx >=32k -> ns 64 -> 256 items      -> 100%
# GLM IS NOT RAGGED: n_grp=4 divides n_cu=256, so the fill cap is already grid-aligned and
# `PLOW_NS_FULL_ABS`'s `aligned = n_cu/gcd(n_grp,n_cu)` rescue is irrelevant here (it is also
# GATED OFF for GLM — it needs `kvh_slide != kvh_full` and GLM has no sliding-window layers).
# UNDER-FILL at short/mid ctx is the lever, not alignment.
#
# ONE BLOB PER (GF, ns) SERVES BOTH CTX POINTS. `PLOW_GLM_NS` REPLACES ns outright
# (`unwrap_or(ns)`), bypassing the `fill` and `kv_tiles` caps, so >64 is reachable — and the
# kernel splits over the LIVE `kv_len` (`d_flash_mla_decode`: `span = cend - first`, `cend` from
# the `kv_len` operand), NOT over the emit-time max ctx. So an 8192-position decode on a
# max-ctx-36864 blob partitions 8192 rows into the pinned ns, exactly as a max-ctx-8192 blob
# would. That is why this emits 8 bundles and not 16.
#
# MATCHED WORK ITEMS ACROSS GF IS THE COMPARISON. `glm_nsplit` computes n_grp from the CONSTANT
# `GLM_MLA_GF = 4`, never from the selected GF, so at GF=8 the real n_grp is 2 and the item count
# `(nh_l/GF)*ns` HALVES. GF=8 therefore needs DOUBLE the ns for the same chip fill:
#   GF=4 ns {16,32,64,128} and GF=8 ns {32,64,128,256}  both give items {64,128,256,512}.
# The §6g-GF8 open question ("GF=8 halves latent traffic AND halves the work items, to first
# order these cancel unless PLOW_GLM_NS doubles") is answered by GF=8@ns128 vs GF=4@ns64.
#
# THE TOP RUNG IS OVER-SUBSCRIPTION, NOT WIDTH. `flash_mla_cus` hands back `min(n_work, n_cu)`
# workgroups and `d_flash_mla_decode` grid-strides `for (w=slice; w<n_work; w+=nblk)`, so the
# 512-item arms dispatch 256 workgroups that each run TWO splits back to back. They test whether
# a finer KV partition still pays once the chip is full, which is a different question from fill
# and is the one `NS_PER` actually controls above 32k:
#   arm        GF ns  n_grp items  flash WGs      | arm        GF  ns  n_grp items  flash WGs
#   gf4ns16     4  16   4     64    64            | gf8ns32     8   32   2     64    64
#   gf4ns32     4  32   4    128   128            | gf8ns64     8   64   2    128   128
#   gf4ns64     4  64   4    256   256            | gf8ns128    8  128   2    256   256
#   gf4ns128    4 128   4    512   256 (x2)       | gf8ns256    8  256   2    512   256 (x2)
# `MlaMergeFold`'s dispatch does NOT move with ns — `mla_fold_cus` sizes it from `nh_l * ceil(v/VT)`
# = 128 — so the merge's O(nsplit) growth lands entirely in its BODY, and any change in its GATE
# is the §6b-i straggler max over the flash's 64 -> 256 producers. `--chain` separates the two.
#
# LABEL THE DIRECTORIES. `build.json` records neither GF nor nsplit, so a gf4ns64 and a gf8ns128
# bundle are indistinguishable on disk — the same trap §6g-GF8 names for GF alone.
#
# SEARCH TRUNCATED, PRICE FULL. `PLOW_GLM_NS` is an EMIT-time knob, so every arm is a fresh blob
# AND a fresh weight load — and the load, not the measurement, is what a lease actually buys
# (glm52_decode.c:224). So the 8-arm grid is emitted with `NLAYERS=8` (3 dense + 5 MoE) and the
# WINNER plus its control are then re-emitted at all 78 layers and priced end-to-end through
# plowrt. The truncated run RANKS; the full run PRICES.
#
# WHAT TRANSFERS AND WHAT DOES NOT — state this, do not hide it. The trade being swept is
# PER-LAYER (flash-decode saving vs O(nsplit) merge growth), and `--chain` reports per-layer
# quantities, so the ranking should carry. What cannot carry is anything that is a property of
# the 78-layer stream as a whole: CU contention between concurrent packets, total chain depth,
# and the CU-0 pileup where the narrow spine ops serialise (§7c: 61% of the gate stall happens
# with <=1 CU in a body). A truncated TOKEN is not the real token and must never be quoted as
# one. If the full-network delta disagrees in SIGN with the truncated ranking, that disagreement
# is the finding.
#
# TOKEN IDENTITY IS THE GATE, NOT A COURTESY. Plow.SplitK: the split reduction equals the
# sequential sum for ANY nsplit, so every arm must decode identical tokens. On GLM-5.2 a wrong
# arm OVER-REPORTS — routing is data-dependent, so garbage activations collapse the router's
# top-k and the expert ops do less work (PLOW_XR_SHUFFLE captured 45% of a ceiling while
# deleting nothing). `nsrun` checks tokens before it reports a time. NOTE the identity here is
# NUMERICALLY-EQUIVALENT, not bit-identical: different split boundaries reassociate the merge.
emitns)
  cd "$WT"
  nix develop --command cargo build --release -p plowc --bin plowc
  # MAXCTX stays at the file default (135168), the SAME value the published `tp<N>`, `gf4-`,
  # `gf8-` and `dsa-`/`dense-` bundles were emitted at. It could be cut to 36864 (the top sweep
  # ctx plus headroom) and every arm would still be internally consistent, but keeping it here
  # makes these rows directly comparable to the recorded §6g ones — max_ctx sizes the KV ring
  # and the activation arena, and `d_flash_mla_decode` splits over the LIVE `kv_len`, so it does
  # not touch the quantity being swept.
  for tp in ${TPS:-4}; do
    for ns in ${GF4_NS:-16 32 64 128}; do
      DSA_ENV="0"; GF_ENV="4"; NS_ENV="$ns"; VARIANT="gf4ns$ns-"; emit_one "$tp"
    done
    for ns in ${GF8_NS_LIST:-32 64 128 256}; do
      DSA_ENV="0"; GF_ENV="8"; NS_ENV="$ns"; VARIANT="gf8ns$ns-"; emit_one "$tp"
    done
  done
  ;;

# ------------------------------------------- ONE ARM OF THE nsplit SWEEP, OP-LEVEL + IDENTITY
# `VARIANT=gf4ns64- NSCTXS='8192 32768' ./glm52_tpctx_sweep.sh nsrun 4`
#
# This is an OP-LEVEL MICROBENCH and it is deliberately NOT the end-to-end instrument. §0-BENCH
# reserves the token number for plowrt through the shared client; `runarm` above is that path.
# What `glm52_decode --sweep` buys is the MECHANISM: the docstring's claim is that the flash
# saving and the merge growth trade against each other, and only a per-op trace can show the
# trade rather than its sum. It also skips the prefill — `--sweep` patches `in.kvlen` directly,
# so a 32768-position decode step costs a step, not the 72 s TTFT a real 32k prompt costs.
#
# THE SWEEP'S OWN ms/tok IS SYNTHETIC (garbage KV, so the MoE router sees garbage) and must NOT
# be quoted as a token. `--gen` in the SAME process decodes from position 0 with real numerics
# and is what the identity check reads; the 4-minute 183 GiB/rank weight load is paid once for
# both, which is the only reason this fits in a lease.
nsrun)
  tp="${2:?tp}"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  b="$(bundle "$tp")"
  [ -f "$b/model.pkt" ] || { echo "!! $b/model.pkt missing — run '$0 emitns' first"; exit 1; }
  D="${NS_OBJ:-/home/lava/plow/build-amd/nssweep-objs}"
  # WHICH OBJECT AN ARM NEEDS, AND WHY THE ANSWER IS NOT "the newest one".
  #
  # The GF=8 flash-decode instantiation is compiled only under `-DPLOW_GLM_GF8_ARM=1`
  # (op_attention.h, DEFAULT 0 since 3344543). Without it `i[7]=8` falls into the `else` and runs
  # the GF=4 body, so a `gf8*` arm measured against a default object is a GF=4 arm wearing a GF=8
  # label — the §6g-GF8 bug, from the other side.
  #
  # AND THE ARM IS NOT FREE TO MERELY CARRY. Its presence grows i_decode.co 313 KB -> 362 KB and
  # costs +32% decode (3344543, bisected on a byte-identical packet), because in a PERSISTENT
  # MEGAKERNEL every packet body shares one instruction stream. So:
  #   * a GF=4-vs-GF=8 comparison MUST run both arms on the SAME arm-present object, or it
  #     measures object size and not GF;
  #   * a SHIPPING number must come from the default, arm-ABSENT object;
  #   * the two sets of numbers are not comparable to each other and must not be tabulated
  #     together without saying which object produced each.
  # The size of i_decode.co is the label, so check it and say what it implies.
  sz=$(stat -c%s "$D/i_decode.co")
  arm8=$([ "$sz" -gt 330000 ] && echo yes || echo no)
  echo "   object: i_decode.co $sz B, GF=8 arm present: $arm8"
  case "${VARIANT}:$arm8" in
    gf8*:no) echo "!! $tag needs -DPLOW_GLM_GF8_ARM=1; this object would run GF=4 under a GF=8 label."
             exit 1 ;;
    gf4*:yes) echo "   (arm-present GF=4 control — comparable to the gf8 arms, NOT to a shipping number)" ;;
  esac
  out="${NS_OUT:-/home/lava/models/glm52_nssweep}"; mkdir -p "$out"
  tag="${VARIANT%-}"
  echo "########## nsrun ARM=$tag TP$tp  bundle=$b  objs=$D"
  cd "$D"
  PLOW_INTERP="$D/interp_decode.elf" PLOW_TRACE_RAW="$out/tr_$tag" \
    ./glm52_decode "$b/model.pkt" "$CKPT" --tp "$tp" --steps "${NSTEPS:-21}" \
      --sweep "$(echo ${NSCTXS:-8192 32768} | tr ' ' ',')" --gen "${NSGEN:-24}" \
    2>&1 | tee "$out/nsrun_$tag.log"
  echo "===== per-op chain, ARM=$tag"
  for c in ${NSCTXS:-8192 32768}; do
    f="$out/tr_$tag.tp$tp.ctx$c.bin"
    [ -f "$f" ] || continue
    nix develop --command python3 "$WT/scripts/glm52_trace_analyze.py" \
      "$out/tr_$tag.insts.txt" "$f" --ctx "$c" --tp "$tp" --chain \
      | tee -a "$out/nsrun_$tag.log"
  done
  ;;

# ALL EIGHT ARMS IN ONE LEASE. Each arm costs a ~4-minute 183 GiB/rank weight load and then a
# few seconds of stepping, so the lease is dominated by loads — taking eight separate leases
# would add eight queue waits for nothing. This is ONE phase; release after it.
nsall)
  tp="${2:?tp}"
  for v in ${NS_ARMS:-gf4ns16 gf4ns32 gf4ns64 gf4ns128 gf8ns32 gf8ns64 gf8ns128 gf8ns256}; do
    VARIANT="$v-" bash "${BASH_SOURCE[0]}" nsrun "$tp" || echo "!! ARM $v FAILED (rc=$?)"
  done
  ;;

# ---------------------------------------------------------------- tuning gate
# GEMM tiles are consulted ONLY by `tunedb::gemm_op_case`, and the DECODE program has zero
# Gemm ops (every decode matmul is Gemv/GemvQkv/GemvGlu/GemvFp8Blk), so stale tuning cannot
# move ms/token. It DOES leave PREFILL and therefore TTFT unmeasured — and a context sweep
# is overwhelmingly a TTFT measurement. So this gate is load-bearing for THIS script
# specifically, and `run` refuses to start without it.
tune)
  OBJD="${2:?objdir with a freshly built test_kernels.elf}"
  bash "$WT/scripts/rebench_tune_gemm.sh" "$OBJD" /tmp/glm_ctx_tune.jsonl
  cd "$WT"
  nix develop --command cargo run --release -p plowc --bin plowc -- \
      tune ingest --gpu mi350 --root . --db tuning \
      --samples /tmp/glm_ctx_tune.jsonl --campaign glm52-ctx-sweep
  nix develop --command cargo test -p devgen --test tuned_tile_selection
  ;;

# The gate `run` makes, standalone and GPU-free.
#
# `published_measurements_reach_the_compiler_and_change_its_answer` is THE gate: it fails
# exactly when selection has silently reverted to the analytical model, which is the state
# that makes a TTFT number unmeasured. It must pass, and `run` refuses without it.
#
# `the_narrow_shapes_agree_between_model_and_hardware` is a DIFFERENT claim — that on the
# M=128 class the campaign only ever CONFIRMS the analytical model. Three independent
# campaigns against build gfx950-a168b6e2e77e1975 falsify it on two shapes, and NEITHER IS
# A GLM-5.2 SHAPE (medians over the 3 passes, ns):
#
#   Gemma-26B router     128x128x2816   128x128 = 28828  <  64x128 = 29590   model says 64x128
#   Gemma-12B k global   128x512x3840   128x128 = 43362  <  64x128 = 47639   model says 64x128
#   GLM-5.2 router       128x256x6144    64x128 = 49351  < 128x128 = 62749   AGREE
#   GLM-5.2 kv_a_proj    128x576x6144    64x128 = 63405  < 128x128 = 72806   AGREE
#   Kimi kv_a_proj       128x576x7168    64x128 = 70695  < 128x128 = 81856   AGREE
#   Gemma-26B gate/up    128x2112x2816   64x128 = 43891  < 128x128 = 49146   AGREE
#
# So every GLM-5.2 narrow shape agrees and this sweep is unaffected; the failure is a real
# measured correction to the model on two Gemma shapes, reproduced 3x, and belongs to
# whoever owns `pick_tile`. It is reported, not silenced, and it does not block GLM.
gate)
  cd "$WT"
  # plowrt MUST be the `--features hsa` build. A default `cargo build -p plowrt` produces a
  # binary that selects the CPU REFERENCE BACKEND and serves fluent-looking garbage through a
  # byte-fallback tokenizer — and `target/` is SHARED, so another agent's plain build silently
  # replaces yours. That happened during this campaign: the hsa binary (6.96 MB, 21:34) was
  # clobbered by a 3.51 MB CPU-only one at 21:54, and the next run came up "ready after 2s"
  # (no weights loaded) answering 'koesgysgseyioseyeskuyiksggqocsgy'. The coherence gate caught
  # it, but only after a lease had been taken. Rebuild here, GPU-free, before anything is
  # leased — §0 forbids compiling under `gpulease`, so it cannot live inside `run`.
  nix develop --command cargo build --release -p plowrt --features hsa 2>&1 | tail -2
  n=$(strings target/release/plowrt 2>/dev/null | grep -c "HSA backend selected")
  [ "${n:-0}" -ge 1 ] || { echo "!! plowrt has NO HSA backend — it would serve CPU garbage."; exit 1; }
  echo "plowrt: hsa build OK ($(stat -c%s target/release/plowrt) bytes)"

  log=/tmp/glm_ctx_tunegate.log
  nix develop --command cargo test -p devgen --test tuned_tile_selection >"$log" 2>&1 || true
  grep -E "^test |test result" "$log" | tail -6
  # DECODE_ONLY=1 downgrades the tuning assertion to a loud warning, and it is the ONLY thing it
  # downgrades — the HSA check above stays hard, because a CPU-only plowrt invalidates every
  # number rather than one column. The gate exists because a ctx sweep is overwhelmingly a TTFT
  # measurement; an ITL/TPOT A/B between two blobs that differ ONLY in a decode packet field is
  # not, and the DECODE program contains no `Gemm` op for `gemm_op_case` to have an opinion about.
  # The cost of refusing anyway is real and was paid here: the campaign gained 32 unmeasured
  # prefill shapes mid-run (461489d), the gate flipped between two arms of the SAME A/B, and the
  # control arm was lost twice while the candidate ran. Setting this DISCARDS the TTFT column —
  # say so wherever the numbers are reported.
  if grep -q "test published_measurements_reach_the_compiler_and_change_its_answer \.\.\. ok" "$log"
  then :
  elif [ -n "${DECODE_ONLY:-}" ]; then
    echo ">>> TUNING IS STALE and DECODE_ONLY=1 is set. Proceeding: prefill tiles come from the"
    echo ">>> analytical model, so EVERY TTFT CELL BELOW IS UNMEASURED and must not be reported."
    echo ">>> ITL / TPOT are unaffected — the decode program has no Gemm op."
  else
    echo "!! tuning is STALE — run '$0 tune <objdir>' first. TTFT would be UNMEASURED."
    echo "!! (decode-only A/B? re-run with DECODE_ONLY=1 and DISCARD the TTFT column.)"
    exit 1
  fi
  grep -q "the_narrow_shapes_agree_between_model_and_hardware ... ok" "$log" \
    || echo ">>> NOTE: narrow-shape model/hardware disagreement — see the header. No GLM shape affected."
  ;;

# ------------------------------------------------- ONE ARM OF THE DSA A/B, plowrt ONLY
# `VARIANT=dsa- ./glm52_tpctx_sweep.sh runarm 8`
#
# Same gate, same coherence gate, same client, same per-ctx prompt counts as `run` — it is
# `run` with the vLLM phase removed, because the vLLM column is a PROPERTY OF THE CHECKPOINT
# and does not change between plow's two arms. Paying its 3110 s health wait twice to
# reproduce the same row would cost more lease time than the whole plow A/B.
#
# The bundle directory is echoed into the log because build.json CANNOT distinguish the arms.
runarm)
  tp="${2:?tp}"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  b="$(bundle "$tp")"
  [ -f "$b/model.pkt" ] || { echo "!! $b/model.pkt missing — run '$0 emitdsa' first"; exit 1; }
  cd "$WT"
  bash "${BASH_SOURCE[0]}" gate || exit 1
  echo "########## ARM=${VARIANT:-<published-dense>} TP$tp  bundle=$b"
  echo "########## ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}  objs=$(readlink -f "$b/hsaco")"
  echo "===== coherence, ${VARIANT}TP$tp"
  bash "$WT/scripts/rebench_glm_coherence.sh" "$b" "$PORT" glm-5.2 2>&1 | tail -30
  echo "===== end coherence"
  MAP=""; WMAP=""
  for L in $CTXS; do MAP="$MAP $L:$(npr "$L")"; WMAP="$WMAP $L:$(nwarm "$L")"; done
  echo "prompt counts:$MAP"
  echo "warm-ups     :$WMAP"
  echo "===== plow, ${VARIANT}TP$tp"
  IN_LENS="$CTXS" CONCS="$CONC" NPROMPT="${NPROMPT:-8}" OUTLEN="$OUTLEN" \
    NPROMPT_MAP="$MAP" NWARM_MAP="$WMAP" \
    bash "$WT/scripts/bench_plowrt_serve.sh" "$b" "$PORT" glm-5.2 zai-org/GLM-5.2-FP8 2>&1 \
    | tee "$OUT/plow_${VARIANT}tp$tp.log"
  echo "===== end plow"
  echo "===== integrity audit, ${VARIANT}TP$tp"
  # `grep -c` PRINTS 0 *and* exits 1 when it matches nothing, so the old `|| echo 0` appended a
  # SECOND zero: `bad` became the two-line string "0\n0", `[ "$bad" = 0 ]` failed, and a clean run
  # printed "markers: 0" immediately followed by "!! DISCARD OR FLAG". Observed on a run whose
  # cells were all good. `|| true` keeps `set -e` happy without inventing a value.
  bad=$(grep -ciE "admission shed|stream ended with no terminal chunk" \
          "${LOG:-/tmp/plowrt_bench_$PORT.log}" 2>/dev/null || true)
  echo "shed/truncation markers in the plow server log: $bad"
  [ "$bad" = 0 ] || echo "!! DISCARD OR FLAG the plow cells above."
  echo "===== end integrity audit"
  ;;

# ---------------------------------------------------------------- the matrix
run)
  tp="${2:?tp}"
  [ "$tp" = 2 ] && { echo "!! TP2 does not fit — see the header. Not run."; exit 2; }
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  b="$(bundle "$tp")"
  [ -f "$b/model.pkt" ] || { echo "!! $b/model.pkt missing — run '$0 emit' first"; exit 1; }
  cd "$WT"
  bash "${BASH_SOURCE[0]}" gate || exit 1

  echo "########## TP$tp  ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
  # The digest every record in `tuning/` is keyed to, printed beside the numbers so a reader
  # can tell a measured tile from a fallback to the analytical model without leaving the log.
  # (`--quant bf16` finds nothing by design: `gemm_op_case` renders bf16 as the `/None`
  # suffix, so the filter is for the quantized rungs. The `build:` line is what is wanted.)
  echo "build digest: $(nix develop --command "$WT/target/release/plowc" \
      tune best --gpu mi350 --root "$WT" --db tuning --quant bf16 2>/dev/null \
      | sed -n 's/^build digest *: *//p')"

  # 1. CORRECTNESS BEFORE TIMING (knob-contract §5). A 128k dense run that produces garbage
  #    is a bug report, not a benchmark row. Reads real text out of two buckets and checks
  #    the first SSE chunk carries CONTENT (the 63f9957 TTFT artefact must stay dead).
  echo "===== coherence, TP$tp"
  bash "$WT/scripts/rebench_glm_coherence.sh" "$b" "$PORT" glm-5.2 2>&1 | tail -30
  echo "===== end coherence"

  # The per-ctx prompt/warm-up counts, built ONCE and handed to both engines byte-identically.
  # `NPROMPT_MAP` / `NWARM_MAP` are `<len>:<n>` pairs the bench scripts look up per input
  # length; unset, both scripts behave exactly as before. Building it here rather than inside
  # the bench scripts is what keeps the two engines' counts provably the same.
  MAP=""; WMAP=""
  for L in $CTXS; do MAP="$MAP $L:$(npr "$L")"; WMAP="$WMAP $L:$(nwarm "$L")"; done
  echo "prompt counts:$MAP"
  echo "warm-ups     :$WMAP"

  # 2. plow. ONE server load, every ctx swept inside it — the load is 167 s and paying it
  #    per ctx point would cost more than the measurement.
  echo "===== plow, TP$tp"
  IN_LENS="$CTXS" CONCS="$CONC" NPROMPT="${NPROMPT:-8}" OUTLEN="$OUTLEN" \
    NPROMPT_MAP="$MAP" NWARM_MAP="$WMAP" \
    bash "$WT/scripts/bench_plowrt_serve.sh" "$b" "$PORT" glm-5.2 zai-org/GLM-5.2-FP8 2>&1 \
    | tee "$OUT/plow_tp$tp.log"
  echo "===== end plow"

  # 3. vLLM, MATCHED: same client binary, same backend, same dataset, same TP, and
  #    --max-model-len 131072 so its KV budget is the same context this sweep asks for.
  #    --dataset-name random with NO --random-prefix-len: prompts share no prefix, so the
  #    prefix cache should have nothing to hit. VERIFY that from the hit rate vLLM prints
  #    (grepped out below) rather than trusting the argument (task #21).
  echo "===== vLLM, TP$tp"
  IN_LENS="$CTXS" CONCS="$CONC" NPROMPT="${NPROMPT:-8}" OUTLEN="$OUTLEN" MAXLEN="$MAXCTX" \
    NPROMPT_MAP="$MAP" NWARM_MAP="$WMAP" \
    DTYPE_ARGS="--dtype auto" EXTRA_ENV="${EXTRA_ENV:-VLLM_ROCM_USE_AITER=1}" \
    bash "$WT/scripts/bench_vllm_chat.sh" zai-org/GLM-5.2-FP8 "$tp" 2>&1 \
    | tee "$OUT/vllm_tp$tp.log"
  grep -iE "prefix cache hit rate|sparse|indexer|dsa" "$OUT"/vllm_tp$tp.log | head -20 || true
  echo "===== end vLLM"

  # 4. BENCHMARK-INTEGRITY AUDIT. Two plowrt paths render a FAILED request as a SUCCESSFUL
  #    one on the wire, so `vllm bench serve` scores it as a completed request:
  #      * `StreamChunk::Err` renders as `finish_reason:"stop"` (serve/chat.rs:401), so an
  #        admission shed (mux.rs:562-587, which drops every live slot) arrives as a normal
  #        completion carrying the error text as content;
  #      * a stream that ends with no terminal chunk (fixed receiver-side in a5f4618) was
  #        scored as a short but successful request.
  #    Both inflate tok/s while deflating mean output length — which reads exactly like a good
  #    result. The wire shape was deliberately left unchanged so cells stay comparable, so the
  #    filtering is the caller's job. A cell whose server log contains either marker is not
  #    reportable.
  echo "===== integrity audit, TP$tp"
  # `grep -c` PRINTS 0 *and* exits 1 when it matches nothing, so the old `|| echo 0` appended a
  # SECOND zero: `bad` became the two-line string "0\n0", `[ "$bad" = 0 ]` failed, and a clean run
  # printed "markers: 0" immediately followed by "!! DISCARD OR FLAG". Observed on a run whose
  # cells were all good. `|| true` keeps `set -e` happy without inventing a value.
  bad=$(grep -ciE "admission shed|stream ended with no terminal chunk" \
          "${LOG:-/tmp/plowrt_bench_$PORT.log}" 2>/dev/null || true)
  echo "shed/truncation markers in the plow server log: $bad"
  [ "$bad" = 0 ] || echo "!! DISCARD OR FLAG the plow cells above — see the note in this script."
  # Independent cross-check: every request must have produced exactly OUTLEN tokens.
  for L in $CTXS; do
    f="/tmp/vllmbench_glm-5.2_in${L}_c1.log"
    [ -f "$f" ] && awk -v l="$L" -v o="$OUTLEN" '
      /Successful requests/ {n=$NF}
      /Total generated tokens/ {g=$NF}
      END {printf "  ctx %-7s n=%-3s generated=%-6s expected=%-6s %s\n",
                  l, n, g, n*o, (g==n*o ? "ok" : "MISMATCH — truncation")}' "$f"
  done
  echo "===== end integrity audit"
  ;;

# ------------------------------------------------- LONG-CONTEXT CORRECTNESS
# `rebench_glm_coherence.sh` tops out at a ~2.1k prompt, so it proves the T=128/512/2048
# buckets and NOTHING about the regime this sweep is actually about. The 8192 bucket, the
# 16-chunk cover, and dense attention over 128k KV rows are all unexercised by it — and the
# recorded failure mode at long ctx is DEGENERATE TEXT, not a crash, so it has to be read.
#
# This is a needle probe: a fact is planted at a known depth inside ~120k tokens of filler
# and the model is asked for it. A wrong answer means the 128k row of the table is an
# instrument reading, not a result. `vllm bench serve`'s random-token prompts cannot make
# this check for you — their outputs are meaningless by construction.
longcoherence)
  tp="${2:?tp}"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  b="$(bundle "$tp")"; cd "$WT"
  LOG=/tmp/glm_longcoh_$PORT.log
  setsid nix develop -c ./target/release/plowrt serve --assets "$b" --port "$PORT" \
    >"$LOG" 2>&1 &
  SRV=$!
  trap 'kill -TERM -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null; sleep 2;
        kill -KILL -"$SRV" 2>/dev/null; sleep 2' EXIT
  for i in $(seq 1 900); do
    kill -0 $SRV 2>/dev/null || { echo "!! server died:"; tail -30 "$LOG"; exit 1; }
    curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 1
  done
  # THE BACKEND GATE. Without it this probe is worthless, and that is not hypothetical: its
  # first-ever run came up in 5 s on `hsa_init failed: 4104`, silently selected the CPU
  # reference backend, answered '' at every depth and printed a confident FAIL. A CPU-backend
  # FAIL says nothing about the GPU 128k row. `run` inherits its gate from
  # rebench_glm_coherence.sh; this path had none, so it gets its own.
  if grep -q "CPU reference backend active" "$LOG"; then
    echo "!! plowrt selected the CPU REFERENCE BACKEND — this probe measures NOTHING."
    grep -E "HSA probe failed|hsa_init" "$LOG" | head -3
    exit 1
  fi
  grep -qE "HSA backend selected|hsa=true" "$LOG" || echo ">>> WARN: no HSA banner in $LOG"
  # NEEDLE_TOKENS is a LIST, so one server load can walk a ctx ladder. That matters when the
  # probe FAILS: "degenerate at 119k" and "degenerate at every length" are different bugs, and
  # a reload costs 160-255 s per rung.
  python3 - "$PORT" "${NEEDLE_DEPTHS:-0.1 0.5 0.9}" "${NEEDLE_TOKENS:-119000}" <<'PY'
import json, sys, urllib.request
port, depths = sys.argv[1], [float(x) for x in sys.argv[2].split()]
targets = [int(x) for x in sys.argv[3].split()]

def ask(text, max_tokens):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": "glm-5.2",
                         "messages": [{"role": "user", "content": text}],
                         "max_tokens": max_tokens, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return (r["choices"][0]["message"]["content"],
            r.get("usage", {}).get("prompt_tokens"))

FILL = "The river was quiet and the lanterns burned low along the far bank. "
NEEDLE = "The secret passphrase is CRIMSON-FALCON-{tag}. "

# CALIBRATE THE REPEAT COUNT AGAINST THE SERVER'S OWN TOKENIZER — do not guess it.
# The original constant (N=17000, "~7 tokens per repeat") was off by 2.2x: this filler is
# ~15 tokens per repeat, so it built a 255,033-token prompt against a 135,168-token blob.
# That is past max_ctx, i.e. exactly the KV-ring overrun this repo has hit three times, and
# it would have been read as a model failure. One cheap 200-repeat request fixes it exactly.
_, probe_tok = ask(FILL * 200, 1)
per = probe_tok / 200.0
print(f"  calibration: {per:.2f} tok/repeat", flush=True)

ok = True
for target in targets:
    N = max(1, int(target / per))
    for d in depths:
        tag = f"{int(d*100):02d}"
        at = int(N * d)
        body = FILL * at + NEEDLE.format(tag=tag) + FILL * (N - at)
        q = (body + "\n\nQuestion: what is the secret passphrase stated somewhere above? "
             "Answer with the passphrase only.")
        got, ptok = ask(q, 24)
        hit = f"CRIMSON-FALCON-{tag}" in got
        ok &= hit
        print(f"  ~{target:>7} tok  depth {d:>4}  prompt_tokens={ptok}  "
              f"{'PASS' if hit else 'FAIL'}  {got!r}", flush=True)
print(">>> long-context coherence:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
  ;;

# ------------------------------------------------- ONE ENGINE, RE-RUNNABLE
# The vLLM half alone, so a failed vLLM phase does not cost another plowrt weight load (167 s
# at TP4, 255 s at TP8) or another full plow sweep.
#
# NOTE ON `VLLM_ROCM_USE_AITER=1`, which `run` passes by default and this does NOT:
# on this image it DEADLOCKS GLM-5.2 at startup. All four TP workers sat on
# `[aiter] waiting for baton release at .../lock_module_gemm_a8w8_blockscale` with zero new
# log lines for 90 s and the engine never became healthy — the JIT build lock for the
# block-scale GEMM module is contended across the TP workers and never releases. The same log
# also shows `shape is M:8192, N:2624, K:6144, not found tuned config in
# /tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv, will use default conf`, i.e. even when it
# does start there is no tuned gfx950 AITER config for GLM's shapes (knob-contract §6g).
# So AITER buys vLLM nothing here and costs a deadlock; unset is the config that runs.
vllmonly)
  tp="${2:?tp}"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  cd "$WT"
  MAP=""; WMAP=""
  for L in $CTXS; do MAP="$MAP $L:$(npr "$L")"; WMAP="$WMAP $L:$(nwarm "$L")"; done
  echo "########## vLLM TP$tp  ROCR=${ROCR_VISIBLE_DEVICES:-<unset>}"
  echo "prompt counts:$MAP"
  IN_LENS="$CTXS" CONCS="$CONC" NPROMPT="${NPROMPT:-8}" OUTLEN="$OUTLEN" MAXLEN="$MAXCTX" \
    NPROMPT_MAP="$MAP" NWARM_MAP="$WMAP" \
    DTYPE_ARGS="--dtype auto" EXTRA_ENV="${EXTRA_ENV:-}" \
    bash "$WT/scripts/bench_vllm_chat.sh" zai-org/GLM-5.2-FP8 "$tp" 2>&1 \
    | tee "$OUT/vllm_tp$tp.log"
  # Prefix caching must have nothing to hit: `--dataset-name random` with no
  # `--random-prefix-len` (confirmed `random_prefix_len=0` in the client's own arg dump).
  # Verify from the SERVER's hit rate rather than trusting the flag (task #21).
  grep -iE "prefix cache hit rate" "$OUT/vllm_tp$tp.log" | tail -5 || true
  ;;

table)  # collate whatever logs exist into the report table
  python3 "$WT/scripts/glm52_tpctx_sweep_table.py" "$OUT"
  ;;

*) sed -n '2,80p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
