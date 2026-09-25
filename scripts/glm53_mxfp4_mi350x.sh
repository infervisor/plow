#!/usr/bin/env bash
# GLM-5.3 MXFP4 on 8x MI350X (gfx950), TP8 — the validated recipe of docs/amd/glm53-mxfp4-mi350x.md.
#
#   scripts/glm53_mxfp4_mi350x.sh overlay CKPT_OUT           # CPU: runtime checkpoint overlay
#   scripts/glm53_mxfp4_mi350x.sh emit PKT_OUT               # CPU: packet (inside nix develop)
#   scripts/glm53_mxfp4_mi350x.sh objects PKT OBJ_OUT        # CPU: gfx950 objects for PKT
#   scripts/glm53_mxfp4_mi350x.sh assets PKT CKPT ASSETS_OUT # CPU: serve bundle
#   scripts/glm53_mxfp4_mi350x.sh validate PKT OBJ CKPT ASSETS OUT   # 8 GPUs, submit through gpuq:
#     python3 scripts/bench/gpuq.py submit scripts/glm53_mxfp4_mi350x.sh validate ...
#
# GLM_RECIPE=dsa: DSA sparse attention (vLLM's top-2048 indexer; overlay needs --dsa).
# GLM_MAX_CTX / GLM_SEQ override the context and prefill bucket ladder (default 16384 / 128..16384).
# GLM_EMIT_EXTRA / GLM_OBJ_EXTRA="K=V ..." override the recipe env (later assignment wins).
# v9 (2026-09-25): packet sha16 3b8b96fd6339691d; prefill 1k/8k/16k 140/388/736 ms,
# decode M1 37.7 ms/token, M128 177.9 ms/step; top-1 == vLLM 0.29 oracle T1024..T8192.
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREPPED="${GLM_PREPPED:-/opt/models/plow-glm53-mxfp4-prepped-20260923-1aS85M}"  # scripts/glm53_prep_quark.py
QUARK="${GLM_QUARK:-/opt/models/GLM-5.3-MXFP4-AttnFP8-amd-4992911b}"
PLOWRT="${PLOWRT_BIN:-$WT/target/release/plowrt}"

MAX_CTX="${GLM_MAX_CTX:-16384}"
SEQ="${GLM_SEQ:-128,512,1024,2048,4096,8192,16384}"
EMIT_ENV=(
  PLOW_MXFP4=1 PLOW_GLM_MOE_STAGE1_NATIVE=1 PLOW_GLM_DSA=0 PLOW_UNISEG=0
  PLOW_MLA_PREFILL=full:$SEQ PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32,64,128
  PLOW_GEMV_MM=8 PLOW_GEMV_WALK=1
  PLOW_GLM_QKVA_W8A8=1 PLOW_GLM_OPROJ_W8A8=1 PLOW_GLM_MLA_W8A8=1 PLOW_GLM_MLA_MHA=1
  PLOW_GLM_MOE_SHARED_FOLD=1 PLOW_GLM_DECODE_SHARED_FOLD=1 PLOW_GLM_NORM_Q128=1 PLOW_GLM_QUANT_NARROW=1
  PLOW_GLM_SEQ_PAR=1 PLOW_GLM_SEQ_PAR_PROJ=1 GLM_GROUP=1 PLOW_GLM_FUSE_B1=1 GLM_FUSE_XRN=1
  PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_DECODE_NORM_ROWS=1 GLM_SPINE_CUS=64 PLOW_GLM_DECODE_GLUE_CUS=1
)
if [ "${GLM_RECIPE:-dense}" = dsa ]; then
  EMIT_ENV+=(PLOW_GLM_DSA=topk PLOW_GLM_DSA_PF=1 PLOW_GLM_DSA_PF_SPAN=3 PLOW_GLM_FUSE_ROPE=0
             PLOW_GLM_DSA_PF_B8=1)
fi
OBJ_ENV=(
  PLOW_MXFP4=1 PLOW_MOE_PREFILL=1 PLOW_MOE_PF_A4W4=1 PLOW_MLA_PF_TR16=1 PLOW_DECODE_BATCH=128
  PLOW_GEMV_MM=8 PLOW_GEMV_WALK=1 PLOW_MOE_PF_DOWN_SWEEP=1 PLOW_MLA_FOLD_MFMA=1 PLOW_MOE_PF_A4W4_BK=256
  PLOW_COMBINE_VEC=1 PLOW_MOE_ROUTER_PF_WAVE=1 PLOW_MLA_MHA=1 PLOW_GEMV_MFMA4=1
  PLOW_XR_SCHED=aiter PLOW_XR_SCHED_NWG=40 PLOW_XR_SCHED_NWG_SRS=16 PLOW_FP8_BLK_DMA=1 PLOW_FP8_BLK_KW=1
  PLOW_MERGE_UNROLL4=1 PLOW_GEMV_F32_COL=1 PLOW_MOE_STAGE1_PIPE=1 PLOW_MOE_GLU_KW=1
  PLOW_MOE_DOWN_SWEEP_LINE=1 PLOW_MOE_ALIGN_WAVES=1
)
if [ "${GLM_RECIPE:-dense}" = dsa ]; then
  OBJ_ENV+=(PLOW_MLA_SPARSE=1 PLOW_DSA_SELECT_V2=1 PLOW_DSA_IDX_QPW=1 PLOW_HNR_ILP=1)
fi
# The MHA prefill arm traps on kv_len != rows (prefix-cache hits, continuation chunks).
RUN_ENV=(PLOW_PREFIX_CACHE=0 PLOW_AMD_DECODE_MIN_RUNG=1 PLOW_TP_NO_AUDIT=0 PLOW_TP_AGREE_EVERY=1)

cmd="${1:-}"; shift || true
case "$cmd" in
overlay)
  dsa=(); [ "${GLM_RECIPE:-dense}" = dsa ] && dsa=(--dsa)
  python3 "$WT/scripts/glm53_mxfp4_overlay.py" "$1" "${dsa[@]}" ;;
emit)
  out="$1"; mkdir -p "$(dirname "$out")"
  env "${EMIT_ENV[@]}" ${GLM_EMIT_EXTRA:-} PLOW_VERIFY_BIN="${PLOW_VERIFY_BIN:-$WT/lean-plow/.lake/build/bin/plow_verify}" \
    "$WT/target/release/plowc" --hf-dir "$PREPPED" --gpu mi350 --arch gfx950 --num-gpus 8 --n-cu 256 \
    --max-ctx "$MAX_CTX" --batch 128 --seq "$SEQ" --out "$out"
  sha256sum "$out/model.pkt" ;;
objects)
  env "${OBJ_ENV[@]}" ${GLM_OBJ_EXTRA:-} PLOW_HSACO_CONFIG="$1/plow_config.h" bash "$WT/scripts/build_gfx950.sh" "$2" ;;
assets)
  pkt="$1"; ckpt="$2"; out="$3"; mkdir -p "$out"
  for f in model.pkt build.json plow_config.h lean-checks.json weights.json; do
    [ -e "$pkt/$f" ] && ln -sfn "$pkt/$f" "$out/$f"
  done
  ln -sfn "$ckpt" "$out/checkpoint"
  for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
    [ -e "$QUARK/$f" ] && ln -sfn "$QUARK/$f" "$out/$f"
  done ;;
validate)
  pkt="$1"; obj="$2"; ckpt="$3"; assets="$4"; out="$5"; mkdir -p "$out"
  export "${RUN_ENV[@]}"
  bench=("$PLOWRT" amd-bench --blob "$pkt/model.pkt" --hsaco "$obj" --checkpoint "$ckpt" --tp 8)
  "${bench[@]}" --prefill-sweep 1024,2048,4096,8192,16384 --prefill-reps 5 > "$out/sweep.log" 2>&1
  grep PFSWEEP "$out/sweep.log"
  seq -s, 100 1123 > "$out/p1024.ids"
  for m in 1 2 4 8 16 32 64 128; do
    batched=(); [ "$m" = 1 ] || batched=(--batched --active-batch "$m")
    "${bench[@]}" --prompt "@$out/p1024.ids" --steps 16 "${batched[@]}" > "$out/decode-M$m.log" 2>&1
    echo "M$m $(grep -oE '[0-9.]* ms/(step|token)' "$out/decode-M$m.log" | tail -1)"
  done
  for spec in "8 8064 128" "1 8064 128"; do
    read -r c i o <<< "$spec"
    PLOWRT_BIN="$PLOWRT" PB_ALLOW_HAZARD="PLOW_PREFIX_CACHE PLOW_AMD_DECODE_MIN_RUNG" \
      bash "$WT/scripts/bench/glm53-mxfp4-plow.sh" "$out/serve-c${c}_in$i" "$assets" "$obj" "$c" "$i" "$o" \
      > "$out/serve-c${c}_in$i.log" 2>&1 || echo "serve c$c rc=$?"
    grep -hE "Mean TTFT|Mean TPOT|Total token throughput|Failed" "$out/serve-c${c}_in$i"/*.bench.log \
      "$out/serve-c${c}_in$i"/*/*.bench.log 2>/dev/null | head -4
  done ;;
*)
  sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
