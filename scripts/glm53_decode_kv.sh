#!/usr/bin/env bash
# GLM-5.3 decode-side KV-format / split-policy campaign on 8x MI300X (gfx942).
#
# Three arms, all TP4, all emitted with the SAME confound set so the only
# difference between `ctl` and `fp8` is the latent cache dtype:
#
#   ctl   bf16 latent, NO fuse-rope, PF_NS=1, decode rung 1 only
#   fp8   e4m3 latent + per-row f32 scale (PLOW_GLM_FP8_KV=1), same everything else
#   dsa   bf16 latent, DSA gate armed at a lowered crossover (PLOW_GLM_DSA=1)
#
# `PLOW_GLM_FUSE_ROPE`, `PLOW_GLM_PF_NS>1`, `PLOW_GLM_OFOLD` and a decode BATCH
# ladder are all refused by the fp8-latent emitter (t7/i6 are spent on the scale
# strip, and the fp8 latent writer has no batch-ring form) — so the control drops
# them too rather than comparing against the shipped knob set.
#
#   ./scripts/glm53_decode_kv.sh emit ctl|fp8|dsa|dense66k [maxctx]
#   ./scripts/glm53_decode_kv.sh serve <arm> <port>
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${PLOW_CKPT:-/workspace/models/GLM-5.3-plow-lite}"
RAW="${GLM_RAW:-/workspace/models/GLM-5.3-FP8}"
OBJ="${PLOW_HSACO:-/app/plow/build-glm53/hsaco}"
OUT="${GLM53_DIR:-$WT/build-glm53}"
BIN="${PLOW_BIN_DIR:-$WT/target-glm53/release}"
LADDER="${LADDER:-full:128,512,2048,8192}"
LEASE="/app/plow/perf-data/tools/gpulease"
# The MEASURED gfx942/MI300X tile store. It lives in the main checkout and holds
# 120 records at the current family digest (gfx942-cae47801184a4196) that the
# committed copy does not — without it every dense-GEMM tile in the blob falls
# back to the analytical model and the blob is not comparable with `tp4-long`.
TUNEDB="${PLOW_TUNEDB:-/app/plow/tuning}"

case "${1:?emit|serve|stop}" in

emit)
  arm="${2:?ctl|fp8|dsa|dense66k|fp8131k|ctl131k}"; MAXCTX="${3:-32768}"
  b="$OUT/tp4-$arm"; mkdir -p "$b"
  # `dsa` needs its own DEFAULT max-ctx: the emitter arms the sparse decode only
  # above its 65536 crossover (`GlmCfg::dsa`), so a 32768 blob with PLOW_GLM_DSA=1
  # emits the DENSE path and the arm would silently be a second control.
  fp8kv=0; dsa=0
  case "$arm" in
    ctl) ;;
    fp8) fp8kv=1 ;;
    dsa) dsa=1; MAXCTX="${3:-66560}" ;;
    dense66k) MAXCTX="${3:-66560}" ;;
    fp8131k) fp8kv=1; MAXCTX="${3:-131072}" ;;
    ctl131k) MAXCTX="${3:-131072}" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  env GLM_FULL=1 PLOW_FP8=1 PLOW_GLM_FP8_KV="$fp8kv" PLOW_GLM_DSA="$dsa" \
      PLOW_MLA_PREFILL="$LADDER" PLOW_MLA_PF_V2=1 PLOW_GLM_PF_NS=1 \
      GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1 \
      PLOW_GLM_FUSE_SEAM=1 PLOW_GLM_FUSE_B1=1 \
      PLOW_MOE_PF_DET=1 \
      PLOW_DECODE_BATCH_LADDER=1 PLOW_TUNEDB="$TUNEDB" \
    "$BIN/plowc" --hf-dir "$CKPT" --emit devblob --gpu MI300X --arch gfx942 \
      --max-ctx "$MAXCTX" --n-cu "${NCU:-0}" --num-gpus 4 --out "$b/model.pkt" || exit 1
  ln -sfn "$CKPT" "$b/checkpoint"
  ln -sfn "$OBJ"  "$b/hsaco"
  for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
    [ -e "$RAW/$f" ] && ln -sfn "$RAW/$f" "$b/$f"
  done
  cat > "$b/weights.json" <<JSON
{ "network": "glm-5.3", "gpu": "mi300x", "num_gpus": 4, "parallel": "tp",
  "weight_shared": false, "weight": null, "kv": null, "fusion": null, "buckets": [],
  "static_tensors": [], "static_tensors_file_emitted": false, "weight_tiling": null }
JSON
  ls -l "$b/model.pkt" | awk '{printf "   model.pkt %.1f MB\n", $5/1048576}'
  ;;

serve)
  b="${2:?bundle dir}"; port="${3:?port}"
  [ -f "$b/model.pkt" ] || { echo "!! $b/model.pkt missing"; exit 1; }
  grep -aq libhsa-runtime64 "$BIN/plowrt" || { echo "!! plowrt has no HSA"; exit 1; }
  exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-14400}" PLOW_TP_NO_AUDIT=1 \
    "$LEASE" -n 4 "glm53-kv-$(basename "$b")" \
    nix develop /app/plow --command bash "$WT/scripts/glm53_serve_inner.sh" "$b" "$port" "$OBJ" "$BIN"
  ;;

stop)
  pat="${2:?assets pattern}"
  pids=$(pgrep -f "plowrt serve --assets $pat" || true)
  [ -z "$pids" ] && { echo "no plowrt serving $pat"; exit 0; }
  echo "stopping: $pids"; kill -TERM $pids 2>/dev/null
  for i in $(seq 1 60); do pgrep -f "plowrt serve --assets $pat" >/dev/null 2>&1 || break; sleep 2; done
  pgrep -f "plowrt serve --assets $pat" >/dev/null 2>&1 && { kill -KILL $pids 2>/dev/null; sleep 5; }
  echo "stopped"
  ;;

*) echo "usage: $0 {emit|serve|stop} ..."; exit 2 ;;
esac
