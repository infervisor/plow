#!/usr/bin/env bash
# Driver for the DSA (sparse-attention) qualification on this worktree. Exists so the
# gpulease -> nix develop -> bash chain is one plain command with no nested quoting.
#
# WT is this worktree, not /app/plow: plowc and plowrt are built here and the tuning store
# is keyed on the toolchain + preprocessed-source digest, so a campaign measured against
# another tree's plowc is stale on arrival.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$WT/target-glm53/release"
OBJ="${PLOW_HSACO:-/app/plow/build-glm53/hsaco}"
CKPT="${PLOW_CKPT:-/workspace/models/GLM-5.3-plow-lite}"
OUT="$WT/build-glm53"
LEASE="/app/plow/perf-data/tools/gpulease"

case "${1:?tune|emit|serve}" in

tune)
  exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-21600}" "$LEASE" -n 1 glm53-dsa-tune \
    nix develop /app/plow --command "$WT/scripts/glm53_tune_gemm_inner.sh" \
      "$WT" "$OBJ" "$CKPT" "${2:-/tmp/glm53-tune-tp4.jsonl}" "${3:-4}"
  ;;

# emit <arm> <maxctx> — `dsa` arms PLOW_GLM_DSA=1, `dense` forces it off. Everything else is
# held identical between the two, INCLUDING PLOW_GLM_FUSE_ROPE, which the DSA gate refuses
# (the q-rope fold and the gather arm both want t[7]) and which the dense arm therefore also
# gives up, so the A/B measures sparse attention and not a lost fold.
emit)
  arm="${2:?dsa|dense}"; ctx="${3:-131072}"; b="$OUT/tp4-$arm"
  case "$arm" in
    dsa)   dsa=1 ;;
    dense) dsa=0 ;;
    *) echo "!! arm must be dsa or dense"; exit 2 ;;
  esac
  mkdir -p "$b"
  nix develop /app/plow --command env \
      GLM_FULL=1 PLOW_FP8=1 \
      PLOW_MLA_PREFILL="${LADDER:-full:128,512,2048,8192}" PLOW_MLA_PF_V2=1 PLOW_GLM_PF_NS=2 \
      PLOW_GLM_DSA="$dsa" \
      GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1 \
      PLOW_GLM_FUSE_SEAM=1 PLOW_GLM_FUSE_B1=1 PLOW_MOE_PF_DET=1 \
      PLOW_DECODE_BATCH_LADDER=1 \
    "$BIN/plowc" --hf-dir "$CKPT" --emit devblob --gpu MI300X --arch gfx942 \
      --max-ctx "$ctx" --n-cu 0 --num-gpus 4 --out "$b/model.pkt" 2>&1 | tail -4 || exit 1
  ln -sfn "$CKPT" "$b/checkpoint"
  ln -sfn "$OBJ"  "$b/hsaco"
  for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
    [ -e "${GLM_RAW:-/workspace/models/GLM-5.3-FP8}/$f" ] &&
      ln -sfn "${GLM_RAW:-/workspace/models/GLM-5.3-FP8}/$f" "$b/$f"
  done
  cat > "$b/weights.json" <<JSON
{ "network": "glm-5.3", "gpu": "mi300x", "num_gpus": 4, "parallel": "tp",
  "weight_shared": false, "weight": null, "kv": null, "fusion": null, "buckets": [],
  "static_tensors": [], "static_tensors_file_emitted": false, "weight_tiling": null }
JSON
  ls -l "$b/model.pkt" | awk '{printf "   %s model.pkt %.1f MB\n", "'"$arm"'", $5/1048576}'
  ;;

# attrib <ctx> — decode-only timing + a packet trace for BOTH arms under ONE lease.
# `amd-bench` drives the decode program directly (no HTTP, no prefill), which is the only
# way to see the indexer chain's per-op cost separated from everything the mux overlaps.
attrib)
  ctx="${2:-32768}"; steps="${3:-32}"
  exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-21600}" "$LEASE" -n 4 glm53-dsa-attrib \
    nix develop /app/plow --command bash -c '
      set -u
      export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
      for arm in dsa dense; do
        echo "===== $arm ctx='"$ctx"' steps='"$steps"'"
        PLOW_TRACE_RAW="/tmp/trace-$arm.bin" PLOW_MLA_PF_V2=1 \
          "'"$BIN"'/plowrt" amd-bench --blob "'"$OUT"'/tp4-$arm/model.pkt" \
            --hsaco "'"$OBJ"'" --checkpoint "'"$CKPT"'" \
            --ctx '"$ctx"' --steps '"$steps"' --tp 4 2>&1 | tail -6
      done'
  ;;

serve)
  arm="${2:?dsa|dense}"; port="${3:?port}"; b="$OUT/tp4-$arm"
  [ -f "$b/model.pkt" ] || { echo "!! $b/model.pkt missing"; exit 1; }
  grep -aq libhsa-runtime64 "$BIN/plowrt" || { echo "!! plowrt has no HSA"; exit 1; }
  exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-21600}" "$LEASE" -n 4 "glm53-$arm" \
    nix develop /app/plow --command bash "$WT/scripts/glm53_serve_inner.sh" "$b" "$port" "$OBJ" "$BIN"
  ;;
esac
