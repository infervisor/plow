#!/usr/bin/env bash
# build_gemma4_h100_assets.sh — assemble plowrt serve-asset dirs for gemma-4 on
# H100 NVL (sm_90a, 132 SMs).
#
#   scripts/build_gemma4_h100_assets.sh <model-dir> <out-root> [max_ctx]
#
# Produces <out-root>/{bf16,fp8}/ , each a directory `plowrt serve --assets` accepts:
#     model.pkt  interp_sm90a.cubin  interp_sm90a_pf.cubin
#     tokenizer.json  weights.json  checkpoint/
#
# Adapted from perf-data/gemma26b-plowrt-sweep.md (sm_120, 188 SM) — the only
# deltas are -arch/cubin names and n_cu=132. NO GPU is used here; this is
# compile + file layout only, so it does NOT take a gpulease.
set -euo pipefail

MODEL="${1:?usage: build_gemma4_h100_assets.sh <model-dir> <out-root> [max_ctx]}"
OUT="${2:?usage: build_gemma4_h100_assets.sh <model-dir> <out-root> [max_ctx]}"
CTX="${3:-8192}"
NCU="${NCU:-132}"                       # H100 NVL SM count
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUBIN_DIR="${CUBIN_DIR:-/workspace/assets/cubin-sm90a}"
PLOWC="$ROOT/target/release/plowc"       # workspace bin dir; gemma4 is a bin target
GEMMA4="$ROOT/target/release/gemma4"
FP8_TWIN="${FP8_TWIN:-$MODEL/fp8-full-plow}"

[ -x "$GEMMA4" ] || { echo "FATAL: $GEMMA4 missing — build with: nix develop -c cargo build --release -p plowc" >&2; exit 1; }
for f in interp_sm90a.cubin interp_sm90a_pf.cubin; do
  [ -f "$CUBIN_DIR/$f" ] || { echo "FATAL: $CUBIN_DIR/$f missing — run scripts/build_sm90a_cubin.sh" >&2; exit 1; }
done

# ---- packets ---------------------------------------------------------------
# PLOW_NS_FULL_ABS=33 is the H100 value: aligned = n_cu/gcd(n_grp,n_cu) =
# 132/gcd(4,132) = 33 (see scripts/build_sm90a_cubin.sh's note). The sm_120 sweep
# used 48; on H100 that costs +41-47%. It is an emit-time decode-scheduling knob
# baked into the packet; correctness holds at any value, only speed depends on it.
mkdir -p "$OUT/bf16" "$OUT/fp8"
echo ">>> emitting bf16 packet (ctx=$CTX, n_cu=$NCU)"
PLOW_UNISEG=1 PLOW_NS_FULL_ABS=33 "$GEMMA4" "$MODEL" "$CTX" "$OUT/bf16/model.pkt" "$NCU"

if [ -d "$FP8_TWIN" ]; then
  echo ">>> emitting fp8 packet (twins at $FP8_TWIN)"
  PLOW_UNISEG=1 PLOW_NS_FULL_ABS=33 PLOW_FP8=1 \
    "$GEMMA4" "$MODEL" "$CTX" "$OUT/fp8/model.pkt" "$NCU"
else
  echo "!!! fp8 twins absent ($FP8_TWIN) — run perf-data/tools/quantize_fp8.py first; skipping fp8"
  rmdir "$OUT/fp8" 2>/dev/null || true
fi

# ---- per-precision asset dirs ----------------------------------------------
# The engine scans every *.safetensors under checkpoint/ and maps tensors BY NAME
# (bf16 = "model.*", fp8 = "fp8/model.*"), so an fp8 dir carries BOTH sets with
# non-colliding FILE names.
link_common() {  # <ckpt-dir>
  local c="$1"
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json; do
    [ -f "$MODEL/$f" ] && ln -sf "$MODEL/$f" "$c/$f"
  done
}

for prec in bf16 fp8; do
  d="$OUT/$prec"; [ -d "$d" ] || continue
  ck="$d/checkpoint"; rm -rf "$ck"; mkdir -p "$ck"
  ln -sf "$CUBIN_DIR/interp_sm90a.cubin"    "$d/interp_sm90a.cubin"
  ln -sf "$CUBIN_DIR/interp_sm90a_pf.cubin" "$d/interp_sm90a_pf.cubin"
  ln -sf "$MODEL/tokenizer.json"            "$d/tokenizer.json"
  # The README's minimal `{"buckets": []}` does NOT deserialize: plow_asset::Manifest
  # requires network/gpu/num_gpus/parallel/weight_shared, none of which carry
  # #[serde(default)]. `network` is not cosmetic — plowrt registers the model under
  # it (orch/registry.rs:27) and it is the name clients pass as "model".
  cat > "$d/weights.json" <<JSON
{
  "network": "$(basename "$MODEL")",
  "gpu": "H100 NVL",
  "num_gpus": 1,
  "parallel": "tp",
  "weight_shared": false,
  "buckets": []
}
JSON

  if [ "$prec" = bf16 ]; then
    for s in "$MODEL"/*.safetensors; do ln -sf "$s" "$ck/$(basename "$s")"; done
  else
    for s in "$MODEL"/*.safetensors;   do ln -sf "$s" "$ck/bf16-$(basename "$s")"; done
    for s in "$FP8_TWIN"/*.safetensors; do ln -sf "$s" "$ck/fp8-$(basename "$s")"; done
  fi
  link_common "$ck"
  echo ">>> $d ready ($(ls "$ck"/*.safetensors | wc -l) safetensors)"
done

echo ">>> done: $OUT"
