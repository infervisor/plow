#!/usr/bin/env bash
# build_gemma_family_sm120.sh — compile every Gemma-4 model for sm_120 and
# assemble one `plowrt serve` asset dir per (model, precision, batch).
#
#   scripts/build_gemma_family_sm120.sh [OUT_ROOT] [BATCHES]
#     OUT_ROOT default /root/gpu-assets-family, BATCHES default "1"
#
# Every packet is compiled with the campaign's shipping flags:
#   PLOW_UNISEG=1        mandatory — segmented programs trap the coarse-only
#                        sm_120 interpreter (see plans/p9-26b-campaign.md)
#   PLOW_NS_FULL_ABS=48  pairs with the GF_FULL=4 cubins below
# fp8 packets additionally carry PLOW_FP8_HEAD (fp8 lm_head twin) — reported as
# its own row, since vLLM's fp8 recipe keeps lm_head bf16.
#
# NOTE: cmake/nvcc must run OUTSIDE `nix develop` (nix glibc lands in RUNPATH
# and the binary segfaults before main). cargo is fine inside.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-/root/gpu-assets-family}"
BATCHES="${2:-1}"
PLOWC="$ROOT/target/release/plowc"
CTX="${CTX:-132096}"
NCU="${NCU:-188}"

# model dir : slug : fp8-twin dir ("" = no fp8 twin available)
MODELS=(
  "/workspace/models/gemma-4-12B-it:gemma-4-12b-it:/workspace/models/gemma-4-12B-it/fp8"
  "/workspace/models/gemma-4-31B-it:gemma-4-31b-it:"
  "/workspace/models/gemma-4-26B-A4B-it:gemma-4-26b-a4b-it:/workspace/models/gemma-4-26B-A4B-it/fp8-full-plow"
)

echo "=== Gemma-4 family → sm_120 assets (ctx $CTX, n_cu $NCU, batches: $BATCHES) ==="
command -v nvcc >/dev/null || export PATH=/usr/local/cuda/bin:$PATH
nix develop "$ROOT" --command cargo build --release -p plowc --bin plowc

# One cubin pair for every asset dir: GF_FULL=4 (measured win at long ctx) and
# self-describing arena size, which `plowrt serve` reads back at load.
CUBIN_TMP="$(mktemp -d)"
sed 's/-DPLOW_NV_FA_GF=2/-DPLOW_NV_FA_GF=2 -DPLOW_NV_FA_GF_FULL=4/g' \
    "$ROOT/scripts/build_sm120_cubin.sh" > "$CUBIN_TMP/build_cubin.sh"
bash "$CUBIN_TMP/build_cubin.sh" "$CUBIN_TMP/interp_sm120.cubin"

for entry in "${MODELS[@]}"; do
  IFS=":" read -r mdir slug fp8dir <<< "$entry"
  [ -d "$mdir" ] || { echo "SKIP $slug (no $mdir)"; continue; }
  for b in $BATCHES; do
    for prec in bf16 fp8; do
      [ "$prec" = fp8 ] && [ -z "$fp8dir" ] && continue
      name="$slug-$prec-b$b"
      dir="$OUT_ROOT/$name"
      mkdir -p "$dir"
      echo "--- $name"
      # plowc emits the PLOWDEV model.pkt AND a servable weights.json into
      # "$dir" in one shot (bundle mode) — the checkpoint's lowercased dir name
      # is the network slug, matching "$slug". Replaces the former gemma4 bin +
      # hand-written stub manifest.
      env PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 PLOW_DECODE_BATCH="$b" \
          ${prec:+$([ "$prec" = fp8 ] && echo "PLOW_FP8=1 PLOW_FP8_HEAD=1")} \
          "$PLOWC" --hf-dir "$mdir" --emit devblob --max-ctx "$CTX" --n-cu "$NCU" \
          --gpu "RTX 6000 Pro Blackwell" --out "$dir" | tail -1
      cp "$CUBIN_TMP"/interp_sm120*.cubin "$dir/"
      cp "$mdir/tokenizer.json" "$dir/"
      ln -sfn "$mdir" "$dir/checkpoint"
      [ -n "$fp8dir" ] && [ "$prec" = fp8 ] && ln -sfn "$fp8dir" "$dir/fp8"
    done
  done
done
rm -rf "$CUBIN_TMP"
echo "=== done → $OUT_ROOT"
