#!/usr/bin/env bash
# Compile the entire Gemma 4 family through plowc targeting MI350X.
# Generates all assets for the production bucket ladder:
#   Decode: batch 1/8/32, seq 1
#   Prefill: batch 1, seq 128/512/2048/8192
#
# This script is meant to run on the MI350X box (with sufficient RAM for the
# egglog exploration). CI uses the smaller test in
# crates/plowc/tests/gemma_family_mi350x.rs instead.
#
# Usage: nix develop --command bash scripts/compile_gemma_family.sh [OUT_DIR]

set -euo pipefail

OUT_DIR="${1:-out}"
EXAMPLES="crates/plowc/examples"

MODELS=(
    "transformer_block_gemma4_12b:gemma4-12b-mi350x"
    "transformer_block_gemma4_31b:gemma4-31b-mi350x"
    "moe_gemma4_26b_a4b:gemma4-26b-moe-mi350x"
)

BATCHES="1,8,32"
SEQS="1,128,512,2048,8192"
GPU="MI350X"
PAGE_KIB=16

echo "=== Gemma Family MI350X Full Compilation ==="
echo "GPU: $GPU  |  page: ${PAGE_KIB} KiB  |  batches: $BATCHES  |  seqs: $SEQS"
echo "Output root: $OUT_DIR"
echo ""

for entry in "${MODELS[@]}"; do
    IFS=":" read -r json_stem out_name <<< "$entry"
    json_path="$EXAMPLES/${json_stem}.json"
    model_out="$OUT_DIR/$out_name"

    if [ ! -f "$json_path" ]; then
        echo "ERROR: $json_path not found"
        exit 1
    fi

    echo "--- Compiling $json_stem → $model_out ---"
    mkdir -p "$model_out"

    cargo run -p plowc --release -- \
        --net "$json_path" \
        --gpu "$GPU" \
        --num-gpus 1 \
        --parallel tp \
        --batches "$BATCHES" \
        --seqs "$SEQS" \
        --phases prefill,decode \
        --page-kib "$PAGE_KIB" \
        --counter-elim \
        --scope-narrow \
        --prefetch \
        --sram-fit \
        --emit-sample \
        --emit-tokenize \
        --out "$model_out"

    echo ""
    echo "  ✓ $(ls "$model_out"/*.pkt 2>/dev/null | wc -l) packet streams"
    echo "  ✓ weights.json: $(wc -c < "$model_out/weights.json") bytes"
    echo "  ✓ assets.json:  $(wc -c < "$model_out/assets.json") bytes"
    echo ""
done

echo "=== Done. All assets in $OUT_DIR/ ==="
echo ""
echo "Asset summary:"
for entry in "${MODELS[@]}"; do
    IFS=":" read -r _ out_name <<< "$entry"
    model_out="$OUT_DIR/$out_name"
    echo "  $out_name:"
    echo "    packets:  $(ls "$model_out"/*.pkt 2>/dev/null | wc -l) files"
    echo "    maps:     $(ls "$model_out"/*.map.json 2>/dev/null | wc -l) files"
    echo "    footprint: $model_out/footprint.csv"
    if [ -f "$model_out/assets.json" ]; then
        echo "    HBM peak: $(grep -o '"total_hbm_peak": [0-9]*' "$model_out/assets.json" | head -1)"
    fi
done
