#!/usr/bin/env bash
# Regenerate the GLM-5.3 TP8 gfx942 SERVING OBJECT SET for one packet.
#
# The set that serves GLM-5.3 on MI300X is not one build: it is the persistent-interpreter family
# (scripts/build_gfx942.sh, derived from the packet's plow_config.h so the low-rung tiers and the
# opt-in arms match what the packet asks for) plus four adapters around pinned vendor code
# objects, each of which its own script hash-checks against the qualified ABI. A set missing one
# of them does not fail loudly at build time -- it fails at load, or silently serves a slower
# route -- which is why they are assembled here rather than by hand.
#
#   scripts/build_glm53_gfx942_serving_objects.sh OUT_DIR ASSETS_DIR VENDOR_DIR   (inside nix develop)
#
# ASSETS_DIR is the packet directory plowc wrote (holds model.pkt + plow_config.h).
# VENDOR_DIR holds the pinned vendor blobs under their shipped names. An EXISTING serving object
# set is a valid VENDOR_DIR -- the vendor blobs are copied verbatim into the set, so the previous
# set is how you regenerate the next one:
#
#   scripts/build_glm53_gfx942_serving_objects.sh /tmp/glm53-objects /tmp/glm53/assets \
#       /tmp/tp-glm53-pswz/serving-safe
#
# The 64-row fmoe object (`..._psx_64x256.co`) is passed as build_moe_aiter.sh's FOURTH argument
# and is REQUIRED here, together with an adapter rebuilt from this tree: the AITER MoE adapter only
# exports `plow_moe_aiter_tile64_abi_1` when built alongside it, and an adapter without that marker
# cannot serve the 64-row tile however many .co files sit beside it (PLOW_MOE_AITER_TILE64,
# docs/flags-reference.md). The frozen /tmp/tp-glm53-pswz/serving-safe set is exactly that case:
# the 64-row .co is present, its adapter predates the marker.
#
# The small/split MLA prefill objects (interp_mla_small*, interp_mla_split_fp8kv*) come from
# build_gfx942.sh like the rest of the interpreter family. A fresh GLM packet needs them and plowrt
# refuses to load without them (d60bdeb2); that is the gap that broke serving-safe for HEAD packets.
set -euo pipefail

if [[ $# != 3 ]]; then
    sed -n '2,30p' "$0" >&2
    exit 2
fi
out=$(mkdir -p "$1" && cd "$1" && pwd)
assets=$2
vendor=$3
root=$(cd "$(dirname "$0")/.." && pwd)

[ -f "$assets/plow_config.h" ] || { echo "no $assets/plow_config.h -- point ASSETS_DIR at the packet plowc wrote" >&2; exit 2; }
for blob in fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co \
            fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3.co \
            fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_psx_64x256.co \
            mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co \
            glm_lt_gfx942.elf; do
    [ -f "$vendor/$blob" ] || { echo "no $vendor/$blob -- VENDOR_DIR must hold the pinned vendor objects" >&2; exit 2; }
done

echo ">>> interpreter family (packet-derived)"
PLOW_HSACO_CONFIG=$assets "$root/scripts/build_gfx942.sh" "$out"

echo ">>> TP DSA adapter"
"$root/scripts/build_dsa_tp.sh" "$out"

echo ">>> AITER MoE adapter + the 32x256, flat 16x128 and 64x256 fmoe objects"
"$root/scripts/build_moe_aiter.sh" "$out" \
    "$vendor/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co" \
    "$vendor/fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3.co" \
    "$vendor/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_psx_64x256.co"

echo ">>> AITER sparse MLA adapter + the QH8 object"
# --single-pass: one AITER attention launch instead of two splits plus a reduce. In flow at the
# served 8192-row chunk (prior 65536) the route is 2746 -> 2466 us/layer, -22 ms/chunk; bench
# 49.08 -> 49.91 out tok/s; retrieval 18/18 (tracker #22). The route engages whenever the
# adapter exports the single-pass kernel, so this flag is what makes it the served default.
"$root/scripts/build_mla_sparse_aiter.sh" "$out" "$vendor/mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co" --single-pass

# The pinned hipBLASLt projection image is already UNBUNDLED in a serving set, and
# build_glm_lt.sh asserts exactly this sha256 after unbundling, so copying it is the same
# artefact by a shorter path. Run build_glm_lt.sh directly when starting from the bundled .co.
echo ">>> pinned hipBLASLt projections"
expected=efa5b0365bedc2effa52265c85eded14fb63febd9c067bab37138d99db607db5
actual=$(sha256sum "$vendor/glm_lt_gfx942.elf")
if [[ ${actual%% *} != "$expected" ]]; then
    echo "glm_lt_gfx942.elf does not match the qualified gfx942 projection image" >&2
    exit 1
fi
cp -f "$vendor/glm_lt_gfx942.elf" "$out/glm_lt_gfx942.elf"

# Only a packet emitted with PLOW_GLM_FOLD_LT routes MlaMergeFold to the FP32 hipBLASLt fold; the
# loader then needs its pinned GEMM image (from VENDOR_DIR) and the adapter built from this tree.
if python3 -c 'import json, sys; k = {x["id"]: x["value"] for x in json.load(open(sys.argv[1]))["emit_config"]["knobs"]}; sys.exit(k.get("glm_fold_lt") != "true")' "$assets/build.json"; then
    echo ">>> native FP32 MLA prefill fold (packet emitted with PLOW_GLM_FOLD_LT)"
    [ -f "$vendor/glm_fold_lt_gfx942.elf" ] || { echo "no $vendor/glm_fold_lt_gfx942.elf -- the packet carries native MLA folds" >&2; exit 2; }
    (cd "$root" && scripts/build_glm_fold_lt.sh "$out" "$vendor/glm_fold_lt_gfx942.elf")
fi

echo ">>> serving object set at $out"
ls "$out"/*.elf | wc -l | sed 's/^/  interpreter + adapter objects: /'
ls -d "$out"/lowrung* 2>/dev/null | wc -l | sed 's/^/  low-rung tier dirs: /'
ls "$out"/*.co | wc -l | sed 's/^/  pinned vendor objects: /'
