#!/usr/bin/env bash
# Emit a Gemma-4-31B gfx950 devblob with and without L2-domain placement, and show what changed.
#
# The OFF blob must be byte-identical to a pre-change build, and the ON blob must report 8 domains
# on the DECODE program only (prefill keeps its wave-class segments). Run INSIDE nix — plowc is a
# Rust binary and needs the nix toolchain, unlike the ROCm objects.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${PLOW_CKPT:-$(readlink -f /home/lava/plow/build-amd/g31b-bf16/checkpoint)}"
OUT="${1:-/tmp/l2place}"
CTX="${PLOW_CTX:-4096}"
PLOWC="$REPO/target/release/plowc"

mkdir -p "$OUT/off" "$OUT/on"

echo "=== OFF (control) ==="
"$PLOWC" --hf-dir "$CKPT" --emit devblob --arch gfx950 --gpu mi355x --n-cu 256 \
  --max-ctx "$CTX" --out "$OUT/off/model.pkt" 2>&1 | grep -Ei 'l2 placement|prog |packets' | head -20

echo
echo "=== ON (PLOW_L2_PLACE=1) ==="
PLOW_L2_PLACE=1 "$PLOWC" --hf-dir "$CKPT" --emit devblob --arch gfx950 --gpu mi355x --n-cu 256 \
  --max-ctx "$CTX" --out "$OUT/on/model.pkt" 2>&1 | grep -Ei 'l2 placement|prog |packets' | head -20

echo
echo "=== blob sizes ==="
ls -l "$OUT/off/model.pkt" "$OUT/on/model.pkt" | awk '{print "   ", $NF, $5"B"}'
if cmp -s "$OUT/off/model.pkt" "$OUT/on/model.pkt"; then
  echo "   blobs IDENTICAL -- placement did not engage"
else
  echo "   blobs DIFFER -- placement engaged"
fi
