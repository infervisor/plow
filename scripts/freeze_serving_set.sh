#!/usr/bin/env bash
# Freeze a SERVING SET: the three artifacts that together reproduce a measured serve, plus the
# record of how they were run.
#
# WHY ALL THREE, AND WHY TOGETHER. `build.json` reproduces the COMPILE and carries a
# `pairing.hash` the loader checks, but a packet alone does not serve: it needs the object set it
# was paired against and a `plowrt` whose PlowProgram layout matches. Every one of those was a
# live failure in this campaign —
#   * a plowrt built without `--features hsa` refused the blob outright (hsa=false),
#   * objects from a different tree gave "kernarg segment is 424 B ... the code object is STALE",
#   * objects built without an arm the packet requires gave "none of [plow_moe_pf_det_arm] is in
#     its symbol table",
#   * and the serve ENV that decides throughput (tiers, chunk, packed-prefill route) appears in
#     none of them.
# A directory holding the packet, the objects, the binary and the replay is the smallest thing
# that answers "what produced this number".
#
#   scripts/freeze_serving_set.sh <assets-dir> <objdir> <plowrt> <out-dir> [serve.log]
#
# The checkpoint is NOT copied — it is hundreds of GB and is named, with its own identity, in
# `build.json`. This freezes what the campaign built, not what it read.
set -euo pipefail
ASSETS="${1:?assets dir (holds model.pkt + build.json)}"
OBJ="${2:?object dir (interp_*.elf)}"
PLOWRT="${3:?plowrt binary}"
OUT="${4:?destination}"
SERVELOG="${5:-}"

[ -f "$ASSETS/model.pkt" ] || { echo "no $ASSETS/model.pkt" >&2; exit 2; }
[ -f "$ASSETS/build.json" ] || { echo "no $ASSETS/build.json" >&2; exit 2; }
[ -x "$PLOWRT" ] || { echo "not executable: $PLOWRT" >&2; exit 2; }
mkdir -p "$OUT/assets" "$OUT/hsaco"

# The packet and its manifest. Symlinks are NOT followed for `checkpoint` (see above); everything
# else in the bundle is small and is copied as content so the set is self-contained.
cp -f "$ASSETS/model.pkt" "$ASSETS/build.json" "$OUT/assets/"
for f in weights.json plow_config.h; do
  [ -f "$ASSETS/$f" ] && cp -f "$ASSETS/$f" "$OUT/assets/"
done
# Tokenizer-side files are what the served PROMPT depends on, and a bundle missing them silently
# falls back to the built-in builders — the defect this campaign found in the frozen bundles.
# Copy content, not the link.
for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
  [ -e "$ASSETS/$f" ] && cp -fL "$ASSETS/$f" "$OUT/assets/" 2>/dev/null || true
done
printf '%s\n' "$(readlink -f "$ASSETS/checkpoint" 2>/dev/null || echo '<none>')" > "$OUT/assets/CHECKPOINT_PATH"

# Objects, including the low-rung tier subdirectories: plowrt discovers those by layout, so a set
# that drops them serves the wide object at every rung and silently loses ~25% output tok/s.
cp -f "$OBJ"/*.elf "$OUT/hsaco/" 2>/dev/null || true
[ -f "$OBJ/build_defines.json" ] && cp -f "$OBJ/build_defines.json" "$OUT/hsaco/"
for d in "$OBJ"/lowrung*; do
  [ -d "$d" ] || continue
  mkdir -p "$OUT/hsaco/$(basename "$d")"
  cp -f "$d"/*.elf "$d"/build_defines.json "$OUT/hsaco/$(basename "$d")/" 2>/dev/null || true
done

cp -f "$PLOWRT" "$OUT/plowrt"

# The serve half of the record. `build.json` has the emit replay; this is the runtime replay, which
# nothing else stores.
if [ -n "$SERVELOG" ] && [ -f "$SERVELOG" ]; then
  sed 's/\x1b\[[0-9;]*m//g' "$SERVELOG" | grep -aE "serve replay|decode tiers|packed prefill|AMD engine ready|L2 hierarchical" \
    > "$OUT/serve-replay.txt" 2>/dev/null || true
fi

{
  echo "frozen:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git:         $(git rev-parse HEAD 2>/dev/null || echo '<not a repo>')"
  echo "assets_src:  $ASSETS"
  echo "objdir_src:  $OBJ"
  echo "plowrt_src:  $PLOWRT"
  echo "pairing:     $(python3 -c "import json;print(json.load(open('$ASSETS/build.json')).get('pairing',{}).get('hash','?'))" 2>/dev/null || echo '?')"
  echo "objects:     $(ls "$OUT/hsaco"/*.elf 2>/dev/null | wc -l) (+ $(ls -d "$OUT/hsaco"/lowrung* 2>/dev/null | wc -l) tier dirs)"
  echo "hsa_linked:  $(grep -ac libhsa-runtime64 "$OUT/plowrt" >/dev/null 2>&1 && echo yes || echo NO)"
} > "$OUT/FROZEN.txt"
cat "$OUT/FROZEN.txt"
du -sh "$OUT"
