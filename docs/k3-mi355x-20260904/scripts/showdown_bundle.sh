#!/usr/bin/env bash
# Build the served showdown bundle from the worktree source: Lean-verified packet, paired objects, pinned binaries.
# Usage: showdown_bundle.sh <bundle-dir> [extra plowc env, e.g. PLOW_GQ_ORDER=asap]
set -euo pipefail
wt=/home/lava/plow/.claude/worktrees/d1-moe-decode-rule; model=/home/shaswot/models/Kimi-K3
B=${1:?bundle dir}; shift
mkdir -p "$B/bin" "$B/hsaco"
cd "$wt"
export PLOW_VERIFY_BIN="$wt/lean-plow/.lake/build/bin/plow_verify"
nix develop -c env PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix K3_FULL=1 PLOW_VERIFY_BIN="$PLOW_VERIFY_BIN" "$@" \
  ./target/release/plowc --hf-dir "$model" --emit devblob --arch gfx950 --gpu mi350 --num-gpus 8 --parallel tp \
  --max-ctx 16384 --n-cu 256 --out "$B/assets" >"$B/emit.log" 2>&1
ln -sfn "$model" "$B/assets/checkpoint"; ln -sfn "$model/tokenizer.json" "$B/assets/tokenizer.json"
echo "lean: $(jq -c '.lean | {verified, oracle, reason}' "$B/assets/build.json")"
echo "tiles: $(jq -c '.tuning' "$B/assets/build.json"); pairing $(jq -r .pairing.hash "$B/assets/build.json"); sha $(sha256sum "$B/assets/model.pkt" | cut -c1-16); decode segs $(jq -c '[.programs[] | select(.kind=="decode")] | length' "$B/assets/build.json")"
nix develop -c bash -c 'exec cmake "$@" -DPLOW_HSACO_HIPCC="$PLOW_HIPCC" -DPLOW_HSACO_BUNDLER="$PLOW_BUNDLER"' bash \
  -S runtime -B "$B/cmake" -DPLOW_GFX950_HSACO=ON -DPLOW_HSACO_ARCH=gfx950 \
  -DPLOW_HSACO_CONFIG="$B/assets/plow_config.h" -DPLOW_HSACO_DIR="$B/hsaco" \
  -DPLOW_HSACO_DECODE_INVENTORY_PRUNE=ON -DPLOW_HSACO_DECODE_MLA_SEGMENTS=ON -DPLOW_HSACO_KDA_KEY_FACTOR=OFF \
  -DPLOW_HSACO_MOE_DECODE_GROUPED=ON >"$B/config.log" 2>&1
nix develop -c cmake --build "$B/cmake" --target gfx950_hsaco -j 24 >"$B/build.log" 2>&1
echo "objects: $(ls "$B/hsaco" | grep -c '\.elf$') ($(ls "$B/hsaco" | grep -c moe_decode_grouped) grouped)"
nix develop -c cargo build --release -p plowrt --features hsa >"$B/build-plowrt.log" 2>&1
cp -f target/release/plowrt "$B/bin/plowrt"; cp -f target/release/plowc "$B/bin/plowc"
git rev-parse HEAD > "$B/source-head.txt"
echo BUNDLE_DONE
