#!/usr/bin/env bash
# Emit the flag-free default packet and compare it with the gated stack-3 packet.
set -uo pipefail
wt=/home/lava/plow/.claude/worktrees/d1-moe-decode-rule; model=/home/shaswot/models/Kimi-K3
out=/tmp/k3-defaultcheck; rm -rf "$out"; mkdir -p "$out"
cd "$wt"
nix develop -c env PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix K3_FULL=1 PLOW_VERIFY_BIN="$wt/lean-plow/.lake/build/bin/plow_verify" \
  ./target/release/plowc --hf-dir "$model" --emit devblob --arch gfx950 --gpu mi350 --num-gpus 8 --parallel tp \
  --max-ctx 16384 --n-cu 256 --out "$out/assets" > "$out/emit.log" 2>&1
echo "rc=$?"
echo "default: $(sha256sum "$out/assets/model.pkt" | cut -c1-16) pairing $(jq -r .pairing.hash "$out/assets/build.json") lean $(jq -c '.lean|{verified,oracle}' "$out/assets/build.json")"
echo "stack3:  $(sha256sum /tmp/k3-stack3/assets/model.pkt | cut -c1-16) pairing $(jq -r .pairing.hash /tmp/k3-stack3/assets/build.json)"
grep -c "want gfx950" "$out/emit.log"
