#!/bin/sh
# ONE block, for kernel work. Layer 2 is representative of the bulk: compress_ratio 2 so it
# carries the gathered attention, plus the MoE and the mHC. Layers 0/1 are window-only and 14 has
# Engram, so neither is the common case.
cd /app/plow/.claude/worktrees/dsv41-bringup || exit 1
D=/workspace/dsv41-blk2
mkdir -p "$D"
rm -f "$D/blk2.pkt" "$D/emit.log"
PLOW_MLA_PREFILL=8192 ./target/release/plowc \
  --hf-dir /workspace/models/DeepSeek-V4.1-Flash --block 2 --num-gpus 8 \
  --gpu mi300x --arch gfx942 --max-ctx 8192 --no-knob-verify \
  --out "$D/blk2.pkt" > "$D/emit.log" 2>&1
echo "emit rc=$? bytes=$(ls -l $D/blk2.pkt 2>/dev/null | awk '{print $5}')"
tail -3 "$D/emit.log"
