#!/usr/bin/env bash
# Emit Gemma-4-31B gfx950 devblobs at PLOW_DECODE_BATCH = 1,2,4,8.
# Rust runs INSIDE nix; plowc needs no GPU and no ROCm.
set -u
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT=/home/lava/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
MAXCTX="${MAXCTX:-16384}"
for B in 1 8 16; do
  OUT=/home/lava/plow/build-amd/g31b-db$B
  echo "######## PLOW_DECODE_BATCH=$B -> $OUT (max-ctx $MAXCTX)"
  PLOW_DECODE_BATCH=$B "$WT/target/release/plowc" \
     --hf-dir "$CKPT" --emit devblob --arch gfx950 --gpu mi355x \
     --n-cu 256 --max-ctx "$MAXCTX" --out "$OUT" 2>&1 | tail -12
  python3 - "$OUT/build.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("  -> decode_batch:",d['shapes'].get('decode_batch'),"gv_mm_max:",d.get('tuning',{}).get('gv_mm_max'))
PY
done
