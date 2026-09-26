#!/usr/bin/env bash
# emit_t3.sh <out-dir> [extra plowc args] — Chatterbox T3 packet (+ objects unless EMIT=devblob).
set -eo pipefail
source /root/tts-work/cuda-env.sh
WT=/root/plow/.claude/worktrees/tts-veena-chatterbox
cd $WT
export PLOW_VERIFY_BIN=$WT/lean-plow/.lake/build/bin/plow_verify
OUT=${1:?out}; shift
/root/tts-work/target/release/plowc --hf-dir /root/tts-work/hf-t3 --gpu h200 --arch sm_90a \
  --max-ctx ${MAXCTX:-2048} --emit ${EMIT:-devblob+cubin} --out "$OUT" "$@"
