#!/usr/bin/env bash
# emit_veena.sh <out-dir> [extra plowc args...]  — Veena packet + paired sm_90a objects.
set -eo pipefail
source /root/tts-work/cuda-env.sh
WT=/root/plow/.claude/worktrees/tts-veena-chatterbox
cd $WT
export PLOW_VERIFY_BIN=$WT/lean-plow/.lake/build/bin/plow_verify
VEENA=$(ls -d /root/tts-work/hf/hub/models--maya-research--Veena/snapshots/*)
OUT=${1:?out dir}; shift
mkdir -p "$(dirname $OUT)"
time /root/tts-work/target/release/plowc --hf-dir "$VEENA" --gpu h200 --arch sm_90a \
  --max-ctx ${MAXCTX:-2048} --emit ${EMIT:-devblob+cubin} --out "$OUT" "$@"
ls -la "$OUT"
