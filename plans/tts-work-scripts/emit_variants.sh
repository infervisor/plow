#!/usr/bin/env bash
# Veena fusion variants on top of the GF3 packet, objects with MMA_B1=0.
set -e
for v in kvhnr:PLOW_FUSE_KV_HNR merge:PLOW_FUSE_MERGE; do
  name=${v%%:*}; knob=${v##*:}
  env PLOW_TTS_PROFILE=veena $knob=1 PLOW_EXTRA_DEFINES_UNUSED=1 bash /root/tts-work/emit_veena.sh /root/tts-work/assets/veena-$name > /root/tts-work/emit_$name.log 2>&1 || { grep -E "panicked|error" /root/tts-work/emit_$name.log | head -n 3; continue; }
  bash /root/tts-work/mk_trace_asset.sh /root/tts-work/assets/veena-$name /root/tts-work/assets/veena-$name-b0 -DPLOW_NV_GEMV_MMA_B1=0 | grep -c built
  mkdir -p /root/tts-work/assets/veena-$name-b0/codec
  echo "$name ok: $(/root/tts-work/target/release/plowrt disasm /root/tts-work/assets/veena-$name --program 1 2>/dev/null | grep -c '^#') insts"
done
