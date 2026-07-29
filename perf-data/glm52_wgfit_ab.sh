#!/usr/bin/env bash
# GLM-5.2 TP4 decode A/B: MlaMergeFold + fusion-A GemvQkv sized to their own work items.
# ONE lease, controls interleaved at positions 1/3/5. Device object is HEAD's for BOTH arms
# (runtime/ is untouched by this change) -- the ONLY difference is the packet's `blocks` field.
set -u
D=/home/lava/models/glm52_wgfit
W=/home/lava/models/GLM-5.2-plow
OUT=$D/raw.txt
: > "$OUT"
for arm in ctl fit ctl fit ctl; do
  echo "########## ARM $arm  $(date -Is)" | tee -a "$OUT"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  PLOW_INTERP=$D/i_base.elf "$D/glm52_decode" "$D/$arm.pkt" "$W" \
      --tp 4 --sweep 1024 --steps 65 --gen 24 2>&1 | tee -a "$OUT"
done
echo "########## DONE $(date -Is)" | tee -a "$OUT"
touch "$D/run.done"
