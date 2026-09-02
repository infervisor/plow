#!/usr/bin/env bash
# Interleaved A/B for L2-domain placement on Gemma-4-31B decode, gfx950.
#
# INTERLEAVED, not sequential, because §6b-STALE: a knob measured against one interpreter does not
# transfer, and `GLM_MOE_CORESIDENT=2` measured -1.09, +1.32 and -0.79 across three folds each
# correct against its own control. So both arms run in the same lease, alternating, and every fold
# reports its own pair.
#
# Arm A (control):   unplaced blob + objects built without -DPLOW_L2_PLACE_DISPATCH
# Arm B (treatment): placed blob   + objects built with it, PLOW_L2_PLACE_DISPATCH=1 to pass the
#                    devblob F_L2DOM load guard.
#
# Run INSIDE nix (plowrt is nix-linked) but under a lease. `gpulease` exports both
# ROCR_ and HIP_VISIBLE_DEVICES to the same absolute id and they COMPOSE, so HIP_ is unset here
# (knob-contract §0a).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${PLOW_AB_DIR:-/tmp/l2place}"
CKPT="${PLOW_CKPT:-$(readlink -f /home/lava/plow/build-amd/g31b-bf16/checkpoint)}"
RT="$REPO/target/release/plowrt"
STEPS="${STEPS:-64}"
FOLDS="${FOLDS:-3}"
PROMPT="${PROMPT:-2,106,1645,108,7154,1701,532,573,6996,529,8043,236881,107,108,106,2516,108}"

unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

run() { # <arm> <blob> <objs> <extra-env>
  local tag="$1" blob="$2" objs="$3" env="$4"
  env $env "$RT" amd-bench --blob "$blob" --hsaco "$objs" --checkpoint "$CKPT" \
      --prompt "$PROMPT" --steps "$STEPS" 2>&1 \
    | tee "$D/raw.$tag.txt" \
    | grep -Ei 'ms/token|error:|panic' | sed "s/^/[$tag] /"
  # TOKEN IDENTITY. Placement is a SCHEDULING change: it moves which workgroup runs which packet
  # and nothing else, so the ids must not move at all. If they do, the mapping is wrong.
  grep -A2 -i 'greedy decode' "$D/raw.$tag.txt" | tail -n +2 > "$D/ids.$tag.txt"
}

for f in $(seq 1 "$FOLDS"); do
  echo "########## fold $f ##########"
  echo "--- A: control (unplaced) ---"
  run "A$f" "$D/off/model.pkt" "$D/objs_off" ""
  echo "--- B: treatment (L2-placed) ---"
  run "B$f" "$D/on/model.pkt" "$D/objs_on" "PLOW_L2_PLACE_DISPATCH=1"
done
