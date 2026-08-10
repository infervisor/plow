#!/usr/bin/env bash
# §6g-WALK PHASE B: DOES THE WALK AT MM=8 RECOVER B=16?
#
# The falsifiable prediction the walk study left, verbatim: "the walk at MM=8 serving t=16
# should recover B=16 from 142.4 back toward 202.3, because it restores both fusions and the
# non-spilling rung. If it does not, §5 is wrong and the B=16 loss is something else."
#
# §6g-BATCH's device ceiling (Gemma-4-31B bf16, TP1, `amd-bench --batched`, blob AND objects
# matched at each B): 57.9 / 106.5 / 141.7 / **202.3** / 142.4 tok/s at B=1/2/4/8/16. This
# reproduces the two cells that matter with the SAME instrument, which is what makes the
# comparison legal — `amd-bench` is banned for headline vLLM numbers (§0-BENCH) and is the
# correct instrument here precisely because the ladder it is being compared against came from
# it.
#
# THREE ARMS, blob and object matched in each:
#   b8ctl     PLOW_DECODE_BATCH=8                            MM=8,  walk 0   (the 202.3 cell)
#   b16ctl    PLOW_DECODE_BATCH=16                           MM=16, walk 0   (the 142.4 cell)
#   b16walk   PLOW_DECODE_BATCH=16 PLOW_GEMV_MM=8 WALK=1     MM=8,  walk 1   (the new arm)
#
# The walk arm differs from b16ctl in THREE ways at once, and they cannot be separated by this
# experiment — say so rather than attribute the result to one of them:
#   1. no MM=16 accumulator spill (16 -> 4 scratch ops, per the build's own register readback)
#   2. `fuse_qkv` and `glu_fused` come back on, because `gemv_staged_rows` bounds the LDS
#      staging at min(MM,t)*hidden = 8*5376 = 43008 <= 73728 instead of 16*5376 = 86016
#   3. i_decode.co is 552 KB instead of 848 KB (-35%), and §6g-GF8-REGRESSION established that
#      decode-object SIZE alone can cost +32% inside the persistent megakernel
# A positive result confirms the composite, not §5's mechanism specifically.
#
# TOKEN IDENTITY BEFORE ANY ms. `--prompt` with two distinct prompts makes `--batched` compare
# each slot against the FIRST SLOT SHARING ITS PROMPT; comparing everything against slot 0
# reported green while B=16's slots 13/14/15 were wrong (§6g-BATCH's third silent-corruption
# bug). That gate is the whole reason the fusion was disabled rather than fixed.
#
#   $1 phase: emit | run
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:?emit|run}"
CKPT="${GEMMA31B:-/home/lava/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475}"
AB="${AB:-/home/lava/models/walkab}"
PLOWC="${PLOWC:-$WT/target/release/plowc}"
PLOWRT="${PLOWRT:-$WT/target/release/plowrt}"
# Two DIFFERENT prompts, so the same-prompt cross-check has something to compare.
PROMPT="${PROMPT:-2,1596,563,573,6996,529,9822,235336;2,4029,603,573,6221,576,4557,235336}"

emit_one() { # <tag> <batch> <gemv_mm> <walk>
  local tag=$1 b=$2 mm=$3 w=$4
  mkdir -p "$AB/$tag"
  echo "=== emit $tag  batch=$b mm=${mm:-auto} walk=$w"
  PLOW_DECODE_BATCH="$b" PLOW_GEMV_MM="$mm" PLOW_GEMV_WALK="$w" PLOW_TUNE_DUMP=1 \
    nix develop -c "$PLOWC" --hf-dir "$CKPT" --emit devblob --max-ctx 4096 --n-cu 256 \
      --out "$AB/$tag/model.pkt" > "/tmp/walkab_emit_$tag.log" 2>&1 \
    || { echo "emit FAILED, see /tmp/walkab_emit_$tag.log"; tail -20 "/tmp/walkab_emit_$tag.log"; return 1; }
  ln -sfn "$CKPT" "$AB/$tag/checkpoint"
  ln -sfn "$CKPT/tokenizer.json" "$AB/$tag/tokenizer.json"
  cp /home/lava/plow/build-amd/g31b-db16/weights.json "$AB/$tag/weights.json"
  echo "  gemv census: $(grep -c TUNEDUMP_GEMV "/tmp/walkab_emit_$tag.log") lines, \
$(grep TUNEDUMP_GEMV "/tmp/walkab_emit_$tag.log" | awk '{print $NF}' | sort | uniq -c | tr '\n' ' ')"
  echo "  decode arms: $(python3 - "$AB/$tag/build.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for p in d['programs']:
    if p['kind']=='decode': print(' '.join(a for a in p['arms'] if 'Gemv' in a))
PY
)"
}

run_one() { # <tag> <objdir>
  local tag=$1 obj=$2
  echo "############ $tag  (objects $obj)"
  # UNDER `nix develop`: plowrt is nix-linked, and outside the shell its ELF interpreter is a
  # missing /nix/store glibc — which reports as "No such file or directory" on a file that is
  # plainly there (the design notes §0a). Running under the lease is fine; only COMPILING
  # under it is forbidden.
  nix develop -c "$PLOWRT" amd-bench --blob "$AB/$tag/model.pkt" --hsaco "$obj" \
      --checkpoint "$CKPT" --prompt "$PROMPT" --steps 65 --ctx 1024 --batched 2>&1 \
    | grep -E "slots agree|slot [0-9]+ and|tpot|aggregate|batched decode|MISMATCH|rror|refus|Error"
  echo
}

case "$PHASE" in
  emit)
    emit_one b8ctl   8  ""  0
    emit_one b16ctl  16 ""  0
    emit_one b16walk 16 8   1
    ;;
  run)
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    run_one b8ctl   /home/lava/plow/build-amd/walk-ctl-b8
    run_one b16ctl  /home/lava/plow/build-amd/walk-ctl-b16
    run_one b16walk /home/lava/plow/build-amd/walk-mm8-b16
    ;;
  *) echo "usage: $0 emit|run" >&2; exit 2;;
esac
