#!/usr/bin/env bash
# Single-block plow-vs-vLLM comparison under FP8, both Gemma-4-26B-A4B layer kinds.
#
# block_e2e.sh emits the block with a bare `plowc --emit devblob+cubin`, so the precision comes
# entirely from the emit ENV. These are the FP8 recipe's own emit/objects vars
# (gemma4-26b-a4b.h100.fp8-ctx16k.toml), not invented:
#   PLOW_FP8 + PLOW_W8A8   — W8A8 is NOT optional on a MoE model: with PLOW_FP8 alone `moe_pf` is
#                            false and the grouped fp8 MoE prefill never turns on.
#   PLOW_SEG_PURE_GEMM=fp8 / PLOW_PF_SEG_PURE=fp8 — the fp8 segment class (block_e2e defaults this
#                            to 1, which is the BF16 class and faults the block).
#   PLOW_SEG_FA512 / PLOW_SEG_FA256_GQA2 — [FP8-TAX] both defaults are bf16-gated.
#   PLOW_BUILD_W8A8 + the three PLOW_BUILD_* — the object set; -DPLOW_NV_W8A8=1 is mandatory or the
#                            w8a16 body misreads the operands.
set -uo pipefail
W=/home/lava/plow/.claude/worktrees/gemma4-26b-beat-vllm
T=/home/lava/.claude/jobs/ef9d0e7f/tmp
MODEL=/opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it
cd "$W" || exit 1

export GPU_LEASE_TIMEOUT=43200
# emit
export PLOW_FP8=1 PLOW_W8A8=1
export PLOW_SEG_PURE_GEMM=fp8
export PLOW_PF_SEG_PURE=fp8
export PLOW_SEG_FA512=1 PLOW_SEG_FA256_GQA2=1
export PLOW_SLIDING_NS_CAP=1 PLOW_SLIDING_NS_GRID=1
export PLOW_ATTENTION_DECODE_BALANCE_GF=4
export PLOW_TMA_GEMM=1 PLOW_FA_MMAQK=3
export PLOW_FUSE_ARGMAX=0 PLOW_GEMMA_MOE_ROUTER_EXACT=0
# objects
# The fp8 WEIGHTS live in a separate pre-quantised MIXED checkpoint (bf16 shards + one fp8
# shard), which is what `MISSING WEIGHT: fp8/...experts.gate_up_proj` was telling us: the bf16
# dir has no fp8 tensors. It is a symlink farm with no HF index and non-standard shard names, so
# it cannot be the vLLM half's model dir -- point only plowrt at it (engine.rs: "Point
# PLOW_CHECKPOINT at the weights"), and leave $MODEL as the real HF dir for plowc and vLLM.
export PLOW_CHECKPOINT=/opt/dlami/nvme/tmp/fp8-campaign/gemma26b-w8a8/checkpoint-mixed
export PLOW_BUILD_W8A8=1 PLOW_BUILD_FA_GQA2_PAIR=1
export PLOW_BUILD_PFATTN_HD256_BKV32=1 PLOW_BUILD_PFATTN_HD256_GQA2_BKV32=1

run () {  # $1 cfg  $2 layer  $3 tag
  echo "########## FP8 single block: $3 (layer $2, $1)"
  OUT="/opt/dlami/nvme/tmp/agent-geom/blkfp8-$3" \
  BATCH=1,4 CTX=128,1024 \
    bash "$W/scripts/block_e2e.sh" "$MODEL" "perf-data/block-configs/$1" "$2" 2048 \
    > "$T/blkfp8_$3.log" 2>&1
  echo "$3 rc=$?"
  grep -nE "plow|vllm|ratio|FAIL|Error|error|refus|0\.[0-9]+x|[0-9]+\.[0-9]+x" "$T/blkfp8_$3.log" \
    | tail -32
  echo
}

run gemma4-26b-a4b-sliding.json 0 sliding
run gemma4-26b-a4b-full.json 5 full
echo "===== BLKFP8 DONE"
