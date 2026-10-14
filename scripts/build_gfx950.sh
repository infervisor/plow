#!/usr/bin/env bash
# Build the gfx950 (MI350X / CDNA4) persistent-interpreter code objects + the chat harness.
#
# Produces, into the output dir (default /tmp/rtb, or $1, or $PLOW_BUILD_DIR):
#   interp_prefill.elf  plow_interp_gfx950        8 waves — GEMM + flash-prefill (class-8 segments)
#   interp_decode.elf   plow_interp_dec_gfx950    8 waves — GEMV + flash-decode
#   interp_flash.elf    plow_interp_flash_gfx950  4 waves / FA_DC=256 — the class-4 flash_prefill
#                                                 SEGMENT only (segmented dispatch)
#   test_kernels.elf    golden __device__ wrappers (share the SAME op_*.h the interpreter runs)
#   chat                the closed-loop host harness (gemma4_chat.c)
#
# EVERY GUARD BELOW EXISTS BECAUSE A STALE ARTIFACT LIED — a test printed "CORRECT" against a binary
# that had not compiled, or a fresh interpreter ran an old .pkt and the model spoke confident nonsense.
#   - `set -euo pipefail`      : a failed hipcc STOPS the script, it does not leave the old .elf.
#   - `rm -f` before compiling : a build that dies leaves NO artifact to run by mistake.
#   - test_kernels is BUILT    : skipping it once made `t_attn` test a stale kernel (three false passes).
#   - the register-cliff check : over budget => HSA_STATUS_ERROR_INVALID_ISA at runtime, caught here.
#   - the freshness table      : printed every time, so a stale timestamp is visible at a glance.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="$(realpath -m -- "${1:-${PLOW_BUILD_DIR:-/tmp/rtb}}")"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
# Discover the bundler from the INSTALLED ROCm instead of pinning a version. This
# was pinned to 7.0.2 and the path does not exist on a 7.2.4 box, so the build died
# on the first machine that had the GPU. $PLOW_BUNDLER still overrides.
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
case "${PLOW_QKVA_WAVE1:-0}" in
  0) ;;
  1) INC="$INC -DPLOW_QKVA_WAVE1=1" ;;
  *) echo "PLOW_QKVA_WAVE1 must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_MLA_P_BF16:-0}" in
  0) ;;
  1) INC="$INC -DPLOW_MLA_P_BF16=1" ;;
  *) echo "PLOW_MLA_P_BF16 must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_MLA_PF_TR16:-0}" in
  0) FLASH_TR16="" ;;
  1) FLASH_TR16="-DPLOW_MLA_PF_SV=1 -DPLOW_MLA_PF_TR16=1" ;;
  *) echo "PLOW_MLA_PF_TR16 must be 0 or 1" >&2; exit 2 ;;
esac
# FA_MLA_PF2_DEFER=1 (default unset = the running-max V2 softmax): the deferred-frame online
# softmax in the FLASH objects, as gfx942 ships it (op_attention_common.h; not bit-identical —
# gated there on the decode-oracle error class and long-context needles). FA_MLA_PF2_FRAME /
# FA_MLA_PF2_FASTBF pass through likewise.
case "${FA_MLA_PF2_DEFER:-}" in
  '') ;;
  0|1) FLASH_TR16="$FLASH_TR16 -DFA_MLA_PF2_DEFER=$FA_MLA_PF2_DEFER" ;;
  *) echo "FA_MLA_PF2_DEFER must be 0 or 1" >&2; exit 2 ;;
esac
[ -z "${FA_MLA_PF2_FRAME:-}" ] || FLASH_TR16="$FLASH_TR16 -DFA_MLA_PF2_FRAME=$FA_MLA_PF2_FRAME"
[ -z "${FA_MLA_PF2_FASTBF:-}" ] || FLASH_TR16="$FLASH_TR16 -DFA_MLA_PF2_FASTBF=$FA_MLA_PF2_FASTBF"
# PLOW_MLA_PF2_DPP=1: the V2 softmax's 16-lane butterflies as DPP moves (bit-identical).
case "${PLOW_MLA_PF2_DPP:-0}" in
  0) ;;
  1) FLASH_TR16="$FLASH_TR16 -DPLOW_MLA_PF2_DPP=1" ;;
  *) echo "PLOW_MLA_PF2_DPP must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_MLA_PF_SNAKE (default 1; =0 rolls back): the V2 MLA prefill deals its (q-tile, head, split)
# items heaviest-first, rounds boustrophedon over the workgroups (op_attention_common.h; bit-identical,
# item->WG map only). GLM-5.3-Flash TP8 MI350X: flash 29.6 -> 20.6 ms per 8k chunk at c0=0, 66.0 ->
# 57.9 at c0=8k; 8k prefill 272.2 -> 264.0 ms, 16k 598.6 -> 583.4; streams byte-equal.
case "${PLOW_MLA_PF_SNAKE:-1}" in
  0) ;;
  1) FLASH_TR16="$FLASH_TR16 -DPLOW_MLA_PF_SNAKE=1" ;;
  *) echo "PLOW_MLA_PF_SNAKE must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_KDA_CARRY_PIPE=1 / PLOW_KDA_CARRY_KTR16=1: regstate KDA carry load schedule and
# transposed key-tile reads (op_kda_carry_regstate.h; both bit-identical).
KDA_CARRY_DEFS=""
for k in PLOW_KDA_CARRY_PIPE PLOW_KDA_CARRY_KTR16; do
  case "${!k:-0}" in
    0) ;;
    1) KDA_CARRY_DEFS="$KDA_CARRY_DEFS -D$k=1" ;;
    *) echo "$k must be 0 or 1" >&2; exit 2 ;;
  esac
done
case "${PLOW_MLA_SPARSE:-0}" in
  0) ;;
  1) [ "${PLOW_MLA_PF_TR16:-0}" = 1 ] || { echo "PLOW_MLA_SPARSE=1 requires PLOW_MLA_PF_TR16=1" >&2; exit 2; }
     FLASH_TR16="$FLASH_TR16 -DPLOW_MLA_SPARSE_ARM=1" ;;
  *) echo "PLOW_MLA_SPARSE must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_MLA_MHA:-0}" in
  0) ;;
  1) [ "${PLOW_MLA_PF_TR16:-0}" = 1 ] || { echo "PLOW_MLA_MHA=1 requires PLOW_MLA_PF_TR16=1" >&2; exit 2; }
     FLASH_TR16="$FLASH_TR16 -DPLOW_MLA_MHA_ARM=1" ;;
  *) echo "PLOW_MLA_MHA must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_MLA_PF_SCALE_HOIST:-0}" in
  0) FLASH_SCALE_HOIST="" ;;
  1) FLASH_SCALE_HOIST="-DFA_MLA_PF_SCALE_HOIST=1" ;;
  *) echo "PLOW_MLA_PF_SCALE_HOIST must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_MOE_PF_A4W4_BK:-128}" in
  128) A4W4_BK="" ;;
  256) A4W4_BK="-DMPF4_BK=256" ;;
  *) echo "PLOW_MOE_PF_A4W4_BK must be 128 or 256" >&2; exit 2 ;;
esac
# PLOW_XR_SCHED=aiter: the 16-byte prefill collective schedule (op_collective.h
# PLOW_XR_SCHED_AITER) on capped data workgroups, as build_gfx942.sh ships it. Bit-identical
# (strict rank order); default off here until the gfx950 in-model A/B.
case "${PLOW_XR_SCHED:-twoshot}" in
  twoshot|off) ;;
  aiter) A4W4_BK="$A4W4_BK -DPLOW_XR_SCHED_AITER=1 -DPLOW_XR_SCHED_NWG=${PLOW_XR_SCHED_NWG:-24} -DPLOW_XR_SCHED_NWG_RS=${PLOW_XR_SCHED_NWG_RS:-8} -DPLOW_XR_SCHED_AG_U=${PLOW_XR_SCHED_AG_U:-1} -DPLOW_XR_SCHED_NWG_SRS=${PLOW_XR_SCHED_NWG_SRS:-8}"
         [ -n "${PLOW_XR_SCHED_NWG_SAG:-}" ] && A4W4_BK="$A4W4_BK -DPLOW_XR_SCHED_NWG_SAG=$PLOW_XR_SCHED_NWG_SAG" ;;
  *) echo "PLOW_XR_SCHED must be aiter or twoshot" >&2; exit 2 ;;
esac
# Decode-object arms (appended to DEC below; A4W4_BK reaches only the prefill MoE object).
DEC_KNOBS=""
case "${PLOW_FP8_BLK_KW:-0}" in
  0) ;;
  1) DEC_KNOBS="$DEC_KNOBS -DPLOW_FP8_BLK_KW=1" ;;
  *) echo "PLOW_FP8_BLK_KW must be 0 or 1" >&2; exit 2 ;;
esac
# FA_MERGE_UNROLL4=1: four splits in flight in d_flash_merge (gfx942's form; gfx950 kept the serial
# merge for bit-identity). Decode's 64-split merge is a dependent-load chain without it.
case "${PLOW_GEMV_F32_COL:-0}" in
  0) ;;
  1|2) DEC_KNOBS="$DEC_KNOBS -DPLOW_GEMV_F32_COL=$PLOW_GEMV_F32_COL" ;;
  *) echo "PLOW_GEMV_F32_COL must be 0, 1 or 2" >&2; exit 2 ;;
esac
case "${PLOW_MLA_DEC_MINPER:-0}" in
  0) ;;
  *[!0-9]*) echo "PLOW_MLA_DEC_MINPER must be a row count" >&2; exit 2 ;;
  *) DEC_KNOBS="$DEC_KNOBS -DPLOW_MLA_DEC_MINPER=$PLOW_MLA_DEC_MINPER" ;;
esac
case "${PLOW_MOE_GLU_KW:-0}" in
  0) ;;
  1) DEC_KNOBS="$DEC_KNOBS -DPLOW_MOE_GLU_KW=1" ;;
  *) echo "PLOW_MOE_GLU_KW must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_MOE_DEC_W8_SPLITN: the W8A8 decode GLU on 128-column tiles when 256-column tiles would
# leave half the workgroups idle (moe_pf_w8a8.h). Bit-identical. Default on.
case "${PLOW_MOE_DEC_W8_SPLITN:-1}" in
  0) ;;
  1) DEC_KNOBS="$DEC_KNOBS -DPLOW_MOE_DEC_W8_SPLITN=1" ;;
  *) echo "PLOW_MOE_DEC_W8_SPLITN must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_MERGE_UNROLL4:-0}" in
  0) ;;
  1) DEC_KNOBS="$DEC_KNOBS -DFA_MERGE_UNROLL4=1" ;;
  *) echo "PLOW_MERGE_UNROLL4 must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_XR_TAGGED (default 1; 0 = rollback): decode one-shot XReduce publishes tagged data words and
# spins on the tags instead of the xctr gate (op_collective.h; the CMake build's default).
# Bit-identical. A message wider than the tag slot (batched rungs) keeps the gate one-shot.
case "${PLOW_XR_TAGGED:-1}" in
  0) ;;
  1) DEC_KNOBS="$DEC_KNOBS -DPLOW_XR_TAGGED=1" ;;
  *) echo "PLOW_XR_TAGGED must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_XR_DEC_AGG (default 1; 0 = rollback): device-local aggregation of the two-shot's all-gather
# signal (PLOW_XR_AGG, build_gfx942.sh's default) in the decode objects, for packets emitted with
# PLOW_XR_DEC_TWOSHOT_MIN. Bit-identical; inert without decode two-shots.
# PLOW_FA_DEC_KL=8: decode flash K-phase with eight lanes per KV row (gfx942's map; one 128-byte
# line per row per instruction instead of 64 rows per instruction) plus the interleaved row->wave
# map. NOT bit-identical (the per-row dot is summed in a different order); 1 = shipped map.
case "${PLOW_FA_DEC_KL:-1}" in
  1) ;;
  8) DEC_KNOBS="$DEC_KNOBS -DFA_DEC_KL=8 -DFA_DEC_ILV=1" ;;
  *) echo "PLOW_FA_DEC_KL must be 1 or 8" >&2; exit 2 ;;
esac
# PLOW_XR_DEC_MLP (default 1; 0 = rollback): the peer-batched reduce (PLOW_XR_MLP, op_collective.h)
# in the decode objects only. Measured null for gfx942's bandwidth-bound prefill reduce-scatter; a decode one-shot past
# the tag slot (B=8: 64 KiB, one element per thread) is the latency-bound case it was written for.
# Bit-identical; B=8 all-reduce 25.2 -> 18.5 us (gfx950 TP8 trace).
case "${PLOW_XR_DEC_MLP:-1}" in
  0) ;;
  1) DEC_KNOBS="$DEC_KNOBS -DPLOW_XR_MLP=1" ;;
  *) echo "PLOW_XR_DEC_MLP must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_XR_DEC_VEC (default 0 = off; 2/4/8): the decode one-shot reduce loads that many bf16 per
# peer per thread (PLOW_XR_VEC, op_collective.h) for the batched rungs' gate path. Bit-identical.
# MEASURED SLOWER, keep 0 (tp_allreduce_bench cold, 8x MI350X, [128,4096] at 256 WGs: 52.3 us
# scalar vs 59.6 (V=4) / 59.0 (V=8)). The lever there is fewer workgroups: PLOW_XR_DEC_CUS=64
# gives 31.3 us (B=32 shape 27.1 -> 15.6, B=8 18.5 -> 11.5).
case "${PLOW_XR_DEC_VEC:-0}" in
  0) ;;
  2|4|8) DEC_KNOBS="$DEC_KNOBS -DPLOW_XR_VEC=$PLOW_XR_DEC_VEC" ;;
  *) echo "PLOW_XR_DEC_VEC must be 0, 2, 4 or 8" >&2; exit 2 ;;
esac
# PLOW_XR_TAG_SLOT (default 20480; bytes, multiple of 256): the tagged one-shot slot of the decode
# objects. 90112 lets B=8 x hidden 4096 take the tagged path. Serve with the same PLOW_XR_TAG_SLOT.
# Bench (cold, 64 WGs): B=8 message 11.5 us gate -> 7.6 us tagged; B=32 15.9 -> 15.3 (no gain).
case "${PLOW_XR_TAG_SLOT:-20480}" in
  20480) ;;
  *[!0-9]*) echo "PLOW_XR_TAG_SLOT must be a byte count" >&2; exit 2 ;;
  *) if [ $((PLOW_XR_TAG_SLOT % 256)) -ne 0 ] || [ "$PLOW_XR_TAG_SLOT" -lt 20480 ]; then
       echo "PLOW_XR_TAG_SLOT must be a multiple of 256, >= 20480" >&2; exit 2
     fi
     DEC_KNOBS="$DEC_KNOBS -DPLOW_XR_TAG_SLOT_BYTES=${PLOW_XR_TAG_SLOT}u -DPLOW_XR_TAG_SLOT_BYTES_N=$PLOW_XR_TAG_SLOT" ;;
esac
case "${PLOW_XR_DEC_AGG:-1}" in
  0) ;;
  1) DEC_KNOBS="$DEC_KNOBS -DPLOW_XR_AGG=1" ;;
  *) echo "PLOW_XR_DEC_AGG must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_DECODE_BODIES (default 1; =0 rolls back): bit-identical decode-body axes. MoE down/GLU and bf16 GEMV
# lane-group maps (op_moe.h [FP8-DECODE-DOWN-LG]/[FP8-DECODE-X2], op_gemm_common.h
# [BF16-GEMV-NARROWK-LG]); the value-identical grouped-MoE axes the K3 prefill object ships plus
# the A-gather hoist (the batched rungs run MoeGroup*Pf); GemmSmall at BK=128 (the batched rungs'
# 16-64-workgroup projections are k-tile-latency bound; each output's MFMA k order is unchanged);
# the prefill GemvF32 tile at the 128-row rung; the wide StateStepG column prefetch.
# GLM-5.3-Flash MI350X TP8 (traced): B=8 30.4 -> 24.5 ms, B=128 82.2 -> 57.0 ms, rung 1 unchanged.
case "${PLOW_DECODE_BODIES:-1}" in
  0) ;;
  1) DEC_KNOBS="$DEC_KNOBS -DPLOW_MOE_DEC_LG=1 -DPLOW_GEMV_LG=1 -DPLOW_MOE_DEC_X2=1 -DGM_SM_BK=128"
     DEC_KNOBS="$DEC_KNOBS -DPLOW_MOE_TILE_BINSEARCH=1 -DPLOW_MOE_PF_EPI=1 -DPLOW_MOE_PF_BRAW=1 -DPLOW_COMBINE_VEC=1"
     DEC_KNOBS="$DEC_KNOBS -DPLOW_MOE_PF_GH=2 -DPLOW_MOE_ALIGN_PAR_PREFIX=1 -DPLOW_GEMV_F32_TILE_DEC=128 -DPLOW_KDA_WIDE_PF=1"
     # Uniform GEMV weight descriptors. The low-rung tiers also scalarize num_records, which
     # removes the per-load readfirstlane waterfall (tier 8: 219 -> 0; B=8 GEMV 30.5 -> 23.3 us
     # per hop). On the wide object that costs B=128 (64 -> 88 ms), and that object keeps DPP
     # wave sums instead (GemmSmall 114 -> 67 us per hop against the bare set).
     if [ -n "${PLOW_DECODE_TIER:-}" ]; then
       DEC_KNOBS="$DEC_KNOBS -DGV_USCALAR=1 -DPLOW_RSRC_SCALAR_N=1"
     else
       DEC_KNOBS="$DEC_KNOBS -DGV_USCALAR=1 -DPLOW_WAVE_RED_DPP=1"
     fi ;;
  *) echo "PLOW_DECODE_BODIES must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_FP8_BLK_DMA:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_FP8_BLK_DMA=1" ;;
  *) echo "PLOW_FP8_BLK_DMA must be 0 or 1" >&2; exit 2 ;;
esac
# Prefill-object arms (moe_down_a4w4_sweep.h, mla_fold_pf_mfma.h); same packet ABI as default.
case "${PLOW_MOE_PF_DOWN_SWEEP:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_MOE_PF_DOWN_SWEEP=1" ;;
  *) echo "PLOW_MOE_PF_DOWN_SWEEP must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_MOE_ALIGN_WAVES:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_MOE_ALIGN_WAVES=1" ;;
  *) echo "PLOW_MOE_ALIGN_WAVES must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_MOE_DOWN_SWEEP_LINE:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_MOE_DOWN_SWEEP_LINE=1" ;;
  *) echo "PLOW_MOE_DOWN_SWEEP_LINE must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_MLA_FOLD_MFMA:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_MLA_FOLD_MFMA=1" ;;
  *) echo "PLOW_MLA_FOLD_MFMA must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_K3_PF_FOLD_MFMA=1 (default off): the K3 prefill object's MlaMergeFold takes the MFMA fold
# for up to 4 KV splits (GLM-5.3-Flash prefill runs ns=2): the scalar fold's split merge, then a
# bf16-rounded latent into the batched MFMA GEMM. Not bit-identical to the scalar fold.
K3_PF_ATTN=""
case "${PLOW_K3_PF_FOLD_MFMA:-0}" in
  0) ;;
  1) K3_PF_ATTN="-DPLOW_MLA_FOLD_MFMA=1 -DPLOW_MLA_FOLD_MFMA_NS=4" ;;
  *) echo "PLOW_K3_PF_FOLD_MFMA must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_MOE_ROUTER_PF_WAVE:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_MOE_ROUTER_PF_WAVE=1" ;;
  *) echo "PLOW_MOE_ROUTER_PF_WAVE must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_DSA_SELECT_V2=1: DSA prefill top-k (op 118) through d_index_select_pf_v2 (same set).
case "${PLOW_DSA_SELECT_V2:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_DSA_SELECT_V2=1" ;;
  *) echo "PLOW_DSA_SELECT_V2 must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_DSA_SELECT_V3=1: op 118 through dsa_select_v3.h (same set); fused select+union when the
# packet asks (emit PLOW_GLM_DSA_SEL_UNION=1).
case "${PLOW_DSA_SELECT_V3:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_DSA_SELECT_V3=1" ;;
  *) echo "PLOW_DSA_SELECT_V3 must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_DSA_IDX_QPW=1: DSA prefill indexer score (op 117) through d_index_score_pf_rowq (bit-identical).
case "${PLOW_DSA_IDX_QPW:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_DSA_IDX_QPW=1" ;;
  *) echo "PLOW_DSA_IDX_QPW must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_DSA_IDX_FP8=1: DSA prefill indexer score (op 117) with vLLM 0.29 ROCm's FP8 arithmetic
# (runtime/amd/dsa_score_fp8.h). Changes the scores (not bit-identical to the bf16 arms); wins over QPW.
case "${PLOW_DSA_IDX_FP8:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_DSA_IDX_FP8=1" ;;
  *) echo "PLOW_DSA_IDX_FP8 must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_HNR_ILP=1: HeadNormRope HD=128 interleaved (DSA indexer rope) four heads per wave pass.
case "${PLOW_HNR_ILP:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_HNR_ILP=1" ;;
  *) echo "PLOW_HNR_ILP must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_DSA_PREP=1: DSA indexer prep fusions (dsa_prep.h; packet HeadNormRope pair_mode 3). Also
# turned on below when the packet's plow_config.h requires it.
case "${PLOW_DSA_PREP:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_DSA_PREP=1" ;;
  *) echo "PLOW_DSA_PREP must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_COMBINE_VEC:-0}" in
  0) ;;
  1) A4W4_BK="$A4W4_BK -DPLOW_COMBINE_VEC=1" ;;
  *) echo "PLOW_COMBINE_VEC must be 0 or 1" >&2; exit 2 ;;
esac
if [ -n "${PLOW_HSACO_CONFIG:-}" ]; then
  [ -f "$PLOW_HSACO_CONFIG" ] || { echo "missing PLOW_HSACO_CONFIG: $PLOW_HSACO_CONFIG" >&2; exit 2; }
  cfg_dir="$(dirname -- "$PLOW_HSACO_CONFIG")"
  cfg_name="$(basename -- "$PLOW_HSACO_CONFIG")"
  INC="$INC -I$cfg_dir -DPLOW_CONFIG=\"$cfg_name\""
fi
decode_inventory_prune="${PLOW_HSACO_DECODE_INVENTORY_PRUNE:-auto}"
DSA_FLASH=""
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -Eq '^#define PLOW_PACKET_OBJECT_REQUIRES "([^"]* )?PLOW_DSA_PF_ARM=1( |")' "$PLOW_HSACO_CONFIG"; then
  DSA_FLASH="-DPLOW_DSA_PF_ARM=1"
fi
# The zero-rope (NoPE) V2 MLA prefill arm, FLASH OBJECT ONLY (interp.hip PLOW_MLA_PF2_NOPE_ARM;
# gfx942 spells it PLOW_MLA_PF_NOPE=1): GLM-5.3-Flash and DeepSeek-V4 packets require it and the
# loader refuses a flash object without `plow_mla_pf2_nope_arm`.
NOPE_FLASH=""
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -Eq '^#define PLOW_PACKET_OBJECT_REQUIRES "([^"]* )?PLOW_MLA_PF2_NOPE_ARM=1( |")' "$PLOW_HSACO_CONFIG"; then
  NOPE_FLASH="-DPLOW_MLA_PF2_NOPE_ARM=1"
fi
# GLM-5.3-Flash (NoPE + K3 packet): the gfx950 transpose-read PV arm is the default in its flash
# objects (same operands, same MFMA order: every validate stream byte-equal; 16k prefill flash
# 107 -> 79 ms per 8192 chunk). PLOW_MLA_PF_TR16=0 rolls back.
if [ -n "$NOPE_FLASH" ] && [ -z "${PLOW_MLA_PF_TR16:-}" ] &&
   grep -Eq '^#define PLOW_PACKET_OBJECT_REQUIRES "([^"]* )?PLOW_K3=1( |")' "$PLOW_HSACO_CONFIG"; then
  FLASH_TR16="-DPLOW_MLA_PF_SV=1 -DPLOW_MLA_PF_TR16=1 $FLASH_TR16"
fi
# GLM-5.3-Flash (NoPE + K3 packet) gets the arms qualified on it alone. PLOW_GEMV_F32_VEC: the
# 8-wide GEMV walks (op_gemm_common.h) reassociate the f32 sums of every object's mHC mix and
# router GEMVs, so they are qualified per model.
GLM53_FLASH=0
if [ -n "$NOPE_FLASH" ] &&
   grep -Eq '^#define PLOW_PACKET_OBJECT_REQUIRES "([^"]* )?PLOW_K3=1( |")' "$PLOW_HSACO_CONFIG"; then
  GLM53_FLASH=1
  INC="$INC -DPLOW_GEMV_F32_VEC=1"
fi
if [ -n "${PLOW_HSACO_CONFIG:-}" ] && [ "${PLOW_DSA_PREP:-0}" = 0 ] &&
   grep -Eq '^#define PLOW_PACKET_OBJECT_REQUIRES "([^"]* )?PLOW_DSA_PREP=1( |")' "$PLOW_HSACO_CONFIG"; then
  A4W4_BK="$A4W4_BK -DPLOW_DSA_PREP=1"
fi
case "${decode_inventory_prune,,}" in
  auto) [ -n "${PLOW_HSACO_CONFIG:-}" ] && decode_inventory_prune=1 || decode_inventory_prune=0 ;;
  1|on|true|yes) decode_inventory_prune=1 ;;
  0|off|false|no) decode_inventory_prune=0 ;;
  *) echo "PLOW_HSACO_DECODE_INVENTORY_PRUNE must be ON or OFF" >&2; exit 2 ;;
esac
if [ "$decode_inventory_prune" = 1 ] && [ -z "${PLOW_HSACO_CONFIG:-}" ]; then
  echo "PLOW_HSACO_DECODE_INVENTORY_PRUNE requires PLOW_HSACO_CONFIG" >&2
  exit 2
fi
need_decode_mla=0
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -qx '#define PLOW_PACKET_HAS_DECODE_MLA_SEGMENTS 1' "$PLOW_HSACO_CONFIG"; then
  [ "$ARCH" = gfx950 ] || {
    echo "decode MLA segment objects are supported only on gfx950" >&2; exit 2; }
  [ "$decode_inventory_prune" = 1 ] || {
    echo "decode MLA segment objects require packet-paired inventory pruning" >&2; exit 2; }
  need_decode_mla=1
fi
need_moe_prefill_ep=0
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -qx '#define PLOW_PACKET_REQUIRES_MOE_PREFILL_EP 1' "$PLOW_HSACO_CONFIG"; then
  [ "$ARCH" = gfx950 ] || {
    echo "prefill expert-parallel objects are supported only on gfx950" >&2; exit 2; }
  need_moe_prefill_ep=1
fi
mkdir -p "$OUT"; cd "$OUT"

if [ "$ARCH" = gfx950 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/rope_cache_bf16_gfx950.elf" glm_rope_cache_bf16 64 4 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_rope_cache_bf16_abi_1 \
    -cuid=plow_rope_cache_bf16_abi_1 \
    -I"$R/amd" -I"$R/common" "$R/amd/rope_cache.hip"
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/rope_indexer_cache_bf16_gfx950.elf" glm_rope_indexer_cache_bf16 64 4 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_rope_indexer_cache_bf16_abi_1 \
    -cuid=plow_rope_indexer_cache_bf16_abi_1 \
    -I"$R/amd" -I"$R/common" "$R/amd/rope_indexer_cache.hip"
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/indexer_decode_gfx950.elf" glm_indexer_decode_prepare 64 4 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_indexer_decode_abi_2 \
    -cuid=plow_indexer_decode_abi_2 \
    -I"$R/amd" -I"$R/common" "$R/amd/indexer_decode.hip"
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/indexer_prefill_gfx950.elf" glm_indexer_prefill_prepare 64 4 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_indexer_prefill_abi_1 \
    -cuid=plow_indexer_prefill_abi_1 \
    -I"$R/amd" -I"$R/common" "$R/amd/indexer_prefill.hip"
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/indexer_prefill_append_gfx950.elf" glm_indexer_prefill_append 64 4 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_indexer_prefill_append_abi_1 \
    -cuid=plow_indexer_prefill_append_abi_1 \
    -I"$R/amd" -I"$R/common" "$R/amd/indexer_prefill_append.hip"
  # PLOW_GLM_HC_MIX_RAW: one-row decode mHC mix + pre/norm tail (GemvF32 marked raw).
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/hc_mix_decode_gfx950.elf" plow_hc_mix_decode_r1 128 2 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_hc_mix_decode_1 \
    -I"$R/amd" -I"$R/common" "$R/amd/hc_mix_decode.hip"
  # PLOW_MOE_W8A8_DEC: decode-rung W8A8 grouped MoE GLU + DOWN (marked op 85/86 pair).
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/moe_w8a8_dec_gfx950.elf" plow_moe_w8a8_dec_glu 512 1 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe_w8a8_dec_abi_1 \
    -I"$R/amd" -I"$R/common" "$R/amd/moe_w8a8_dec.hip"
fi

# Delete FIRST. A build that fails must leave nothing behind to run.
rm -f i_prefill.co i_decode.co i_flash.co tk.co \
      interp_prefill.elf interp_decode.elf interp_flash.elf test_kernels.elf \
      i_prefill_gq.co i_decode_gq.co i_flash_gq.co \
      interp_prefill_gq.elf interp_decode_gq.elf interp_flash_gq.elf \
      i_decode_mla.co i_decode_mla_gq.co \
      interp_decode_mla.elf interp_decode_mla_gq.elf \
      i_decode_fp8.co i_decode_fp8_gq.co interp_decode_fp8.elf interp_decode_fp8_gq.elf \
      i_decode_fp8kv.co i_decode_fp8kv_gq.co interp_decode_fp8kv.elf interp_decode_fp8kv_gq.elf \
      i_flash_fp8kv.co i_flash_fp8kv_gq.co interp_flash_fp8kv.elf interp_flash_fp8kv_gq.elf \
      i_prefill_fp8kv_mla_moe_a4w4_full.co i_prefill_fp8kv_mla_moe_a4w4_full_gq.co \
      interp_prefill_fp8kv_mla_moe_a4w4_full.elf interp_prefill_fp8kv_mla_moe_a4w4_full_gq.elf \
      i_prefill_mla_moe.co i_prefill_mla_moe_gq.co \
      interp_prefill_mla_moe.elf interp_prefill_mla_moe_gq.elf \
      kda_decode_fused_gfx950.co kda_decode_fused_gfx950.elf \
      kda_decode_wide_gfx950.co kda_decode_wide_gfx950.elf \
      kda_chunk_intra_cached_gfx950.co kda_chunk_intra_cached_gfx950.elf \
      kda_chunk_intra_wave_items_gfx950.co kda_chunk_intra_wave_items_gfx950.elf \
      attn_res_f32mix_gfx950.co attn_res_f32mix_gfx950.elf \
      kda_chunk_carry_regstate_gfx950.co kda_chunk_carry_regstate_gfx950.elf \
      kda_chunk_wu_lean_gfx950.co kda_chunk_wu_lean_gfx950.elf \
      kda_chunk_wu_lean_keys_gfx950.co kda_chunk_wu_lean_keys_gfx950.elf \
      kda_chunk_carry_regstate_keyfeed_gfx950.co \
      kda_chunk_carry_regstate_keyfeed_gfx950.elf \
      kda_chunk_key_factor_wu_gfx950.co kda_chunk_key_factor_wu_gfx950.elf \
      kda_chunk_key_factor_carry_gfx950.co kda_chunk_key_factor_carry_gfx950.elf \
      xreduce_attnres_gfx950.co xreduce_attnres_gfx950.elf \
      moe_stage1_mxfp4_gfx950.co moe_stage1_mxfp4_gfx950.elf \
      moe_stage2_mxfp4_gfx950.co moe_stage2_mxfp4_gfx950.elf \
      moe_combine_gfx950.co moe_combine_gfx950.elf \
      moe_ep_align_gfx950.co moe_ep_align_gfx950.elf \
      moe_ep_stage2_gfx950.co moe_ep_stage2_gfx950.elf \
      moe_ep_combine_gfx950.co moe_ep_combine_gfx950.elf \
      moe_decode_grouped_mxfp4_gfx950.co moe_decode_grouped_mxfp4_gfx950.elf \
      mla_materialized_hd192_v128_gfx950.co mla_materialized_hd192_v128_gfx950.elf \
      mla_materialize_pack_gfx950.co mla_materialize_pack_gfx950.elf

genco() { # <extra-defs> <out.co>
  hipcc --offload-arch="$ARCH" -O3 -w $1 --genco "$R/amd/interp.hip" -o "$2" $INC
}
unbundle() { # <in.co> <out.elf>
  "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input="$1" --output="$2"
}

KDA_FUSED_ELFS=""
need_kda_fused=1
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   ! grep -qx '#define PLOW_PACKET_HAS_KDA_DECODE_FUSED 1' "$PLOW_HSACO_CONFIG"; then
  need_kda_fused=0
fi

KDA_INTRA_CACHED_ELFS=""
if [ "$ARCH" = gfx950 ] && [ "${PLOW_KDA_INTRA_CACHED:-0}" = 1 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/kda_chunk_intra_cached_gfx950.elf" plow_kda_chunk_intra_cached_gfx950 96 1 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_kda_intra_cached_abi_1 \
    "$R/amd/kda_chunk_intra_cached.hip"
  KDA_INTRA_CACHED_ELFS="kda_chunk_intra_cached_gfx950.elf"
fi

KDA_INTRA_WAVE_ITEMS_ELFS=""
case "${PLOW_KDA_INTRA_WAVE_ITEMS:-0}" in
  0|1) ;;
  *) echo "PLOW_KDA_INTRA_WAVE_ITEMS must be 0 or 1" >&2; exit 2 ;;
esac
need_kda_intra_wave_items=${PLOW_KDA_INTRA_WAVE_ITEMS:-0}
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -qx '#define PLOW_PACKET_REQUIRES_KDA_INTRA_WAVE_ITEMS 1' "$PLOW_HSACO_CONFIG"; then
  need_kda_intra_wave_items=1
fi
if [ "$need_kda_intra_wave_items" = 1 ] && [ "$ARCH" != gfx950 ]; then
  echo "manifest-required KDA intra wave-item object is supported only on gfx950" >&2
  exit 2
fi
if [ "$ARCH" = gfx950 ] && [ "$need_kda_intra_wave_items" = 1 ]; then
  [ -n "${PLOW_HSACO_CONFIG:-}" ] || {
    echo "PLOW_KDA_INTRA_WAVE_ITEMS=1 requires PLOW_HSACO_CONFIG for packet pairing" >&2
    exit 2
  }
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/kda_chunk_intra_wave_items_gfx950.elf" \
    plow_kda_chunk_intra_wave_items_gfx950 96 2 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_kda_intra_wave_items_abi_1 \
    "$R/amd/kda_chunk_intra_wave_items.hip"
  KDA_INTRA_WAVE_ITEMS_ELFS="kda_chunk_intra_wave_items_gfx950.elf"
fi

ATTN_RES_F32MIX_ELFS=""
need_attn_res_f32mix=${PLOW_ATTNRES_F32MIX:-0}
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -qx '#define PLOW_PACKET_REQUIRES_ATTN_RES_F32MIX 1' "$PLOW_HSACO_CONFIG"; then
  need_attn_res_f32mix=1
fi
if [ "$need_attn_res_f32mix" = 1 ] && [ "$ARCH" != gfx950 ]; then
  echo "manifest-required f32-mix AttnRes object is supported only on gfx950" >&2
  exit 2
fi
if [ "$ARCH" = gfx950 ] && [ "$need_attn_res_f32mix" = 1 ]; then
  [ -n "${PLOW_HSACO_CONFIG:-}" ] || {
    echo "PLOW_ATTNRES_F32MIX=1 requires PLOW_HSACO_CONFIG for packet pairing" >&2
    exit 2
  }
  # -fno-slp-vectorize: packed-f32 FMA formation costs ~30 VGPRs here (see the source header).
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/attn_res_f32mix_gfx950.elf" \
    plow_attn_res_f32mix_gfx950 168 3 \
    $INC -fno-slp-vectorize -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_attn_res_f32mix_abi_1 \
    "$R/amd/attn_res_f32mix_gfx950.hip"
  ATTN_RES_F32MIX_ELFS="attn_res_f32mix_gfx950.elf"
fi

KDA_CARRY_REGSTATE_ELFS=""
case "${PLOW_KDA_CARRY_REGSTATE:-0}" in
  0|1) ;;
  *) echo "PLOW_KDA_CARRY_REGSTATE must be 0 or 1" >&2; exit 2 ;;
esac
need_kda_carry_regstate=${PLOW_KDA_CARRY_REGSTATE:-0}
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -qx '#define PLOW_PACKET_REQUIRES_KDA_CARRY_REGSTATE 1' "$PLOW_HSACO_CONFIG"; then
  need_kda_carry_regstate=1
fi
if [ "$need_kda_carry_regstate" = 1 ] && [ "$ARCH" != gfx950 ]; then
  echo "manifest-required KDA carry regstate object is supported only on gfx950" >&2
  exit 2
fi
if [ "$ARCH" = gfx950 ] && [ "$need_kda_carry_regstate" = 1 ]; then
  [ -n "${PLOW_HSACO_CONFIG:-}" ] || {
    echo "PLOW_KDA_CARRY_REGSTATE=1 requires PLOW_HSACO_CONFIG for packet pairing" >&2
    exit 2
  }
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/kda_chunk_carry_regstate_gfx950.elf" \
    plow_kda_chunk_carry_regstate_gfx950 256 1 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_kda_carry_regstate_abi_1 $KDA_CARRY_DEFS \
    "$R/amd/kda_chunk_carry_regstate.hip"
  KDA_CARRY_REGSTATE_ELFS="kda_chunk_carry_regstate_gfx950.elf"
fi

KDA_WU_LEAN_ELFS=""
case "${PLOW_KDA_WU_LEAN:-0}" in
  0|1) ;;
  *) echo "PLOW_KDA_WU_LEAN must be 0 or 1" >&2; exit 2 ;;
esac
need_kda_wu_lean=${PLOW_KDA_WU_LEAN:-0}
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -qx '#define PLOW_PACKET_REQUIRES_KDA_WU_LEAN 1' "$PLOW_HSACO_CONFIG"; then
  need_kda_wu_lean=1
fi
if [ "$need_kda_wu_lean" = 1 ] && [ "$ARCH" != gfx950 ]; then
  echo "manifest-required lean KDA Wu object is supported only on gfx950" >&2
  exit 2
fi
if [ "$ARCH" = gfx950 ] && [ "$need_kda_wu_lean" = 1 ]; then
  [ -n "${PLOW_HSACO_CONFIG:-}" ] || {
    echo "PLOW_KDA_WU_LEAN=1 requires PLOW_HSACO_CONFIG for packet pairing" >&2
    exit 2
  }
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/kda_chunk_wu_lean_gfx950.elf" \
    plow_kda_chunk_wu_lean_gfx950 256 2 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_kda_wu_lean_abi_1 \
    "$R/amd/kda_chunk_wu_lean.hip"
  KDA_WU_LEAN_ELFS="kda_chunk_wu_lean_gfx950.elf"
fi

# DevOp::GemmGroupPf: the grouped prefill projection segment (packet-required only).
GEMM_GROUP_ELFS=""
if [ "$ARCH" = gfx950 ] && [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -qx '#define PLOW_PACKET_REQUIRES_GEMM_GROUP_PF 1' "$PLOW_HSACO_CONFIG"; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/gemm_group_pf_gfx950.elf" \
    plow_gemm_group_pf_gfx950 256 2 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_REQUIRED_MARKER=plow_gemm_group_pf_abi_1 \
    "$R/amd/gemm_group_pf.hip"
  GEMM_GROUP_ELFS="gemm_group_pf_gfx950.elf"
fi

KDA_CARRY_KEYFEED_ELFS=""
case "${PLOW_KDA_CARRY_KEYFEED:-0}" in
  0|1) ;;
  *) echo "PLOW_KDA_CARRY_KEYFEED must be 0 or 1" >&2; exit 2 ;;
esac
need_kda_carry_keyfeed=${PLOW_KDA_CARRY_KEYFEED:-0}
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -qx '#define PLOW_PACKET_REQUIRES_KDA_CARRY_KEYFEED 1' "$PLOW_HSACO_CONFIG"; then
  need_kda_carry_keyfeed=1
fi
if [ "$need_kda_carry_keyfeed" = 1 ] && [ "$ARCH" != gfx950 ]; then
  echo "manifest-required KDA carry keyfeed objects are supported only on gfx950" >&2
  exit 2
fi
if [ "$ARCH" = gfx950 ] && [ "$need_kda_carry_keyfeed" = 1 ]; then
  [ -n "${PLOW_HSACO_CONFIG:-}" ] || {
    echo "PLOW_KDA_CARRY_KEYFEED=1 requires PLOW_HSACO_CONFIG for packet pairing" >&2
    exit 2
  }
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/kda_chunk_wu_lean_keys_gfx950.elf" \
    plow_kda_chunk_wu_lean_keys_gfx950 256 2 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_kda_wu_lean_abi_1 \
    "$R/amd/kda_chunk_wu_lean_keys.hip"
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/kda_chunk_carry_regstate_keyfeed_gfx950.elf" \
    plow_kda_chunk_carry_regstate_keyfeed_gfx950 256 1 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_kda_carry_keyfeed_abi_1 $KDA_CARRY_DEFS \
    "$R/amd/kda_chunk_carry_regstate_keyfeed.hip"
  KDA_CARRY_KEYFEED_ELFS="kda_chunk_wu_lean_keys_gfx950.elf kda_chunk_carry_regstate_keyfeed_gfx950.elf"
fi

# PLOW_MLA_PF_DENSE=1: the dedicated NoPE MLA flash-prefill object plowrt routes eligible flash
# segments to when served with PLOW_MLA_PF_DENSE=1 (exec/amd_mla_pf_dense.rs). iterative-ilp keeps
# the QK fragment loads in flight (measured 8k chunk 0.826 -> 0.809 ms/layer, 16k 2.267 -> 2.223).
MLA_PF_DENSE_ELFS=""
case "${PLOW_MLA_PF_DENSE:-0}" in
  0|1) ;;
  *) echo "PLOW_MLA_PF_DENSE must be 0 or 1" >&2; exit 2 ;;
esac
if [ "${PLOW_MLA_PF_DENSE:-0}" = 1 ] && [ "$ARCH" != gfx950 ]; then
  echo "PLOW_MLA_PF_DENSE=1 is supported only on gfx950" >&2
  exit 2
fi
if [ "$ARCH" = gfx950 ] && [ "${PLOW_MLA_PF_DENSE:-0}" = 1 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/mla_pf_dense_gfx950.elf" plow_mla_pf_dense_gfx950 512 1 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_mla_pf_dense_abi_1 -mllvm -amdgpu-sched-strategy=iterative-ilp \
    "$R/amd/mla_pf_dense_gfx950.hip"
  MLA_PF_DENSE_ELFS="mla_pf_dense_gfx950.elf"
fi

KDA_KEY_FACTOR_ELFS=""
case "${PLOW_KDA_KEY_FACTOR:-0}" in
  0|1) ;;
  *) echo "PLOW_KDA_KEY_FACTOR must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$ARCH" = gfx950 ] && [ "${PLOW_KDA_KEY_FACTOR:-0}" = 1 ]; then
  for role in wu carry; do
    bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
      "$OUT/kda_chunk_key_factor_${role}_gfx950.elf" \
      "plow_kda_chunk_key_factor_${role}_gfx950" 160 3 \
      $INC \
      -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
      -DPLOW_REQUIRED_MARKER="plow_kda_key_factor_${role}_1" \
      "$R/amd/kda_chunk_key_factor_${role}.hip"
  done
  KDA_KEY_FACTOR_ELFS="kda_chunk_key_factor_wu_gfx950.elf kda_chunk_key_factor_carry_gfx950.elf"
fi

XR_ATTNRES_ELFS=""
if [ "$ARCH" = gfx950 ] && [ "${PLOW_XR_ATTNRES:-0}" = 1 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/xreduce_attnres_gfx950.elf" plow_xreduce_attnres_gfx950 256 2 \
    $INC -DPLOW_K3=1 -DPLOW_XR_ATTNRES=1 -DPLOW_LEAN_OBJECT=1 \
    -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_xr_attnres_wave64_nospill_1 \
    "$R/amd/xreduce_attnres_fused.hip"
  XR_ATTNRES_ELFS="xreduce_attnres_gfx950.elf"
fi
if [ "$ARCH" = gfx950 ] && [ "$need_kda_fused" = 1 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/kda_decode_fused_gfx950.elf" plow_kda_decode_fused_256x16_v2 128 4 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_kda_decode_fused_256x16_2 \
    "$R/amd/kda_decode_fused.hip"
  # Descriptor version 3 (PLOW_KDA_DECODE_WIDE): the batched-rung chain through the decode
  # object's own bodies, so it carries the decode object's bit-identical body defines.
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/kda_decode_wide_gfx950.elf" plow_kda_decode_wide_v3 128 2 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_BUCKET_DECODE=1 -DPLOW_KDA_WIDE_PF=1 \
    -DPLOW_WAVE_RED_DPP=1 -DPLOW_REQUIRED_MARKER=plow_kda_decode_wide_3 \
    "$R/amd/kda_decode_wide.hip"
  KDA_FUSED_ELFS="kda_decode_fused_gfx950.elf kda_decode_wide_gfx950.elf"
fi

# Opt-in lean MoE body variants requested at emit (PLOW_MOE_STAGE1_BODY / PLOW_MOE_STAGE2_BODY).
MOE1_BODY_DEFS=""; MOE1_MAXREG=192; MOE2_BODY_DEFS=""
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -qx '#define PLOW_OBJECT_MOE_STAGE1_BODY 1' "$PLOW_HSACO_CONFIG"; then
  MOE1_BODY_DEFS="-DPLOW_MOE1_BODY=1"; MOE1_MAXREG=256
fi
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -qx '#define PLOW_OBJECT_MOE_STAGE2_BODY 1' "$PLOW_HSACO_CONFIG"; then
  MOE2_BODY_DEFS="-DPLOW_MOE2_BODY=1"
fi
# Pipelined token-gather stage-1 (reuse_kernel.hip PLOW_MOE1_BODY=2); the runtime picks its ABI
# from the object's plow_moe1_a4_token_gather_1 marker.
case "${PLOW_MOE_STAGE1_PIPE:-0}" in
  0) ;;
  1) MOE1_BODY_DEFS="-DPLOW_MOE1_BODY=2"; MOE1_MAXREG=256 ;;
  *) echo "PLOW_MOE_STAGE1_PIPE must be 0 or 1" >&2; exit 2 ;;
esac

MOE_STAGE2_ELFS=""
if [ "$ARCH" = gfx950 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/moe_stage2_mxfp4_gfx950.elf" plow_moe2_mxfp4_16x16x128_gfx950 100 2 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe2_mxfp4_stage2_abi_3 $MOE2_BODY_DEFS \
    "$R/bench/amd/lean_moe_stage2_ref/native_kernel.hip"
  MOE_STAGE2_ELFS="moe_stage2_mxfp4_gfx950.elf"
fi

MOE_STAGE1_ELFS=""
if [ "$ARCH" = gfx950 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/moe_stage1_mxfp4_gfx950.elf" plow_moe1_a4_reuse_16x16x128_gfx950 "$MOE1_MAXREG" 2 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe1_a4_reuse_abi_1 $MOE1_BODY_DEFS \
    "$R/bench/amd/lean_moe_stage1_ref/reuse_kernel.hip"
  MOE_STAGE1_ELFS="moe_stage1_mxfp4_gfx950.elf"
fi

MOE_COMBINE_ELFS=""
if [ "$ARCH" = gfx950 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/moe_combine_gfx950.elf" plow_moe_combine_fixed_order_gfx950 64 4 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe_combine_fixed_order_abi_1 \
    "$R/bench/amd/lean_moe_combine_ref/kernel.hip"
  MOE_COMBINE_ELFS="moe_combine_gfx950.elf"
fi

# Dedicated block-fp8 grouped-MoE prefill object (GLU / down, W8A16 and W8A8), routed for the
# segments PLOW_MOE_FP8_PF_LEAN=1 packets isolate; bit-identical to the interpreter arms.
MOE_FP8_PF_ELFS=""
if [ "$ARCH" = gfx950 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/moe_fp8_pf_gfx950.elf" plow_moe_fp8_glu_pf_gfx950 256 1 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_REQUIRED_MARKER=plow_moe_fp8_pf_abi_2 \
    "$R/amd/moe_fp8_pf.hip"
  MOE_FP8_PF_ELFS="moe_fp8_pf_gfx950.elf"
fi

MOE_EP_ELFS=""
if [ "$need_moe_prefill_ep" = 1 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/moe_ep_align_gfx950.elf" plow_moe_ep_filter_align_gfx950 64 2 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe_ep_filter_align_no_spill_1 \
    "$R/bench/amd/moe_ep_boundary/filter_align.hip"
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/moe_ep_stage2_gfx950.elf" plow_moe2_ep_full_i_16x16x128_gfx950 128 2 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe2_ep_full_i_vgpr_le_128 \
    "$R/bench/amd/moe_ep_boundary/stage2_full_i.hip"
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/moe_ep_combine_gfx950.elf" plow_moe_ep_combine_gfx950 64 2 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe_ep_combine_no_spill_1 \
    "$R/bench/amd/moe_ep_boundary/combine.hip"
  MOE_EP_ELFS="moe_ep_align_gfx950.elf moe_ep_stage2_gfx950.elf moe_ep_combine_gfx950.elf"
fi

MOE_DECODE_GROUPED_ELFS=""
if [ "$ARCH" = gfx950 ] && [ "${PLOW_MOE_DECODE_STANDALONE:-0}" = 1 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/moe_decode_grouped_mxfp4_gfx950.elf" \
    plow_moe_decode_grouped_glu_mxfp4_gfx950 96 5 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe_decode_grouped_mxfp4_abi_1 \
    "$R/amd/moe_decode_grouped_mxfp4.hip"
  MOE_DECODE_GROUPED_ELFS="moe_decode_grouped_mxfp4_gfx950.elf"
fi

MLA_MATERIALIZED_ELFS=""
if [ "$ARCH" = gfx950 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/mla_materialized_hd192_v128_gfx950.elf" \
    plow_mla_materialized_hd192_v128_gfx950 256 2 \
    -std=c++20 -I"$R/amd/third_party/aiter_opus" \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_mla_materialized_opus_abi_1 \
    "$R/amd/mla_materialized_opus.hip"
  MLA_MATERIALIZED_ELFS="mla_materialized_hd192_v128_gfx950.elf"
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/mla_materialize_pack_gfx950.elf" \
    plow_mla_materialize_pack_gfx950 64 4 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_mla_materialize_pack_abi_1 \
    "$R/amd/mla_materialize_pack.hip"
  MLA_MATERIALIZED_ELFS="$MLA_MATERIALIZED_ELFS mla_materialize_pack_gfx950.elf"
fi

# DECODE BATCH BUCKET -> PLOW_GEMV_MM. THIS ROUTE WAS MISSING, and it is why batched decode
# produced exactly one non-zero logits row on AMD while devgen emitted a fully batch-aware
# program (build.json decode_batch:4, gv_mm_max:4, prog 3 (T=4), 4x KV cache — all correct).
#
# The multi-row GEMV arms have always existed here: gemv_rows<MM>, gemv_glu_rows<MM>,
# gemv_qkv_rows<MM> and the fp8/mxfp4/fp8_blk/dma variants all carry `float acc[MM]`, predicate
# each row on `m < M`, and write C[m*N + n]. What did not exist was anything DEFINING
# PLOW_GEMV_MM, so every AMD decode object compiled at op_gemm.h's default of 1 and wrote row 0
# only — rows 1..B-1 stayed zero and every sequence but the first sampled token 0. §4's bug
# shape exactly: the arm exists, is correct, is register-gated, and NOTHING ROUTES TO IT.
#
# The knob NAMES diverge across backends, which is how this hid in plain sight: the tuning
# pipeline (crates/devgen/src/manifest.rs, crates/tunedb/src/decode.rs) emits `GV_MM_MAX`, which
# is the NVIDIA op_gemm.cuh knob. AMD's is PLOW_GEMV_MM, and nothing anywhere set it.
#
# It is a COMPILE-TIME bucket on purpose. Do NOT replace this with a runtime LADDER like
# NVIDIA's gemv_walk(). That WAS built here, measured, and REMOVED (op_gemm.h:2163): a runtime
# `if (M<=1) ... else if (M<=2) ...` chain inlines every instantiation, so the allocator budgets
# for the widest arm and the decode object went to arch 148 + agpr 128 = 276 registers — over
# the 256 a wave may use at 2 waves/SIMD, and that single switch is what blocked the 8-wave
# dispatch for every other op, GEMM included. One instantiation with a runtime M and an
# `m < M` predicate serves every M <= MM, which is why the ladder buys nothing here.
#
# THE LADDER AND THE OUTER LOOP ARE NOT THE SAME THING, and 276 measured the ladder. A
# SINGLE-RUNG walk over M > MM (`PLOW_GEMV_WALK`, op_gemm.h, default 0) inlines ONE body, so
# no register union forms: measured +1 VGPR at MM=4 and +0 everywhere else, occ 2, AGPR 0,
# spill unchanged. It is the WIDTH that costs, not the loop — MM=32/64 stay at 256/occ 2 but
# spill 0.35-0.53 scratch ops per FMA INSIDE the weight-stream loop (MM=1 is 0.022), and
# NVIDIA's shallow-unroll rescue does not port because this megakernel has 8 registers of
# headroom where sm_120 has 43. Full table + verdict: the design notes §6g-WALK.
#
# next_pow2, clamped to PLOW_GEMV_MAXM (16). Register cost measured on ROCm 7.2.4 / gfx950:
#   MM=1  248/occ2/spill 0     MM=2  248/occ2/spill 0 (free)     MM=4  252/occ2/spill 0
#   MM=8  256/occ2/spill 19    <- at the cap AND spilling; the bucket stops paying here.
# B=1 keeps MM=1, so it is byte- and register-identical to a pre-batch build.
#
# THE BUCKET IS NOW SELF-DECLARING. op_gemm.h emits `plow_gemv_mm_cap_$GVMM` into every object
# it compiles, named for PLOW_GEMV_MM itself, and plowrt's `check_gemv_capacity` refuses to run
# a packet whose widest GEMV asks for more rows than the object advertises. Nothing here has to
# be kept in sync with it: the symbol IS the macro. But note the corollary for OLD trees — an
# object built before that marker advertises nothing, and a batch>1 packet against it is refused
# rather than silently served, so a tree serving B>1 must be rebuilt with this script once.
# Verified on ROCm 7.2.4/gfx950, MM=1 decode object with and without the marker:
# 248 vgpr / 0 agpr / 100 sgpr / 288 B private, IDENTICAL. It costs 4 bytes of .data.
GVMM="${PLOW_DECODE_BATCH:-1}"
case "$GVMM" in ''|*[!0-9]*) GVMM=1;; esac
[ "$GVMM" -lt 1 ] && GVMM=1
P2=1; while [ "$P2" -lt "$GVMM" ]; do P2=$((P2 * 2)); done
[ "$P2" -gt 16 ] && P2=16
GVMM="$P2"
# THE BUCKET AND THE BATCH CAN NOW BE DECOUPLED, and that is the whole §6g-WALK experiment.
#
# Until now `PLOW_GEMV_MM` was DERIVED from `PLOW_DECODE_BATCH` with no way to override it, so
# `min(t, gv_mm)` was identically `t` and the fusion-gate fix the walk study asked for would
# have been a no-op. The two numbers answer different questions:
#   PLOW_DECODE_BATCH  how many sequences the PROGRAM serves      (t, packet-side)
#   PLOW_GEMV_MM       how many rows one GEMV pass handles        (MM, object-side)
# With `PLOW_GEMV_WALK=1` the kernel loops `ceil(t/MM)` times, so MM < t is legal and is the
# ONLY way to serve t=16 without the MM=16 spill (16 scratch ops/FMA, 5536 B/lane) AND without
# losing `fuse_qkv`/`glu_fused` to the `t*hidden > GM_LDS_HALVES` gate.
#
# WITHOUT the walk, MM < t is SILENT CORRUPTION — rows MM..t-1 are never written. plowrt refuses
# that pairing (`check_gemv_capacity` against `plow_gemv_mm_cap_<N>`), so the mistake is caught
# at load rather than in the output, but do not make it on purpose.
if [ -n "${PLOW_GEMV_MM:-}" ]; then
  case "$PLOW_GEMV_MM" in ''|*[!0-9]*) echo "PLOW_GEMV_MM must be a number" >&2; exit 1;; esac
  [ "$PLOW_GEMV_MM" -lt 1 ] || [ "$PLOW_GEMV_MM" -gt 16 ] && {
    echo "PLOW_GEMV_MM must be 1..16 (PLOW_GEMV_MAXM)" >&2; exit 1; }
  if [ "$PLOW_GEMV_MM" -lt "$GVMM" ] && [ "${PLOW_GEMV_WALK:-0}" != "1" ]; then
    echo "REFUSING: PLOW_GEMV_MM=$PLOW_GEMV_MM < batch $GVMM with PLOW_GEMV_WALK unset." >&2
    echo "  Without the walk, gemv_rows<MM> writes rows 0..MM-1 and leaves the rest STALE." >&2
    exit 1
  fi
  GVMM="$PLOW_GEMV_MM"
fi
WALK="${PLOW_GEMV_WALK:-0}"
# Every DECODE object AND its register check must carry the same bucket, or the cliff gate
# validates an object that is not the one that ships.
DEC="-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=$GVMM -DPLOW_GEMV_WALK=$WALK $DEC_KNOBS"
case "${PLOW_GEMV_MFMA4:-0}" in
  0) ;;
  1) DEC="$DEC -DGV_MFMA4=1" ;;
  *) echo "PLOW_GEMV_MFMA4 must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_MOE_TILE_BINSEARCH:-0}" in
  0) ;;
  1) DEC="$DEC -DPLOW_MOE_TILE_BINSEARCH=1" ;;
  *) echo "PLOW_MOE_TILE_BINSEARCH must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$decode_inventory_prune" = 1 ]; then
  DEC="$DEC -DPLOW_DECODE_INVENTORY_PRUNE=1"
fi
echo "   decode GEMV batch bucket: PLOW_GEMV_MM=$GVMM walk=$WALK inventory_prune=$decode_inventory_prune (PLOW_DECODE_BATCH=${PLOW_DECODE_BATCH:-1})"

for B in 0 1; do
  N=$([ "$B" -eq 0 ] && echo prefill || echo decode)
  D=$([ "$B" -eq 0 ] && echo "-DPLOW_BUCKET_DECODE=0" || echo "$DEC")
  genco "$D" "i_$N.co"
  unbundle "i_$N.co" "interp_$N.elf"
done

DECODE_MLA_ELFS=""
if [ "$need_decode_mla" = 1 ]; then
  genco "$DEC -DPLOW_BUCKET_DECODE_MLA=1" i_decode_mla.co
  unbundle i_decode_mla.co interp_decode_mla.elf
  DECODE_MLA_ELFS="interp_decode_mla.elf"
fi

# SEGMENTED-DISPATCH flash object: prefill op set at 4 waves / FA_DC=256, compiling ONLY the class-4
# flash_prefill segment (PLOW_BUCKET_FLASH). 1 wave/SIMD => 512-reg budget; the Q-hoist spills Q to
# on-chip scratch by design (cheaper than the L2 re-read it replaces).
genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 $DSA_FLASH $NOPE_FLASH $FLASH_TR16" i_flash.co
unbundle i_flash.co interp_flash.elf

# GLOBAL-QUEUE variant (Experiment E1). Same op set / tiles / wave count as the static prefill+decode
# objects — ONLY the scheduling loop differs (one shared atomic cursor vs static per-CU streams), so an
# A/B isolates scheduling. GLOBAL QUEUE IS THE DEFAULT scheduler (E1: decode win on both models, prefill
# a MEASURED -8.4% prefill win on 31B too, 163 vs 178 ms at T=1024 — see interp.hip) — build the
# _gq objects unless PLOW_NO_GQ=1 asks for a static-only build. PLOW_GQ_BATCH
# stays 1 (raising it starves parallelism, measured 6-8x slower).
BUILD_GQ=1; [ -n "${PLOW_NO_GQ:-}" ] && BUILD_GQ=0
GQB="${PLOW_GQ_BATCH:-1}"
# FP8 DECODE object (PLOW_FP8=1). A SEPARATE decode interpreter carrying the fp8 w8a16 GEMV arms
# (GEMV_FP8 / GEMV_GLU_FP8) IN ADDITION TO the bf16 GEMV_GLU/QKV arms — they used to replace them,
# which dropped the one bf16 GEMM every fp8 PREFILL packet still emits (the lm_head) and produced
# all-zero logits; see the design notes §4. Additive is register-free on ROCm 7.2.4
# (248/occ 2/spill 0 either way). Mirrors Gemma's OWN decode flags (FA_DEC_VPIPE default 0, no
# PLOW_FLASH_HD128 — Gemma runs segmented flash), only adding PLOW_FP8=1. Built only when asked.
BUILD_FP8=0; [ "${PLOW_FP8:-0}" = 1 ] && BUILD_FP8=1
# FP8 KV-CACHE decode object (PLOW_FP8_KV=1). e4m3 K/V storage+read halves the decode KV stream.
# The DECODE object swaps FLASH_DECODE for the fp8 flash + adds HeadNormRopeFp8, and keeps the fp8
# GEMV weight arms. The KV axis stays a SWAP where the weight axis is additive: a packet's KV is
# uniformly one encoding, and the 4-wave flash object is where the register budget is tight.
# `-DPLOW_FP8=1` here is the WEIGHT axis, not implied by fp8-KV — the cmake table composes the two
# separately now, so a bf16-weight/fp8-KV object is expressible. No FA_DEC_VPIPE (bf16-only).
# Only the DECODE object is built here — the synthetic-KV decode sweep never runs prefill. Gemma's
# real-prefill fp8kv path (a separate 4-wave flash object) is out of scope for the timing sweep.
BUILD_FP8KV=0; [ "${PLOW_FP8_KV:-0}" = 1 ] && BUILD_FP8KV=1
# MXFP4 DECODE object (PLOW_MXFP4=1). OCP microscaling e2m1 weights + one E8M0 scale per 32 K.
# Its own flag rather than a mode of PLOW_FP8: they are alternative encodings of the same linear,
# so no object wants both, and separating them keeps the fp8 object's register budget untouched.
BUILD_MXFP4=0; [ "${PLOW_MXFP4:-0}" = 1 ] && BUILD_MXFP4=1
# MLA PREFILL object (PLOW_MLA_PREFILL=1). Adds FLASH_MLA_PREFILL + FLASH_GATHER_PREFILL and the
# latent epilogue (MLA_MERGE_FOLD / O_UV_FOLD) to the PREFILL bucket -- the arms whose absence
# meant Kimi K2.7 / DeepSeek / GLM 5.2 could decode on gfx950 but could not prefill through their
# own attention. Its own flag because the prefill bucket sits AT the 256/occ-2 cliff and a Gemma
# prefill object must not pay for MLA it never runs.
BUILD_MLA=0; [ "${PLOW_MLA_PREFILL:-0}" = 1 ] && BUILD_MLA=1
# MoE PREFILL object (PLOW_MOE_PREFILL=1). Adds the grouped-expert prefill ops 83-87
# (MOE_ROUTER_TOPK_PF / MOE_ALIGN_PF / MOE_GROUP_GLU_PF / MOE_GROUP_DOWN_PF / MOE_COMBINE_PF) —
# the FFN half of an MLA-family prompt. THE FLAG EXISTED IN interp.hip AND NOTHING HERE EVER
# DEFINED IT, so ops 83-87 were compiled into ZERO shipped objects: the same §4 shape as
# PLOW_GEMV_MM and interp_prefill_mxfp4 above, an arm that is written, register-gated and
# unreachable. An MLA prefill object with no MoE arm is attention-complete and FFN-incomplete
# (the expert packets fall through to `default:` and write NOTHING), which is why this builds
# the MLA+MoE COMBINATION and PLOW_MOE_PREFILL=1 turns the MLA arm on with it — exactly the
# `interp_prefill_mla_moe` row of runtime/CMakeLists.txt, and what a GLM-5.2 / Kimi K2.7 /
# DeepSeek whole-layer prefill object must be. Measured cost on ROCm 7.2.4 / gfx950: identical
# to the bf16 prefill, 256 VGPR / 0 AGPR / occ 2 / spill 2 — the grouped GEMM costs occupancy
# nothing, and its 81920 B double-buffered tile fits inside the interpreter's existing arena.
BUILD_MOE=0; [ "${PLOW_MOE_PREFILL:-0}" = 1 ] && { BUILD_MOE=1; BUILD_MLA=1; }
# L2-DOMAIN DISPATCH (PLOW_L2_PLACE=1). Makes a workgroup take its global-queue window from the
# XCD it is PHYSICALLY running on (HW_REG_XCC_ID) instead of from the host's `cur_seg`, so all 8
# domains drain concurrently in one launch and a consumer reads its producer out of the same XCD's
# 4 MiB L2 instead of across the fabric. Pairs with a blob built by `plowc` under PLOW_L2_PLACE=1.
#
# Applied to the DECODE object in place rather than built as a fourth object, and both halves of
# that matter:
#   - IN PLACE, because a separate object would export the SAME `plow_interp_dec_gfx950_gq`
#     symbol and the two could not co-load. Making it a build-recipe choice needs no runtime
#     kernel selection at all.
#   - DECODE ONLY, because a placed PREFILL packet does not exist to run: `seg` carries the wave
#     class the host relaunches at, and `Builder::finish` declines placement for any program with
#     more than one wave class. Decode has no FlashPrefill op, so every op is class 8, `seg` is
#     uniformly 0, and there is nothing to overwrite. A prefill twin would be an arm with nothing
#     routing to it — the §4 shape this tree keeps rediscovering.
#
# SAFE IN BOTH DIRECTIONS, so the pairing cannot silently corrupt: an L2 object given an UNPLACED
# blob sees `prog.l2_domains == 0` and falls back to `cur_seg` (byte-identical behaviour), and a
# placed blob given a NON-L2 runtime is refused at load by `devblob.rs`'s F_L2DOM guard.
#
# Measured cost on ROCm 7.2.4 / gfx950 (`scripts/l2_regcheck.sh`): 248 VGPR / 0 AGPR / occ 2 /
# spill 0 — IDENTICAL to the plain GQ decode object. The XCC id lands in a scalar register, so
# the hottest loop in the interpreter pays nothing for it.
L2D=""
HIERD=""
HIERPF=""
HIER_STATUS="off"
if [ "${PLOW_L2_PLACE:-1}" = 1 ]; then
  L2D="-DPLOW_L2_PLACE_DISPATCH"
  if [ "${PLOW_GATE_HIER:-1}" = 1 ]; then HIERD="-DPLOW_GATE_HIER=1"; HIER_STATUS="on"; fi
  echo "   per-XCD packet queues: prefill+decode L2 dispatch; decode hierarchy=$HIER_STATUS"
fi
case "${PLOW_GATE_HIER_PF:-0}" in
  0) ;;
  1) [ -n "$L2D" ] || { echo "PLOW_GATE_HIER_PF=1 requires PLOW_L2_PLACE=1" >&2; exit 2; }
     HIERPF="-DPLOW_GATE_HIER=1" ;;
  *) echo "PLOW_GATE_HIER_PF must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$BUILD_GQ" = 1 ]; then
  for B in 0 1; do
    N=$([ "$B" -eq 0 ] && echo prefill || echo decode)
    D=$([ "$B" -eq 0 ] && echo "-DPLOW_BUCKET_DECODE=0 $L2D $HIERPF" || echo "$DEC $L2D $HIERD")
    genco "$D -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" "i_${N}_gq.co"
    unbundle "i_${N}_gq.co" "interp_${N}_gq.elf"
  done
  # The L2 decode object is register-checked below with the same 256/occ-2 budget as the plain
  # one — it is the SAME object with one extra scalar read, and if that ever costs occupancy the
  # build must fail rather than ship a decode kernel at occ 1.
  # GQ flash object (Gemma's segmented 4-wave flash segment under the global queue).
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 $DSA_FLASH $NOPE_FLASH $FLASH_TR16 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" i_flash_gq.co
  unbundle i_flash_gq.co interp_flash_gq.elf
  if [ "$need_decode_mla" = 1 ]; then
    genco "$DEC $L2D $HIERD -DPLOW_BUCKET_DECODE_MLA=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_decode_mla_gq.co
    unbundle i_decode_mla_gq.co interp_decode_mla_gq.elf
    DECODE_MLA_ELFS="$DECODE_MLA_ELFS interp_decode_mla_gq.elf"
  fi
fi

# FP8 decode objects (static + GQ), each swapping the bf16 GEMV_GLU/QKV arms for the fp8 ones. Same
# flags as Gemma's bf16 decode above, only adding PLOW_FP8=1. The bf16 objects are unaffected.
if [ "$BUILD_FP8" = 1 ]; then
  genco "$DEC -DPLOW_FP8=1" i_decode_fp8.co
  unbundle i_decode_fp8.co interp_decode_fp8.elf
  if [ "$BUILD_GQ" = 1 ]; then
    genco "$DEC -DPLOW_FP8=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_decode_fp8_gq.co
    unbundle i_decode_fp8_gq.co interp_decode_fp8_gq.elf
  fi
  # w8a8 PREFILL object. The fp8 GEMM arms SWAP for the bf16 ones (an fp8 program emits no bf16
  # GEMM), so this stays under the SAME 256/occ-2 cliff as the bf16 prefill — checked below.
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_FP8=1" i_prefill_fp8.co
  unbundle i_prefill_fp8.co interp_prefill_fp8.elf
  if [ "$BUILD_GQ" = 1 ]; then
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_FP8=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" i_prefill_fp8_gq.co
    unbundle i_prefill_fp8_gq.co interp_prefill_fp8_gq.elf
  fi
fi

# FP8 KV-CACHE decode objects (static + GQ). Swap FLASH_DECODE for the fp8 flash + add HeadNormRopeFp8,
# keeping the fp8 GEMV weight arms. bf16/fp8-weight objects above are unaffected.
if [ "$BUILD_FP8KV" = 1 ]; then
  FP8KV_WEIGHTS="-DPLOW_FP8=1"
  if [ "$BUILD_MXFP4" = 1 ]; then
    FP8KV_WEIGHTS="-DPLOW_MXFP4=1"
    if [ "$BUILD_MOE" = 1 ] && [ "$GVMM" -gt 1 ]; then
      FP8KV_WEIGHTS="$FP8KV_WEIGHTS -DPLOW_MOE_PREFILL=1 -DPLOW_MOE_PF_A4W4=1"
    fi
  fi
  genco "$DEC $FP8KV_WEIGHTS -DPLOW_FP8_KV=1" i_decode_fp8kv.co
  unbundle i_decode_fp8kv.co interp_decode_fp8kv.elf
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 -DPLOW_FP8_KV=1 $DSA_FLASH $NOPE_FLASH $FLASH_TR16 $FLASH_SCALE_HOIST" i_flash_fp8kv.co
  unbundle i_flash_fp8kv.co interp_flash_fp8kv.elf
  if [ "$BUILD_GQ" = 1 ]; then
    genco "$DEC $FP8KV_WEIGHTS -DPLOW_FP8_KV=1 $L2D $HIERD -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_decode_fp8kv_gq.co
    unbundle i_decode_fp8kv_gq.co interp_decode_fp8kv_gq.elf
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 -DPLOW_FP8_KV=1 $DSA_FLASH $NOPE_FLASH $FLASH_TR16 $FLASH_SCALE_HOIST -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" i_flash_fp8kv_gq.co
    unbundle i_flash_fp8kv_gq.co interp_flash_fp8kv_gq.elf
  fi
fi

# MXFP4 objects (static + GQ), decode AND prefill.
#
# THE PREFILL OBJECT WAS NAMED AND NEVER BUILT. `scripts/gfx950_objects.py:161` routes an
# mxfp4 packet's prefill bucket to `interp_prefill_mxfp4`, and nothing here produced it — so a
# host following the object table asked for a file that did not exist. This is the same §4
# shape as the arm it carries: `PLOW_DOP_GEMM_MXFP4` had already been written, register-gated,
# and left unreachable once (interp.hip:854 records that), and building only the decode half
# left it unreachable a second way. Measured: 256 VGPR / 0 AGPR / occ 2 / spill 2, i.e. the
# bf16 prefill budget unchanged — the fp4 arms are additive (w4a16 mixes with bf16 by
# construction) and they cost nothing.
MXFP4_ELFS=""
if [ "$BUILD_MXFP4" = 1 ]; then
  MXDEC="$DEC -DPLOW_MXFP4=1"
  if [ "$BUILD_MOE" = 1 ] && [ "$GVMM" -gt 1 ]; then
    MXDEC="$MXDEC -DPLOW_MOE_PREFILL=1 -DPLOW_MOE_PF_A4W4=1"
  fi
  genco "$MXDEC" i_decode_mxfp4.co
  unbundle i_decode_mxfp4.co interp_decode_mxfp4.elf
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1" i_prefill_mxfp4.co
  unbundle i_prefill_mxfp4.co interp_prefill_mxfp4.elf
  MXFP4_ELFS="interp_decode_mxfp4.elf interp_prefill_mxfp4.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    genco "$MXDEC $L2D $HIERD -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_decode_mxfp4_gq.co
    unbundle i_decode_mxfp4_gq.co interp_decode_mxfp4_gq.elf
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" i_prefill_mxfp4_gq.co
    unbundle i_prefill_mxfp4_gq.co interp_prefill_mxfp4_gq.elf
    MXFP4_ELFS="interp_decode_mxfp4.elf interp_decode_mxfp4_gq.elf interp_prefill_mxfp4.elf interp_prefill_mxfp4_gq.elf"
  fi
fi

# MLA prefill objects (static + GQ).
MLA_ELFS=""
if [ "$BUILD_MLA" = 1 ]; then
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1" i_prefill_mla.co
  unbundle i_prefill_mla.co interp_prefill_mla.elf
  MLA_ELFS="interp_prefill_mla.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" i_prefill_mla_gq.co
    unbundle i_prefill_mla_gq.co interp_prefill_mla_gq.elf
    MLA_ELFS="interp_prefill_mla.elf interp_prefill_mla_gq.elf"
  fi
fi

if [ "$BUILD_MLA" = 1 ] && [ "$BUILD_MXFP4" = 1 ]; then
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1 -DPLOW_MLA_PREFILL=1" i_prefill_mxfp4_mla.co
  unbundle i_prefill_mxfp4_mla.co interp_prefill_mxfp4_mla.elf
  MLA_ELFS="$MLA_ELFS interp_prefill_mxfp4_mla.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1 -DPLOW_MLA_PREFILL=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" i_prefill_mxfp4_mla_gq.co
    unbundle i_prefill_mxfp4_mla_gq.co interp_prefill_mxfp4_mla_gq.elf
    MLA_ELFS="$MLA_ELFS interp_prefill_mxfp4_mla_gq.elf"
  fi
fi

# MLA+MoE prefill objects (static + GQ) — the whole-layer prefill object.
MOE_ELFS=""
if [ "$BUILD_MOE" = 1 ]; then
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1" i_prefill_mla_moe.co
  unbundle i_prefill_mla_moe.co interp_prefill_mla_moe.elf
  MOE_ELFS="interp_prefill_mla_moe.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" i_prefill_mla_moe_gq.co
    unbundle i_prefill_mla_moe_gq.co interp_prefill_mla_moe_gq.elf
    MOE_ELFS="interp_prefill_mla_moe.elf interp_prefill_mla_moe_gq.elf"
  fi
fi

if [ "$BUILD_MOE" = 1 ] && [ "$BUILD_MXFP4" = 1 ]; then
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1 -DPLOW_MOE_PF_A4W4=1 -DPLOW_MXFP4=1 $A4W4_BK" i_prefill_mla_moe_a4w4_full.co
  unbundle i_prefill_mla_moe_a4w4_full.co interp_prefill_mla_moe_a4w4_full.elf
  MOE_ELFS="$MOE_ELFS interp_prefill_mla_moe_a4w4_full.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1 -DPLOW_MOE_PF_A4W4=1 -DPLOW_MXFP4=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF $A4W4_BK" i_prefill_mla_moe_a4w4_full_gq.co
    unbundle i_prefill_mla_moe_a4w4_full_gq.co interp_prefill_mla_moe_a4w4_full_gq.elf
    MOE_ELFS="$MOE_ELFS interp_prefill_mla_moe_a4w4_full_gq.elf"
  fi
fi

if [ "$BUILD_FP8KV" = 1 ] && [ "$BUILD_MOE" = 1 ] && [ "$BUILD_MXFP4" = 1 ]; then
  FP8KV_A4W4="-DPLOW_BUCKET_DECODE=0 -DPLOW_FP8_KV=1 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1 -DPLOW_MOE_PF_A4W4=1 -DPLOW_MXFP4=1 $A4W4_BK"
  genco "$FP8KV_A4W4" i_prefill_fp8kv_mla_moe_a4w4_full.co
  unbundle i_prefill_fp8kv_mla_moe_a4w4_full.co interp_prefill_fp8kv_mla_moe_a4w4_full.elf
  MOE_ELFS="$MOE_ELFS interp_prefill_fp8kv_mla_moe_a4w4_full.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    genco "$FP8KV_A4W4 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" i_prefill_fp8kv_mla_moe_a4w4_full_gq.co
    unbundle i_prefill_fp8kv_mla_moe_a4w4_full_gq.co interp_prefill_fp8kv_mla_moe_a4w4_full_gq.elf
    MOE_ELFS="$MOE_ELFS interp_prefill_fp8kv_mla_moe_a4w4_full_gq.elf"
  fi
fi

# GEMMA-4 26B-A4B MoE objects (PLOW_GEMMA_MOE=1).                            [GEMMA4-MOE-AMD]
#
# Nineteen opcodes (61-77, 81/82) that the sm_120 interpreter has dispatched since the 26B-A4B
# bring-up and the AMD one had NO arm for — so the model could not run on AMD at all, and the way
# it could not run was SILENT: this interpreter's dispatch `default:` writes nothing. Two flags,
# because the two halves have different costs:
#
#   PLOW_MOE_GEMMA     ops 61-72, the DECODE family (router split, fused-gate_up experts,
#                      combines). Wave-per-output GEMVs; measured FREE on gfx942 — the decode
#                      object is 251 VGPR / occ 2 / 0 spill / 64520 B LDS with and without it.
#   PLOW_MOE_GEMMA_PF  ops 73-77 + 81/82, the grouped-expert PREFILL. A second full MFMA body;
#                      measured 256 VGPR / occ 2 / spill 2 / 64520 B, i.e. also free at the
#                      cliff, but it is its own flag for the reason PLOW_MOE_PREFILL is: a model
#                      that never prefills MoE should not carry the GEMM.
#
# The DECODE object gets both halves when a batched decode is asked for: at PLOW_DECODE_BATCH > 1
# the decode program groups by expert exactly as prefill does (see the note above the arms in
# interp.hip), so the grouped GEMM has to be there too.
BUILD_GEMMA_MOE=0; [ "${PLOW_GEMMA_MOE:-0}" = 1 ] && BUILD_GEMMA_MOE=1
# Kimi-K3 / GLM-5.3-Flash packets (`PLOW_K3=1` in PLOW_PACKET_OBJECT_REQUIRES): the loader resolves
# `interp_prefill_k3_moe` and `interp_decode_k3` (exec/amd/object.rs `PrefillArm`), i.e. the
# MLA+MoE prefill object and the decode object under their K3 names. The paired config header
# already derives PLOW_K3 / PLOW_KDA_CHUNK from the packet; -DPLOW_K3=1 restates it as the
# gfx942 rows do.
# K3_PF_MOE: grouped-MoE prefill axes, all value-identical (op_moe.h notes). GLM-5.3-Flash TP8
# 4k prefill, MI350X: GroupDownPf 58.6 -> 20.2 ms, GroupGluPf 36.4 -> 24.3 ms (tile binsearch over
# 288 experts, down-epilogue row-metadata hoist, raw fp8 B held across the MFMAs), CombinePf
# 23.2 -> 6.3 ms (vector arm).
K3_PF_MOE="-DPLOW_MOE_TILE_BINSEARCH=1 -DPLOW_MOE_PF_EPI=1 -DPLOW_MOE_PF_BRAW=1 -DPLOW_COMBINE_VEC=1"
# Value-identical K3 prefill arms, default on (=0 rolls back): the wave-per-token router tail
# (op 83, now up to 320 experts; GLM-5.3-Flash has 288) and the all-wave align scatter (op 84).
case "${PLOW_K3_PF_ROUTER_WAVE:-1}" in
  0) ;;
  1) K3_PF_MOE="$K3_PF_MOE -DPLOW_MOE_ROUTER_PF_WAVE=1" ;;
  *) echo "PLOW_K3_PF_ROUTER_WAVE must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_K3_PF_ALIGN_WAVES:-1}" in
  0) ;;
  1) K3_PF_MOE="$K3_PF_MOE -DPLOW_MOE_ALIGN_WAVES=1" ;;
  *) echo "PLOW_K3_PF_ALIGN_WAVES must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_K3_PF_XR_SCHED=aiter (opt-in, default off): the 16-byte collective schedule (bit-identical
# data movement, op_collective.h) on capped workgroups — two-shot NWG/NWG_RS, seam reduce-scatter
# NWG_SRS, seam all-gather NWG_SAG. PLOW_K3_PF_FOLD_TB (default 8 on GLM-5.3-Flash packets, 0 on
# other K3 packets; 0 rolls back): the token-blocked MlaMergeFold (op_attention_common.h). The
# gfx950 V=128 oracle saw packed-FMA rounding moves, so it is qualified per model: GLM-5.3-Flash
# (V=256) validate streams byte-equal at every rung and 4k-60k prompts, fold 9.8 -> 3.8 ms at 4k.
case "${PLOW_K3_PF_XR_SCHED:-0}" in
  0) ;;
  aiter) K3_PF_MOE="$K3_PF_MOE -DPLOW_XR_SCHED_AITER=1 -DPLOW_XR_SCHED_NWG=${PLOW_XR_SCHED_NWG:-48} -DPLOW_XR_SCHED_NWG_RS=${PLOW_XR_SCHED_NWG_RS:-16} -DPLOW_XR_SCHED_NWG_SRS=${PLOW_XR_SCHED_NWG_SRS:-16} -DPLOW_XR_SCHED_NWG_SAG=${PLOW_XR_SCHED_NWG_SAG:-48}" ;;
  *) echo "PLOW_K3_PF_XR_SCHED must be 0 or aiter" >&2; exit 2 ;;
esac
# PLOW_K3_PF_SP_VEC (default 0; 2/4/8): the seq-par seam reduce-scatter/all-gather (ops 25/26)
# move that many bf16 per peer access (PLOW_XR_SP_VEC, op_collective.h). Bit-identical.
# MEASURED, keep 0: rs+ag at 1024 rows 106.6 -> 94.0 us (V=8) but 4096 rows 244.6 -> 362.6 and
# 8192 rows 416.7 -> 720.9 (tp_allreduce_prefill_bench TP_PHASES, hidden 4096, 256 WGs).
case "${PLOW_K3_PF_SP_VEC:-0}" in
  0) ;;
  2|4|8) K3_PF_MOE="$K3_PF_MOE -DPLOW_XR_SP_VEC=$PLOW_K3_PF_SP_VEC" ;;
  *) echo "PLOW_K3_PF_SP_VEC must be 0, 2, 4 or 8" >&2; exit 2 ;;
esac
# PLOW_RN_ROWS (default 1 here; gfx942 ships 2): d_rmsnorm issues R rows of loads per workgroup
# before reducing any (op_norm.h multi-row arm; same element map and tree, bit-identical). Plain
# prefill rows only: a packed program keeps the single-row loop.
case "${PLOW_RN_ROWS:-1}" in
  0|1) ;;
  2|4) K3_PF_MOE="$K3_PF_MOE -DPLOW_RN_ROWS=$PLOW_RN_ROWS" ;;
  *) echo "PLOW_RN_ROWS must be 1, 2 or 4" >&2; exit 2 ;;
esac
# PLOW_HC_POST_ROWS (default 1): the mHC HyperConnPost issues R band rows of loads per workgroup
# before storing (op_hyperconn.h; same arithmetic, bit-identical).
case "${PLOW_HC_POST_ROWS:-1}" in
  0|1) ;;
  2|4) K3_PF_MOE="$K3_PF_MOE -DPLOW_HC_POST_ROWS=$PLOW_HC_POST_ROWS" ;;
  *) echo "PLOW_HC_POST_ROWS must be 1, 2 or 4" >&2; exit 2 ;;
esac
# PLOW_GEMV_F32_PF_TILE=RxCN (default 2x3, the plain arm): the prefill GemvF32 (mHC mix, N = 24,
# K = 16384) output tile per wave; each output keeps its lane-to-k map and term order (bit-identical).
case "${PLOW_GEMV_F32_PF_TILE:-2x3}" in
  2x3) ;;
  4x3|2x6|4x6) K3_PF_MOE="$K3_PF_MOE -DPLOW_GEMV_F32_PF_R=${PLOW_GEMV_F32_PF_TILE%x*} -DPLOW_GEMV_F32_PF_CN=${PLOW_GEMV_F32_PF_TILE#*x}" ;;
  *) echo "PLOW_GEMV_F32_PF_TILE must be 2x3, 4x3, 2x6 or 4x6" >&2; exit 2 ;;
esac
FOLD_TB_DEFAULT=0
if [ "$GLM53_FLASH" = 1 ]; then FOLD_TB_DEFAULT=8; fi
case "${PLOW_K3_PF_FOLD_TB:-$FOLD_TB_DEFAULT}" in
  0|1) ;;
  2|4|8) K3_PF_MOE="$K3_PF_MOE -DPLOW_MLA_FOLD_TB=${PLOW_K3_PF_FOLD_TB:-$FOLD_TB_DEFAULT}" ;;
  *) echo "PLOW_K3_PF_FOLD_TB must be 0, 2, 4 or 8" >&2; exit 2 ;;
esac
# PLOW_MOE_PF_DEEP (default 1; =0 rolls back): two raw register stages over the LDS ping-pong in
# the grouped k-loop, with the A-gather index hoisted per tile (PLOW_MOE_PF_GH=1), so each load
# has two MFMA phases to land instead of one. Bit-identical (same LDS bytes, same MFMA order).
case "${PLOW_MOE_PF_DEEP:-1}" in
  0) ;;
  1) K3_PF_MOE="$K3_PF_MOE -DPLOW_MOE_PF_GH=1 -DPLOW_MOE_PF_DEEP=1" ;;
  *) echo "PLOW_MOE_PF_DEEP must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_MOE_PF_HDR (default 1; =0 rolls back): a tile's expert from a register copy of the tile
# prefix (ballot count) instead of the binary search's nine dependent loads per tile. Bit-identical.
case "${PLOW_MOE_PF_HDR:-1}" in
  0) ;;
  1) K3_PF_MOE="$K3_PF_MOE -DPLOW_MOE_PF_HDR=1" ;;
  *) echo "PLOW_MOE_PF_HDR must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_GEMM_PF_REPICK (default 1 here, =0 rolls back): a dense prefill GEMM whose emitted rung fills
# under PLOW_GEMM_PF_FILL_PCT (75) of its workgroups at the chunk's live M takes the largest rung
# that does (interp.hip gemm_pf_repick; bit-identical, the same MFMA and k order).
case "${PLOW_GEMM_PF_REPICK:-1}" in
  0) ;;
  1) K3_PF_MOE="$K3_PF_MOE -DPLOW_GEMM_PF_REPICK=1 -DPLOW_GEMM_PF_FILL_PCT=${PLOW_GEMM_PF_FILL_PCT:-75}" ;;
  *) echo "PLOW_GEMM_PF_REPICK must be 0 or 1" >&2; exit 2 ;;
esac
# PLOW_GEMM_PF_SM_BK=128: the 64x128 GemmSmall rung (and its fp32 router twin) at BK=128 in the K3
# prefill objects only -- half the k-tiles for the N<=256, K=4096 projections that run on 64-128
# workgroups. Arithmetic-preserving (BK only moves the k-tile barriers; op_gemm_common.h). Default
# 128 (=64 rolls back): gemm_tile_sweep on MI350X put e2 (64x128x128) 5-25% ahead of c4 on every
# N<=256, K=4096 GLM-5.3-Flash shape, and validate streams stayed byte-equal.
case "${PLOW_GEMM_PF_SM_BK:-128}" in
  64) ;;
  128) K3_PF_MOE="$K3_PF_MOE -DGM_SM_BK=128" ;;
  *) echo "PLOW_GEMM_PF_SM_BK must be 64 or 128" >&2; exit 2 ;;
esac
K3_ELFS=""
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
   grep -Eq '^#define PLOW_PACKET_OBJECT_REQUIRES "([^"]* )?PLOW_K3=1( |")' "$PLOW_HSACO_CONFIG"; then
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1 -DPLOW_K3=1 $K3_PF_MOE $K3_PF_ATTN" i_prefill_k3_moe.co
  unbundle i_prefill_k3_moe.co interp_prefill_k3_moe.elf
  genco "$DEC -DPLOW_K3=1" i_decode_k3.co
  unbundle i_decode_k3.co interp_decode_k3.elf
  K3_ELFS="interp_prefill_k3_moe.elf interp_decode_k3.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1 -DPLOW_K3=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF $K3_PF_MOE $K3_PF_ATTN" i_prefill_k3_moe_gq.co
    unbundle i_prefill_k3_moe_gq.co interp_prefill_k3_moe_gq.elf
    genco "$DEC $L2D $HIERD -DPLOW_K3=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_decode_k3_gq.co
    unbundle i_decode_k3_gq.co interp_decode_k3_gq.elf
    K3_ELFS="$K3_ELFS interp_prefill_k3_moe_gq.elf interp_decode_k3_gq.elf"
  fi
fi
# PACKED-PREFILL FAMILY OBJECTS (PLOW_PACKED_PREFILL_CONSUMERS=1; default on when the packet
# carries packed-prefill siblings, `=0` skips them): the three lean objects such a packet opens
# by name (exec/amd.rs `load_packed_family`), the gfx950 twins of build_gfx942.sh's rows. The
# flash object takes the NoPE V2 arm when the packet requires it (marker
# `plow_packed_prefill_mla_nope_1`). Not in ALL_ELFS: the default-object check there refuses
# consumer arms by design.
PACKED_ELFS=""
PK_DEFAULT=0
if [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
  grep -q '^#define PLOW_PACKET_HAS_PACKED_PREFILL_TOPOLOGY 1' "$PLOW_HSACO_CONFIG"; then
  PK_DEFAULT=1
fi
if [ "${PLOW_PACKED_PREFILL_CONSUMERS:-$PK_DEFAULT}" = 1 ]; then
  rm -f interp_packed_mla_norm*.elf interp_packed_mla_flash*.elf interp_packed_kda*.elf
  PK_NORM="-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_BUCKET_PACKED_MLA_NORM=1 -DPLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS=1"
  PK_FLASH="-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 $NOPE_FLASH $FLASH_TR16 -DPLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS=1"
  PK_KDA="-DPLOW_BUCKET_DECODE=0 -DPLOW_K3=1 -DPLOW_BUCKET_PACKED_KDA=1 -DPLOW_PACKED_PREFILL_KDA_CONSUMERS=1"
  for fam in norm flash kda; do
    case $fam in
      norm) AX="$PK_NORM"; S=interp_packed_mla_norm ;;
      flash) AX="$PK_FLASH"; S=interp_packed_mla_flash ;;
      kda) AX="$PK_KDA"; S=interp_packed_kda ;;
    esac
    genco "$AX" "i_pk_$fam.co"
    unbundle "i_pk_$fam.co" "$S.elf"
    PACKED_ELFS="$PACKED_ELFS $S.elf"
    if [ "$BUILD_GQ" = 1 ]; then
      genco "$AX -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" "i_pk_${fam}_gq.co"
      unbundle "i_pk_${fam}_gq.co" "${S}_gq.elf"
      PACKED_ELFS="$PACKED_ELFS ${S}_gq.elf"
    fi
  done
  echo "   packed-prefill family objects:$PACKED_ELFS"
fi
GMOE_ELFS=""
if [ "$BUILD_GEMMA_MOE" = 1 ]; then
  # `if`, not `[ ... ] && GMD=...`: the file already records what that pattern costs under
  # `set -e` when it lands last in a block.
  GMD="-DPLOW_MOE_GEMMA=1"
  if [ "$GVMM" -gt 1 ]; then GMD="$GMD -DPLOW_MOE_GEMMA_PF=1"; fi
  genco "$DEC $GMD" i_decode_gmoe.co
  unbundle i_decode_gmoe.co interp_decode_gmoe.elf
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MOE_GEMMA_PF=1" i_prefill_gmoe.co
  unbundle i_prefill_gmoe.co interp_prefill_gmoe.elf
  GMOE_ELFS="interp_decode_gmoe.elf interp_prefill_gmoe.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    genco "$DEC $GMD -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_decode_gmoe_gq.co
    unbundle i_decode_gmoe_gq.co interp_decode_gmoe_gq.elf
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MOE_GEMMA_PF=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" i_prefill_gmoe_gq.co
    unbundle i_prefill_gmoe_gq.co interp_prefill_gmoe_gq.elf
    GMOE_ELFS="$GMOE_ELFS interp_decode_gmoe_gq.elf interp_prefill_gmoe_gq.elf"
  fi
  # The marker symbols plowrt's `check_moe_gemma_arms` reads out of .symtab. Absence is what it
  # refuses on, so a build that silently dropped the flag must fail HERE and not at first token.
  for e in $GMOE_ELFS; do
    case "$e" in
      *decode*) want=plow_moe_gemma_arms_1;;
      *)        want=plow_moe_gemma_pf_arms_1;;
    esac
    "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/llvm-nm "$e" | grep -q "$want" || {
      echo "FAIL: $e does not advertise $want — the Gemma MoE arms were not compiled in"; exit 1; }
  done
fi

# Golden wrappers call the SAME __device__ functions the interpreter does, so rebuild them with it.
hipcc --offload-arch="$ARCH" -O3 -w --genco "$R/amd/test_kernels.hip" -o tk.co $INC
unbundle tk.co test_kernels.elf

# HOST harness. Two COHERENT pairings, never a mix:
#   default            SYSTEM gcc in a CLEAN env against the system ROCm. The nix shell's gcc
#                      bakes a RUNPATH to nix glibc 2.42 via LD_RUN_PATH while the ELF
#                      interpreter stays the system 2.35 — the mismatch aborts with
#                      "*** stack smashing detected ***", which reads as a buffer overflow
#                      and is not one. Hence env -i + /usr/bin/gcc + the RUNPATH gate.
#   PLOW_HOST_CC=<cc>  the nix pairing (the dev shell exports it): ROCM_PATH there is the nix
#                      ROCm SDK, whose libhsa-runtime64 carries nix-glibc symbol versions the
#                      system ld cannot resolve (undefined arc4random@GLIBC_2.36). The nix cc
#                      sets interpreter AND runpath to the SAME glibc, so that side is the
#                      consistent one — and the RUNPATH gate must NOT run, a nix binary is
#                      supposed to carry one.
if [ -n "${PLOW_HOST_CC:-}" ]; then
  "$PLOW_HOST_CC" -O2 -std=gnu11 -o chat \
      "$R/tests/gemma4_chat.c" "$R/amd/hsa_backend.c" \
      -I"${ROCM_PATH:-/opt/rocm}"/include -L"${ROCM_PATH:-/opt/rocm}"/lib \
      -Wl,-rpath,"${ROCM_PATH:-/opt/rocm}"/lib -lhsa-runtime64 -lm
else
  /usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o chat \
      "$R/tests/gemma4_chat.c" "$R/amd/hsa_backend.c" \
      -I"${ROCM_PATH:-/opt/rocm}"/include -L"${ROCM_PATH:-/opt/rocm}"/lib -lhsa-runtime64 -lm
  readelf -d chat | grep -qi runpath && { echo "FAIL: RUNPATH leaked into the host binary"; exit 1; }
fi

# REGISTER CLIFF as a build error. The 8-wave interpreters must stay <= 256 (2 waves/SIMD); the
# 4-wave flash object <= 512 (1 wave/SIMD), and ITS VGPR spill is INTENTIONAL (Q-hoist to scratch).
check() { # <name> <defs> <max-total> <min-occ>
  local U V A O S
  U=$(hipcc --offload-arch="$ARCH" -O3 -w $2 --genco \
        -Rpass-analysis=kernel-resource-usage "$R/amd/interp.hip" -o /dev/null $INC 2>&1)
  V=$(echo "$U" | grep -oP 'VGPRs: \K\d+' | head -1)
  A=$(echo "$U" | grep -oP 'AGPRs: \K\d+' | head -1)
  O=$(echo "$U" | grep -oP 'Occupancy \[waves/SIMD\]: \K\d+' | head -1)
  S=$(echo "$U" | grep -oP 'VGPRs Spill: \K\d+' | head -1)
  printf "   %-8s VGPR=%-3s AGPR=%-3s total=%-3s occ=%s spill=%s\n" "$1" "$V" "$A" "$((V + A))" "$O" "$S"
  if [ "$((V + A))" -gt "$3" ] || [ "$O" -lt "$4" ]; then
    echo "BUILD FAILED: $1 over the register cliff (total $((V + A)) > $3, or occ $O < $4)." >&2
    rm -f "interp_$1.elf"; exit 1
  fi
}
check prefill "-DPLOW_BUCKET_DECODE=0" 256 2
check decode  "$DEC" 256 2
check flash   "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 $DSA_FLASH $NOPE_FLASH $FLASH_TR16" 512 1
if [ "$need_decode_mla" = 1 ]; then
  check decode_mla "$DEC -DPLOW_BUCKET_DECODE_MLA=1" 256 2
fi
# The GQ loop adds a shared cursor + claim broadcast; it must NOT push prefill past 256/occ-2. If it
# does and no minimal-live-set fix recovers it, that is itself a recordable E1 finding (GQ incompatible
# with occ-2 prefill). These run only when the GQ objects were built.
GQ_ELFS=""
if [ "$BUILD_GQ" = 1 ]; then
  check prefill_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" 256 2
  check decode_gq  "$DEC $L2D -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
  check flash_gq   "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 $DSA_FLASH $NOPE_FLASH $FLASH_TR16 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" 512 1
  if [ "$need_decode_mla" = 1 ]; then
    check decode_mla_gq "$DEC $L2D $HIERD -DPLOW_BUCKET_DECODE_MLA=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
  fi
  GQ_ELFS="interp_prefill_gq.elf interp_decode_gq.elf interp_flash_gq.elf"
fi
# The fp8 arms swap (not add) against the bf16 GEMV_GLU/QKV arms, so they must stay under the same
# 256/occ-2 cliff. Checked only when the fp8 objects were built.
FP8_ELFS=""
if [ "$BUILD_FP8" = 1 ]; then
  check decode_fp8 "$DEC -DPLOW_FP8=1" 256 2
  check prefill_fp8 "-DPLOW_BUCKET_DECODE=0 -DPLOW_FP8=1" 256 2
  FP8_ELFS="interp_decode_fp8.elf interp_prefill_fp8.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    check decode_fp8_gq "$DEC -DPLOW_FP8=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
    check prefill_fp8_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_FP8=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" 256 2
    FP8_ELFS="interp_decode_fp8.elf interp_decode_fp8_gq.elf interp_prefill_fp8.elf interp_prefill_fp8_gq.elf"
  fi
fi
FP8KV_ELFS=""
if [ "$BUILD_FP8KV" = 1 ]; then
  check decode_fp8kv "$DEC $FP8KV_WEIGHTS -DPLOW_FP8_KV=1" 256 2
  check flash_fp8kv "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 -DPLOW_FP8_KV=1 $DSA_FLASH $NOPE_FLASH $FLASH_TR16 $FLASH_SCALE_HOIST" 512 1
  FP8KV_ELFS="interp_decode_fp8kv.elf interp_flash_fp8kv.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    check decode_fp8kv_gq "$DEC $FP8KV_WEIGHTS -DPLOW_FP8_KV=1 $L2D $HIERD -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
    check flash_fp8kv_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 -DPLOW_FP8_KV=1 $DSA_FLASH $NOPE_FLASH $FLASH_TR16 $FLASH_SCALE_HOIST -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" 512 1
    FP8KV_ELFS="interp_decode_fp8kv.elf interp_decode_fp8kv_gq.elf interp_flash_fp8kv.elf interp_flash_fp8kv_gq.elf"
  fi
fi

if [ "$BUILD_MXFP4" = 1 ]; then
  # Conflict resolution: DECODE arms must use $DEC, which carries -DPLOW_GEMV_MM=$GVMM.
  # Dropping it here would silently revert batched decode to one row for the mxfp4 object
  # only — exactly the class of bug that made PLOW_GEMV_MM unreachable in the first place.
  # The prefill_mxfp4 checks are new and are kept as written.
  check decode_mxfp4 "$MXDEC" 256 2
  # The prefill twin carries the fp4 GEMM at all five tile rungs. Its accumulators are the
  # bf16 ones (w4a16 dequantizes in the B-fetch and issues a bf16 MFMA), and every rung is
  # smaller than the 256x256 arm the object already had, so it must stay at the same cliff.
  check prefill_mxfp4 "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1" 256 2
  if [ "$BUILD_GQ" = 1 ]; then
    check decode_mxfp4_gq "$MXDEC $L2D $HIERD -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
    check prefill_mxfp4_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" 256 2
  fi
fi

if [ "$BUILD_MLA" = 1 ]; then
  check prefill_mla "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1" 256 2
  # `if`, not `[ ... ] && check`: under `set -e` a false `[ ]` as the LAST command of this block
  # is the block's status and kills the script right here — silently, after the objects are
  # already built. Only reachable with PLOW_NO_GQ=1, which is exactly the rare path nobody runs.
  if [ "$BUILD_GQ" = 1 ]; then
    check prefill_mla_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" 256 2
  fi
fi
if [ "$BUILD_MLA" = 1 ] && [ "$BUILD_MXFP4" = 1 ]; then
  check prefill_mxfp4_mla "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1 -DPLOW_MLA_PREFILL=1" 256 2
  if [ "$BUILD_GQ" = 1 ]; then
    check prefill_mxfp4_mla_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1 -DPLOW_MLA_PREFILL=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" 256 2
  fi
fi

# The MoE arms are a SECOND full MFMA body in a bucket that already sits at the 256/occ-2 cliff,
# so the combination is checked, not assumed. Measured: unchanged at 256/0/occ 2/spill 2.
if [ "$BUILD_MOE" = 1 ]; then
  check prefill_mla_moe "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1" 256 2
  if [ "$BUILD_GQ" = 1 ]; then
    check prefill_mla_moe_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" 256 2
  fi
fi
if [ "$BUILD_MOE" = 1 ] && [ "$BUILD_MXFP4" = 1 ]; then
  check prefill_mla_moe_a4w4_full "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1 -DPLOW_MOE_PF_A4W4=1 -DPLOW_MXFP4=1 $A4W4_BK" 256 2
  if [ "$BUILD_GQ" = 1 ]; then
    check prefill_mla_moe_a4w4_full_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1 -DPLOW_MOE_PF_A4W4=1 -DPLOW_MXFP4=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF $A4W4_BK" 256 2
  fi
fi
if [ "$BUILD_FP8KV" = 1 ] && [ "$BUILD_MOE" = 1 ] && [ "$BUILD_MXFP4" = 1 ]; then
  check prefill_fp8kv_mla_moe_a4w4_full "$FP8KV_A4W4" 256 2
  if [ "$BUILD_GQ" = 1 ]; then
    check prefill_fp8kv_mla_moe_a4w4_full_gq "$FP8KV_A4W4 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" 256 2
  fi
fi

# Same argument for the Gemma-4 MoE halves: the decode family is wave-per-output GEMVs and the
# prefill family is a second MFMA body, and both go into buckets already at the cliff.
if [ "$BUILD_GEMMA_MOE" = 1 ]; then
  check decode_gmoe "$DEC $GMD" 256 2
  check prefill_gmoe "-DPLOW_BUCKET_DECODE=0 -DPLOW_MOE_GEMMA_PF=1" 256 2
  if [ "$BUILD_GQ" = 1 ]; then
    check decode_gmoe_gq "$DEC $GMD -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
    check prefill_gmoe_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_MOE_GEMMA_PF=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D $HIERPF" 256 2
  fi
fi

ALL_ELFS="interp_prefill.elf interp_decode.elf interp_flash.elf test_kernels.elf $K3_ELFS $DECODE_MLA_ELFS $KDA_FUSED_ELFS $KDA_INTRA_CACHED_ELFS $KDA_INTRA_WAVE_ITEMS_ELFS $ATTN_RES_F32MIX_ELFS $KDA_CARRY_REGSTATE_ELFS $KDA_WU_LEAN_ELFS $GEMM_GROUP_ELFS $KDA_CARRY_KEYFEED_ELFS $MLA_PF_DENSE_ELFS $KDA_KEY_FACTOR_ELFS $XR_ATTNRES_ELFS $MOE_STAGE1_ELFS $MOE_STAGE2_ELFS $MOE_COMBINE_ELFS $MOE_FP8_PF_ELFS $MOE_EP_ELFS $MOE_DECODE_GROUPED_ELFS $MLA_MATERIALIZED_ELFS $GQ_ELFS $FP8_ELFS $FP8KV_ELFS $MXFP4_ELFS $MLA_ELFS $MOE_ELFS $GMOE_ELFS"

# Every interpreter is compiled against the packed-prefill PlowProgram tail. This is an ABI
# marker, not a claim that descriptor-consuming math arms are enabled.
for e in $ALL_ELFS; do
  case "$e" in
    interp_*.elf)
      symbols=$("${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/llvm-nm "$e")
      grep -q plow_packed_prefill_abi_1 <<<"$symbols" || {
        echo "FAIL: $e does not advertise plow_packed_prefill_abi_1"; exit 1; }
      if grep -qE 'plow_packed_prefill_(mla|kda)_consumers_1' <<<"$symbols"; then
        echo "FAIL: default $e unexpectedly enables packed-prefill consumers"; exit 1
      fi
      ;;
  esac
done

for e in $DECODE_MLA_ELFS; do
  symbols=$("${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/llvm-nm "$e")
  grep -q plow_decode_mla_segment_object_1 <<<"$symbols" || {
    echo "FAIL: $e omits the decode MLA segment marker"; exit 1; }
  grep -q plow_packet_hash_lo <<<"$symbols" && grep -q plow_packet_hash_hi <<<"$symbols" || {
    echo "FAIL: $e omits the packet pairing stamp"; exit 1; }
done

for e in interp_flash.elf ${BUILD_GQ:+interp_flash_gq.elf}; do
  [ -f "$e" ] || continue
  symbols=$("${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/llvm-nm "$e")
  grep -q plow_mla_pf_v2_arm_1 <<<"$symbols" || {
    echo "FAIL: $e omits the gfx950 default MLA-prefill V2 arm"; exit 1; }
  if [ -n "$DSA_FLASH" ]; then
    grep -q plow_dsa_pf_arm <<<"$symbols" || {
      echo "FAIL: $e omits the packet-required gathered sparse-prefill arm"; exit 1; }
  fi
done

# Packed consumers are a separate opt-in object axis. Default production objects retain the
# ABI tail but no descriptor-consuming arms or resource cost.

# INSTRUCTION-SELECTION gate. The register check above catches a kernel that will not launch; it
# does NOT catch one that launches, is correct, and is silently 4x slow because the backend picked
# a narrow MFMA or widened an fp4 operand. With no GPU on the dev box that failure is otherwise
# invisible, so assert on the disassembly. Skipped when no expectations file is present.
EXPECT="$REPO/scripts/asm_expect_gfx950.json"
if [ -f "$EXPECT" ] && command -v python3 >/dev/null; then
  echo "   --- instruction-selection audit ---"
  if [ "${PLOW_GEMV_MFMA4:-0}" = 1 ]; then
    DEFAULT_ELFS=""
    MFMA4_ELFS=""
    for e in $ALL_ELFS; do
      case "$e" in
        interp_decode_mxfp4*.elf) MFMA4_ELFS="$MFMA4_ELFS $e" ;;
        *) DEFAULT_ELFS="$DEFAULT_ELFS $e" ;;
      esac
    done
    python3 "$REPO/scripts/asm_audit.py" --expect "$EXPECT" $DEFAULT_ELFS | tail -20
    if [ -n "$MFMA4_ELFS" ]; then
      python3 "$REPO/scripts/asm_audit.py" \
        --expect "$REPO/scripts/asm_expect_gfx950_mfma4.json" $MFMA4_ELFS | tail -20
    fi
  elif [ -n "${PLOW_HSACO_CONFIG:-}" ] &&
       grep -qx '#define PLOW_PACKET_HAS_INDEX_SCORE 1' "$PLOW_HSACO_CONFIG"; then
    DEFAULT_ELFS=""
    DSA_DECODE_ELFS=""
    QUARK_FP8KV_ELFS=""
    for e in $ALL_ELFS; do
      case "$e" in
        interp_decode_mxfp4*.elf) DSA_DECODE_ELFS="$DSA_DECODE_ELFS $e" ;;
        interp_decode_fp8kv*.elf|interp_flash_fp8kv*.elf|interp_prefill_fp8kv_mla_moe_a4w4_full*.elf)
          QUARK_FP8KV_ELFS="$QUARK_FP8KV_ELFS $e" ;;
        *) DEFAULT_ELFS="$DEFAULT_ELFS $e" ;;
      esac
    done
    python3 "$REPO/scripts/asm_audit.py" --expect "$EXPECT" $DEFAULT_ELFS | tail -20
    if [ -n "$DSA_DECODE_ELFS" ]; then
      python3 "$REPO/scripts/asm_audit.py" \
        --expect "$REPO/scripts/asm_expect_gfx950_dsa_decode_mxfp4.json" \
        $DSA_DECODE_ELFS | tail -20
    fi
    if [ -n "$QUARK_FP8KV_ELFS" ]; then
      python3 "$REPO/scripts/asm_audit.py" \
        --expect "$REPO/scripts/asm_expect_gfx950_quark_fp8kv.json" \
        $QUARK_FP8KV_ELFS | tail -20
    fi
  else
    python3 "$REPO/scripts/asm_audit.py" --expect "$EXPECT" $ALL_ELFS | tail -20
  fi
fi

# Freshness is the whole point of this script: print it, every time.
ls -l --time-style=+%H:%M:%S $ALL_ELFS chat \
  | awk '{print "   ", $NF, $5"B", $6}'

# Low-rung decode tiers (PLOW_DECODE_TIERS=<w,...>, as build_gfx942.sh): plowrt serves a decode
# rung t <= w from `lowrung<w>/interp_decode*.elf` beside the object dir, so a one-row rung does
# not run the widest object's GEMV body. Each tier is this script again at PLOW_DECODE_BATCH=w,
# one GEMV pass of w rows (PLOW_GEMV_MM=min(w,16)), from the same packet config — a tier stamps
# the packet hash and is refused under any other packet. Only its decode objects are kept.
if [ -n "${PLOW_DECODE_TIERS:-}" ] && [ -z "${PLOW_DECODE_TIER:-}" ]; then
  for w in ${PLOW_DECODE_TIERS//,/ }; do
    case "$w" in ''|*[!0-9]*) echo "PLOW_DECODE_TIERS must be a comma list of widths" >&2; exit 1;; esac
    if [ "$w" -ge "${PLOW_DECODE_BATCH:-1}" ]; then continue; fi  # the wide object serves it
    mm=$w; [ "$mm" -gt 16 ] && mm=16
    echo ""
    echo ">>> low-rung decode tier: width $w -> $OUT/lowrung$w"
    ( unset PLOW_DECODE_TIERS
      PLOW_DECODE_TIER="$w" PLOW_DECODE_BATCH="$w" PLOW_GEMV_MM="$mm" PLOW_GEMV_WALK="${PLOW_GEMV_WALK:-1}" \
        "$REPO/scripts/build_gfx950.sh" "$OUT/lowrung$w" ) || exit 1
    find "$OUT/lowrung$w" -maxdepth 1 -type f ! -name 'interp_decode*' -delete
  done
fi
