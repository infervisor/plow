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
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/rtb}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
# Discover the bundler from the INSTALLED ROCm instead of pinning a version. This
# was pinned to 7.0.2 and the path does not exist on a 7.2.4 box, so the build died
# on the first machine that had the GPU. $PLOW_BUNDLER still overrides.
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
INC="-I$R/amd -I$R/common"
if [ -n "${PLOW_HSACO_CONFIG:-}" ]; then
  [ -f "$PLOW_HSACO_CONFIG" ] || { echo "missing PLOW_HSACO_CONFIG: $PLOW_HSACO_CONFIG" >&2; exit 2; }
  cfg_dir="$(dirname -- "$PLOW_HSACO_CONFIG")"
  cfg_name="$(basename -- "$PLOW_HSACO_CONFIG")"
  INC="$INC -I$cfg_dir -DPLOW_CONFIG=\"$cfg_name\""
fi
decode_inventory_prune="${PLOW_HSACO_DECODE_INVENTORY_PRUNE:-auto}"
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
mkdir -p "$OUT"; cd "$OUT"

# Delete FIRST. A build that fails must leave nothing behind to run.
rm -f i_prefill.co i_decode.co i_flash.co tk.co \
      interp_prefill.elf interp_decode.elf interp_flash.elf test_kernels.elf \
      i_prefill_gq.co i_decode_gq.co i_flash_gq.co \
      interp_prefill_gq.elf interp_decode_gq.elf interp_flash_gq.elf \
      i_decode_mla.co i_decode_mla_gq.co \
      interp_decode_mla.elf interp_decode_mla_gq.elf \
      i_decode_fp8.co i_decode_fp8_gq.co interp_decode_fp8.elf interp_decode_fp8_gq.elf \
      i_decode_fp8kv.co i_decode_fp8kv_gq.co interp_decode_fp8kv.elf interp_decode_fp8kv_gq.elf \
      i_prefill_mla_moe.co i_prefill_mla_moe_gq.co \
      interp_prefill_mla_moe.elf interp_prefill_mla_moe_gq.elf \
      kda_decode_fused_gfx950.co kda_decode_fused_gfx950.elf \
      kda_chunk_intra_cached_gfx950.co kda_chunk_intra_cached_gfx950.elf \
      kda_chunk_intra_wave_items_gfx950.co kda_chunk_intra_wave_items_gfx950.elf \
      kda_chunk_key_factor_wu_gfx950.co kda_chunk_key_factor_wu_gfx950.elf \
      kda_chunk_key_factor_carry_gfx950.co kda_chunk_key_factor_carry_gfx950.elf \
      xreduce_attnres_gfx950.co xreduce_attnres_gfx950.elf \
      moe_stage1_mxfp4_gfx950.co moe_stage1_mxfp4_gfx950.elf \
      moe_stage2_mxfp4_gfx950.co moe_stage2_mxfp4_gfx950.elf \
      moe_combine_gfx950.co moe_combine_gfx950.elf \
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
  KDA_FUSED_ELFS="kda_decode_fused_gfx950.elf"
fi

MOE_STAGE2_ELFS=""
if [ "$ARCH" = gfx950 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/moe_stage2_mxfp4_gfx950.elf" plow_moe2_mxfp4_16x16x128_gfx950 100 2 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe2_mxfp4_stage2_abi_3 \
    "$R/bench/amd/lean_moe_stage2_ref/native_kernel.hip"
  MOE_STAGE2_ELFS="moe_stage2_mxfp4_gfx950.elf"
fi

MOE_STAGE1_ELFS=""
if [ "$ARCH" = gfx950 ]; then
  bash "$R/cmake/hipcc_hsaco.sh" hipcc "$BUN" "$ARCH" \
    "$OUT/moe_stage1_mxfp4_gfx950.elf" plow_moe1_mxfp4_bk256_gfx950 192 2 \
    $INC -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe1_mxfp4_stage1_abi_1 \
    "$R/bench/amd/lean_moe_stage1_ref/native_kernel.hip"
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
DEC="-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=$GVMM -DPLOW_GEMV_WALK=$WALK"
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
genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1" i_flash.co
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
HIER_STATUS="off"
if [ "${PLOW_L2_PLACE:-1}" = 1 ]; then
  L2D="-DPLOW_L2_PLACE_DISPATCH"
  if [ "${PLOW_GATE_HIER:-1}" = 1 ]; then HIERD="-DPLOW_GATE_HIER=1"; HIER_STATUS="on"; fi
  echo "   per-XCD packet queues: prefill+decode L2 dispatch; decode hierarchy=$HIER_STATUS"
fi
if [ "$BUILD_GQ" = 1 ]; then
  for B in 0 1; do
    N=$([ "$B" -eq 0 ] && echo prefill || echo decode)
    D=$([ "$B" -eq 0 ] && echo "-DPLOW_BUCKET_DECODE=0 $L2D" || echo "$DEC $L2D $HIERD")
    genco "$D -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" "i_${N}_gq.co"
    unbundle "i_${N}_gq.co" "interp_${N}_gq.elf"
  done
  # The L2 decode object is register-checked below with the same 256/occ-2 budget as the plain
  # one — it is the SAME object with one extra scalar read, and if that ever costs occupancy the
  # build must fail rather than ship a decode kernel at occ 1.
  # GQ flash object (Gemma's segmented 4-wave flash segment under the global queue).
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $L2D" i_flash_gq.co
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
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_FP8=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_prefill_fp8_gq.co
    unbundle i_prefill_fp8_gq.co interp_prefill_fp8_gq.elf
  fi
fi

# FP8 KV-CACHE decode objects (static + GQ). Swap FLASH_DECODE for the fp8 flash + add HeadNormRopeFp8,
# keeping the fp8 GEMV weight arms. bf16/fp8-weight objects above are unaffected.
if [ "$BUILD_FP8KV" = 1 ]; then
  genco "$DEC -DPLOW_FP8=1 -DPLOW_FP8_KV=1" i_decode_fp8kv.co
  unbundle i_decode_fp8kv.co interp_decode_fp8kv.elf
  if [ "$BUILD_GQ" = 1 ]; then
    genco "$DEC -DPLOW_FP8=1 -DPLOW_FP8_KV=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_decode_fp8kv_gq.co
    unbundle i_decode_fp8kv_gq.co interp_decode_fp8kv_gq.elf
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
  genco "$DEC -DPLOW_MXFP4=1" i_decode_mxfp4.co
  unbundle i_decode_mxfp4.co interp_decode_mxfp4.elf
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1" i_prefill_mxfp4.co
  unbundle i_prefill_mxfp4.co interp_prefill_mxfp4.elf
  MXFP4_ELFS="interp_decode_mxfp4.elf interp_prefill_mxfp4.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    genco "$DEC -DPLOW_MXFP4=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_decode_mxfp4_gq.co
    unbundle i_decode_mxfp4_gq.co interp_decode_mxfp4_gq.elf
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_prefill_mxfp4_gq.co
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
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_prefill_mla_gq.co
    unbundle i_prefill_mla_gq.co interp_prefill_mla_gq.elf
    MLA_ELFS="interp_prefill_mla.elf interp_prefill_mla_gq.elf"
  fi
fi

# MLA+MoE prefill objects (static + GQ) — the whole-layer prefill object.
MOE_ELFS=""
if [ "$BUILD_MOE" = 1 ]; then
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1" i_prefill_mla_moe.co
  unbundle i_prefill_mla_moe.co interp_prefill_mla_moe.elf
  MOE_ELFS="interp_prefill_mla_moe.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_prefill_mla_moe_gq.co
    unbundle i_prefill_mla_moe_gq.co interp_prefill_mla_moe_gq.elf
    MOE_ELFS="interp_prefill_mla_moe.elf interp_prefill_mla_moe_gq.elf"
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
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_MOE_GEMMA_PF=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" i_prefill_gmoe_gq.co
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
check flash   "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1" 512 1
if [ "$need_decode_mla" = 1 ]; then
  check decode_mla "$DEC -DPLOW_BUCKET_DECODE_MLA=1" 256 2
fi
# The GQ loop adds a shared cursor + claim broadcast; it must NOT push prefill past 256/occ-2. If it
# does and no minimal-live-set fix recovers it, that is itself a recordable E1 finding (GQ incompatible
# with occ-2 prefill). These run only when the GQ objects were built.
GQ_ELFS=""
if [ "$BUILD_GQ" = 1 ]; then
  check prefill_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
  check decode_gq  "$DEC $L2D -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
  check flash_gq   "-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 512 1
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
    check prefill_fp8_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_FP8=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
    FP8_ELFS="interp_decode_fp8.elf interp_decode_fp8_gq.elf interp_prefill_fp8.elf interp_prefill_fp8_gq.elf"
  fi
fi
FP8KV_ELFS=""
if [ "$BUILD_FP8KV" = 1 ]; then
  check decode_fp8kv "$DEC -DPLOW_FP8=1 -DPLOW_FP8_KV=1" 256 2
  FP8KV_ELFS="interp_decode_fp8kv.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    check decode_fp8kv_gq "$DEC -DPLOW_FP8=1 -DPLOW_FP8_KV=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
    FP8KV_ELFS="interp_decode_fp8kv.elf interp_decode_fp8kv_gq.elf"
  fi
fi

if [ "$BUILD_MXFP4" = 1 ]; then
  # Conflict resolution: DECODE arms must use $DEC, which carries -DPLOW_GEMV_MM=$GVMM.
  # Dropping it here would silently revert batched decode to one row for the mxfp4 object
  # only — exactly the class of bug that made PLOW_GEMV_MM unreachable in the first place.
  # The prefill_mxfp4 checks are new and are kept as written.
  check decode_mxfp4 "$DEC -DPLOW_MXFP4=1" 256 2
  # The prefill twin carries the fp4 GEMM at all five tile rungs. Its accumulators are the
  # bf16 ones (w4a16 dequantizes in the B-fetch and issues a bf16 MFMA), and every rung is
  # smaller than the 256x256 arm the object already had, so it must stay at the same cliff.
  check prefill_mxfp4 "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1" 256 2
  if [ "$BUILD_GQ" = 1 ]; then
    check decode_mxfp4_gq "$DEC -DPLOW_MXFP4=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
    check prefill_mxfp4_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_MXFP4=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
  fi
fi

if [ "$BUILD_MLA" = 1 ]; then
  check prefill_mla "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1" 256 2
  # `if`, not `[ ... ] && check`: under `set -e` a false `[ ]` as the LAST command of this block
  # is the block's status and kills the script right here — silently, after the objects are
  # already built. Only reachable with PLOW_NO_GQ=1, which is exactly the rare path nobody runs.
  if [ "$BUILD_GQ" = 1 ]; then
    check prefill_mla_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
  fi
fi

# The MoE arms are a SECOND full MFMA body in a bucket that already sits at the 256/occ-2 cliff,
# so the combination is checked, not assumed. Measured: unchanged at 256/0/occ 2/spill 2.
if [ "$BUILD_MOE" = 1 ]; then
  check prefill_mla_moe "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1" 256 2
  if [ "$BUILD_GQ" = 1 ]; then
    check prefill_mla_moe_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
  fi
fi

# Same argument for the Gemma-4 MoE halves: the decode family is wave-per-output GEMVs and the
# prefill family is a second MFMA body, and both go into buckets already at the cliff.
if [ "$BUILD_GEMMA_MOE" = 1 ]; then
  check decode_gmoe "$DEC $GMD" 256 2
  check prefill_gmoe "-DPLOW_BUCKET_DECODE=0 -DPLOW_MOE_GEMMA_PF=1" 256 2
  if [ "$BUILD_GQ" = 1 ]; then
    check decode_gmoe_gq "$DEC $GMD -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
    check prefill_gmoe_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_MOE_GEMMA_PF=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB" 256 2
  fi
fi

ALL_ELFS="interp_prefill.elf interp_decode.elf interp_flash.elf test_kernels.elf $DECODE_MLA_ELFS $KDA_FUSED_ELFS $KDA_INTRA_CACHED_ELFS $KDA_INTRA_WAVE_ITEMS_ELFS $KDA_KEY_FACTOR_ELFS $XR_ATTNRES_ELFS $MOE_STAGE1_ELFS $MOE_STAGE2_ELFS $MOE_COMBINE_ELFS $MLA_MATERIALIZED_ELFS $GQ_ELFS $FP8_ELFS $FP8KV_ELFS $MXFP4_ELFS $MLA_ELFS $MOE_ELFS $GMOE_ELFS"

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
  python3 "$REPO/scripts/asm_audit.py" --expect "$EXPECT" $ALL_ELFS | tail -20
fi

# Freshness is the whole point of this script: print it, every time.
ls -l --time-style=+%H:%M:%S $ALL_ELFS chat \
  | awk '{print "   ", $NF, $5"B", $6}'
