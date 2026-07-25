#!/usr/bin/env bash
# Build the gfx950 (MI350X / CDNA4) interpreter for a HEAD_DIM=128 model (Llama-3.1 / Qwen3).
#
# Differs from build_gfx950.sh (Gemma, head_dim 256/512) in exactly two ways:
#   - the PREFILL interpreter is built with -DPLOW_FLASH_HD128 so the D=128 flash arm runs INLINE
#     on the 8-wave interpreter (D=128 fits the 256-reg / 2-wave budget; see the QCH=2 note in
#     op_attention.h). Gemma reaches flash through a separate 4-wave object; Qwen does not need one.
#   - NO interp_flash.elf is built. The segmented 4-wave flash object is a Gemma-only speedup; with
#     it absent gemma4_chat runs every segment on the 8-wave interpreter (correct, and for D=128 the
#     inline flash already runs there). Its thread count would also mismatch (chat auto-picks 512 for
#     D=128); leaving it out avoids that entirely.
#
# Same freshness/register-cliff guards as build_gfx950.sh: an over-budget interp is INVALID_ISA at
# load, caught here.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/rtb}}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUN="${PLOW_BUNDLER:-/opt/rocm-7.0.2/lib/llvm/bin/clang-offload-bundler}"
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"; cd "$OUT"

# Qwen GEMM tile: 192x256 beats the 256x256 default by ~1% end-to-end prefill and is register-legal
# (E1 tiling sweep; both tiles land at 128+128/occ2 inside the interpreter). Only prefill uses the GEMM.
# NB: GM_SLICE=32 (halving the MEM/MFMA barriers on this tile) is +7-9% STANDALONE on gate/up+down
# but NEUTRAL end-to-end (4k 182->181, 8k 491->490 ms; the serving regime eats it -- op_gemm.h SLICE
# comment). Left at the default 16; the knob exists for the occ-1 deep-pipeline work.
QTILE="-DGM_BM=192"
# GLOBAL QUEUE IS THE DEFAULT scheduler (E1: decode win on both models, prefill neutral). Build the _gq
# objects unless PLOW_NO_GQ=1 asks for a static-only build. PLOW_GQ_BATCH stays 1.
BUILD_GQ=1; [ -n "${PLOW_NO_GQ:-}" ] && BUILD_GQ=0
GQB="${PLOW_GQ_BATCH:-1}"
# FP8 DECODE object (PLOW_FP8=1). A SEPARATE decode interpreter that carries the fp8 w8a16 GEMV arms
# (GEMV_FP8 / GEMV_GLU_FP8) in place of the bf16 GEMV_GLU/QKV arms, so its register footprint stays
# under the same 256/occ-2 cliff. Built only when asked; the bf16 objects above are unaffected.
BUILD_FP8=0; [ "${PLOW_FP8:-0}" = 1 ] && BUILD_FP8=1
# FP8 KV-CACHE objects (PLOW_FP8_KV=1). fp8 e4m3 K/V storage+read halves the decode KV stream. The
# DECODE object swaps FLASH_DECODE for the fp8 flash + adds HeadNormRopeFp8 (and keeps the fp8 GEMV
# weight arms, since an fp8-KV pkt is also fp8-weight); the PREFILL object swaps FLASH_PREFILL for
# the fp8 flash (bf16 GEMM weights untouched) + adds HeadNormRopeFp8. Built only when asked. No
# FA_DEC_VPIPE here: that V-prefetch is a bf16-only path (op_attention.h) the fp8 flash does not use.
BUILD_FP8KV=0; [ "${PLOW_FP8_KV:-0}" = 1 ] && BUILD_FP8KV=1

# interp_flash.elf is DELETED too though we never build it: a stale one from a Gemma build in a
# reused $OUT gets loaded by the harness and launched at the wrong width -> HSA_STATUS_ERROR_INVALID_ISA
# (the "stale artifact lied" footgun the build headers warn about). Qwen runs flash inline (HD128).
rm -f i_prefill.co i_decode.co tk.co interp_prefill.elf interp_decode.elf test_kernels.elf interp_flash.elf \
      i_prefill_gq.co i_decode_gq.co interp_prefill_gq.elf interp_decode_gq.elf \
      i_decode_fp8.co i_decode_fp8_gq.co interp_decode_fp8.elf interp_decode_fp8_gq.elf \
      i_prefill_fp8kv.co i_decode_fp8kv.co interp_prefill_fp8kv.elf interp_decode_fp8kv.elf \
      i_prefill_fp8kv_gq.co i_decode_fp8kv_gq.co interp_prefill_fp8kv_gq.elf interp_decode_fp8kv_gq.elf

genco()    { hipcc --offload-arch="$ARCH" -O3 -w $1 --genco "$R/amd/interp.hip" -o "$2" $INC; }
unbundle() { "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --input="$1" --output="$2"; }

genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_FLASH_HD128 $QTILE" i_prefill.co
unbundle i_prefill.co interp_prefill.elf
# FA_DEC_VPIPE=8: prefetch d_flash_decode's first V-group over the softmax barriers (op_attention.h).
# A win only on the D=128/GF=2 decode arm; the depth is allocator-fragile (4 spills to AGPR and
# regresses), 8 lands cleanly at 237 VGPR/occ2. Qwen-only; Gemma's build leaves it at the 0 default.
genco "-DPLOW_BUCKET_DECODE=1 -DFA_DEC_VPIPE=8" i_decode.co
unbundle i_decode.co interp_decode.elf

# GLOBAL-QUEUE objects (the DEFAULT scheduler). D=128 flash runs INLINE in the 8-wave prefill_gq
# object, so Qwen needs no separate flash_gq object — one shared cursor covers every op.
if [ "$BUILD_GQ" = 1 ]; then
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_FLASH_HD128 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $QTILE" i_prefill_gq.co
  unbundle i_prefill_gq.co interp_prefill_gq.elf
  genco "-DPLOW_BUCKET_DECODE=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB -DFA_DEC_VPIPE=8" i_decode_gq.co
  unbundle i_decode_gq.co interp_decode_gq.elf
fi

# FP8 decode objects (static + GQ), each swapping the bf16 GEMV_GLU/QKV arms for the fp8 ones.
if [ "$BUILD_FP8" = 1 ]; then
  genco "-DPLOW_BUCKET_DECODE=1 -DPLOW_FP8=1 -DFA_DEC_VPIPE=8" i_decode_fp8.co
  unbundle i_decode_fp8.co interp_decode_fp8.elf
  if [ "$BUILD_GQ" = 1 ]; then
    genco "-DPLOW_BUCKET_DECODE=1 -DPLOW_FP8=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB -DFA_DEC_VPIPE=8" i_decode_fp8_gq.co
    unbundle i_decode_fp8_gq.co interp_decode_fp8_gq.elf
  fi
fi

# FP8 KV objects (prefill + decode, static + GQ). The prefill object dequants the e4m3 cache at the
# LDS stage (bf16 GEMM weights unchanged); the decode object runs the fp8 flash + fp8 GEMV arms.
if [ "$BUILD_FP8KV" = 1 ]; then
  genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_FLASH_HD128 -DPLOW_FP8_KV=1 $QTILE" i_prefill_fp8kv.co
  unbundle i_prefill_fp8kv.co interp_prefill_fp8kv.elf
  genco "-DPLOW_BUCKET_DECODE=1 -DPLOW_FP8=1 -DPLOW_FP8_KV=1 -DFA_DEC_VPIPE=8" i_decode_fp8kv.co
  unbundle i_decode_fp8kv.co interp_decode_fp8kv.elf
  if [ "$BUILD_GQ" = 1 ]; then
    genco "-DPLOW_BUCKET_DECODE=0 -DPLOW_FLASH_HD128 -DPLOW_FP8_KV=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $QTILE" i_prefill_fp8kv_gq.co
    unbundle i_prefill_fp8kv_gq.co interp_prefill_fp8kv_gq.elf
    genco "-DPLOW_BUCKET_DECODE=1 -DPLOW_FP8=1 -DPLOW_FP8_KV=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB -DFA_DEC_VPIPE=8" i_decode_fp8kv_gq.co
    unbundle i_decode_fp8kv_gq.co interp_decode_fp8kv_gq.elf
  fi
fi

hipcc --offload-arch="$ARCH" -O3 -w --genco "$R/amd/test_kernels.hip" -o tk.co $INC
unbundle tk.co test_kernels.elf

/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o chat \
    "$R/tests/gemma4_chat.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d chat | grep -qi runpath && { echo "FAIL: RUNPATH leaked into the host binary"; exit 1; }

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
check prefill "-DPLOW_BUCKET_DECODE=0 -DPLOW_FLASH_HD128 $QTILE" 256 2
# decode checks MUST carry FA_DEC_VPIPE=8 to match the built elf — the V-prefetch costs +58 VGPR
# (237 total), so checking without it would validate the wrong, smaller config and miss a cliff trip.
check decode  "-DPLOW_BUCKET_DECODE=1 -DFA_DEC_VPIPE=8"                          256 2
GQ_ELFS=""
if [ "$BUILD_GQ" = 1 ]; then
  check prefill_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_FLASH_HD128 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $QTILE" 256 2
  check decode_gq  "-DPLOW_BUCKET_DECODE=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB -DFA_DEC_VPIPE=8" 256 2
  GQ_ELFS="interp_prefill_gq.elf interp_decode_gq.elf"
fi
FP8_ELFS=""
if [ "$BUILD_FP8" = 1 ]; then
  check decode_fp8 "-DPLOW_BUCKET_DECODE=1 -DPLOW_FP8=1 -DFA_DEC_VPIPE=8" 256 2
  FP8_ELFS="interp_decode_fp8.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    check decode_fp8_gq "-DPLOW_BUCKET_DECODE=1 -DPLOW_FP8=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB -DFA_DEC_VPIPE=8" 256 2
    FP8_ELFS="interp_decode_fp8.elf interp_decode_fp8_gq.elf"
  fi
fi
FP8KV_ELFS=""
if [ "$BUILD_FP8KV" = 1 ]; then
  check prefill_fp8kv "-DPLOW_BUCKET_DECODE=0 -DPLOW_FLASH_HD128 -DPLOW_FP8_KV=1 $QTILE" 256 2
  check decode_fp8kv  "-DPLOW_BUCKET_DECODE=1 -DPLOW_FP8=1 -DPLOW_FP8_KV=1 -DFA_DEC_VPIPE=8" 256 2
  FP8KV_ELFS="interp_prefill_fp8kv.elf interp_decode_fp8kv.elf"
  if [ "$BUILD_GQ" = 1 ]; then
    check prefill_fp8kv_gq "-DPLOW_BUCKET_DECODE=0 -DPLOW_FLASH_HD128 -DPLOW_FP8_KV=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB $QTILE" 256 2
    check decode_fp8kv_gq  "-DPLOW_BUCKET_DECODE=1 -DPLOW_FP8=1 -DPLOW_FP8_KV=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=$GQB -DFA_DEC_VPIPE=8" 256 2
    FP8KV_ELFS="interp_prefill_fp8kv.elf interp_decode_fp8kv.elf interp_prefill_fp8kv_gq.elf interp_decode_fp8kv_gq.elf"
  fi
fi

ls -l --time-style=+%H:%M:%S \
  interp_prefill.elf interp_decode.elf test_kernels.elf chat $GQ_ELFS $FP8_ELFS $FP8KV_ELFS \
  | awk '{print "   ", $NF, $5"B", $6}'
