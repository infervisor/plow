#!/usr/bin/env bash
# build_sm90a_cubin.sh — the prebuilt interpreter module `plowrt serve` loads.
#
#   scripts/build_sm90a_cubin.sh <out.cubin>
#
# Compiles runtime/nvidia/interp_sm90a.cu to the two cubins `plowrt serve`
# loads:
#   1. the DECODE object <out.cubin>, committed recipe
#      (perf-data/gemma4-12b-plow-sm120-decode.md):
#        -arch=sm_90a -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_FA_GF_FULL=4
#        (208 regs on H100 → 1 block/SM; GQ sched)
#      symbol _Z12interp_sm90a11PlowProgram
#      GF_FULL=4 fuses the hd512 full-attention decode GQA group 4-wide, halving
#      KV re-reads vs the GF=2 default. CONFIRMED OPTIMAL ON H100 with H100 data
#      (runtime/nvidia/experiments/fa_gf_full_h100_ab.cu): beats GF=2 by 17-33%
#      and GF=8 by 15-39% at every context 8k..128k. GF=8 wins nowhere here.
#      Occupancy-neutral (megakernel REG stays 208 at GF 2/4/8 -> 1 block/SM);
#      the GF=4 arm itself is 104 regs with zero spills. relL2 1.68e-03.
#
#      RUNTIME COMPANION: set PLOW_NS_FULL_ABS=33 on H100 (NOT the RTX value 48).
#      nsplit must land the work on exactly one item per SM:
#          aligned = n_cu / gcd(n_grp, n_cu) = 132 / gcd(4,132) = 33
#      This is a cliff, not a slope: ns=33 -> 34 costs +67% at 128k (136 items on
#      132 SMs leaves 4 SMs running two, and d_flash_merge waits on them). The
#      inherited RTX value 48 puts 192 items on 132 SMs and costs +41% @128k /
#      +47% @32k; ns=24 is equally bad (+43%). ns=32 is an equally good tie.
#      For batching the aligned point scales as 33/n_batch (B=2 -> 16, B=4 -> 8),
#      since n_work = n_batch * n_grp * nsplit. Correctness holds at any nsplit;
#      only the speed depends on it.
#      NOTE: crates/plowc/src/bin/gemma4.rs already computes this rule
#      (n_cu/gcd(n_grp,n_cu)) but gates it on c.kvh_full >= 4, and this shape is
#      kvh_full = 2, so the rule never fires -- hence the explicit override.
#   2. the PREFILL object <out>_pf.cubin (same source, -DPLOW_NV_PREFILL=1):
#        the tiled-GEMM + FLASH_PREFILL megakernel (236 regs, 77.5 KiB smem),
#      symbol _Z15interp_sm90a_pf11PlowProgram. `exec::gpu` loads it as
#      <assets>/interp_sm90a_pf.cubin (PLOW_NV_CUBIN_PF overrides); absent, the
#      serve path falls back to decode-only prompt consumption.
# Both are verified to carry their kernel symbol.
#
# Runs with a clean environment: under `nix develop`, CPATH points nvcc's host
# pass at nix glibc headers that conflict with the CUDA math headers.
set -euo pipefail

OUT="${1:?usage: build_sm90a_cubin.sh <out.cubin>}"
# The prefill path is DERIVED from this one (`${OUT%.cubin}_pf.cubin`), so an
# argument without the suffix writes decode to `<x>` and prefill to
# `<x>_pf.cubin` — a pair that then gets copied into a bundle as
# `interp_sm90a` + `interp_sm90a.cubin`, i.e. the prefill object sitting under
# the decode object's name. Refuse the ambiguity at the source.
case "$OUT" in
  *.cubin) ;;
  *) echo "FATAL: <out> must end in .cubin (the prefill object is written to" >&2
     echo "       \${out%.cubin}_pf.cubin; '$OUT' would swap the pair)." >&2
     exit 1 ;;
esac
# PLOW_ROOT lets a worktree build its OWN modified source instead of /root/plow.
HERE="${PLOW_ROOT:-.}"
SRC="$HERE/runtime/nvidia/interp_sm90a.cu"
# PLOW_NVCC selects the toolchain. It MUST NOT be newer than what the installed
# driver accepts, or the cubin loads with CUDA_ERROR_INVALID_IMAGE at runtime
# (e.g. a CUDA 13 cubin on a 570.x / CUDA 12.8 driver). Check `nvidia-smi`.
NVCC="${PLOW_NVCC:-/usr/local/cuda/bin/nvcc}"
CUDA_BIN="$(dirname "$NVCC")"
# The clean environment every nvcc/cuobjdump call below runs under (see the
# CPATH note above). PLOW_NVCC_PATH replaces the PATH when the toolchain does
# not live in /usr (the nix sandbox has no /usr/bin — the host gcc must come
# from the caller); NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS pass through because
# nix's nvcc receives its -ccbin that way, and `env -i` would strip it.
NVENV=(env -i PATH="${PLOW_NVCC_PATH:-$CUDA_BIN:/usr/bin:/bin}")
[ -n "${NVCC_PREPEND_FLAGS:-}" ] && NVENV+=(NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS")
[ -n "${NVCC_APPEND_FLAGS:-}" ] && NVENV+=(NVCC_APPEND_FLAGS="$NVCC_APPEND_FLAGS")
KSYM=_Z12interp_sm90a11PlowProgram
OUT_PF="${OUT%.cubin}_pf.cubin"
KSYM_PF=_Z15interp_sm90a_pf11PlowProgram

if [ "${PLOW_BUILD_GEMV_CTA512_ROLE:-0}" = "1" ]; then
  "${NVENV[@]}" "$NVCC" -arch=sm_90a -O3 -cubin \
    -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
    -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF_FULL=4 \
    -DPLOW_NV_MLA=0 -DPLOW_NV_DSA=0 -DPLOW_NV_MAMBA=0 \
    -o "$OUT" "$HERE/runtime/nvidia/interp_sm90a_gemv512.cu"
  "${NVENV[@]}" cuobjdump -symbols "$OUT" | grep -q _Z20interp_sm90a_gemv51211PlowProgram
  exit 0
fi

# DEAD-ARM GATING (h100-interp arm-ablation). MLA (DeepSeek/GLM/Kimi latent attn), MAMBA (Nemotron
# SSD) and DSA (GLM sparse indexer) are op families a Gemma model NEVER emits. Compiling them OUT of
# these Gemma serving objects (the flags default ON in the source, so every OTHER build — the sm_120
# cubins and the op-test — stays byte-identical) shrinks the DECODE cubin ~43% (2.46->1.39 MB, REG
# 208->177, stack 1024->0 B) and the PREFILL stack frame (1744->672 B). It does NOT drop the prefill
# REG=255 ceiling — that is owned by the LIVE Hopper wgmma GEMM arms (d_gemm_sm90/d_gemm_w8a8_sm90),
# not by these dead arms — but it removes their contribution to size/stack. A gated-out opcode traps.
GEMMA_GATE="-DPLOW_NV_MLA=0 -DPLOW_NV_MAMBA=0 -DPLOW_NV_DSA=0"

# H100 DECODE ARM FIXES (perf-data/gemma26b-h100-gemv-mlp.md). Four independent defects, each
# found by opcode ablation rather than inspection. Every flag defaults to 0 in the sources, so
# every sm_120 object stays BYTE-IDENTICAL -- verified by sha256 on BOTH the decode and prefill
# cubins against a clean `git archive HEAD` build, re-checked after each round.
#
#   PLOW_NV_GEMV_RB          the fused norm+GLU expert arm issued SCALAR 2 B loads and
#                            recomputed the normalized x for each of 5632 output channels per
#                            layer (837 GB/s where the bytes support ~2200). Stages x once per
#                            CTA and gives each warp several weight streams. Also carries the
#                            warp-parallel router top-k, which had been running a 128-expert
#                            softmax + 1024 serial argmax iterations on THREAD 0 -- 1.026 ms,
#                            13% of the token, now ~0.
#   PLOW_MOE_DOWN_LANESPLIT  the MoE-down arms have a SHORT K (I_moe=704), so per-row cost
#                            amortises over almost nothing. Splits the warp into 8-lane
#                            sub-groups, one output row each: the rows-in-flight row-blocking
#                            wanted, but with one accumulator per lane and NO wv[][] array, so
#                            it costs no registers. Short-K only -- it loses on long-K arms.
#   PLOW_NV_FA_WPR           flash gave each THREAD a whole KV row, so a warp instruction
#                            scattered 32 requests D*2 bytes apart. Warp-per-row makes a D=256
#                            row one coalesced 512 B load: flash 1.079 -> 0.686 ms (2.53x at
#                            occ-2/fp8, with the non-flash remainder unchanged -- i.e. it moved
#                            only the arm it names).
#   PLOW_NV_FP8_RB=4         fp8 halves the weight bytes but not the x widening (~28
#                            instructions per 8 bytes vs bf16's 24 per 16), leaving the fp8
#                            dense GEMV compute-bound at 1046 GB/s. Row-blocking widens x once
#                            per chunk and reuses it across RB rows.
#
# MEASURED at 1 block/SM, ctx=1024, medians, gpu_lifecycle PASS on both precisions:
#   bf16 9.267 -> 6.196 ms (1.50x)   fp8 7.465 -> 5.330 ms (1.40x)
# NOTE the 4 here is the occ-1 optimum; at 2 blocks/SM the register cap is 128 and FP8_RB=2
# wins instead. That inversion is why these belong in the tuner (tuning/README-decode-tuner.md)
# rather than as one hand-set constant -- GV_MOE_UN and PLOW_NS_ABS both went stale mid-campaign
# exactly this way.
GEMV_RB="-DPLOW_NV_GEMV_RB=1 -DPLOW_MOE_DOWN_LANESPLIT=1 -DPLOW_NV_FA_WPR=1 -DPLOW_NV_FP8_RB=4"

# TUNER HOOK. scripts/tune_decode_sweep.sh appends knob overrides here
# (-DPLOW_NV_FORCE_MINBLK=2 -DGV_UNROLL=4 …) so the sweep builds the SHIPPED
# recipe plus one delta, instead of carrying a copy of it that silently rots
# away from this file. Empty by default, so every normal build — and every
# sm_120 build, which uses a different translation unit anyway — is unchanged.
EXTRA="${PLOW_EXTRA_DEFINES:-}"
# Decoder-only native FP8 M1 TMA experiment; requires AOT weight-map handles.
if [ "${PLOW_BUILD_FP8_M1_TMA:-0}" = "1" ]; then
  EXTRA="$EXTRA -DPLOW_NV_TMA_GEMM=1 -DPLOW_NV_FP8_M1=1 -DPLOW_NV_FP8_M1_FAST_ACCUM=1 -DPLOW_NV_FP8_M1_BK1024=1 -DPLOW_NV_FP8_M1_XCACHE=1 -DPLOW_NV_FP8_M1_TMA=1 -DPLOW_NV_QUANT_FP8_VLLM=1"
  export PLOW_BUILD_TMA_GEMM=1
fi


# PLOW_BUILD_TMA_GEMM=1: opt-in TMA + warp-specialized prefill GEMM (op_gemm_sm90.cuh,
# port of tma_ws_gemm_bf16.cu's ws_tma winner). Prefill object only — decode never
# dispatches the tiled GEMM. Requires packets emitted with tensormap gen-tensors
# (devgen PLOW_TMA_GEMM=1); without them every GEMM falls back to the cp.async body,
# so the flag alone is inert. OFF by default until it passes numeric gates on
# real Hopper hardware (the interp_sm90a.cu header's standing rule for wgmma paths).
PF_EXTRA=""
if [ "${PLOW_BUILD_TMA_GEMM:-0}" = "1" ]; then
  PF_EXTRA="-DPLOW_NV_TMA_GEMM=1"
fi

# PLOW_BUILD_W8A8=1: opt-in TRUE-fp8 (w8a8 QGMMA) prefill GEMM arms. On sm_90a the
# default fp8 prefill is w8a16 dequant into an EMULATED mma.sync.e4m3 (12x F2FP +
# 2x HMMA — zero fp8 tensor-core benefit; measured 1.6-1.7x SLOWER than bf16 prefill
# on GH200). PLOW_NV_W8A8 swaps the GEMM_*_FP8 opcodes to d_gemm_w8a8_sm90 (QGMMA,
# measured 1.7-1.9x FASTER than bf16 wgmma) + QUANT_FP8. PGM90_FP8_PROMOTE=1 rides
# along: fp8 wgmma does not accumulate in true f32 and the error grows with K
# (1.14e-3 @K=3840 eats ~70% of the 1.6e-3 budget); DeepGEMM two-level promotion
# every 128 k-elems cuts it ~10.9x (tma_ws_gemm_fp8.cu correctness table).
# Packets must be emitted with PLOW_W8A8=1 (QuantFp8 + the w8a8 t[] layout) —
# the opcodes change MEANING, so cubin and packet must pair.
# PLOW_BUILD_GEMV_HEAD=1 (T23/PX-6): compile the M=1 lm_head GEMV arm into the prefill
# objects; pair with packets emitted PLOW_PF_GEMV_HEAD=1 (else byte-identical).
if [ "${PLOW_BUILD_GEMV_HEAD:-0}" = "1" ]; then
  PF_EXTRA="$PF_EXTRA -DPLOW_NV_PF_GEMV_HEAD=1"
fi

if [ "${PLOW_BUILD_W8A8:-0}" = "1" ]; then
  # PLOW_W8A8_PROMOTE=0 opts out of two-level accumulation (A/B knob; accuracy drops
  # 1.04e-4 -> 1.14e-3 relL2 at K=3840, still under the 1.6e-3 oracle budget).
  PF_EXTRA="$PF_EXTRA -DPLOW_NV_W8A8=1 -DPGM90_FP8_PROMOTE=${PLOW_W8A8_PROMOTE:-1}"
fi

# PLOW_BUILD_SEG=1: ALSO build the segmented-prefill object pair (T9c/T10 design,
# first wired on Hopper by the gh200 prefill campaign):
#   <out>_pfseg.cubin   — the FAT segmented object (every prefill arm, occ-1); runs the
#                         flash/norm wave-class segments. Symbol *_pfseg.
#   <out>_pfgemm.cubin  — the LEAN GEMM segment object (PLOW_NV_SEG_GEMM=1: flash arms
#                         compiled OUT, launch_bounds(256,2) => 128 regs, TMA ring NS=2 so
#                         2 blocks/SM fit); runs the GEMM wave-class segments. *_pfgemm.
# plowrt launches the segments of one prefill program in order, alternating modules;
# packets must be emitted WITHOUT PLOW_UNISEG (and want PLOW_SEG_CLASS_SLICE=1 +
# PLOW_NO_GLU_FUSE=1 so GEMM segments slice to 2*n_cu and stay 64-acc).
if [ "${PLOW_BUILD_SEG:-0}" = "1" ]; then
  OUT_PFSEG="${OUT%.cubin}_pfseg.cubin"
  OUT_PFGEMM="${OUT%.cubin}_pfgemm.cubin"
  # PLOW_BUILD_FATLITE=1 (T14): build the fat object FATLITE — flash-prefill arms stripped,
  # 128-reg cap, occ-2. Requires PLOW_SEG_FA512=all packets (flash never classes to fat)
  # and wants PLOW_SEG_SLICE_ALL=1 so light filling ops slice to 2*n_cu.
  FATLITE_GATE=""
  if [ "${PLOW_BUILD_FATLITE:-0}" = "1" ]; then
    # STAGES=3 shrinks the TMA-GEMM arena claim to the occ-2 budget (99376 B), same as the
    # lean object — at the default 4 stages the claim alone (132160 B) pins occupancy to 1.
    FATLITE_GATE="-DPLOW_NV_FATLITE=1 -DPGM90_TMA_STAGES=3"
  fi
  "${NVENV[@]}" \
    "$NVCC" -arch=sm_90a -O3 -cubin \
    -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
    -DPLOW_NV_PREFILL=1 -DPLOW_NV_SEGMENTS=1 -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 $FATLITE_GATE \
    -DPLOW_NV_EMBED_SMEM=1 $GEMMA_GATE $GEMV_RB $EXTRA $PF_EXTRA \
    -o "$OUT_PFSEG" "$SRC"
  "${NVENV[@]}" \
    cuobjdump -symbols "$OUT_PFSEG" | grep -q "_pfseg" || { echo "FATAL: _pfseg symbol missing" >&2; exit 1; }
  echo "built $OUT_PFSEG ($(stat -c%s "$OUT_PFSEG") B)"
  # PLOW_BUILD_GEMM_ONLY=1 (T11): build the lean object PURE — every non-GEMM arm
  # stripped (PLOW_NV_GEMM_ONLY=1) so ptxas gives the wgmma bodies probe-grade register
  # allocation. Pair with packets emitted PLOW_SEG_PURE_GEMM=1 PLOW_NO_GLU_FUSE=1 and
  # serve-time PLOW_PF_SEG_PURE=1, or a light-op segment lands on this object and traps.
  GEMM_ONLY_GATE=""
  if [ "${PLOW_BUILD_GEMM_ONLY:-0}" = "1" ]; then
    GEMM_ONLY_GATE="-DPLOW_NV_GEMM_ONLY=1"
  elif [ "${PLOW_BUILD_SEG_NOGLU:-0}" = "1" ]; then
    # Middle rung: classic segmentation, fused-GLU arms stripped (dead under
    # PLOW_NO_GLU_FUSE packets) so the 128-reg lean object compiles spill-free.
    GEMM_ONLY_GATE="-DPLOW_NV_SEG_NOGLU=1"
  fi
  # PLOW_BUILD_SEG_WS=1: dispatch the warp-specialized setmaxnreg twin in the lean
  # object (PLOW_NV_SEG_WS — deadlocked in the wide TU; re-tested per TU shape).
  if [ "${PLOW_BUILD_SEG_WS:-0}" = "1" ]; then
    GEMM_ONLY_GATE="$GEMM_ONLY_GATE -DPLOW_NV_SEG_WS=1"
  fi
  # PLOW_BUILD_SEG_WS_ENTRY=1: ws twin with the ONCE-per-launch register split (needs a
  # pure-GEMM packet stream; see PLOW_NV_SEG_WS_ENTRY in op_gemm_sm90.cuh).
  if [ "${PLOW_BUILD_SEG_WS_ENTRY:-0}" = "1" ]; then
    GEMM_ONLY_GATE="$GEMM_ONLY_GATE -DPLOW_NV_SEG_WS=1 -DPLOW_NV_SEG_WS_ENTRY=1"
  fi
  # PLOW_BUILD_WS_BN256=1 (T13): BM64/BN256 m64n256k32 mainloop in the ws-entry object.
  if [ "${PLOW_BUILD_WS_BN256:-0}" = "1" ]; then
    GEMM_ONLY_GATE="$GEMM_ONLY_GATE -DPGM90_WS_BN256=1"
  fi
  # PLOW_BUILD_GEMM_UNI256=1 (T15): the lean object as the UNIFORM m128n256 occ-1 body
  # (both warpgroups math, 255-reg budget, deep TMA ring). Mutually exclusive with the
  # ws-entry knobs — it overrides them.
  if [ "${PLOW_BUILD_GEMM_UNI256:-0}" = "1" ]; then
    GEMM_ONLY_GATE="-DPLOW_NV_GEMM_ONLY=1 -DPGM90_UNI_BN256=1"
  fi
  # PLOW_BUILD_GEMM_OCC1=1 (T20): the bf16 lean object — arm-stripped, occ-1, 255-reg
  # budget (no 128-reg cap; the uniform bf16 TMA body spills there). For bf16 bundles.
  if [ "${PLOW_BUILD_GEMM_OCC1:-0}" = "1" ]; then
    GEMM_ONLY_GATE="-DPLOW_NV_GEMM_ONLY=1 -DPLOW_NV_SEG_OCC1=1"
  fi
  # PLOW_BUILD_GEMM_WS384=1 (T31): the lean object at 384 threads — dedicated TMA producer
  # warpgroup + two 224-reg m64n256 consumers (the cuBLAS shape). Loader reads the
  # plow_block_pfgemm global and launches accordingly.
  if [ "${PLOW_BUILD_GEMM_WS384:-0}" = "1" ]; then
    GEMM_ONLY_GATE="-DPLOW_NV_GEMM_ONLY=1 -DPGM90_UNI_BN256=1 -DPLOW_NV_SEG_WS384=1"
  fi
  if [ "${PLOW_BUILD_GEMM_SMALL_BF16:-0}" = "1" ]; then
    GEMM_ONLY_GATE="-DPLOW_NV_GEMM_ONLY=1 -DPLOW_NV_SEG_OCC1=1 -DPLOW_NV_TMA_GEMM=1 -DPLOW_NV_SEG_SMALL_BF16=1"
  fi
  if [ "${PLOW_BUILD_GEMM_M64N64:-0}" = "1" ]; then
    GEMM_ONLY_GATE="-DPLOW_NV_GEMM_ONLY=1 -DPLOW_NV_TMA_GEMM=1 -DPLOW_NV_SEG_M64N64=1"
  fi
  if [ "${PLOW_BUILD_GEMM_M64N128:-0}" = "1" ]; then
    GEMM_ONLY_GATE="-DPLOW_NV_GEMM_ONLY=1 -DPLOW_NV_TMA_GEMM=1 -DPLOW_NV_SEG_M64N128=1"
  fi
  "${NVENV[@]}" \
    "$NVCC" -arch=sm_90a -O3 -cubin \
    -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
    -DPLOW_NV_PREFILL=1 -DPLOW_NV_SEGMENTS=1 -DPLOW_NV_SEG_GEMM=1 -DPGM90_TMA_STAGES=3 \
    -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 $GEMM_ONLY_GATE \
    -DPLOW_NV_EMBED_SMEM=1 $GEMMA_GATE $GEMV_RB $EXTRA $PF_EXTRA \
    -o "$OUT_PFGEMM" "$SRC"
  "${NVENV[@]}" \
    cuobjdump -symbols "$OUT_PFGEMM" | grep -q "_pfgemm" || { echo "FATAL: _pfgemm symbol missing" >&2; exit 1; }
  echo "built $OUT_PFGEMM ($(stat -c%s "$OUT_PFGEMM") B)"

  # PLOW_BUILD_FA512=1 (T12): ALSO build the dedicated hd512 flash object <out>_pffa.cubin
  # (PLOW_NV_FA_ONLY=1: only the FLASH arms; hd256 compiled out — hd512 segments, class 2,
  # launch here). PLOW_BUILD_FA_WG=1 selects the <512,64,16> wgmma arm (re-test of the
  # in-megakernel refutation in a stripped TU); default px4 <512,32,16>.
  if [ "${PLOW_BUILD_FA512:-0}" = "1" ]; then
    OUT_PFFA="${OUT%.cubin}_pffa.cubin"
    FA_WG=""
    if [ "${PLOW_BUILD_FA_WG:-0}" = "1" ]; then
      FA_WG="-DPLOW_NV_FA512_WG=1"
    fi
    # PLOW_BUILD_FA_HD256=1: FA object also carries the hd256 sliding arm (class 'all').
    if [ "${PLOW_BUILD_FA_HD256:-0}" = "1" ]; then
      FA_WG="$FA_WG -DPLOW_NV_FA_ONLY_HD256=1"
    fi
    # PLOW_BUILD_FA_WGITEM=1 (T30): warpgroup-per-work-item hd256 flash (forces BKV=32).
    if [ "${PLOW_BUILD_FA_WGITEM:-0}" = "1" ]; then
      FA_WG="$FA_WG -DPLOW_NV_FA_WGITEM=1"
    fi
    # PLOW_BUILD_FA_ROPE=1 (T16): rope arm in the FA object (classing v2).
    if [ "${PLOW_BUILD_FA_ROPE:-0}" = "1" ]; then
      FA_WG="$FA_WG -DPLOW_NV_FA_ROPE=1"
    fi
    "${NVENV[@]}" \
      "$NVCC" -arch=sm_90a -O3 -cubin \
      -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
      -DPLOW_NV_PREFILL=1 -DPLOW_NV_SEGMENTS=1 -DPLOW_NV_FA_ONLY=1 $FA_WG \
      -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 \
      -DPLOW_NV_EMBED_SMEM=1 $GEMMA_GATE $GEMV_RB $EXTRA $PF_EXTRA \
      -o "$OUT_PFFA" "$SRC"
    "${NVENV[@]}" \
      cuobjdump -symbols "$OUT_PFFA" | grep -q "_pffa" || { echo "FATAL: _pffa symbol missing" >&2; exit 1; }
    echo "built $OUT_PFFA ($(stat -c%s "$OUT_PFFA") B)"
  fi
fi

"${NVENV[@]}" \
  "$NVCC" -arch=sm_90a -O3 -cubin \
  -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
  -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_FA_GF_FULL=4 -DPLOW_NV_EMBED_SMEM=1 $GEMMA_GATE $GEMV_RB $EXTRA \
  -o "$OUT" "$SRC"

if ! "${NVENV[@]}" \
    cuobjdump -symbols "$OUT" | grep -q "$KSYM"; then
  echo "FATAL: $KSYM not found in $OUT — kernel name/signature changed;" >&2
  echo "       update exec::gpu's kernel-name constant (or set PLOW_NV_KERNEL)." >&2
  exit 1
fi
echo "built $OUT ($(stat -c%s "$OUT") B), kernel $KSYM present"

"${NVENV[@]}" \
  "$NVCC" -arch=sm_90a -O3 -cubin \
  -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
  -DPLOW_NV_PREFILL=1 -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1 $GEMMA_GATE $GEMV_RB $EXTRA $PF_EXTRA \
  -o "$OUT_PF" "$SRC"

if ! "${NVENV[@]}" \
    cuobjdump -symbols "$OUT_PF" | grep -q "$KSYM_PF"; then
  echo "FATAL: $KSYM_PF not found in $OUT_PF — prefill kernel name/signature" >&2
  echo "       changed; update exec::gpu's KERNEL_PF constant (or PLOW_NV_KERNEL_PF)." >&2
  exit 1
fi
echo "built $OUT_PF ($(stat -c%s "$OUT_PF") B), kernel $KSYM_PF present"

# fp8-KV variants (rtx-19 E3, PLOW_BUILD_FP8KV=1): same two objects + -DPLOW_FP8_KV=1, which
# compiles in the e4m3 KV op-arms (HEADNORM_ROPE_FP8 / FLASH_DECODE_FP8). The default objects above
# stay byte-identical (fp8 arms are behind the flag). fp8-KV composes with fp8 WEIGHTS at runtime:
# the fp8 GEMV opcodes are already in the decode object, selected by the packet.
if [ "${PLOW_BUILD_FP8KV:-0}" = "1" ]; then
  OUT_KV="${OUT%.cubin}_fp8kv.cubin"
  OUT_PF_KV="${OUT%.cubin}_pf_fp8kv.cubin"
  "${NVENV[@]}" \
    "$NVCC" -arch=sm_90a -O3 -cubin \
    -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
    -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_FA_GF_FULL=4 -DPLOW_NV_EMBED_SMEM=1 -DPLOW_FP8_KV=1 $GEMMA_GATE $GEMV_RB $EXTRA \
    -o "$OUT_KV" "$SRC"
  "${NVENV[@]}" \
    cuobjdump -symbols "$OUT_KV" | grep -q "$KSYM" || { echo "FATAL: $KSYM missing in $OUT_KV" >&2; exit 1; }
  echo "built $OUT_KV ($(stat -c%s "$OUT_KV") B), kernel $KSYM present"
  # fp8 prefill dequants at the smem stage, so it needs the PIPE=0 synchronous-staging arm
  # (cp.async cannot convert fp8 inline). -DPLOW_NV_FA_PIPE=0 selects it; decode is PIPE-agnostic.
  #
  # PLOW_FP8_KV_FASTPF=1 opts back into the pipelined arm, matching the CMake
  # path (runtime/CMakeLists.txt, which measures the PIPE=0 arm at 5.4x on a
  # 127k context). Without an override here this script could only ever build
  # the slow arm, while CMake could build both.
  #
  # $PF_EXTRA belongs here as much as on the other four prefill objects: it
  # carries -DPLOW_NV_TMA_GEMM / -DPLOW_NV_W8A8, so omitting it made
  # PLOW_BUILD_TMA_GEMM=1 a silent no-op on exactly this object.
  FA_PIPE_KV="-DPLOW_NV_FA_PIPE=0"
  if [ "${PLOW_FP8_KV_FASTPF:-0}" = "1" ]; then
    FA_PIPE_KV="-DPLOW_NV_FA_PIPE=1"
  fi
  "${NVENV[@]}" \
    "$NVCC" -arch=sm_90a -O3 -cubin \
    -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
    -DPLOW_NV_PREFILL=1 -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1 -DPLOW_FP8_KV=1 $GEMMA_GATE $GEMV_RB $EXTRA $PF_EXTRA \
    $FA_PIPE_KV \
    -o "$OUT_PF_KV" "$SRC"
  "${NVENV[@]}" \
    cuobjdump -symbols "$OUT_PF_KV" | grep -q "$KSYM_PF" || { echo "FATAL: $KSYM_PF missing in $OUT_PF_KV" >&2; exit 1; }
  echo "built $OUT_PF_KV ($(stat -c%s "$OUT_PF_KV") B), kernel $KSYM_PF present"
fi
