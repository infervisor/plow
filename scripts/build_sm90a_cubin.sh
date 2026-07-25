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
# PLOW_ROOT lets a worktree build its OWN modified source instead of /root/plow.
HERE="${PLOW_ROOT:-/root/plow}"
SRC="$HERE/runtime/nvidia/interp_sm90a.cu"
# PLOW_NVCC selects the toolchain. It MUST NOT be newer than what the installed
# driver accepts, or the cubin loads with CUDA_ERROR_INVALID_IMAGE at runtime
# (e.g. a CUDA 13 cubin on a 570.x / CUDA 12.8 driver). Check `nvidia-smi`.
NVCC="${PLOW_NVCC:-/usr/local/cuda/bin/nvcc}"
CUDA_BIN="$(dirname "$NVCC")"
KSYM=_Z12interp_sm90a11PlowProgram
OUT_PF="${OUT%.cubin}_pf.cubin"
KSYM_PF=_Z15interp_sm90a_pf11PlowProgram

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

env -i PATH="$CUDA_BIN":/usr/bin:/bin \
  "$NVCC" -arch=sm_90a -O3 -cubin \
  -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
  -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_FA_GF_FULL=4 -DPLOW_NV_EMBED_SMEM=1 $GEMMA_GATE $GEMV_RB $EXTRA \
  -o "$OUT" "$SRC"

if ! env -i PATH="$CUDA_BIN":/usr/bin:/bin \
    cuobjdump -symbols "$OUT" | grep -q "$KSYM"; then
  echo "FATAL: $KSYM not found in $OUT — kernel name/signature changed;" >&2
  echo "       update exec::gpu's kernel-name constant (or set PLOW_NV_KERNEL)." >&2
  exit 1
fi
echo "built $OUT ($(stat -c%s "$OUT") B), kernel $KSYM present"

env -i PATH="$CUDA_BIN":/usr/bin:/bin \
  "$NVCC" -arch=sm_90a -O3 -cubin \
  -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
  -DPLOW_NV_PREFILL=1 -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1 $GEMMA_GATE $GEMV_RB $EXTRA \
  -o "$OUT_PF" "$SRC"

if ! env -i PATH="$CUDA_BIN":/usr/bin:/bin \
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
  env -i PATH="$CUDA_BIN":/usr/bin:/bin \
    "$NVCC" -arch=sm_90a -O3 -cubin \
    -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
    -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_FA_GF_FULL=4 -DPLOW_NV_EMBED_SMEM=1 -DPLOW_FP8_KV=1 $GEMMA_GATE $GEMV_RB $EXTRA \
    -o "$OUT_KV" "$SRC"
  env -i PATH="$CUDA_BIN":/usr/bin:/bin \
    cuobjdump -symbols "$OUT_KV" | grep -q "$KSYM" || { echo "FATAL: $KSYM missing in $OUT_KV" >&2; exit 1; }
  echo "built $OUT_KV ($(stat -c%s "$OUT_KV") B), kernel $KSYM present"
  # fp8 prefill dequants at the smem stage, so it needs the PIPE=0 synchronous-staging arm
  # (cp.async cannot convert fp8 inline). -DPLOW_NV_FA_PIPE=0 selects it; decode is PIPE-agnostic.
  env -i PATH="$CUDA_BIN":/usr/bin:/bin \
    "$NVCC" -arch=sm_90a -O3 -cubin \
    -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
    -DPLOW_NV_PREFILL=1 -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1 -DPLOW_FP8_KV=1 $GEMMA_GATE $GEMV_RB $EXTRA \
    -DPLOW_NV_FA_PIPE=0 \
    -o "$OUT_PF_KV" "$SRC"
  env -i PATH="$CUDA_BIN":/usr/bin:/bin \
    cuobjdump -symbols "$OUT_PF_KV" | grep -q "$KSYM_PF" || { echo "FATAL: $KSYM_PF missing in $OUT_PF_KV" >&2; exit 1; }
  echo "built $OUT_PF_KV ($(stat -c%s "$OUT_PF_KV") B), kernel $KSYM_PF present"
fi
