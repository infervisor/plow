#!/usr/bin/env bash
# build_sm120_cubin.sh — the prebuilt interpreter module `plowrt serve` loads.
#
#   scripts/build_sm120_cubin.sh <out.cubin>
#
# Compiles runtime/nvidia/interp_sm120.cu to the two cubins `plowrt serve`
# loads:
#   1. the DECODE object <out.cubin>, committed recipe
#      (perf-data/gemma4-12b-plow-sm120-decode.md):
#        -arch=sm_120a -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2   (150 regs, GQ sched)
#      symbol _Z12interp_sm12011PlowProgram
#   2. the PREFILL object <out>_pf.cubin (same source, -DPLOW_NV_PREFILL=1):
#        the tiled-GEMM + FLASH_PREFILL megakernel (236 regs, 77.5 KiB smem),
#      symbol _Z15interp_sm120_pf11PlowProgram. `exec::gpu` loads it as
#      <assets>/interp_sm120_pf.cubin (PLOW_NV_CUBIN_PF overrides); absent, the
#      serve path falls back to decode-only prompt consumption.
# Both are verified to carry their kernel symbol.
#
# Runs with a clean environment: under `nix develop`, CPATH points nvcc's host
# pass at nix glibc headers that conflict with the CUDA math headers.
set -euo pipefail

OUT="${1:?usage: build_sm120_cubin.sh <out.cubin>}"
# PLOW_ROOT lets a worktree build its OWN modified source instead of /root/plow.
HERE="${PLOW_ROOT:-/root/plow}"
SRC="$HERE/runtime/nvidia/interp_sm120.cu"
NVCC=/usr/local/cuda/bin/nvcc
KSYM=_Z12interp_sm12011PlowProgram
OUT_PF="${OUT%.cubin}_pf.cubin"
KSYM_PF=_Z15interp_sm120_pf11PlowProgram

env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  "$NVCC" -arch=sm_120a -O3 -cubin \
  -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
  -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_FA_GF_FULL=4 -DPLOW_NV_EMBED_SMEM=1 \
  -o "$OUT" "$SRC"

if ! env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
    cuobjdump -symbols "$OUT" | grep -q "$KSYM"; then
  echo "FATAL: $KSYM not found in $OUT — kernel name/signature changed;" >&2
  echo "       update exec::gpu's kernel-name constant (or set PLOW_NV_KERNEL)." >&2
  exit 1
fi
echo "built $OUT ($(stat -c%s "$OUT") B), kernel $KSYM present"

env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  "$NVCC" -arch=sm_120a -O3 -cubin \
  -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
  -DPLOW_NV_PREFILL=1 -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1 \
  -o "$OUT_PF" "$SRC"

if ! env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
    cuobjdump -symbols "$OUT_PF" | grep -q "$KSYM_PF"; then
  echo "FATAL: $KSYM_PF not found in $OUT_PF — prefill kernel name/signature" >&2
  echo "       changed; update exec::gpu's KERNEL_PF constant (or PLOW_NV_KERNEL_PF)." >&2
  exit 1
fi
echo "built $OUT_PF ($(stat -c%s "$OUT_PF") B), kernel $KSYM_PF present"

# Device stochastic sampler (plan stage 4). A standalone object (not the
# megakernel) the engine loads next to the decode cubin when PLOW_DEV_SAMPLE=1
# and launches after the decode kernel to write each stochastic row's token
# into in.ids. `<out>` dir gets sample_sm120.cubin. Independent of the two
# objects above; a build failure here is non-fatal to serving (host sampling).
SRC_SMP="$HERE/runtime/nvidia/sample_sm120.cu"
OUT_SMP="$(dirname "$OUT")/sample_sm120.cubin"
if [ -f "$SRC_SMP" ]; then
  if env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
      "$NVCC" -arch=sm_120a -O3 -cubin -o "$OUT_SMP" "$SRC_SMP" 2>/dev/null &&
     env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
      cuobjdump -symbols "$OUT_SMP" | grep -q "plow_sample"; then
    echo "built $OUT_SMP ($(stat -c%s "$OUT_SMP") B), kernel plow_sample present"
  else
    echo "WARN: sampler cubin build failed (device sampling will fall back to host)" >&2
  fi
fi

# fp8-KV variants (rtx-19 E3, PLOW_BUILD_FP8KV=1): same two objects + -DPLOW_FP8_KV=1, which
# compiles in the e4m3 KV op-arms (HEADNORM_ROPE_FP8 / FLASH_DECODE_FP8). The default objects above
# stay byte-identical (fp8 arms are behind the flag). fp8-KV composes with fp8 WEIGHTS at runtime:
# the fp8 GEMV opcodes are already in the decode object, selected by the packet.
if [ "${PLOW_BUILD_FP8KV:-0}" = "1" ]; then
  OUT_KV="${OUT%.cubin}_fp8kv.cubin"
  OUT_PF_KV="${OUT%.cubin}_pf_fp8kv.cubin"
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
    "$NVCC" -arch=sm_120a -O3 -cubin \
    -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
    -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1 -DPLOW_FP8_KV=1 \
    -o "$OUT_KV" "$SRC"
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
    cuobjdump -symbols "$OUT_KV" | grep -q "$KSYM" || { echo "FATAL: $KSYM missing in $OUT_KV" >&2; exit 1; }
  echo "built $OUT_KV ($(stat -c%s "$OUT_KV") B), kernel $KSYM present"
  # fp8 prefill dequants at the smem stage, so it needs the PIPE=0 synchronous-staging arm
  # (cp.async cannot convert fp8 inline). -DPLOW_NV_FA_PIPE=0 selects it; decode is PIPE-agnostic.
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
    "$NVCC" -arch=sm_120a -O3 -cubin \
    -I "$HERE/runtime/common" -I "$HERE/runtime/nvidia" \
    -DPLOW_NV_PREFILL=1 -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1 -DPLOW_FP8_KV=1 \
    -DPLOW_NV_FA_PIPE=0 \
    -o "$OUT_PF_KV" "$SRC"
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
    cuobjdump -symbols "$OUT_PF_KV" | grep -q "$KSYM_PF" || { echo "FATAL: $KSYM_PF missing in $OUT_PF_KV" >&2; exit 1; }
  echo "built $OUT_PF_KV ($(stat -c%s "$OUT_PF_KV") B), kernel $KSYM_PF present"
fi
