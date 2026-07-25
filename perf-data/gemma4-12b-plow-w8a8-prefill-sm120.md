# gemma-4-12B w8a8 fp8 prefill GEMM on sm_120 (RTX PRO 6000 Blackwell) — T7 L2

The compute-bound fix T6 pointed at. T6-L2 w8a16 (fp8 weight, bf16 activation, dequant-to-bf16
-in-smem) measured NEGATIVE for speed because prefill GEMM is COMPUTE-bound (each weight is reused
across many output rows), so halving the DRAM weight stream buys nothing. w8a8 attacks the compute
wall directly: `mma.sync.m16n8k32.e4m3` tensor cores (2x the bf16 rate, rtx-05: fp8 peak 503.8 vs
bf16 209.5 TFLOP/s) with BOTH operands e4m3 and ONE k32 mma per BK=32 tile (vs bf16's two k16).

## Status: COMPLETE (T8, 2026-07-19) — emitter wired, e2e measured. MEASURED POSITIVE.

The device kernels + interp dispatch (`PLOW_NV_W8A8=1`) and the emitter path (`gemma4.rs`,
`PLOW_W8A8=1`) are complete and validated end-to-end. **w8a8 is the first prefill lever to BEAT plow
bf16** (−5…−16.5%), but still 1.56–2.29× behind vLLM fp8 TTFT. Full table, per-op trace, and the
vLLM-fp8 divergence-budget comparison are in `gemma4-12b-plow-prefill-sm120.{md,json}` under the
`T8-w8a8-prefill` campaign. Emitter: ONE shared `DevOp::QuantFp8` per activation site (qkv/o/gate-up/
down, 4/layer) feeding `GEMM_FP8`/`GEMM_GLU_FP8` re-pointed to `t1=xq` + `t3=a_scale`; the shared
quant is required (a per-proj quant races the xq buffer). Dep-graph verified (280/280 GEMM_FP8 wait
on their QuantFp8); `PLOW_W8A8` unset ⇒ byte-identical emission. pkts need `PLOW_UNISEG=1` (the
sm_120 persistent harness runs only the coarse single-segment path).

The measured e2e sweep (plow bf16 A/B control | plow w8a8 | Δ | vLLM fp8 TTFT | ratio):
4k 513.97 | 429.41 | −16.5% | 244.71 | 1.75×; 16k 2169.97 | 1918.41 | −11.6% | 1220.76 | 1.57×;
32k 5058.93 | 4554.95 | −10.0% | 2438.73 | 1.87×; 64k 12979.20 | 11964.73 | −7.8% | 7663.76 | 1.56×;
128k 37351.31 | 35485.36 | −5.0% | 15520.48 | 2.29×.

## Measured (GPU, sm_120, one t7-prefill lease, 2026-07-19)

Fragment layout + tiled-GEMM correctness (`experiments/fp8_verify.cu`, `fp8_gemm_w8a8_probe.cu`):

| probe | result |
|---|---|
| e4m3 m16n8k32 fragment layout | 0/128 exact mismatches — PASS |
| w8a8 tiled GEMM 128x128x64 (staging + frags + two-scale epilogue) | 0/16384 exact — PASS |
| w8a8 GLU 128x128x64 (gelu-tanh) | relL2 1.6e-3 — PASS |

Oracle (`sm120_interp_op_test`, e4m3-aware tolerance):

| case | relL2 (gate: dequant-both ref) | quant-err vs full-precision |
|---|---|---|
| quant_fp8 (x2, scale + xq) | 0 (bit-exact) | — |
| gemm_w8a8 (x4, incl lm_head a_row0) | 0 .. 5.2e-5 | **~3.6% (0.036)** |
| gemm_glu_w8a8 (x2, gelu/silu) | 5.3e-5 .. 9.6e-5 | — |

`sm120_interp_op_test: ok` (whole suite, incl. the L1 flash edges, all PASS).

## Numerics characterization (the honest w8a8 divergence)

The GATE compares against a reference that DEQUANTIZES both operands (kernel and ref see the SAME
e4m3 values): e4m3*e4m3 products are exact in f32, so the gate charges only the k-accumulation order
+ bf16 output round → **~1e-3 relL2** (tighter than w8a16, since the activation is now also e4m3-exact
rather than bf16-rounded). This proves the KERNEL is arithmetically correct.

Separately, the w8a8 output vs the FULL-PRECISION (unquantized) matmul is **relL2 ~3.6%** — this is
the e4m3 quantization error itself (both activation AND weight rounded to 3-mantissa-bit e4m3, max
rel 2^-4 = 6.25%/elt, partially cancelling over the K-sum). This is INHERENT to w8a8 and matches
standard w8a8 practice (vLLM fp8 drifts from its own bf16 similarly). Per the campaign rule, an fp8
mode with documented quantization error is acceptable IF the divergence matches standard w8a8 — it
does. ptxas 238 regs / 0 spill / occ-1 UNCHANGED; decode cubins byte-identical.
