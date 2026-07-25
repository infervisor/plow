# rtx-19 E3 — fp8 (e4m3) KV cache (`PLOW_FP8_KV`)

Store K/V in the decode/prefill flash cache as **e4m3** (1 byte/elem, HALF the bf16
bytes) with a **per-row f32 dequant scale** (amax→448). Write: `d_headnorm_rope_fp8`
(op_norm.cuh). Read: `d_flash_decode<...,FP8KV=true>` dequant in the inner loop,
`d_flash_prefill<...,FP8KV=true>` dequant at the smem stage. Default OFF ⇒ bf16 KV,
byte-identical.

- GPU: RTX PRO 6000 Blackwell, sm_120, 188 SM, 96 GiB. Toolchain: **system nvcc 13.0** (not nix).
- Twin config: fp8 weights `-DPLOW_FP8=1` + `/workspace/models/gemma-4-31B-it/fp8` (29.9 GiB).
- 12B = fast correctness loop; 31B fp8-weights+fp8-KV = headline vs vLLM fp8kv.

## Build gate

- **Default byte-identical (flag off):** decode + prefill cubins SHA256-match the main
  build. `673288677424c587912c95af61d506e99f32ef10768b4b298f8b613ce045233b` (decode).
- **ptxas / spill (no spills anywhere):**
  - decode: 210 regs (fp8-KV) vs 212 (bf16), STACK 0.
  - prefill fp8-KV (`PLOW_NV_FA_PIPE=0`): 240 regs, STACK 0.
- CMake fix (this run): the fp8-KV prefill objects (`_pf/_seg/_gemm`) build with
  `PLOW_NV_FA_PIPE=0` — the `FLASH_PREFILL_FP8` arm dequants at the smem stage and only
  exists on the synchronous-staging build (cp.async can't convert fp8 inline). Decode is
  PIPE-agnostic; bf16-KV runs use the default (PIPE=1) harness, unaffected.

## Parity gate — PASS (fp8-KV band, not bit-exact)

Real text (`prompt_tpot`, 3587 natural tokens), fp8-KV vs bf16-KV:

| model | first token | identical greedy prefix | step-0 logits relL2 | argmax |
|------|------------|------------------------|--------------------|--------|
| 12B (bf16 w) | MATCH | 21 tokens | 3.12% (prefill) / 5.89% (decode) | top-3 identical |
| 31B (fp8 w)  | MATCH (236865) | ≥30 tokens | 2.67% | MATCH |

Band = fp8-KV lossy (e4m3, 3 mantissa bits, per-row scale) — relL2 2.7–5.9%, top-1 agree.
vLLM-fp8kv-class accuracy; greedy drifts after ~21 tokens (12B). NOT bit-exact (expected).
The degenerate periodic-input case shows an 82% step-0 relL2 artifact (bf16 prefill is
anomalously overconfident there); real text is the valid signal.

## Decode TPOT — 12B (bf16 weights), fp8-KV vs bf16-KV

ITL is extremely stable (sd < 0.05 ms, < 0.2% of mean); median ≈ mean.

| ctx | bf16-KV ms/tok | fp8-KV ms/tok | Δ |
|-----|---------------|---------------|-----|
| 4k  | 18.784 | 19.139 | **+1.9%** |
| 16k | 19.294 | 19.667 | **+1.9%** |
| 32k | 19.974 | 20.417 | **+2.2%** |
| 64k | 21.289 | 21.772 | **+2.3%** |

12B single-user decode is **~2% SLOWER** with fp8-KV at every ctx — the per-element dequant
ALU exceeds the KV-bandwidth saved (the 22 GiB weight stream dwarfs even a 6 GiB KV read at
64k). No crossover for 12B. **Honest negative.**

## Decode TPOT — 31B (fp8 weights), fp8-KV vs bf16-KV

| ctx | bf16-KV ms/tok | fp8-KV ms/tok | Δ |
|-----|---------------|---------------|-----|
| 4k  | 26.482 | 26.907 | +1.6% |
| 16k | 27.454 | 27.654 | +0.7% |
| 32k | 28.961 | 28.619 | **−1.2% (fp8-KV FASTER)** |

31B crosses over by 32k: the larger KV (16 kv-heads) makes halving the KV read beat the
dequant ALU. At long ctx fp8-KV **extends the decode lead**; at short ctx it is a slight loss.

## 31B vs vLLM (decode TPOT ms/tok)

| ctx | plow fp8w+fp8kv | vLLM fp8 | vs fp8 | vLLM bf16 | vs bf16 |
|-----|-----------------|----------|--------|-----------|---------|
| 4k  | 26.907 | 26.16 | +2.9% | 45.20 | **−40%** |
| 16k | 27.654 | 27.80 | **−0.5%** | 46.93 | **−41%** |
| 32k | 28.619 | 29.86 | **−4.2%** | 49.14 | **−42%** |

plow fp8w+fp8-KV beats vLLM fp8kv at ≥16k and stays ~−41% vs vLLM bf16 everywhere.

## KV VRAM halved + 31B multi-user batch cap

KV per slot (emitter self-report): 12B@64k 6.00→3.04 GiB; **31B@32k 15.00→7.61 GiB (−49%)**.

31B multi-user, 96 GiB card, fp8 weights 29.9 GiB + ~6 GiB overhead ⇒ ~60 GiB KV budget, ctx 32k:

| KV dtype | GiB/slot | concurrent 31B slots |
|----------|----------|----------------------|
| bf16 | 15.00 | **B=4** (the current cap) |
| fp8  | 7.61 | **B=7–8** |

**Halving the KV bytes doubles the KV-limited batch cap: B=4 → B=7–8.** This is the decisive
win — it lifts the 31B multi-user cap the plan targeted.

## Verdict

fp8-KV **passes parity** (vLLM-fp8kv-class, top-1 agree). It does **not** speed up short-ctx
single-user decode (dequant ALU costs ~2%), but **extends the decode lead at long ctx**
(31B −1.2% @32k, crossover by 32k) and — the headline — **doubles the 31B multi-user batch
cap from B=4 to B=7–8** by halving KV bytes. Ship behind the flag; enable for long-ctx and
multi-user 31B serving, leave off for short-ctx single-user latency.

### Honest negatives
- Dequant ALU eats the win at short ctx: 12B +2% at all ctx (no crossover even at 64k); 31B +1.6% @4k.
- Accuracy drift is real: ~3–6% logits relL2, greedy diverges after ~21 tokens (12B). Acceptable fp8-KV band, not bit-exact.
- fp8 prefill needs the PIPE=0 (synchronous-staging) object, so fp8-KV prefill TTFT is ~20–30% slower than the cp.async bf16 prefill (prefill only; decode unaffected).
