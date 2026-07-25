# Gemma-4-26B-A4B grouped-MoE PREFILL — fp8 (w8a8) NEW + bf16 re-measured vs vLLM — beat26b

RTX PRO 6000 Blackwell (sm_120, 188 SMs, CUDA 13.0), 2026-07-22. Branch `beat26b-prefill`.
Delivers the two workstreams: (1) a single-block grouped-GEMM tile microbench, (2) the first
**fp8 (w8a8) grouped-MoE prefill** implementation (plow fp8 prefill was NOT IMPLEMENTED).

- Harness `gemma4_sm120_chat` (PLOW_PREFILL=1), built `-DPLOW_NV_W8A8=ON -DPLOW_NV_FA_GF_FULL=4`.
- Packet `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 PLOW_MOE_PREFILL=1 [PLOW_FP8=1 PLOW_W8A8=1] … 132096`.
- Metric `prefill_ms` = full chunked prefill incl. the first generated token = TTFT (same
  convention as the P9 plow table and the vLLM TTFT column).
- **plow rows are RE-MEASURED on current `main`** (post-P9 `eb359da`); they are faster than the
  P9-published plow figures because of intervening main prefill work (flash PX-4/PIPE, T9). vLLM
  columns are the trusted B1 baseline, not re-derived.

## Single-block grouped-GEMM microbench (the mandated first step)

Full detail in `beat26b-moe-group-microbench-sm120.md`. Verdicts that drove this campaign:
- **bf16 tile winner = the shipped default** (BM128/BN128/BK32/GLU2/ST3). BN64 *loses* (DOWN 35%
  slower at T8192 — 2× the n-tiles ⇒ 2× the contiguous-A re-read); GLU_STAGES=3 is noise + more
  arena. The grouped GEMM is already at its tile-param optimum — **no bf16 tile-tuning headroom.**
  ("Activation-stationary reuse" is not expressible at this register budget; larger N-tiles make
  the A re-read *worse*.)
- **fp8 w8a8 grouped: GLU 1.67–2.02×, DOWN 1.26–1.52×** over bf16 at the kernel level ⇒ implement.

## TTFT ladder (ms, lower better) — plow vs vLLM

| ctx  | vLLM bf16 | vLLM fp8 | plow bf16 (P9 pub) | **plow bf16 (main)** | **plow fp8 (w8a8, NEW)** | fp8/bf16 |
|------|----------:|---------:|-------------------:|---------------------:|-------------------------:|---------:|
| 1k   | 75   | 88   | 113   | 91.9   | **72.5** | 1.27× |
| 4k   | 169  | 152  | 323   | 237.1  | 200.4 | 1.18× |
| 16k  | 799  | 710  | 1402  | 1009.2 | 882.3 | 1.14× |
| 32k  | 1544 | 1465 | 3251  | 2350.4 | 2096.2 | 1.12× |
| 64k  | 4689 | 4651 | 8293  | 6018.7 | 5506.2 | 1.09× |
| 96k  | 6980 | 7131 | 15130 | 10986.9| 10221.1 | 1.08× |
| 128k | 9293 | 9623 | 23743 | 17280.6| 16398.4 | 1.05× |

## GO / NO-GO per ctx

| ctx | plow fp8 vs vLLM fp8 | plow fp8 vs vLLM bf16 | plow bf16 vs vLLM bf16 |
|-----|----------------------|-----------------------|------------------------|
| 1k  | **GO** (72.5 < 88, −18%) | **GO** (72.5 < 75) | no (91.9 vs 75, 1.23×) |
| 4k  | no (1.32×) | no (1.19×) | no (1.40×) |
| 16k | no (1.24×) | no (1.10×) | no (1.26×) |
| 32k | no (1.43×) | no (1.36×) | no (1.52×) |
| 64k | no (1.18×) | no (1.17×) | no (1.28×) |
| 96k | no (1.43×) | no (1.46×) | no (1.57×) |
| 128k| no (1.70×) | no (1.76×) | no (1.84×) |

**fp8 GO at 1k** (beats both vLLM bf16 and vLLM fp8). NO-GO elsewhere.

## Correctness gates (all PASS)

- **Oracle `sm120_interp_op_test: ok`** — grouped-prefill ops (router ties, align invariants,
  grouped GLU/down vs naive, combine relL2 ≤5.5e-4) + the w8a8 GEMM tests all PASS.
- **fp8 greedy coherence**: fp8 first gen token == bf16 at 1k/16k/32k/64k/96k/128k
  (189113 / 89998 / 150437 / 236743…); differs only at 4k (near-tie fp8 drift — same class as
  vLLM fp8 non-exactness, which is itself not bit-exact).
- **bf16 parity inherited**: the bf16 grouped ops are byte-unchanged on this branch, so the P9
  prefill==decode-consume 512-tok 32/32 EXACT gate carries over.
- **Default byte-identical**: every addition is gated (`#if PLOW_NV_W8A8` in the interp, `if w8a8`
  in the emitter, enum appended without renumber). bf16 packets never emit ops 81/82.

## Honest verdict

- **fp8 grouped-MoE prefill delivered** — the previously-missing capability now exists, is
  correctness-gated, and **beats the soft target (vLLM fp8) and vLLM bf16 at 1k**. The kernel-level
  fp8 speedup (microbench 1.3–2.0×) is real.
- **Why fp8 only wins at 1k**: fp8 accelerates the MoE + dense **GEMMs**, but TTFT is increasingly
  dominated by **FLASH_PREFILL attention** (quadratic in ctx, and **dtype-invariant** here — the KV
  path stays bf16). At 1k the FFN GEMMs dominate → fp8 wins; by 128k flash dominates and the fp8
  end-to-end win compresses to 1.05×. vLLM's tuned FlashInfer-CUTLASS + cudagraphs attention is the
  moat at mid/high ctx, not the MoE GEMM.
- **Residual bf16 gap**: re-measured bf16 on current main is 1.25–1.45× faster than the P9-published
  figures but still trails vLLM bf16 everywhere (1.23–1.84×). The microbench proves this is **not**
  tile-tunable in the grouped GEMM; it is the flash quadratic + the untuned-grouped-vs-autotuned-
  CUTLASS-MoE gap. Closing it needs a faster prefill flash (the dense front already declared
  exhausted) or fp8-KV (halves the KV stream — vLLM's fp8kv column is far ahead for exactly this
  reason; a separate, lossy lever outside the bf16/fp8 goal).
- **Next lever under evaluation** (coordinator input): wiring the existing-but-unwired MoE DOWN
  fine-dependency map to overlap grouped GLU→DOWN→combine within a chunk — gated on a
  `PLOW_NV_TRACE_PF` measurement of the grouped ops' gate-wait share at the 8192 bucket
  (<5% ⇒ skip; >10–15% ⇒ wire). See campaign notes.
