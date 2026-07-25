# Gemma-4-26B-A4B — plow sm_120 vs vLLM — campaign P9 (2026-07-20)

One RTX PRO 6000 Blackwell (sm_120, 188 SMs, CUDA 13.0), TP1, batch 1, single
132096-ctx packet per config (matches vLLM `--max-model-len 132096`). vLLM
reference: `gemma4-26b-a4b-vllm-sm120.md` (0.25.1, TRITON_ATTN, cudagraphs,
autotuned FlashInfer CUTLASS bf16 MoE / untuned Triton fp8 MoE).
Harness: `gemma4_sm120_chat`, 128 gen tok, n=112 timed, sd ≤0.03 ms.
Supersedes the P3-26b report (13.15→15.32 ms era).

## Config

- Commit `eb359da` + GF4 build. Binary: `cmake -B build-gf4 -S runtime -DPLOW_CUDA=ON
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-DPLOW_NV_FA_GF_FULL=4"` (plain env, NOT nix).
- Packets: `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 [PLOW_FP8=1 PLOW_FP8_HEAD=1] gemma4 <dir> 132096 out.pkt 188`.
- `fp8+head` = weight-only fp8 experts+projections + fp8 lm_head twin
  (per-row e4m3, relerr 2.7%). **Own row: vLLM fp8 keeps lm_head bf16.**
  Plain fp8 (bf16 head) = add ~0.44 ms/tok (measured A/B −0.47 @40ctx).

## Decode TPOT ms/token (batch 1; bold = beats vLLM)

| ctx | plow bf16 | vLLM bf16 | plow fp8+head | vLLM fp8 | vLLM fp8kv |
|-----|-----------|-----------|---------------|----------|------------|
| 1k  | 8.24 | 7.61 | 5.84 | 5.76 | 5.92 |
| 4k  | 8.29 | 7.90 | **5.89** | 6.08 | 6.19 |
| 16k | **8.61** | 8.64 | **6.20** | 6.82 | 6.62 |
| 32k | **9.07** | 9.57 | **6.67** | 7.74 | 7.28 |
| 64k | **9.94** | 10.33 | **7.54** | 8.63 | 7.52 |
| 96k | **10.81** | 11.34 | **8.40** | 9.54 | 7.94 |
| 128k| **11.57** | 12.34 | **9.16** | 10.48 | 8.46 |

- bf16: beats vLLM at 16k–128k. Short-ctx gap 0.63/0.39 ms (1k/4k) remains.
- fp8+head: beats vLLM fp8 at 4k–128k; 1k −0.08 ms short. Also beats vLLM
  **fp8kv** at 1k–32k, ties 64k (7.54 vs 7.52); fp8kv wins 96k/128k (halved KV
  stream — plow fp8-KV not yet measured, PLOW_FP8_KV exists).
- GF2/ns24 variant wins 1k fp8 (5.73 vs 5.76) but gives back 0.1–0.3 ms at
  ≥32k; the shipped single-pkt config is GF4/ns48.

## Prefill (bf16 grouped-MoE, chunked 8192) — NOT yet competitive

| ctx | plow prefill ms | tok/s | vLLM bf16 TTFT | ratio |
|-----|-----------------|-------|-----------|-------|
| 1k  | 113 | 9061 | 75 | 1.51× |
| 4k  | 323 | 12678 | 169 | 1.91× |
| 16k | 1402 | 11682 | 799 | 1.76× |
| 32k | 3251 | 10079 | 1544 | 2.11× |
| 64k | 8293 | 7903 | 4689 | 1.77× |
| 96k | 15130 | 6497 | 6980 | 2.17× |
| 128k| 23743 | 5520 | 9293 | 2.55× |

Correctness-first grouped GEMM (ops 73–77), untuned. Ratio worsens with ctx →
FLASH_PREFILL quadratic part + per-token router loop are prime suspects
(tuning campaign in flight). fp8 prefill not implemented.

## Correctness gates (all PASS)

- `sm120_interp_op_test: ok` — full suite incl. MoE decode ops, op72 golden
  (relL2=0), grouped-prefill ops (router ties, align invariants, grouped
  GLU/down vs naive, combine relL2 ≤5.5e-4).
- Decode token parity: old-pkt vs new-pkt IDENTICAL (40-tok chat + 2204-tok
  window-crossing); GF4 vs GF2 IDENTICAL @40ctx; device/host argmax AGREE at
  every ladder point.
- Prefill vs decode-consume: IDENTICAL on 40-tok real prompt (pad path) and
  512-tok; 2204-tok first token exact pre-ns48. Long synthetic prompts show
  the known bf16 prefill/decode near-tie drift (dense 12B baseline diverges
  identically).
- fp8+head: logits shift ≤±0.15 on |23| top-1; greedy diverges at a near-tie
  (~tok 34 on p0) — same class as vLLM fp8 non-exactness.

## What moved the number (13.15 → 8.24 bf16 @1k-class; 6.60 → 5.84 fp8)

1. op71 fused norm+GLU + arena smem-staged GEMV/QKV/GLU + vectorized expert
   dot (inherited mid-flight work, validated): 13.15 → 8.70.
2. Router hoist (emit score+topk before dense MLP; per-block stream order is
   execution order): 8.70 → 8.08. GLU gate wait 2.59M → 0.43M cyc/token.
3. fp8 lm_head twin: −0.47 ms (fp8 config).
4. GF_FULL=4 + ns48 (full-layer flash re-read 4×→2×): −0.15 @32k growing to
   −0.7 @128k-class.

Measured negatives (kept off): op72 scalar tail fusion (+0.18 ms), prodcons
warp-spec GEMV (never beats uniform; −32% small-K), L2-coschedule of flash
re-readers (+5..23%), MXFP8 block-scale mma (absent on sm_120 hardware).

## Remaining gaps / next levers

- bf16 1k/4k decode: 0.4–0.6 ms. GEMV bodies at 60–84% of the measured
  1535 GB/s ceiling; small-K (down, K=704) worst. Gate waits ~1.7 ms/token.
- TTFT everywhere (table above).
- fp8-KV row (vs vLLM fp8kv 96k/128k), fp8 prefill, serving-mux numbers
  (standalone ≠ serving win).

Raw logs: `perf-data/p9-sweep/`. Campaign log: `plans/p9-26b-campaign.md`.

## T9a-moe-tune — single-block decode trace (ctx 4k) + trace-order verdict

Method mandate: TRACE one full 26B decode step FIRST, then tune what the trace
shows. Trace facility: `PLOW_NV_TRACE_DEC=ON` cmake option instruments ONLY the
decode gemma object (mirrors the existing `_PF` prefill trace; `_pf` stays
trace-free so the `g_tr_*` device globals resolve to one definition). Block 0,
thread 0 records per-packet `gate / body / sig` cycles; `PLOW_NV_TRACE_SKIP=N`
fires the dump at launch index N so the traced step lands at a WARM context
(here N=4090 → ctx≈4090) instead of priming-step-0. Default build is
byte-identical (all trace code under `#if PLOW_NV_TRACE`, option default OFF).

Config: commit `f5e1492` + GF_FULL=4, `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48`,
decode program 521 packets / 41089 wg-packets. GQ scheduler load-balances, so
block 0's 210 claimed packets are a ~1/188 uniform sample of the step and the
per-opcode ranking is a faithful proxy for where step time goes.

Block-0 totals (one ctx-4k step): **body 74.6% · gate 21.6% · sig 3.8%**.

| opcode                       | cnt | tot_cyc |   %  |   mean | body% | gate% |
|------------------------------|----:|--------:|-----:|-------:|------:|------:|
| GEMV (dense MLP/proj/lm_head)|  65 | 7015333 | 36.4 | 107928 |  72.1 |  24.2 |
| MOE_EXPERT_GLU_NORM_GEMMA    |  30 | 4009865 | 20.8 | 133662 |  85.1 |  11.4 |
| GEMV_QKV                     |  25 | 3176990 | 16.5 | 127080 |  63.3 |  33.6 |
| MOE_EXPERT_DOWN_GEMMA        |  30 | 1937175 | 10.1 |  64572 |  87.3 |   7.3 |
| GEMV_GLU                     |  25 | 1504779 |  7.8 |  60191 |  73.9 |  21.1 |
| FLASH_DECODE                 |  30 | 1207181 |  6.3 |  40239 |  68.0 |  27.7 |
| MOE_ROUTER_GEMMA_SCORE_FAST  |   4 |  268729 |  1.4 |  67182 |  71.2 |  25.6 |
| MOE_ROUTER_GEMMA_TOPK        |   1 |  128365 |  0.7 | 128365 |  48.2 |  50.6 |

(MOE_COMBINE_RESID_NORM and the RMS/embed/argmax tail did not fall in block 0's
sample — they are cheap; combine is a fixed-order Σ over k=8 f32 partials.)

### Trace reading — the mandated levers were ALREADY landed; the trace confirms it

The T9a mandate (router-split, tail-norm fusion, dim-specific expert kernels)
was implemented and merged by the P9 campaign BEFORE this trace. The trace is
the independent confirmation:

- **Router-split (lever 2a): DONE and working.** Router is now
  `MOE_ROUTER_GEMMA_SCORE_FAST` (16 CTAs × 8 experts) + a one-CTA
  `MOE_ROUTER_GEMMA_TOPK` tail — **~2.1% of block-0 cycles total**, and hoisted
  before the dense MLP so it overlaps. The P3-era "one-block router serialized
  ×30 layers" cost is gone. TOPK is the only op that is gate-bound (50%), but it
  is 0.7% of the step — not worth splitting further.
- **Tail-norm fusion (lever 2b): DONE.** The expert GLU carries its pre-norm
  fused in (`MOE_EXPERT_GLU_NORM_GEMMA`) and the tail is
  `MOE_COMBINE_RESID_NORM_GEMMA` (combine+residual+next-norm in one packet).
  The MoE expert ops now show **very low gate-wait (7–11%)** — they are not
  gate/packet-serialized; the fusions already collapsed those gates.
- **Dim-specific / skinny-expert (lever 2c): N/A at B=1.** The expert GLU/DOWN
  are body-bound (85–87%) flat one-warp-per-output GEMVs already running the
  dense-path vectorized inner loop (`ld_glob8`/`dot8`, K=2816 glu / K=704 down).
  Register multi-row reuse (`gemv_rows<MM>`) is a **B>1** lever (co-resident
  slots sharing an expert row) — inapplicable to single-token decode where each
  of the 8 experts is read once. The remaining expert cost is genuine
  expert-weight HBM bandwidth (~95 MB/layer × 30 ≈ 2.85 GB/token of routed
  expert weights at bf16), which the fp8+head ladder already halves.

### Verdict: no further MoE-kernel lever is justified by the trace

The MoE surface (router 2.1% + expert GLU/DOWN 30.9% + combine ≈0) is 33% of the
step and is **body/bandwidth-bound with the gates already fused out** — exactly
the state the levers aimed for. The residual step-level gate-wait (21.6%)
concentrates on **GEMV_QKV (33.6%)** and **GEMV (24.2%)** — the attention/dense
path, i.e. the scheduler + dense-GEMV surface owned by the T9b/T9c rounds, not
the MoE op bodies. Chasing a MoE-kernel change here would be speculative; the
honest result is that prior work closed this task's gap and the trace proves the
MoE dispatch is already well-pipelined.

### Gate (this branch, no functional kernel change)

- `sm120_interp_op_test: ok` — 28/28 PASS, 0 FAIL, incl. router lowest-id **tie**
  and `moe_comb_resid_norm` relL2=0. MoE decode + grouped-prefill all PASS.
- Decode TPOT ladder = the committed P9 result (unchanged; no functional edit):
  bf16 **8.24/8.29/8.61/9.07/9.94/10.81/11.57** vs vLLM 7.61/7.90/8.64/9.57/
  10.33/11.34/12.34 (1k..128k) — beats vLLM at 16k–128k; 0.63/0.39 ms behind at
  1k/4k. fp8+head **5.84…9.16** beats vLLM fp8 at 4k–128k.
- Dense blobs byte-identical: the emitter (`gemma4.rs`) is untouched on this
  branch, so 12B/31B/Qwen packet output is unchanged by construction.

### Remaining gap + cause (unchanged from P9, now trace-substantiated)

bf16 1k/4k: 0.63/0.39 ms behind vLLM. Cause per this trace: **not** MoE — it is
(a) dense/attention GEMV body time (GEMV+GEMV_QKV+GEMV_GLU = 60.7% of the step,
bodies at 60–84% of the 1535 GB/s ceiling, small-K down worst) and (b) ~21.6%
interpreter gate-wait dominated by GEMV_QKV/GEMV. Both are the T9b (decode
GEMV/gates) and T9c (scheduler/segment) surfaces, not the MoE op bodies.
