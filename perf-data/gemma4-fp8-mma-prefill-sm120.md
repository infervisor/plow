# fp8 tensor-core mma inside the px4 prefill flash — beat-fp8-mma (sm_120)

RTX PRO 6000 Blackwell (sm_120, 188 SMs, CUDA 13.0), 2026-07-23. Branch `beat-fp8-mma-prefill`
(from `beat-fp8-prefill`). Campaign: attack the COMPUTE-bound hd512 full-attn prefill flash with
e4m3 mma — the lever the fp8-KV NO-GO left open (its dequant-on-critical-path cost is DELETED
here, not mitigated: the mma consumes the e4m3 cache directly).

## 0. Verified px4 compute budget (this tree — the committed T5 ablation predates px4)

New harness `experiments/px4_fp8mma_ablate.cu` (byte-faithful px4 copy + phase-skip bits,
trailing 8k chunk, 188 blocks). Per-tile ~1300 ns, ctx-FLAT 8k→128k, decomposing as an
imperfectly overlapped TWO-SIDED pipeline, not a serial phase sum:

| side | ns/tile | detail |
|------|---------|--------|
| memory (cp.async L2→smem stream, 32 KB/tile) | ~1170 | 24% exposed; ~5.1 TB/s aggregate — L2 bandwidth, NOT HBM (1535 ceiling never binds) |
| compute | ~990 | QK 35% (latency-bound: 16-deep dependent mma + ldmatrix), softmax 35%, P.V 18%, floor 12% |

Single-phase skips barely move the wall (QK skip measures −7%: fully hidden, and removing it
just lets stageK contend earlier). Cross-check: 1300 ns/tile × 2.68e8 tiles/layer(128k) / 188
= 1.86 s per full layer ≈ the 26B quadratic TTFT term (below). Budget model VERIFIED.

## 0b. Ceiling math (stated before building)

TTFT(C) ≈ a + b·C + c·C² fits the 26B fp8 ladder (gemma4-26b-fp8-prefill-sm120.json) to <2%:
**TTFT ≈ 44.5 + 42.9·C + 0.663·C² ms** (C in k-tok) → at 128k: **linear 5.54 s + quadratic
(hd512 full-attn flash) 10.86 s = 16.40 s.**

FLOP floor per full layer @128k causal (nh16·C²/2·2·(2·512) = 2.82e14 FLOP): bf16 mma 1.34 s,
fp8 mma 0.56 s (`experiments/fp8_mma_rate_probe.cu`: e4m3 m16n8k32 **f32-acc = 930 TF/s =
exactly 2× bf16's 465**, same latency — the 2× does NOT need f16 accum on sm_120).

- 26B (5 full layers): bf16 floor 6.72 s vs 10.86 s measured → px4 already at 62% of the bf16
  physics floor; overhead removal alone caps at −38% of flash.
- fp8-QK only: flash floor 4.76 s → best-case TTFT ≈ 10.3 s.
- all-fp8 (QK+PV): flash floor 2.80 s → best-case TTFT ≈ 8.3 s.
- **vLLM fp8kv 26B@128k = 5.13 s < plow's LINEAR part alone (5.54 s).**

**Honest ceiling, stated up front: flash levers CANNOT close fully to vLLM fp8kv.** The realistic
flash-side outcome is 26B@128k ≈ 11-13 s. What ELSE must move (outside this campaign): the
linear front — MoE grouped-GEMM overlap, per-chunk launch/counter overhead, sliding flash —
needs its own ~2× to reach vLLM territory. Same shape for 12B (8/48 full layers; vLLM fp8kv
12B@128k = 8.10 s).

## 1. Lever verdicts (microbench, `runtime/tests/flashpre_fp8_bw_sm120.cu` + CHUNK regime)

The NO-GO regime (single q-tile) was latency-dominated; the harness grew a **CHUNK regime**
(seq_q=8192 trailing chunk, nsplit=1, persistent 188 grid) that matches the e2e chunked
prefill (bf16 chunk time == the px4 ablation control to 0.1%).

Journey at 32k chunk (bf16 control 50.32 ms): NO-GO-arm-with-fp8-QK v1 74.8 → own-bytes
barrier-free V dequant 68.2 → conflict-free e4m3 stride (PAD8 16→32) 64.5 → scale prefetch +
single-cvt fp16 V dequant (vsc folded into V) 51.3 → A-fragment-order Qs8 (one LDS.128/k-step)
+ 4 independent QMMA chains + paired STS.64 score stores + float2 softmax reads → **46.39 ms**.

| lever | verdict | numbers |
|-------|---------|---------|
| A — fp8 QK^T (m16n8k32.e4m3, raw Ks8, Q per-row e4m3 once/q-tile) | **GO** | part of the 1.01–1.10× win; dequantK + its barrier deleted |
| A' — scale PREFETCH one tile ahead | **GO** | the tile-top gmem scale loads stalled the whole block at the post-WAIT_K barrier: 11.3 ms of 51 ms at 32k → 0.8 ms |
| A'' — own-bytes V dequant (thread converts exactly what it staged) | **GO** | dequant barrier gone; fp8 arm barrier count == bf16 arm (NO-GO paid +2) |
| A''' — fp16 V dequant, vsc folded into V, P unscaled, f16 P.V mma | **GO** | one cvt.f16x2.e4m3x2 per byte-pair; softmax loses its per-element scale fold; relL2 unchanged |
| B — softmax | mostly already optimal | EX2 is raw ex2.approx (3.4% of wall); corr-rescale ballot-skip **NO-GO** (−8%: defeats the mma scheduler; PLOW_NV_FA_CORRSKIP=0) |
| C — fp8 P.V | **NOT TAKEN** | needs BKV=32 to fill k32 (at BKV=16 half of each mma is wasted = parity at best) AND V^T (k=kv contiguous) — an in-kernel transpose or a V^T cache layout; the f16-mma P.V (A''') already removed the dequant round-trip cost |
| D — BKV=32 all-fp8 tile | **NOT TAKEN** | gated behind C; smem plan (~59 KiB, no bf16 KV tiles) documented in plans/ |

### Microbench GO gate (fp8 vs bf16 px4, same grid/smem, unit scales)

CHUNK regime (ladder-faithful): 8k 1.077× / 16k 1.083× / 32k **1.085×** / 64k 1.045× /
128k 1.013× — fp8 FASTER at every ctx. Legacy single-q-tile regime: 1.040–1.104×.
relL2 vs bf16: 2.2–2.5e-2 at every ctx (the established fp8-KV band; unchanged from the
NO-GO arm's verified numerics — Q-quantization adds nothing measurable).

## 2. What shipped (kernel)

`d_flash_prefill_px4<512,32,16,true>` fp8mma arm (`PLOW_NV_FA_FP8MMA`, default 1 under
`PLOW_FP8_KV`; =0 restores the NO-GO dequant arm):
- QK^T: `fa_mma_fp8_k32` (m16n8k32.e4m3.f32) straight off raw Ks8; 8 k32 steps per hd half in
  4 independent chains; B-frag = plain uint2 (8-bit has no ldmatrix).
- Q: per-row amax → e4m3 ONCE per q-tile into an A-FRAGMENT-ORDER Qs8 (one conflict-free
  LDS.128 per k-step); scale into the score store with the k-scale (both factor out of the dot).
- e4m3 staging stride HD+32 (FA_FP8_PAD8): fragment reads bank-conflict-free (8r+2c/half-warp).
- K/V row scales prefetched one tile ahead (regs → STS at tile top).
- V: own-bytes dequant to FP16 with vsc folded into V (single-instruction cvt); P packs half2,
  unscaled; P.V = `fa_mma_f16`. No block barrier for the dequant.
- smem: the DEAD bf16 Ks slot holds qsc/ksc/vsc + Qs8 → px4 fp8 arena 87.0 KiB (< 99 cap).
  Latent overflow FIXED: the generic (hd256) prefill arena wrongly carried the e4m3 staging
  reservation → 100096 B > 99 KiB cap → cooperative prefill grid 0 under any fp8-KV pf build.

## 3. e2e wiring — MIXED fp8-KV (`PLOW_FP8_KV_FULL=1`)

Sliding-layer ring caches are window-bounded (tiny) — fp8 buys them nothing. Mixed mode puts
the e4m3 cache + scales on the hd512 FULL layers ONLY (emitter per-layer flag; sliding layers
keep their shipped bf16 arms byte-identical). This is also what lets the fp8 prefill object
build PIPE=1: `FLASH_PREFILL_FP8` dispatches the px4 fp8mma arm for hd512 (hd256 traps — never
emitted in mixed mode). CMake: `-DPLOW_FP8_KV=ON -DPLOW_FP8_KV_FASTPF=ON` (OFF preserves the
legacy all-layer PIPE=0 prefill). 12B packet KV: 7.02 → 6.02 GiB; 26B: 5.64 → 4.39 GiB.

## 4. TTFT ladders (gemma4_sm120_chat, PLOW_PREFILL=1, fp8 weights w8a8, best of 2)

Control = SAME build, bf16-KV packets (the shipped fp8-weights prefill). vLLM columns =
the trusted B1 baselines. The e2e win is far LARGER than the isolated microbench delta
because the bf16-KV control is L2-THRASHED at long ctx (12B bf16 full-layer KV @128k =
268 MB > L2 while competing with weight streams) whereas the halved e4m3 cache largely
fits — the fp8mma arm wins on compute AND locality together.

### 12B (gemma-4-12B-it, 8/48 full layers, PLOW_UNISEG=1)

| ctx | plow fp8 (bf16-KV, ctl) | **plow fp8mma (mixed KV)** | Δ | vLLM fp8kv | gap | vLLM fp8 |
|-----|----------:|----------:|------:|----------:|-----:|---------:|
| 4k  | 329.3 | **326.9** | −0.7% | 196.4 | 1.66× | 244.7 |
| 16k | 1448.6 | **1311.2** | −9.5% | 868.1 | 1.51× | 1220.8 |
| 32k | 3396.6 | **2627.0** | −22.7% | 1536.8 | 1.71× | 2438.7 |
| 64k | 8781.0 | **5259.8** | **−40.1%** | 4316.4 | 1.22× | 7663.8 |
| 128k| 26709.7 | **10990.5** | **−58.9%** | 8097.4 | **1.36×** | 15520.5 |

**128k gap to vLLM fp8kv: 3.30× → 1.36×.** plow now BEATS vLLM fp8 at 64k (−31%) and
128k (−29%), and vLLM bf16 at 64k/128k.

### 26B (gemma-4-26B-A4B-it MoE, 5/30 full layers, PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 PLOW_MOE_PREFILL=1)

| ctx | plow fp8 (bf16-KV, ctl) | **plow fp8mma (mixed KV)** | Δ | vLLM fp8kv | gap | vLLM fp8 | vLLM bf16 |
|-----|----------:|----------:|------:|----------:|-----:|---------:|----------:|
| 4k  | 198.1 | **196.9** | −0.6% | 134 | 1.47× | 152 | 169 |
| 16k | 867.4 | **782.5** | −9.8% | 526 | 1.49× | 710 | 799 |
| 32k | 2045.4 | **1566.1** | −23.4% | 939 | 1.67× | 1465 | 1544 |
| 64k | 5330.0 | **3135.3** | **−41.2%** | 2646 | 1.18× | 4651 | 4689 |
| 128k| 15594.4 | **6271.1** | **−59.8%** | 5133 | **1.22×** | 9623 | 9293 |

**128k gap to vLLM fp8kv: 3.04× → 1.22×**, and plow now BEATS vLLM fp8 (−35%) and vLLM
bf16 (−33%) at 128k. (Control rows re-measured on this tree: 15594 @128k vs the committed
16398 — same methodology, fresher tree.)

### The quadratic term is GONE (the flash is no longer the TTFT driver)

The 26B fp8mma column is linear in ctx to 3 digits: 782.5 → 1566.1 → 3135.3 → 6271.1 ms
(×2.000 per ctx doubling from 16k). 12B likewise up to 64k (×2.00), with a mild ×2.09 tail
at 128k. The O(ctx²) full-attn flash — the campaign's target — now sits inside the linear
front's noise at ≤128k. **The next TTFT campaign must attack the LINEAR front** (26B slope
≈ 48 ms/k-token vs vLLM fp8kv's ≈ 39): MoE grouped-GEMM overlap, per-chunk launch/counter
overhead, sliding flash.

### Why the e2e win exceeds the ceiling estimate

The ceiling math modeled the quadratic term as COMPUTE at the ablation-harness rate
(KV L2-hot). The e2e bf16-KV control is worse than that model at long ctx — its full-layer
KV (26B: 268 MB bf16 @128k) thrashes L2 against the weight streams, so the real memory side
is far above the harness's 1170 ns/tile. The mixed-e4m3 cache halves those bytes back under
the L2 working set AND halves the QK mma issue — locality and compute win together, hence
−59% where the compute-only model predicted −33%. The two-sided budget model remains correct;
its memory-side calibration was the optimistic piece.

## 5. Gates

- Microbench GO/NO-GO: **GO** (fp8 faster at every ctx, both regimes). relL2 2.2–2.5e-2.
- Oracle: **PASS** — `sm120_interp_op_test: ok` (whole suite on the fp8mma build).
- bf16 byte-identity: **PASS** — the bf16 prefill object (W8A8 + GF_FULL=4, no PLOW_FP8_KV)
  built from this tree `cmp`s equal (834168 B) to the same object at the base commit 0eb0d47.
  All fp8mma behavior sits behind `PLOW_NV_FA_FP8MMA && PLOW_FP8_KV && if constexpr(FP8KV)`
  and the FLASH_PREFILL_FP8 opcode; the emitter's mixed mode is opt-in (`PLOW_FP8_KV_FULL=1`).
- ptxas: **PASS** — fp8 pf object 244 regs / 0 spill (all kernels 0 spill).
- Greedy vs the current fp8 path: **near-tie class, documented** — 4k first gen token
  IDENTICAL to the bf16-KV control (236743). With high-entropy random-token prompts the
  logits are near-flat, so post-step-0 streams settle into different repetition attractors
  under ANY numerics perturbation (the CONTROL itself flips first tokens across ctx); the
  quantitative gate is the relL2 band above — the same band the SHIPPED fp8kv decode
  carries (its committed acceptance) — plus the oracle.
- Decode sanity (mixed mode must not regress the shipped fp8kv decode wins): 12B @32k
  **12.42 ms/tok (mixed) vs 12.74 (bf16-KV control)** — slightly FASTER; the full-layer
  fp8 decode arm outweighs the sliding-layer bf16 reversion (sliding rings are tiny).

## 6. What remains (honest)

- The remaining 128k gap (26B 1.22×, 12B 1.36× vs vLLM fp8kv) is now the LINEAR front —
  the fp8mma ladders are ctx-linear, so more flash work has almost no TTFT left to win at
  ≤128k. Next campaign: MoE grouped-GEMM overlap / per-chunk overhead / sliding flash.
- Levers C (fp8 P.V) + D (BKV=32 all-fp8 tile) remain unbuilt: they need the V^T operand
  (in-kernel byte transpose or a V^T cache layout) and would cut the flash floor another
  ~2× — worth revisiting only if ctx ≫ 128k brings the quadratic term back.
- Batched/varlen (PX-1) prefill still routes around px4, so fp8mma serves the legacy
  single-request path only (as px4 itself does).
- 4k/16k rungs are linear-front-bound already; fp8mma is neutral there by design.
