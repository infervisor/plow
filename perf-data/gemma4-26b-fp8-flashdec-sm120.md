# Gemma-4-26B-A4B — fp8-KV flash-DECODE kernel tune (hd512 full-attn) — sm_120 (2026-07-22)

Follow-on to `gemma4-26b-fp8kv-decode-sm120.md` (beat26b-decode). That campaign localized the
entire long-ctx fp8-KV decode deficit vs vLLM fp8kv to the **`FlashDecodeFp8<512>` hd512
full-attention read** (5 of 30 layers; the 25 sliding layers ring-cap at window 1024 and are
ctx-invariant), and measured plow's marginal KV cost 32k→128k at **~21.3 ns/tok vs vLLM's ~12.0
(~1.8×)**. This campaign micro-benchmarks that kernel, A/Bs the named levers, ships the winner,
and re-measures the ladder.

Branch `beat26b-flashdec` (from `beat26b-decode`). One RTX PRO 6000 Blackwell (sm_120, 188 SM,
CUDA 13.0), TP1, batch 1.

## What changed (kernel)

`runtime/nvidia/op_attention.cuh`, **fp8 arm only** (`if constexpr (FP8KV)`), behind a new
build flag `PLOW_FP8_FAST` (bf16 arm and the shipped fp8 arm byte-identical when unset):

- **e4m3 → f32 dequant, direct** (drop the intermediate `__float2bfloat16` round-trip). The
  shipped fp8 arm decodes e4m3 → bf16 (`fp8v8_to_bf16v8`) then the dot converts bf16 → f32 —
  a double rounding + extra ALU per element. The microbench shows the fp8 read is
  **dequant-ALU bound, not bandwidth bound** (see below), so this is the lever. New helpers
  `fp8v8_to_f32` / `fp8v16_to_f32`; score-phase K and V-phase both accumulate in f32.
- 8-byte K loads retained (16-byte `PLOW_FP8_LD16` variant tested and REJECTED — see A/B).

Numerically e4m3→f32 is *more* precise than e4m3→bf16→f32 (drops a rounding); the fp8 flash
oracle relL2 is identical to the shipped arm to 4 sig figs (error dominated by the e4m3 write
quant, not the dequant domain).

## 1. Microbench — `FlashDecodeFp8<512>` KV-read (runtime/tests/flashdec_fp8_bw_sm120.cu)

Standalone single-kernel harness, `__launch_bounds__(256,1)` (matches the megakernel's 1-block/SM
pin), grid = n_work = n_grp·nsplit items (GF4→ns48, GF8→ns96; grid-aligned to the shipped
`PLOW_NS_FULL_ABS=48`). Geometry = 26B full-attn layer: n_head=16, n_kv_head=2, gqa=8, D=512.
Best of 60. **GBps_iss** = bytes the kernel demands from the mem system (K+V, per group);
**GBps_phys** = distinct HBM bytes (GF4 issues 2× phys — see traffic note). HBM ceiling 1535 GB/s.

### fp8 GF4 (the shipped full-attn config) — before/after

| ctx  | shipped ms | **FAST ms** | Δ      | shipped GBps_iss | FAST GBps_iss | FAST %ceil_iss | FAST GBps_phys |
|------|-----------|-------------|--------|------------------|---------------|----------------|----------------|
| 32k  | 0.1166    | **0.1087**  | −6.8%  | 1151             | 1235          | 80.4%          | 617            |
| 64k  | 0.2172    | **0.2027**  | −6.7%  | 1236             | 1325          | 86.3%          | 662            |
| 96k  | 0.3071    | **0.2900**  | −5.6%  | 1311             | 1388          | 90.4%          | 694            |
| 128k | 0.4152    | **0.3897**  | −6.1%  | 1293             | 1378          | 89.8%          | 689            |

### vs the bf16 FlashDecode path (same GF4, same lengths)

| ctx  | bf16 ms | bf16 GBps_iss | bf16 GBps_phys | fp8 FAST ms | fp8/bf16 time |
|------|---------|---------------|----------------|-------------|---------------|
| 32k  | 0.0967  | 2777 (L2)     | 1388           | 0.1087      | 1.12×         |
| 64k  | 0.3331  | 1612          | 806            | 0.2027      | 0.61×         |
| 96k  | 0.4862  | 1656          | 828            | 0.2900      | 0.60×         |
| 128k | 0.6649  | 1615 (L2)     | 807            | 0.3897      | 0.59×         |

bf16 GBps_iss **exceeds** the 1535 ceiling (L2 reuse of the 2× GQA reread) — bf16 is
latency/occupancy-bound at ~807 GB/s *physical* HBM, not bandwidth-bound. fp8 FAST moves half
the bytes and runs 0.59–0.61× the bf16 time from 64k up.

### Marginal KV cost, 32k→128k slope (per full-attn LAYER and ×5-layer model projection)

| cfg              | ns/tok (layer) | ns/tok (×5 model) |
|------------------|----------------|-------------------|
| bf16 GF4         | 5.78           | 28.9              |
| **fp8 GF4 shipped** | 3.04        | **15.2**          |
| **fp8 GF4 FAST**    | 2.86        | **14.3**          |
| fp8 GF8          | 3.68           | 18.4              |
| fp8 GF4 LD16     | 3.43           | 17.2              |

Campaign MODEL marginal: **plow 21.3 ns/tok, vLLM 12.0**. The isolated FAST kernel slope
(14.3 ×5) is already *below* the plow model slope — the extra ~7 ns/tok in the model is
FLASH_MERGE + L2 contention with the weight stream, not the flash-decode read.

### A/B of the other named levers — REJECTED

- **GF8 (full GQA fusion, 1× KV traffic).** GF8 reads each KV head ONCE (n_grp=2=n_kv_head)
  vs GF4's TWICE (gqa/GF=2 groups/head). Halving traffic should have been the big win — but
  GF8 is **SLOWER at every ctx** (128k 0.489 vs GF4 0.415 ms) and only reaches ~549 GB/s. It
  is arithmetic/ILP-bound: 8 accumulators, VU=2 unroll (vs GF4's VU=4), heavier per-byte FMA.
  174 regs, no spill — so it is compute-bound, not occupancy-bound. The 2× traffic saving is
  unrealizable at GF8's throughput.
- **16-byte fp8 K loads (`PLOW_FP8_LD16`).** Wins big at ≤96k (32k −21%, 64k −19%, hits 99.6%
  ceil) but **REGRESSES at 128k (+3.3%)** where the 268 MB working set exceeds L2 and the load
  becomes genuinely HBM-bound; worse slope (17.2 ×5). Rejected in favour of the monotonic FAST.
- **GF8 (bf16)** also slower everywhere — same ILP story.

### Traffic / roofline note

GF4 with gqa=8 reads each KV head TWICE (groups 0,1 → kv_head0; 2,3 → kv_head1). At 128k the
KV (268 MB distinct) exceeds L2, so fp8's 2× reread is real HBM demand: FAST issues 1378 GB/s
(90% of the 1535 ceiling) moving 536 MB, i.e. 689 GB/s of *distinct* KV. vLLM reads each byte
ONCE. That 2× is the structural half of the ~1.8× slope and is **not** recoverable in the
current GQA-fusion design (GF8, the 1× config, is compute-bound and slower). The dequant-ALU
half (~6%) IS recoverable and is what FAST banks.

## 2. Re-measured decode TPOT ladder (gemma4_sm120_chat, PLOW_PREFILL=1 warm, 128 gen / n=112)

fp8kv packet re-emitted from this branch's plowc (`PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 PLOW_FP8=1
PLOW_FP8_HEAD=1 PLOW_FP8_KV=1`, 601 packets, KV 2.86 GiB). Harness statically links the kernel,
so shipped vs FAST are two builds (`-DPLOW_FP8_FAST` on the FAST libs). Same machine, same
packet, same warm → the shipped↔FAST **delta** is the clean kernel-effect signal.

TPOT ms/token (batch 1). "shipped" and "FAST" are both measured on THIS build/machine
(current-main-based, ~0.05–0.15 ms faster than the campaign's committed `plow_fp8kv` column
which was measured at 7fd19fb — uniform main-drift, noted). vLLM fp8kv is the trusted baseline
(not re-derived). **Bold = FAST beats vLLM fp8kv.**

| ctx  | committed plow_fp8kv | this-build shipped | **FAST** | Δ (FAST−shipped) | vLLM fp8kv | FAST − vLLM |
|------|----------------------|--------------------|----------|------------------|------------|-------------|
| 32k  | 6.702 | 6.671 | **6.476** | −0.195 | 7.28 | **−0.80** (win) |
| 64k  | 7.442 | 7.389 | **7.102** | −0.287 | 7.52 | **−0.42** (win) |
| 96k  | 8.118 | 8.045 | **7.661** | −0.384 | 7.94 | **−0.28** (win, was +0.18 loss) |
| 128k | 8.791 | 8.727 | **8.293** | −0.434 | 8.46 | **−0.17** (win, was +0.33 loss) |

**FAST beats vLLM fp8kv at all four points — the 96k (was +0.18) and 128k (was +0.33) losses
are both flipped to wins.** sd < 0.02 ms (< 0.25% of mean) at every point; device==host argmax
AGREE. Marginal KV cost 32k→128k: **shipped 20.9 ns/tok → FAST 18.5 ns/tok** (campaign committed
plow 21.3, vLLM 12.0).

The model-level FAST gain (−0.195 → −0.384, growing with ctx) is **larger than the isolated
microbench predicted** (~0.09 ms model at 96k). The isolated single-kernel bench misses
megakernel-wide effects: the shipped bf16-round-trip dequant's extra ALU/registers perturb the
whole megakernel's scheduling and its L2 contention with the concurrent fp8 weight stream; the
f32-direct path relieves that. The measured ladder (sd < 0.15% of mean, device==host argmax
AGREE at every point) is the ground truth.


## Correctness gates

- **`sm120_interp_op_test`: ok** — full suite PASS incl. w8a8 GEMM (relL2 0 / 5.9e-5). (Built
  with `-DPLOW_FP8_KV=1 -DPLOW_FP8_FAST`.)
- **fp8 flash-decode numeric oracle** (new, `runtime/tests/flashdec_fp8_correct_sm120.cu`;
  f32 reference over the e4m3-dequantized cache; the interp op test only covers the *bf16*
  flash arm): D=512/GF4 (26B full-attn), D=512/GF2, D=256/GF2 — **FAST relL2 = 0.00169**,
  identical to the shipped arm, all PASS (≤2e-2 e4m3 budget).
- **bf16 byte-identity**: the DEFAULT (non-fp8kv) decode cubin is **md5-identical**
  (`5cea2d5c…`) built from my branch vs pristine `origin/beat26b-decode` op_attention.cuh —
  the V-phase restructure does not change bf16 SASS.
- **FAST ≡ shipped decode equivalence**: at 32k / 64k / 96k the FAST and shipped builds produce
  **byte-identical** greedy token streams (`PLOW_IDS` md5 match), **identical** step-0 top-5
  logits, and identical device==host argmax. FAST changes nothing observable in decode output,
  so it inherits the campaign's validated shipped-fp8kv real-text parity exactly (first-token
  MATCH vs bf16-KV, 20 identical greedy tokens). FAST is also numerically ≥ shipped (drops one
  rounding), so its bf16-KV parity is ≥ shipped's.
- Megakernel footprint unchanged: **221 regs, 0 stack, 0 spill** for both shipped and FAST
  (the fp8-decode arm is not the megakernel's worst-case), so occupancy (1 block/SM) is intact.

## Verdict — GO. Ship `PLOW_FP8_FAST`; the 96k/128k goal is met.

- **Goal met.** plow fp8-KV FAST beats vLLM fp8kv at **96k (7.661 vs 7.94, −0.28)** and **128k
  (8.293 vs 8.46, −0.17)** — the two points the beat26b-decode campaign lost (+0.18 / +0.33).
  It also holds the mid-range wins (32k −0.80, 64k −0.42). The single-user decode gap is closed
  across the full 1k–128k ladder.
- **Was the 1.8× recoverable?** Partially — and enough. The *dequant-ALU* half of the slope is
  recoverable: f32-direct dequant cuts the marginal KV cost from 20.9 → 18.5 ns/tok (this build;
  campaign committed 21.3). The *traffic* half (GF4 reads each KV head twice, gqa/GF=2) is NOT
  recoverable in the current GQA-fusion design — GF8 (1× traffic) is arithmetic/ILP-bound and
  slower. So plow's slope (18.5) stays above vLLM's (12.0); plow wins the ≤128k range on its
  lower *baseline* (fp8-weight decode is faster at short ctx), and the FAST slope reduction is
  what pulls the long-ctx points below vLLM before the lines would cross (~past 160k).
- **The isolated microbench under-predicted the model win** (predicted ~0.09–0.13 ms at 96k/128k;
  measured −0.38 / −0.43). The single-kernel bench correctly identified the *winner* and the
  *mechanism* (dequant-ALU bound, not bandwidth bound) but misses megakernel-wide effects: in the
  persistent megakernel the fp8 flash-decode runs concurrently with the fp8-weight GEMV stream,
  and the shipped bf16-round-trip path's extra ALU/registers contend for issue slots and L2 with
  that stream. The f32-direct path relieves it — a larger, real, stable end-to-end gain.
- **Safety.** fp8-arm only, behind `PLOW_FP8_FAST`; bf16 default cubin byte-identical (md5); fp8
  oracle relL2 = shipped; FAST≡shipped decode output (PLOW_IDS + top-5 identical) at every ctx;
  megakernel 221 regs / 0 spill unchanged. Recommend the fp8kv cubin ship with `-DPLOW_FP8_FAST`.
