# rtx-19 E4 — tensor-core fp8 (w8a8) batched-decode weight GEMM vs FFMA fp8 GEMV

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120a, 188 SM, 100 KB smem/SM).
Harness: `runtime/tests/e4_tc_fp8_decode_sm120.cu` (built system nvcc 13.0, `-arch=sm_120a -DPLOW_NV_W8A8=1`).
Raw JSON: `perf-data/rtx19-e4-tc-fp8-decode.json`. Every GPU run via `gpulease e4-tcfp8`.

## What this measures

The current fp8 decode weight op is a **w8a16 FFMA GEMV** (`op_gemm.cuh` `gemv_rows_fp8`/`dot8_fp8`:
fp8 weight, bf16 activation, per-col scale). `gemv_walk` (GV_MM_MAX=8) re-reads the weight
`ceil(M/8)×` for a decode batch M — so a B=32 step streams every weight **four times**.

The E4 twin — `k_tc_fp8<BM,BN,NW,STAGES>` — is a **w8a8 tensor-core GEMM**: PX-2's
`mma.sync.m16n8k32.e4m3` mainloop (both operands e4m3, `pgm_mma_fp8_k32` + the `pgm_sw8` fp8
swizzle from `d_gemm_w8a8`) + ZG-0's **split-K** skinny-N tiling (job = N-tile × K-slice, raw f32
partials `atomicAdd`'d into a global buffer, the two scales applied ONCE in a finalize pass so
split-K stays exact) + the two-scale epilogue `acc·a_scale[m]·w_scale[n]`. It streams the weight
**once** regardless of M. Config: M≤16 → BM16/BN128/8warp/S4; M≤32 → BM32/BN64/4warp/S4; split-K
chosen so N-tiles×ksplit ≥ 4·SM.

GB/s convention (ZG-0's): **logical single-weight-read = N·K·1 byte / time**. TC reads the weight
once so its logical GB/s stays near the wall; FFMA re-reads `ceil(M/8)×` so its logical GB/s
collapses with M. The crossover is the ratio **TC/FFMA**.

## Correctness gate

- **mma gate** (the real one): TC-fp8(w8a8) vs its own from-quantized-operands f32 w8a8 oracle —
  isolates the tensor-core mma + split-K + two-scale epilogue. Threshold **0.05** (ZG-0's proven
  bf16-GEMM max-abs/RMS bound; e4m3 is coarser so 0.05 is apt). **PASS on all 48 shape×M cells.**
  At M=1 the error is **exactly 0.0** (split-K atomicAdd+finalize reproduces the oracle bit-for-bit);
  it rises to ≤2.0e-2 only at M=32 on the largest-K 31B shapes — pure f32-accumulation-order
  divergence between the mma tree and the linear oracle, well inside the e4m3 budget.
- **twin fidelity** (informational, not a hard gate): TC-fp8(w8a8) vs the current w8a16 GEMV.
  Max-element delta ≈ 0.09–0.13 on synthetic uniform ±0.25 data — this is the inherent
  **activation-requant** difference (w8a8 quantizes the activation to e4m3; w8a16 keeps it bf16),
  exactly the "within the w8a8 e4m3 tolerance" the plan calls for. Real token identity must be
  judged on a serving run (see Integration).

## ptxas (sm_120a, `-DPLOW_NV_W8A8=1`)

| kernel | registers | spill |
|---|---|---|
| `k_tc_fp8<16,128,8,4>` (M≤16) | 54 | 0 B |
| `k_tc_fp8<32,64,4,4>`  (M≤32) | 77 | 0 B |

No spills, no stack frame. Default byte-identical: the new kernel lives entirely in the test TU;
`op_gemm.cuh` and the decode megakernel are **untouched**, so the FFMA decode path is unchanged.

## Per-op GB/s — Gemma-4-12B (clean signal)

fp8 cold-read ceilings (all-SM streaming read of THIS tensor): qkv 853, o_proj 790, gate/up 958,
down 953 GB/s. (Achievable fp8-byte bandwidth is ~55–62% of the 1535 GB/s bf16 wall — the fp8
weights are half the bytes but the streaming read tops out lower.)

| shape (N×K) | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| **qkv** 8192×3840 — TC / FFMA GB/s | 640 / 627 | 614 / 615 | 614 / 512 | 591 / 334 | 615 / 201 |
| &nbsp;&nbsp;TC/FFMA | 1.02× | 1.00× | **1.20×** | **1.77×** | **3.06×** |
| **o_proj** 3840×4096 — TC / FFMA | 512 / 466 | 513 / 439 | 496 / 349 | 480 / 202 | 480 / 114 |
| &nbsp;&nbsp;TC/FFMA | **1.10×** | **1.17×** | **1.42×** | **2.37×** | **4.22×** |
| **gate/up** 15360×3840 — TC / FFMA | 800 / 823 | 800 / 779 | 800 / 694 | 779 / 440 | 720 / 263 |
| &nbsp;&nbsp;TC/FFMA | 0.97× | 1.03× | **1.15×** | **1.77×** | **2.74×** |
| **down** 3840×15360 — TC / FFMA | 758 / 640 | 748 / 524 | 748 / 395 | 739 / 228 | 703 / 124 |
| &nbsp;&nbsp;TC/FFMA | **1.18×** | **1.43×** | **1.90×** | **3.24×** | **5.65×** |

## Per-op GB/s — Gemma-4-31B (fp8 headline)

fp8 cold-read ceilings: qkv 1001, o_proj 878, gate/up 1027, down 1017 GB/s.

| shape (N×K) | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| **qkv** 16384×5376 — TC / FFMA | 804 / 915 | 796 / 878 | 789 / 748 | 782 / 468 | 694 / 267 |
| &nbsp;&nbsp;TC/FFMA | 0.88× | 0.91× | **1.06×** | **1.67×** | **2.60×** |
| **o_proj** 5376×8192 — TC / FFMA | 768 / 624 | 742 / 598 | 717 / 478 | 717 / 287 | 633 / 163 |
| &nbsp;&nbsp;TC/FFMA | **1.23×** | **1.24×** | **1.50×** | **2.50×** | **3.88×** |
| **gate/up** 21504×5376 — TC / FFMA | 801 / 949 | 801 / 911 | 790 / 748 | 773 / 453 | 724 / 256 |
| &nbsp;&nbsp;TC/FFMA | 0.84× | 0.88× | **1.06×** | **1.71×** | **2.83×** |
| **down** 5376×21504 — TC / FFMA | 784 / 784 | 773 / 701 | 758 / 520 | 763 / 292 | 693 / 149 |
| &nbsp;&nbsp;TC/FFMA | 1.00× | **1.10×** | **1.46×** | **2.61×** | **4.64×** |

## Aggregate decode weight-GEMM throughput (all 5 projections/layer)

Decode is HBM-weight-bandwidth-bound, so the per-token weight-GEMM time ≈ Σ(bytes/GB·s) over
{qkv, o, gate, up, down}. Speedup = FFMA time / TC time (batch-amortized; the fixed ~2 ms/token
intercept and attention are unchanged and not counted here):

| batch B | 12B TC/FFMA | 31B TC/FFMA |
|---|---|---|
| 1  | 1.05× | 0.93× |
| 8  | **1.38×** | **1.20×** |
| 32 | **3.67×** | **3.34×** |

## Crossover verdict

**TC-fp8 widens the batched-decode lead, and the lead grows monotonically with B.**

- **From B=1** on the skinny-N ops — 12B o_proj (1.10×) & down (1.18×), 31B o_proj (1.23×) —
  where FFMA starves the 188 SMs (few N-tiles) and split-K floods them.
- **From B=8** on the wide-N ops (qkv, gate/up) on both models (1.06–1.20×); below that, at B=1–4,
  the current fp8 FFMA GEMV is at parity or up to ~16% ahead on those shapes (see negatives).
- The lead then grows to **2.6–5.65× at B=32** on every shape, because FFMA re-reads the weight
  `ceil(B/8)×` while TC streams it once. Aggregate weight-GEMM: **3.67× (12B) / 3.34× (31B) at B=32.**

This is the multi-user decode win: exactly where vLLM batches (B≥8), TC-fp8 pulls decisively ahead.

## Honest negatives

1. **B=1–4 wide-N is NOT a TC win.** On qkv & gate/up, at B≤4 the current fp8 FFMA GEMV ties or
   beats TC-fp8 (31B gate/up B=1 = 0.84×). Those shapes already give FFMA enough N-parallelism, and
   the fp8 `dot8_fp8` FFMA is efficient at M=1. So the aggregate at **31B B=1 is 0.93× (a small
   regression)**; TC-fp8 crosses over by B≈4–8. The unqualified ZG-0 "beats FFMA from batch 1"
   holds for bf16 but for **fp8 it is shape-dependent**: from B=1 on skinny-N, from B=8 on wide-N.
2. **TC-fp8 is not bandwidth-saturating at these fp8 decode shapes.** It reaches only ~40–52% of the
   1535 GB/s bf16 wall = ~75–80% of the fp8 cold-read ceiling (e.g. 12B qkv TC 614 vs 853 cold-read).
   Unlike ZG-0's bf16 result (94–96% of ceiling), the fp8 small-M path leaves ~20–25% headroom —
   a config/pipeline tuning opportunity (deeper stages, BK, ksplit) not yet captured. The crossover
   vs FFMA is already decisive regardless.
3. **Live serving tok/s + token identity are follow-on** (see Integration) — the numbers above are
   the kernel-level throughput proxy (the same methodology as `runtime/tests/gemv_batch_sm120.cu`,
   "tokens/s-equivalent … with no scheduler in the way").

## Integration note (why not yet wired into the live megakernel)

The decode interpreter is a **single persistent megakernel** launch. Split-K accumulation across
blocks needs a **global f32 partials buffer + a separate finalize/scale pass** (blocks owning
different K-slices of the same output can't share registers) — i.e. a two-pass decode GEMM op,
which the single-launch design deliberately avoids. Wiring TC-fp8 into serving therefore also needs
the decode program to emit a per-op fp8 **activation quant** (`d_quant_fp8`) + the per-row a_scale /
per-col w_scale tensors before each weight op (the model is currently driven w8a16, activation bf16).
Both are additive serving-layer changes; behind the default-off flag the megakernel stays
byte-identical. The kernel, the correctness gate, and the crossover data here are the prerequisite
proof that the op is worth wiring.
