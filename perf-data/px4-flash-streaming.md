# PX-4 — streaming/restructured FlashAttention for the 128k full-layer tail (rtx-11)

RTX PRO 6000 Blackwell (sm_120a, 188 SMs, L2 128 MiB, 1792 GB/s), CUDA 13.0, 2026-07-21.
Branch `px4-flash-streaming`. Scope: **only the 8 FULL hd512 causal layers** (nh16/nkv1,
O(ctx²)) — the 128k prefill driver. The 40 sliding hd256 layers are window-capped and
left byte-identical.

## What changed

`d_flash_prefill_px4<512,32,16>` in `runtime/nvidia/op_attention.cuh`
(`PLOW_NV_FA_PX4=1` default; hd512 arm only, dispatched from `d_flash_prefill`):

1. **Register softmax fused into the P.V mma A-fragment.** Each P.V warp computes P for
   its own 16 query rows directly in the mma `m16n8k16` A layout (two quad-`shfl` row
   reductions per lane), with `m`/`l`/`corr` register-resident. The T5 `Ps[BQ][BKV]` smem
   buffer, its `ldmatrix`, and one `__syncthreads` are **gone**.
2. **8-warp QK.** The HD-512 contraction splits into two hd halves (`SsA` hd 0..255 /
   `SsB` hd 256..511) summed at the softmax read — halves the dependent-mma-chain depth
   and puts the 4 QK-idle warps of the T5 grid to work (T5's QK used only 4 of 8 warps).
3. **Optional TMA** (`cp.async.bulk` + `mbarrier` per-row KV staging, `PLOW_NV_FA_TMA=1`),
   A/B'd against the default `cp.async.cg`.

ptxas: `k_fa_pre<512,32,16>` **96 regs / 0 spill** (was ~210 in T5), prefill `_pf`
megakernel 240 regs / 0 spill / occ-1, dynamic-smem union **unchanged 79.75 KiB** (hd256
sliding arm dominates the union; the PX4 hd512 arena shrinks to 69.0 KiB).

## Baseline correction (important)

The pre-build roofline (`rtx-11-px4-roofline.md`) quoted **21.9 / 147.2 / 1019.4 ms** as
"the current mma flash op". That is the **PIPE=0 (T4 synchronous-staging)** path, *not* the
shipping default. The shipping default is **PIPE=1 (T5 cp.async KV-stream)**, measured here
at **11.85 / 78.61 / 345.6 ms** — already ~3× faster than the roofline baseline at 128k. All
tables show both; PX-4's honest win is measured vs T5.

## Correctness gates (before any perf claim)

- **Oracle** (`sm120_interp_op_test`, T4-FIXED sensitive f32 ref, masked→0): full
  flash-prefill suite **PASS in both staging arms** — 26 `flash_pre` cases (causal /
  sliding straddle / hd256+512 / fused ns=1 / split ns>1 + merge / ragged / chunked
  q_pos0>0 / soft-softmax) at relL2 1.5e-3…2.4e-3 (same bf16-P band as T5); whole suite
  `ok`; wave64 negctrl still FAILs.
- **Token identity** (gemma-4-12B, PLOW_PREFILL=1 vs decode-only reference): first-gen
  token **AGREES at ctx 256 / 4k / 32k / 128k (all 236770)**. PX4 prefill step-0 top-5
  logits are **bit-identical** to the T5 baseline prefill. device==host argmax AGREE.

## Microbench — hd512 FULL op (fa_pv_bench, grid 188/256, seq_q=8192 tail chunk, ms/op)

| seq_kv | T4 (roofline base) | T5 (shipping) | **PX4** | PX4 vs T5 | PX4 vs T4 | PX4-TMA |
|--------|--------------------|---------------|---------|-----------|-----------|---------|
| 8k     | 21.95              | 11.85         | **9.08**   | **−23.4%** | −58.6% | 20.73 |
| 32k    | 147.29             | 78.61         | **59.68**  | **−24.1%** | −59.5% | 138.68 |
| 128k   | 1020.6             | 345.6         | **261.96** | **−24.2%** | −74.3% | 610.59 |

## Achieved DRAM BW @128k (derived — ncu unavailable, ERR_NVGPUCTRPERM)

ncu is blocked in this container (`RmProfilingAdminOnly=1`). BW derived from bytes/time:
`bytes = 68.99 GB` (= roofline FA-re-read floor 38.5 ms × 1792 GB/s), floors from the roofline.

| variant | ms | BW GB/s | % of 1792 | × off 38.5 ms FA floor | × off 139.7 ms compute floor |
|---------|-----|---------|-----------|------------------------|------------------------------|
| T4 (roofline base) | 1020.6 | 67.6  | 3.8%  | 26.5× | 7.31× |
| T5 (shipping)      | 345.6  | 199.6 | 11.1% | 8.98× | 2.47× |
| **PX4**            | 261.96 | **263.4** | **14.7%** | **6.80×** | **1.87×** |

PX4 closes the 128k gap to the FA-re-read floor from T4's 26.5× to **6.8×**, and to the
compute floor from 7.3× to **1.87×**.

## Phase ablation (px4_fa_ablate.cu — the ncu substitute), hd512 @128k, full = 329.99 ms

| phase | cost ms | % |
|-------|---------|---|
| softmax (T5 smem-reduce path)   | 118.0 | 36% |
| QK (4/8 warps active)           | 70.3  | 21% |
| cp.async staging exposure       | 54.0  | 16% |
| P.V                             | 18.8  | 6%  |
| loop + barrier floor            | 10.3  | 3%  |

**cp.async-issue-removal saves only 16%, and per-tile time is flat 8k→128k ⇒ the hd512
flash is NOT DRAM-bandwidth-bound; it is compute/latency-bound (softmax + QK).** This is
why PX-4 attacked the compute geometry and why TMA (a pure streaming lever) is a negative.

## End-to-end prefill (gemma-4-12B, PLOW_PREFILL=1, batch1, greedy)

| ctx  | baseline T5 ms | **PX4 ms** | Δ | first tok |
|------|----------------|------------|-----|-----------|
| 4k   | 513.2          | 505.9      | −1.4%  | 236770 |
| 32k  | 5050.4         | 4677.1     | −7.4%  | 236770 |
| 128k | 37363.5        | **31511.3**| **−15.7%** | 236770 |

Win grows with ctx as the 8 full-layer O(ctx²) flash share rises; at 128k it saves **5.85 s**
of a 37.4 s prefill. Only the 8 full layers changed, so the op-level −24% dilutes to −15.7%
e2e. Baseline 37363 ms reproduces the committed T5 e2e (37223 ms @131000).

## Honest negatives

- **TMA (cp.async.bulk + mbarrier) is ~2× SLOWER** than cp.async.cg at every ctx
  (20.7/138.7/610.6 vs 9.08/59.68/261.96 ms). sm_120 single-CTA TMA issues one bulk copy
  per contiguous row from ONE thread (no `.tensor` 2D tiling to batch the 16 BKV rows),
  serializing 16 per-row descriptors, while cp.async.cg spreads 16B lines across all 256
  threads. Matches rtx-05 (`gemv_transport`: TMA refuted for small/1-D streams). Kept
  opt-in behind `PLOW_NV_FA_TMA=0`.
- **The bandwidth framing was wrong for this kernel.** The roofline treated PX-4 partly as
  a KV-streaming problem; the ablation disproves that (compute/latency-bound). The measured
  win came from the compute-geometry restructure, not from streaming.
