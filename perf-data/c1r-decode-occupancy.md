# C-1R — decode-architecture reopen: occupancy lever, warp-spec control, and the cp.async win

Date 2026-07-22. Branch `c1r-decode-occupancy` (based on `c1-decode-splitzip`).
Model gemma-4-12B, RTX PRO 6000 Blackwell (188 SM, 1535 GB/s achievable).
Plan: `plans/p9-lossless-compression.md` §C-1R. All kernels **bit-exact** vs bf16 GEMV
(reused the C-1 device oracle: sz/ws/cp output == bf16 output, byte-identical).

## TL;DR — C-1 is UN-KILLED for the B=1 decode path

C-1 killed SplitZip decode because the **naive** per-lane in-register reconstruct kernel
(`gemv_rows_sz`) is 0.62–0.77× (a regression) at the real megakernel geometry (GRID=188,
1 block/SM), and only won under ≥5 blocks/SM oversubscription. C-1's stated reopen
criterion was "move the GEMV to oversubscribed standalone launches." **That criterion is
superseded by a better mechanism found here.**

- **H2b (cp.async inline-smem decompress — the user-proposed design): SURPRISE WIN.**
  Staging the compressed bytes (lo 8 B + cd 4 B/lane/chunk) global→smem via a `cp.async`
  software pipeline and reconstructing **inline from smem, with ALL 8 warps still doing
  FMA (no warp specialization)**, wins **1.25–1.30× at B=1 at GRID=188 — the real
  1-block/SM megakernel geometry — bit-exact**. This directly refutes C-1's "needs ≥5
  blocks/SM." The naive kernel was **load/issue-latency bound** at 1/SM (900–1130 logical
  GB/s ≈ 680–850 GB/s of actual compressed bytes, far under the 1466 GB/s bf16 wall), NOT
  occupancy bound. `cp.async` hides that latency **without any occupancy change**, reaching
  1850–1865 logical GB/s (≈1400 GB/s actual, i.e. the compressed stream now saturates HBM).
- **H1 (occupancy / oversubscription): REAL but inferior and unnecessary.** The naive
  kernel does cross to a win under oversubscription (≈1.13× at 4 blocks/SM, B=1), but that
  path requires **de-fusing** the GEMVs into standalone launches (launch overhead), and it
  under-performs the in-place cp.async kernel. Occupancy is a lever; it is not THE lever and
  it is not needed.
- **H2 (warp-spec producer/consumer): confirmed NO-WIN (loud kill).** Dedicating 4 warps to
  reconstruct and 4 to FMA **halves FMA throughput** and loses 0.32–0.77× — worse than even
  naive-sz. As predicted: warp-spec adds no occupancy and no BW shadow.

**Verdict: KEEP (reopen succeeds) for B=1 via the cp.async kernel.** Projected decode TPOT
18.5 → ~16.6 ms (**−10%**) at B=1, bit-exact, as a **drop-in megakernel kernel swap** (stays
at 1 block/SM, **no launch overhead**). The one shape that is currently neutral (`down`,
K=15360) caps the win; fixing it would push toward −17% (the plan's original C-1 estimate).

## Setup / faithfulness

Same rigor as C-1's kill: the A/B launches the SAME kernels with the interpreter's own
geometry (`slice=blockIdx, nblk=gridDim`), real gemma-4-12B weight bytes (100 MB sample of
real layer-0 weights, `best_base=109` cov 99.971% — matches EXP_BASE=109), 12B decode shapes.
speedup = bf16 time ÷ variant time; logical GB/s = uncompressed bytes ÷ time. Harnesses:
`runtime/tests/sz_batch_sm120.cu` (H1 grid sweep, from C-1), `runtime/tests/sz_ws_sm120.cu`
(H2 warp-spec + H2b cp.async, new), `runtime/tests/launch_ovh_sm120.cu` (H1 launch
accounting, new). Raw: `perf-data/c1r-decode-occupancy-raw.txt`.

## H1 — occupancy sweep: naive `gemv_rows_sz` speedup vs bf16, by blocks/SM (B=1 / MM=1)

| shape (K)        | 1/SM (188) | 2/SM (376) | 4/SM (752) | 5/SM (940) | 8/SM (1504) | 10/SM (1880) |
|------------------|-----------:|-----------:|-----------:|-----------:|------------:|-------------:|
| qkv    (K3840)   | 0.767×     | 1.109×     | **1.137×** | 1.125×     | 1.027×      | 1.075×       |
| o_proj (K4096)   | 0.771×     | 1.093×     | **1.126×** | 1.111×     | 1.028×      | 1.045×       |
| gate/up (K3840)  | 0.753×     | 1.117×     | **1.152×** | 1.108×     | 1.039×      | 1.060×       |
| down   (K15360)  | 0.619×     | 0.952×     | **1.132×** | 1.028×     | 0.890×      | 0.984×       |

Crossover (naive kernel, B=1): K≤4096 shapes cross to a win at **2 blocks/SM**; `down`
(K=15360) needs **4 blocks/SM**. Optimum is **~4 blocks/SM** (GRID≈752), NOT the 5–10× the
task hypothesized — beyond ~5/SM the per-block slice gets too fine (per=3–4 rows, idle
blocks) and the win erodes. Peak naive-kernel win ≈ **1.13–1.15×** (real% ≈ 85–87% of the
1.33× ratio). (MM≥8 rungs are noisier because bf16 itself becomes compute/ILP-bound off the
BW wall; see raw file.)

### H1(b) — launch-overhead accounting (does oversubscription survive net end-to-end?)

Measured per-op kernel-launch cost on this box (`launch_ovh_sm120.cu`):

| mode | us/op |
|------|------:|
| noop back-to-back GRID=188 | 2.05 |
| noop back-to-back GRID=940 | 2.04 |
| tiny dependent (RAW) GRID=940 | 2.05 |
| host enqueue only (`cudaLaunchKernel`) | 1.94 |
| **CUDA-graph replay, 200-chain, GRID=940** | **0.88** |

Weight-GEMV launches/token to de-fuse = 4/layer (qkv, o, gate/up, down) × 48 layers + lm_head
= **193**; a full eager de-fuse of the megakernel ≈ **~485** ops/token.

Added launch overhead/token: 193 × 0.88 µs = **0.17 ms** (graph) … 485 × 2.05 µs = **0.99 ms**
(eager full de-fuse). Weight-pass saving at the 4/SM naive optimum (1.14× on the 15.6 ms bf16
weight pass) ≈ **1.9 ms**. So the oversubscription path **would** be a small net win
(≈ −5..−9% TPOT B=1) **because launch overhead (0.17–0.99 ms) ≪ weight-pass saving (1.9 ms)**
— a PROJECTION (no hybrid built). **But it is moot: the cp.async kernel below wins more,
in-place, at 1 block/SM, with zero launch overhead.**

## H2 + H2b — at the REAL 1-block/SM geometry (GRID=188), B=1 / MM=1

| shape (K)       | bf16 GB/s | naive-sz | warp-spec (H2) | **cp.async (H2b)** |
|-----------------|----------:|---------:|---------------:|-------------------:|
| qkv    (K3840)  | 1468      | 0.762×   | 0.751×         | **1.271×** (1865 GB/s) |
| o_proj (K4096)  | 1426      | 0.760×   | 0.769×         | **1.296×** (1849 GB/s) |
| gate/up (K3840) | 1483      | 0.760×   | 0.756×         | **1.249×** (1852 GB/s) |
| down   (K15360) | 1465      | 0.617×   | 0.578×         | 0.977× (1431 GB/s) |

Reproduced across two independent runs (150-iter confirm) — the 1.25–1.30× on the K≤4096
shapes is stable, not noise. All variants **bit-exact** (0 mismatching outputs).

- **H2 warp-spec (producer/consumer, 4 prod + 4 cons):** loses everywhere, and gets
  **catastrophically worse with batch** (MM=8: 0.32×, MM=16: 0.54×) because it only has 4
  FMA-capable warps — halving the compute the batched rungs need. Confirmed NO-WIN.
- **H2b cp.async cooperative:** wins on the three K≤4096 shapes at 1/SM; `down` (K=15360) is
  neutral (0.977×, no regression) and is **not** fixed by deeper pipeline (CP_D=8 tested,
  identical) — its neutrality is shape-structural, a tuning target. **cp only wins at MM=1
  (B=1)**; at MM≥4 it loses (the kernel is compute-bound there and the smem staging adds
  latency in the critical path). So H2b resurrects the **B=1 accuracy row**, not the
  multi-user (B≥4) throughput thesis.

At **4 blocks/SM** (GRID=752) cp.async reaches **1.41–1.55×** at B=1 (near-full ratio
realization) — but oversubscription is unnecessary given the 1/SM win.

## ptxas (standalone microbench kernels, sm_120a, 0 spill everywhere)

| variant | MM=1 | MM=4 | MM=8 | MM=16 |
|---------|-----:|-----:|-----:|------:|
| bf16 `gemv_rows`      | 80 | 89 | 114 | 98 |
| naive-sz `gemv_rows_sz` | 40 | 55 | 61 | 72 |
| warp-spec (H2)        | 64 | 64 | 64 | 92 |
| **cp.async (H2b)**    | **64** | **64** | **71** | **80** |

cp.async needs **fewer registers than bf16** (64 vs 80 at MM=1) — the reconstruct temporaries
live in the 12 KB smem ring instead of registers — plus 12 KB dynamic smem (trivial at 1
block/SM: 100 KB smem/SM available). No 255-cliff risk. (C-1's fused-megakernel counts were
210/242 regs; the cp path would not raise them.)

## End-to-end projection (B=1), and its assumptions (PROJECTION, not measured)

The right way to realize sz at B=1 is **not** H1's oversubscription (which needs de-fusing
into standalone launches). It is to **swap the naive `gemv_rows_sz` for the cp.async
inline-smem kernel inside the existing 1-block/SM megakernel** — no de-fuse, no launch
overhead, no occupancy change.

- bf16 decode TPOT @4k = 18.5 ms; weight-stream share 84% = **15.6 ms** (§A measured basis).
- cp.async byte-weighted weight-pass speedup at B=1 (per-layer elems: qkv 23.6M @1.27×,
  o 15.7M @1.30×, gate/up 59.0M @1.25×, down 59.0M @0.98×) = **≈1.14×** → 15.6 → 13.7 ms,
  save **1.9 ms**.
- Projected TPOT 18.5 → **~16.6 ms (−10%)** at B=1, bit-exact, in-place (no launch overhead).
- If `down` is fixed to ~1.25× (currently the cap), weight-pass ≈1.26× → TPOT ~15.3 ms
  (**−17%**), matching the plan's original C-1 estimate of ~15.0 ms.

Assumptions / residual risk (honest): (1) the standalone cp measurement is at the megakernel's
exact geometry and with HBM-resident weights, faithful to how the megakernel runs each GEMV —
but it is a **per-op** number; the full e2e needs the sz weight **emitter + on-load encoder**
that C-1 explicitly did not build (out of scope here). (2) Integration cost: the megakernel
must call the cp-staged GEMV and reserve 12 KB smem for the ring (fits at 1 block/SM).
(3) The win is **B=1 only**; batched serving (B≥4) is not helped by cp (compute-bound) — for
B≥4 the only positive lever is naive-sz oversubscription (H1), which is marginal and noisy.

## Verdicts

- **H1 (occupancy lever): CONFIRMED real but INFERIOR + UNNECESSARY.** Naive kernel crosses to
  ~1.13× at 4 blocks/SM (B=1); net of measured launch overhead (0.88 µs graph / 2.05 µs eager
  per op) the projected e2e is a small win — but the cp.async kernel wins more, in-place, at 1
  block/SM, so oversubscription is not the path to take.
- **H2 (warp-spec producer/consumer): KILL — confirmed no-win**, strictly worse than naive-sz
  (0.32–0.77×), bit-exact. Halving FMA warps is the failure mode.
- **H2b (cp.async cooperative inline-smem decompress): KEEP — the win.** Bit-exact 1.25–1.30×
  at B=1 at the real 1-block/SM geometry; ptxas 64 regs / 0 spill / +12 KB smem.

**Updated reopen criterion (supersedes C-1's):** C-1's "needs ≥5 blocks/SM oversubscription"
is **wrong** — it measured only the naive per-lane reconstruct, which was latency-bound, not
occupancy-bound. **SplitZip decode is a win at B=1 at 1 block/SM when the compressed operand
is `cp.async`-staged to smem and decompressed inline (all warps still FMA).** Next step to
close the reopen: build the plowc sz weight emitter + on-load encoder (C-1's unbuilt S3) and
wire `k_szcp` into `interp_sm120.cu`'s GEMV arms; measure the e2e TPOT ladder B=1 and the
per-tensor `down` tuning / opt-out. Do NOT pursue warp specialization or oversubscribed
standalone launches — both are dominated by the in-place cp.async kernel.

## Honest negatives / not done

- **No e2e TPOT/serving measurement.** Requires the sz weight emitter + on-load host encoder
  (C-1's unbuilt S3) + megakernel wiring; the e2e number here is a labeled PROJECTION.
- **cp.async wins B=1 only.** At B≥4 (MM≥4) it loses (compute-bound regime). The multi-user
  throughput thesis is NOT resurrected — only the B=1 bit-exact accuracy row.
- **`down` (K=15360) neutral (0.977×)** — not fixed by pipeline depth; needs its own tuning
  or a per-tensor bf16 opt-out (plan S5).
- **H1 oversubscription optimum is ~4 blocks/SM, not 5–10×** as hypothesized; beyond ~5/SM the
  win erodes.
