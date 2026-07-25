# occ-1 register-deep-prefetch GEMM PoC — VERDICT: **NO-GO**

Standalone bounded PoC (branch `gemm-occ1-poc`, MI350X/gfx950). Tests the single hypothesis the
prefill lever hinged on: does an **occ-1 (4-wave, 512-reg) GEMM with REGISTER-deep prefetch** break
the LDS-read→MFMA dependency chain and lift MfmaUtil above the bf16 ~30%?

**It does not.** The multi-week production rewrite (segmented 4-wave GEMM object, dispatch, tile
geometry) is **NOT worth it as specified**. The read→MFMA-dependency diagnosis is *incomplete*; the
real cap is the **MFMA accumulator-latency chain**, which occ-1 operand prefetch cannot touch.

## What was built
`gemm_occ1_poc.hip` — standalone 4-wave (256-thread, 2×2 grid), `__launch_bounds__(256,1)` ⇒ occ-1,
512-VGPR budget. LDS K-tile double-buffered; inside each tile the BK/16 MFMA-K slices are contracted
through a **register ring of depth D**: all D future slices' A/B fragments are read into distinct
registers before the MFMA that consumes slice s fires — **no s_barrier inside the slice loop**. Depth
swept 2/3/4 (=full-tile decoupling). Correctness spot-checked vs f64 CPU ref (worst rel ≤ 0.004).
Driver `gemm_occ1_bench.c`; MfmaUtil via `rocprofv3 --pmc MfmaUtil MemUnitStalled LdsBankConflict`.

## The decisive numbers (M=4096, Qwen3-4B prefill shapes; peak 1660 TF/s)

MfmaUtil is measured **under rocprof** (which lowers absolute values but is internally consistent —
every row shares identical profiler conditions). TF/s is un-profiled.

| config | tile | waves/occ | q_proj TF/s · MfmaUtil | gate/up TF/s · MfmaUtil | down TF/s · MfmaUtil |
|---|---|---|---|---|---|
| occ1 **d2** | 128×256 | 4 / **occ-1** | 545 · 24.0% | 580 · 25.4% | 467 · 19.7% |
| occ1 **d3** | 128×256 | 4 / occ-1 | 543 · 23.6% | 571 · 25.2% | 460 · 19.3% |
| occ1 **d4** | 128×256 | 4 / occ-1 | 548 · 24.2% | 580 · 25.4% | 467 · 19.5% |
| occ1 d3 | 192×256 | 4 / occ-1 | 460 · 19.7% | 571 · 24.5% | 645 · 28.8% |
| occ1 d2 | 256×256 | 4 / occ-1 | 615 · 27.0% | 585 · 25.3% | 557 · 23.3% |
| occ1 d3 | 256×256 | 4 / occ-1 | **625** · 26.9% | 593 · 25.4% | 562 · 23.5% |
| occ1 d4 | 128×128 | 4 / occ-1 | 478 · 20.7% | 505 · 21.3% | 495 · 21.1% |
| **bf16 gemm_c5** (prod) | 192×256 | 8 / occ-2 | 587 · 26.1% | **716 · 34.1%** | **806 · 39.9%** |
| bf16 gemm_c0 | 256×256 | 8 / occ-2 | 525 · 23.4% | 523 · 22.7% | 531 · 22.6% |
| bf16 gemm_c3 | 128×128 | 8 / occ-2 | 497 · 21.2% | 511 · 21.8% | 498 · 20.7% |
| **hipBLASLt** (given) | — | — | 2612 | 2329 | 2779 |

Register/occupancy table (compile, `-Rpass-analysis=kernel-resource-usage`, all occ-1, 512 budget):

| config | VGPR | AGPR | total | occ | spill | LDS |
|---|---|---|---|---|---|---|
| occ1 128×256 d2/d3/d4 | 188–190 | 128 | 316–318 | 1 | 0 | 108 KiB |
| occ1 192×256 d3 | 254 | 192 | 446 | 1 | 0 | 118 KiB |
| occ1 256×256 d2/d3 | 256 | 256 | **512** | 1 | 8–16 | 144 KiB |
| occ1 128×128 bk128 d3/d4/d6 | 198 | 64 | 262 | 1 | 0 | 136 KiB |

## Why it is a NO-GO (the four load-bearing facts)

1. **Prefetch depth is completely INERT.** d2 ≡ d3 ≡ d4 in both MfmaUtil and TF/s, at every tile
   (e.g. 128×256: 24.0/23.6/24.2%). Reading operands 2–4 slices ahead of the MFMA changes nothing ⇒
   the LDS-read→MFMA *operand* path was never the bottleneck. The hypothesis is directly refuted.
2. **Not memory, not bank conflicts.** MemUnitStalled ≈ 0.0–0.2%, LdsBankConflict ≈ 0 across every
   occ-1 config. (Also refutes the "12.7% bank-conflict" suspicion for this kernel.)
3. **The ONLY lever that moved MfmaUtil is accumulator width** (= # independent MFMA chains per
   wave): 128×128 (4 acc) ~21% → 128×256 (8 acc) ~24% → 256×256 (16 acc) ~27%. Each MFMA feeds the
   next MFMA to the *same* accumulator (a RAW chain); utilization is capped by how many independent
   accumulator chains are in flight to hide the ~fixed MFMA execution latency. This is
   **MFMA-issue/accumulator-latency bound**, which operand prefetch cannot address.
4. **occ-1's 512 registers are consumed by the ACCUMULATOR, not free for prefetch.** 256×256 (16 acc)
   already hits 512 and spills; you cannot widen further at occ-1. And it plateaus at ~27% / 37% peak
   — *below* the production 8-wave 192×256 tile (gemm_c5: 34–40% MfmaUtil, 716/806 TF/s), because
   8 waves × 12 acc puts more total in-flight MFMA chains on each SIMD than one 512-reg wave can.

Net: best occ-1 (256×256 d3) = 625/593/562 TF/s vs production gemm_c5 = 587/**716**/**806**. occ-1
wins only q_proj; loses gate/up and down; **both are ~4× short of hipBLASLt**. Step 4 (fp8 on top) was
gated on bf16 lifting MfmaUtil — it did not, so fp8 layering was not pursued (it would ride the same
capped pipeline; recall fp8-gemm already measured fp8 K64 *dropping* MfmaUtil to 15%).

## What actually caps it / where the library's 50–90% comes from
MFMA execution-latency hiding needs **many independent MFMA accumulator chains resident at once** =
(waves/SIMD) × (accumulators/wave), bounded by the register file — *plus* the sustained power/clock
envelope, *plus* fp8's wide-K `mfma_scale_f32_32x32x64_f8f6f4`. Fewer waves (occ-1) REDUCES the chain
count and pushed MfmaUtil the wrong way at matched tiles. The productive levers are the opposite of
this PoC's premise: keep/raise occupancy, maximize accumulator tile within the register file, and lean
on fp8's 2× wide-K instruction — none of which require the segmented 4-wave rewrite.

Raw per-run rocprof CSVs: `build_occ1/occ1_results/` (`final_summary.csv`, `throughput.csv`,
`summary.csv`, and `*_s{0,3,4}/*_counter_collection.csv`).
