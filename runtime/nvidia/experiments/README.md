# sm_120 experiment harnesses

Standalone measurement code from the RTX/sm_120 campaign. **Not part of the
production build** — each file is compiled on its own, e.g.

    nvcc -arch=sm_120a -O2 -o /tmp/x setmaxnreg_probe.cu && /tmp/x

Kept because each one answers a question that cost real time to settle, and the
answer is only trustworthy because the harness can be re-run.

| file | question it answers | result |
|---|---|---|
| `setmaxnreg_{probe,occ,design}.cu` | Does `setmaxnreg` execute on sm_120a? | YES, but needs `__maxnreg__` (NOT `__launch_bounds__`, which makes ptxas silently drop it). `dec` returns registers to a **CTA-local** pool, so it does NOT raise blocks/SM — measured flat at 2.00. |
| `gemv_transport.cu` | TMA vs `cp.async.cg` vs `__ldg` for decode weight streaming | TMA is **real** (SASS `UTMALDG.2D`/`UBLKCP.S.G`, identical to sm_90a) but the **slowest of 37 configs**. `cp.async.cg` wins. Refuted for 1-D streaming. |
| `kv_l2.cu` | Does `cudaAccessPolicyWindow` on KV help? | +13-21% on the window, but it fits ~400 tokens of 36-layer KV ⇒ <0.02 ms/tok. Effectively nil. |
| `warpspec_ab.cu` | Does warp specialization pay on decode GEMV? | NO. Costs +5..+24 regs and 4-33 KB smem, cuts occupancy, never beat uniform except +1.1% on `down` (within noise). |
| `fp8_{probe,verify,gemv}.cu` | fp8 e4m3 fragment layout + GEMV throughput | Layout derived by one-hot probe, verified **bit-exact**. FFMA dequant-on-load beats `mma.sync` at batch 1: 3.203e-07 vs 2.488e-02 rel err (mma forces fp8 activations). |
| `splitzip_{codec,gemv,bitexact}.cu` | Lossless bf16 exponent compression, decoded into SRAM | **1.300x** fused into the GEMV, bit-exact. Decompressing to HBM (the paper's mode) is refuted at 849 GB/s. **Flag-gated OFF** by default. |
| `op_moe_w64.cuh` | wave64->warp32 mutant of `op_moe.cuh` | Negative control. Killed at fu relL2 3.215e+00 — proves the MoE test can detect a transliterated reduction. |
| `t3_pipe_probe.cu` | Can the prefill GEMM's mma B operand be read from a NATURAL `[n][k]` smem tile (cp.async-friendly) with `ldmatrix.x2` non-`.trans`, instead of the transpose-scatter `[k][n]` + `.trans`? | YES, **bit-identical**: 0/2048 register mismatches vs the in-tree `[k][n]+.trans` path, matches f32 CPU ref. Unblocked the T3 real-cp.async pipeline (op_gemm.cuh `d_gemm`/`d_gemm_glu`). |
| `gemv_warpspec_prod_cons.cu` | REVIVAL: does row-double-buffered producer/consumer GEMV (2 producer + 6 consumer warps, depth-S smem ring) beat uniform one-warp-per-row at decode shapes? | **NO — DROP.** Was never runnable (3 bugs: divergent `__syncthreads`, misused `expect_tx`, L2-measuring harness — all fixed here). Now correct: prodcons **never wins**, best case ties (S=3 ≈ uniform on lm_head 1503 vs 1507 GB/s, N4096 1136 vs 1143). Big **loss on small-K** `down` K704 (387 vs 573 GB/s, **−32%**): 6-consumer cooperative reduction is starved when K is small. Depth sweep: S=3 best, S=2 worst, S=4≈S=3. Uniform `d_gemv` already hits 84% HBM (1507 GB/s = 98% of the 1535 GB/s achievable ceiling) on lm_head. Confirms warpspec_ab: BW-bound GEMV gains nothing from load/compute split. Cost: +~28 KB smem (ring), no reg win. |
| `mxfp8_probe.cu` | Does `mma.sync.kind::mxf8f6f4.block_scale` (e4m3 + UE8M0/32-elem scale) work on sm_120a? Gates the MXFP8 roadmap. | **NO — hardware lacks it.** ptxas rejects outright: *"Instruction 'mma with block scale' not supported on .target 'sm_120'"* (+ `.kind::mxf8f6f4`/`.block_scale`/`.scale_vec::1X` all unsupported). block_scale MMA is a tcgen05 (sm_100a datacenter) feature; sm_120 has warp-level `mma.sync` only. **Fallback measured:** plain `m16n8k32` e4m3 is **bit-exact** (0/16384) at **933 TFLOPS**. MXFP8 here must SW-dequant the UE8M0 scale and feed plain e4m3/FFMA. |
| `gemv_dimspec.cu` | bf16 MoE expert GEMV (M=1): flat vs split-K×2 vs N-tile/warp vs fused gate+up, at gate/up (N704 K2816) and down (N2816 K704). | **flat is right; fuse gate+up.** gate/up: flat 1090, split-K 1118, N-tile 1103 GB/s — all within noise. down (small-K): flat 1101 best, **split-K −30%** (775), N-tile −4%. **Fused gate+up: 1346 vs 1096 GB/s (+23%)** — x read once, one launch (validates production `d_gemv_glu`). Regs: flat 35, split-K 39, N-tile 40, fused 36. |
| `fp8_gemv_moe.cu` | fp8 (w8a16) MoE GEMV: `dot8_fp8` UN=4 vs UN=8, and fused gate+up; does fp8 reach bf16's %HBM? | **UN=4 wins, fuse gate+up.** gate/up: UN=4 1070 vs UN=8 985 GB/s (UN=8 spills 32→42 regs, −occ) — confirms production `GV_UNROLL_FP8=4`. down: UN=4 858 vs UN=8 781. **Fused fp8 gate+up: 1284 vs 1078 GB/s (+19%)** (validates `d_gemv_glu_fp8`). **fp8 matches bf16 %HBM on big-K** (gate/up 60% vs 61%) but **lags on small-K** (down 48% vs 61%): half the bytes ⇒ dequant/reduction overhead less hidden. Regs: UN=4 32, UN=8 42, fused 46. |
| `fa_gf_full_ab.cu` | On the 26B FULL layer (kvh=2, hd512, GQA8), does raising the flash-decode fusion factor GF_FULL 2→4→8 (fewer KV re-reads) or L2-coscheduling the GF=2 re-reads cut decode time? Drives the REAL `d_flash_decode<512,GF>`. | **GF=4 + refilled nsplit wins**: −38% @32k, **−10% @64k, −17% @128k** vs GF2/ns24. GF=8 wins only @32k (−43%) and REGRESSES @64k+. L2-coschedule (same-KV items adjacent) is **refuted** (−5..−19%, worse). No spills at any GF (GF2/4/8 = 55/74/128 regs). GF4 needs nsplit doubled (`PLOW_NS_FULL_ABS=48`) or it underfills and loses. Ship path already wired: `-DPLOW_NV_FA_GF_FULL=4` + `PLOW_NS_FULL_ABS=48`, both opt-in, 31B byte-identical by default. |
| `hopper_cluster_multicast.cu` | H100-only (sm_120a has no clusters at all): do thread-block clusters + DSMEM + `cp.async.bulk.tensor .multicast::cluster` cut weight traffic for prefill GEMM/MoE? | **NO — DO NOT PURSUE.** All three primitives verified working (cluster 8 portable/16 non-portable; DSMEM checksums exact; SASS `UTMALDG.2D.MULTICAST [UR4],[UR8],UR11` vs plain `UTMALDG.2D`, kernels otherwise instruction-identical). But multicast **never wins**: 0.78–0.94x at C=2/4/8, both N=4096 and N=15360. Cause: the C duplicate reads are **L2 hits, not DRAM** — at C=2 the baseline sustains 5996 GB/s L2→SM on top of a 2998 GB/s DRAM stream, so L2 has ~2x the DRAM ceiling spare. Multicast removes the non-scarce resource, does not reduce smem write volume (C x 16 KB still lands in C CTAs), and adds DSMEM mbarrier round-trips. A 2x-deeper pipeline at equal smem reproduces the tie, so it is not a producer-concurrency artifact. **Two integration blockers regardless:** (a) `cuLaunchCooperativeKernel` cannot carry a cluster dim (needs `cuLaunchKernelEx` with COOPERATIVE+CLUSTER_DIMENSION), and grid 132 = 4x33 means C=8 divides only at even occupancy, C=16 never; (b) **fatal** — every interpreter block walks its OWN stream (`interp_sm120.cu` `prog.stream_ofs[cu]`) with per-entry counter gates, so the C CTAs of a cluster are in different ops of different layers; multicast needs all C inside one tile loop and a cluster barrier across divergent streams **deadlocks**. That is a scheduling-model change, not a kernel change. |
| `hopper_warpspec_prefill.cu` | H100 re-test of the sm_120a `setmaxnreg` refutation, on the COMPUTE-bound prefill GEMM (the sm_120a verdict was measured on BW-bound decode GEMV): does producer/consumer warp specialization + register donation beat the uniform wgmma baseline? | **setmaxnreg is REAL on Hopper (unlike the sm_120a CTA-local finding) but is never the winner; warp specialization alone IS worth it, per-shape.** ptxas does region-based allocation: at MS=2, entry=128, both at 2 blk/SM, the `clamp` control spills 1024 B/thread (408 STL/LDL) at **33 TF/s** while `smr` has **zero spills at 183 TF/s** — a 5.5x rescue at identical SM register cost, so the donation really does reach the consumer. **But there is nothing to donate:** below ~88 producer registers throughput drops 20-45% with NO spills (staging address-gen starves, in-flight cp.async collapses); usable window prod in [88,120], i.e. <=80 donatable. CUTLASS/DeepGEMM `dec` to 24-40 only because their producer is **TMA** (one thread, a descriptor, a barrier) — structural, not a tuning miss. At every shape `ws_free`/`ws_clamp` >= `ws_smr`. **Warp specialization by itself:** +11.3% @(512,4096,3840), **+22.8% @(200,4096,3840)** (ragged M tail: producer keeps streaming through the consumer epilogue), -0.8% @N=15360 — and it cuts registers unaided (134->90 MS=1, 168->154 MS=2) because staging address math stops sharing live ranges with accumulators. relL2 3.78e-06 identical across all 7 variants. **Verdict:** do NOT integrate setmaxnreg; DO consider the producer/consumer split, but the winning (MS,NS,variant) differs per shape, so it belongs in the arch-gemm tuning selection, not a global switch. **Precondition for the CUTLASS-style win: move the producer from cp.async to TMA first**, then re-test setmaxnreg. |
| `fa_gf_full_h100_ab.cu` | H100 re-sweep of GF_FULL x nsplit x context on the Gemma-4-26B FULL layer (kvh=2, hd512, GQA8), since the shipped GF=4 was chosen on RTX data and H100 has 132 SM / 60 MB L2 / HBM3 instead of 188 SM / 128 MB / GDDR7. Drives the real `d_flash_decode<512,GF>` + `d_flash_merge<512>`. | **GF=4 CONFIRMED optimal on H100 at every context 8k-128k; but nsplit must be 33, not the RTX 48.** GF=4/ns=33 beats best GF=2 by 17/26/30/32/33% and best GF=8 by 39/15/15/15/16% at 8k/16k/32k/64k/128k. **GF=8 wins nowhere on H100** (on RTX it at least won at 32k): at 128k it reaches only 978 GB/s of a 3683 GB/s ceiling — no longer memory bound but issue/ALU bound (8 hd512 accumulators, 140 regs). GF=2 is bandwidth-capped (4x re-read = 3112 GB/s logical at 128k, ~85% of the machine's streaming limit spent on redundant traffic; 60 MB L2 absorbs almost none of 512 MB of KV). **nsplit is a grid-alignment CLIFF, not a slope:** ns=33 -> 34 costs **+67%** at 128k (136 items on 132 SMs leaves 4 SMs running two and d_flash_merge waits). Law is `aligned = n_cu/gcd(n_grp,n_cu)` = 132/4 = **33** (same law as the RTX 188-SM finding, different constant). The inherited RTX ns=48 costs **+41% @128k / +47% @32k**; ns=24 +43%. ns=32 ties. Scales as 33/n_batch when batching. Regs sm_90a GF2/4/8 = **79/104/140, zero spills anywhere** (richer than RTX's 55/74/128); megakernel decode cubin is **REG:208 at all three GF** so the choice is occupancy-neutral. Correctness: worst relL2 1.68e-03 across all GF x nsplit x context, 0 failures. L2 methodology: HOT ~= COLD everywhere (<1% at >=16k) because the kernel is fill/latency bound (1545 GB/s at 8k vs 3683 ceiling), so L2 residency never distorted the ranking. **Emitter note:** `crates/plowc/src/bin/gemma4.rs` already has the right rule but gates it on `c.kvh_full >= 4` while this shape is kvh_full=2, so it never fires — widen the gate or ship PLOW_NS_FULL_ABS=33. |
| `splitzip_h100_ab.cu` | H100 re-test of SplitZip lossless bf16 exponent compression, which WON on sm_120a (1.30x fused into GEMV) but ships flag-gated OFF. Hypothesis was that H100's smaller L2 + higher compute:BW ratio should make it pay MORE. | **NO — DO NOT ENABLE ON sm_90a. The hypothesis is INVERTED.** Bit-exactness PASS (real Gemma-4 bf16 weights, 89.1 M elements, codec round-trip memcmp EQUAL, both negative controls detected; fused-GEMV output memcmp EQUAL on all 5 shapes). Compression **1.3314x** achieved (escape rate 0.0194%) = the hard speedup ceiling. Measured A/B at M=1 steady state: **0.828x** q_proj, 0.836x kv_proj, 0.834x gate/up, **0.716x** down, 0.848x lm_head — a LOSS everywhere; short-burst is worse still (0.55-0.72x). **Root cause:** H100's higher compute:BW ratio is a TENSOR-CORE fact, while SplitZip decode runs on the INT/CUDA-core pipe. Issue slots per weight byte moved: RTX PRO 6000 ~2.4 warp-inst/elem vs H100 NVL **~0.53** — **4.6x FEWER** scalar slots per element than the GPU where it won. Decomposition: the two-plane layout alone caps at 1.16-1.23x (a zero-decode ablation with identical loads still only reaches that: 24 B across two streams sustains 2.0-2.2 TB/s vs 2.4 TB/s for one 32 B stream), and reconstruct ALU costs a further 29-40% at byte-identical traffic. Removing the escape pass recovers only to 0.92-0.94x and is not lossless. Occupancy is NOT the limiter (raw 37 regs/6 blk_SM vs SZ 43/5; forcing 32 regs hits 100% occ but spills and is 5-10% slower). No cheaper decode can close a 29-40% ALU deficit plus a 10-14% layout deficit — the p9 kill-bar (<~4 SASS ops/elem hidden in the load shadow) is unsatisfiable at H100's issue-slots-per-byte. Only untested niche: batched rungs M>=8 where reconstruct amortizes xM, outside the M=1 decode target. |

## Hardware facts these established (RTX 5090, sm_120a, GB202)

- Measured streaming ceiling **1673 GB/s** (93% of the 1792 GB/s spec). Any
  figure above this is invalid by construction — check for a working set that
  fits the ~100 MB L2.
- **No thread-block clusters, no DSM** (cluster size 1). NVIDIA's generic
  "cc 12.x" table lumps 12.0 with 12.1 and wrongly lists clusters as available.
- TMA available; `.multicast::cluster` is not (it needs clusters).
- Warp = 32. The AMD reference is wave64 — every reduction must be re-derived.

## Hardware facts (RTX PRO 6000 Blackwell Server Edition, sm_120a — the p9-proto card)

The last four rows above were measured on the **RTX PRO 6000** (188 SMs, 96 GB GDDR7,
**L2 = 128 MB**), not the RTX 5090. Notes for anyone re-running:

- Measured streaming ceiling **1535 GB/s** (simple grid-stride `float4` read of 2 GB;
  `d_gemv` on lm_head reaches 1507, i.e. 98% of it). Spec is 1790 GB/s; treat 1535 as the
  real ceiling. `%HBM` in the four files above is vs the 1790 spec (so ~86% max attainable).
- **L2 is 128 MB** — bigger than the 5090's ~96 MB. Any weight tensor < 128 MB is L2-resident
  on a repeat loop, so every one of these harnesses replicates weights past ~400 MB and cycles
  buffers to force cold HBM reads. Skipping this reads L2 and reports impossible >100% figures.
- Standalone GEMV kernels must use `blockIdx.x` as the column slice; passing a fixed `slice=0`
  (as the persistent-megakernel signature invites) makes all blocks recompute the same rows.
- block-scale / MXFP8 tensor-core MMA is **absent** (sm_100a-only); e4m3 `mma.sync` = 933 TFLOPS.

## Hardware / toolchain facts (H100 NVL, sm_90a — Hopper)

Measured on H100 NVL (132 SMs, 95 GB HBM3), CUDA 13.0, driver 570.

- Streaming ceiling: **3727 GB/s short-burst, but only ~3526 GB/s sustained**, and a real memory-bound decode loop sees
  **2.2-2.4 TB/s**. THIS CARD IS POWER-CAPPED AT 310 W (max 400) and DVFS-limited: sustained SM clock drops to **600-615 MHz**
  on a memory-bound GEMV vs 1305-1350 MHz on a compute-heavier kernel. Short-burst bandwidth numbers DO NOT hold in a decode
  loop -- always report steady state, and note that raising the power cap moves results toward the burst numbers.
- **L2 measures 60 MB** on this card (not the 50 MB often quoted). Still less than half the RTX PRO 6000's 128 MB, so
  harnesses must replicate weights past ~500 MB and cycle buffers to force cold HBM reads.
- Clusters, DSMEM and TMA multicast all EXIST here (they do not on sm_120a) — see `hopper_cluster_multicast.cu`, which
  verifies all three and then refutes multicast for plow's shapes.
| `cluster_gq_probe.cu` | The three cluster/DSM/multicast cases the earlier refutation left open, re-examined against plow's DEFAULT global-queue scheduler (a static-audit argued cluster-cooperative claiming should be viable under GQ since all C blocks land on one entry). | **ALL THREE STILL REFUTED; measurement overrides the audit.** Mechanisms real (SASS: cluster.sync->UCGABAR_ARV/WAIT+MEMBAR.ALL.GPU, mapa+ld.shared::cluster, UTMALDG.2D.MULTICAST). **P1 cluster-cooperative GQ claim: NOT VIABLE** — a single claim is correct at full grid (memcheck 0 errors) but costs 4.6x (349->1602 ns, two cluster.sync vs one atomicAdd), and SUSTAINED claiming FAULTS ('unspecified launch failure') — a forward-progress failure of the cluster.sync spin under the 310 W DVFS collapse, re-introducing the deadlock class in the persistent-grid regime. So GQ does NOT dissolve the deadlock in practice. **P2 multicast off-L2: no crossover** — 0.78-0.98x swept 30 MB->1440 MB (24x L2); co-resident cluster CTAs read the shared tile simultaneously (reuse distance ~0) so it is ALWAYS an L2 hit regardless of working-set size. **P2' multicast on COMPUTE-BOUND wgmma GEMM (fast.cu regime, the gap P2 left): still does not pay** — marginal and sign-flipping: BM=64 +1-4%, BM=128 (the tile plow uses) 0.97-0.99x; the benefit decays to negative as arithmetic intensity rises (cross-CTA mbarrier round-trips dominate once the shared-operand load is no longer the bottleneck). SASS: 2x UTMALDG.2D.MULTICAST, cluster.sync only at prologue/teardown (none in the k-loop, so P1's spin-fault does not apply to a launch-time cluster GEMM). Reconciles with fast.cu (a modest margin-op inside a warp-specialized kernel, not a structural win). **P3 DSM reductions lose** — split-K combine 1.15-1.33x slower, flash split-KV merge 2.76-3.35x slower (relL2 1.5-1.7e-3, correct): cluster.sync is far costlier than a coalesced HBM read and HBM was never the bottleneck for these reductions. **Verdict: do not build a cluster/DSM/multicast packet or interpreter extension for plow — these features optimize resources plow is not bound by (L2 not HBM), and the barrier costs more than it saves.** |

### MANDATORY benchmarking harness on this H100 (read before trusting any number)

This card is **power-capped at 310 W** (max 400) and thermally/DVFS limited. Under
sustained wgmma the SM clock collapses **1785 -> ~700 MHz within ~2 seconds**. A
back-to-back timing loop therefore measures the power cap, not the kernel:

- the MoE grouped probe re-run UNCHANGED in a naive loop reports **86 TF/s** where
  its careful harness reports **163** -- a 1.9x error, entirely measurement artifact
- a memory-bound GEMV sustains only **600-615 MHz** vs 1305-1350 MHz for a
  compute-heavier kernel, so DVFS also *reorders* A/B comparisons between kernels
  with different DRAM intensity (this is why SplitZip looks better in steady state
  than in burst -- and worse if you raise the cap)
- short-burst bandwidth here reads 3.4-3.7 TB/s but a real decode loop sees
  **2.2-2.4 TB/s**

Required harness for any perf claim on this card:
1. **short bursts** (~2 ms of work, adaptively sized) -- never a long back-to-back loop
2. **25 ms idle gaps** between bursts so the card recovers clocks
3. **rotated round-robin** across the variants being compared, not variant-at-a-time
4. **min-of-N** (N >= 12) rounds, not mean
5. **sample the SM clock with NVML during the run** and report it; if it fell below
   ~1350 MHz the number is contaminated
6. defeat L2 (60 MB) by replicating weights past ~500 MB and cycling buffers

`tma_ws_moe_group.cu` implements all six and is the reference harness to copy.
Numbers produced without this are noise, regardless of how stable they look.

### GPU is shared -- take the lock

Multiple agents/sessions use this card. Serialize every GPU run with
`flock /tmp/plow_gpu.lock <cmd>`. Note that at least one observed job did NOT honor
the lock (91 GB resident, 95% util), which depressed a concurrent sweep's absolutes
by ~2x -- so also check `nvidia-smi` before trusting absolute numbers, and prefer
same-run ratios when you cannot guarantee an idle card.

### nvcc: how to actually get wgmma on sm_90a

`wgmma` (and `wgmma.fence`) are **`sm_90a`-only** opcodes, and whether nvcc gives
you them depends on the OUTPUT MODE, not just `-arch`. Verified matrix:

| invocation | wgmma |
|---|---|
| `-arch=sm_90a -cubin` (what `scripts/build_sm90a_cubin.sh` does) | **accepted**, HGMMA emitted |
| `-arch=sm_90a -o <executable>` | **REJECTED** — "not supported on .target 'sm_90'" |
| `-gencode arch=compute_90a,code=sm_90a -o <executable>` | **accepted**, HGMMA emitted |
| `-arch=native` (on H100) | **REJECTED** — resolves to `sm_90`, no `a` suffix |

So the interpreter cubin build is fine as written, but **every standalone
harness/probe/benchmark executable in this directory must use the `-gencode`
form**. Four independent probes hit the executable case and lost time to it.
`crates/plowrt/tests/cuda_gpu.rs` uses `-arch=native`; harmless today (it only
builds a trivial vadd) but a latent trap for any future Hopper-feature test.

### Hopper has no native fp8 `mma.sync`

One `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`:

| target | SASS |
|---|---|
| sm_90a (H100) | **12x `F2FP.F16.E4M3.UNPACK_B` + 2x `HMMA.16816.F32`** (emulated: e4m3 -> f16, f16 tensor core, half k-depth) |
| sm_120a (consumer Blackwell) | **1x `QMMA.16832.F32.E4M3.E4M3`** (native) |

On Hopper the fp8 tensor-core pipe is reachable **only** through wgmma
(`QGMMA`). `op_gemm.cuh`'s `pgm_mma_fp8_k32` and the MoE fp8 twins therefore get
**zero** fp8 tensor-core benefit on H100 — 14 instructions doing the work of 1.

### Smem swizzle is a PERFORMANCE requirement for wgmma, not a correctness one

A swizzle-free core-matrix-blocked operand layout passes a numeric oracle at
identical relL2 but runs **84 TF/s vs 177 TF/s** (bf16 GEMM probe): its 1024 B
row-core stride puts every row-core on one smem bank -> 8-way conflicts on both
the `cp.async` store and the wgmma operand read. Use the 128-Byte swizzle
(physical `row*64 + ((c ^ (row&7))*8)`, LBO=16 B, SBO=1024 B, swizzle bits
[63:62]=1, tile base 1024 B aligned, k16 substep advances the START ADDRESS by
+32 B only).

### wgmma probe results (all oracle-validated on H100, see the `wgmma_*` files)

| probe | speedup vs `mma.sync` | note |
|---|---|---|
| `wgmma_flash_prefill_probe.cu` | **2.82x** (43.6 vs 15.5 TF/s) | biggest win; softmax must stay in-fragment, V must be MN-major |
| `wgmma_bf16_probe.cu` | **1.64x** @N=15360, 1.28x @N=4096 | 177 TF/s; BM=128 config reaches 219 TF/s |
| `wgmma_fp8_probe.cu` | **1.48x / 1.40x** | `QGMMA`; fp8 accumulate is not true f32, wants DeepGEMM two-level promotion |
| `wgmma_moe_group_probe.cu` | 1.07-1.12x | swizzle-free, staging-bound; retest with 128B swizzle pending |
| `gemv_lds_vectorize_probe.cu` | n/a — **refuted** | decode GEMV already LDS.128; A/B flat null, decode is HBM-bound |

The opcode swap alone is necessary but not sufficient: these tiles are
`cp.async`-staging-bound (a BK=64 stage issues ~1536 `cp.async` against 4
`HGMMA`). The full Hopper win needs **TMA + producer/consumer warp
specialization + 128x256 tiles**.
