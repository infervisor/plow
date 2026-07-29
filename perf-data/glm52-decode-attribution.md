# GLM-5.2 TP4 block-fp8 decode — per-opcode attribution of one token

**Measured 2026-07-28.** 4x gfx950 (MI355X), real weights (`/home/lava/models/GLM-5.2-plow`,
715 GB, 183 GiB/rank), ctx 1024, `--sweep 1k --steps 65`, `PLOW_TRACE_RAW` on rank 0.
Instrument: `runtime/tests/glm52_decode.c:421` + `scripts/glm52_token_attrib.py` (new; companion
to the per-block `scripts/glm52_trace_analyze.py`, same trace format).

**§0-BENCH.** Every number here comes from the C harness, which is an EXPERIMENT instrument.
Nothing in this file may be placed next to a vLLM number. This is plow-internal attribution only.

Leases: `gpulease -n 4` (gpus 1,2,3,4), no contention warning on any run.

---

## 0. TL;DR — where GLM's 34 ms goes

| finding | size |
|---|--:|
| **`MLA_MERGE_FOLD` (op 57)** — 78 packets, 111 µs each, **0.6% of HBM peak** | **8.69 ms — 25% of the token** |
| GEMV family (`Gemv`+`GemvQkv`+`GemvGlu`+dense) at 23–25% of peak | 9.75 ms |
| MoE routed experts (`45`/`46`) at 28% / 12% of peak | 4.76 ms |
| `FlashMlaDecode` at 0.5% of peak | 2.74 ms |
| 156 XReduce collectives | 2.41 ms |
| 545 one-workgroup packets (RmsNorm/Residual/RouterTopk) | 2.32 ms |
| idle (no packet body live anywhere on the chip) | 1.45 ms |

**The roofline premise in the campaign notes is wrong and it matters.** GLM's *active weights per
GPU per token measured off the packet* are **19.20 GB**, not ~9.9 GB — because **only the MoE
experts and the 3 dense FFNs are block-fp8. Every MLA projection, `o_proj`, the shared expert and
`lm_head` are still bf16** (`o_proj.weight` = 49152 KB for [6144,4096] = 2 B/elt). So the floor is
**3.10 ms at 6200 GB/s, and we are 11.2x above it**, not 20x above 1.6 ms. Of the 19.20 GB,
**13.36 GB is bf16 and only 5.83 GB is block-fp8**, so converting the MLA projections + `o_proj` +
shared expert + `lm_head` to fp8 would remove 6.68 GB = **1.08 ms of floor** on its own. The
"latency not bandwidth" conclusion still holds, with less headroom than advertised.

**One measured fix already found (§5): a single template parameter on `d_mla_merge_fold` is
−1.99 ms/token (34.361 → 32.376), and the remaining 6.7 ms of that op is a separate, larger
kernel defect.**

---

## 1. Per-opcode budget — baseline `GLM_MOE_CORESIDENT=1`

`glm52_tp4_64k.pkt`, 2756 ops, 297,569 workgroup-packets.
Sweep median of 65 = **34.607 ms/token**; the traced step = **34.681 ms**; the table sums to the
traced step *by construction* (see §1.1). Campaign record for this blob is 34.149 ms — this run is
**+1.3%**, which is the tracing store (one 40-B record per workgroup-packet) plus run-to-run drift.

| opcode | pkts | ms | %tok | µs/pkt | wg | **eff wg** | GB | pkt GB/s | %6200 | tok GB/s | %6200 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **MlaMergeFold (57)** | 78 | **8.690** | **25.1%** | 111.4 | 256 | **28.3** | 0.327 | 38 | **0.6%** | 38 | 0.6% |
| Gemv (10) | 229 | 4.435 | 12.8% | 19.5 | 256 | 197.4 | 6.537 | 1461 | 23.6% | 1474 | 23.8% |
| GemvQkv (22) | 156 | 3.763 | 10.9% | 24.1 | 256 | 195.1 | 5.459 | 1451 | 23.4% | 1451 | 23.4% |
| FlashMlaDecode (50) | 78 | 2.740 | 7.9% | 35.1 | 256 | 197.5 | 0.092 | 34 | **0.5%** | 34 | 0.5% |
| MoeExpertDownFp8Blk (46) | 600 | 2.579 | 7.4% | 32.6 | 32 | 28.8 | 1.888 | 96 | 1.6% | 732 | 11.8% |
| XReduce (24) | 156 | 2.414 | 7.0% | 15.5 | 256 | 160.1 | — | — | — | — | — |
| MoeExpertGluFp8Blk (45) | 600 | 2.176 | 6.3% | 28.4 | 32 | 28.8 | 3.776 | 221 | 3.6% | 1735 | 28.0% |
| HeadNormRope (3) | 156 | 1.584 | 4.6% | 13.3 | 256 | 165.6 | ~0 | — | — | — | — |
| GemvGlu (19) | 75 | 1.353 | 3.9% | 25.3 | 256 | 193.8 | 0.944 | 497 | 8.0% | 697 | 11.2% |
| RmsNorm (1) | 313 | 1.246 | 3.6% | 4.8 | **1** | 1.0 | 0.002 | 2 | 0.0% | 2 | 0.0% |
| MoeCombine (43) | 75 | 0.955 | 2.8% | 12.7 | 256 | 117.6 | — | — | — | — | — |
| Residual (4) | 156 | 0.541 | 1.6% | 3.5 | **1** | 1.0 | — | — | — | — | — |
| MoeRouterTopk (56) | 75 | 0.527 | 1.5% | 14.1 | **1** | 1.0 | — | — | — | — | — |
| DenseGluFp8Blk (47) | 3 | 0.103 | 0.3% | 34.5 | 256 | 166.9 | 0.113 | 1095 | 17.7% | 1095 | 17.7% |
| GemvFp8Blk (44) | 3 | 0.059 | 0.2% | 19.5 | 256 | 182.7 | 0.057 | 966 | 15.6% | 966 | 15.6% |
| Embed (6) | 1 | 0.049 | 0.1% | 49.2 | 256 | 156.1 | ~0 | — | — | — | — |
| ArgmaxFin (18) | 1 | 0.007 | 0.0% | 7.5 | 1 | 1.0 | — | — | — | — | — |
| Argmax (17) | 1 | 0.007 | 0.0% | 6.8 | 64 | 55.7 | — | — | — | — | — |
| **IDLE / GATE-STALL** | — | **1.451** | **4.2%** | — | — | — | — | — | — | — | — |
| **TOTAL** | **2756** | **34.681** | 100% | | | | **19.195** | | | 554 | **8.9%** |

Layer classes: MoE L3-77 **31.74 ms (91.5%)**, dense L0-2 1.07 ms (3.1%), embed+head 0.42 ms (1.2%).

Per-CU decomposition (the §7 shape): **body 17.09 ms | gate-stall 16.77 ms | gap/launch 0.82 ms**.
Every workgroup spends **48% of the token spinning on a closed counter** — the serial-chain
signature, not protocol overhead.

### 1.1 How the ms column is computed (and why it sums)

* An op instance owns the wall interval `[min(t_ready), max(t_end)]` over its workgroups — the
  window in which some workgroup of that op is executing a body.
* The step span is swept; every elementary interval is split **equally among the op instances live
  in it**. Intervals with nothing live go to IDLE.
* Therefore `Σ per-op ms + idle ≡ traced step`, exactly, with no double counting, and genuine
  concurrency (co-resident experts) shows up as a *reduction* rather than as overlap.
* `pkt GB/s` = bytes / Σ(per-packet wall) — judges **one packet against its own roofline**.
  `tok GB/s` = bytes / attributed ms — machine utilisation while the op owns the token. They differ
  only where packets of the same op run concurrently (the 8 routed experts).

### 1.2 `eff wg` — why "256 workgroups" is a lie

`wg` counts workgroups that *recorded* the packet. `eff wg = Σbody / max(body)` is the width a
perfectly balanced packet of the same total work would need. `MLA_MERGE_FOLD` dispatches 256 and
has **eff wg 28**: 16 workgroups grind for ~107 µs while 240 fall out of the work loop in ~4 µs.
Measured body-tick histogram, `inst 9` (10 ns/tick):

```
cu   0..15  : 10580 .. 11124 ticks   (106-111 us)   <- 16 workgroups do all the work
cu  16..255 :   404 ..   750 ticks   (4.0-7.5 us)   <- 240 workgroups arrive and leave
```

### 1.3 HBM byte accounting

Weight operands are summed from the sidecar's declared tensor bytes. Four corrections, because
the declared size is not what is read:

| op | correction |
|---|---|
| 45/46 (+48/49) | `t[3]/t[4]` are `[E][3]` u64 **pointer tables** (6 KB), not weights. Bytes derived from `i[1],i[2]`: glu = 2·I·H + scales, down = H·I + scales, fp8 1 B/elt, f32 [128,128] scale grid. |
| 50/54 (flash) | `kv.*` is declared at `max_ctx` (64k). Only `ctx` rows are touched: `ctx × 1152 B` (kv_lora 512 + rope 64, bf16). |
| 6 (Embed) | one row (12 KB), not the 1.9 GB table. |
| 24 (XReduce) | fabric bytes, not HBM. Priced separately as time only. |

---

## 2. The starvation census — GLM's answer to §7b

§7b found 63% of Gemma's packets on ≤4 of 256 CUs, costing ~1.8 ms. GLM is **worse, and the
badness has moved**: it is no longer mostly 1-CU norms, it is *nominally wide packets that are
internally starved*.

**By dispatched workgroups:**

| wgs | pkts | %pkts | ms | %token |
|---|--:|--:|--:|--:|
| 1 | 545 | 19.8% | 2.322 | 6.7% |
| 4–32 | 1200 | 43.5% | 4.755 | 13.7% |
| 33–128 | 1 | 0.0% | 0.007 | 0.0% |
| 255–256 | 1010 | 36.6% | 26.146 | 75.4% |

**By EFFECTIVE width (`Σbody/max(body)`) — the honest one:**

| eff wgs | pkts | %pkts | ms | %token |
|---|--:|--:|--:|--:|
| 1 | 545 | 19.8% | 2.322 | 6.7% |
| 4–32 | 1278 | 46.4% | **13.446** | **38.8%** |
| 33–128 | 93 | 3.4% | 1.160 | 3.3% |
| 129–255 | 840 | 30.5% | 16.303 | 47.0% |

* **≤4 effective workgroups: 545 packets (19.8%), 2.32 ms (6.7% of the token).**
  These are `RmsNorm` ×313, `Residual` ×156, `MoeRouterTopk` ×75, `ArgmaxFin` ×1 — all emitted on
  `one = vec![0u32]`, literally one CU of 256.
* **≤32 effective workgroups: 1823 packets (66.1%), 15.77 ms — 45.5% of the token.**
* **≤128: 1916 packets (69.5%), 16.93 ms — 48.8% of the token.**
* Nothing in this program ever reaches a *balanced* 256. The best-filled family (GEMV) sits at
  eff wg ≈ 195–220.

The Gemma lesson generalises but with a twist: on GLM the ≤32 bucket is dominated not by the
1-CU norms (2.32 ms) but by **`MLA_MERGE_FOLD` (8.69 ms) and the 32-CU expert slices (4.76 ms)**.
The 1-CU norms are now a rounding error next to a 256-wide packet that only uses 16 CUs.

---

## 3. The top 3 losers — ops furthest below what their shape allows

The control that proves the GEMV kernel body is fine: **`lm_head`, N=154880 K=6144 bf16, one
packet, 1.903 GB in 0.326 ms = 5834 GB/s = 94.1% of the 6200 GB/s ceiling.** Everything below uses
the same machine.

### Loser #1 — `MLA_MERGE_FOLD` (op 57): 8.69 ms/token, 0.6% of peak

* Shape per packet: merge `nsplit=16` latent partials then fold `olat[512] @ W_uv[h][512,256]`,
  `n_head_local = 16` (TP4 shard of 64), `DK=512`, `V=256`, W_uv **bf16**.
* Bytes per packet: 4.19 MB `v_absorb` + 0.52 MB `opart` (f32) = **4.72 MB in 111 µs = 42 GB/s**.
* **Two independent defects, both in `runtime/amd/op_attention.h:2294` `d_mla_merge_fold<DK,VT>`:**

  1. **Occupancy.** `runtime/amd/interp.hip:1095` instantiates `d_mla_merge_fold<512, **256**>`.
     With `VT == V`, `vtiles = 1` and `n_work = n_batch·n_head·vtiles = **16**`. 16 of 256
     workgroups. The kernel's own doc-comment names `VT` as the occupancy knob and
     `runtime/amd/test_kernels.hip:415` already instantiates `<512, 32>` — **the arm exists, is
     correct, and nothing routes to it** (knob-contract §4's recurring bug shape).
     Measured cost of the omission: **−1.99 ms/token** to fix (§5).
  2. **No memory-level parallelism in the fold, and half the workgroup idle.** The fold is
     ```c
     for (unsigned v = v0 + tid; v < v1; v += PLOW_THREADS)         /* PLOW_THREADS = 512 */
         for (unsigned l = 0; l < DK; l++)                          /* DK = 512 */
             acc += olds[l] * bf2f(wv[(size_t)l * V + v]);
     ```
     One thread per output column, a **512-deep serially dependent chain with one strided global
     load per iteration** (stride `V*2 = 512 B`), no unroll, no independent accumulators, no LDS
     staging of `W_uv`. 111 µs / 512 = **217 ns per iteration ≈ exactly one uncached memory
     latency** — the loop is not pipelined at all. And with `V=256 < PLOW_THREADS=512`, threads
     256–511 never enter the loop, so even the 16 working workgroups are half idle.
     After the VT fix this term is still **6.66 ms** — it is the larger half of the defect.
* Sizing the prize: 4.72 MB/layer at even 50% of peak is ~1.5 µs/packet → **~0.12 ms/token**.
  **~8.5 ms of a 34.4 ms token is sitting in this one kernel.**

### Loser #2 — the K=512 down-projections: 3.56 ms/token at 12% / 8% of peak

| op | shape (M,N,K) | dtype | ms | tok GB/s | %6200 |
|---|---|---|--:|--:|--:|
| `MoeExpertDownFp8Blk` ×600 | (1, 6144, **512**) per expert | fp8 + f32 [128,128] grid | 2.579 | 732 | 11.8% |
| `Gemv` shared-expert `down_proj` ×75 | (1, 6144, **512**) | bf16 | 0.977 | 483 | 7.8% |

Compare their gate/up twins on the *same* weights with K and N swapped:
`MoeExpertGluFp8Blk` (1, 512, **6144**) reaches 1735 GB/s (28.0%) — **2.4x better**.
`GemvGlu` (1, 512, 6144) reaches 697 GB/s vs the down's 483.

**Why:** every wave-dot in this family reduces over K with 64 lanes
(`wave_dot_fp8_blk`, `op_moe.h`). At **K = 512 that is 8 elements per lane — a single `dot8` pass
and then a `wave_sum` cross-lane reduction.** The reduction tree and the loop prologue cost more
than the one load they amortise, so the op is pure latency. The gate/up direction (K=6144) gets 12
passes per lane and pipelines.

**K=512 is a TP4 artefact**: `imoe_l = moe_intermediate_size / tp = 2048/4`. TP8 would make it 256
and this worse; **EP (whole experts, `imoe_e = 2048`) makes it 2048 and is the structurally correct
fix** — which is exactly the §6g-KNOBS argument for EP, now with a measured mechanism behind it.

### Loser #3 — `FlashMlaDecode` (op 50): 2.74 ms/token, 35 µs/packet

* Shape: `n_head_local=16`, `ctx=1024`, latent `DK=512` + rope `64`, `nsplit=16`, head-fusion
  `GF=2` at this ctx → `n_grp = 8`, so **128 of 256 workgroups** get a work item
  (`n_batch·n_grp·nsplit = 128`). Measured body distribution is uniform, so this is a *dispatch*
  under-fill, not a straggler.
* Bytes: single-stream latent footprint `1024 × 1152 = 1.18 MB` → **34 GB/s (0.5%)**. Charging the
  head-group re-read (8 groups) gives 9.4 MB → 268 GB/s (4.3%). Either denominator is a disaster.
* 35 µs to consume 1024 KV rows of 1152 B with 16 query heads is latency, not bandwidth: at ctx
  1024 each of the 128 work items owns 64 positions = 72 KB, i.e. one CU moving 2 GB/s.
* This is the op §6g says is *flat* to 64k under DSA. Flat and expensive: it is a fixed ~2.7 ms
  toll on every token at every context.

**Honourable mentions.** `mlp.gate` router score GEMV (1, **256**, 6144) — 0.860 ms at 274 GB/s
(4.4%), because N=256 outputs spread over 2048 waves leaves 7/8 of them idle; plus its 1-CU
`MoeRouterTopk` tail at 0.527 ms. **1.39 ms/token to route.** And `o_proj` (1, 6144, 4096) bf16 at
1728 GB/s (27.9%) — 3.4x below what the same kernel achieves on `lm_head`, because 6144 output rows
over 2048 waves is 3 rows/wave and never builds a pipeline.

---

## 4. `GLM_MOE_CORESIDENT=1` vs `=2`, op by op

Both traced back-to-back in one lease, same weights, same harness.

| | sweep median | traced step |
|---|--:|--:|
| `glm52_tp4_64k.pkt` (cores=1) | 34.607 ms | 34.681 ms |
| `glm_cores2.pkt` (cores=2) | 33.520 ms | 33.528 ms |
| **delta** | **−1.09 ms** | **−1.15 ms** |

> **Discrepancy, stated rather than hidden.** The campaign record for cores=2 is **31.145 ms**
> (−3.00 vs baseline). On these two on-disk blobs today I measure **−1.09 ms**, not −3.00. Both
> runs were leased, uncontended, median-of-65, and the baseline reproduces its own record to
> +1.3%. The blobs differ in size (31.4 MB vs 29.6 MB) and the recorded −3.00 may have been taken
> against `glm52_tp4.pkt` (15.7 MB, a different `max_ctx`) rather than `glm52_tp4_64k.pkt`.
> **Anyone quoting 31.145 should re-derive it before building on it.**

Where the −1.15 ms comes from (fractional-attribution ms, so these add up):

| opcode | cores=1 | cores=2 | delta | what changed |
|---|--:|--:|--:|---|
| `Gemv` — shared-expert `down_proj` | 0.977 | 0.387 | **−0.590** | moved to its own 28-CU slice, overlapped with the routed experts |
| `GemvGlu` — shared-expert gate/up | 1.353 | 0.653 | **−0.700** | same |
| `Gemv` — `o_proj` | 2.272 | 2.125 | −0.147 | knock-on (earlier arrival) |
| `GemvQkv` | 3.763 | 3.587 | −0.176 | knock-on |
| `MoeRouterTopk` | 0.527 | 0.417 | −0.110 | knock-on |
| `MoeExpertGluFp8Blk` | 2.176 | 2.574 | **+0.398** | expert slice 32 → 28 CUs (`tk+1` parts) |
| `MoeExpertDownFp8Blk` | 2.579 | 2.756 | **+0.177** | same |
| `XReduce` | 2.414 | 2.548 | +0.134 | more skew to absorb at the rendezvous |
| all others | | | −0.05 | |
| **net** | 33.230 | 32.122 | **−1.108** | |

**Reading.** cores=2 buys exactly one thing: it takes the shared expert's **1.45 ms** off the
critical path by running it concurrently with the routed experts, and pays **0.58 ms** for
narrowing every routed-expert slice from 1/8 to 1/9 of the chip. Net −1.15.

**What a further win of this shape looks like — and its ceiling.** The lever is
*overlap something already-ready with the expert window*, and the shared expert was the only
routing-independent tenant available. There is nothing else in a GLM block that is ready at
`c_rn2` and not already running, so **this lever is now exhausted**; pushing it further only
narrows the expert slices and costs more than it saves (which is exactly why `GLM_GROUP=1` loses
2.88 ms). The remaining MoE lever is the *other* direction, EP: `imoe_e` 512 → 2048 fixes Loser #2
without touching concurrency.

---

## 5. Confirming experiment — the `MLA_MERGE_FOLD` occupancy defect is real and priced

One-token change, `runtime/amd/interp.hip:1095`: `d_mla_merge_fold<512, 256>` →
`d_mla_merge_fold<512, 32>` (the instantiation `test_kernels.hip:415` already carries). Both
objects built from the same source tree with the same `hipcc` line, both run in one lease on the
same `glm52_tp4_64k.pkt`:

| interp object | sweep median | traced step | `MLA_MERGE_FOLD` ms | eff wg | µs/pkt |
|---|--:|--:|--:|--:|--:|
| `<512, 256>` (shipping) | 34.361 ms | 34.556 ms | 8.678 | 28.3 | 111.3 |
| `<512, 32>` | **32.376 ms** | 32.206 ms | **6.656** | **124.8** | 85.3 |
| delta | **−1.985 ms (−5.8%)** | −2.350 | **−2.023** | | |

Per-CU gate-stall falls 16.77 → 12.23 ms; nothing else moves by more than 0.2 ms.

Two caveats, both load-bearing:

* **`VT=32` is a diagnostic, not a ship-ready change.** It fixes occupancy (16 → 128 working
  workgroups) but leaves the fold at 85 µs and 0.8% of peak, and it makes the merge 8x redundant
  while using only 32 of 512 threads in the fold. The right fix is to rewrite the fold as the
  wave-cooperative reduction the MoE down-projection already uses (or unroll `l` with independent
  accumulators + LDS-stage `W_uv`), which should take the op to ~0.2 ms rather than 6.7 ms.
* **Token identity: VERIFIED.** `--gen 24` on the same pkt/weights, both objects, prompt
  `100,264,6722,315,9822,374`: **24/24 ids identical**, and identical to the reference string in
  knob-contract §6g — `264 5777 9125 1948 498 323 279 6372 315 264 3162 2025 429 6147 498 311 653
  3654 2513 429 1035 5937 387 458`. `VT` is a pure work-partition (each v-tile re-merges `olat`
  identically), as the kernel's doc-comment claims. The full HF-coherence gate is still owed
  before shipping, but nothing about the change is numerically suspect.

---

## 6. What this says about the 24.93 ms bar (plow-internal reasoning only)

Recoverable, all measured or directly derived from the table above, none of it speculative
kernel work beyond what the trace already proves:

| lever | size | basis |
|---|--:|---|
| `MLA_MERGE_FOLD` occupancy (`VT`) | −1.99 ms | **measured, token-identical**, §5 |
| `MLA_MERGE_FOLD` fold loop (MLP/LDS) | ~−6.3 ms | derived: 4.72 MB/pkt at 50% of peak = 1.5 µs vs 85 µs |
| `FlashMlaDecode` dispatch fill + latency | ≤−2.4 ms | derived: 0.5% of peak today |
| MLA/`o_proj`/shared/`lm_head` bf16 → block-fp8 | −1.08 ms of *floor* | 13.36 GB of the 19.20 GB is bf16 |
| EP (`imoe_e` 512 → 2048) on the down-projections | ≤−2.0 ms | Loser #2: both downs at their own gate/up rate |

The 1-CU norms (2.32 ms) and the collectives (2.41 ms) are **not** where the win is, and §7a
already says fusing gates does not pay. The win is three kernels.

Cross-check on the collectives: §6g-KNOBS prices all 156 of them at 3.84 ms via `PLOW_NO_XREDUCE`.
This trace attributes **2.41 ms to the `XReduce` packets themselves**; the remaining ~1.4 ms of
that knob is the 75 `Residual` packets it also deletes plus the cross-rank skew the rendezvous
absorbs. The two numbers are consistent and measure different things — the knob prices the
*collective plus its graph*, the trace prices the *packet*.

---

## 7. Reproduction

```bash
# trace (each run: ~4-5 min weight load, then a 65-step median + one traced step)
gpulease -n 4 glm52-trace sg render -c '
  cd /home/lava/models/glm52_tp &&
  PLOW_TRACE_RAW=/tmp/glmtr/base ./glm52_decode glm52_tp4_64k.pkt \
      /home/lava/models/GLM-5.2-plow --tp 4 --sweep 1k --steps 65'

# attribute
python3 scripts/glm52_token_attrib.py /tmp/glmtr/base.insts.txt \
        /tmp/glmtr/base.tp4.ctx1024.bin --tp 4 --traced-ms 34.681 --csv /tmp/glmtr/base.csv
```

**Environment note worth adding to knob-contract §0a: the ROCR device index is NOT the `rocm-smi`
card index.** `gpulease -n 4` exported `ROCR_VISIBLE_DEVICES=1,2,3,4`; the four ranks physically
landed on `rocm-smi` **card0, card1, card2, card7**, while a foreign process leased as gpu0 sat on
card3. The lease is still sound (the sets are disjoint and the permutation is a bijection), but
**do not audit contention by matching lease ids against `rocm-smi` card ids** — match by VRAM
occupancy instead.
