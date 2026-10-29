# 27 — Zero-copy KV: shared pages for live and cached rows

**Status: design (2026-10-28), measured inputs, not implemented.** Builds on the KV block store
of `docs/arch/26-kv-block-store.md` (B.6a as implemented). Question: how to remove the copy a
prefix hit pays today and the second copy of every cached row a live slot holds, for every model
class plowrt serves (dense, sliding-window hybrids, MoE, recurrent hybrids, MLA) and every
backend (CUDA, HIP, CPU, later TP), without making a kernel step slower. Two families were
evaluated: (A) designs that keep today's flat per-slot addressing in the kernels, and (B)
kernel-native software pages. **Recommendation.** The design that scales is B — live and
cached rows share one pool of fixed-size pages (page = today's 32-row store block), the kernels
read through a per-sequence page table whose ids are resident before a work item starts, and the
store becomes an index over those pages. No family-A option scales: slot parking reaches 1–22 %
of hit tokens at realistic slot counts and, memory-bounded, either doubles copy traffic or loses up to 14 points of hit rate (§2), VMM aliasing
costs 40 µs per 2 MiB mapping (§4.1), and pointer swaps cannot share a window that the next token
writes. But B is also the only candidate that can slow a step (§4.4, §4.5), and on today's 26B
traffic the copies are ≈ 1 % of engine time. So: ship the flat-pipeline fixes first (F1, no
attention-kernel change, most of the measured cost), run the real-kernel paging gate beside it
(P0, kill criterion = any serving rung slower than the flat object beyond noise), and page
`Prefix`, then `Window`, only behind that gate (§6.3).

## 1. What a hit costs today

### 1.1 Bytes per attach

Every hit gathers the matched boundary's blocks into the slot's flat memory (`kv_blocks.rs`
`blocks_gather`, one `plow_kv_segcopy` launch). Per class, for a 5.7K-token resume (the
production-fidelity plan's mean, §2):

| packet | `Window` (sliding rings + fp8 scales) | `Prefix` (full layers) | `State` | per attach |
|---|---|---|---|---|
| Gemma-4 26B-A4B FP8 | 1024 × 104,000 B = 106.5 MB | 10,320 B/row → 59 MB | — | **165 MB** |
| Gemma-4 12B FP8 | 1024 × 166,400 B = 170 MB | 8,256 B/row → 47 MB | — | **217 MB** |
| Gemma-4 E4B BF16 | 512 × 40 KiB = 21 MB | 16 KiB/row → 93 MB | — | **114 MB** |
| Qwen3.8 27B BF16 | — | 64 KiB/row → 374 MB | 146.8 MiB (GDN 48 × 3 MiB + conv) | **528 MB** |

(Geometry from each checkpoint's `config.json` and the 26B server log: ring 2048 rows, window
1024, 128 slots. Qwen3.8's measured mean hit is 3.1K tokens, 357 MB.) A system-prompt hit
(first turn, 1.5–2.1K tokens) still copies the whole window: 122–128 MB on 26B.

### 1.2 Time per attach, and the engine-thread share

Store counters (`plowrt_kv_store_*`) over 25-minute production-fidelity replays (4 cells, rates
1.8 and 2.7), gpu03 H100:

| run | gathers | engine-thread µs / gather (of which drain) | µs / publish | gather + publish, share of wall |
|---|---|---|---|---|
| 26B FP8, store, host tier (`fix-g26-own2`) | 9,037 | 672 (127) | 920 | 0.39 % + 0.54 % |
| 12B FP8, store, no host tier (`grid-t0ao`) | 7,662 | 487 (26) | 1,658 | 0.24 % + 1.09 % |
| 12B FP8, store, host tier (`grid-uncap`) | 8,861 | **6,335** (236) | 1,070 | **3.75 %** + 0.61 % |
| Qwen3.8 BF16, exactness probe (`qwen-prefix/exact/bin2`) | 38 | 625 (555) | 1,129 | 0.12 % of tick time |

`gather_us` is engine-thread time (planning, segment-table upload, fence waits, launch, and a
drain outside a launch window), not GPU copy time; the copy itself is ≈ 0.1 ms of GPU for 165 MB
(§4.2). Two readings:

* Device-only, the copies cost **≈ 1 % of engine time** on today's traffic (5–9 requests in
  flight). Removing them is worth under a millisecond of TTFT per hit, not a throughput step.
* With the host tier, 12B's gathers cost **6.3 ms each on the engine thread** (host-resident
  blocks: 0.8M block restores per run). That is the largest copy cost measured, and it is a host
  transfer and scheduling cost that no device-side zero-copy design removes (§6.3, F1).

### 1.3 Device memory held twice

A live slot holds its own copy of every cached row it attached, and the store keeps the
original: 165 MB per in-flight 26B request (≈ 1.5 GB at 9 in flight, 4.5 GB at 27). On top, the
live sliding ring is a power of two ≥ window + chunk (2048 rows for a 1024 window), so a live
26B slot carries 200 MiB of ring for 101 MiB of window. Qwen3.8 is worst: 374 MB of `Prefix`
rows plus 147 MiB of state per session, held in both places while it runs.

## 2. Where the hit tokens come from (decides how far family A goes)

Production-fidelity plan (`agentic_turns.py --prod-fidelity`; `rate` = sessions/s, ~4.06 turns
per session), unbounded-cache attribution (method of `prefix_oracle.py`), 3 seeds × 1800 s:

| arm | req/s | oracle hit | same-session resume | cross-session (system prompt) |
|---|---|---|---|---|
| rate 1.8, think 45 s | 7.2 | 79.5 % | 88.1 % | 11.9 % |
| rate 2.7, think 20 s | 10.95 | 81.0 % | 89.2 % | 10.8 % |
| rate 2.7, think 10 s | 10.95 | 81.4 % | 89.5 % | 10.5 % |

The measured gpu03 replays agree (first turns hold 7.5–10 % of cached tokens). Two facts bound
any same-session-only scheme:

* **The next turn never resumes at the previous turn's end.** It re-renders the reply and
  diverges before it (`exec/gpu/prefix.rs:891`, 0 of 1223 attaches on Gemma-4), so it attaches
  at the previous *prompt* end `p` and re-prefills the reply. A resumed slot is usable only while
  its ring still holds rows `(p − w, p]` (reply ≤ ring − window rows), and for a recurrent model
  never: the in-place state has moved past `p`.
* **Think time is long against slot turnover.** A slot freed by a finished turn is needed again
  within 1–5 s at these rates; the next turn of its session arrives 10–45 s later.

Slot-park simulation (discrete-event over the same plans; `kv-zerocopy/hitcomp/a1_sim.py`,
`hybrid_sim.py`): share of all hit tokens a parked slot serves without a copy.

| arm | 32 slots (no end signal) | 64 slots (no end signal) | memory-bounded park, 44 GB, ring charged 2048 rows (H) | same, store-first eviction, ring charged at the window (H2sf) |
|---|---|---|---|---|
| 1.8 / 45 s | 1.1 % | 5.7 % | 1 % (no end signal) / 4 % (end signal), hit rate = store's | 28 / 37 %, hit −1.1 / −1.2 pt |
| 2.7 / 20 s | 0.5 % | 5.1 % | 1 % / 12 % | 32 / 44 %, hit −1.0 / 0 pt |
| 2.7 / 10 s | 3.2 % | 21.7 % | 1 % / 88 % | 64 / 88 %, hit −0.1 / 0 pt |

Under memory pressure a parked session must be demoted into the store before its slot is
reused — one ~165 MB scatter — so a park-first policy pays ≈ one demotion copy per request *on
top of* the gather (twice today's copy traffic) and stays near-empty; a store-first policy
charged at the real 2048-row ring loses 5–14 points of hit rate, because a parked slot costs
1.6× the bytes of the same session in the store. Only the variant that charges the parked ring at
exactly the window keeps the hit rate, and then serves 28–64 % of hits zero-copy. **Slot parking
is the retention cache in another form, with worse byte efficiency; it does not reach the
cross-session 11 % or recurrent models at all.**

## 3. How SGLang and vLLM 0.29 own KV

Sources read at SGLang `main` e0df901d681a (paths under `python/sglang/srt/`), vLLM v0.29.0
98dff2a81d74, flash-attention 98c1cee0ed59.

| | SGLang | vLLM 0.29 |
|---|---|---|
| device prefix hit | no KV bytes copied: `write_cache_indices` (`mem_cache/allocation.py:56-103`) writes the matched pool indices into `req_to_token[req, :prefix_len]` (int32, 4 B/token) | no KV bytes copied: `save_new_computed_blocks` → `BlockPool.touch` (`v1/core/block_pool.py:702-716`), `ref_cnt += 1` |
| default granule | `page_size = 1` (`arg_groups/overrides.py:1350-1372`); MLA backends force 32/64 (`overrides.py:1153-1209`) | `block_size = 16` (`config/cache.py:79,317`), raised for hybrid Mamba models (`platforms/interface.py:880-925`) |
| kernel consequence | the FA3 backend passes `req_to_token` rows as the page table (`layers/attention/flashattention_backend.py:924-1001`) | FA3 accepts multiples of 16 (`v1/attention/backends/flash_attn.py:112-116`) |
| sliding window | separate SWA pool, `full_to_swa_index_mapping` (`mem_cache/allocator/swa.py:41-153,477-514`); out-of-window tokens freed every extend (`mem_cache/common.py:75-115`); an evicted interior window is tombstoned and a hit truncates to the last point with a full live window (`unified_cache/components/swa.py:60-68,357-394`) | out-of-window blocks replaced by the null block but keep their hash, hittable until reused (`single_type_kv_cache_manager.py:632-668,911-1000`) |
| recurrent state on hit | **copied**: the matched checkpoint is the COW source of a fresh per-request slot (`unified_cache/components/mamba.py:197-226`), copied on the forward stream before the step (`model_executor/model_runner.py:1971-1986` → `MambaPool.copy_from`, `memory_pool.py:1163-1209`), then updated in place | `align` (default): **copied** from the cached block into the request's block before the step (`v1/worker/mamba_utils.py:1433-1537`, fused `precopy_mamba_align_fused_kernel`); `all`: the kernel gathers the initial state from the cached block and writes checkpoints out of place (`model_executor/layers/mamba/mamba_mixer2.py:856-935`) |
| host tier | HiCache: per-layer loads overlapped with the forward (`managers/cache_controller.py:1125-1170`, `LayerDoneCounter`), a warp-gather kernel reading pinned host memory (`kernels/aot/csrc/kvcacheio/transfer.cu`), loaded pages become shared tree pages | `--kv-offloading-size` → `OffloadingConnector`; `cuMemcpyBatchAsync` on a side stream (`v1/kv_offload/cpu/gpu_worker.py:512-680`), whole-request load before scheduling, no layer overlap |
| MLA | token-indexed latent pool (`memory_pool.py:4760-4872`), FlashMLA forces 64-token pages | FlashMLA 64, CUTLASS MLA 128, FlashInfer MLA 32/64 |

What they pay: neither copies attention rows on a hit, and both still copy recurrent state
(vLLM's `all` mode avoids the pre-copy at the price of materialising a gathered temporary).
Their kernels pay for fine pages: FA3 takes its TMA paged path only when `page_size % kBlockN ==
0` and the query side is prefill-like (`hopper/flash_api.cpp:415-424`); at page 1 or block 16 it
always runs `PagedKVNonTMA`, which forces PackGQA and disables clusters
(`flash_fwd_launch_template.h:212`). Page size is a kernel-performance parameter, not a cache
parameter: both projects decouple it from match granularity only partially (SGLang rounds
matches down to the page, vLLM hashes per block).

## 4. Measurements on this H100

All in one lease on gpu03 (`/opt/dlami/nvme/lava-tts/kv-zerocopy/probe/zc_probe.cu`, plain-shell
nvcc 12.9, driver 595.91).

### 4.1 VMM map/alias latency

`cuMemGetAllocationGranularity`: minimum = recommended = **2 MiB**. Per call, median over 3 ×
64 extents (32 at 128 MiB):

| extent | `cuMemCreate` | `cuMemMap` | `cuMemSetAccess` RW (p90) | RO alias | `cuMemUnmap` (p90) |
|---|---|---|---|---|---|
| 2 MiB | 16 µs | 0.6 µs | 39 µs (49) | 42 µs | 22 µs (24) |
| 8 MiB | 15 µs | 0.6 µs | 39 µs (44) | 40 µs | 22 µs (24) |
| 32 MiB | 15 µs | 0.7 µs | 40 µs (55) | 40 µs | 23 µs (31) |
| 128 MiB | 16 µs | 0.7 µs | 42 µs (70) | 42 µs | 23 µs (32) |

* The cost is **per mapping, not per byte**: one `cuMemSetAccess` over 64 contiguous extents
  takes 2.5 ms (= 64 × 40 µs) and one `cuMemUnmap` over them 1.45 ms; batching the range saves
  nothing. Read-only access is accepted and costs the same.
* A map + access issued while a 22 ms kernel runs on another stream did not wait for it (44 µs a
  pair) and did not slow it.
* Applied to 26B FP8: a sliding window cannot be aliased (a slot's ring is written by the next
  token, so the whole ring would need a COW). The full-layer rows of a 5.7K resume are 20 head
  tracks × 2 extents of 4096 rows: 40 × 40 µs = **1.6 ms to map plus 0.9 ms to unmap later**,
  against **≈ 0.1 ms of GPU** to copy the same 59 MB. Moving a whole parked session between
  slots by remapping (50 ring tensors × 2 granules + 40) is ≈ 5.7 ms + 3 ms. The vAttention
  figures for A100 (map 2 µs, access 38 µs, unmap 34 µs) carry over almost unchanged; VMM aliasing
  is a capacity tool, not an attach path.

### 4.2 Copy bandwidth (D2D scatter/gather, H2D)

| payload | CE contiguous | SM gather, 8 KiB segments | SM gather, 16 KiB | `cudaMemcpyBatchAsync`, 8 KiB | H2D pinned (CE) | D2H pinned (CE) | SM read of mapped host |
|---|---|---|---|---|---|---|---|
| 3.3 MB | 7 µs | 6 µs | 6 µs | 310 µs | 0.23 ms | 0.37 ms | 0.25 ms |
| 13.3 MB | 11 µs | 10 µs (1.35 TB/s) | 9 µs | 1.18 ms | 0.92 ms | 1.56 ms | 0.99 ms |
| 35 MB | 30 µs | 30 µs | 30 µs | 3.06 ms | 2.42 ms | 4.08 ms | 2.61 ms |
| 106.5 MB | 77 µs (1.38 TB/s) | 80 µs (1.33 TB/s) | 80 µs | 8.69 ms (12 GB/s) | 7.36 ms (14.5 GB/s) | 12.6 ms (8.5 GB/s) | 7.92 ms (13.4 GB/s) |

* A device gather is **bandwidth-cheap**: the whole 26B attach (165 MB) is ≈ 0.12 ms of GPU,
  12B's 217 MB ≈ 0.16 ms, Qwen3.8's 528 MB ≈ 0.4 ms. The SM gather kernel matches the copy
  engine on contiguous data; segment size does not matter above 8 KiB.
* The batched copy-engine API is host-bound at ≈ 0.65 µs per segment (12–20 GB/s): it is not a
  gather mechanism for 32-row blocks.
* **The host link is the expensive copy**: 14.5 GB/s H2D and 8.5 GB/s D2H on this box (doc 26
  assumed ~25 GB/s); a host hit of a 26B session is ≈ 11 ms, a 12B one ≈ 15 ms, which is why
  12B's host-tier gathers cost 6.3 ms of engine thread each (§1.2). Reading mapped host memory
  from the SMs runs at the same 13.4 GB/s.

### 4.3 Interference of a hidden copy with a bandwidth-bound decode

Twenty back-to-back synthetic decode reads (B32, ctx 8K, 2.1 GB each; 13.87 ms alone) with a
copy started on a second stream:

| concurrent copy | decode × 20 | Δ |
|---|---|---|
| none | 13.87 ms | — |
| 106.5 MB D2D, copy engine (0.41 ms) | 14.01 ms | +0.13 ms |
| 106.5 MB SM gather, 8 KiB segments (0.61 ms) | 13.99 ms | +0.12 ms |
| 4 × 106.5 MB D2D, copy engine | 14.19 ms | +0.32 ms |
| 106.5 MB H2D (7.4 ms) | 14.11 ms | +0.24 ms |

A hidden copy costs a bandwidth-bound step about its own HBM traffic (≈ 1.2 µs per MB copied):
hiding moves the copy off the critical path but does not make it free, and a copy-engine copy
does not beat an SM gather on interference.

### 4.4 Address indirection in a head-major decode read

Synthetic decode read with today's geometry (512-byte rows, 8 KV heads, 256-row tiles, 4 warps,
online softmax over bf16 K/V): flat addressing against a per-sequence page table, page base
loaded per row or hoisted once per page, pages laid out in order or shuffled across the pool.

µs per launch (median of 7 × 50); identical outputs between flat and per-row lookup:

| B × ctx | flat | P=32 base per row | hoisted per page, in order: P=32 / 64 / 128 / 256 | hoisted, shuffled pages: 32 / 64 / 128 / 256 |
|---|---|---|---|---|
| 1 × 1K | 18.1 | 20.4 | 18.5 / 18.0 / 17.8 / 17.6 | 18.4 / 18.0 / 17.8 / 17.7 |
| 1 × 8K | 31.7 | 33.5 | 35.6 / 35.4 / 35.2 / 35.1 | 35.4 / 35.5 / 35.2 / 35.1 |
| 8 × 1K | 31.6 | 33.6 | 35.6 / 35.3 / 35.3 / 35.2 | 35.4 / 35.8 / 35.1 / 35.0 |
| 8 × 8K | 176.7 | 176.0 | 178.7 / 178.7 / 179.9 / 180.0 | 182.9 / 183.8 / 180.6 / 180.9 |
| 32 × 1K | 92.3 | 92.6 | 94.0 / 93.7 / 94.7 / 95.2 | 94.0 / 94.4 / 94.5 / 94.4 |
| 32 × 8K | 693.8 | 687.4 | 680.7 / 681.5 / 686.1 / 683.8 | 685.4 / 687.7 / 686.5 / 685.5 |

* **Bandwidth is not the cost**: at B32 × 8K (2.1 GB per launch) every paged variant is within
  1 % of flat, shuffled pages included, and page size 32 → 256 changes nothing.
* **Latency is**: where a work item is one 256-row tile (B1 × 8K and B8 × 1K split into 66
  items of 256 rows to fill 132 SMs), the page-id load is a dependent global load in front of the
  item's first row and costs **+11 %** even with P = 256, more than a per-row lookup the warps can
  overlap (+6 %). Hoisting per page only helps when the item spans many pages.
* So the design point is not "hoist per page" but **"page ids resident before the item starts"**:
  loaded with the query (they are per sequence, not per head), or prefetched for the next item —
  the persistent decode body already computes the next item's base early on its bulk-prefetch
  path (`op_attention.cuh:2433-2449`). This is the P0 kernel question, and §4.5 says the answer
  must also hold at whole-object level.

### 4.5 The existing per-row slot indirection on the production decode

`PLOW_NV_ROW_SLOT` (`op_attention.cuh:2381-2386`, `op_norm.cuh:1347`, `interp_sm120.cu:49-53`)
already replaces `slot = b` with `in.rowslot[b]` once per work item in every decode reader and
the fp8 writer. Its first deployment measured +0.6..1.5 % on plain decode (commit `0282348a3`),
which is why span-verify packets carry `_rs` twins. Re-measured here: Kit 2.2 26B FP8 packet,
`step_bench`, the shipped decode objects against their `_rs` twins, 2 interleaved rounds.

| B × ctx | shipped object, ms/step (r1, r2) | `_rs` twin, ms/step (r1, r2) | Δ |
|---|---|---|---|
| 1 × 1024 | 4.282, 4.286 | 4.322, 4.326 | +0.9 % |
| 8 × 1024 | 6.424, 6.422 | 6.493, 6.498 | +1.1 % |
| 32 × 1024 | 13.053, 13.061 | 13.087, 13.087 | +0.2 % |
| 1 × 8192 | 4.356, 4.357 | 4.404, 4.407 | +1.1 % |
| 8 × 8192 | 7.912, 7.919 | 8.026, 8.025 | +1.4 % |
| 32 × 8192 | 15.767, 15.768 | 15.914, 15.921 | +0.9 % |

Token digests identical on every rung; rounds agree to 0.1 %. The cost is real and it is **not
the lookup**: at B1 attention is a few percent of the step, yet the step loses 0.9–1.1 %. One
runtime-checked indirection compiled into the persistent interpreter perturbs code generation
across the whole object (register allocation, the branch at every addressing site). The lesson
for any indirection — slot table or page table — is that it must be a compile-time property of
a separate object (as the `_rs` twins already are) and gated per rung, never a runtime branch
in the shared body.

### 4.6 The "paging is a measured null" record

`crates/devgen/src/lib.rs:2723` ("a byte-repack (token-major, or vLLM-style paging) is a
measured null here", cited by doc 26 B.9 at a stale line) predates the repository's first
public commit (`b84deb91b`, 2026-07-25); no experiment, rung or number survives in the tree,
the docs or the plans, and the only nearby rationale (`runtime/common/dev_isa.h:1608-1625`)
argues token-major against head-major row strides. It concerns *re-packing bytes* for decode
bandwidth, which head-major already wins; it says nothing about address indirection with
head-major rows kept inside a page, and it is not evidence for or against this design. Two
measured data points do exist: token-major interleaved rows in production for Qwen3.8 cost
−0.6..−0.9 % of tok/s for allocation granularity (`recipes/infervisor/qwen3.8-27b/
sm90a-h100-tp1.toml:191-196`), and the per-item slot indirection of §4.5.

## 5. Candidates

Scores against today (block store, copy per hit). "Copies removed" counts hit-path bytes on the
production plan; "risk" is kernel step-time risk.

| | mechanism | copies removed | memory saved | step-time risk | model coverage | backends | effort |
|---|---|---|---|---|---|---|---|
| A1 park in slot | keep a finished slot's rows in place for the session's next turn | 1–5 % of hits at 32–64 slots; 28–64 % memory-bounded at −1 pt hit (§2); none for state | negative unless the ring shrinks to the window | none while a resumed slot sits inside the live rung; anywhere else it needs the row-slot table (+0.2..1.4 %, §4.5) | dense and SWA; not recurrent (state moved past `p`), not cross-session | all (exists as session retention without the VMM store, `serve/session.rs`, `gpu.rs:8253`) | small |
| A2 per-slot / per-class base table | one base pointer per CTA; hit = swap a whole ring or state buffer | a whole-window swap still needs a whole-window COW at the first write (decode writes into the ring) → none for `Window`; state: see B-State | none | +0.2..1.4 % as compiled today (§4.5) | narrow | all | small |
| A3 VMM aliasing | map store pages into slot VA | full-layer extents only (2 MiB = 4096 rows per head track); windows need a whole-ring COW | `Prefix` duplicates, coarse | none | dense full layers, MLA latent | CUDA, HSA (KFD unmap ~120 µs); not CPU | medium; already built and removed once (B4) |
| A4 lazy COW of forked windows | share a window until the first write | needs page granularity to be lazy; whole-ring otherwise | — | — | — | — | needs B |
| A5 hide copies | restore ahead of admission, paced spills, layer-wise host loads | none (moves them off the critical path) | none | interference §4.3 | all | all | small–medium |
| **B1 shared pages** | live and cached rows in one page pool; kernels read through a page table, page ids resident before a work item starts; `Window` = ring of pages with tail clone; `State` = out-of-place first update | all device-hit row copies; host hits load into pages that become shared | every duplicate; live ring shrinks to window + chunk pages | §4.4 + per-rung gates | all classes, MLA natively | CUDA, HIP, CPU, TP (same page ids per rank) | large |

## 6. Recommended design

### 6.1 Shape

* **One page pool per class**, page = `P` rows of every tensor of the class, laid out
  `[tensor][head][P rows][hd]` plus `[tensor][head][P] f32` scales — exactly today's store block
  layout at `P = 32`, so a store block *is* a page. The store keeps its index, chain hashes,
  states, dependents, holds, host tier and eviction; it stops owning a second copy.
* **A sequence is a page table** per class (`u32` page ids, logical page `j` = rows
  `[jP, (j+1)P)`). Writers allocate the pages of a chunk before its launch (today's
  `ensure_rows`). Publish seals the pages up to the boundary (refcount, no copy). Attach installs
  the boundary's page ids. A sealed page is never written.
* **`Window(w)` is a ring of pages.** The table keeps the last `⌈(w + chunk)/P⌉ + 1` logical
  pages; a page leaving the window is released unless a boundary retains it. The `row & mask`
  ring disappears from the kernels; a live sequence holds window + chunk rows, not the next
  power of two.
* **Boundary tail.** Boundaries stay 32-aligned (hash granularity unchanged). With `P > 32` an
  attach at `p mod P ≠ 0` clones the partially filled shared page before the first write: at most
  `P − 32` rows (26B: 10 MB window + 1 MB rows at `P = 128`), against 165 MB today.
* **`State` out of place.** The packed GDN prefill already reads per-sequence initial states from
  a contiguous staging buffer the engine gathers into on every launch (`gdn_prefill.cpp:54-82`,
  `qwen_gdn.rs:239`); an attached sequence's first chunk gathers from the store checkpoint
  instead of the slot. The attach copy disappears with no kernel change; the decode step
  (`op_qwen_gdn.cuh:167-186`, in place, already slot-indirected) is unchanged. Speculative
  rollback keeps restoring from the checkpoint, as B3 does.
* **Kernels.** Decode: the per-item base (`op_attention.cuh:2415-2421`, fp8 bases 2476-2488)
  becomes a per-page base. The page ids of an item are loaded before its first row is needed —
  with the query, or for the next item while the current one streams (the bulk-prefetch path
  already computes the next item's base, 2433-2449) — because a dependent id load in front of a
  one-tile item costs +11 % (§4.4); split ranges are aligned to `P` so a tile crosses at most one
  page (the split start at 2397-2401 is unaligned today). Paged bodies are separate objects
  selected per packet, never a runtime branch in the flat body (§4.5). Prefill:
  the per-slot rank-3 tensor maps (`gpu.rs:2030-2114`) become one rank-4 map per class over the
  pool, `[hd, P, kvh, pages]`, with the page id as the fourth coordinate per tile (needs `P ≥
  box_rows`); the descriptor-free path already issues one `cp.async.bulk` per row and takes a
  per-row page base. Writers (`op_norm.cuh:1123-1143, 1347-1360`) index the page table at the
  row they write. MLA latent rows are token-major with one head, which is the natural page
  layout; the AMD MLA decode already reads rows through an index (`GATHER`, DSA top-k).

### 6.2 Coverage

| class | `Prefix` | `Window` | `State` | notes |
|---|---|---|---|---|
| dense (Gemma-4 full layers, Qwen full-attention) | pages | — | — | Qwen's interleaved rows are one track of `heads × hd`, paged the same way |
| sliding hybrids (Gemma-4 26B/12B/E4B, KV-shared layers) | pages | ring of pages | — | KV-shared layers read the source layer's table |
| MoE (26B-A4B, GLM, Kimi) | as their attention | as their attention | — | experts do not touch KV |
| recurrent hybrids (Qwen3.8 GDN, Kimi KDA) | pages | — | checkpoint read by the first chunk | capture op unchanged (B3) |
| MLA (GLM-5.3, DeepSeek, Kimi) | latent + rope (+ index, scale) pages, one head | — | — | replaces the AMD `SharedPrefix` multi-map |

| backend | what changes |
|---|---|
| CUDA (H100) | decode bodies, prefill maps, writers, `kv_segcopy` for tail clones and host loads |
| HIP (MI300X/MI350X) | the same addressing in `runtime/amd` (`op_attention_common.h:570-724`), HSA VMM no longer needed for sharing; MLA already index-addressed |
| CPU (Xeon) | the donor-slot `copy_nonoverlapping` (`serve/cpu_serve.rs:446-487`) becomes a table install; a 165 MB copy is ≈ 2–3 ms of a socket's memory bandwidth per hit, twenty times the GPU's |
| TP > 1 | each rank pages its own head shard; one page-id table is valid on every rank if pools allocate in lock step; the store key carries the rank (doc 26 B.8) |

### 6.3 Phases

Two tracks. Track 1 ships the measured savings with no attention-kernel change; track 2 is the
paging decision, gated on a real-kernel measurement that may stop it.

| phase | content | gate | expected effect |
|---|---|---|---|
| **F1** flat pipeline (≈ 1–2 weeks, ships first) | (a) host-resident blocks restored ahead of admission onto device slots on the restore stream, never read in place by a step's gather, and the engine-thread part of a host gather (12B: 6.3 ms) attributed and moved off the thread; (b) the device gather's engine-thread overhead decomposed and cut (0.67 ms against ≈ 0.12 ms of GPU for 165 MB: segment-table rebuild and synchronous upload, fence planning, the drain outside a launch window) and the publish drain (0.92 ms) made asynchronous behind the scatter's event; (c) `State` attach through the first chunk's initial-state gather (one copy instead of two) | replay engine-thread share and TTFT p50/p99 against today's build at rates 1.8/2.7, think 10/20/45 s; Qwen exactness + FP32 gate | 26B: most of the 0.93 % engine-thread share; 12B with host tier: up to 3.75 %; Qwen3.8: −147 MiB copy per attach |
| **P0** paging gate (≈ 1 week, runs beside F1) | real persistent objects, compile-time paged variant of the decode bodies, the writers and the prefill stager; identity, shuffled and fragmented tables; page ids loaded with the query and prefetched for the next item; first-item cold vs warm table; realistic page-boundary offsets; `P ∈ {32, 64, 128}`; the same object with paging compiled out as the codegen control; `step_bench` B1/8/32/64/128 × ctx 1K/4K/8K/16K, the one-tile-item rungs (B1 × 8K, B8 × 1K) first; prefill throughput; `ptxas` resources and SASS | **kill** if any serving rung is slower than the flat object by more than the run-to-run noise (≈ 0.2 %, §4.5) with an identity table | decides whether track 2 exists, and `P` |
| **P2** paged `Prefix` (≈ 2–3 weeks, only if P0 passes) | full-attention rows and MLA latent rows: pool = store slabs, publish = seal, attach = install, no tail (P = 32) or a tail clone | P0 gates on the full packet; exactness (logits equal to the copy path); lifecycle faults | removes 59 / 47 / 93 / 374 MB per attach (26B / 12B / E4B / Qwen) and the live duplicates; long-context and MLA attach become O(table); Qwen BF16 ≈ 0.4 GB more cache per in-flight request |
| **P3** paged `Window` (≈ 2–3 weeks, only if P2 holds in production) | ring of pages; `kv_ring` proof replaced by a page-table obligation (live pages distinct per writer, no write to a sealed page, no reuse before the last reader's event) | as P2 plus window-edge sweeps (rows 1023/1024/1025, 4094/4095/8191), speculative rollback across a page edge | removes the 106–170 MB window gather (65–78 % of attach bytes) and the 2× ring (≈ 100 MiB per live 26B slot); the riskiest phase: windows are short, latency-bound reads (§4.4) |
| **P4** | HIP (dense + MLA, replaces `SharedPrefix` and the per-slot route), CPU (table install instead of the donor memcpy), TP (lock-step pools) | per-backend rung + replay gates | one cache design on every backend |

Production effect, stated plainly. On today's 26B traffic the copies are ≈ 1 % of engine time
and 1–2 GB of device memory, and plowrt already leads vLLM 0.29 on the same stream at 20 s and
45 s think (doc 26 B.6a: 78.4 / 71.7 % hit against 72.6 / 60.6 %, lower TTFT and TPOT). F1
removes most of the measured cost without touching a kernel. Paging adds structural parity with
vLLM — no hit copy, no duplicate rows — and its margin is where copies are large: Qwen3.8 (528 MB
per attach), long contexts (the `Prefix` copy grows linearly, the window does not), MLA models,
CPU serving, and forks (parallel sampling, agent branches) that copy a whole window per branch
today. The accounting that decides it is `extra decode + extra prefill + page management <
exposed copy time removed`, in the same denominator; on 26B a decode regression of a few tenths
of a percent already cancels it. Neither track moves the hit rate by itself (doc 26 §1.2:
capacity and session-end signals do); freed device memory is worth ≈ +1–4 points at 20–45 s
think without a host tier (the simulation's 41 → 48 GB column) and little with one.

### 6.3a Review findings folded in

An adversarial review of the measured plan raised these; each is now part of the design or the
gates:

* **Indirection cost is codegen as much as latency.** Separate objects remove the runtime branch,
  not register-allocation or scheduling changes across the persistent interpreter (§4.5); P0
  therefore compares against the paged object with paging compiled out, and inspects SASS for
  where the id loads land and how long they stay live. Preloading the *first* item's ids during
  the previous operation creates cross-operation live registers: measured, not assumed.
* **TMA over pages.** A rank-4 map `[hd, P, kvh, pages]` is legal on H100 (32-bit coordinates,
  box dimensions 1–256, inner box bytes ≤ the swizzle span), but the box must not cross a page:
  `row offset + box_rows ≤ P`. The producer loads upcoming page ids while consumers compute; K, V
  and the f32 scales keep separate maps (one per tensor family, not one per class). Small `P`
  multiplies transfers and barriers; the descriptor-free per-row bulk path is the fallback.
* **`P` and publication.** Boundaries are 32-aligned and publishes happen per prompt end; with
  `P > 32` almost every attach would clone a tail page and every publish would seal a partial
  page. `P = 32` (page = store block) is the default unless P0 shows a larger page is needed for
  the kernels, in which case partial pages stay private and only whole pages are shared.
* **Window ownership.** `⌈(w + chunk)/P⌉ + 1` bounds one sequence's live span, not physical
  ownership: retained boundaries, speculative writes (`chunk` includes the maximum draft rows),
  partial pages and in-flight readers are charged too. A page is recycled only after the last
  reader's event; a sealed page is never a draft target; rollback restores the table and the
  recurrent state, and an accepted prefix inside a speculative page keeps a valid length.
  Page allocation per crossing is engine-thread work: allocated per chunk ahead of the launch,
  from a per-class free list, never per step.
* **`State` through the staging gather removes one copy, not both.** The staging copy (store
  checkpoint → staging) remains; the checkpoint → slot copy goes. Requirements: the checkpoint is
  the state at exactly `p` (B3's chunk cut already captures it there), conv history travels with
  it, the attached sequence's first launch is a prefill chunk (true for every resume: the reply
  and the new user turn are re-prefilled), and the single-sequence path keeps today's copy.
* **Host hits.** Paging cannot remove the PCIe transfer (14.5 GB/s: 100 MB ≈ 6.9 ms), and a
  host-mapped page read in place by every decode step would pay it repeatedly; host pages are
  restored once, ahead of admission. Layer-wise restore needs per-layer readiness inside the
  persistent launch, so it is a kernel change, not an F1 item.

### 6.4 Packet ABI additions (runtime contract 3)

Generic, per cache tensor in the live-KV manifest (`plow-asset/src/live_kv.rs`), no model
names:

* `class`: `prefix` | `window{w}` | `state` (today inferred by the runtime from `window`/`mask`).
* `paging`: `{page_rows P, pool tensor, scale pool tensor}`; `mask`/`stride` keep their meaning
  for flat packets, and a paged packet declares `stride = P`, no `mask`.
* Per-launch inputs: `in.page_table.<class>` (`[rows][max_pages] u32`) and, for writers, the
  write rows' page ids; replaces `in.rowslot` for paged classes.
* `recurrent_state`: per tensor, whether the first chunk accepts a separate initial-state source
  (the packed GDN path does).
* Object capability symbol `plow_kv_paged_abi = 1`; a packet or object without it serves through
  the flat layout and the block store's copies, unchanged.

### 6.5 P0 result (2026-10-28): paging gate FAILS, track 2 stops

Measured on the 12B FP8 packet (`interp_sm90a_fp8kv` with its recipe flags, RG8/RG8S hd256
sliding and TC16 hd512 decode bodies), H100, `step_bench` 64 steps after 8 warm-up, ABAB rounds.
Arms are the same object built three ways: `flat`; `pg32` (`PLOW_NV_KV_PAGED=32`: row =
`table[slot][r/32] + r%32`, head-major inside the page, page id loaded per row from an L1-resident
module table); `pg32r` (`PLOW_NV_KV_PAGED_REG=1`: the item's page ids, up to 64, loaded with the
query into registers and broadcast by warp shuffle). With an identity table both paged arms give
token-stream digests identical to `flat` at every rung and round (an address-only change). Shuffled tables
(`step_bench --kv-pages 1|2`) were not measured: the identity table already fails the gate.

| rung | flat ms/step | pg32 | pg32r |
|---|---|---|---|
| B1 × 8K | 7.341 (spread 0.08 %) | 7.485 (+1.97 %) | 7.516 (+2.39 %) |
| B8 × 1K | 7.931 | 8.184 (+3.19 %) | 8.225 (+3.71 %) |
| B32 × 8K | 12.865 | 13.266 (+3.12 %) | — |
| B32 × 1K | 11.769 | 12.114 (+2.93 %) | — |

B1, B8: three ABAB rounds; B32: two. Round-to-round spread ≤ 0.35 % in every arm (flat ≤ 0.1 %), an
order of magnitude below the deltas. B1 × 16K did not run (prompt + steps exceed the compiled
16,384 context); it was not needed for the verdict.

Resources (`ptxas`): the persistent interpreter stays at 255 registers with no local memory
(spills 0), stack 240 → 256 bytes, shared 40,592 bytes unchanged; the standalone rider bodies move
148 → 158 / 142 registers (`rider_flash512`, pg32 / pg32r), 104 → 106 / 128 (`rider_flash256`),
103 → 104 / 128 (`light_fp8_flash256`).

Both first rungs exceed the kill threshold (≈ 0.2 %) by 10–18×, and loading page ids up front
is slower than the per-row lookup, so the cost is not the table load latency but the extra
address arithmetic and register pressure in the 255-register persistent body. Per the §6.3 rule
the writers, prefill stager and BF16 variants were not built: track 2 (P2/P3) stops, and the
flat pipeline (F1) is the KV design. Re-opening it needs a decode body with register headroom
(a non-persistent attention kernel), not a different page size.

### 6.6 F1 result (2026-10-29)

Production-mix replays, F1 against its base in one lease (2 reps, one rate per model; strict
reports against the vLLM 0.29 arms). Store time on the engine thread, as a share of tick time:
12B FP8 with the 64 GiB host tier 13.0 → 0.53 % (gather 8.9 ms → 63 µs), 26B FP8 3.06 → 1.37 %
(gather 715 → 58 µs; publish drain 89 µs → 0), Qwen3.8 BF16 4.35 → 1.50 % (gather 7.1 ms → 86 µs).
The 12B gain reaches the tail: TTFT P99 1.02x → 0.73x vLLM, TPOT P99 0.74x → 0.50x, both reps.
26B and Qwen3.8 P99s move within their rep-to-rep spread. Attached-prefix outputs are
bit-identical to the copy path with the matmul algorithms pinned (Qwen3.8 BF16 and 26B FP8,
40 of 40 cached turns); FP32 gates pass. The largest single cost had been the stream drain of
a gather outside a launch window, not the copy itself. Evidence:
`/opt/dlami/nvme/lava-tts/kv-f1p0/` (`rep/`, `gates/`, `l2/`).

## 7. Validation

0. **F1.** Kernels untouched: `step_bench` digests identical to the base packet; replay
   engine-thread time per gather/publish and TTFT p50/p99 against today's build, same stream; the
   host-tier arms at 10 s think (where doc 26 B.6a measured the tier's tail cost).
1. **Per rung.** `step_bench` digests identical between the flat and the paged object with an
   identity table (address-only change: bitwise-equal KV, scales and logits), then shuffled and
   fragmented tables; B1/8/32/64/128 × ctx 1K/4K/8K/16K and the 4094/4095/4096/8191/8192
   boundaries (`step_boundary.sh`); `op_roof.py` against the calibrated roofline; `ptxas`
   registers, spills, stack and shared memory per arm. Step time within the run-to-run noise at
   every rung, or the rung keeps the flat object.
2. **Exactness.** Attach-then-decode logits equal to the copy path's at ctx 1000–12000 across
   window and block edges, multi-turn with sibling branches (doc 26 §4.1 protocol); on `State`
   models the cold-then-attached probe.
3. **Lifecycle.** Page refcounts under eviction, spill, restore, cancellation mid-attach,
   speculative rollback across a page edge, slot reuse; poison sweeps.
4. **Numerics.** FP32 gate at c1/c8/c32 (trimmed mean / p95 / over-limit; p99 informational).
5. **Serving.** Production-fidelity replay against vLLM 0.29 (prod flags, with and without
   offload) at rates 1.8/2.7 and think 10/20/45 s, 3 repetitions; final numbers only from
   `campaign.py report` in the strict format; report copy bytes, engine-thread time, device bytes
   per class, TTFT/TPOT p50/p99 and goodput.

## 8. Open questions

1. `P` per class: the P0 measurement decides; 32 makes a page a store block and removes the tail
   clone, 128 suits the TMA box and the decode tile.
2. Whether to keep the flat path for packets that never cache (ASR/TTS), or page everything for
   one code path.
3. Speculative decoding: drafts write past the committed row; with pages a rejected draft frees
   at most one page, and a draft never writes a sealed page — to be proven in the page-table
   obligation.
4. Whether the production think distribution (doc 26 P0(b)) makes a session-end signal common;
   it decides how much the freed memory is worth.
5. Whether long-context traffic (16K–128K, where the `Prefix` copy grows linearly and MLA models
   live) is expected soon; it raises track 2's value independently of today's 5.7K mean.

Evidence (campaign scratch, outside the repository): probe source and outputs
`/opt/dlami/nvme/lava-tts/kv-zerocopy/probe/` (`zc_probe.cu`, `probe1.out`, `probe2.out`,
`rs_ab/`), hit-composition and park simulations `/opt/dlami/nvme/lava-tts/kv-zerocopy/hitcomp/`,
store-counter scan `/opt/dlami/nvme/lava-tts/kv-zerocopy/attach/scan.py`, upstream checkouts
`/opt/dlami/nvme/lava-tts/kv-zerocopy/upstream/`.
