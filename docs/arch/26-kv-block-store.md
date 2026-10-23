# 26 — KV block store: a content-addressed prefix cache under flat slots

**Status: design (2026-10-23), Codex-consulted, not implemented.** Replaces the boundary-snapshot
prefix cache of `docs/arch/22-kv-cache.md` §7–§9 and the AMD `SharedPrefix` with one store.
Model kernels are unchanged.

## 1. Why

Production Gemma-4 26B-A4B FP8 on gpu04, from the engines' own metrics:

| | req/s | TTFT mean | ITL mean | prefix hit (tokens) | uncached tok/req |
|---|---|---|---|---|---|
| plowrt Kit 2.1 (MTP on), 10-21 day avg | 5.85 | 110 ms | 16.3 ms | 74.4% | — |
| plowrt Kit 2.2 (MTP on), 10-22 06:22–07:02 | 10.2 | 142–370 | 19.8–22 | 63–65% | 2,002 |
| vLLM 0.29 (prod flags), 10-22 day avg | 6.2 | 116 | 13.6 | 80.8% | ~1,100 |

These are different days and different arrival rates, so they rank nothing by themselves. On gpu03
with the **same recorded request stream** (`scripts/bench/agentic_turns.py --open-loop
--prod-fidelity`: production prompt shape, think median 10 s; two interleaved rounds agree
within noise):

| rate 1.8 → 6.5 req/s | TTFT p50/p99 | TPOT p50/p99 | hit |
|---|---|---|---|
| vLLM 0.28 | 82 / 371 | 11.5 / 30.8 | 75.7% |
| plowrt, MTP off | 41 / 271 | 10.8 / 23.0 | 74.4% |
| plowrt, MTP on | 40 / 238 | 5.2 / 19.1 | 74.6% |

At rate 2.7 (9.2–9.7 req/s) vLLM 0.28 saturates (TPOT p99 227 ms, goodput 81%) while plowrt
holds TTFT 65/438, TPOT 11.6/45.9, hit 74.8%. The vLLM 0.29 arms are queued (`ab.sh`,
`ab-long.sh`). Per token plowrt leads; per retained session it does not, and that is what
production at 10 req/s exposes. The cause is the snapshot cache's insertion cost per turn, below.

### 1.1 Where the bytes go today (26B FP8, `exec/gpu/prefix.rs:682-751`)

Per token: 25 sliding layers × 2 × 8 heads × 256 B = 100 KiB, +1.6 KiB FP8 ring scales; 5 full
layers × 2 × 2 heads × 512 B = 10 KiB, +80 B scales. A boundary snapshot at prompt end `p` is

```
106.5 MB  sliding windows: 1024 rows × 104,000 B       (fixed, every turn, never shared)
 ≤ 42 MB  full-layer rows p mod 4096 × 10 KiB           (private sub-block tail)
 ~0.5 MB  full-layer scale prefix [0, p) × 80 B
```

plus whole 4096-row full-layer VMM blocks shared by multi-mapping (40 MiB per block over the
10 tracks × 2 heads). Prod counters: 41 GB cache = 31.7 GB snapshots (261 live, ~121 MB) +
9 GB shared blocks. Retire-on-publish keeps one boundary per session.

### 1.2 Capacity model: retention time, not occupancy

Eviction is LRU over boundaries. A boundary survives until `capacity` bytes of newer insertions
have passed it, so its retention time is

```
T = cache_bytes / (turn_rate × bytes_inserted_per_turn)
hit ≈ ceiling × P(think ≤ T) + shared-prefix hits
```

where `ceiling` is the stream's own repeat fraction and `think` is the measured turn-gap
distribution, not the harness's. For the replay plan the unbounded-cache oracle
(`/home/lava/scratch/kv-store-p0/oracle.py`: exact prefix match, 32-row floor plus the 9-row
prompt-end slack) is **81.1%** at rates 1.8 and 2.7 (81.75% unquantised); both engines measured
~5 pt below it at 6.5 req/s with 10 s think, so even the short replay is retention-bound, not at
the ceiling. Everything the design changes is in the denominator:

| engine | bytes inserted per turn | T at 10 req/s, 41–48 GB |
|---|---|---|
| plowrt today | a new 106–121 MB snapshot **every** turn, overlap or not | ~34 s |
| vLLM 0.29 | new tokens × ~120 KiB: ~12–36 MB for a continuing session, the whole prompt on a miss | ~80–100 s (pool size to be measured, §4.0) |
| plowrt + store | new rows × 114 KiB: ~11–34 MB continuing, ~165 MB for a new session | ~80–100 s (same mechanism) |

That is the whole gap on identical traffic: today a continuing session's turn displaces another
session's entire boundary; vLLM's and the store's turns displace only their new rows. (The
earlier `S = λ·L` occupancy framing predicted 23–47% for the measured 63–65%; it is wrong
because retention is set by insertion churn, not by population.)

**Both engines share the per-session byte floor.** An independent 5.7K-token session's state is
the window (106 MB) plus `Prefix` rows (5.7K × 10.3 KiB = 59 MB) ≈ 165 MB in both engines once
vLLM's hybrid manager has freed the sliding rows outside the window
(`SlidingWindowManager.remove_skipped_blocks`, vLLM 0.29 `v1/core/single_type_kv_cache_manager.py:632`:
freed blocks keep their hash in the free queue in free order, so under pressure they are the
first of that request's blocks to be reused). The window cannot be rebuilt from fewer rows — each
sliding layer's window depends on the previous layer's window back to row 0 — only re-prefilled
from the tokens. So the design does not shrink the floor. It (a) matches vLLM's incremental
insertion, (b) extends retention into host memory for both classes (249 GB RAM on gpu03 and
gpu04), (c) demotes provably-unused boundaries ahead of LRU, and (d) keeps the per-token lead by
leaving every kernel alone. At equal hit, the same-stream numbers above are the margin.

## 2. Design

### 2.1 Row classes

Every cache tensor the packet declares belongs to a **row class**: a retention rule plus the
tensors it owns, each with its own source layout descriptor (base, slot stride, head stride,
row stride, `mask`, dtype, scale tensor). Row `r` of a class is the concatenation of row `r` of
its tensors.

| class | retention | 26B FP8 row bytes | tensors |
|---|---|---|---|
| `Prefix` | every row `[0, p)` | 10 KiB + 80 B | full-layer `kv.{l}.{k,v}` (+ `_scale`); MLA `ckv`/`krot`/`kidx`/`scale` |
| `Window(w)` | rows `[p − w, p)` | 100 KiB + 1.6 KiB | sliding-layer `kv.{l}.{k,v}` (+ ring scales) |
| `State` | one blob per boundary, captured **at** row `p` | model-defined | recurrent state (GDN, KDA, Mamba): matrices, conv history, counters |

Classes come from the live-KV manifest (`crates/plow-asset/src/live_kv.rs`: `window`, `mask`,
`stride`, `layout` per cache) and `serve_manifest::KvGeometry`; a packet whose caches do not
map onto a class refuses the store rather than guessing. Retention rules are descriptors, so
sink-plus-window or chunk-local attention are new rules, not new code paths. Multiple window
sizes in one model are separate `Window(w)` classes.

`State` is what Qwen3.8 27B and Kimi-K3 lack today ("recurrent GDN state has no prefix
snapshot", `docs/bringup/results/qwen38-27b-h100/comparison.md`). Its contract is explicit:
the packet exposes a capture op at an aligned row during prefill (the state after exactly `p`
rows, not the slot's current state after the floor-32 rounding) and a restore op; speculative
rollback must restore it too. A model without the capture op gets `Prefix`/`Window` only, and
its boundaries are the exact prompt end.

### 2.2 Blocks and keys

A **block** is `R` rows of one class, `R` = the packet's boundary quantum (32 today; hash
granularity, physical block size and legal checkpoint alignment are three separate parameters
of the class, derived per model and rejected when they do not divide). Layout `[tensor][row][hd]`
so each tensor's rows are one run. On 26B FP8 a `Window` block is 3.3 MB, a `Prefix` block
330 KB. Blocks live in per-class slabs — device, and optionally pinned host (§2.6) — with fixed
block size per class (free lists, no fragmentation, bounded pointer tables).

Identity: `(namespace, class, chain hash of tokens [0, (i+1)·R))` with the chained FxHash the
radix `PrefixCache` already uses (`memory/vmm.rs:5990`), verified on lookup by comparing the
whole matched prefix's tokens, as today. `namespace` = packet hash + cache format + adapter +
positional config + a multimodal-input digest, so one store can serve more than one packet
and never aliases across formats.

Blocks are **immutable once sealed**. A block is created `Filling`, becomes `Ready` when its
scatter copy's completion event has fired, and only `Ready` blocks are visible to lookup. A
partially written tail (rows past the last full `R`) is never a block; it is re-prefilled, as
the 9-row prompt-end slack is today. Two publishers filling the same key race by key: the
second finds `Filling` and waits on the first's event instead of copying.

The radix tree remains the index. A node at depth `i` holds per class a block ref with its
residency (`Device`, `Host`, `Migrating`, `Absent`) and two counts: **dependents** (boundaries
that need it) and **transfer holds** (gathers, spills, restores in flight). A **boundary**
record at row `p` (`PROMPT_END`, `SHARED_END`, `CHECKPOINT` as today) owns its `State` blobs
and depends on `Prefix` blocks `[0, p/R)` and `Window` blocks `[p/R − w/R, p/R)`.

### 2.3 Publish (incremental, transactional)

Trigger points are unchanged (`exec/gpu/prefix.rs:913-1068`: prompt end, shared end, chunk
checkpoint, awaited checkpoint). At boundary `p`:

1. **Check the copy window.** `written − p ≤ ring − window` must hold at the moment the copy
   is enqueued, on the engine stream, after the producing launch and before any launch that
   can write rows ≥ `p + (ring − window)`. An old shared boundary or an oversized chunk that
   violates this is refused (counter), never copied late.
2. Walk the chain. Per node and class: `Ready`/`Filling` → take a dependency; `Absent` and
   retained by this boundary → allocate from the slab (evicting per §2.5 if needed; failure
   aborts the publish and releases what it took), enqueue the scatter, record the event.
3. Copy `State` blobs captured at `p`.
4. **Commit**: only after every class succeeded, publish the boundary and then release the
   previous same-session boundary's dependencies (retire-on-publish). Blocks whose dependents
   and holds both reach 0 are not freed — they go to the slab's reclaim list in LRU order,
   still `Ready` and hashed, exactly like vLLM's free queue, so an older boundary of the same
   session can still hit until the space is actually needed.

Per-turn cost becomes the new rows only: ~100–300 rows → 10–35 MB scattered, against 106–121 MB
copied today; a cold session (no overlap) still copies its full state (~165 MB), and the gate
measures the mix (§4). Strict publish (`PLOW_VMM_STRICT_PUBLISH`) and the orphaned-block
hazard disappear: blocks are owned by the store, never by a slot.

### 2.4 Attach

For a prompt, pick the deepest boundary `p ≤ match` whose dependencies are all `Ready` and
resident (device or host) and whose `State` blobs exist; `Filling` blocks are waited on
(`PLOW_PREFIX_INFLIGHT_WAIT` semantics) or skipped by policy. Then one **gather** per class:
`Prefix` blocks into the slot's full-layer rows (after `ensure_rows(p)`), `Window` blocks into
ring positions `(r & mask)`, `State` into the slot's state buffers. Each block gathered takes a
transfer hold until the gather's event fires; the slot's first launch waits on that event
(today's `deferred_attach` discipline). Host-resident blocks are H2D copies straight into the
slot's rows from the pinned slab, enqueued on the same stream and covered by the same event;
a device gather kernel never dereferences a host pointer.

Cost model, to be replaced by measurement (§4): a 5.7K-token prefix is 165 MB → ~330 MB of HBM
traffic D2D, ≈ 0.15–0.3 ms; from host ≈ 4.3 ms (window only) to 6.6 ms (all classes) at the
measured ~25 GB/s. Today's attach copies the 106–121 MB snapshot D2D plus one VMM multi-map call
per (track, head, block).

`kv_gather`/`kv_scatter` are **utility kernels** (block pointer table, per-tensor descriptor,
row count), one launch per class, CUDA and HIP, with a per-tensor `memcpy2D` fallback. No
attention or norm object changes; the live-KV contract, the `kv_ring.v1` Lean obligation, the
TMA maps and FP8 scale addressing are untouched.

### 2.5 Eviction and demotion

A boundary is **demoted** (its dependencies released, blocks to the reclaim list) or
**evicted** (blocks reclaimed); a block with transfer holds is never reclaimed, and reclaiming
a block invalidates every boundary that depends on it first. Victim order:

1. reclaim-list blocks, LRU (no decision);
2. **demotion candidates**: boundaries of requests that were aborted or disconnected before
   the first token, and session-TTL-expired boundaries. This is a *prediction*, not a death
   certificate — a retry can follow an abort — so demotion only drops the pin; the blocks stay
   hashed on the reclaim list and the next turn still hits if the space was not needed;
3. LRU over live boundaries; `SHARED_END` (system prompts) ranked by attach count;
4. a victim block spills to the host slab if one is armed (§2.6), else is reclaimed.

An authoritative session-close signal (request header) would turn (2) from prediction into
reclamation; it needs the prod request mix (no bodies are captured) and is a separate change.

### 2.6 Host tier

Per class, a pinned host slab (`PLOW_KV_STORE_HOST_MIB`, default 0), allocated at bring-up in
bounded chunks with a bounded number of transfers in flight. Both classes tier: a session is
165 MB wherever it lives, so a 150 GB tier holds ~900 extra sessions. Migration protocol: a
block being spilled stays `Device` until its D2H event fires, then flips to `Host`; a restore
pins the host source until the H2D event fires; a concurrent attach on a `Migrating` block waits
on the event rather than racing it. The earlier whole-snapshot host test (60 GB: +0.8 pt hit,
worse p99) ran at 1.8 req/s where retention was not binding, and 120 GB pinned at bring-up
killed the server once; §4.0 repeats it at the production rate with bounded allocation and the
crash root-caused before P2. vLLM 0.29 has CPU offloading (`--kv-offloading-size`); prod does not
run it, and §4 compares against both configurations.

### 2.7 What VMM still does

Per-slot sparse growth of the full-layer tracks (`ensure_rows`, premap) and the live rings.
Multi-mapping of whole 4096-row blocks is kept through P1–P4 and removed (P5) only if the
measured `Prefix` gather cost and copy amplification are within the gate; until then an
aligned whole block attaches by map and the store copies the rest. Cached `Prefix` rows then
exist once in the store and once per live slot (≈59 MB × in-flight slots, ~1.5 GB at 27 slots).

### 2.8 Hardware and model coverage

| | today | store |
|---|---|---|
| NVIDIA H100, TP=1 (Gemma-4 12B/26B) | VMM multi-map + snapshot copy | `Prefix` + `Window` + scales — **P1 scope** |
| NVIDIA TP>1 | none | not in scope; the key carries a rank/shard id so a later TP attach can require a boundary `Ready` on every rank before any rank proceeds |
| AMD MI300X/MI325X (GLM-5.3, DeepSeek, Kimi MLA) | `exec/amd/shared_prefix.rs`, VMM groups + tail copy | same store, `Prefix` over `ckv/krot/kidx/scale` with per-tensor descriptors (different row cadences allowed), HIP gather |
| AMD per-slot route (`exec/amd/prefix.rs`) | carried state + ring windows, same slot only | `Window` + `State`, cross-slot |
| Qwen3.8 27B GDN, Kimi KDA | prefix cache off | `Prefix` + `State` once the capture op exists |
| CPU Xeon (26B BF16) | donor-slot memcpy | host slab is the store; gather = memcpy; NUMA placement per slab |
| Apple | none | not in scope |

The store is one Rust module (`memory/kv_store.rs`) over an ops trait: `alloc_slab`,
`scatter`, `gather`, `copy_h2d/d2h`, `event`/`wait`. Backends implement it with their own
synchronisation (CUDA/HIP events, synchronous copies on CPU); the index, boundary records,
counts, state machine, eviction and host tier are shared.

### 2.9 Rejected alternatives

- **Paged slots (vLLM block tables).** Every sliding read/write site (`op_attention.cuh` decode
  bodies and prefill stager, `op_norm.cuh` writers, TMA maps, FP8 scale indexing) would take a
  `(slot, logical block) → physical` lookup; `live_kv::validate` and the `kv_ring` proof assume
  `mask == stride − 1`; `crates/devgen/src/lib.rs:2378` records paging as a measured null on
  these kernels. The store keeps the flat slot and copies instead.
- **512-row blocked full-layer layout** for zero-copy `Prefix` mapping: a kernel addressing
  change for ≤ 3% of cache bytes. Not part of this proposal.
- **Recomputing the window on attach**: not exact (§1.2); exact recompute is the full re-prefill
  the miss path already does.
- **Lower-precision window storage**: changes numerics; the FP32 gate owns that budget.

## 3. Expected outcome vs vLLM 0.29

| | vLLM 0.29 device-only (prod flags) | vLLM 0.29 + CPU offload | plowrt today | plowrt + store |
|---|---|---|---|---|
| bytes / retained session | ~165 MB | ~165 MB | ~156 MB | ~165 MB |
| bytes inserted / continuing turn | new tokens × 120 KiB | same | 106–121 MB | new rows × 114 KiB |
| retention tiers | device | device + host | device | device + host, both classes |
| dead-boundary demotion | none | none | TTL 150 s | abort/disconnect/TTL |
| per-hit attach | 0 (in place) | H2D of offloaded blocks | ~121 MB D2D + map calls | ~165 MB D2D or H2D |
| recurrent-state models | Mamba/GDN prefix caching | same | off | `State` class |
| TTFT / TPOT at equal hit, same stream, 6.5 req/s | 82 / 11.5 ms (0.28) | — | 40 / 10.8 (MTP off), 40 / 5.2 (MTP on) | unchanged |

The claim to verify, predeclared: at the production arrival rate and above, plowrt + store
reaches a token hit rate within 2 pt of the oracle ceiling while holding its same-stream
TTFT/TPOT, so its SLO goodput (TTFT ≤ 200 ms, TPOT ≤ 30 ms) exceeds vLLM 0.29's by ≥ 20% at
10 req/s and its maximum sustainable rate by ≥ 20%, against vLLM both device-only and with
offloading at the same host budget. Anything less is a tie and does not justify P5.

## 4. Gates

Each phase stops at a failing gate. Replays use identical recorded arrivals, prompts, output
lengths and a cold cache (the harness on this branch lets explicit `--think-*` flags win over
`--prod-fidelity`, which silently reset them to 10 s before); MTP-off arms are separate from
MTP-on; the final numbers come from
`campaign.py report` in the strict format of `docs/bringup/agent-tools.md`.

0. **Measure first (P0).** (a) The oracle ceiling of the recorded stream from the prompts alone
   (unbounded cache, exact prefix match): done, 81.1% (§1.2). (b) The actual think-time distribution from prod
   timestamps. (c) vLLM 0.29's real pool: allocated KV bytes, groups, page sizes, hybrid-manager
   status, from its startup log — not `tokens × 120 KiB`. (d) The cheapest hypothesis: today's
   snapshot cache with a bounded host tier (60 GB, crash root-caused) at rates 2.7 and 4.5. If
   retention alone meets §3, the store must justify itself on the remaining rows of §3.
1. **Exactness.** Attach-then-decode logits (not only token digests) equal to the uncached
   prefill at ctx 4094/4095/8191 and window edges (`p − w ± 32`), BF16 and FP8, both layer
   kinds; the existing poison and boundary sweeps; multi-turn checkpoints and speculative
   rollback on a `State` model.
2. **Lifecycle faults.** Forced eviction, slab exhaustion, delayed copies, concurrent
   attach/publish/spill on one key, cancellation mid-gather, slot reuse, hash collisions,
   partial-block boundaries, `Migrating` races — each with value checks.
3. **Kernels untouched, interference bounded.** `step_bench` digests identical to the base
   packet; serving ITL/TTFT under load within 2% of today's at rates 1.8 and 2.7 with the store
   on; achieved gather/scatter bandwidth, copy amplification, pinned RSS and queue delay
   reported.
4. **Serving win (§3).** Steady-state arms: warm-up longer than one retention turnover, then
   ≥ 3 session lifetimes at rates 2.7, 4.5 and overload, 3 repetitions, with offered vs
   completed, backlog, timeouts and cancellations; ablations separating incremental publish,
   demotion and host tier.
5. **Numerics.** FP32 gate at c1/c8/c32 unchanged.
6. **Soak.** 1 h at rate 2.7: flat RSS, device mapped bytes, `kv_store_*` counters consistent,
   no engine-thread stall > 1 ms attributable to the store (pfx_trace).

## 5. Phases

| | content | gate |
|---|---|---|
| P0 | measurements of §4.0; vLLM 0.29 device-only and offload arms on gpu03 | — |
| P1 | `memory/kv_store.rs`, CUDA ops, `Prefix` + `Window` (+ scales), block state machine, gather/scatter kernels, publish/attach at today's trigger points, today's eviction order, behind `PLOW_KV_STORE=1` (default off); TP=1 | 1, 2, 3, 4 device-only |
| P2 | demotion (§2.5) and the per-class host tier | 4 incl. offload arm, 6 |
| P3 | AMD: `SharedPrefix` and the per-slot route on the store; HIP gather | AMD rung + replay gates, when a box is available |
| P4 | `State` class: capture op in the packet contract; GDN (Qwen3.8 27B), KDA (Kimi-K3) | 1 (state), 5 and replay on Qwen3.8 |
| P5 | remove VMM multi-map, snapshot pool, strict publish, stale/park/recycle; rewrite `docs/arch/22` §7–9 | copy-amplification gate; all production recipes re-qualified |

Metrics: `plowrt_kv_store_{blocks,bytes}{class,tier}`, `publish_{blocks,bytes,refused}_total`,
`attach_{hits,misses,rows,wait}_total`, `attach_from_host_{total,bytes}`, `demote_{abort,
disconnect,ttl}_total`, `evict_{lru,spill}_total`, `migrations_inflight`. The `plowrt_prefix_*`
names stay for the kept semantics (hits, tokens attached/lost, boundaries).

## 6. Open questions

1. `R` per model: 32 matches today's boundary quantum; the gather cost is per row, not per
   block, so larger `R` only coarsens dedup. Decide on P0's oracle at 32 vs 64.
2. A session-close header for authoritative reclamation (§2.5).
3. The `State` capture op's cost on the prefill rungs (one state copy per aligned checkpoint).
4. Pinned-host allocation strategy (bring-up chunks vs on demand) after the 120 GB crash is
   root-caused.
