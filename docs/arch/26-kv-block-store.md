# 26 — KV block store: a content-addressed prefix cache under flat slots

**Status: design (2026-10-23), not implemented.** Replaces the boundary-snapshot prefix cache of
`docs/arch/22-kv-cache.md` §7–§9 and the AMD `SharedPrefix` with one store. Kernels are unchanged.

## 1. Why

Production Gemma-4 26B-A4B FP8 on gpu04, same traffic class, measured from the engines' own
metrics (`docs/runtime/prefix-cache.md`, vLLM `/metrics`):

| | req/s | TTFT mean | ITL mean | prefix hit | uncached tok/req |
|---|---|---|---|---|---|
| plowrt Kit 2.1 (MTP on), 10-21 day avg | 5.85 | 110 ms | 16.3 ms | 74.4% | — |
| plowrt Kit 2.2 (MTP on), 10-22 06:22–07:02 | 10.2 | 142–370 | 19.8–22 | 63–65% | 2,002 |
| vLLM 0.29 (prod flags), 10-22 day avg | 6.2 | 116 | 13.6 | 80.8% | ~1,100 |

On gpu03 with the **same request stream** (`scripts/bench/agentic_turns.py --open-loop
--prod-fidelity`, think median 45 s) the per-token picture is the opposite:

| rate 1.8 → 6.5 req/s | TTFT p50/p99 | TPOT p50/p99 | hit |
|---|---|---|---|
| vLLM 0.28 | 82 / 371 | 11.5 / 30.8 | 75.7% |
| plowrt, MTP off | 41 / 271 | 10.8 / 23.0 | 74.4% |
| plowrt, MTP on | 40 / 238 | 5.2 / 19.1 | 74.6% |

At rate 2.7 (9.2–9.7 req/s) vLLM 0.28 saturates (TPOT p99 227 ms, goodput 81%) while plowrt holds
TTFT 65/438 and TPOT 11.6/45.9 with 74.8% hit. The hit rates match whenever the traffic matches.
plowrt's prefix cache is not slower; it holds fewer sessions per byte of retained state at high
arrival rates, and every turn re-copies ~120 MB. Both are design properties of the snapshot
cache, not tuning.

### 1.1 Where the bytes go today (26B FP8, `exec/gpu/prefix.rs:682-751`)

Per token: 25 sliding layers × 2 × 8 heads × 256 B = 100 KiB (+1.6 KiB FP8 ring scales); 5 full
layers × 2 × 2 heads × 512 B = 10 KiB (+80 B scales). A boundary snapshot at prompt end `p_a` is

```
106.5 MB  sliding windows, 1024 rows × 104,000 B        (fixed, every turn, never shared)
 ≤ 42 MB  full-layer rows p_a mod 4096 × 10 KiB          (the private sub-block tail)
 ~0.5 MB  full-layer scale prefix [0, p_a) × 80 B
```

plus whole 4096-row full-layer VMM blocks shared by multi-mapping (40 MiB per block across the
10 tracks × 2 heads). Prod counters: 41 GB cache = 31.7 GB snapshots (261 live, ~121 MB) + 9 GB
shared blocks → **~156 MB per retained session**. Retire-on-publish keeps one boundary per
session, so each turn copies a new 106–121 MB snapshot and frees the previous one: 335 GB/h of
copy plus VMM unmap/remap on the engine path, for no change in retained bytes.

### 1.2 The capacity model

Let `S` = sessions alive (turns still to come), `λ` = session arrival rate, `L` = session
lifetime. `S ≈ λ·L`. A session's next turn hits only if its boundary survived, so

```
hit ≈ ceiling × min(1, capacity_bytes / (S × bytes_per_session))
```

The ceiling is the traffic's own repeat fraction (~81% on the prod stream). At 10 req/s and
the replay's session shape (≈5 turns, 45 s median think) `S` is ~450–900; at 156 MB/session the
41 GB device cache holds ~260, which is the 63–65% hit Kit 2.2 measured. At 6.2 req/s `S`
halves and vLLM's 48 GB pool (419,746 tokens × 120 KiB) holds it: 80.8%.

**Both engines sit at the same per-session floor.** vLLM's hybrid KV manager keeps a sliding
layer's blocks only inside the window (older ones are freed to the LRU head) and the full
layers at 10 KiB/token: ≈158 MB for a 5.7K-token session. The window is exact state — it
cannot be recomputed from fewer rows, because each sliding layer's K/V for the window depends
on the previous layer's window, and so on back to row 0. So the design cannot win by shrinking
the floor; it wins by (a) retaining more sessions for the same device bytes (reclaiming what
is provably dead), (b) extending retention into host memory (249 GB on both boxes), (c) paying
the incremental cost per turn instead of a full snapshot, and (d) keeping the per-token lead
(kernels untouched). At the ceiling, plowrt's TTFT is 2× and ITL 13% (MTP off) to 2× (MTP on)
better on the same stream; that is the margin this design protects.

## 2. Design

### 2.1 Row classes

Every cache tensor the packet declares belongs to a **row class**. A class is a retention rule
plus the tensors it owns; row `r` of a class is the concatenation of row `r` of its tensors.

| class | retention | 26B FP8 row bytes | tensors |
|---|---|---|---|
| `Prefix` | every row `[0, p)` | 10 KiB + 80 B | full-layer `kv.{l}.{k,v}` (+ `_scale`), MLA `ckv/krot/kidx/scale` |
| `Window(w)` | rows `[p − w, p)` | 100 KiB + 1.6 KiB | sliding-layer `kv.{l}.{k,v}` (+ ring scales) |
| `State` | one blob per boundary | model-defined | recurrent state (GDN, KDA, Mamba), `kv.48..52.*`-style drafter KV if ever cached |

The classes come from the live-KV manifest (`crates/plow-asset/src/live_kv.rs`: `window`,
`mask`, `stride`, `layout` per cache) and from `serve_manifest::KvGeometry`; a packet with an
unknown cache kind refuses the store rather than guessing. `State` is what Qwen3.8 27B and
Kimi-K3 lack today ("recurrent GDN state has no prefix snapshot",
`docs/bringup/results/qwen38-27b-h100/comparison.md`).

### 2.2 Blocks

A **block** is `R = 32` rows of one class (the boundary quantum already used by every publish
point), stored contiguously as `[tensor][row][hd]` so one tensor's rows are one run. Sizes on
26B FP8: `Window` block 3.3 MB, `Prefix` block 330 KB. Blocks live in per-class device slabs
(fixed size per class → free lists, no fragmentation), with an optional pinned-host slab per
class (§2.6). A block's identity is `(class, chain hash of tokens [0, (i+1)·R))`, the same
chained FxHash the radix `PrefixCache` already uses (`memory/vmm.rs:5990`), verified by token
compare on lookup.

The radix tree stays the index. A node at depth `i` holds, per class, an optional block ref
(device, host, or absent) and a refcount. A **boundary** record at row `p` (`PROMPT_END`,
`SHARED_END`, `CHECKPOINT`, as today) holds its `State` blobs and pins the blocks it needs:
`Prefix` blocks `[0, p/R)`, `Window` blocks `[p/R − w/R, p/R)`.

### 2.3 Publish (incremental)

At a boundary `p` (prompt end, shared end, chunk checkpoint — unchanged trigger points,
`exec/gpu/prefix.rs:913-1068`):

1. Walk the chain `[0, p/R)`. For each node, per class: block present → touch; absent and the
   class retains that row range for this boundary → allocate a block and enqueue a scatter copy
   from the slot (ring rows at `(r & mask)`, full-layer rows at `r`) on the engine stream, after
   the launch that produced them.
2. Copy `State` blobs.
3. Record the boundary; release the previous same-session boundary's pins (retire-on-publish).
   A `Window` block whose refcount drops to 0 goes back to the free list (it is outside every
   live window, exactly as vLLM frees it); a `Prefix` block stays LRU-evictable.

Per-turn cost becomes the new rows only: a 50-token reply plus a user message is ~100–300 rows
→ 10–33 MB scattered, against 106–121 MB copied today. The sliding ring holds `ring ≥ window +
stage` rows, so every window block a boundary needs is still in the ring when its publish runs
(the existing `written − p_a ≤ ring − window` rule, `prefix.rs:929`). Publishes are stream-
ordered with the producing launch; there is no hazard with later rows because later rows only
overwrite ring positions outside the window.

Strict-publish (`PLOW_VMM_STRICT_PUBLISH`) and its orphaned-block hazard disappear: blocks are
owned by the store, never by a slot, so a slot's rows can never be referenced before they are
copied out.

### 2.4 Attach

For a prompt, find the deepest boundary `p ≤ match` whose pins are all resident (device or
host) and whose `State` blobs exist. Then one **gather** per class: `Prefix` blocks `[0, p/R)`
into the slot's full-layer rows (after `ensure_rows(p)`), `Window` blocks into ring positions
`(r & mask)`, `State` blobs into the slot's state buffers. Host-resident blocks copy H2D
directly into the slot (no device staging).

A gather of a 5.7K-token prefix moves 58 MB + 106 MB ≈ 165 MB: ~0.1 ms at device bandwidth,
~6.6 ms if the window comes from host (PCIe Gen5, measured ~25 GB/s). Today's attach already
copies the 106–121 MB snapshot D2D plus VMM multi-map calls, so the device case is not slower
and the CPU-side driver work (map/unmap per block per head per track) is gone.

The gather/scatter are **utility kernels** (`kv_gather`, `kv_scatter`: a pointer table of
block bases, per-tensor offsets, row count), one launch per class, CUDA and HIP, with a
per-tensor `memcpy2D` fallback. They are not model kernels: no change to any attention or
norm object, the live-KV contract, the `kv_ring.v1` Lean obligation, TMA maps, or FP8 scale
addressing. `step_bench` digests and times must be identical before/after (§4).

### 2.5 Eviction

One policy, in victim order, replacing fine → dead-leaf → tree-LRU:

1. unpinned blocks (window-trimmed, superseded turns): free list, no decision;
2. boundaries of **ended sessions**: HTTP abort (11–17% of prod requests), client disconnect
   before the first token, explicit session close — demoted immediately instead of waiting out
   `PLOW_PREFIX_SESSION_TTL_S`;
3. TTL-expired boundaries, then LRU over the rest, `SHARED_END` (system prompts) ranked by
   attach count rather than recency;
4. a victim spills to the host slab if one is armed, else frees.

(2) is the first lever vLLM lacks: its LRU cannot know a session died. (3) keeps today's
session semantics. A think-time-predicted order (evict the boundary least likely to return
next) is a later experiment, gated on the replay.

### 2.6 Host tier

Per class, a pinned host slab (`PLOW_KV_STORE_HOST_MIB`, default 0). `Window` blocks are the
bulk (106 MB per session) and the ones worth tiering: at 150 GB the tier holds ~900 extra
sessions, which at 10 req/s is the whole alive population (§1.2). Spill is async off the
engine thread; restore is part of the attach gather. The earlier whole-snapshot host test
(60 GB: +0.8 pt hit, worse p99) ran at 1.8 req/s where device capacity was not binding; the
gate here is the prod-rate replay (§4). vLLM 0.29 has CPU offloading (`--kv-offloading-size`)
but prod does not run it; this is the second lever.

### 2.7 What VMM still does

Per-slot sparse growth of the full-layer tracks (`ensure_rows`, premap) and the live rings.
Multi-mapping, shared handles, `Stale`/park/recycle, deferred reclaim and stale reserve are
no longer needed once the store is qualified (§5, P5). Cached `Prefix` rows exist once in the
store and once in each slot that is live on them (≈58 MB × ~27 in-flight ≈ 1.5 GB, 3% of the
cache) — the price of keeping the slot flat.

### 2.8 Hardware and model coverage

| | today | store |
|---|---|---|
| NVIDIA H100 (Gemma-4 12B/26B) | VMM multi-map + snapshot copy | `Prefix` + `Window` + scales |
| AMD MI300X/MI325X (GLM-5.3, DeepSeek, Kimi MLA) | `exec/amd/shared_prefix.rs`, VMM groups + tail copy | same store, `Prefix` class over `ckv/krot/kidx/scale`, HIP gather |
| AMD per-slot route (`exec/amd/prefix.rs`) | carried state + ring windows, same slot only | `Window` + `State`, cross-slot |
| Qwen3.8 27B GDN, Kimi KDA | prefix cache off | `Prefix` (full layers) + `State` (GDN state per boundary) |
| CPU Xeon (26B BF16) | donor-slot memcpy | host slab is the store; gather = memcpy |
| Apple | none | not in scope |

The store is one Rust module (`memory/kv_store.rs`) over a small ops trait: `alloc_slab`,
`copy_d2d`, `copy_h2d/d2h`, `gather`, `scatter`, `fence`. Backends implement the trait; the
radix index, boundary records, refcounts, eviction and host tier are shared.

### 2.9 Rejected alternatives

- **Paged slots (vLLM block tables).** Every sliding read/write site (`op_attention.cuh`
  decode bodies and prefill stager, `op_norm.cuh` writers, the TMA maps, FP8 scale indexing)
  would take a `(slot, logical block) → physical` lookup, and `live_kv::validate` and the
  `kv_ring` proof assume `mask == stride − 1`. `crates/devgen/src/lib.rs:2378` records paging as
  a measured null on these kernels. The store keeps the flat slot and copies instead.
- **512-row blocked full-layer layout** (`[block][layer][kv][head][row][hd]`, VMM-mappable at
  5 MiB) for zero-copy `Prefix` sharing: a kernel addressing change for ≤ 3% of cache bytes.
  Kept as an optional later step gated on `step_bench` showing no decode regression.
- **Recomputing the window on attach** (store only `Prefix`, re-prefill the last 1024 rows):
  not exact — each sliding layer's window depends on the previous layer's window back to row
  0 (§1.2).
- **Lower-precision window storage**: changes numerics; the FP32 gate owns that budget.

## 3. Expected outcome vs vLLM 0.29

| | vLLM 0.29 (prod flags) | plowrt today | plowrt + store |
|---|---|---|---|
| bytes / retained session | ~158 MB | ~156 MB | ~160 MB device, 106 MB of it tierable |
| sessions retained (26B FP8, 1× H100) | ~300 (48 GB pool) | ~260 (41 GB) | ~260 device + ~6 per host GB |
| per-turn publish bytes | 0 (in place) | 106–121 MB + map/unmap | 10–33 MB |
| per-hit attach cost | 0 | ~121 MB D2D + driver calls | ~165 MB D2D (~0.1 ms) or +6.6 ms from host |
| dead-session reclaim | LRU only | TTL 150 s | immediate |
| hit at prod rate (10 req/s, model §1.2) | ~65% device-only | 63–65% | ceiling (~81%) with ≥ 100 GB host |
| recurrent-state models | Mamba/GDN prefix caching (0.11+) | off | `State` class |
| TTFT / ITL at equal hit (same stream) | 82 / 11.5 ms | 40 / 10.8 (MTP off), 40 / 5.2 (MTP on) | unchanged |

The claim to verify: at the production arrival rate, the store reaches the traffic ceiling hit
rate while vLLM (device-only) does not, and per-token latency stays where it is.

## 4. Gates

Each phase stops at a failing gate.

1. **Exactness.** Attach-then-decode token digests identical to the uncached prefill at ctx
   4094/4095/8191 and at window edges (`p − w ± 32`), BF16 and FP8, both layer kinds; the
   existing poison and boundary sweeps.
2. **Kernels untouched.** `step_bench` digests and times identical to the base packet (same
   binary, same objects), B1–B128, ctx 1K/4K/16K.
3. **Serving.** The prod-fidelity replay at rates 1.8 and 2.7 (`scripts/bench/agentic_turns.py
   --open-loop --prod-fidelity --think-median-s 45 --think-sigma 1.0 --think-max-s 600`, 10 min
   arms) against the current snapshot cache and against vLLM 0.29: hit ≥ both, TTFT and TPOT
   p50/p99 ≤ today's, 0 errors. A rate-4.5 arm for the host tier.
4. **Numerics gate.** FP32 gate at c1/c8/c32 unchanged (regression only; no kernel changed).
5. **Soak.** 1 h at rate 2.7: flat RSS, device mapped bytes, `kv_store_*` counters consistent;
   no engine-thread stall > 1 ms attributable to the store (pfx_trace).

## 5. Phases

| | content | gate |
|---|---|---|
| P1 | `memory/kv_store.rs`, CUDA ops, `Prefix` + `Window` classes (+ scales), gather/scatter kernels, publish/attach wired at today's trigger points behind `PLOW_KV_STORE=1` (default off) | 1, 2, 3 (device only) |
| P2 | eviction policy (§2.5) incl. abort/disconnect demotion; host tier per class | 3 at rate 2.7 and 4.5, 5 |
| P3 | AMD: `SharedPrefix` and the per-slot route on the store; HIP gather | AMD rung + replay gates, when a box is available |
| P4 | `State` class: GDN (Qwen3.8 27B), KDA (Kimi-K3) | FP32 gate + replay on Qwen3.8 |
| P5 | remove VMM multi-map, snapshot pool, strict publish, stale/park/recycle; `docs/arch/22` §7–9 rewritten | all production recipes re-qualified |
| P6 (optional) | 512-row blocked full-layer layout for zero-copy `Prefix` | `step_bench` no regression |

Metrics: `plowrt_kv_store_{blocks,bytes}{class,tier}`, `publish_{blocks,bytes}_total`,
`attach_{hits,misses,rows}_total`, `attach_from_host_total`, `evict_{dead_session,ttl,lru,
spill}_total`, `sessions_ended_total`. The `plowrt_prefix_*` names stay for the kept
semantics (hits, tokens attached/lost, boundaries).

## 6. Open questions

1. `R = 32` vs 64/128: smaller `R` is finer dedup and larger pointer tables; the gather kernel
   cost is per row, not per block. Decide on the replay.
2. Should `Prefix` blocks of a hit be shared by multi-map when they align to 4096 rows (keep
   that one VMM feature) — saves 40 MiB of copy per aligned block at the cost of keeping the
   shared-handle code. Default no; measure attach time first.
3. Session end signals beyond abort/disconnect: a request header, or `finish_reason = stop`
   without a session id. Needs the prod request mix (no bodies are captured).
4. Host tier placement: pinned slab at bring-up (today's 1 GiB chunks) vs on demand; 120 GB
   pinned killed the server on gpu03 once — root-cause before P2.
