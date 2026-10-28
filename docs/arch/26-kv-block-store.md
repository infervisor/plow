# 26 — Prefix cache retention and the KV block store

**Status: design (2026-10-23), reviewed, retention simulated; Part A implemented (A1, A2); B1 and B3 implemented and on by default (exactness gate passed 2026-10-23, replay and Qwen gates below); B2 parked (no AMD box); B4 on CUDA: the snapshot copy path, its pool and pitched knobs and the store switch are gone and `Prefix` blocks hold every full-layer row, so nothing is multi-mapped (replay 7, 2026-10-23: without a host tier the store is above the snapshot path and vLLM 0.29 on hit rate, TTFT and TPOT at 10 s think and 2.3 points under vLLM's hit rate at 45 s think, capacity-bound; with the host tier (replay 15, 2026-10-24) 71.7 % at 45 s think and 78.4 % at 20 s think with lower TTFT and TPOT than vLLM and than the store without the tier at 45 s; at 10 s think the tier still costs TPOT and tails, attributed to a per-tick cost under heavy spilling amplified by the speculation policy, B.6a); the AMD snapshot path remains (§5). Merge prep 2026-10-25: fourth review round without a blocking finding; every H100 recipe with a usable packet passes its own gate on the branch binary (Gemma-4 E4B and 12B with the store and host tier armed, Veena, Orpheus, Chatterbox, Chatterbox-MTL, both Qwen3-ASR sizes re-emitted), the Qwen3.8 production path is numerically unchanged, and the main-vs-branch benches lose nothing (`docs/bringup/results/gemma26b-h100-20261023/comparison.md`). Production pass 2026-10-25 (same doc, "Production pass"): every BF16/FP8 recipe re-emitted from the final source and gated with the store armed; Qwen3.8 BF16 now serves with the prefix cache on (B3 exact attach, c1/c8/c32 gate PASS, agentic c8 TTFT 3.6x better; c32 retention is store-capacity-bound beside 48 live slots); Qwen3.8 FP8 gained a 1..48 decode ladder (the ladder validator had silently kept native FP8 rungs on the widest launch) and scores against a same-precision vLLM peer, but stays uncompetitive on throughput; the 12B strict report is MATCHED and a loss; the E4B recipe's packet exposed a kv-shared tail bug (the prefill-bucket tail path uploaded the chain's request table but not the attention's, so the first generated token attended over the first prompt token alone; latent on main since the attention table arrived with the Qwen3.8 chain), fixed on the branch and the packet passes its gate with the tail route on. Packet-ABI slice 1 (2026-10-25): the NVIDIA engine no longer decides from model knowledge which interpreter objects may run a W8A8 program or where a recurrent model's state lives; the packet declares both (`object_requirements` per module, `recurrent_state` slot-major buffers) under runtime contract 2, an undeclared W8A8 or gated-delta-net packet is refused with a re-emit message, and the four affected packets (Qwen3.8 BF16/FP8, Gemma-4 12B and 26B FP8) were re-emitted with unchanged pairing hashes and re-gated; E4B and 26B BF16 load unchanged. Packet-ABI slice 2 (2026-10-26): serving choices come from the packet too; `serve.json` version 2 declares the tool-call syntax, the reasoning-trace markers and, for a checkpoint without a chat template, the built-in prompt builder, the runtime compiles the template with those declarations and probes nothing from its text or the vocabulary (version-1 and legacy packets keep their probes), every full-model emitter writes the section, and the production packets' prompt bytes and token ids are frozen as goldens (`crates/plowrt/tests/fixtures/serve_golden/`).**
Part A says what actually moves the production hit rate and is implementable on today's
snapshot cache. Part B is the content-addressed block store that replaces the boundary
snapshots of `docs/arch/22-kv-cache.md` §7–§9 and the AMD `SharedPrefix` with one substrate,
justified by extensibility and publish cost — not by hit rate. Model kernels are unchanged in
both parts.

## 1. The problem

Production Gemma-4 26B-A4B FP8 on gpu04, from the engines' own metrics:

| | req/s | TTFT mean | ITL mean | prefix hit (tokens) | uncached tok/req |
|---|---|---|---|---|---|
| plowrt Kit 2.1 (MTP on), 10-21 day avg | 5.85 | 110 ms | 16.3 ms | 74.4% | — |
| plowrt Kit 2.2 (MTP on), 10-22 06:22–07:02 | 10.2 | 142–370 | 19.8–22 | 63–65% | 2,002 |
| vLLM 0.29 (prod flags), 10-22 day avg | 6.2 | 116 | 13.6 | 80.8% | ~1,100 |

Different days and arrival rates, so this ranks nothing. On gpu03 with the **same recorded
request stream** (`scripts/bench/agentic_turns.py --open-loop --prod-fidelity`: production
prompt shape, think median 10 s; two interleaved rounds agree within noise):

| rate 1.8 → 6.5 req/s | TTFT p50/p99 | TPOT p50/p99 | hit |
|---|---|---|---|
| vLLM 0.28 | 82 / 371 | 11.5 / 30.8 | 75.7% |
| plowrt, MTP off | 41 / 271 | 10.8 / 23.0 | 74.4% |
| plowrt, MTP on | 40 / 238 | 5.2 / 19.1 | 74.6% |

At rate 2.7 (9.2–9.7 req/s):

| 9.7 req/s | TTFT p50/p99 | TPOT p50/p99 | goodput | hit |
|---|---|---|---|---|
| vLLM 0.28 | 230 / 1323–1630 | 36–37 / 227–262 | 81% | 60.1% |
| vLLM 0.29 (prod version) | 118 / 592 | 16.9 / 61.6 | 100% | 75.6% |
| plowrt, MTP off | 69 / 524–590 | 20.0 / 44.6 | 100% | 74.4–74.6% |
| plowrt, MTP on | 64–65 / 438–469 | 11.6–13.0 / 45.9–46.2 | 100% | 74.8% |

vLLM 0.29 is the real baseline: it does not saturate at this rate, and without MTP its TPOT p50
is *better* than plowrt's (16.9 vs 20.0 ms) while its TTFT is 1.8× worse. plowrt's per-token
lead at high rate is therefore TTFT always, TPOT only with MTP on. Hit rates are equal on equal
traffic. The 45 s-think long arms are queued (`/home/lava/scratch/g26-vs-vllm/ab-long.sh`). What
production exposes is retention at 10 req/s.

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
9 GB shared blocks ≈ **156 MB per retained session**; retire-on-publish keeps one boundary per
session.

**Both engines share this floor.** An independent 5.7K-token session is the window (106 MB)
plus `Prefix` rows (5.7K × 10.3 KiB = 59 MB) ≈ 165 MB in vLLM 0.29 too, once its hybrid
manager has freed the sliding rows outside the window (`SlidingWindowManager.
remove_skipped_blocks`, `v1/core/single_type_kv_cache_manager.py:632`; freed blocks keep their
hash in the free queue in free order, so under pressure they are the first of that request's
blocks to be reused). The window cannot be rebuilt from fewer rows — each sliding layer's window
depends on the previous layer's window back to row 0 — only re-prefilled from the tokens.

### 1.2 What moves the hit rate: the retention simulation

`scripts/bench/prefix_retention_sim.py` replays the harness plan (30 min, seed 7) through an
LRU-by-bytes cache with the per-session bytes above, for the snapshot model (every turn inserts
a fresh ~121 MB boundary and retires the previous) and an incremental model (a turn inserts only
its new rows; the session's blocks move to MRU on a hit, as vLLM's and a block store's do). The
in-simulation ceiling is 75% (it omits first-turn system-prompt hits; the unbounded-cache oracle
of the same plan, `scripts/bench/prefix_oracle.py`, is **81.1%** with those, 32-row floor and
9-row slack included).
"ended-first" evicts sessions whose last turn has run before any live session — an oracle bound
on what a session-end signal buys.

| 10.8 req/s (rate 2.7) | LRU, 41 GB | incremental, 48 GB | ended-first, 41 GB | 101 GB | 191 GB | 191 GB + ended-first |
|---|---|---|---|---|---|---|
| think median 10 s | 73.4% | 74.2% (74.4% snapshot at 48 GB) | 75.0% | 75.0% | 75.0% | 75.0% |
| think median 20 s | 55.2% | 60.8% (62.3%) | **74.9%** | 74.7% | 75.0% | 75.0% |
| think median 45 s (σ 1.0) | 28.7% | 32.9% (34.0%) | 41.1% | 60.7% | 72.9% | 75.0% |

At 7.2 req/s and 10 s think every column is 74.4–74.9%: the cache is not binding, which is
why the earlier 60 GB host-tier test at that rate measured +0.8 pt and why the replay's 75%
matches both engines. The snapshot column is consistent with production (74.4% at 5.85 req/s,
63–65% at 10.2) at a think median somewhere in 15–20 s, but aggregate hit rates do not pin the
distribution; measuring it is P0's first item.

Three conclusions, each contrary to a claim an earlier draft of this document made:

1. **Incremental insertion does not raise retention.** Under LRU a hit moves the whole session to
   MRU whether the engine re-inserts a snapshot or touches blocks, so reuse distance is the
   bytes of *distinct sessions* touched since — identical in both models. vLLM's per-token
   cache has no retention advantage over the snapshot cache at equal capacity, and a block
   store would have none either.
2. **LRU is close to the right order among live sessions** for this think distribution: a
   lognormal hazard rises only below its mode (seconds here) and falls after it, so beyond the
   first few seconds the longest-idle session is the least likely to return next. This is not a
   proof of optimality (sessions differ in size and token value); it says a cleverer live-session
   order is not where the points are.
3. **The levers the simulation supports are capacity and knowing which sessions ended.** A
   session-end signal is worth up to +20 pt at 20 s think (to the ceiling) and +12 pt at 45 s;
   host capacity is worth +20 to +44 pt at 45 s. Both apply to today's snapshot cache unchanged.
   The simulation's limits: whole-session recency (no block-level eviction or partial-prefix
   hits), one seed, and no production think distribution — P0 validates it against
   hand-checkable traces, more seeds and the measured gaps before any column is quoted as a
   prediction.

### 1.3 Where the lost tokens are (measured 2026-10-23, gpu03, plowrt MTP, 10-min arms)

Per-request split of the replay's lost prompt tokens (`load.json`, first turn of a session vs
later turns):

| arm | hit | first turns: share of requests / token hit / zero-hit | later turns: token hit / zero-hit | lost tokens first : later |
|---|---|---|---|---|
| 45 s think, rate 1.8 (prod-like) | 47.6 % | 36 % / 13.5 % / 57 % | 66 % / 24 % | 58 : 42 |
| 20 s think, rate 2.7 (saturated) | 51.3 % | 28 % / 14.0 % / 56 % | 65 % / 24 % | 48 : 52 |
| 20 s think, rate 2.7, 60 GB host tier | 52.6 % | 29 % / 15.4 % / 52 % | 67 % / 22 % | 50 : 50 |

Two losses of comparable size, with different fixes:

* **First turns miss the shared system prompt.** A first turn can only hit its app's system
  prompt (a quarter to a third of its tokens); 57 % of first turns attach nothing although
  the four shared ends exist (1054 shared-end publishes, 682 shared attaches). A vLLM-style
  pool hits the system prompt on every first turn. Attributed (store replay, 2026-10-23): no
  first turn missed before memory pressure began and 24–66 % missed after it. Under load every
  session boundary sits inside its TTL, so the shared ends were the only unprotected victims of
  the boundary LRU and were evicted and republished in a loop (267–1276 shared-end publishes
  per 10 min). Fix: a shared end that served two attaches is shielded like a session inside
  its TTL (B.5).
* **A quarter of later turns find their boundary evicted** (zero cached tokens), the
  retention loss Part A addresses; the 20 s arms are saturated (goodput 63–73 %, 32–56 in
  flight), so their latencies measure overload and the host tier's +1.3 pt there is a lower
  bound.

Against vLLM 0.29 on the same 45 s stream (60.6 %), closing the later-turn loss alone (to ~88 %)
is worth ~14 pt and the first-turn loss ~6.5 pt; either closes the gap, both pass it.

## Part A — Retention (the production fix)

### A.1 Host tier for boundary snapshots

Today's snapshot host tier (`PLOW_VMM_HOST_CACHE_MIB`, pinned 1 GiB chunks, 4 spills in flight,
`vmm.rs:1933-2010`) is the right mechanism and was tested at the wrong rate. gpu03 and gpu04
have 249 GB RAM (238 GB available on gpu04). A spilled snapshot keeps its whole-block `Prefix`
handles **on device** (the radix blocks it alone keeps alive, `evict_unit`), ≈ 35 MB per session
today (9 GB / 261), so the effective budget is `41 GB = dev_snapshots × 121 MB + all_sessions ×
35 MB` with `host_snapshots = host_GB / 121 MB`: a 100 GB host tier gives ≈ 826 host + 77 device
= ~900 retained sessions against 261 today (≈ the simulation's 141 GB total, between its 101
and 191 GB columns). Allocation after model load in bounded chunks with partial-allocation
fallback and transfer backpressure, after the 120 GB bring-up crash is root-caused (P0).
Root cause (kernel log, 2026-10-22 06:44 UTC): the host OOM killer took plowrt (123 GB pinned
RSS) beside two co-tenant compiler builds of ~27 GB each; nothing in the tier itself failed.
Implemented: the cap is cut to `MemAvailable` minus an eighth of `MemTotal` (floor 16 GiB) at
bring-up, with a warning. The reserve is derived, not a knob.
Restore cost per host hit: 121 MB over PCIe Gen5 at the measured ~25 GB/s ≈ 4.8 ms on the
attach path, against a 40–70 ms TTFT; spill is async off the engine thread. Host-hit latency,
queue delay, spill/restore bytes, pinned RSS and p99 interference are reported separately from
the hit rate.
vLLM 0.29 has the equivalent (`--kv-offloading-size`); production does not run it, and the
gates compare against both configurations.

### A.2 Session-end demotion

A boundary whose session has ended is dead weight until LRU reaches it; with 150 s of TTL
protection and 2.7 sessions/s ending at 10.8 req/s, dead state alone (≈ 67 GB) exceeds the
device cache, so the TTL protects nothing under pressure. Signals, strongest first:

1. an explicit end-of-session request header from the client (new, small API addition: the
   session header exists, `SESSION_PROMPT_REWRITE` relies on it) — authoritative reclamation;
2. HTTP abort or disconnect before the first token (11–17% of prod requests) — predictive;
3. TTL expiry — predictive, already present.

(1) is the only one that reaches the oracle column; (2) and (3) demote rather than reclaim: the
boundary loses its pin but stays hashed and attachable until the space is actually needed, so a
wrong prediction costs LRU position and, under pressure, a miss — not nothing. P0 measures the
signals' precision and the reuse they lose; how much of the gap (2) alone closes depends on the
prod request mix and is measured, not assumed.

Ordering and ownership. The end header rides the session's final request; the request still
publishes its prompt-end boundary (cheap, and a retry may hit it), and the boundary is recorded
with a per-session **generation** and marked terminal *after* that publish, so a later
publication of the same generation cannot re-pin it. A new request with the same session id
starts the next generation and re-pins on attach (a retry is just that). Demotion releases only
that session's own pin: `SHARED_END` boundaries, other sessions' dependencies and transfer holds
are untouched, and a boundary with in-flight attaches is demoted only when they complete.
Abort/disconnect means the client closed the connection before the response completed;
a normally completed response or stream is not a signal. Implemented: `X-Session-End: 1`
(body `metadata.session_end`) and the mux's disconnect flag reach `retire_slot`, which demotes
after the turn-end publish (`VmmKv::demote_session`: TTL cleared, boundary moved to the cold
band, counted per signal). Both signals are always on; neither adds a knob. Tests: end-before-publish,
concurrent turns of one session, retry after end, cancellation mid-attach — with value checks.

### A.3 What Part A does not change

Per-turn publish still copies 106–121 MB and retires the previous snapshot (335 GB/h on the
engine stream, ~93 MB/s, plus VMM unmap/remap calls); recurrent-state models stay without a
prefix cache; AMD keeps its own `SharedPrefix`. Those are Part B's reasons.

## Part B — The KV block store

### B.1 Row classes

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
sink-plus-window or chunk-local attention are new rules, not new code paths; several window
sizes in one model are separate `Window(w)` classes.

`State` is what Qwen3.8 27B and Kimi-K3 lack today ("recurrent GDN state has no prefix
snapshot", `docs/bringup/results/qwen38-27b-h100/comparison.md`). Its contract is explicit:
the packet exposes a capture op at an aligned row during prefill (the state after exactly `p`
rows, not the slot's current state after the floor-32 rounding) and a restore op; speculative
rollback must restore it too. A model with recurrent state and no capture op gets **no attach**
(prefix cache off, as today): its attention rows are useless without the state. Drafter KV
(`kv.48..52.*`-style tensors) is not cached by any class; a spec-decode hit re-primes the
drafter as it does after a miss.

### B.2 Blocks and keys

A **block** is `R` rows of one class, `R` = the packet's boundary quantum (32 today; hash
granularity, physical block size and legal checkpoint alignment are three separate parameters
of the class, derived per model and rejected when they do not divide). Layout `[tensor][row][hd]`
so each tensor's rows are one run. On 26B FP8 a `Window` block is 3.3 MB, a `Prefix` block
330 KB. Blocks live in per-class slabs — device, and optionally pinned host (B.6) — with fixed
block size per class (free lists, no fragmentation, bounded pointer tables).

Identity: `(namespace, class, chain hash of tokens [0, (i+1)·R))` with the chained FxHash the
radix `PrefixCache` already uses (`memory/vmm.rs:5990`), verified on lookup by comparing the
whole matched prefix's tokens, as today. `namespace` = packet hash (which pins the checkpoint
digest, cache format and positional config) + adapter id + a multimodal-input digest, so one
store can serve more than one packet and never aliases across weights or formats.

Blocks are **immutable once sealed**. A block is created `Filling`, becomes `Ready` when its
scatter copy's completion event has fired, or `Failed` if the copy or its allocation failed;
only `Ready` blocks are visible to lookup, and a `Failed` block is freed once its event has
fired (never before — the slab region may still be a DMA target). A partially written tail
(rows past the last full `R`) is never a block; boundaries are always `R`-aligned (floored, as
today's prompt end is), and the tail is re-prefilled like the 9-row slack. A boundary shorter
than the window pins `Window` blocks `[0, p/R)` (the window clamps at row 0). Two publishers
reaching the same key verify identity (namespace and the matched prefix's tokens) before
coalescing; the second then waits on the first's event instead of copying, and a `Failed`
outcome propagates to every waiter, which aborts its own publish.

The radix tree remains the index. A node at depth `i` holds per class a block ref with its
residency (`Device`, `Host`, `Migrating`, `Absent`) and two counts: **dependents** (boundaries
that need it) and **transfer holds** (gathers, spills, restores in flight). A **boundary**
record at row `p` (`PROMPT_END`, `SHARED_END`, `CHECKPOINT` as today) owns its `State` blobs
and depends on `Prefix` blocks `[0, p/R)` and `Window` blocks `[p/R − w/R, p/R)`.

### B.3 Publish (incremental, transactional)

Trigger points are unchanged (`exec/gpu/prefix.rs:913-1068`: prompt end, shared end, chunk
checkpoint, awaited checkpoint). At boundary `p`:

1. **Check the copy window.** `written − p ≤ ring − window` must hold at the moment the copy
   is enqueued, on the engine stream, after the producing launch and before any launch that
   can write rows ≥ `p + (ring − window)`. An old shared boundary or an oversized chunk that
   violates this is refused (counter), never copied late.
2. Walk the chain. Per node and class: `Ready`/`Filling` → take a dependency; `Absent` and
   retained by this boundary → allocate from the slab (evicting per B.5 if needed), enqueue the
   scatter, record the event.
3. Copy `State` blobs captured at `p`.
4. **Commit**: only after every class's copies have been submitted successfully, publish the
   boundary (its blocks may still be `Filling`; attach waits on them) and then release the
   previous same-session boundary's dependencies (retire-on-publish). A failure after some
   copies were submitted commits nothing: the boundary is never recorded, the previous one
   keeps its pins, and the new blocks go `Failed` and are freed after their events. Blocks
   whose dependents and holds both reach 0 are not freed — they go to the slab's reclaim list
   in LRU order, still `Ready` and hashed, like vLLM's free queue, so an older boundary of the
   same session can still hit until the space is actually needed. A retired boundary stays
   discoverable as an unpinned record until the first of its blocks is reclaimed, which
   invalidates it. A block that goes `Failed` after its boundary was committed invalidates
   that boundary the same way; no attach can have used it, because attach waits for `Ready`.
5. **Reservation**: a publish first reserves its incremental working set (new blocks plus a
   bounded pointer-table slot) from the slab; if eviction cannot free it without touching held
   or `Filling` blocks, the publish is refused (counter), never partially written.
6. **Source lifetimes**: a slot is not retired or re-admitted until every scatter reading it
   has fired its event (today's `vmm_snap_copy` drain, kept); pointer tables are a per-stream
   ring reused only behind the launch event that consumed them.

Per-turn copy becomes the new rows only: ~100–300 rows → 10–35 MB scattered, against 106–121 MB
today; a cold session still copies its full state (~165 MB), and the gate measures the mix. This
is a bandwidth and engine-thread saving, not a retention gain (§1.2). Blocks are owned by the
store, never by a slot, so the store has no orphaned-block hazard; while whole-block VMM
multi-mapping is retained (B.7) the existing strict-publish check stays in force for the mapped
blocks.

### B.4 Attach

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

### B.5 Eviction and demotion

A boundary is **demoted** (its dependencies released, blocks to the reclaim list) or
**evicted** (blocks reclaimed); a block with transfer holds is never reclaimed, and reclaiming
a block invalidates every boundary that depends on it first. Victim order: reclaim-list blocks
(LRU, no decision); demotion candidates per A.2 (ended, aborted, disconnected, TTL-expired);
LRU over live boundaries; a victim block spills to the host slab if one is armed, else is
reclaimed. **Shielded** boundaries (a session boundary inside its TTL, a `SHARED_END` that has
served at least two attaches and is not cold) are victims only once nothing unshielded is
left, in the device LRU, the host LRU and the spill victim choice alike.

### B.6 Host tier

Per class, a pinned host slab allocated in bounded chunks with a bounded number of transfers
in flight; both classes tier (a session is 165 MB wherever it lives). The tier is on by
default and sized by the A1 rule (`MemAvailable` minus an eighth of RAM, at least 16 GiB, at
start); growth re-reads `MemAvailable` every 16 chunks and stops for good once the reserve
would be touched. `PLOW_VMM_HOST_CACHE_MIB` names a size instead, under the same rule. Migration protocol: a block being spilled stays `Device` until its D2H
event fires, then flips to `Host`; a restore pins the host source until the H2D event fires; a
concurrent attach on a `Migrating` block waits on the event rather than racing it. A host
allocation or D2H failure leaves the block `Device` and the eviction falls through to reclaim;
an H2D failure fails that attach, which is unwound as today (`AttachUnwound`, the slot
re-prefills) while the block stays `Host`; a cancelled request releases its transfer holds only
when the in-flight event fires.

Device budget with the tier armed, per class: `slab_bytes = cache_cap − live rings (ring_rows ×
Window row bytes × admitted slots) − Prefix rows duplicated in live slots − in-flight transfer
reservation (max transfers × block size)`; the host slab is sized independently and both are
reported as `kv_store_bytes{class,tier}`.

### B.6a As implemented (B1, B3)

`crates/plowrt/src/memory/kv_store.rs` is the store (classes, chained-hash keys with an
independent check hash, `Filling`/`Ready`/`Failed`, dependents and transfer holds, release-order
reclaim, one byte budget per tier shared by every class, per-class slabs grown in 8 MiB chunks,
async spill with a fence per block and a synchronous restore). The device budget is the pool's
pressure budget: free memory after load minus the eviction floor, set when pressure eviction is
armed (an explicit `PLOW_VMM_CACHE_MIB` caps both the same way); the first replay with a quarter
of the card as the budget held 21 GB against the snapshot path's 40 GB and lost 15 points of hit
rate. Under pressure the pool asks the store to `shrink` before it evicts a boundary; a slab hands
out free slots from its fullest chunk so that emptier chunks keep draining, and `shrink` releases
whole chunks whose remaining blocks are all reclaimable (emptiest first), so a chunk's blocks are
of one age and a few boundary evictions free a chunk rather than scattering holes.

The second replay (budget 48 GiB) held only 27 GB of blocks with 209 live boundaries against
the snapshot path's 40 GB and 245: boundaries die by LRU scattered over chunks, so chunks rarely
emptied and the pressure loop evicted further boundaries to free whole ones. `shrink` now
compacts: it drops the oldest unreferenced blocks in other chunks to make holes, moves the
emptiest chunk's live blocks into them (one drained device copy each, `compactions_total`,
`compact_bytes_total`) and releases the chunk; a chunk with a held, filling or spilling block
is left alone. `device_reserved_bytes` (used plus holes) beside `device_bytes` shows the packing.

Host tier, as measured on the first replay with a 60 GB tier: every spilled block was copied
back by the next publish that met it (143k synchronous restores in ten minutes) and the host
victim search scanned every block per spilled block, which held the engine thread (TPOT p50 8 →
50 ms). Now a publish that meets a host-resident block refills it from the rows it is publishing
(one more segment of its launch, the host slot freed; `refills_total`), a restore happens on a
gather only, and host victims come from a host reclaim list in release order, like device ones.
A gather reads a host-resident block in place: the tier is pinned and device-mapped, so the
block's host bytes are one more segment of the gather's launch (`host_reads_total`), nothing is
copied back, and the block is promoted to the device only by the next publish that meets it.
The memcpy fallback (no copy kernel) still restores first. Review fixes on the same change: `shrink`
checks every live chunk before returning 0 (a bounded candidate scan could hide a releasable
chunk while the pool evicted boundaries for nothing), a held host block is never refilled (a
gather may be reading it), reclaim lists are per class (no re-walking other classes' entries),
and a released chunk index is reused by the next growth. `crates/plowrt/src/exec/gpu/kv_blocks.rs`
is the CUDA side: four classes — `window` (32 rows of every sliding ring and its fp8 ring scales),
`tail` (32 rows of the full layers' partial block and their fp8 scales), `scale` (one whole
block of the full layers' fp8 scale prefix) and `state` (one block per boundary: every
`state.*` tensor's per-slot bytes, the recurrent state after the boundary's rows) — each block
keyed by the chained FxHash of the tokens up to its last row. Copies are segment tables (one
16-byte-aligned run per tensor and head per block) executed by one `plow_kv_segcopy` launch on
the engine stream (`runtime/nvidia/kv_segcopy.cu`, shipped as PTX embedded in the binary and
JIT-compiled by the driver at load; a backend without it runs the table as a memcpy loop).

The boundary tree of `vmm.rs` keeps the policy: a snapshot is `SnapStore::Blocks{handle, state}`
and the pool tells the store through the `SnapBlocks` hook when a boundary dies (`release`), when
the device LRU would spill it (`spill`, the whole boundary; a full store host tier first gives up
the pool's LRU host boundary under the same protection rules as the snapshot tier) and when an
attach needs it back (`restore`). `VmmKv::publish_blocks` runs the engine's acquire-scatter-seal
with the pool lock released; a store that is full refuses the acquire, the engine evicts one pool
boundary (`reclaim_one`) and retries. The pool charges each publish the bytes its new blocks
added, so a turn costs its new rows, not a whole window. The pool's own host tier is not armed
under the store; `PLOW_VMM_HOST_CACHE_MIB` caps the store's pinned slabs instead. Multi-mapping of
whole full-layer blocks is unchanged (B.7).

Replay 7 (2026-10-23, `Prefix` blocks, no host tier, gpu03, 10-minute production-fidelity replay):
at 10 s think and rate 2.7 the store hits 78.6 % with TTFT 53/358 ms and TPOT 10.1/37.7 ms
(p50/p99) against the snapshot path's 74.6 % / 68/645 / 14.0/46.5 and vLLM 0.29's 75.6 % /
119/592 / 16.9/61.6. Two fixes got it there: a slot block the driver could not create evicted
boundaries whose blocks only fed the store's reclaim list (free memory stayed at the floor; now
the pool asks the store to `shrink` first), and the KV admission budget counted only free memory
plus pool blocks, so once the store's slabs held the card the budget fell to the floor and the
mux queued requests (27286 budget denials; the slabs count as held). At 45 s think and rate 1.8
the store hits 58.3 % (snapshot path 48.0 %, vLLM 60.6 %) with TTFT 87/401 ms and TPOT
6.3/34.6 ms against vLLM's 147/569 and 13.5/53.0: 15.8 % of prompt tokens are lost to boundaries
the device LRU dropped at a median age of 45.5 s, which is the think-time median, with the card
at its floor throughout; the store holds about 236 boundaries at about 170 MB each (two thirds
of a boundary is its 1024-row sliding window). The remaining gap at 45 s is capacity, so the
lever is the host tier.

The host tier under the store, as first measured (auto tier of 214 GB, session-end demotion): the
45 s arm hit 60.0 % with TTFT p99 7.4 s and TPOT p99 401 ms, the 10 s arm fell to 66.1 %. Four
causes, each attributed from its counters and fixed in turn, with the arm re-run after each:

1. The pool's OOM policy, inherited from its own slot tier where a host slot had to be freed for
   a spill, evicted cold host boundaries ahead of the device victim and fell back to the host
   LRU, which with the store frees no device bytes (1145 boundaries spilled, 1143 evicted from
   the host again: the tier was a pass-through). Device pressure never evicts a host-resident
   store boundary now; the store's host cap is enforced at the spill. With it the 45 s arm hit
   71.3 %, the loss left being session-end demotion.
2. The host grower kept 256 MiB of pinned chunks ahead per class, so a burst of evictions emptied
   the spares, the spill was refused and the boundary dropped (403 of 1548). 1 GiB ahead per
   class.
3. Gauges for the longest single host allocation, shrink, publish and gather showed the longest
   pinned 8 MiB allocation at 247 ms and the longest publish at 250 ms: `cuMemHostAlloc` faults
   the pages in (page reclaim, zeroing) under the driver lock and every CUDA call of the engine
   thread waited behind it. A chunk is now an anonymous populated mapping registered after
   (`cuMemHostRegister`, device address checked equal to the host address, which the gather
   kernel relies on). The 45 s arm: 71.6 %, TTFT 72/448 ms, TPOT 7.2/36.2 ms, goodput 100 %.
4. At 10 s think every spill happened at publish time on the engine thread: the acquire found no
   slot, the pool spilled the LRU boundary, the retry found the slots still in flight and the loop
   spilled the next boundary until one copy landed (9 ms per publish against 1.5 ms without the
   tier; the device tier drained to 21 GB). A spill-ahead thread per store keeps 2 GiB of device
   headroom (free slots, reclaimable blocks, growth room) by spilling the LRU device boundary,
   unshielded first, and retires completed spills; the publish path's reclaim is the fallback.
   One `shrink` call also releases at most 128 MiB so the store lock is dropped between steps.

Replay 10 (2026-10-23) then showed two more costs: a host gather drained inside the engine
stream (3.9 of 5.3 s of gather time; with the copy kernel no boundary was ever restored, every
host-resident block was read in place over PCIe), and the longest shrink stayed at about 150 ms
whatever the bytes released. Fixed on 2026-10-24 (replays 11-15), each attributed from its
counters:

5. The arrival prefetch (`PrefixProbe::prefetch`, one pool lock per request in the mux) covers
   store boundaries: the store queues one H2D per host block on a restore stream of its own
   (`Where::Restoring`, both slots kept until `reap` sees the fence; `restores_total`,
   `restore_blocks_total`) and a gather of a block still in flight orders the engine stream after
   the fence (`cuStreamWaitEvent`, `gather_fenced_blocks_total`) instead of blocking the host. A
   block no device slot can be made for stays on the host and is read in place as before. A held
   block's fence is not queried until the hold drops: a query that reports done consumes it, and
   the first build left gathered blocks `Restoring` for good (286 failed admissions in five
   minutes).
6. Compaction copies drain on a DtoD stream of their own, not behind the spill backlog, with the
   store lock held only to plan (a transfer hold on each moving block, destinations taken) and to
   commit (a block another transfer held meanwhile keeps its slot). Released chunks are freed on a
   store worker thread: a device free synchronises the context, so each shrink on the engine
   thread had waited for the spill stream's queued copies; until the free lands the bytes count as
   coming free (`pending_spill_bytes`). Longest shrink 150 ms → 30 ms, shrink time 1.75 s → 0.24 s
   per 10 s-think arm.
7. At 10 s think a request is admitted within milliseconds of arriving, before its restore lands,
   and a boundary the pool holds on the device shares blocks spilled with another boundary that
   the pool never asked the store to restore (17k in-place host reads per arm). The prefetch now
   also runs the store's restore for a device boundary, and admission holds a request up to 50 ms
   while the store reports its boundary's restore in flight (`restore_holds_total`), so the copy is
   paid by that request's queueing rather than by every slot in the batch. In-place host reads
   fell to 2.4k and fence-ordered gathers to 0.9k per arm.
8. Not the cause, measured and reverted: transparent huge pages for the pinned chunks (6.2 ms per
   8 MiB against 4.3 ms, the kernel's direct compaction inside the populate, and the compaction
   stalls reached every thread; TTFT p99 20 s in the last minute). Registration itself is not the
   lever either: a 16 GB tier finishes growing within two minutes (2.4k chunks, 9 s of
   registration in total) and degrades exactly like the auto tier.

Measured on the final build (`120a2202`, replay 15): at 45 s think the store with the host tier
hits 71.7 % (vLLM 60.6 %, no host 58.3 %, ideal 73.9 %) with TTFT 70/376 ms and TPOT 6.5/31.8 ms
against the no-host arm's 87/401 and 6.3/34.6 and vLLM's 147/569 and 13.5/53.0, goodput 100 %:
above both on every metric. At 20 s think it hits 78.4 % (no host 71.7 %, vLLM 72.6 %, ideal
78.5 %) with TTFT 53/446 ms and TPOT 8.1/41.1 ms against vLLM's 100/489 and 12.2/42.1. At 10 s
think the tier adds one point (79.7 % against 78.6 %) and still costs: TTFT 65/707 ms and TPOT
16.5/89 ms against the no-host arm's 53/358 and 10.1/37.7 (TTFT p99 was 1070 ms and TPOT p99
148 ms before items 6-7). The cost is steady state, not the tier's growth: a 15-minute arm holds
81-83 % with TPOT p50 17-24 ms throughout. Its mechanism, from the per-tick histograms
(`plowrt_tick_duration_seconds`) and the speculation counters: the tier's tick-time cost is small
on average (decode ticks +3-6 %, prefill ticks +11 %) but has a tail the no-host arm lacks
(about 100 decode ticks over 25 ms and 25 over 100 ms per arm against 8 and 0), and the longer
turns raise the in-flight batch from 7 to 11-14 rows, where the speculation policy's gain falls
under its margin (drafts fit in the same decode rung) and drafting stops on 20-30 % of decode
ticks against 1 %: a second, self-sustaining equilibrium. With speculation off on both arms (the
production setting) the tier's cost is a fifth of that at p50 (TPOT 19.9 against 16.7 ms) but the
tails remain (TTFT p99 757 against 360 ms, TPOT p99 94 against 42 ms), so a per-tick cost under
heavy spilling is still there; what it is needs a profiler trace of the engine thread under
spilling (candidates: the spill-ahead thread's driver calls and 2 ms polling, copy-engine
contention with the tick's own transfers, the remaining fence-ordered gathers). The strict
comparison keeps `PLOW_VMM_HOST_CACHE_MIB=0` at its 10 s think; the tier is the right setting
where the think time is long enough for the device alone to lose boundaries (production's is
about 45 s). Known teardown message since the prefetch: `cuMemGetInfo` after the context is
torn down logs "CUDA context poisoned" once at exit.

Recurrent models (B3): the Qwen3.8 packet keeps head-interleaved rows in its full-attention
caches (the LIVE manifest's `layout`), which the prefix geometry had mapped head-major, so the
first prefix-mode request faulted at row 1024 of a slot; the geometry now folds those heads into
the row (one track of `heads × hd` per layer), and the mapping, the tail blocks and the partial
copies follow the rows the kernels write. The serve manifest's KV geometry also skips
`linear_attention` layers instead of rejecting the config. The prefill chunk that would cross the prompt-end boundary is cut to end
exactly there (`state_capture_cap`; only when at least the smallest prefill bucket is left to
it) and the boundary is published at that chunk end, while the state tensors hold the state
after its rows (`state_capture_publish`); the later prompt-end publish finds it and touches it.
A boundary the slot has moved past is not publishable for such a model (no state to capture),
so a prompt end evicted during decode is not restored at the turn end. Prefix caching for a
model with recurrent state is accepted only with the store on.

Copy backend, measured (gpu03 H100, `kvstore-probe/segcopy`, random 8 KiB segments over a 2 GB
arena, second repetition):

| segments × bytes | memcpy loop submit / GPU | segcopy kernel submit / GPU |
|---|---|---|
| 12,800 × 8 KiB (a full window gather) | 32.1 ms / 32.1 ms (3.3 GB/s) | 0.07 ms / 0.16 ms (644 GB/s) |
| 1,200 × 8 KiB (one turn's new rows) | 3.0 ms / 3.0 ms | 0.01 ms / 0.02 ms |
| 2,560 × 16 KiB | 6.3 ms / 6.3 ms | 0.02 ms / 0.05 ms (886 GB/s) |
| 12,800 × 128 B (ring scales) | 31.3 ms / 31.3 ms | 0.05 ms / 0.07 ms |

A memcpy loop costs ~2.5 µs per segment whatever its size; the kernel is launch-bound at
~0.1 ms. The store runs the kernel (PTX embedded); the loop is the fallback only.

Not a knob: the store has no switch; the reserve, chunk size and class set are derived.

Known limits of B1 (accepted until B4, each gated rather than assumed):
* Dedup is keyed by tokens, as the radix tree's whole-block sharing already is; a 32-row block
  computed under one prefill bucket or inherited prefix stands in for the same tokens computed
  under another. KV rows and their fp8 scales travel in one block, filled together. The
  exactness gate (store vs snapshot path, greedy multi-turn at block and window edges) and the
  FP32 gate are the contract, not the hash.
* Slab chunks belong to the class that grew them; the budget moves between classes only when a
  chunk is released (`shrink`, run first under device pressure). The pool's byte charges are per
  publish (new blocks) and approximate; `plowrt_kv_store_{device,host}_bytes` are the truth.
  Growth asks the backend for free memory every 8 chunks and stops while a chunk would take it
  under the pool's floor: without that check the pool shrank the store and the next publish
  grew it right back (58k chunk allocations and 66 GB of compaction copies in a 10-minute
  replay, 24 s under the pool lock; `plowrt_kv_store_{device_grow,shrink,spill}_us_total`
  attribute it).
* Publish and non-window gathers drain the engine stream before sealing or releasing blocks
  (the snapshot path drained the same way); a launch-window gather keeps its holds until the
  step's drain. Hash chains are recomputed per publish (16K tokens, microseconds) and the
  segment table is uploaded synchronously per copy (~600 KB); both are below the copy itself.

### B.7 What VMM still does

Per-slot sparse growth of the full-layer tracks (`ensure_rows`, premap) and the live rings.
On CUDA the store holds every full-layer row of a boundary (`Prefix` blocks, B4): the radix
tree indexes boundaries only, a publish takes no block references, an attach maps the slot's
own blocks and gathers every row into them, and the strict-publish scale hazard is moot.
Cached `Prefix` rows exist once in the store and once per live slot (≈ 59 MB × in-flight
slots, ~1.5 GB at 27). The whole-block multi-map code remains for the AMD shared prefix (B2).

### B.8 Hardware and model coverage

| | today | store |
|---|---|---|
| NVIDIA H100, TP=1, one packet per process (Gemma-4 12B/26B) | VMM multi-map + snapshot copy | `Prefix` + `Window` + scales — **B1 acceptance scope**; multi-packet namespaces, TP and CPU/NUMA below are extensions outside it |
| NVIDIA TP>1 | none | future; the key carries a rank/shard id so a later TP attach can require a boundary `Ready` on every rank before any rank proceeds |
| AMD MI300X/MI325X (GLM-5.3, DeepSeek, Kimi MLA) | `exec/amd/shared_prefix.rs`, VMM groups + tail copy | same store, `Prefix` over `ckv/krot/kidx/scale` with per-tensor descriptors (different row cadences allowed), HIP gather |
| AMD per-slot route (`exec/amd/prefix.rs`) | carried state + ring windows, same slot only | `Window` + `State`, cross-slot |
| Qwen3.8 27B GDN, Kimi KDA | prefix cache off | `Prefix` + `State` once the capture op exists |
| CPU Xeon (26B BF16) | donor-slot memcpy | host slab is the store; gather = memcpy; NUMA placement per slab |
| Apple | none | not in scope |

The store is one Rust module (`memory/kv_store.rs`) over an ops trait: `alloc_slab`,
`scatter`, `gather`, `copy_h2d/d2h`, `event`/`wait`. Backends implement it with their own
synchronisation (CUDA/HIP events, synchronous copies on CPU); the index, boundary records,
counts, state machine, eviction and host tier are shared.

### B.9 Rejected alternatives

- **Paged slots (vLLM block tables).** Every sliding read/write site (`op_attention.cuh` decode
  bodies and prefill stager, `op_norm.cuh` writers, TMA maps, FP8 scale indexing) would take a
  `(slot, logical block) → physical` lookup; `live_kv::validate` and the `kv_ring` proof assume
  `mask == stride − 1`; `crates/devgen/src/lib.rs:2378` records paging as a measured null on
  these kernels. The store keeps the flat slot and copies instead.
- **512-row blocked full-layer layout** for zero-copy `Prefix` mapping: a kernel addressing
  change for ≤ 3% of cache bytes. Not part of this proposal.
- **Recomputing the window on attach**: not exact (§1.1); exact recompute is the full re-prefill
  the miss path already does.
- **Lower-precision window storage**: changes numerics; the FP32 gate owns that budget.
- **The block store as the retention fix**: §1.2 shows it is not one.

## 3. Expected outcome vs vLLM 0.29

| | vLLM 0.29 device-only (prod flags) | vLLM 0.29 + CPU offload | plowrt today | plowrt + Part A | + Part B |
|---|---|---|---|---|---|
| bytes / retained session | ~165 MB | ~165 MB | ~156 MB | ~156 MB | ~165 MB |
| retention tiers | device | device + host | device | device + host (snapshots) | device + host (blocks) |
| session-end demotion | none | none | TTL 150 s | header / abort / TTL | same |
| hit at 10.8 req/s, think 20 s (sim) | 61% (48 GB) | ~75% | 55% (41 GB) | ~75% | same |
| hit at 10.8 req/s, think 45 s (sim) | 33% | ~61% (100 GB) | 29% | 61% + end signal → up to 75% | same |
| per-turn publish | in place | in place | 106–121 MB + map/unmap | same | 10–35 MB |
| recurrent-state models | Mamba/GDN prefix caching | same | off | off | `State` |
| TTFT / TPOT at equal hit, same stream, 6.5 req/s | 82 / 11.5 ms (0.28) | — | 40 / 10.8 (MTP off), 40 / 5.2 (MTP on) | unchanged | unchanged |

Predeclared hypotheses for Part A, over a finite load envelope (rates 2.7 and 4.5 of the
prod-fidelity plan, think 20 s and 45 s, 100 GB host tier, end signal present on the replay's
final turns and, separately, absent): (i) token hit rate within 2 pt of the oracle ceiling —
the oracle's 6 pt above the simulation's 75% is the first-turn system-prompt share, four ~2K
prompts that are always resident, which the measurement confirms rather than assumes; (ii)
same-stream TTFT/TPOT held; hence (iii) SLO goodput (TTFT ≤ 200 ms, TPOT ≤ 30 ms) ≥ 20% above
vLLM 0.29's and maximum sustainable rate ≥ 20% above, against vLLM device-only and with
offloading at the same host budget, device KV+cache budget equal to vLLM's allocated KV bytes.
These are hypotheses: the per-token margins above are vLLM 0.28 numbers and the hit-rate tie
with offloading follows from the simulation, not from a measurement. Acceptance rule: 3
repetitions per arm, paired by repetition; the mean paired difference's 95% confidence interval
(t, n = 3) must lie entirely above the threshold. Part B's hypothesis is narrower: publish bytes
−70% or better on the replay's turn mix, attach latency within 2% of today's, `State` models
cached, with Part A's hit rate preserved — and a measured engine-thread or p99 benefit from the
copy reduction, since ~93 MB/s of average copy traffic is not a cost by itself.

## 4. Gates

Each phase stops at a failing gate. Replays use identical recorded arrivals, prompts, output
lengths and a cold cache (the harness on this branch lets explicit `--think-*` flags win over
`--prod-fidelity`, which silently reset them to 10 s before); MTP-off arms are separate from
MTP-on; final numbers come from `campaign.py report` in the strict format of
`docs/bringup/agent-tools.md`.

0. **Measure first (P0).** (a) Oracle ceiling of the replay plan: done, 81.1% (§1.2). (b) The
   production think-time distribution and the fraction of sessions whose end is detectable
   (abort/disconnect) — from timestamps, no request bodies. (c) vLLM 0.29's real pool: done
   on gpu03 with the prod flags — "Available KV cache memory: 44.07 GiB", 419,749 tokens
   (≈ 112.6 KB/token), 25.6× concurrency at 16K; the 48 GB simulation column holds. (d) The
   120 GB pinned bring-up crash root-caused. (e) Today's snapshot cache with a bounded 60–100 GB
   host tier at rates 2.7 and 4.5 with think 20 s and 45 s — the retention columns of §1.2
   measured, not simulated.
1. **Exactness.** Attach-then-decode logits (not only token digests) equal to the snapshot
   path's attach (the qualified production path) at ctx 1000–12000 across the window edges and
   the 4094/4095/4096/8191/8192 boundaries, FP8 KV, both layer kinds, multi-turn with a
   sibling branch per conversation: measured 0 of 40 turns differing (2026-10-23, store build
   41196f68 against a803cf92 with the store off). The uncached prefill is **not** a valid
   reference: the suffix prefill runs different GEMM shapes, so any attach differs from the
   cold prefill at the first logprob (37 of 40 turns on the snapshot path itself, text flips
   only on the flat-distribution 12K random-word prompts). The existing poison and boundary
   sweeps; on a `State` model the within-path check (the same server probed twice, cold then
   attached) and the FP32-reference gate at c1.
2. **Lifecycle faults (Part B).** Forced eviction, slab exhaustion, delayed copies, concurrent
   attach/publish/spill on one key, cancellation mid-gather, slot reuse, hash collisions,
   partial-block boundaries, `Migrating` races — each with value checks.
3. **Kernels untouched, interference bounded.** `step_bench` digests identical to the base
   packet; serving ITL/TTFT under load within 2% of today's at rates 1.8 and 2.7; achieved
   copy bandwidth, copy amplification, pinned RSS and queue delay reported.
4. **Serving win (§3).** Steady-state arms: warm-up longer than one retention turnover, then
   ≥ 3 session lifetimes at rates 2.7, 4.5 and overload, 3 repetitions, with offered vs
   completed, backlog, timeouts and cancellations; ablations separating host tier, demotion
   signal and (Part B) incremental publish.
5. **Numerics.** FP32 gate at c1/c8/c32 unchanged.
6. **Soak.** 1 h at rate 2.7: flat RSS, device mapped bytes, counters consistent, no
   engine-thread stall > 1 ms attributable to the cache (pfx_trace).

## 5. Phases

| | content | gate |
|---|---|---|
| P0 | §4.0 measurements; vLLM 0.29 device-only and offload arms on gpu03 | — |
| A1 | bounded snapshot host tier (today's mechanism, allocation after load, chunked); cap derived from host RAM, no new knob | 0(e), 3, 4, 6 |
| A2 | session-end header in the serve API + demotion on header/abort/disconnect (A.2), always on, metrics `session_end_{header,abort}_total` | 4 with the end-signal ablation, 5 |
| A3 | production deploy of A1+A2 on gpu04 (user go) and the same-hour comparison the Grafana endpoint allows | strict report |
| B1 | `memory/kv_store.rs`, CUDA ops, `Prefix` + `Window` (+ scales), block state machine, gather/scatter kernels, publish/attach at today's trigger points, today's eviction order, on by default once its gates passed (the switch went with B4's first step); TP=1; tier and demotion carried over | 1, 2, 3, 4 device-only, 5 |
| B2 | AMD: `SharedPrefix` and the per-slot route on the store; HIP gather. Parked: the pool substrate (`SnapStore::Blocks`, `SnapBlocks`, `publish_blocks`) compiles under `hsa`; the engine-side gather needs a box | AMD rung + replay gates, when a box is available |
| B3 | `State` class: the engine's `state.*` tensors captured at a chunk end aligned to the boundary (B.6a); GDN (Qwen3.8 27B) first, KDA (Kimi-K3) when its state tensors follow the same contract. Measured 2026-10-23 (one server probed twice, 300–8191-token conversations): a cold turn and its attached replay are bit-identical, state included, whenever the publisher's rows were computed in the same prefill launch shape; an attach whose shared blocks came from a longer publisher (a 1K–8K-row chunk against the probe's 832-row chunk) differs at rounding scale (first-token logprob within 1e-3), as the snapshot path and vLLM's cache do, because an attach returns the publisher's KV. The FP32 gate's kl_p95 sits 7 % over its limit with the store on against 6 % under on the no-prefix path (trimmed means 0.0112 vs 0.0102): the gate's margins were calibrated on the no-prefix path and need re-measuring on the packed-prefix path before B3 is qualified. Merge prep 2026-10-25: the production FP8 recipe serves with the prefix cache off, and on that path the branch's c1 and c8 captures equal the 2026-10-20 qualifying run digit for digit (the recipe's present decisive-drop allowance of 0.022 fails that qualified packet at c1 on main as well, two flips against the 0.05 the qualifying run used; c8 and c32 fail identically on both, requests refused under the single-slot decode); within the prefix path an attach is exact (0 of 16 turns differing), so prefix-on Qwen stays opt-in until the margins are re-measured there | 1 (state), 5 and replay on Qwen3.8 |
| B4 | done on CUDA: the contiguous snapshot copies, `PLOW_VMM_SNAP_{PITCHED,POOL}` and `PLOW_KV_STORE` removed, the store is unconditional, `Prefix` blocks hold every full-layer row (no multi-map, no strict publish in this mode). `docs/arch/22` §7–9 rewritten. Remaining: the AMD snapshot and multi-map path (B2); ring park/recycle stay (live-ring admission latency, not boundary storage) | exactness + FP32 gates, replay vs the multi-map build and vLLM |

Metrics (Part B): `plowrt_kv_store_{blocks,bytes}{class,tier}`, `publish_{blocks,bytes,
refused}_total`, `attach_{hits,misses,rows,wait}_total`, `attach_from_host_{total,bytes}`,
`evict_{lru,spill}_total`, `migrations_inflight`; the `plowrt_prefix_*` names stay for the kept
semantics.

Knobs: the work adds no permanent tuning knob. Behaviour that qualifies becomes the default;
the store's rollback switch existed only for its qualification window and is gone.

## 6. Open questions

1. The end-of-session header's shape and whether the production clients can send it; without
   it A2 is limited to the abort/disconnect fraction P0(b) measures.
2. `R` per model: 32 matches today's boundary quantum; the gather cost is per row, not per
   block, so larger `R` only coarsens dedup.
3. The `State` capture op's cost on the prefill rungs (one state copy per aligned checkpoint).
4. Whether the simulation's per-session byte model holds for vLLM 0.29 once P0(c) gives the
   real pool and page sizes.
