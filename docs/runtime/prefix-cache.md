# Prefix caching and unified batching

Both selectors default on for AMD and NVIDIA. Runtime selection also requires
compatible assets; enabling a selector does not qualify an unsupported execution
path. Check startup selection messages and actual token-batch dispatch logs.

| Control | Default | Effect |
|---|---|---|
| `--prefix-cache` / `PLOW_PREFIX_CACHE` | `true` | Permit prefix reuse on either backend. |
| `--token-batch` / `PLOW_TOKEN_BATCH` | `true` | Permit unified prefill/decode dispatch. |
| `--vmm-cache-memory-utilization` / `PLOW_VMM_CACHE_MEMORY_UTILIZATION` | `0.05` | Prefix block/snapshot budget per engine as a fraction of device memory, in the unit of vLLM's `--gpu-memory-utilization` (4 GiB on an 80 GiB H100); zero allows OOM-driven eviction only. |
| `--vmm-cache-mib` / `PLOW_VMM_CACHE_MIB` | unset | Explicit budget in MiB, overriding the fraction; zero allows OOM-driven eviction only. |

Use `--prefix-cache=false` or `PLOW_PREFIX_CACHE=0` to disable reuse. On NVIDIA,
this also overrides `PLOW_VMM_PREFIX=1`. `PLOW_VMM_PREFIX=0` remains a
backend-specific opt-out. Disable unified dispatch separately with
`--token-batch=false` or `PLOW_TOKEN_BATCH=0`. Explicit fusion takes precedence
over unified batching.

NVIDIA uses the shared VMM prefix cache on compatible assets. AMD currently
reuses a prefix within the same physical sequence slot, retaining flat KV and
snapshotting recurrent state, sliding-window rows and any FP8 KV scales. Its
snapshots are bounded and evictable; an allocation OOM skips caching. Prefix
publication keeps a final input row available for sampling. Cancellation of a
miss invalidates its old prefix before new KV writes.

AMD prefix reuse is unavailable with its current lazy VMM allocation path,
unknown KV layouts, or a snapshot larger than the configured budget. Explicit
fusion retains its existing execution path. Tensor-parallel prefix snapshots
require all ranks; AMD unified batching still requires its single-GPU packet
and code-object capabilities. All compiled decode and prefill rungs remain
available to ordinary execution.

The GLM-5.3 MI300X TP8 screen selects prefix caching with the default settings.
Unified batching is requested by default but declines with
`token batching does not support tensor parallelism`; that run uses ordinary
execution. Enabling the selector does not remove this implementation limit.

Fresh single-GPU Hopper builds emit packed-request metadata by default for
BF16 weights and BF16-activation FP8 weights (`--fp8` or `--w8a16`) with BF16 KV.
`--emit-packed-prefill=false` disables that emission. Activation-FP8 packing
(`--w8a8`) remains explicit opt-in; packed FP8-KV execution is not qualified.
AMD retains its own packed-program contract and does not receive NVIDIA request
metadata, including when `--emit-packed-prefill=true` is selected.

Host tests and emitted AMD BF16/FP8-KV packet checks cover the snapshot layout,
window wrap, configuration and scheduling invariants. H100 tests cover actual
BF16/FP8-weight dispatch. AMD device correctness/performance and sustained
production qualification remain pending; build compatibility alone is not that
evidence. The H100 readiness report lives with the campaign's raw `perf-data`, which is
kept out of source control.

## Retention policy on the CUDA VMM pool: the boundary tree (`PLOW_PREFIX_POLICY`, 2026-10)

Mechanism (unchanged since 2026-09): the full-attention layers' KV is shared zero-copy as 2 MiB
granules through a token-verified radix tree of whole blocks; everything else a prompt needs to
resume at a row inside a block (the sliding layers' last `window` rows, the full-layer rows past
the last whole block, FP8 scales) is a private *boundary snapshot* (`vmm_snap_bytes`, ~118 MiB per
prompt end on Gemma-4 26B FP8, a stub on uniform full attention). Pressure eviction keeps free
device memory above `PLOW_VMM_CACHE_MIN_FREE_MIB`.

Policy (`crates/plowrt/src/memory/vmm_policy.rs`, `Policy::Tree`, the default):

* `Snap { id, kind, chain, parent, children, users, last_used, store }`. `kind` is what the
  publisher said (`PROMPT_END`, `SHARED_END`, `CHECKPOINT`; a dedup or a touch ORs). `parent` is
  canonical at publish time (historical: a shorter boundary published later does not adopt the
  existing longer ones, so a parent can be an ancestor further up than the nearest retained
  boundary): the longest retained boundary strictly below this one on its chain, found by
  the same walk an attach does, so an identical boundary published from two attach points has
  one parent. `children` are the retained boundaries whose parent this is; every free unlinks
  synchronously and re-parents the freed boundary's children to its parent.
* Eviction: one LRU key, `last_used`, over snapshots with `users == 0` that no in-flight waiter
  awaits (`Await { chain, rows }`, set by `inflight_prefix`, cleared at the waiter's attach,
  cancellation or next sequence). The unit is the snapshot plus the radix leaves only it kept
  attachable. No MRU exemption, no pin tier, no shared tier, no recency window.
* Recency moves on consumer demand only: an attach refreshes the attached boundary and every
  ancestor on its chain (a hot system prompt end stays warm while conversations continue through
  it); the turn-end pass touches the conversation's prompt end (metadata only, before the
  ring-overflow check, so a long reply never stops the refresh) and restores it if pressure
  evicted it during the decode. A republish of an identical boundary refreshes nothing.
* Structural retire: when `C` is published (or re-published) with parent `B`, `B` is freed iff
  `B.children == {C}`, `B.users == 0`, `B.pending == 0`, `B` is not awaited, `B` is not a
  `SHARED_END`, and `C` survived its own trim. The same rule runs when a sibling's eviction or a
  hold's release leaves one continuation. `pending` counts requests that attached `B` and have
  not continued it (a publish below it, a touch of it, or a strict-publish refusal of a boundary
  another request already published from the same chain): a parallel pair's second request
  whose own prompt end strict publish refused, a cancelled request. Each is an expected
  continuation without a boundary of its own, so `B` waits for it and only LRU takes `B`; the
  count never drops for a request that ended without continuing. A later request that extends
  `B` and diverges below `C` (a regenerate) recomputes from `B`'s parent: a miss, never a wrong
  attach (identity stays the token-verified radix path plus the tail tokens).
* Strict publish (FP8 KV: `resolve_prefix_hazard`) is policy-independent and unchanged: a
  request that rendered blocks another request already published cannot publish a boundary
  over them. A regenerate below a live sibling branch, and the second of a parallel pair
  admitted without waiting on the first, therefore keep their attach point (`pending`) but get
  no boundary of their own; a system prompt shared across conversations gets its shared end
  only from a pair that overlapped in prefill (`inflight_prefix`), never from a conversation that
  arrived after its twin finished (`cell_strict_publish_refuses_a_sequential_shared_end`).
* `SHARED_END` boundaries (published because another recent or in-flight prompt shares that
  prefix) are never structurally retired: their second continuation is expected. They survive by
  LRU refresh alone. A shared end within 64 rows of the owner's prompt end is the same history (a
  parallel pair, a retry) and is not published at all: the waiter attaches the prompt end.
* `X-Session-Id` has no cache meaning. Turn-end output boundaries are not published.

Invariants (`vmm_sim_tests::cell_*`, `policy::check_tree`): a held, awaited or in-transfer
snapshot is never freed on any path (LRU, retire, radix leaf eviction, strict publish's tail
eviction; a node's snapshots are freed one by one so each unlinks while the rest are indexed); `parent.rows < rows` and the parent lies on the chain; children sets hold
retained ids only; a retire needs exactly one retained continuation.

### Host tier: level 2 of the LRU (`PLOW_VMM_HOST_CACHE_MIB`, tree policy only)

Production's loss is retention: at 300-370 requests/min the device holds a prompt end for 47-57 s
against a median think time of 45 s, so a quarter of the turns come back after their end was
evicted and re-render ~900 rows from a block-aligned ancestor. The device cannot grow; pinned
host memory can. With the tier armed the device LRU's victim is *spilled*, not freed:

* `SnapStore::{Device(va), Spilling{va,slot,fence}, Host(slot), Restoring(slot)}`. A spill is one
  async D2H copy into a fixed host slot (the largest snapshot the engine can publish; chunks of
  1 GiB grow on the `vmm-host-tier` thread a chunk ahead of demand, never on the engine thread).
  The device buffer outlives the fence: `reap_spills` (every trim) frees it once the copy
  retired; the cap trim credits in-flight spills and never waits, OOM reclamation finishes the
  oldest in-flight spill before queueing another; at most 4 are in flight.
* The tier is the same LRU one level down: a device eviction spills; a full tier first drops its
  least recently used host snapshot entirely (slot, node, dead blocks, `host_evictions`). The
  radix blocks of a host-resident boundary stay on the device: the boundary is still found by
  the same walk, `cached_rows` still counts it, admission still sees it.
* An attach to a `Host` snapshot claims it (`Restoring`, so a prefetch or a second attach waits
  instead of copying again) and restores it (`alloc_snapshot` + synchronous H2D) with the pool
  lock released, under `users` so no eviction touches it; a failed restore unwinds the window to
  a miss and leaves the snapshot on the host for the next try. An attach to a `Spilling`
  snapshot waits for the fence and keeps the device copy (the slot goes back). A restored
  snapshot is the MRU (no device/host ping-pong). The mux queues a restore at arrival
  (`PrefixProbe::prefetch`, `Restoring`, held) so the copy overlaps the queueing; the attach
  waits on it when it is still in flight.
* A fence that cannot be queried or fails on wait, or a copy submitted without a fence,
  quarantines the snapshot: the device buffer is valid (a D2H copy only reads it) but never
  freed or spilled again, the host slot is never reused (`host_fence_failures`). The tier
  counts quarantines independently of the snapshot, so teardown leaks the pinned chunks while
  any exists, even after the snapshot itself was evicted.
* Accounting: `cache_bytes`/`snapshot_bytes` stay device-only; `host_snapshots`,
  `host_snapshot_bytes`, `host_capacity_bytes` are the tier's; `snapshots_spilled`,
  `snapshots_restored`, `host_evictions`, `spill_fallbacks` (too large for a slot, no slot, copy
  refused: freed as before), `host_prefetches`.

Invariants (`host_tier_*`, `failed_*`, `prefetch_*`, `cell_host_tier_*` tests): exactly one
owner per buffer (device or host) outside `Spilling`; a `Spilling`/`Restoring` snapshot is never
freed, never a victim, never spilled twice; a held snapshot is never spilled; `users` releases
exactly once per attach, prefetch or restore; a snapshot's blocks go only with the snapshot.

`PLOW_PREFIX_POLICY=legacy` keeps the 0.1.2 flag/tier policy (with the phase-0 fixes: the MRU
exemption is bounded by `SHARED_RECENT_TICKS`, a reuse distance in publishes + attaches; eviction
honours waiters; the turn-end restore runs regardless of `PLOW_PREFIX_CACHE_OUTPUT`; the live ring
is released before the retire trim) for one release.
