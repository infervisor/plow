# Prefix caching and unified batching

Both selectors default on for AMD and NVIDIA. Runtime selection also requires
compatible assets; enabling a selector does not qualify an unsupported execution
path. Check startup selection messages and actual token-batch dispatch logs.

| Control | Default | Effect |
|---|---|---|
| `--prefix-cache` / `PLOW_PREFIX_CACHE` | `true` | Permit prefix reuse on either backend. |
| `--vmm-cache-memory-utilization` / `PLOW_VMM_CACHE_MEMORY_UTILIZATION` | `0.05` | Prefix block/snapshot budget per engine as a fraction of device memory, in the unit of vLLM's `--gpu-memory-utilization` (4 GiB on an 80 GiB H100); zero allows OOM-driven eviction only. |
| `--vmm-cache-mib` / `PLOW_VMM_CACHE_MIB` | unset | Explicit budget in MiB, overriding the fraction; zero allows OOM-driven eviction only. |

Use `--prefix-cache=false` or `PLOW_PREFIX_CACHE=0` to disable reuse. On NVIDIA,
this also overrides `PLOW_VMM_PREFIX=1`. `PLOW_VMM_PREFIX=0` remains a
backend-specific opt-out. Unified prefill/decode dispatch is always selected
where supported; explicit fusion takes precedence over it.

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

## Retention policy on the CUDA VMM pool: the boundary tree (2026-10)

Mechanism (CUDA, 2026-10): every row a boundary needs lives in the KV block store (below) as
32-row content-addressed blocks, one slab per row class — the full-attention layers' rows
(`Prefix`), the sliding rings' last `window` rows (`Window`), a recurrent model's state (`State`),
fp8 scales with their rows — with a pinned host tier beneath the device slabs. Nothing is
multi-mapped: an attach gathers the rows into the slot's private blocks. Pressure eviction keeps
free device memory above `PLOW_VMM_CACHE_MIN_FREE_MIB`.

Policy (`crates/plowrt/src/memory/vmm_policy.rs`, `Policy::Tree`, the only CUDA policy):

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
  attachable. No pin tier, no shared tier, no recency window; the only MRU exemption is the
  soft-cap trim's (`preserve_hot`): it keeps the most recently used boundary while a live request
  maps its blocks, and allocation OOM may still take it.
* Recency moves on consumer demand only: an attach refreshes the attached boundary and every
  ancestor on its chain (a hot system prompt end stays warm while conversations continue through
  it); the turn-end pass touches the conversation's prompt end (metadata only, before the
  ring-overflow check, so a long reply never stops the refresh) and restores it if pressure
  evicted it during the decode. A republish of an identical boundary refreshes nothing.
* Structural retire: when `C` is published (or re-published) with parent `B`, `B` is freed iff
  `B.children == {C}`, `B.users == 0`, `B.pending == 0`, `B.ends == 0`, `B` is not awaited, `B`
  is not a `SHARED_END`, and `C` survived its own trim. `ends` counts the requests whose prompt
  ended at `B` (published it or found it published) that no later prompt end on the same chain
  has superseded: each new prompt end decrements the nearest ancestor with `ends > 0`. A
  parallel pair or a retry ends twice at one boundary and so expects two continuations, which
  may diverge (lab dup-pair replay: 22% of twin replies differed at temperature 0); the first
  continuation does not retire the boundary the second still needs. Without a second
  continuation it goes by LRU. The same rule runs when a sibling's eviction or a
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
* `X-Session-Id` buys bounded retention (`PLOW_PREFIX_SESSION_TTL_S`, default 150 s, `0` = none):
  a session request's prompt end (and the boundary it attached, at admission) gets
  `protect_until = now + TTL`, refreshed by every use (publish, turn-end touch, the next turn's
  attach). Eviction takes a protected boundary only once nothing unprotected is left (the host
  tier's entries included), so the cache cannot deadlock. An expired one is demoted lazily (a
  sweep at most once a second from `trim_cache`, no timer thread): its snapshot, then each
  radix block only it kept attachable, newest first (`PrefixCache::evict_chain_from`); shared
  ancestors stay under LRU. A boundary in use, awaited, mid-transfer, or whose blocks a live
  request still maps waits for the next sweep. A full host tier never drops a protected entry to
  take an unprotected spill (that victim is freed instead).
  `/metrics`: `plowrt_prefix_sessions_protected`, `plowrt_prefix_session_demotions_total`,
  `plowrt_prefix_session_demoted_bytes_total`. Turn-end output boundaries are not published.
* Session end (`X-Session-End: 1`, body `metadata.session_end`, or a client disconnect before the
  response completed): at the slot's retire, after its turn-end publish, the session's prompt end
  loses its TTL and moves to the cold band (`last_used` below every live boundary), so pressure
  takes it before a live session's; the bytes stay until then and a retry or a later turn attaches
  and re-warms it (`VmmKv::demote_session`, `cell_session_end_*`). The header is authoritative, the
  disconnect predictive; neither is a knob. `/metrics`: `plowrt_prefix_session_end_header_total`,
  `plowrt_prefix_session_end_abort_total`.

Invariants (`vmm_sim_tests::cell_*`, `policy::check_tree`): a held, awaited or in-transfer
snapshot is never freed on any path (LRU, retire, radix leaf eviction, strict publish's tail
eviction; a node's snapshots are freed one by one so each unlinks while the rest are indexed); `parent.rows < rows` and the parent lies on the chain; children sets hold
retained ids only; a retire needs exactly one retained continuation.

### Host tier: level 2 of the LRU (`PLOW_VMM_HOST_CACHE_MIB`)

Production's loss is retention: at 300-370 requests/min the device holds a prompt end for 47-57 s
against a median think time of 45 s, so a quarter of the turns come back after their end was
evicted and re-render ~900 rows from a block-aligned ancestor. The device cannot grow; pinned
host memory can. With the tier armed the device LRU's victim blocks are *spilled*, not freed:

* Size: unset = the host-reserve rule, `MemAvailable` minus an eighth of `MemTotal` (at least
  16 GiB) at bring-up; growth re-reads `MemAvailable` every 16 chunks and stops for good once the
  reserve would be touched (`kv store: host tier frozen at the host reserve`). `0` = device only.
  Pinned pages cannot be reclaimed, and a 120 GB tier beside two co-tenant compiler builds was
  OOM-killed on the 249 GB lab box (2026-10-22); the reserve is not a tuning parameter.
* Nothing is pinned at bring-up. A grower thread (`kv-store-host-grower`) allocates chunks ahead,
  up to 256 MiB of spare per class, outside the pool lock: pinning takes the driver context lock,
  and growing under the lock while serving stalled every engine launch for seconds in the lab. A
  spill that finds no spare fails and the pool frees the boundary instead.
* `Where::{Device, Spilling { dev, host, fence }, Host}`. A spill is one async D2H copy per block
  into a host slot of the block's class; the device slot is kept until the fence retires
  (`reap_spills` on every trim; the pressure loop credits in-flight spill bytes and never waits).
  A fence that cannot be queried leaves the block `Spilling` for good: both copies kept, neither
  reused.
* A host-resident block is read in place: its pinned slot is mapped into the device address
  space and the gather kernel reads it over PCIe (`host_reads`); no restore copy and no device
  allocation. The boundary stays in the tree, `cached_rows` still counts it, admission still sees
  it; a later publish that needs the same rows re-fills them on the device.
* Shielded boundaries (a session inside its TTL, a shared end that served two attaches) are
  spilled only once nothing unshielded is left, and a full tier drops its least recently used
  unshielded block rather than a shielded one.
* Accounting: `plowrt_kv_store_*` reports residency per class and tier, `host_grows` and
  `host_grow_us`, `spills` and `spill_us`, `shrinks` and `shrink_us`, `host_reads`, dedup hits,
  fills, reclaims and refusals.

Invariants (`memory::kv_store` tests): one owner per block outside `Spilling`; a block with
transfer holds is never reclaimed or spilled; a `Spilling` block is never a victim and is never
spilled twice; a block's rows go only with the block.

The legacy 0.1.2 flag/tier policy (`Policy::Legacy`) is removed completely across all backends.
Every pool (including AMD's shared-prefix path) now runs the boundary-tree retention policy:
canonical parents, one LRU over unheld boundaries, ancestor refresh on attach, structural retire
when exactly one continuation extends a boundary. `PLOW_PREFIX_POLICY`, `PLOW_PREFIX_CACHE_OUTPUT`
and `PLOW_PREFIX_RETIRE_ATTACHED` were removed with the legacy paths (session pin, turn-end output
boundary, `retire_superseded`). The previous binary is the rollback.

`/metrics`: `plowrt_prefix_snapshots_{live,branch,share}`, `snapshots_{retired,touched}_total`,
`publishes_{prompt_end,shared_end,checkpoint,turn_end}_total`, attach kinds
`attach_{prompt_end,block,shared}_total` + `attach_misses_total` and the rows each kind attached
/ left to recompute `tokens_{attached,lost}_{prompt_end,block,shared}_total`,
`tokens_lost_miss_total`.

### KV block store (CUDA, always on, `docs/arch/26-kv-block-store.md`)

A boundary is a set of content-addressed blocks instead of one contiguous
snapshot: 32-row blocks of the sliding windows (and their fp8 ring scales), 32-row blocks of
every full-layer row with its fp8 scales, and for a recurrent model one state block per
boundary. Nothing is multi-mapped: an attach maps the slot's own blocks and gathers the rows. Each block is keyed by the chained hash of the
tokens up to its last row, so the boundaries of one conversation and conversations on one
system prompt share blocks; a turn's publish copies only its new rows and a retire frees only the
rows nothing else uses. The boundary tree above is unchanged: it still decides which boundary
lives, spills (the store's pinned host tier) or dies; the
store owns the bytes. Copies run as one `plow_kv_segcopy` launch per publish or attach.
The host tier is on by default and sized by the host-reserve rule: `MemAvailable` minus an
eighth of RAM (at least 16 GiB) at start, and it stops growing for good if `MemAvailable` would
fall under that reserve (`kv store: host tier frozen`); `PLOW_VMM_HOST_CACHE_MIB` names a size
instead, under the same rule.
`/metrics`: `plowrt_kv_store_*` (residency per tier, dedup hits, fills, reclaims, refusals,
spills, restores, publishes, gathers, segments, launches). Recurrent models (Qwen3.8 GDN) get
prefix caching only through the store: the prefill chunk is cut at the prompt-end boundary and
the boundary published there, state included. The contiguous snapshot copies of
`docs/arch/22-kv-cache.md` §7 are gone on CUDA: a packet the store refuses has no prefix cache.
