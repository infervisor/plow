# Multi-model S1 switching — chunk pool, KV block pool, preemptive drain (GH200)

Date: 2026-08-07. Box: GH200 480GB (97 871 MiB), SHARED — a co-tenant vLLM held
~62 GiB throughout, so every number below is delta-based, measured by
`examples/s1_switch_bench.rs` (A = gemma-4-12b bf16, 24.3 GiB required;
B = gemma-4-31b fp8, 33.8 GiB required; budget-capped to force single
residency). Branch `worktree-multi-model-mem-opt` on top of main +
`worktree-vmm-weight-slab`.

## Headline

| switch | before (no pool) | after |
|---|---|---|
| A→B (31.5 GiB fp8 in) | ~6.4 s first/cold | **0.93 s** |
| B→A (24 GiB bf16 in)  | ~1.9 s (weight commit + upload) | **0.40 s** |
| drain of a live 512-token generation | O(max_tokens) ≈ 13 s+ | **258 ms** (250 ms deadline, `finish_reason:"preempted"`, 161 tokens delivered) |

Token identity holds across the A→B→A cycle (greedy ids byte-equal).

## What shipped (plan review: `cuMemAlloc` multi-model plan)

* **A — cross-model chunk pool, now default-on for multi-model serve.** The
  slab-branch pool (`VmmOps::pool_take/put`, exact-256 MiB-chunk match) already
  did cross-model reuse; what was missing was planner awareness. The manager
  now credits `pool_bytes()` in its fit check and `pool_trim()`s to the
  target's slab-consumable bytes **inside** the eviction loop — trimming after
  the loop stranded uncreditable pool VRAM and shed a fitting switch by 54 MiB
  (30 GiB victim pool vs a 24 GiB target cap left 5.7 GiB dead). Overhead
  calibration counts both ledgers (`free + pool`) so reused chunks don't make
  a model look smaller. `ModelManager::new` flips the process default on when
  it manages >1 model; `PLOW_SLAB_KEEP=1/0` still force-overrides; single-model
  paths and the lifecycle tests keep release-on-drop.
* **B — KV physical-block reuse pool** (`VmmKv::enable_block_pool`, engines
  arm it off `PLOW_KV_POOL_MIB`, default 512, 0 disables): zero-ref blocks park
  (bounded) instead of `cuMemRelease`; `ensure_rows` draws pool-first so the
  request path pays map+set_access (µs) instead of `cuMemCreate`'s serial
  ~13 GiB/s commit; a `vmm-kv-pool` thread pre-creates ~two window columns at
  load so the first request skips the commit too. Backend-agnostic — AMD gets
  it through the same `VmmKv`.
* **C — preemptive drain** (`ModelMux::preempt`, `FinishReason::Preempted`):
  closes every live stream with the tokens generated so far + honest usage,
  freeing slots on the client-disconnect teardown path. The dispatcher checks
  an atomic at every loop top (a full slot table never reaches the message
  channel between ticks). Manager wiring: `PLOW_DRAIN_TIMEOUT_MS` — unset =
  legacy unbounded graceful drain, N = grace window then preempt, 0 = preempt
  now.
* **D — speculative preload**: after each switch, if the hottest non-resident
  model fits free VRAM + usable pool with NO eviction, load it in the
  background under the switch lock (arrivals coalesce). `PLOW_PRELOAD=0`
  disables. Narrow by design: `load_initial` already greedy-loads co-fitting
  models, so this only re-warms after eviction cycles.
* **AMD**: `HsaBackend` gained the slab pool (`pool_take/put/bytes/trim`,
  released at backend drop) — `PLOW_SLAB_KEEP` now works on ROCr; the KV pool
  rides `VmmKv` unchanged. The S1 manager itself remains CUDA-only (AMD serve
  is one model for the life of the process, by decision of record).

## Skipped, with reasons

* **F — tables-slab consolidation**: measured 2026-08-06 at ~4 ms total for
  decode/prefill table allocs (`vmm-weight-slab-gh200.md`), not the plan's
  claimed 50-100 ms. Noise; not worth the carve-out complexity.
* **E — cross-model KV block pool**: the in-model pool (B) captures the
  steady-state win; cross-model reuse only matches same-geometry models and
  would share one backend pool with the 256 MiB slab chunks. Revisit if a
  same-arch switch fleet materializes.

## Validation

* `cargo test -p plowrt --features cuda --lib`: 152/152 (includes 2 new
  KV-pool tests: recycle+precreate, cap enforcement).
* `gpu_lifecycle` (12b-bf16): VRAM to baseline both cycles, replies "Paris".
* `reload_bench` 12b-bf16 with `PLOW_SLAB_KEEP=1`: load 1.94 s → reload
  0.38 s, token-identical (unchanged from the slab branch — no regression).
* `s1_switch_bench` (new): all phases above.
* `gpu_vmm_prefix`: dedup-ledger helper now counts created+reused (the ledger
  reasons about block *acquisitions*; the pool is invisible to it). The two
  slot>0 tests need the batch-4 bundle (`/root/gpu-assets-b4`) absent on this
  box — they fail `begin_slot` on batch-1 assets with or without this work.
* `--features hsa` compiles clean; HSA behavior untested here (no AMD card on
  this box) — the gfx950 box should run `hsa_vmm` + a serve smoke.
