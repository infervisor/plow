# Multi-model support review + two-model verification — GH200 (2026-08-07)

Post-PR-#55 review of plowrt multi-model (S1 switching) support, verified live with
two models on GH200. Companion design notes cover
spec-decode/MTP + PP + disagg as instances of one abstraction.

## 1. What multi-model is today (decision of record)

S1 switching (`crates/plowrt/src/serve/manager.rs:1-31`): at most the VRAM-fitting
subset of registered models is RESIDENT; a request for a non-resident model
evicts LRU victims until the blob-derived planner says the target fits, then
loads. Co-residency is real (not swap-only): `load_initial` greedily loads in CLI
order while models fit. Per resident model: one dispatcher task + one engine
thread + own slot table + own `VmmKv`. No cross-model scheduler; ticks serialize
via the shared CUDA context + per-engine mutex. Routing = OpenAI `model` field =
manifest network name. PR #55 added: cross-model slab chunk pool (`pool_take/put/
bytes/trim` on `VmmOps`), KV physical-block reuse pool, bounded-drain preemption
(`PLOW_DRAIN_TIMEOUT_MS` → `FinishReason::Preempted`, queued jobs → 429),
speculative preload, pool-aware OOM retry in `create_block`.

## 2. Two-model verification (this session)

Pair: A = 12b-w8a8 (fp8, plan 16.5 GiB), B = 31b-bf16 (plan 66.8 GiB), both
ctx-8k b1 bundles, network names distinct (checkpoint hashes 707f0a / 842da3).
Segmented-prefill knobs global: `PLOW_PF_SEG_DIR + PURE=fp8 + FA512=all + GRAPH=1`
(fp8 classing is safe for the bf16 model — its GEMM segs fall to the flash object).

| Check | Result |
|---|---|
| Co-residency (serve, 86.0 GiB used / 11.2 free) | PASS — both fully resident |
| Planner accuracy | plan vs measured: 16.52/16.61, 66.77/66.87 GiB (~100 MiB, self-calibrating overhead 89-102 MiB) |
| Per-model routing, serial gates | PASS — distinct coherent answers per model |
| Concurrent mixed (4×A + 4×B) | PASS — all correct, +155 MiB KV growth, zero errors |
| `co_resident_pair_interleaves` (gpu test) | PASS — `last_switch` stays None |
| `s1_switch_cycle` 4k phase (budget-forced) | PASS — A→B 4.07 s (load 4.0 s), B→A **0.51 s** (chunk-pool reuse), token identity after switch-back, VRAM within plan bounds every step |
| `s1_switch_cycle` 32k phase | NOT RUN — bundles compiled ctx 8192 (test wants 32k outgoing KV; needs ≥32k-ctx bundle pair) |
| `s1_switch_bench` (example) | ALL PHASES OK — preempt drill: 250 ms deadline → drain 258 ms, stream got 204 tokens + `Some(Preempted)`, switch 4.28 s |
| `long_prefill_does_not_shed_decode_streams` | NOT RUN — needs a batch-4 bundle (`PLOW_MM_ASSETS_C`); none on this box |

31b weight upload: 57.2 GiB at 15.1 GiB/s (page-cache warm; 12b re-load hit
51.8 GiB/s). Cold switch cost is load-dominated; drain/unload are ms-class.

## 3. Findings (review)

Correctness / latent bugs:
1. **Engine lookup by `bundle.network()`, install by slug** (`mux.rs:305,377,963`
   vs `manager.rs:569`). Identical only because CLI passes `slug=None`. Any slug
   override silently degrades to the CPU reference path.
2. **Duplicate network names silently drop a bundle** (`registry.rs:28` plain
   insert). Live on this box: all 12b bundles share network name 707f0a… — a
   12b-fp8 + 12b-bf16 pair is unservable, second dir unreachable, no warning.
3. **Slab pool never trimmed outside a switch**: `pool_trim` only in
   `ensure_resident`/`maybe_preload`. Steady-state KV growth OOMs against VRAM
   parked in the pool (`create_block`'s retry evicts prefix cache but never asks
   the backend to release pooled chunks).
4. **`set_slab_keep_default(true)` is a process-global latch** (`vmm.rs:1143`) —
   never cleared; a transient 2-model probe manager flips retention on for the
   process lifetime.
5. **Residency race**: `ensure_resident` releases the switch lock before
   `state.mux()` — a competing switch in the window yields a misleading 404
   ("no model registered") for a registered-but-just-evicted model.
6. `PLOW_TTFT_LOG` mixes models (process-global counters, valid at concurrency 1
   only, `obs/ttft.rs:6-16`).

Multi-model design gaps (feed the design doc):
7. **Segmented-prefill knobs are process-global env** (`PLOW_PF_SEG_*` read at
   engine load, `exec/gpu.rs:3769`) but classing must match each blob's emit-side
   classing — two models with different emit classing cannot both be served
   optimally. Promote to the asset manifest (the code comment already says so).
8. No cross-model scheduler: no fairness/priority between resident peers; a
   long prefill on A delays B's ticks only via context serialization (unmeasured).
9. LRU thrash has no hysteresis; victim KV destroyed (re-prefill on return);
   KV arenas compiled into the blob, not resizable at admission.
10. `maybe_preload` holds the switch lock across a whole background load
    (third-model requests queue behind it; `PLOW_PRELOAD=0` escape).
11. Preempt latency bounded by one tick (flag read at loop top only) — a 32k
    prefill tick runs to completion first.
12. AMD path entirely outside the manager (one model, no planning).

Stale docs: `manager.rs:31` and `perf-data/m1-multimodel-sm120.md` ("S1 does not
preempt") predate PR #55's preemption. Undocumented knobs: `PLOW_SLAB_KEEP`,
`PLOW_KV_POOL_MIB`, `PLOW_DRAIN_TIMEOUT_MS`, `PLOW_PRELOAD`, `PLOW_WEIGHT_VMM`
missing from `docs/flags-reference.md`.

## 4. Verdict

S1 multi-model is solid for the serve-N-models use case: planner accurate to
~100 MiB, co-residency + switching + preemption all behave as designed, chunk
pool delivers ~8× faster switch-back. The gaps that matter for what comes next
(spec decode = coupled co-resident pair) are #2 (name collisions — draft pairs
will hit it), #3 (pool starves KV under sustained decode), #7 (per-model prefill
knobs), and #8 (no cross-model scheduling policy) — all addressed in the
multi-instance design.
