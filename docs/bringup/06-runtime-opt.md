# Stage 6 — Runtime optimization

Serve the whole model on `plowrt` so TTFT and TPOT stay within budget at the target concurrency
and weights + KV + scratch fit device memory.

**Precondition:** Stage 5 gated out (block correct and at its latency target) and `plowc` emits a
full-model packet that loads. The [`target.md`](target.md) block is filled in; this stage uses
`$VENDOR`, `$GPU`, `$NGPU`, `$PARALLEL`, `$MAXCTX`, `$FEATURES`, `$BW_BOUND`. Nothing here
recompiles the model except the decode width/ladder, which sends you back to Stage 5.

**Gate out (into Stage 7):** target concurrency served with TTFT/TPOT in budget, memory fits, no
spurious shedding, every numerics-changing lever re-verified, and the settings recorded in the
model's `recipes/infervisor/<model>/<label>.toml`.

Kernel, warp, register, attention-split and packet-rung experiments do not run at whole-model
scope. Return them to [Stage 5](05-single-block-sweep.md) / [Stage 8](08-rung-campaign.md).
Whole-model serving is only for what a block cannot represent: admission, cross-request prefill
batching, decode/prefill overlap, KV capacity, queueing and tail latency
([agent-tools.md](agent-tools.md) "HARD RULE").

## The runtime

```
HTTP / UDS  →  per-model mux (continuous batching)  →  device engine(s)
                slot table · admission · hold          persistent / segmented interpreter
```

* The mux (`serve/mux.rs`) holds a fixed slot table sized to the compiled decode width. Each tick
  it admits arrivals into idle slots, runs one batched decode launch over every live slot, and
  frees finished slots.
* The engine is `serve::engine::ServeEngine`:

| variant | feature | shape |
|---|---|---|
| `Cuda` (`exec::gpu::GpuEngine`) | `cuda` | slotted: B sequence slots, chunked prefill, prefix sharing, device sampling, VMM paging, S1 multi-model residency |
| `CudaGroup` (`serve/cuda_group.rs`) | `cuda` | tensor-parallel CUDA packet over a process rank group (rank 0 local, slotted) |
| `Amd` (`exec::amd`, `AmdServe`) | `hsa` | fixed-capacity sequence slots with a compiled decode ladder, per-slot prefix snapshots, greedy on-device sampling, optionally TP |
| `Cpu` (`serve/cpu_serve.rs`) | `cpu` | CPU engine, persistent pinned workers |

* One serve may hold several models and `--dp` ranks, each placed on its own device group
  (`serve::placement`), all resident. VMM paging and S1 residency switching are CUDA-only.
* Memory is one per-device arena with compiler-resolved offsets (`memory::AddressSpace`); KV is
  carved by `memory::streamer::KvArena` over per-head `memory::pool::GrowablePool`s.
  `memory::kv::{BlockAllocator, PageTable}` is not on the serving path.

A lever listed for one engine does not exist on the other; it is not merely untuned.

Reading: `docs/arch/06-runtime.md` (execution model), `docs/arch/09-multi-gpu.md` (TP),
`docs/arch/30-heterogeneous-serving.md` (DP/PD), `docs/arch/13-prefill-chunking.md`. Every flag:
`docs/flags-reference.md`; every knob: `crates/plowrt/src/config.rs`.

## Performance authority

`plowrt bench` is the performance authority. It drives the same `ServeEngine` and mux as `serve`,
excludes load and warmup from the timed interval, and fails without JSON on partial output,
shedding, TP rank disagreement or CPU fallback.

* `plowrt amd-bench` drives `AmdEngine` directly and bypasses production scheduling. Use it only
  for dump comparisons and diagnosing a TP disagreement rank by rank; never report it as
  served-model performance.
* `plowrt amd-probe` runs an unbound zero-weight packet. Its timings are synthetic diagnostics.
* Every GPU process goes through the queue ([agent-tools.md](agent-tools.md) "Queue").

## Levers

Effects below are directions, not targets. Re-measure every size on `$GPU`.

### 1. Concurrency and decode width — both

The decode width and ladder are compiled into the packet (`PLOW_DECODE_BATCH`,
`PLOW_DECODE_BATCH_LADDER`, emit knobs); mux slot *i* is engine slot *i*.

* Scheduling is set by `PLOW_OBJECTIVE`, default `auto`; multi-step depth and device sampling are
  derived from it. Recipes keep `auto` ([agent-tools.md](agent-tools.md)
  "Per-rung gate and recipe selection").
* Decode is bandwidth-bound against `$BW_BOUND`: a wider width raises per-token latency. Pick the
  width from the TPOT budget, not peak throughput. A fixed wide width overpays at partial load.
  For MoE the batching win is weaker: the expert union saturates weight reads at modest width.
* Serve width must match the compiled width; a mismatch runs a predicated remainder arm, slower.

### 2. KV cache and memory fit — both

| lever | flag / env | effect |
|---|---|---|
| prefill chunk / KV ring | `PLOW_MAX_CHUNK` (emit) | ring = `next_pow2(window + chunk − 1)`; a smaller chunk is a fit enabler, not a speed win |
| KV reuse pool | `--kv-pool-mib` / `PLOW_KV_POOL_MIB` | zero-ref KV blocks park instead of being freed; `0` disables |
| AMD VMM KV | `--amd-vmm-kv` / `PLOW_VMM_KV` | automatic when a flat slab cannot fit; explicit value forces the route |

Compute the budget before serving. Terms come from the Stage 1 geometry audit, `$MAXCTX`, the
decode width `B` and the chunk:

```
ring            = min(ctx, next_pow2(window + chunk − 1))
sliding_per_seq = ring × n_slide × kvh_slide × hd_slide × 2 (k,v) × elt
full_per_seq    = ctx  × n_full  × kvh_full  × hd_full  × 2 (k,v) × elt
total           = weights + B × (sliding_per_seq + full_per_seq) + activations
```

* The ring is set by the chunk, with a hard floor at `next_pow2(window)`.
* KV capacity is a `$GPU` property: two parts at one `$ISA` emit identical code and fit different
  concurrencies.
* KV OOM is a typed shed (`admit_into`), not a crash. Shrink the chunk or widen the budget; never
  grow the slot table.

### 3. Prefix caching — always on

| lever | flag / env | scope |
|---|---|---|
| prefix cache (master switch) | `--prefix-cache` / `PLOW_PREFIX_CACHE` (default on) | every route; same-slot snapshots for recurrent/linear-attention models |
| VMM prefix sharing | `--vmm-prefix` / `PLOW_VMM_PREFIX` | shared full-attention KV mapped into every sharer; recipes set `=1` where the packet supports it |
| VMM block size | `--vmm-block-mib` | the real tuning knob for long-context dedup |

* Never disable the prefix cache as a tuning lever. A prefix-cache cost is a plowrt bug to fix;
  `=0` is only an attribution arm.
* Report the hit rate with every prefix-cache number. Bench against a reference with its prefix
  cache on too, with unique prompts per cell (`llm_grid.sh` does this).
* `PrefixCache::lookup`/`insert` compare stored tokens per matched block; a mismatch ends the
  match and counts in `collisions()`.

### 4. Prefill: chunking and interleave — both

Prefill is a launch-count trade: fewer launches are faster but need a bigger KV ring. Measure the
per-launch fixed cost on `$GPU` before choosing a chunk.

| lever | flag / env | effect |
|---|---|---|
| interleave quantum | `--pf-interleave` / `PLOW_PF_INTERLEAVE` | prefill rows per tick while requests decode; unset or 0 = widest compiled launch |
| per-request chunk cap | `--pf-chunk` / `PLOW_PF_CHUNK` | tail-latency tool; usually loses throughput. AMD also rotates isolated admission under it |
| defer decode | `--pf-defer-decode` / `PLOW_PF_DEFER_DECODE` | prefill to completion before decode; trades streaming latency for tok/s; not a default |
| ragged tail | `--amd-ragged-chunk` / `PLOW_RAGGED_CHUNK` (default on) | AMD: last chunk at its real row count. CUDA: MoE ragged tail only |
| segmented prefill | `--pf-seg-*` (nvidia), `--amd-seg-window` (amd) | same-occupancy launch sequence; `--pf-seg-*` are A/B diagnostics, emit classing must match the packet |

* Cross-request co-packed prefill is not numerics-neutral (flash split count is chunk-dependent).
  Gate it before relying on it.
* AMD prefill/decode overlap is not supported: programs share one mutable tensor table. Keep the
  sequential drain.

### 5. Admission — both

| lever | flag | effect |
|---|---|---|
| admission SLO | `--slo-ms` | floor; effective SLO = `max(slo_ms, 8 × service_ms)` |
| cold-start hold | `--max-hold-ms` | batch-formation hold, only when the slot table is empty |
| trace | `--trace` | per-packet timeline at `GET /trace` |

* A shed looks like an emitter bug. Check `plowrt_admit_shed_total` (controller drops live work)
  vs `plowrt_rejected_total` (full table or oversize prompt) before touching the packet.
* A throughput client counts shed requests as successful with few tokens. Raise `--slo-ms` for
  throughput runs and check total generated tokens.

### 6. Load and cold start — both

| lever | flag / env | effect |
|---|---|---|
| weight slab | `--rt-weight-slab` / `PLOW_WEIGHT_SLAB`; `--rt-slab-keep` / `PLOW_SLAB_KEEP` | one allocation for all weights |
| VMM lazy-commit slab | `--weight-vmm` / `PLOW_WEIGHT_VMM`; `--nv-upload-direct` / `PLOW_UPLOAD_DIRECT` | default on for CUDA, off for AMD (regressed there) |
| prefetch | `--rt-prefetch` / `PLOW_PREFETCH`, `--rt-prefetch-threads` / `PLOW_PREFETCH_THREADS` | parallel upload; reader count saturates on host storage, re-fit per host |

### 7. Multi-GPU — TP, DP, PD

* **TP** is the only emit-time mode: `plowc --parallel tp` (`dp`/`pp`/`ep` are not implemented at
  emit). `$NGPU` is the rank count. Host side: `exec::tp`, `exec::amd_tp::AmdTpGroup`,
  `serve/cuda_group.rs`.
* **DP** is a serve-time mode: `plowrt serve --dp N|all|model=N` loads N full copies, each on its
  own device group with its own prefix cache; requests route by session, prefix and load
  (`--route-prefix`, `--route-spill`, `--route-prefix-slack`, `--dp-numa-pin`; `serve/dp.rs`).
* **PD** splits a `--dp` model's ranks into prefill and decode pools: `--pd model=0/1`. The first
  token is sampled on the prefill rank; KV moves through host memory (`serve/pd.rs`).
* EP exists only as a MoE weight-base remap (`orch::moe`); `orch::disagg` is not the serving PD
  path.

TP bring-up without a full serve: `plowrt devices --tp $NGPU` (peer mapping, all-pairs
visibility), then gate performance and rank agreement with `plowrt bench` on the TP assets. Every
rank must emit the identical token stream; a rank whose collective timed out still samples fluent
ids from its shard. TP knobs: `--amd-tp-agree-every`, `--amd-tp-no-audit` (timing only),
`--amd-share-ckpt`.

### 8. Multi-model residency — CUDA only

`serve::manager::ModelManager` keeps the largest VRAM-fitting subset resident and evicts LRU on a
miss. Knobs: `--drain-timeout-ms` / `PLOW_DRAIN_TIMEOUT_MS`, `--preload` / `PLOW_PRELOAD`,
`--vram-budget-mib` / `PLOW_VRAM_BUDGET_MIB`. AMD serves its models and ranks all resident, with
no switching.

Not implemented: multi-tenancy partitions, a host watchdog, `Streamer::execute_reclaim` (no-op).

## Steps

Fix the target first: model, prompt length, target concurrency, TTFT budget, TPOT budget, all for
`$GPU`. Build `plowrt` with `nix develop --command cargo build -p plowrt --release $FEATURES`.
Assets are the Stage 5 emit for `$GPU`/`$ISA` in `$ASSETS`. Every device step runs under the
queue.

1. **Schedule sanity (CPU).** `plowrt simulate --assets $ASSETS --all-buckets --chrome sim.json`.
   No deadlock, every dependency honored. `--bucket decode:1:128` isolates one bucket;
   `--math golden` runs reference numerics.
2. **Engine floor.**
   `plowrt bench --assets $ASSETS --prompt-ids 1,2,3,4 --concurrency 1 --requests 8 --warmup-requests 1 --output-len 64`,
   then at the target concurrency. The JSON records TTFT/TPOT/ITL/E2E, throughput, rungs, TP
   width, runtime settings and packet/object checksums. `--engine-diagnostics` adds AMD prefill
   chunk boundaries, decode rung choices and TP agreement policy. `--trace-raw PATH` writes rank
   0's last decode packet trace. AMD snapshots: `--amd-ctr-snap DIR`, `--amd-tens-snap DIR`
   with `--amd-snap-tensors a,b` and `--amd-snap-slot N` (fail-closed, never overwrite).
3. **Cold prefill sweep.** `plowrt bench --assets $ASSETS --prefill-sweep --prefill-lengths
   512,1024,2048,4096,8192 --prefill-warmups 1 --prefill-reps 3` (every length ≤ `$MAXCTX`;
   `--prompt-ids` for an exact row). It refuses to start with a prefix cache enabled: run it with
   `PLOW_PREFIX_CACHE=0` as a cold-prefill measurement only.
4. **Serve and baseline.** Start through `campaign.py serve <recipe> --assets $ASSETS` and drive
   the target load. Scrape `/metrics`:

   | series | read |
   |---|---|
   | `rate(plowrt_batch_size_sum[5m]) / rate(plowrt_batch_count_total[5m])` | mean batch actually run |
   | `plowrt_admit_shed_total` vs `plowrt_rejected_total` | controller shed vs capacity |
   | `plowrt_utilization`, `plowrt_arrival_rate` | ρ = λ/μ, and λ |

5. **Fit memory.** On `kv arena OOM`, shrink the chunk/ring; confirm with `PLOW_LOAD_PROFILE=1`.
6. **Levers, one at a time**, re-measuring after each: decode width from the TPOT budget →
   chunking/interleave only for TTFT-under-load tails → VMM prefix block size → TP/DP/PD when one
   `$GPU` cannot hold the model or the budget → load knobs only if time-to-serving matters.
7. **Record** every kept setting in the recipe TOML; experimental settings stay in
   `scripts/campaign/recipes/` ([agent-tools.md](agent-tools.md)
   "Per-rung gate and recipe selection").

## Attribution ladder

Cheapest first; drill down only when the current rung is flat. Where `$VENDOR` has no instrument,
skip the rung and say it went unattributed.

| # | question | instrument | vendor |
|---|---|---|---|
| 1 | client or server? | `PLOW_TTFT_LOG=1` — per-request TTFT decomposition; a large unaccounted share means instrument first | both |
| 2 | which class of work? | `PLOW_PF_SEG_TIME=1` — per-class segment totals; perturbs, read shares | nvidia |
| 3 | which op? | object built with `-DPLOW_NV_TRACE=1`, serve with `PLOW_PF_TRACE_LOG=1`; block 0 only, read shares | nvidia |
| 4 | host or device? | `PLOW_STEP_TIME=1` → `gap_us` vs `dev_interp_ms`: host-bound → levers; kernel-bound → Stage 4/5 | nvidia |
| 5 | which stall in the kernel? | vendor profiler (`ncu`, `rocprof`) under the queue | both |
| 6 | is it the silicon? | sample clocks/power/throttle during the bench | both |

Other diagnostics: `PLOW_DSTEP_LOG=1` (per-step host phases), `PLOW_PFX_LOG=1` (prefix cache),
`PLOW_PF_PACKLOG=1` (prefill-vs-decode split per tick), `plowrt disasm $ASSETS --counters
--kernargs --tensors` (static, no device). Hazardous leftovers: [agent-tools.md](agent-tools.md)
"Measurement hazards".

## Gate

1. TTFT in budget at c1 and not blown past at the target concurrency.
2. TPOT in budget at the target concurrency, and as concurrency rises to it.
3. Target concurrency admitted, not KV-OOM shed; `PLOW_LOAD_PROFILE=1` shows headroom.
4. `plowrt_admit_shed_total` ~0 at target load; any 429 is genuine capacity.
5. Every numerics-changing lever (co-packed prefill, ragged tail, prefix sharing) re-verified by
   the recipe's gates ([agent-tools.md](agent-tools.md) "Gates"); TP rank token-identity holds.
6. Settings recorded in the recipe TOML, measured on `$GPU`; nothing carried over from another
   part as already tuned.

## Pitfalls

* An asset built for another part still loads and runs: `--arch` is packet metadata. Check
  `build.json` names `$GPU`/`$ISA` before trusting a number.
* Device init failure falls back to the CPU reference interpreter with a warning and correct text.
  Confirm `backend ready — GPU accelerated` in every serve log.
* Multi-model (CUDA): duplicate network names drop a bundle; `PLOW_PF_SEG_*` are process-global,
  so two models with different emit classing cannot both be optimal.
* On variable-completion traffic, tok/s is confounded by stop position; prefer requests/s.

## Stop and ask

* A target-block row cannot be filled, or `$GPU` is not in `plowc --list-gpus`.
* Full-model emit aborts or `simulate` deadlocks — a Stage 1–5 defect.
* KV cannot fit the target concurrency at any sane chunk — a width/TP/DP/part decision.
* The plan needs an unimplemented mode or a lever that exists only on the other engine.
* A numerics-changing lever helps but no gate can confirm it.
* TP ranks disagree — a correctness blocker.

## Report

* Filled target block; engine, compiled width/ladder, `$NGPU`, prompt length, concurrency.
* Baseline vs tuned TTFT/TPOT/throughput, and which lever moved which number.
* Memory budget (weights / KV / scratch) and the concurrency that fits; prefix hit rate.
* Correctness re-verification per numerics-changing lever.
* Gate decision and the recipe TOML path.

## Code pointers (`crates/plowrt/src/`)

| path | role |
|---|---|
| `serve/mux.rs` — `admit_into`, `Slot`, `MuxConfig` | continuous-batching mux, admission |
| `serve/engine.rs` — `ServeEngine` | engine seam |
| `serve/dp.rs`, `serve/pd.rs`, `serve/placement.rs` | DP routing, PD split, device placement |
| `serve/bench.rs` | `plowrt bench` |
| `serve/policy.rs` | objective → scheduling mechanisms |
| `sched/admission.rs`, `sched/batching.rs` (`formation_window_ms`) | admission, hold |
| `memory/streamer.rs` (`KvArena`), `memory/pool.rs`, `memory/prefix.rs`, `memory/vmm.rs` | KV, prefix cache, VMM |
| `exec/gpu`, `exec/amd`, `exec/amd_tp.rs`, `exec/tp.rs` | device engines, TP |
| `serve/manager.rs` | S1 residency (CUDA) |
| `config.rs` | every `PLOW_*` env var and flag |
