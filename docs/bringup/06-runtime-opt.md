# Stage 6 — Runtime Optimization

> Turn a correct model into a **server that holds up under load**. Serve the
> whole model on `plowrt` so that time-to-first-token (TTFT) and time-per-output-
> token (TPOT) stay within budget at a target concurrency, and the working set
> (weights + KV cache + activation scratch) fits device memory.

**Precondition:** Stage 5 complete — one block is numerically correct and at its
per-block latency target, and `plowc` emits a full-model `model.pkt` that loads.
The [`target.md`](target.md) block is filled in; this stage is written against
`$VENDOR`, `$GPU`, `$NGPU`, `$PARALLEL`, `$MAXCTX`, `$FEATURES` and `$BW_BOUND`,
and **which levers exist at all is a function of `$VENDOR`** — see the table
below. Nothing in this stage recompiles the model; every lever is a `plowrt`
flag or a `PLOW_*` environment variable read at serve time. Choosing the decode
batch width `B` is the one lever that may send you *back* to Stage 5 to
recompile.

**Gate out (into Stage 7, e2e campaign):** the model serves at the target
concurrency with TTFT and TPOT within budget, memory fits (target concurrency is
admitted, not shed on KV OOM), no spurious shedding, and any lever that can
change numerics was re-verified against a correctness gate. The output is a
**serving recipe** — a `plowrt serve` command line + `PLOW_*` knobs + measured
TTFT/TPOT/throughput/memory numbers.

Kernel, warp, register, attention-split, and packet-rung experiments do not run
at whole-model scope here. Return them to Stage 5 and rank them with the existing
single-block or minimal mixed-block harness. Use whole-model serving only for
effects a block cannot represent: admission, cross-request prefill batching,
decode/prefill overlap, KV capacity, queueing, and tail latency.

---

## The runtime, in one picture

```
HTTP / UDS  →  per-model mux (continuous batching)  →  device engine
                slot table · admission · hold          persistent/segmented interpreter
```

* The **mux** (`crates/plowrt/src/serve/mux.rs`) holds a fixed-size **slot
  table** sized to the engine's compiled decode batch. Each tick it admits new
  arrivals into idle slots, runs one batched decode launch that advances every
  live slot by one token (or K tokens under multi-step), and frees finished
  slots — so a fresh request does not wait for prior requests to reach
  `max_tokens` before its first token.
* The **engine** is one of two shapes (`serve/engine.rs`, `enum ServeEngine`):

| `$VENDOR` | module | shape |
|---|---|---|
| nvidia | `exec::gpu::GpuEngine` | **slotted**: B independent sequence slots, chunked prefill, prefix sharing, device sampling. Most runtime levers live here. |
| amd | `exec::amd` (`enum Ranks { One, Tp }`) | fixed-capacity independent sequence slots with a compiled decode ladder and greedy on-device sampling, optionally tensor-parallel. One model per process, no paging, no residency manager. |

The seam is per **vendor**, not per ISA: every NVIDIA `$ISA` reaches
`GpuEngine` and every AMD `$ISA` reaches `exec::amd`. A lever marked for one
vendor below does not exist on the other — it is not merely untuned there.

* Memory is a single per-device **arena** with compiler-resolved offsets
  (`memory::AddressSpace`); KV is carved from it by `memory::streamer::KvArena`
  over per-head `memory::pool::GrowablePool`s.

Read `docs/arch/06-runtime.md` for the execution model (persistent kernel,
segmented dispatch, counter DAG, arena, KV paging), `docs/arch/09-multi-gpu.md`
for TP, `docs/arch/13-prefill-chunking.md` for how a prompt is a sum of compiled
rungs.

---

## The levers

Grouped by what they move. **Platform** is stated as `$VENDOR`: nvidia, amd, or
both. Defaults are the in-tree defaults from `config.rs` / `main.rs` — the full
catalogue of every flag is [`docs/flags-reference.md`](../flags-reference.md);
this section is only the serving-relevant subset and its measured evidence.
Where a lever is design-intent but **not wired**, it says so.

Every measured figure quoted below was taken on the part named with it. None of
them is a target for `$GPU`; they say a lever *has* an effect and roughly where
it comes from. Re-measure the size on `$GPU`.

### 1. Concurrency and batch width — `$VENDOR`: both

The decode batch `B` is **compiled into the blob** (`PLOW_DECODE_BATCH` at
compile time); the mux sizes its slot table to exactly `B` (mux slot *i* IS
engine slot *i*). The first concurrency decision was therefore made in Stage 5.

| lever | flag / env | default | effect |
|---|---|---|---|
| CPU executor threads | `--executors N` | 8 | reference backend only; GPU engines run one dedicated engine thread per model |
| Multi-step decode | `PLOW_MULTISTEP=K` | 8 | device advances K greedy tokens per host round-trip. NVIDIA supports up to 64. AMD TP uses a bounded K≤4 device token ring only when `PLOW_TP_AGREE_EVERY>1`; counter drain/audit still runs every token. Stochastic rows fall back to per-token. |
| Device sampling | `PLOW_DEV_SAMPLE=1` | on (nvidia) | argmax/sample on device, no logits round-trip; pairs with multi-step *(nvidia)* |

> **Pitfall — a wide fixed `B` is not a free win.** Decode is HBM-bandwidth-
> bound against `$BW_BOUND`: doubling `B` ~doubles per-token latency, on any
> part. Where the crossover lands is per-part — on a 12B-class model on RTX
> 5090, `B=16` crossed a 50 ms inter-token SLO at ~2 concurrent users where
> `B=8` held to ~4 (`perf-data/serving-capacity-report.md`,
> `perf-data/b2-concurrency-family.md`). Derive your own crossover on `$GPU`;
> pick `B` from the **TPOT budget**, not the peak-throughput number. A fixed
> large `B` also overpays at partial load — a `B=32` blob costs full width at 16
> live slots. For MoE the batching win is weaker still: the expert union
> saturates weight-read by modest B (measured 26B on H100,
> `perf-data/concurrent-decode-26b-h100.md`).

### 2. KV cache and memory fit — `$VENDOR`: both

The live allocator is `KvArena` (`memory::streamer`) over per-head
`GrowablePool`s. The vLLM-style paged `BlockAllocator`/`PageTable` in
`memory/kv.rs` is **retained but not on the serving path** (superseded by the
`GrowablePool` layout — design-intent for a future page-table-over-pool view).

| lever | flag / env | default | effect |
|---|---|---|---|
| Prefill chunk / KV ring size | (chunk knobs below) | — | ring = `next_pow2(window + chunk − 1)`; smaller chunk → smaller resident KV ring → more sequences fit. Measured 12B: KV ~10.7→3.0 GiB from 8192→1024 chunk for ~2% prefill cost — a **fit enabler, not a speed win** (`perf-data/gemma4-12b-sm120-serving.md`) |
| Per-engine KV reuse pool | `--kv-pool-mib` / `PLOW_KV_POOL_MIB` | 512 | zero-ref physical KV blocks park (bounded) instead of being freed; next request pays a map, not an allocation. 0 disables |
| VMM-backed KV | `--amd-vmm-kv` / `PLOW_VMM_KV` | off | driver-VMM physical backing (`memory::vmm::VmmKv`); block size `--vmm-block-mib` *(amd)* |

KV capacity is a `$GPU` property, not an `$ISA` one — two parts at the same
`$ISA` can differ in HBM capacity and therefore in which concurrencies fit
while emitting identical code.

**Compute the budget before you serve**, rather than discovering it as a startup
refusal or a shed at the target concurrency. Every term comes from the Stage 1
geometry audit except `ctx` (`$MAXCTX`), the decode width `B`, and the chunk:

```
ring            = min(ctx, next_pow2(window + chunk − 1))
sliding_per_seq = ring × n_slide × kvh_slide × hd_slide × 2 (k,v) × elt
full_per_seq    = ctx  × n_full  × kvh_full  × hd_full  × 2 (k,v) × elt
total           = weights + B × (sliding_per_seq + full_per_seq) + activations
```

**The ring is set by the prefill chunk, not by the model.** That is why
`PLOW_MAX_CHUNK` defaults to `next_pow2(window)` on a windowed model and stays
at 8192 when `window == 0`. There is a **hard floor at `next_pow2(window)`**: a
1024 window cannot ring fewer than 2048 rows however small the chunk, so below
that point shrinking the chunk buys launches for nothing. The size of the effect,
as an illustration and not a target: on a 12B-class windowed model at B=16, a
fixed 8192 chunk wanted 21.32 GiB of KV where the window-derived chunk wants
6.09 GiB — the difference between fitting and not fitting beside the weights on
a 32 GiB card.

> **KV OOM is a shed, not a crash.** `admit_into` fails a request whose KV does
> not fit with a typed OOM error rather than holding a slot open under pressure.
> If the target concurrency sheds on KV, shrink the chunk (smaller ring) or widen
> the VRAM budget — do not grow the slot table.

### 3. Prefix caching — `$VENDOR`: split

| lever | flag / env | default | effect |
|---|---|---|---|
| VMM prefix sharing | `--vmm-prefix` / `PLOW_VMM_PREFIX=1` | **off** | shared prefix's full-attention KV held once, `cuMemMap`'d into every sharer (`memory::prefix::PrefixCache`). Measured warm-TTFT 3.6×(4k)→23.8×(128k), dedup ~10 GiB/sharer at 31B/128k, TPOT-neutral (`perf-data/vmm-prefix-v1.md`). Block size is the real knob (`--vmm-block-mib`; 64 MiB @128k won on the measured part). Hit stats on `/metrics` via `VmmStatsHandle`. *(nvidia)* |
| Same-slot prefix cache | `--prefix-cache` / `PLOW_PREFIX_CACHE=1` | off | for recurrent/linear-attn models: MLA KV half is free, recurrent state checkpointed via one batched D2D copy. Measured ~24% lower per-query latency, ~1.9× better median TTFT at 75% hit (`perf-data/archive/k3/k3-prefix-cache-design.md`). *(amd, `$PARALLEL = tp` path)* |

> **Pitfall — report the hit rate with any prefix-cache number.** A cache that
> misses pays snapshot/insert cost with no benefit; at low hit rates the
> same-slot cache is a net *loss*. VMM prefix sharing is off by default pending
> its e2e integration — enable it deliberately and measure. Open safety gap: the
> VMM `BlockHash` collision check is **not implemented** (`memory/vmm.rs`), so a
> hash collision would attach the wrong prefix's KV. Measured-good, not hardened.

### 4. Prefill: chunking, batching, interleave — `$VENDOR`: both

Prefill is a launch-count game: each launch has a fixed cost, so fewer launches
is faster — at the cost of a bigger KV ring. **Measure that fixed cost on
`$GPU`** before choosing a chunk size; it is a per-part constant and it sets
the whole trade. Two recorded fits, as illustrations of the shape and not as
values to reuse: ~32 ms/launch on one NVIDIA part, and `139 ms + 0.943 ms/row`
on an AMD TP4 configuration (`perf-data/glm52-ttft-breakdown.md`).

| lever | flag / env | default | effect |
|---|---|---|---|
| Chunk-interleave quantum | `--pf-interleave N` / `PLOW_PF_INTERLEAVE` | 2048 | rows admitted per tick before decode runs; caps how long a decode stream stalls behind a new prefill. AMD TP re-splits a pending compiled step if the cap shrinks after decode becomes live *(both)* |
| Per-request chunk-row cap | `--pf-chunk N` / `PLOW_PF_CHUNK` | 0 (off) | finer chunking. **Tail-latency tool** — measured a net throughput *loss* at B=8 (`perf-data/serving-capacity-report.md`) *(both; AMD TP uses its compiled ladder)* |
| Disable chunking / interleave | `--pf-no-chunk`, `--pf-no-interleave` | off | A/B and pure-throughput controls *(both)* |
| Chunk cost model | `--pf-chunk-cost N` / `PLOW_PF_CHUNK_COST`, `--pf-cover` | 512 | fixed launch cost in padded-row equivalents — **re-fit this on `$GPU`**, the default is another part's number (`perf-data/chunk-cost-model.md`, `perf-data/rtx12-chunked-packing.md`) *(nvidia)* |
| Cross-request prefill scheduling | `--pf-batch` / `PLOW_PF_BATCH=1` | off | NVIDIA packs chunks into one launch. AMD rotates isolated one-request chunks fairly; true co-packing needs per-row state/KV packet fields *(both)* |
| Throughput mode | `--pf-defer-decode` / `PLOW_PF_DEFER_DECODE=1` | off | run prefill chains to completion before decode. Trades streaming latency for aggregate tok/s. Not a shippable default *(both)* |
| Ragged-tail chunk | `--amd-ragged-chunk` / `PLOW_RAGGED_CHUNK` | **on** | cover a prompt in fewest launches, run the last chunk at its real row count (measured −239 ms @4097 tok on one part; `exec::amd::rebase_chunk_rows`). `=0` restores the padding-vs-launch DP for a controlled A/B — see the flag's own doc in `config.rs` for the quality-gate caveat *(amd)* |
| Segmented prefill | `--pf-seg-*` (nvidia), `--amd-seg-window` (amd) | seg-window on | prefill as a sequence of same-occupancy launches (measured −11%/−12% at 8k/16k on one part — `docs/arch/06-runtime.md`, `perf-data/gemma12b-gh200-prefill-campaign.md`). `--pf-seg-*` are mostly A/B diagnostics; emit-side classing must match the blob |

AMD prefill/decode overlap is not currently safe. All programs share one mutable
`d_tens`; `act.*`, mutable `in.*`, and trace storage are shared, and `kv_rebase`
rewrites that same tensor table between phases. Keep the sequential drain.
Overlap requires phase-local scratch and tensor tables, phase-local KV rebasing
and staging/trace ownership, plus a physical-range non-aliasing proof and
concurrent parity tests.

> **Pitfall — cross-request co-packed prefill is not numerics-neutral.** Greedy
> tokens can differ across chunk boundaries (the flash split count is
> chunk-dependent; `perf-data/px14-batched-prefill-fp8.md`). If bit-identical
> decode matters, keep packing off or gate on a facts probe. AMD's fair mode
> does not co-pack rows and preserves per-request recurrent/KV isolation.
> AMD additionally requires `plow_packed_prefill_abi_1` in every object a
> packed program would route through before it stages non-null descriptors.
> The marker proves the `PlowProgram` ABI only; it does not enable co-packed
> dispatch or claim that KDA/MLA kernels consume the descriptors.

### 5. Admission: SLO, hold, shedding — `$VENDOR`: both

The mux admits, holds, and sheds via `sched::admission` and
`sched::batching::formation_window_ms`.

| lever | flag | default | effect |
|---|---|---|---|
| Admission SLO | `--slo-ms` | 250.0 | treated as a **floor**: effective SLO = `max(slo_ms, 8 × service_ms)`, scaling with the decode batch so a wide blob is not shed wholesale |
| Cold-start hold | `--max-hold-ms` | 8.0 | upper bound on the arrival-rate batch-formation hold; used only when the slot table is empty. With slots draining, the hot path never sleeps |
| Trace | `--trace` | off | per-packet timeline, dumpable at `GET /trace` |

> **Pitfall — the SLO shed masquerades as an emitter bug.** The old serial
> `predicted_wait = live × service_ms` 429'd *every live slot* once
> `live × step_ms` crossed a flat SLO — a B=16 blob shedding all streams at ~8
> users while the kernel was byte-identical and error-free
> (`perf-data/serving-capacity-report.md`, `perf-data/b2-concurrency-family.md`).
> Fixed by a `ceil(live/batch) × service_ms` predictor + the SLO floor. Two
> consequences: (1) a throughput benchmark that sheds counts the 429'd requests
> as *successful* with ~12 tokens each → fake tok/s; **raise `--slo-ms` when
> throughput-benching**. (2) If your served concurrency is 429'd, check
> `plowrt_admit_shed_total` vs `plowrt_rejected_total` before touching the blob —
> distinct signals (shed = controller dropping live work; rejected = a full slot
> table or an oversize prompt).

### 6. Weight load and cold start — `$VENDOR`: both (some nvidia-only)

Faster load is faster time-to-serving; it does not touch steady-state TPOT.
(The part names in the citations below are where those campaigns ran; some —
GH200 among them — are **not** `--gpu` registry entries. Take `$GPU` from
`plowc --list-gpus`, never from a write-up's filename.)

| lever | flag / env | default | effect |
|---|---|---|---|
| Weight slab | `--rt-weight-slab` / `PLOW_WEIGHT_SLAB`; `--rt-slab-keep` / `PLOW_SLAB_KEEP` | on / off | one allocation for all weights. Measured on MI355X: named-tensor alloc ~6.4–8.8 s → ~0.1–0.27 s, wall halved (`perf-data/weight-slab-amd-mi355x.md`) |
| VMM lazy-commit weight slab | `--nv-weight-vmm` / `PLOW_WEIGHT_VMM`; `--nv-upload-direct` / `PLOW_UPLOAD_DIRECT` | on (nvidia) / off (amd) | commit chunks behind the upload. Measured 12B on GH200: load 3.67→2.0 s (`perf-data/vmm-weight-slab-gh200.md`). **Default off on `$VENDOR = amd` — a measured regression there** |
| Prefetch | `--rt-prefetch` / `PLOW_PREFETCH`, `--rt-prefetch-threads` / `PLOW_PREFETCH_THREADS` | 256 / 16 | parallel weight upload. Measured 12B on GH200: cold start 17.55→11.19 s (`perf-data/coldstart-plow-vs-vllm-gh200.md`). Reader count saturates against the host storage, not `$GPU` — re-fit per host |

### 7. Tensor parallelism (TP) — `$VENDOR`: both; the ONLY wired multi-GPU mode

Serving selects TP via `enum Ranks { One, Tp }` (`serve/engine.rs`); the host
side is `exec::tp` (`TpGroup`, `TpRank`, `PeerLayout`) and, on AMD,
`exec::amd_tp::AmdTpGroup`. See `docs/arch/09-multi-gpu.md`.

`$PARALLEL` must be `tp` — the devblob path *errors* on `dp`/`pp`/`ep`, so a
non-TP value is a blocker at emit time, not a slower path. `$NGPU` is the rank
count.

Bring up and time a TP group **without** a full serve:

* `plowrt devices --tp $NGPU` — peer-mapped reduction regions, cross-GPU counter
  tables, all-pairs peer-visibility check (no model).
* `plowrt amd-bench --tp $NGPU` — runs the explicit cross-rank decode audit.
  **Every rank must emit an identical token stream** — that identity is the TP
  correctness gate, not a sanity check (a rank whose collective silently timed
  out still samples fluent ids from its own shard).

TP knobs: `--amd-tp-agree-every N` (agreement interval; serving samples, the
oracle checks every step), `--amd-tp-no-audit` (timing runs), `--amd-share-ckpt`
(shared checkpoint mapping), `--amd-tp-serial-load`.

> **Not wired — say so honestly.** There is **no** unified
> `Parallel { Tp, Dp, Pp, Ep }` selector. Data-parallel (**DP**) and cross-GPU
> pipeline-parallel (**PP**) do not exist as GPU modes. Expert-parallel (**EP**)
> exists only as a MoE weight-base-remap path (`orch::moe`, AMD MoE dispatch),
> not a general selector. Prefill/decode **disaggregation** is a skeleton
> (`orch::disagg`: `Pool`, `KvHandoff` types only, no dispatch). Plan multi-GPU
> serving around TP only.

### 8. Multi-model residency — `$VENDOR`: nvidia only

* **S1 switching** (`serve::manager::ModelManager`) — keep the largest
  VRAM-fitting subset resident; a non-resident request evicts LRU and loads.
  Knobs: `--drain-timeout-ms` / `PLOW_DRAIN_TIMEOUT_MS` (bounded-drain preemption
  on switch — measured, a live 512-tok generation drained O(max_tokens)≈13 s
  unbounded vs 258 ms at a 250 ms deadline), `--preload` / `PLOW_PRELOAD`
  (speculative next-model preload), `--vram-budget-mib` /
  `PLOW_VRAM_BUDGET_MIB` — the budget is a `$GPU` capacity question. See
  `perf-data/m1-multimodel-sm120.md`, `perf-data/multi-model-switch-gh200.md`.
* **`$VENDOR = amd` serve is one model per process** by decision — no
  `ModelManager`, no paging, no residency (`serve/engine.rs` module doc).

> **Planned, not implemented — do not depend on it.** The `docs/arch/06-runtime.md`
> **multi-tenancy** section (isolated counter pools, per-tenant arena partitions,
> priority-weighted scheduling) and the **watchdog** timer (stall detection via
> counter-progress monitoring) are design intent. There is no S2 multi-tenancy
> and no host watchdog in the runtime today. `memory::streamer::Streamer`'s
> reclaim policy (`EvictKv`/`StreamWeights`/`Preempt`) is likewise a skeleton
> whose `execute_reclaim` arms are no-ops.

---

## Step by step

Fix a concrete target first: **model, prompt length, target concurrency, TTFT
budget, TPOT budget**. Everything below is measured against it, on `$GPU`.
Commands assume a built `plowrt` (`nix develop`, then `cargo build -p plowrt
--release $FEATURES`) and Stage 5 assets in `$ASSETS`.

### 0. Sanity — does the schedule even run? (no GPU)

```bash
plowrt simulate --assets $ASSETS --all-buckets --chrome sim.json
```

`simulate` replays the packet stream on the CPU reference with full visibility:
no deadlocks, all dependencies honored, correct memory access. `--bucket
decode:1:128` isolates one bucket; `--math golden` runs reference numerics. The
only whole-schedule instrument that needs no device.

### 1. Direct-engine diagnostics (device, one sequence)

```bash
# Exercise a packet/object pair and inspect its greedy stream:
plowrt amd-bench --blob $ASSETS/model.pkt --hsaco $ASSETS/hsaco \
    --checkpoint $CKPT --prompt 1,2,3,4 --steps 64
```

This output is for bring-up and debugging, not a TPOT or TTFT result. Without
`--checkpoint` the weights are unbound and require `--synthetic-probe`; those
timings are synthetic diagnostics only.

Use the production engine for a one-load repeated prefill sweep (points must all
be ≤ `$MAXCTX`):

```bash
plowrt bench --assets $ASSETS --prefill-sweep \
    --prefill-lengths 512,1024,2048,4096,8192 \
    --prefill-warmups 1 --prefill-reps 3
```

For an exact token row, replace `--prefill-lengths ...` with
`--prompt-ids 1,2,3,4`. The distinct
`plowrt.bench.prefill-sweep.v1` JSON contains one TTFT distribution per
length, the deterministic prompt checksum, and warmup/repetition counts. Any
prefix-cache hit aborts the sweep instead of being reported as cold prefill.
Prefix caching must be disabled for the entire sweep.

### 2. Bring up serving and baseline it

Use the in-process production benchmark for the no-HTTP floor. It drives the
same `ServeEngine` and mux as `serve`, excludes model load and warmup from the
timed interval, and fails without JSON on partial output, shedding, rank
disagreement, or CPU fallback:

```bash
plowrt bench --assets $ASSETS --prompt-ids 1,2,3,4 \
    --concurrency 1 --requests 8 --warmup-requests 1 --output-len 64
```

The JSON records TTFT/TPOT/ITL/E2E distributions, throughput, scheduler rungs,
TP width, runtime settings, packet/object checksums, and checkpoint layout. Add
`--engine-diagnostics` to record bounded production-engine diagnostics. On AMD,
`diagnostics` contains the ordered prefill chunk boundaries, decode rung choices,
and TP agreement policy; overflow fails the benchmark instead of truncating the
record. Diagnostic capture is opt-in so normal benchmark timing is not perturbed.
Add `--trace-raw PATH` to write rank 0's last measured decode packet trace after
the production mux is drained. On AMD, `--amd-ctr-snap DIR` and
`--amd-tens-snap DIR` capture rank-0 state after every decode dispatch through
both `serve` and `bench`; select up to 16 named tensors with
`--amd-snap-tensors a,b` and a sequence with `--amd-snap-slot N`. Invalid
names/slots, oversized selections, capture failures, and filesystem failures
abort the request; existing snapshot filenames are never overwritten. Omitting
the tensor list selects the legacy four names, which still fail load when the
packet does not declare them or their aggregate exceeds 64 MiB. Use `amd-bench`
only for dump comparisons not yet migrated,
synthetic packet probes, and explicit TP correctness audits.

```bash
plowrt serve --assets $ASSETS --port 8080 --executors 8 \
    --max-hold-ms 8 --slo-ms 250
```

Drive it at the target concurrency and prompt length, and scrape `/metrics`:

| series | read |
|---|---|
| `rate(plowrt_batch_size_sum[5m]) / rate(plowrt_batch_count_total[5m])` | windowed mean batch size actually run |
| `plowrt_admit_shed_total` vs `plowrt_rejected_total` | shedding (controller) vs capacity (full table / oversize) |
| `plowrt_utilization`, `plowrt_arrival_rate` | queueing ρ = λ/μ, and λ |

Record baseline TTFT, TPOT, throughput. **If shedding at the target concurrency,
raise `--slo-ms` first and re-measure** (Lever 5 pitfall).

### 3. Fit memory to the target concurrency

If KV OOM sheds (`kv arena OOM` in the log, typed OOM to the client), shrink the
prefill chunk / KV ring so the target concurrency fits, and confirm the weight +
KV + scratch budget:

```bash
PLOW_LOAD_PROFILE=1 plowrt serve --assets $ASSETS --port 8080
```

### 4. Turn the levers, one at a time, and measure

Apply highest-value first, re-measuring TTFT/TPOT/throughput after each and
keeping only what helps *your* target:

1. **Multi-step + device sampling** (`$VENDOR = nvidia`): confirm
   `PLOW_MULTISTEP=8`, `PLOW_DEV_SAMPLE=1` — the largest single decode win.
2. **Batch width `B`** from the TPOT budget (Lever 1 pitfall) — may send you back
   to Stage 5 to recompile at a different `PLOW_DECODE_BATCH`.
3. **Prefill chunking / interleave** if TTFT-under-load (tail) is the problem
   (`--pf-interleave`, `--pf-chunk`); measure — finer chunking often *loses*
   throughput.
4. **Prefix cache** if traffic shares prompt prefixes (`--vmm-prefix` on nvidia,
   `--prefix-cache` on amd recurrent). **Report the hit rate.**
5. **TP** if one `$GPU` cannot hold the model or hit the latency budget. Gate on
   rank token-identity (`amd-bench --tp $NGPU`, `devices --tp $NGPU`).
6. **Load / cold start** (`--rt-prefetch-threads`, weight-slab knobs) if
   time-to-serving matters.

### 5. Diagnostics when a number is off

| instrument | env / flag |
|---|---|
| TTFT timeline (queue→prefill→first token; conc. 1) | `PLOW_TTFT_LOG=1` |
| per-decode-step host-phase breakdown | `PLOW_DSTEP_LOG=1` (+ `--rt-dstep-every N`) |
| prefix-cache timing | `PLOW_PFX_LOG=1` |
| per-tick prefill-vs-decode wall split | `PLOW_PF_PACKLOG=1` |
| packet timeline | `plowrt bench --trace-raw path` (amd decode) or `--trace` + `GET /trace` |
| static blob analysis (no device) | `plowrt disasm $ASSETS --counters --kernargs --tensors` |

**The attribution ladder.** Do not open a profiler first. Walk these in order,
cheapest first, and drill to the next level only when the current one comes back
flat — each rung costs more and perturbs more than the one above it. **The
ordering is the general part; the instruments are not**, and where a rung has no
tool for `$VENDOR` the honest move is to skip it and say the level went
unattributed.

| # | question | instrument | `$VENDOR` |
|---|---|---|---|
| 1 | client or server? | `PLOW_TTFT_LOG=1` — per-request TTFT decomposition (template / tokenize / queue / prefill / detok / unaccounted-HTTP). A large `UNACCOUNTED` means instrument before comparing anything | both |
| 2 | which class of work? | `PLOW_PF_SEG_TIME=1` — wraps every segment launch in device events, logs per-class totals (GEMM / fat / flash) + the slowest segments. Perturbs ~5%: read **shares, not absolutes** | nvidia (segmented prefill stack only) |
| 3 | which op? | build the device object with `-DPLOW_NV_TRACE=1`, serve with `PLOW_PF_TRACE_LOG=1` — per-opcode gate/body/signal cycles from block 0. Block 0 undercounts imbalanced ops several-fold: again shares, not absolutes | nvidia |
| 4 | host or device? | `PLOW_STEP_TIME=1` prints `gap_us=<G> dev_interp_ms=<K>` per decode step. **This is the branch point:** `G ≫ K` → host-bound, go to Lever 1/4; `K ≫ G` → kernel-bound, go back to Stage 4/5; comparable → do the host side first, it is free | nvidia |
| 5 | which stall inside the kernel? | `sudo ncu --set full` + Warp State Statistics (`nsys` cannot intercept a dlopen'd driver, so it does not see plowrt); `rocprof` on the AMD side | vendor profiler |
| 6 | is it the silicon? | sample clocks / power / throttle reasons **during** the bench (`nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active -lms 500`, or `rocm-smi` equivalents) before blaming the part | both |

Rungs 2–4 are NVIDIA-only as instruments: there is no in-tree AMD equivalent of
the per-class segment timer, the per-op cycle trace, or the host-gap split. On
`$VENDOR = amd` attribution runs rung 1, then jumps to rung 5 (`rocprof`) — a
real gap in coverage, and one worth stating in the write-up rather than
substituting a rung-1 number for a rung-4 conclusion.

---

## Success criteria

The model gates into Stage 7 when **all** hold:

1. **TTFT** at the target prompt length is within budget at concurrency 1 and
   does not blow past budget at the target concurrency.
2. **TPOT** at the target concurrency is within budget — and stays under it as
   concurrency rises to the target, not just at concurrency 1.
3. **Memory fits**: the target concurrency is *admitted*, not shed on KV OOM;
   `PLOW_LOAD_PROFILE=1` shows weights + KV + scratch resident with headroom.
4. **No spurious shedding**: `plowrt_admit_shed_total` ~0 at the target load; any
   429s are genuine capacity (`plowrt_rejected_total`).
5. **Correctness held**: every lever that can change numerics (batched prefill,
   ragged tail, prefix sharing) was re-checked against a correctness gate, not
   assumed neutral. On TP, rank token-identity holds.
6. **A recorded recipe**: the `plowrt serve` command + `PLOW_*` knobs with the
   measured TTFT/TPOT/throughput/memory numbers, **stamped with `$GPU`, `$ISA`,
   `$NGPU`/`$PARALLEL` and `$MAXCTX`**. A recipe that does not name the part it
   was measured on is not a result — it will be read as a target on the next
   part and silently be wrong.
7. **Every lever kept was measured on `$GPU`.** Defaults carried over from
   another part's campaign (chunk cost, `B`, prefetch threads) count as
   unmeasured until re-checked here.

---

## Pitfalls (from real serving campaigns)

* **Flat SLO sheds a wide blob.** A constant `--slo-ms` 429s every live stream
  once `live × step_ms` crosses it; looks like a B>8 emitter bug, is the
  admission model (`serving-capacity-report.md`).
* **Shed requests count as "successful" in benchmarks** with ~12 tokens each →
  fake throughput. Raise `--slo-ms` when throughput-benching.
* **A wide `B` doubles latency for the throughput.** Decode is bandwidth-bound;
  pick `B` from the TPOT budget (`b2-concurrency-family.md`).
* **A fixed large `B` overpays at partial load** — bad default for mixed traffic.
* **Finer prefill chunking usually loses throughput** — it multiplies launches
  and re-reads growing KV; it is a tail-latency tool (`serving-capacity-report.md`).
* **Cross-request batched prefill is not numerics-neutral** and is a no-op /
  force-off in common cases (`px14-batched-prefill-fp8.md`).
* **Prefix cache with a low hit rate is a net loss.** Always report the hit rate;
  the VMM `BlockHash` collision check is not yet implemented.
* **VMM weight-slab helps nvidia, hurts amd** — leave the amd default off.
* **Serve width must match the compiled width.** A blob compiled for one decode
  batch (or `-DGV_MM_MAX`) but driven at another routes through a predicated
  remainder arm and runs *slower* (`px10-batched-decode.md`).
* **A blob built for another part still loads and still runs.** `--arch` is
  packet metadata and its default disagrees with `--gpu`'s default by design;
  `PLOW_UNISEG` on `$VENDOR = amd` is warn-and-ignore, not a refusal. Nothing
  stops you serving an asset compiled for the wrong `$GPU`/`$NCU` — check the
  `build.json` before trusting a serving number.
* **Multi-model traps (nvidia):** duplicate network names silently drop a bundle;
  engine looked up by network but installed by slug (a slug override falls to the
  CPU reference path); `PLOW_PF_SEG_*` are process-global, so two models with
  different emit classing can't both be optimal (`multi-model-review-gh200.md`).
* **`gpulease` rc=76 is a false positive** for any serving benchmark on a shared
  box (`glm52-ttft-breakdown.md`).
* **`out_tok_s` / `bench_speed` throughput is confounded** on variable-completion
  benchmarks (prefix cache moves the stop position) — prefer `req_s`
  (`k3-throughput-architecture-review.md`).

---

## Code pointers

| symbol / path (`crates/plowrt/src/`) | role |
|---|---|
| `serve::mux` — `spawn`, `run_one_tick`, `admit_into`, `Slot`, `MuxConfig`, `advance_health` | continuous-batching mux / slot table / admission |
| `serve::engine::{ServeEngine, Ranks}` | per-vendor engine seam (nvidia slotted vs amd single-sequence / TP) |
| `serve::mod::{AppState, app}` | router: `/v1/{chat/,}completions`, tokenizer alignment, metrics, trace, health |
| `sched::admission::{LoadEstimator, Admit, admit}` | admission controller, SLO shed |
| `sched::batching::{select_bucket, formation_window_ms}` | bucket pick + adaptive hold |
| `sched::multistep::MultiStep::for_batch` | multi-step depth from batch size |
| `memory::AddressSpace` (`kv_layer_bases`) | per-device arena, physical rebase |
| `memory::streamer::{KvArena, SlotHandle, KvOom, Streamer}` | live KV allocator (`Streamer` reclaim = skeleton) |
| `memory::pool::GrowablePool` | per-head KV pool (Lean-mirrored) |
| `memory::prefix::{PrefixCache, Match}` | radix prefix cache, head-major runs |
| `memory::vmm::{VmmKv, VmmStatsHandle, WeightSlab, VmmSlab}` | VMM prefix sharing + weight slabs |
| `memory::kv::{BlockAllocator, PageTable}` | vLLM-style paged pool — **retained, off-path** |
| `exec::{ExecutorSet}`, `exec::gpu::GpuEngine`, `exec::amd::AmdEngine` (`rebase_chunk_rows`), `exec::amd_tp::AmdTpGroup` | executors / device engines |
| `exec::tp::{TpGroup, TpRank, PeerLayout}` | TP host side |
| `serve::manager::{ModelManager, BlobPlan, SwitchReport}` | S1 residency / switch (nvidia) |
| `orch::registry::Registry`; `orch::moe`, `orch::disagg` (skeleton) | model registry; EP/disagg |
| `config::{RuntimeConfig, NvidiaRuntimeConfig, AmdRuntimeConfig}` | every `PLOW_*` env var + CLI flag |
| `obs::Metrics::to_prometheus`, `obs::{ttft, dstep, pfx, trace}` | metrics + diagnostics |

Architecture reading: `docs/arch/06-runtime.md` (execution model — note its
multi-tenancy/watchdog sections are marked Planned), `docs/arch/09-multi-gpu.md`
(TP), `docs/arch/05-counter-system.md` (counter DAG),
`docs/arch/13-prefill-chunking.md`.
