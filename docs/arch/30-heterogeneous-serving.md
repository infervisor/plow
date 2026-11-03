# 30 — Heterogeneous serving: one runtime, mixed-vendor devices

**Status: design (2026-11-02), reviewed once (two independent reviews, every blocking finding
folded in), not implemented.** It extends [09](09-multi-gpu.md) (process rank groups, pipeline
stage packets), [22](22-kv-cache.md) (KV layout), [26](26-kv-block-store.md) (prefix store) and
`docs/runtime/data-parallel.md` (`--dp`). It covers one `plowrt serve` driving every GPU of a node
whose GPUs come from more than one vendor. The worked example is a box with two `sm_120a` GPUs
(96 GB, PCIe, peer access between them) and two `gfx1201` GPUs (32 GB, PCIe, large BAR).

**Goal.** One runtime binary and one serving endpoint use every GPU. Data parallel (DP), tensor
parallel (TP), pipeline parallel (PP) and prefill/decode disaggregation (P/D) are chosen by the
placement plan from what each model's packets declare, never from model names. Prefill may run
on one vendor and decode on another. Every supported combination works and is gated first;
performance work follows.

---

## 1. What exists today

| Piece | State | Where |
|---|---|---|
| Vendor choice | One vendor per process. Bring-up probes CUDA, then HSA, then CPU, keeps a single `dyn Backend`, and refuses a bundle whose `manifest.gpu` resolves to another vendor. | `main.rs` `bringup_runtime` |
| Asset target | One target per bundle: `weights.json` `gpu`, one blob `target` fingerprint, one objects dir. The dist index (`ModelIndex`) already lists several target variants per model; each variant/bundle has one target. | `plow-asset` `Manifest`, `dist.rs` |
| Weights | Bound by name from the shared checkpoint, TP-sliced at load. Bundles for two vendors can share one checkpoint. | `asset/serve.rs`, `asset/shard.rs` |
| Placement | A pure planner. `ModelSpec{slug, tp, required, device}` maps to `Group{ordinals, capacity}`. Groups are homogeneous and carry no vendor or ISA. CUDA uses it under the residency manager; AMD uses it with every model resident. | `serve/placement.rs` |
| DP | Ranks `slug#r`, routed by session, then prefix probe, then load. One prefix block size per set. | `serve/dp.rs` |
| TP, one process | AMD: peer VRAM, in-program collectives. | `exec/amd_tp.rs`, `exec/tp.rs` |
| TP, process ranks | NVIDIA, sibling branch: one device per process over a shared page-locked host region (control page, per-rank collective regions, a stage hand-off buffer). Gates are single-writer, a plan fingerprint arms each launch unit, and rank 0 samples. The rank binding and group serve are CUDA-typed today; a per-rank backend trait is in progress. | `exec/proc_group.rs`, `exec/gpu/rank.rs`, `serve/cuda_group.rs` |
| PP | Sibling branch, in progress: `stage<i>/` packet dirs plus a stage manifest, with `act.x` crossing as bf16 `[rows, H]`. | — |
| KV row import | Both engines have `write_kv_rows(plan, src)`. A span planner (`kv_handoff::plan`) covers head-major, seq-major and pooled caches, and a geometry digest (`kv_contract`) exists. Spans preserve layout, start at row 0 and share tensor-table handles. The only user is the CPU-twin head probe. There is no export path. | `exec/kv_handoff.rs`, `plow-asset/kv_contract.rs`, `serve/head.rs` |
| Prefix cache | CUDA: content-addressed block store, raw device bytes, `namespace` hard-coded 0. AMD: separate per-slot snapshot path. Neither format is portable. | [22 §8](22-kv-cache.md), [26](26-kv-block-store.md) |
| Sampling | CUDA applies request sampling. AMD and CPU take the device argmax. | `serve/engine.rs` `honours_sampling` |
| `orch::{disagg, transport}` | Type stubs only. | `orch/` |
| `gfx1201` | Absent: no ISA level, no hwspec entry, no packets. The RDNA4 workstream owns it. | — |

---

## 2. Device model

### 2.1 One serving process, both drivers

The serving process (HTTP, tokenization, admission, routing, host sampling, metrics) loads both
vendor drivers. They are already loaded on demand; the runtime stays static otherwise. It builds
one **device inventory**:

```
Device { id: DevId, vendor, isa: IsaLevel, spec: hwspec, target fingerprint (units, geometry),
         vram_bytes, numa, pci_bus, peers: [DevId] /* same-vendor peer access, probed */ }
DevId  = "nv:<ordinal>" | "amd:<ordinal>"     (the vendor's own enumeration, after its visibility mask)
```

* `--devices` takes `DevId`s. Bare integers keep today's meaning on a single-vendor box. A
  device not named gets no context.
* `ExecutorSet` maps `DevId → backend`: one `CudaBackend` per CUDA device, one `HsaBackend` per
  AMD group, as each vendor path builds them today. A single-vendor process is today's code with
  one list empty.

**Where engines run.** By default every engine runs in the serving process. Process rank groups
(the sibling branch's TP transport) spawn `plowrt rank-worker` children of the same binary,
supervised by the server and torn down with it. They remain allowed: the endpoint, the binary
and the scheduler are still one. This needs a product decision, listed in §12.

**Coexistence is H1's first gate, not an assumption.** Known hazards:

* the HSA runtime installs fault-signal handlers;
* both drivers pin pages and reserve address space;
* rank workers fork after drivers are loaded. Workers must exec, never fork-only.

The gate opens every device, runs a decode on each vendor concurrently, and moves data both ways
through a doubly registered host region (§6) under that load. If coexistence fails, the failing
vendor's engines move to rank-worker processes. Nothing else in this design changes, because every
cross-engine byte already goes through host regions and every cross-engine call through the
leader/worker link.

### 2.2 Engines and composite instances

An **engine** is one model instance's executor on one *device group*: one device, or a
same-vendor TP group. It is a `SeqEngine` implemented by `GpuEngine`, `AmdServe` or a rank group.
A served model is one or more **instances**. Each instance has one of these shapes:

| Shape | Members | Data between members |
|---|---|---|
| `Whole` | one engine | none |
| `Pipeline` | ordered stage engines, each a layer range | the declared boundary tensors (§3.3), per launch unit |
| `Disagg` | a prefill pool and a decode pool of `Whole` engines (`Pipeline` members later) | one KV image per request (§5) |
| `Dp` | N instances of the above behind one model name | none; the router picks one |

**P/D does not pool weight memory.** Each pool member holds the whole model, or a TP/PP slice of
it. A prefill-only packet drops decode rungs and their scratch, not weights. On this box a 32 GB
part can be a P/D prefill member only for a model whose weights plus prefill scratch fit in
32 GB. The planner checks this per rank (§4).

---

## 3. What packets and bundles declare

Everything placement needs is readable from the bundle **without opening a device**, as
`BlobPlan` already is.

### 3.1 Variants: existing bundles, composed locally

A **variant** is an existing, unchanged bundle directory: one target, one packet, its objects,
`serve.json`. Several variants form one served model when the serving profile (§4.2) lists them
under one model name. The loader then checks, and refuses on mismatch:

* the same checkpoint (the `serve_manifest` weight pins);
* the same `serve.json` (chat template, tool and reasoning declarations, stop ids, tokenizer
  digest), so the model answers identically on every device.

The dist schema is not changed first. Its index already lists several target variants per model;
`plowrt pull` fetching every variant present on the host is a later, small change.

### 3.2 Per-variant facts used by placement

| Fact | Source |
|---|---|
| full target fingerprint: ISA, unit geometry, objects pairing | blob `target`, object requirements, as `dist.rs` and the AMD loader already enforce |
| TP degree and shard map | `DevTp`, `weight_shards` |
| PP stage, layer range, boundary tensors | stage manifest (§3.3) |
| per-rank peak memory: weights, KV per slot at `max_ctx`, minimum KV, scratch, overhead | `BlobPlan`, per rank or stage, including prompt plus generation capacity |
| phases served | a new `roles` field in the packet's program metadata: `prefill`, `decode`, or both. Absent means both. |
| sampling capability | engine property: host sampling or device argmax |
| P/D compatibility | `kv_image` digest (§5) |
| PP compatibility | `stage_boundary` digest (§3.3) |
| transport needs | in-process TP requires `peers`; a rank group requires only host memory |

### 3.3 PP stage boundary (seam with the sibling branch)

A pipeline cut is legal only where devgen can name **every** tensor live across it. That is not
always `act.x` alone: a residual or norm carry may cross, and KV-shared layers can read a
producer layer's KV from an earlier stage. Each stage packet carries a `stage_boundary` section:

* `layers`: the layer range;
* `inputs` and `outputs`: named, typed boundary tensors with shapes in `rows` and `H`;
* `kv_owned`: the shared-KV ownership;
* row, slot and position mapping;
* the TP partition of each boundary tensor;
* a digest.

The emitter refuses a cut that needs an undeclared transfer (for example a KV-shared consumer
separated from its producer, until shared KV can be forwarded). Variants of different vendors
form a pipeline when their adjacent stages' digests match.

---

## 4. Placement

### 4.1 Planner

`serve/placement.rs` stays a pure function, generalised from homogeneous ordinals to the
inventory:

```
plan(inventory, models: [ModelRequest], policy, overrides) -> Result<Plan, PlacementError>
ModelRequest { model, variants: [Variant], dp: Dp, shape: ShapeRequest }
ShapeRequest = Auto | Whole | Pipeline{stages} | Disagg{prefill, decode}
Plan         = [Instance { model, key, shape, members: [(variant, role, [DevId])] }]
```

Rules, in order:

1. **Feasibility.** A member runs only on devices whose full target fingerprint matches its
   variant. ISA alone is not enough: unit geometry mismatches drop instruction streams.
   - A TP group is same-vendor and same-fingerprint. It runs in-process only when every pair is a
     peer; otherwise it is a process rank group.
   - Per-rank peak memory must fit: VRAM minus reserve minus co-tenants, with KV for the
     instance's slot count at its max context.
   - Roles must cover the shape: a `Disagg` decode member needs `decode`, and needs `prefill` too
     if local fallback (§7.3) is wanted.
2. **Overrides first.** The operator pins whatever they name. The planner fills in the rest and
   refuses an infeasible pin instead of moving it.
3. **Auto.** `Auto` means `Whole`: the smallest TP that fits, on the fastest matching group. Then
   `--dp all` spreads `Whole` instances over every group any variant fits. `Pipeline` and
   `Disagg` are never automatic: they are performance choices that must be measured, so the
   profile asks for them.
4. **DP sets are capability-uniform.** All ranks of one DP set must honour the same request
   features: sampling, logprobs, speculative decoding, media. Until AMD host sampling lands
   (§11), a mixed-vendor DP set is refused unless the profile restricts it to greedy-only
   (`dp_mix = "greedy"`). In that case a sampled request routes only to sampling ranks, and is
   refused if none exist. The DP prefix probe keeps a block size per rank.
5. **Anti-affinity.** DP ranks never share a device. P/D pools share one only by override.

The plan is logged as a table at start, served at `GET /v1/placement`, and exported as
`plowrt_placement_info{model, instance, role, stage, device, vendor, isa}`.

### 4.2 Operator override: the serving profile

`--plan <file.toml>` (or the serving profile's `[placement]` table), beside today's
`--pin/--place/--dp`, which keep working with `DevId`s:

```toml
[[variant]]
model = "gemma-4-12b"
dir = "models/gemma-4-12b/sm120a-tp1"
[[variant]]
model = "gemma-4-12b"
dir = "models/gemma-4-12b/gfx1201-tp1"

[[instance]]
model = "gemma-4-12b"
shape = "disagg"
prefill = [{ variant = "gfx1201-tp1", devices = ["amd:0"] },
           { variant = "gfx1201-tp1", devices = ["amd:1"] }]
decode  = [{ variant = "sm120a-tp1", devices = ["nv:0"] }]

[[instance]]
model = "qwen3-asr"
devices = ["nv:1"]
```

---

## 5. KV image: the vendor-neutral P/D handoff contract

### 5.1 Scope by state family (v1 is a capability subset)

The decode program reads per-request state of several families. P/D v1 admits a variant pair only
when every family its packets declare is in the **supported** column. The other columns refuse
the pair at load, never at request time.

| Family | v1 | Later | Refused until designed |
|---|---|---|---|
| dense GQA K/V, full and sliding (ring), bf16/f16 | ✅ | | |
| FP8 K/V with per-row(-head) f32 scales | ✅ | | |
| KV-shared layers (no bytes; source layer travels) | ✅ | | |
| recurrent + conv state (`recurrent_state` buffers) | | H6, only with `prefill_updates_state = true` | |
| MLA latent + rope (`ckv`, `krot`) | | after AMD/NV MLA parity | |
| DSA pooled indexer caches (`tokens_per_entry > 1`, packed FP8 blocks, partial pool) | | | ✅ |
| multimodal encoder or cross-attention state, mrope side positions | | after media P/D | ✅ |
| speculative / MTP draft caches and uncommitted draft tokens | | | ✅ (P/D instances run without speculation) |

`kv_contract`'s name-based digest stays for CPU-twin pairs. The image below is a new section,
because cross-vendor packets are separate emits whose tensor tables, native layouts and ring
depths differ.

### 5.2 Section `kv_image` (packet ABI, version 1)

devgen emits this section from the same facts that lay the caches out. It enumerates **every
persistent decode input** of the packet: each `kv.*` cache tensor, each `recurrent_state`
buffer, and each per-sequence host input the decode program reads (positions, generation
counters). A packet whose tensor table has a per-request tensor not covered by the section is
refused for P/D. Entries:

```
Entry {
  name:    "L{l}.k" | "L{l}.v" | "L{l}.k_scale" | "L{l}.v_scale" | "S{i}" | …,   // canonical id
  class:   Append | Window{w} | Shared{source} | State{buffer: recurrent_state index},
  dtype:   bf16 | f16 | f32 | e4m3 | e5m2,
  row:     [heads, dim]                     // Append/Window
  shape:   [..]                             // State: canonical element order
  scale:   none | { entry, granularity: per_row | per_row_head }
  // bindings (not digested):
  tensor:  "<native tensor name>",
  layout:  SeqMajor | HeadMajor{ring} | HeadInterleaved{ring} | SlotMajor,
}
KvImage { version, entries, rope: { applied_in_cache, digest(theta, scaling, partial dims) },
          positions: { kind: linear, header: [pos: u32, consumed: u32] }, digest }
```

The canonical image for frontier `n` and transfer range `[a, n)`:

* **`Append`**: rows `[a, n)`, seq-major `[row][head][dim]`, with scale rows alongside.
* **`Window{w}`**: rows `[max(0, n − w), n)` in token order, with the absolute start in the
  header. The importer wraps rows into its own ring: `row & (ring − 1)`, split at the ring end.
  Ring depths differ freely between sides.
* **`State`**: the buffer at exactly `n`, in canonical element order. The pair requires
  `prefill_updates_state` on the prefill side.
* **`Shared`**: no bytes.
* **Header**: `n`, `a`, the position base, the token history length, the pending first token and
  the producer variant id (§7.2).

Two variants are **P/D-compatible** exactly when their digests are equal. Native layout, names,
ring depth and slot count are bindings. Dtype, scale granularity, rope placement and position
semantics are numerics and must match: an e4m3-per-row-head prefill pairs only with a decode
packet reading exactly that.

### 5.3 Engine API: transactional import

```rust
fn export_image(&mut self, slot, range: Range<u32>, dst: &mut StagedImage) -> Result<ExportTicket>;
fn import_prepare(&mut self, slot, header: &ImageHeader) -> Result<ImportTicket>;
fn import_image(&mut self, t: &mut ImportTicket, src: &StagedImage) -> Result<()>;
fn import_commit(&mut self, t: ImportTicket) -> Result<()>;
fn import_abort(&mut self, t: ImportTicket);
```

**`import_prepare`** does what `begin_slot` does today:

* allocates VMM rows for `n` plus the generation budget;
* resets recurrence and generations;
* sets `pos`, the consumed frontier and the token history.

The slot is invisible to the dispatcher until **`import_commit`**. Commit:

* waits for the import DMA;
* publishes the frontier and enqueues the pending first token as the slot's next input.

**`import_abort`** waits for any in-flight DMA, then frees the slot. A slot is never reused while
a transfer touching it is in flight.

**Layout conversion.** Native and canonical layouts differ (head-major ring ↔ seq-major is a
transpose). v1 does it with **strided 2-D DMA**, one per (entry, head) run, split at ring wrap.
That gives independent source and destination pitches, absolute row ranges and wraparound, and
needs no new kernel. It costs about 2·L·H submissions per image, which is acceptable for the
correctness phase.

The perf phase adds per-vendor **pack/unpack kernels** that gather a slot's state into a
contiguous canonical staging buffer in VRAM (one launch, one DMA). They are screened at block
level against the copy-bandwidth roofline like any kernel.

**Rank groups.** A rank group exports per rank, because each rank holds a KV-head slice. The
image records `tp` and the head range of each piece. v1 requires equal TP on both sides.

### 5.4 Numerics and exactness

* The handoff is **data**: bytes are copied without conversion. The decode engine reads exactly
  what the prefill engine wrote.
* **Image round trip, byte-exact.** For each supported family: export from an engine, import into
  another slot of the same engine, export again, and compare bytes. The test runs at ring wrap,
  `n < w`, `n > ring`, FP8 scales and windowed + full mixes. It is a hardware unit test and needs
  no model gate.
* **Same-variant P/D, token-exact.** Prefill on `nv:0` and decode on `nv:1` with identical
  packets, against `Whole` on one device, under frozen conditions: c1, same chunking, same decode
  rung, greedy. Token ids and logprobs must be identical. A mismatch is a contract bug, never
  rounding.
* **Cross-vendor P/D** is its own **arm**. Prefill bytes differ from what the decode vendor would
  compute, so it passes the FP32-reference gate on its own, at the recipe's concurrencies, against
  the same-concurrency peer capture. The gate's multi-turn agentic cases exercise the mixed-prefix
  path (§7.2) with the cache on.
* **First token.** The prefill engine produces token 1 from its last-row logits. A sampled request
  needs the logit row on the host, where the leader draws with the request's seeded sampler (the
  rank-group leader-sampling path). The AMD engine needs that last-row return (§11). Until it
  lands, A→N P/D is greedy-only and refuses sampled requests to the instance.

---

## 6. Cross-vendor data movement

**Baseline: host staging.** A page-locked `memfd` mapping, aligned to 2 MiB (at least 64 KiB), is
registered with each driver that touches it:

* **CUDA**: host registration, device-mapped and portable.
* **HSA**: memory lock to the agents' pools. This returns an **agent pointer that need not equal
  the host pointer**. Each side keeps its own alias table and never uses the other's pointers.

If one region cannot be registered with both drivers, each vendor gets its own pinned pool and
the host copies between them: one extra copy, still correct.

**Completion protocol.** The exporting engine's host thread waits for its D2H completion (event
or signal). Only then does it release the slab's sequence number, as a single-writer flag. The
importer acquires the flag and issues H2D. The slab returns to the pool only after import
completion. Device-side system-scope atomics on host memory are not relied on (09, MIG
measurement).

**No direct cross-vendor peer access is assumed.** Same-vendor peer access (the NVIDIA pair) is
used where probed. A large-BAR cross-vendor device-to-device path is a perf-phase probe.

**Slabs.** v1 stages the **complete image**: rows `[a, n)` plus windows, state and scales. The
pool is sized for `K` concurrent handoffs at max context, and the plan reports `K`. A request
admitted to P/D holds one slab lease until import commit or abort. Bounded per-chunk streaming
with acknowledgements is a perf-phase item.

**Cost.** PCIe 5.0 x16 moves about 50 GB/s per direction, and the path is two hops (D2H, H2D). A
dense model at 48 layers × 2 × 8 heads × 128 × 2 B holds about 196 KB per token, so 8K tokens is
about 1.6 GB: about 32 ms per hop, about 64 ms unoverlapped. P/D pays only when it removes
prefill interference from decode or when streaming overlaps the copy. The plan reports the
estimate; it does not choose P/D itself.

---

## 7. Scheduling, prefix cache, failure, metrics

### 7.1 Request flow

| Shape | Flow |
|---|---|
| `Whole` / `Dp` | as today |
| `Pipeline` | the leader drives each launch unit through the stages in order, with the arm and fingerprint discipline per stage; sampling happens on the last stage or the leader |
| `Disagg` | see the steps below |

The `Disagg` steps:

1. **Admit.** Admission charges the decode pool's **capacity** (rows and slots, as an accounting
   reservation) and takes a slab lease, so a prefill never starts without a guaranteed landing.
   The physical decode slot is bound only at import (no head-of-line blocking of the decode
   pool's slots during long prefills).
2. **Attach and pin.** Attach and pin the decode engine's prefix hit `d`. Probe results are hints;
   only an attached, pinned hit counts.
3. **Pick the prefill engine.** Choose by load and prefix probe. The prefill engine attaches its
   own hit `p` and computes rows `[p, n)`.
4. **Export** rows `[min(d, n), n)`, plus whole `Window` and `State` entries. Every exported row
   must be materialised in the prefill slot. Full-layer rows below `p` are, because attach gathers
   them into the slot. If `d < p` and a row in `[d, p)` is not materialised, `a` drops to `p` and
   the decode engine recomputes `[d, p)` locally, or the transfer starts at `d` with the prefill
   engine computing from `d`.
5. **Import** with `import_prepare`, then `import_image`, then `import_commit`, and stream from
   there.

If the decode hit already leaves a tail shorter than one prefill chunk (`PLOW_PD_LOCAL_TAIL`), the
decode engine prefills the tail itself and no handoff happens.

### 7.2 Prefix cache across devices

The cache stays on in every engine. In v1, nothing is shared between devices.

* **Provenance.** The block store's `namespace`, hard-coded to 0 today, becomes a hash of
  (checkpoint, decode packet, producer variant).
  - Rows the decode engine computed itself publish under producer = self.
  - Imported rows publish under producer = the prefill variant.
  - An attach matches only blocks of one producer chain, except that a P/D instance may extend an
    imported prefix with locally computed tail rows. That mixed path is exactly what the instance's
    FP32 gate exercises through its multi-turn cases.
  - `Whole` instances are unaffected: producer is always self.
* **Later.** A canonical-image host tier shared by every engine of one model and producer, so a hit
  on one device serves another.

### 7.3 Cancellation, failure, fallback

* **Cancel or timeout** during a handoff: mark the request cancelled, let in-flight DMA drain,
  then `import_abort` and release the slab. Partial imports are never committed.
* **Output commit point.** Token 1 is streamed to the client when the prefill engine produces it.
  After that, the request's history is the prompt plus token 1, and any fallback continues from
  that history: the token is forced as input, never resampled.
* **Fallback** on import failure: the decode engine re-prefills the prompt and then takes token 1
  as input. This is possible only if the decode variant declares the `prefill` role; otherwise
  the request fails cleanly and the prefill member is marked degraded. Fallbacks are counted.
* **Engine failure** fails the engine's in-flight requests and marks the instance or pool member
  down. The DP and P/D routers skip down members.
* A rank group or pipeline with a failed member stays failed (as the sibling branch does).
  Re-placement goes through the control plane, not automatically.
* **`/health`** reports `degraded` while some instance of each model is alive.

### 7.4 Metrics

A P/D request counts once, in both the aggregate and the per-model view:

* The prefill rank records the arrival, queue time, time to first token and the prompt.
* The decode rank records the generated tokens, the completion and the end-to-end time. It
  measures them from the request's arrival, which the import carries.
* The two legs never run at the same time, so each counts as running in turn.
* One shared outcome token owns the request's terminal count (completed, aborted, preempted or
  rejected) until the decode leg is seated. A rank that refuses either leg settles it. If it is
  dropped without being settled, it counts one abort.
* The JSON runtime metrics give a data-parallel set one aggregated row. Each rank's row names its
  set (`rank_of`).

Untested path: a fatal device fault while a prefill slot holds an export sink. The dispatcher
releases the sink with the slot, so the handoff ends with an error. No backend mock can run the
GPU tick to cover this, so the path is checked only by inspection.

Existing per-model metrics gain `device`, `vendor` and `role` labels. New metrics:

- `plowrt_placement_info`
- `plowrt_pd_handoff_bytes_total`
- `plowrt_pd_handoff_seconds{stage=export|import|total}`
- `plowrt_pd_local_tail_total`
- `plowrt_pd_fallback_total{reason}`
- `plowrt_pd_slab_wait_seconds`
- `plowrt_pp_stage_seconds{stage}`
- `plowrt_xfer_bytes_total{link}`

---

## 8. Cross-vendor TP

Ranks on different vendors reduce partial sums made by different GEMM kernels. Every rank then
holds logits that differ in rounding, so per-rank argmax agreement is lost. Cross-vendor TP is
therefore allowed only when all of these hold:

* leader sampling covers every row, with the token broadcast as the next input;
* both variants come from the same TP split (the same `weight_shards` axes and the same
  collective protocol over host memory);
* the arm passes the FP32-reference gate.

The slowest member paces every collective, and mixed PP or P/D gives similar memory and compute
pooling without lockstep. Cross-vendor TP is **last priority**, implemented only if a model fits
no other shape.

---

## 9. Support matrix

Columns: NV = `sm_120a` only, AMD = `gfx1201` only, Mix = both vendors in one served model. For
DP, Mix means ranks of both vendors behind one name; for PP and P/D it means one instance spanning
vendors.

Legend:

- **R**: required, built and gated here.
- **R\***: required, but blocked on the sibling workstream delivering the single-vendor variant
  (RDNA4 for AMD; NV TP/PP).
- **L**: later, after the R set or on demand.
- **—**: out of scope or meaningless.

| Model class | Placement (models per device) | DP NV / AMD / Mix | TP NV / AMD / Mix | PP NV / AMD / Mix | P/D NV / AMD / Mix A→N / Mix N→A |
|---|---|---|---|---|---|
| Dense LLM (GQA, sliding, FP8 KV, kv-shared) | R | R / R\* / R | R\* / R\* / L | R\* / L / R\* | R / L / R / L |
| MoE LLM (same KV families) | R | R / R\* / R | R\* / R\* / — | R\* / L / R\* | R / L / R / L |
| GDN hybrid LLM | R | R / R\* / R | L / L / — | L / L / L | L / L / L (H6) / L |
| MLA / DSA LLM | R | R\* / R\* / L | R\* / R\* / — | L / L / L | — (refused, §5.1) |
| TTS (LM + codec) | R | R / R\* / R | — | — | — |
| ASR (encoder + decoder) | R | R / R\* / R | — | — | L (encoder/decoder split) |
| VAD | R (any device or CPU) | R / R\* / R | — | — | — |

* **Placement** is co-serving: for example LLM decode on `nv:*` and speech on `amd:*`. It is the
  first mixed deliverable and needs only §2–§4.
* **Mix P/D A→N** (prefill on `gfx1201`, decode on `sm_120a`) is the named case. It needs a model
  whose AMD prefill variant fits in 32 GB (§2.2). N→A uses the same code; its gate run follows
  A→N.
* **NV same-vendor P/D** is required because it is the exactness test (§5.4).
* **Mix DP** needs uniform capabilities (§4.1 rule 4): greedy-only until AMD host sampling lands.
* **Mix PP** is R\*: it builds on the sibling branch's stage packets and the `stage_boundary`
  section.

### 9.1 Gates per combination

| Class | Functional gate | Extra |
|---|---|---|
| LLM | FP32-reference gate at c1 and c8 (c32 at promotion) against the same-concurrency peer capture of the same checkpoint; every Mix instance is its own arm | image round trip byte-exact (§5.4); same-variant P/D token-exact; same-vendor PP vs `Whole` token-exact at c1 |
| TTS | the recipe's TTS gate (round-trip CER) | — |
| ASR | WER/CER on the recipe's eval set within threshold | realtime session routing across vendors |
| VAD | frame decisions identical to the reference program | — |

---

## 10. Phases

Each phase ends green (unit tests, hardware smoke on the box, its gates) and merges into the
integration branch.

| Phase | Deliverable | Gate |
|---|---|---|
| **H1** | Coexistence gate (§2.1); inventory, `DevId`, `ExecutorSet` per device; the per-vendor model paths in one process; profile-composed variants (§3.1); `Whole` placement of different models on different vendors | coexistence gate; every box-kit model serves from one process on its device and passes its own gate |
| **H2** | Planner v2 (fingerprint, per-rank peak memory, roles), `--plan`, `/v1/placement`, metrics; DP across vendors, capability-uniform | router and planner unit tests; FP32 gate per rank vendor |
| **H3** | `kv_image` section (devgen), host staging regions (dual registration and the completion protocol), CUDA `export_image` and transactional import, `Disagg` flow; NV same-variant P/D | byte-exact round trip; token-exact same-variant P/D; FP32 gate |
| **H4** | AMD export/import, AMD last-row logits for host sampling; Mix P/D A→N, then N→A | FP32 gate c1/c8 on each Mix arm; dense, then MoE |
| **H5** | Mix PP on stage packets with `stage_boundary` | same-vendor PP token-exact; Mix PP FP32 gate |
| **H6** | `State` entries: GDN P/D | round trip; FP32 gate |
| **H7** | Cross-vendor TP, only under §8's conditions | FP32 gate |

**Perf phase** (after the R set is green). Each item is measured against the same model as
`Whole` and against the peer, at block level first where a kernel is involved:

- pack/unpack kernels;
- per-chunk streaming of the image, overlapped with prefill;
- a large-BAR device-to-device probe;
- P/D admission policy (which requests go P/D, by prompt length and load);
- pipeline micro-batching;
- a shared canonical-image host tier.

---

## 11. Seams needed from the sibling workstreams

**NV TP/PP:**

* (a) a per-rank backend trait in the rank-group serve, so a rank or stage can run on HSA;
* (b) host-region registration behind a trait (CUDA host register / HSA memory lock) that keeps
  per-driver aliases, uses 2 MiB alignment, and supports named regions beyond the one hand-off
  buffer, so P/D staging reuses it;
* (c) the stage manifest's `stage_boundary` section (§3.3) and each stage's target fingerprint;
* (d) leader sampling as an engine-independent call, used for the P/D first token.

**RDNA4:**

* (a) the `gfx1201` ISA level and hwspec entry, with `from_probe` resolving the R9700;
* (b) `gfx1201` variants of the box models, with the same `serve.json` and KV dtype choices as
  the `sm_120a` variants wherever P/D is wanted;
* (c) slotted serving with the prefix cache on;
* (d) a last-row logits return, so the host can sample;
* (e) strided D2H/H2D for export and import.

**Bring-up:**

* a both-vendor lease in gpulease;
* gpuq submission of a both-vendor job.

---

## 12. Open decisions

1. **"One process."** Is a supervised `rank-worker` child of the same binary acceptable inside
   "one plowrt"? This design says yes: one endpoint, one binary, one scheduler. It is needed by the
   NV process-rank TP transport and as the coexistence fallback.
2. **Box-kit model set** for the gates. It is not yet confirmed; the AMD 32 GB limit decides which
   models can be P/D prefill members.
