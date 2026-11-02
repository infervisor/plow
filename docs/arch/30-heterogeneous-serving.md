# 30 — Heterogeneous serving: one runtime, mixed-vendor devices

**Status: design (2026-11-02), not implemented.** It extends [09](09-multi-gpu.md) (process rank
groups, pipeline stage packets), [22](22-kv-cache.md) (KV layout), [26](26-kv-block-store.md)
(prefix store) and `docs/runtime/data-parallel.md` (`--dp`). It covers one `plowrt serve` driving
every GPU of a node whose GPUs come from more than one vendor. The worked example is a box with
two `sm_120a` GPUs (96 GB, PCIe, peer access between them) and two `gfx1201` GPUs (32 GB, PCIe,
large BAR).

Goal: one runtime binary serves models on every GPU. Data parallel (DP), tensor parallel (TP),
pipeline parallel (PP) and prefill/decode disaggregation (P/D) are decided by the placement plan
and the packets each model ships, never by model names. Prefill may run on one vendor and decode
on another. Every combination is made to work first. Performance work follows.

---

## 1. What exists today

| Piece | State | Where |
|---|---|---|
| Vendor choice | One vendor per process. Bring-up probes CUDA, then HSA, then CPU, and refuses a bundle whose `manifest.gpu` resolves to another vendor. | `main.rs` `bringup_runtime` |
| Asset target | One target per bundle: `weights.json` `gpu`, one blob `target` fingerprint, one objects dir. The dist schema keys variants by `(vendor, isa, sku)`, one target each. | `plow-asset` `Manifest`, `dist.rs` `Target` |
| Weights | Bound by name from the shared checkpoint, with TP slicing at load. Two vendors' bundles can share one checkpoint. | `asset/serve.rs`, `asset/shard.rs` |
| Placement | Pure planner (`ModelSpec{slug, tp, required, device}` → `Group{ordinals, capacity}`). Groups are homogeneous and carry no vendor or ISA. CUDA uses it under the residency manager. AMD uses it with every model resident. | `serve/placement.rs` |
| DP | Ranks `slug#r` behind one model. The router uses session affinity, then prefix probe, then least load. | `serve/dp.rs` |
| TP, one process | AMD: peer VRAM and in-program collectives. | `exec/amd_tp.rs`, `exec/tp.rs` |
| TP, process ranks | NVIDIA (branch nv-tp-pp): one device per process. A shared page-locked host region carries the control page, per-rank collective regions and a stage hand-off buffer. Gates are single-writer. Every launch unit arms on a plan fingerprint. Rank 0 samples. | `exec/proc_group.rs`, `exec/gpu/rank.rs`, `serve/cuda_group.rs` |
| PP | nv-tp-pp: `stage<i>/` packet dirs plus a stage manifest. `act.x` crosses as bf16 `[rows, H]` through the shared host region. Each stage's backend comes from its stage packet's target. | nv-tp-pp |
| KV row import | The engine method `write_kv_rows(plan, src)` exists on both engines, along with a span planner over head-major and seq-major caches and a geometry digest. Its only user is the CPU-twin head probe. There is no export path. | `exec/kv_handoff.rs`, `plow-asset/kv_contract.rs`, `serve/head.rs` |
| Prefix cache | CUDA: a content-addressed block store, with raw device bytes keyed by token-hash chains and `namespace` 0. AMD: a separate per-slot snapshot path. Neither format is portable. | [22 §8](22-kv-cache.md), [26](26-kv-block-store.md) |
| Sampling | CUDA applies the request's sampling. AMD and CPU take the device argmax. | `serve/engine.rs` `honours_sampling` |
| `orch::{disagg, transport}` | Stubs (types only). | `orch/` |
| `gfx1201` | Absent: no ISA level, no hwspec entry, no packets. The amd-rdna4 workstream owns it. | — |

---

## 2. Device model

### 2.1 One process, both drivers

The serving process loads both vendor drivers (they are already loaded on demand, and the runtime
stays a static binary otherwise). It builds one **device inventory**:

```
Device { id: DevId, vendor, isa: IsaLevel, spec: hwspec, vram_bytes, numa, pci_bus,
         peers: [DevId] /* same-vendor peer access, probed */ }
DevId  = "nv:<ordinal>" | "amd:<ordinal>"     (ordinal = the vendor's own enumeration)
```

* `--devices` takes `DevId`s: `nv:0,nv:1,amd:0,amd:1`. Bare integers keep today's meaning, CUDA
  ordinals on a CUDA-only box. `CUDA_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` still mask
  devices before enumeration. The bring-up probe opens only the devices the plan uses: a device
  that is not named gets no context.
* The `Backend` set becomes per device rather than per process: `ExecutorSet` holds an
  `Arc<CudaBackend>` per CUDA device and an `HsaBackend` per AMD group, as each vendor path
  builds them today.
* A process with one vendor behaves exactly as today. Mixed mode is the same code with both lists
  non-empty.

**Coexistence risk.** Both drivers in one address space is untested on this box. Specific
concerns are signal handlers, the fork/exec of rank workers, and pinned-page accounting. H1 starts
with a gate: open every device, register one host region with both drivers, and copy through it
in both directions concurrently with a decode on each vendor. If that fails, the fallback is
already built: engines of the failing vendor run in **rank-worker processes** of the same binary
(nv-tp-pp's `plowrt rank-worker`). The rest of this design is unchanged, because everything
between engines already goes through host regions and the leader/worker link.

### 2.2 Engines and hosts

An **engine** is one model instance's executor on one *device group*: one device, or a same-vendor
TP group. It is `GpuEngine` (CUDA), `AmdServe` (HSA) or a process rank group. Two ways to host an
engine:

* **In-process**: today's CUDA single-GPU and AMD single-process TP.
* **Rank group**: a leader plus `rank-worker` processes (nv-tp-pp). After nv-tp-pp's backend-trait
  refactor, a rank is "a `SeqEngine` on some backend". A group's ranks therefore need not share a
  vendor at the transport level. Cross-vendor TP is still gated separately (§7).

The leader process owns HTTP, tokenization, admission, routing, sampling of host-drawn rows,
prefix-route state and metrics. Each engine keeps its own dispatcher thread (`exec/engine_thread.rs`),
slots and prefix cache.

### 2.3 Composite instances

A served model is one or more **instances**. An instance is one of four shapes:

| Shape | Members | Data between members |
|---|---|---|
| `Whole` | one engine (TP inside it) | none |
| `Pipeline` | ordered stage engines, each running a layer range | activations `[rows, H]` per launch unit |
| `Disagg` | a prefill pool and a decode pool, each a set of `Whole` or `Pipeline` engines | KV image of a prompt (§5), once per request |
| `Dp` | N instances of the above behind one model name | none; the router picks one |

`Dp` over `Disagg` is allowed: P/D pools are themselves the DP dimension.

---

## 3. What packets and manifests declare

The rule: everything placement needs is readable from the asset **without opening a device**, as
`BlobPlan` already is.

### 3.1 Multi-target assets

An asset (one model, one checkpoint) may carry several **variants**:

```
<asset>/
  targets.json                 # plow-asset section `asset_targets`, version 1
  checkpoint -> …              # shared, pinned by `serve_manifest` weight pins
  sm_120a-tp1/   (today's bundle layout: weights.json, model.pkt, objects/, serve.json …)
  sm_120a-tp2/
  gfx1201-tp1/
  gfx1201-pp2/stage0/ stage1/  # nv-tp-pp stage layout
```

`targets.json` lists the variants:

```
{ "version": 1,
  "model": "<served name>",
  "checkpoint_pin": "<sha of the weight pins>",
  "variants": [
    { "dir": "sm_120a-tp1", "isa": "sm_120a", "tp": 1, "pp": 1,
      "roles": ["prefill", "decode"],        // which phases its programs serve
      "stages": null,                        // or [{ "dir": "stage0", "layers": [0, 24] }, …]
      "plan": { "weights": …, "kv_per_slot": …, "other": …, "min_kv": … },   // BlobPlan, per rank
      "kv_image": "<digest>",                // §5; absent = cannot take part in P/D
      "act_handoff": "<digest>" }            // PP boundary contract; absent = not a stage set
  ] }
```

* A directory with no `targets.json` is a single-variant asset, read from its `weights.json`
  exactly as today. Nothing changes for existing kits.
* Every variant's `serve.json` must match: chat template, tool and reasoning declarations, stop
  ids and tokenizer digest. The loader compares them and refuses a mismatch. A model answers the
  same on every device.
* `roles` lets an emitter ship a **prefill-only** variant, with no decode rungs, for a prefill
  pool. That cuts VRAM on the 32 GB parts.
* The dist schema moves from one `Target` per bundle to a list of variants. `plowrt pull` fetches
  the variants whose ISA is present on the host. The variant layout under `objsets/<vendor>/<isa>/`
  already matches.
* hwspec gains the `gfx1201` entry (owned by amd-rdna4). `registry::from_probe` must not require
  an exact unit count for a known ISA plus SKU. Today an unlisted part resolves to no spec and its
  bundle is refused.

### 3.2 What a variant contributes to placement

| Fact | Source |
|---|---|
| target ISA → which devices can run it | blob `target` + `targets.json` `isa` |
| TP degree | `DevTp.n_gpu` (as today) |
| PP stages and their layer ranges | stage manifest (nv-tp-pp) |
| per-rank memory: weights, KV per slot, minimum KV, overhead | `BlobPlan`, per rank or stage |
| phases served | `roles` |
| P/D compatibility | `kv_image` digest (§5) |
| PP compatibility across variants | `act_handoff` digest: dtype, shape `[rows, H]`, residual/norm state crossing the boundary |
| transport needs | TP in one process needs `peers`; a process rank group needs only host memory |

---

## 4. Placement

### 4.1 Planner

`serve/placement.rs` stays a pure function, generalised from homogeneous ordinals to the
inventory:

```
plan(inventory, models: [ModelRequest], policy, overrides) -> Result<Plan, PlacementError>
ModelRequest { model, variants: [Variant], dp: Dp, shape: ShapeRequest }
ShapeRequest = Auto | Whole{variant?} | Pipeline{variants per stage?} | Disagg{prefill, decode}
Plan         = [Instance { model, key, shape, members: [(variant, [DevId])] }]
```

Rules, in order:

1. **Feasibility.** A member runs only on devices whose ISA matches its variant. A TP group is
   same-vendor and same-ISA. It is in-process only if every pair is a peer; otherwise it is a
   process rank group. A member fits if its `BlobPlan.required` fits the device's capacity:
   VRAM minus the reserve minus co-tenants.
2. **Overrides first.** The operator pins whatever they name (§4.2). The planner fills in the
   rest and refuses an infeasible pin instead of moving it.
3. **Auto policy.** `Auto` means `Whole`: the smallest TP that fits, on the fastest
   matching-ISA group (ranked by the hwspec peak of the variant's dominant phase). Then `--dp all`
   spreads `Whole` instances over every group that fits any variant, across vendors.
   `Pipeline` and `Disagg` are never chosen automatically. They are perf choices that must be
   measured, so the operator or the recipe asks for them.
4. **Anti-affinity.** DP ranks never share a device, as today. P/D pools may share a device only
   with an explicit override.
5. **Spread or pack.** These apply to co-tenancy as today (`Place`).

The plan is logged as a table at start, served at `GET /v1/placement`, and exported as
`plowrt_placement_info{model, instance, role, stage, device, vendor, isa}`.

### 4.2 Operator override

`--plan <file.toml>`, or a `[placement]` table in the serving profile, beside today's
`--pin/--place/--dp`:

```toml
[[instance]]
model = "gemma-4-12b"
shape = "disagg"
prefill = [{ variant = "gfx1201-tp1", devices = ["amd:0"] },
           { variant = "gfx1201-tp1", devices = ["amd:1"] }]
decode  = [{ variant = "sm_120a-tp1", devices = ["nv:0"] }]

[[instance]]
model = "qwen3-asr"
shape = "whole"
devices = ["nv:1"]

[[instance]]
model = "big-moe"
shape = "pipeline"
stages = [{ variant = "sm_120a-pp2", stage = 0, devices = ["nv:0"] },
          { variant = "gfx1201-pp2", stage = 1, devices = ["amd:0"] }]
```

`--pin slug@N` keeps working, with `N` read as a `DevId` or a bare ordinal.

---

## 5. KV image: the vendor-neutral P/D handoff contract

### 5.1 Why a new contract

The existing `kv_contract` digest compares per-slot byte extents of `kv.*` tensors by **name and
tensor-table index**. That suffices for a CPU twin emitted from the same layout. It does not
suffice across vendors:

* the two packets are separate emits whose tensor tables need not align;
* head-major vs seq-major, ring depth and `--max-ctx` legitimately differ per vendor;
* recurrent and conv state is excluded on purpose.

### 5.2 Section `kv_image` (packet ABI, version 1)

Emitted by devgen for every packet whose programs write per-request state. It describes a
**canonical image**: the bytes one sequence's state occupies at frontier `n`. The layout is fixed
by the contract, not by either packet.

```
KvImage { version: 1,
  entries: [ Entry {
    name:  "L{l}.k" | "L{l}.v" | "L{l}.ckv" | "L{l}.krot" | "L{l}.k_scale" | "L{l}.idx" | "L{l}.state.<i>" | …,
    class: Append | Window{w} | State | Shared{source_layer},
    dtype: bf16 | f16 | f32 | e4m3 | e5m2 | u8,
    row:   [heads, dim]                // Append/Window: one token's row, canonical order head-major within the row
    shape: [..]                        // State: canonical shape (e.g. [heads, dk, dv]); conv: [channels, k-1]
    scale: { granularity: per_row_head | per_row | per_tensor, tensor: "<entry name>" } | none,
    tensor: "<this packet's tensor name>",   // binding, NOT part of the digest
    layout: SeqMajor | HeadMajor{ring} | HeadInterleaved | SlotMajor   // native layout, NOT part of the digest
  } ],
  rope: { applied_in_cache: bool, digest: "<theta, scaling, partial dims, position base>" },
  positions: { kind: linear | mrope{sections}, extra_per_seq: [..] },
  digest: sha256(version, canonical entries without the binding fields, rope, positions) }
```

The canonical image for `n` tokens:

* `Append`: rows `[0, n)`, seq-major `[n][heads][dim]`, followed by its scale rows.
* `Window{w}`: rows `[max(0, n−w), n)` in token order. The image carries the absolute start, so
  any ring depth on either side works.
* `State`: the state at exactly `n` (recurrent, conv, AttnRes-style snapshots), in the canonical
  shape.
* `Shared`: no bytes. Kv-shared layers reuse their source layer's entry.
* Per-sequence side values the decode program reads (position base, mrope offsets) travel in the
  image header.

Two variants are **P/D-compatible** exactly when their digests are equal. Native layout, tensor
names, ring depth and slot count are bindings, so a head-major ring packet can pair with a
seq-major one. Dtype, scale granularity, rope placement and position semantics cannot differ:
they are numerics, not layout. A prefill variant writing e4m3 KV with one scale per row and head
pairs only with a decode variant reading exactly that.

### 5.3 Engine API

Two methods on each engine (CUDA, AMD, rank group; the CPU engine for tests), driven by the
`kv_image` bindings:

```rust
fn export_image(&mut self, slot, rows: Range<u32>, dst: &mut HostImage) -> Result<()>;   // native → canonical
fn import_image(&mut self, slot, rows: Range<u32>, src: &HostImage) -> Result<()>;       // canonical → native
```

* `rows` lets a decode engine that already holds a prefix (its own prefix cache, §6) import only
  the tail `[hit, n)`. `Window` and `State` entries always travel whole at `n`.
* Head-major ↔ canonical is a strided copy: one span per (layer, head) and per contiguous run, as
  `kv_handoff::plan` already computes. The import side reuses `write_kv_rows`. Export is its
  mirror, a D2H span copy, which neither engine has yet. A device gather or scatter kernel per
  layout is an optimisation for the perf phase.
* `HostImage` lives in a host staging region (§6) and is registered with the exporting and the
  importing driver, so the bytes are copied once in each direction and never pass through a CPU
  memcpy.
* A rank group exports per rank (each rank holds its KV-head slice). The image therefore carries
  `tp` and the head range per piece. v1 requires equal TP on both sides, or TP1 to TP1. Re-slicing
  across different TP degrees is a later extension.

### 5.4 Numerics

* The handoff is **data**. Bytes are copied without conversion, so a decode engine running from
  an imported image reads exactly the bytes the prefill engine wrote.
* The bytes differ from those the decode vendor would itself compute, because the kernels differ.
  The P/D instance is therefore a different **arm** and passes the FP32-reference gate in its own
  right, at the recipe's concurrencies, against the same-concurrency peer capture.
* An **exactness test** pins the contract independently of any vendor. Same-variant P/D (prefill
  on `nv:0`, decode on `nv:1`, identical packets) must produce token ids and logprobs
  bit-identical to the whole-instance run with the same chunking. The test runs at decode rungs 1
  and 8 and at windowed, FP8-KV and recurrent-state contexts. Any mismatch is a contract or
  export/import bug, never rounding.
* First token: the prefill engine's last-row logits produce token 1. For a sampled (non-argmax)
  request, the prefill engine returns the logit row to the leader, which draws with the request's
  seeded sampler. This is the same leader-sampling path nv-tp-pp built for rank groups. The
  decode engine then continues from the token as usual. An AMD engine that only argmaxes on
  device needs that logit-row return; amd-rdna4 owns it.

---

## 6. Cross-vendor data movement

* **Baseline: host staging regions.** A page-locked `memfd` mapping is registered with every
  driver that touches it: CUDA host registration (device-mapped, portable) and HSA memory lock to
  the agents' pools. Copies are each vendor's async DMA H2D or D2H on its own stream or queue.
  Completion crosses vendors by **single-writer flags** in the region, with the protocol of
  nv-tp-pp's control page. The writer releases a sequence number and the reader acquires it.
  System-scope atomics on mapped host memory are not relied on; see 09's MIG measurement.
* **Direct cross-vendor peer access is not assumed.** Same-vendor peer access (the NVIDIA pair)
  is used where probed, for TP and for same-vendor P/D. A cross-vendor device-to-device path
  through a large-BAR window is a perf-phase probe, not a dependency.
* **Regions.** One leader-owned pool per link (prefill pool → decode pool, stage i → stage i+1).
  Each is a ring of fixed slabs sized from the plan: KV image bytes per token × the prefill
  chunk, or the `act_handoff` size. Slabs are leased per request and returned on import, so
  host memory is bounded. A request that cannot get a slab waits in admission (§7). It is not
  started.
* **Cost model.** PCIe 5.0 x16 moves about 50 GB/s per direction, and two DMA hops (D2H, then
  H2D) cost about 2× the bytes. For P/D, the KV image is `n × bytes_per_token`. For example,
  48 layers × 2 × 8 heads × 128 × 2 B is about 196 KB per token, or about 1.6 GB at 8K tokens:
  about 32 ms per hop, about 64 ms total. P/D pays only when it overlaps prefill (stream the image
  per prefill chunk, §8 P-phase) or frees the decode engine from prefill interference. The planner
  reports the estimate. It does not pick P/D on its own.

---

## 7. Scheduling, prefix cache, failure, metrics

### 7.1 Request flow

| Shape | Flow |
|---|---|
| `Whole` / `Dp` | as today (DP router: session → prefix → load) |
| `Pipeline` | the leader drives every stage per launch unit, in stage order, with nv-tp-pp's arm/fingerprint discipline per stage; decode steps flow stage 0 → … → last; sampling on the last stage (or the leader) |
| `Disagg` | admit → **reserve a decode slot first** (a prompt is never prefilled with nowhere to land) → pick a prefill engine (prefix probe + load, the DP router's rules over the prefill pool) → prefill in chunks, exporting each completed chunk's rows into the request's slab → token 1 → import the image into the reserved decode slot → enqueue on the decode dispatcher |

Admission is global in the leader. A P/D request holds a decode-slot reservation and a slab lease
for its whole prefill. Both are released on cancel, error or timeout.

### 7.2 Prefix cache across devices

The cache stays on in every engine, as each engine's own cache. Nothing is shared between devices
in v1.

* **Whole/DP:** unchanged.
* **Disagg:** the decode engine holds the most of a session's history: prompt plus generated rows.
  The router therefore probes the **decode** engine first. If its hit leaves a tail shorter than a
  threshold (`PLOW_PD_LOCAL_TAIL`, default one prefill chunk), the tail is prefilled locally on the
  decode engine with no handoff. Otherwise the prefill engine prefills from its own cache hit, and
  only rows `[decode_hit, n)` plus whole `Window` and `State` entries are exported and imported
  (§5.3 `rows`). Each engine publishes into its own cache as today.
* **Namespacing:** the block store's `namespace` (0 today) becomes the variant's packet hash
  (doc 26 §B design). Two variants never confuse each other's blocks, even if a future host tier
  is shared.
* **Later:** a canonical-image host tier shared by every engine of a model, so a hit on one
  device serves another. This is not v1.

### 7.3 Failure

* An engine failure fails its in-flight requests with a clean error and marks the instance down.
  The DP router already skips down ranks, and the P/D router skips down pool members.
* Handoff failure (slab timeout, import error, digest mismatch at load, which is refused at
  start) re-runs the prefill on the decode engine once and counts it. The request is not lost
  unless the decode engine is down too.
* A rank group or pipeline with a failed member stays failed, as nv-tp-pp does. Re-placement is
  operator-driven: reload through the control plane. No automatic re-planning in v1.
* `/health` reports `degraded` while any instance of each model is alive.

### 7.4 Metrics

All existing per-model metrics gain `device`, `vendor` and `role` labels. New:

* `plowrt_placement_info`
* `plowrt_pd_handoff_bytes_total`
* `plowrt_pd_handoff_seconds{stage=export|import|total}`
* `plowrt_pd_local_tail_total`
* `plowrt_pd_fallback_total{reason}`
* `plowrt_pd_slab_wait_seconds`
* `plowrt_pp_stage_seconds{stage}`
* `plowrt_xfer_bytes_total{link}`

---

## 8. Cross-vendor TP

Ranks on different vendors would reduce partial sums produced by different GEMM kernels. Every
rank then takes the argmax of logits that differ in rounding, and greedy agreement between ranks
is no longer guaranteed. Cross-vendor TP is therefore allowed only:

* with leader sampling from rank 0's reduced logits, broadcast as the next input, which nv-tp-pp
  already does for sampled rows; it would apply to every row here;
* with both variants emitted from the same TP split (the same weight-shard axes and the same
  collective protocol over the host region);
* after the arm passes the FP32-reference gate.

It is **last priority**. On this box the 32 GB parts would bound a 2+2 TP group at 4 × 32 GB of
usable memory, with the slowest member pacing each collective. Mixed PP and P/D give the same
memory pooling without lockstep. The plan implements it only if a model that fits no other shape
needs it.

---

## 9. Support matrix

Columns: NV = `sm_120a` only, AMD = `gfx1201` only, Mix = both vendors in one instance (DP: ranks
of both vendors behind one model). Legend: **R** = required, built and gated in this workstream; **R\*** =
required, but depends on the other workstream landing the single-vendor variant (amd-rdna4 for
AMD, nv-tp-pp for NV TP/PP); **L** = later (after the R set, perf phase or on demand); **—** =
out of scope or meaningless.

| Model class | Placement (different models per device) | DP NV / AMD / Mix | TP NV / AMD / Mix | PP NV / AMD / Mix | P/D NV / AMD / Mix (A→N, N→A) |
|---|---|---|---|---|---|
| Dense LLM | R | R / R\* / R | R\* / R\* / L | R\* / L / R | R / L / R, L |
| MoE LLM | R | R / R\* / R | R\* / R\* / — | R\* / L / R | L / L / R |
| GDN hybrid LLM | R | R / R\* / R | L / L / — | L / L / L | L / L / R (recurrent `State`) |
| TTS (LM + codec) | R | R / R\* / R | — | — | — |
| ASR (encoder + decoder) | R | R / R\* / R | — | — | L (encoder on one device, decoder on another) |
| VAD | R (any device or CPU) | R / R\* / R | — | — | — |

Notes:

* "Placement" covers co-serving: for example, LLM decode on `nv:*`, TTS and ASR on `amd:*`. That is
  the first mixed deliverable and needs only §2 and §3.
* P/D Mix A→N (prefill on `gfx1201`, decode on `sm_120a`) is the user's named case and the
  dense-LLM priority. N→A runs on the same code once A→N passes; it is listed L only for its gate
  run.
* Same-vendor NV P/D (`nv:0` → `nv:1`) is required because it is the exactness test of §5.4.
* MoE TP needs nv-tp-pp's expert sharding. MoE P/D uses the same KV image as dense. The model
  class changes nothing in the handoff.
* GDN P/D is the test of `State` entries. GDN PP needs per-stage recurrent state only, which stays
  local.

### 9.1 Gates per combination

| Class | Functional gate | Extra |
|---|---|---|
| LLM (all) | FP32-reference gate at c1 and c8 (c32 for promotion) against the same-concurrency peer capture of the same checkpoint | P/D same-variant exactness (§5.4); PP same-vendor vs whole: token-exact at c1 |
| TTS | the recipe's TTS gate (round-trip CER through the ASR gate) | — |
| ASR | WER/CER on the recipe's eval set within its threshold | realtime session routing across vendors |
| VAD | frame decisions identical to the reference program | — |

A mixed instance is an arm of its own: its gate is not inherited from its members' gates.

---

## 10. Phases

Each phase ends green, with unit tests, a hardware smoke on the box and its gates, and is merged
into the integration branch.

| Phase | Deliverable | Gate |
|---|---|---|
| **H1** | Dual-driver process: inventory, `DevId`, per-device backends, the coexistence gate (§2.1); multi-model placement across vendors (`Whole` on any device); `targets.json` multi-variant assets + dist list | every model in the box kit served from one process on its device, own gate passes; coexistence gate passes or the rank-worker fallback is taken |
| **H2** | Planner v2 + `--plan` file + `/v1/placement` + metrics; DP across vendors (Mix DP) | DP Mix: FP32 gate per rank vendor; router tests |
| **H3** | `kv_image` section (devgen) + `export_image`/`import_image` (CUDA first) + host staging regions + `Disagg` flow; same-variant NV P/D | exactness test bit-identical; FP32 gate |
| **H4** | AMD export/import + leader sampling for P/D first token; Mix P/D A→N, then N→A | FP32 gate c1/c8 on the Mix arm; dense first, then MoE |
| **H5** | Mixed PP on nv-tp-pp stage packets (stage backend per variant; `act_handoff` digest) | PP Mix vs FP32 gate; same-vendor PP token-exact |
| **H6** | `State` entries: GDN P/D; ASR encoder/decoder split if wanted | FP32 gate / WER |
| **H7** | Cross-vendor TP, only if §8's conditions hold for a model that needs it | FP32 gate |

**Perf phase (after the R set is green):**

* stream the KV image per prefill chunk, overlapped with the next chunk;
* device gather and scatter kernels for export and import;
* a cross-vendor large-BAR device-to-device probe;
* P/D policies: which requests go P/D, chosen by prompt length and load;
* pipeline micro-batching across stages;
* a shared canonical-image host tier.

Each item is measured against the same instance in `Whole` shape and against the peer, starting
at block level where a kernel is involved.

---

## 11. Seams needed from the other workstreams

| From | Seam |
|---|---|
| nv-tp-pp | (a) the per-rank backend trait in the rank-group serve, so a rank or stage can be HSA; (b) host-region registration behind a trait (CUDA host register / HSA memory lock) that staging regions reuse, plus named regions beyond the single hand-off buffer, or a sibling region type built on the same trait; (c) the stage manifest carries each stage's target ISA and an `act_handoff` contract (dtype, shape, residual state) that devgen digests; (d) leader sampling exposed as an engine-independent call the P/D first token can use |
| amd-rdna4 | (a) the `gfx1201` ISA level and hwspec entry, and `from_probe` resolving the R9700; (b) `gfx1201` variants of the box's models; (c) slotted serving with prefix cache on (as on gfx950); (d) a last-row logits return for host sampling; (e) span D2H/H2D on the engine for export and import |
| bring-up | a lease that holds devices of both vendors (gpulease), and gpuq submitting a both-vendor job |
