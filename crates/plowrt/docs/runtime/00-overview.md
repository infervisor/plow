# plowrt — runtime overview

`plowrt` is the host runtime for plow. `plowc` compiles a model into per-bucket
**assets** (`.pkt` packet streams, `.map.json` address maps, `weights.json`,
sidecars); `plowrt` loads them, drives the device, coordinates the
persistent-kernel executors, and serves an OpenAI-compatible API.

The authoritative per-subsystem documentation is the **module rustdoc** — each
`src/**/*.rs` opens with the design for that subsystem and cites the compiler
contract it implements. This file is the map; the topic files alongside it go
deep on the two areas most often asked about (counter memory placement, static
binary + drivers).

## Performance model (read first)

The runtime is latency-critical; per-token cost compounds. The rules the code
follows:

- **Nothing allocates, locks, or syscalls on the per-token / per-packet path.**
  Counters and queues are lock-free atomics; buffers are preallocated and reused.
- **Counter pool** (`exec/counters.rs`): cache-line-padded `AtomicU64`,
  `Acquire`/`Release` ordering, `#[inline]` load/add. See `counter-memory.md` for
  where the cells physically live.
- **Packet queue** (`exec/queue.rs`): bounded SPSC ring, power-of-two mask,
  head/tail on separate cache lines, no allocation after construction.
- **Cold paths** (asset load, bringup, config, registry) may allocate and lock
  freely (`parking_lot`, `serde`).
- Fast maps (`rustc-hash`), inline small vecs (`smallvec`), POD casts
  (`bytemuck`) on the warm paths.

## Subsystem map

| § | Module | What it does |
|---|--------|--------------|
| A | `asset/` | Load + validate a compiled `ModelBundle` (mmap `.pkt`, `Program::decode`, deserialize sidecars). |
| B | `device/` | `Backend` trait; real `cpu` reference backend; `cuda`/`rocm` dlopen backends (features). |
| C | `memory/` | `AddressSpace` (one arena/segment, slot rebase), `weights`, `kv` (paged blocks). |
| M | `memory/dma.rs` | Host↔device DMA plane: pinned pool, role streams, transfer policy. |
| N | `memory/streamer.rs` | Weights/KV tiered streamer (offload, prefetch, HBM reclaim). |
| D | `exec/` | Counter pool, packet queue, OOB channel, indirection table, executor set. |
| OOB | `exec/oob.rs` | Bidirectional side-channel: host→exec control, exec→host events. |
| E | `sched/` | Per-iteration bucket select + dispatch + completion. |
| I | `sched/admission.rs`, `batching.rs` | Queuing-theory admission (λ/μ/ρ) + bucket/pkt-stream selection. |
| J | `exec/health.rs` | Bringup verify; optional counter-space monitor (deadlock + progress). |
| K | `obs/` | Metrics (`/metrics`) + per-task timeline (Chrome trace). |
| F | `orch/` | Slug→bundle registry, router, multimodal pipeline; MoE, disagg, speculative. |
| G | `serve/` | axum OpenAI API: chat (stream + non-stream), models, healthz, metrics. |
| H | `text/` | Tokenizer, sampler (greedy/top-k/top-p/min-p), guided decoding. |

## Data flow (one request)

```
POST /v1/chat/completions {model, messages}
  → serve::chat            resolve slug → orch::Registry
  → text::tokenizer        prompt → token ids
  → sched::batching        (phase, batch, seq) → compiled bucket → .pkt stream
  → exec::ExecutorSet      counter pool + run stream (CPU: interpret; GPU: enqueue + poll milestone)
  → text::sample           logits → token (device-side by default; host fallback for logprobs/guided)
  → serve::stream          SSE chat.completion.chunk deltas → [DONE]
```

## Status

Real and exercised by tests: asset loading, address map + KV paging, counter
pool, packet queue, OOB channel, CPU reference interpreter (counter-gated,
deadlock-aware), counter-space monitor, admission/batching math, registry +
pipeline, the OpenAI API surface, sampler.

Typed seams (documented TODO): GPU op numerics (golden bodies), the CUDA/HIP
driver entry points past `cuInit`/`hipInit` (dlopen is wired), safetensors
weight bytes (`hub` feature), and the production lock-free OOB rings.
