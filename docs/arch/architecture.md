# Plow Architecture — Index

> **Scope:** This page is the landing point for the `docs/arch/` set. Plow (née
> Infervisor) is an ahead-of-time compiler and persistent-kernel runtime for
> transformer inference across heterogeneous accelerators (NVIDIA, AMD, CPU,
> DPU). The compiler (`plowc`) lowers a model and a hardware topology into
> per-executor `.pkt` streams plus JSON sidecars; the runtime (`plowrt`) loads
> those artifacts and executes them on persistent-kernel interpreters that
> coordinate through a shared atomic-counter pool, keeping the CPU off the
> critical path.

The numbered chapters below are the single source of truth. This file used to
duplicate them and is now an index only.

---

## Chapters

| # | Document | What it covers |
|---|----------|----------------|
| [00](00-overview.md) | **Overview & Index** | System identity, design philosophy, boundaries, data flow, key decisions |
| [01](01-compiler-pipeline.md) | **Compiler Pipeline** | Egglog rewriting half + list-scheduling half and the interface between them |
| [02](02-tile-graph.md) | **Tile Dependency Graph** | The central IR (DmaIn/Compute/DmaOut DAG) that connects rewriting to scheduling |
| [03](03-scheduler.md) | **Scheduler** | Machine model, interval-conflict list scheduling, and post-schedule passes |
| [04](04-packet-abi.md) | **Packet ABI** | Wire format, opcode encoding, record/body types, resource kinds |
| [05](05-counter-system.md) | **Counter Coordination** | Scope tiers, counter determination, wavefront clustering |
| [06](06-runtime.md) | **Runtime** | Execution model, interpreter loop, memory regions, KV cache, backends |
| [07](07-cost-model.md) | **Cost Model** | Hardware abstraction, tile candidate enumeration and selection |
| [08](08-formal-verification.md) | **Formal Verification** | Lean 4 checkpoint architecture (A–F) and universal theorems |
| [09](09-multi-gpu.md) | **Multi-GPU & Parallelism** | TP/PP/EP/DP strategies and collective lowering |
| [10](10-implementation-status.md) | **Implementation Status** | What exists in the checkout vs. target design, crate by crate |
| [11](11-tuning-coverage.md) | **Tuning Coverage** | Per operator family: distinct kernels, knobs, oracles, blockers |
| [12](12-using-the-tuner.md) | **Using the Tuner** | `plowc tune`, reading the output, taking a measurement |
| [13](13-prefill-chunking.md) | **Prefill Chunking** | Bucket ladder, the ragged tail, ragged-M |
| [14](14-amd-arch-divergence.md) | **AMD Arch Divergence** | gfx942 vs gfx950: one source tree, what forks, the tripwire |
| [15](15-dispatch-audit-and-knob-manifest.md) | **Dispatch Audit & Knob Manifest** | The dispatch audit and the knob manifest |
| [16](16-object-contract.md) | **Object Contract** | What a packet requires of its device objects |
| [17](17-unified-token-batch.md) | **Unified Token Batch** | Decode and prefill rows in one launch (always on) |
| [18](18-serving-metrics.md) | **Serving Metrics** | Model-scoped Prometheus metrics, vLLM compatibility, request/tick accounting and lifecycle |
| [19](19-packet-extensions.md) | **Packet Extensions** | Adding prefill buckets, decode rungs, packed siblings and token-batch bodies to a packet that is already compiled, without re-emitting it |
| [20](20-asr-pipelines.md) | **ASR Packet Pipelines** | Qwen causal and Nemotron RNNT compilation, runtime, batching and serving |
| [21](21-decode-context-parallelism.md) | **Decode Context Parallelism** | DCP design |
| [22](22-kv-cache.md) | **KV Cache** | Head-major ring layout, chunk-sized rings, the write and read paths, fp8 KV, slots, the VMM allocator and the prefix cache |
| [26](26-kv-block-store.md) | **Prefix Cache Retention and the KV Block Store (design)** | What moves the production hit rate (capacity tiers, session-end demotion; the retention simulation against vLLM 0.29) and the content-addressed row-class store that unifies NVIDIA/AMD/CPU and adds recurrent state |
| [23](23-jit-rungs.md) | **JIT Rungs (future plan)** | Building a new decode rung or prefill bucket at runtime on prebuilt cubins: measured emit cost, what is packet-only, the pairing and verifier guardrails, the staged path from `plowc extend` to a hot-add |
| [24](24-tts-pipelines.md) | **TTS Packet Pipelines** | TTS compilation and serving |
| [25](25-dspark-pending.md) | **DSpark Block Drafter (pending)** | Pending support |
| [27](27-kv-zero-copy.md) | **Zero-copy KV** | Shared pages for live and cached rows; measured results |
| [28](28-library-attention.md) | **Library Attention** | Library attention kernels inside plowrt, per rung |
| [29](29-techniques.md) | **Techniques** | Per technique: what the op computes, its packet ABI, kernels and objects, KV/state class, backends |

---

## Techniques index

Models share techniques, so the runtime is documented per technique ([29](29-techniques.md)); models are examples there.

| Family | Techniques |
|---|---|
| Attention | [Causal GQA flash attention (prefill, split-KV decode, FP8 KV)](29-techniques.md#causal-gqa-flash-attention-prefill-split-kv-decode-fp8-kv), [Sliding-window attention and hybrid sliding/global layouts](29-techniques.md#sliding-window-attention-and-hybrid-slidingglobal-layouts), [KV-shared layers (KV-shared tail)](29-techniques.md#kv-shared-layers-kv-shared-tail), [Attention sinks and logit softcap](29-techniques.md#attention-sinks-and-logit-softcap), [Library attention route](29-techniques.md#library-attention-route) |
| Latent, sparse and memory attention; residual-stream variants | [Multi-head latent attention (MLA)](29-techniques.md#multi-head-latent-attention-mla), [Sparse attention with a lightning indexer (DSA)](29-techniques.md#sparse-attention-with-a-lightning-indexer-dsa), [Compressed KV attention (learned pooling, CSA)](29-techniques.md#compressed-kv-attention-learned-pooling-csa), [Engram conditional memory (hashed n-gram lookup)](29-techniques.md#engram-conditional-memory-hashed-n-gram-lookup), [Hyper-connections (mHC: multi-stream residual)](29-techniques.md#hyper-connections-mhc-multi-stream-residual), [Attention residuals (AttnRes: softmax mix over block residuals)](29-techniques.md#attention-residuals-attnres-softmax-mix-over-block-residuals) |
| Recurrent mixers | [Recurrent state (shared contract for the mixers below)](29-techniques.md#recurrent-state-shared-contract-for-the-mixers-below), [Short causal depthwise convolution (conv history)](29-techniques.md#short-causal-depthwise-convolution-conv-history), [Gated DeltaNet (GDN)](29-techniques.md#gated-deltanet-gdn), [Kimi Delta Attention (KDA)](29-techniques.md#kimi-delta-attention-kda), [Mamba-2 selective scan (SSD)](29-techniques.md#mamba-2-selective-scan-ssd) |
| Mixture of experts | [MoE routing (router + top-k)](29-techniques.md#moe-routing-router--top-k), [MoE experts: GLU / down, shared experts, combine](29-techniques.md#moe-experts-glu--down-shared-experts-combine), [Expert parallelism](29-techniques.md#expert-parallelism), [Muse-style gated residual and norms](29-techniques.md#muse-style-gated-residual-and-norms) |
| Precision and encoding | [FP8 weights, BF16 activations (W8A16)](29-techniques.md#fp8-weights-bf16-activations-w8a16), [FP8 weights and activations (W8A8)](29-techniques.md#fp8-weights-and-activations-w8a8), [Block-scaled FP8 (128x128 and 32x32 grids)](29-techniques.md#block-scaled-fp8-128x128-and-32x32-grids), [MXFP4](29-techniques.md#mxfp4), [FP8 KV cache](29-techniques.md#fp8-kv-cache), [Library GEMM routes (cuBLASLt)](29-techniques.md#library-gemm-routes-cublaslt) |
| Multi-token drafting and speculative decode | [Chain drafter (MTP assistant)](29-techniques.md#chain-drafter-mtp-assistant), [Block drafter (DSpark)](29-techniques.md#block-drafter-dspark), [Verify and acceptance (runtime)](29-techniques.md#verify-and-acceptance-runtime) |
| Media and speech inputs | [Multimodal soft tokens (vision and audio encoders)](29-techniques.md#multimodal-soft-tokens-vision-and-audio-encoders), [Encoder overlay rows](29-techniques.md#encoder-overlay-rows), [Speech front-ends and codecs (pointer)](29-techniques.md#speech-front-ends-and-codecs-pointer) |

## Workspace at a glance

Cargo workspace, 13 crates under `crates/` (Rust) plus the `tools/bench` member, a C/CUDA/HIP device runtime under
`runtime/` and Lean 4 proofs under `lean-plow/`:

- **Frontend / IR:** `nn-graph` (symbolic operator graph IR, model hub; folds the
  former `frontend`)
- **Compiler core:** `rewrite`, `costmodel`, `schedule`, `packet`, `plowc`
- **Device-blob emitter:** `devgen` — a **non-optional** dependency of `plowc`, and the
  path every shipping GPU asset is emitted through (`plowc --emit devblob`)
- **Hardware / kernel / tuning registries:** `hwspec`, `kernelcaps`, `tunedb`
- **Verification:** `lean_verify`
- **Shared schema:** `plow-asset` (compiler↔runtime boundary types)
- **Runtime host:** `plowrt` (serve, simulate, mux, executor pool)

> [!IMPORTANT]
> **`devgen`, not `rewrite` + `schedule`, is what emits a shipping packet.** devgen does not
> depend on `rewrite` or `schedule` (`rewrite` is a dev-dependency only), and every device
> kernel and fused opcode in a shipped packet is hand-written in `devgen`. Since 2026-09-14
> (`PLOW_EMIT_REWRITE`, default on, e64b6d840) `plowc` extracts the egglog fused graph from the
> checkpoint config and passes its fused sites to devgen, which emits a hand-written fused
> opcode only where the extracted graph fused that site (falling back to the hand fusions when
> no rewrite exists for the architecture). The scheduling half (`schedule`) is still not on the
> devblob path. See [01 — Compiler Pipeline](01-compiler-pipeline.md)'s opening warning.

See [10 — Implementation Status](10-implementation-status.md) for the full
crate-by-crate breakdown, the build profile, and the device-ISA reconciliation.
