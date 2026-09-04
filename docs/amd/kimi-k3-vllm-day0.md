# vLLM's Kimi-K3 day-0 post — targets, and five ideas that apply to GLM-5.2 today

> Deprecated performance reference (2026-09-02). The figures below are historical,
> cross-hardware research and must not be used as the Kimi-K3 vLLM baseline. The
> same-box 8×MI355X baseline is `perf-data/kimi-k3-vllm-mi355x-baseline.md`.

Source: <https://vllm.ai/blog/2026-07-27-k3> — "Kimi K3 Is Here: Efficient Day-0 Support on vLLM",
2026-07-27, vLLM Team and Inferact. Fetched 2026-07-28.

Companion to `docs/amd/kimi-k3-atom-day0.md` (AMD/ATOM). Where AMD published **no** performance
numbers, vLLM published plenty — but read the hardware line carefully.

## 0. SCOPE DECISION (user, 2026-07-28): DO NOT RUN vLLM FOR K3.

**K3 is a plow-only bring-up. We do not stand up vLLM for it.** Their published TP8 batch-1 figure —
**111 tok/s = 9.01 ms/token** — is the **TARGET REFERENCE** we aim at, taken as given from this post.

This is a deliberate departure from §0-BENCH's "every plow-vs-vLLM number comes from `vllm bench
serve` against a plowrt endpoint", and the honesty rule that replaces it is simple:

> **9.01 ms/token is an ASPIRATION TARGET on DIFFERENT HARDWARE (GB300 NVL72), not a head-to-head
> result.** Any K3 number we report says so in the same sentence. We never write "plow beats vLLM on
> K3" off this comparison — only "plow reaches X ms/token against vLLM's published 9.01 on Blackwell".

§0-BENCH still governs GLM-5.2 unchanged: that comparison is on one box, both engines, same client.

Why this is reasonable rather than a dodge: nobody — not vLLM, not AMD — has published a K3 number
on MI355X, and vLLM's own ROCm path is bring-up ("broader tuning on the roadmap"). A vLLM-on-MI355X
run we produced ourselves would be measuring an untuned path and would tell us less than the
arithmetic below already does.

## 1. The published numbers, and what they are NOT

| config | batch 1, per user |
|---|--:|
| TP8 | **111 tok/s** (9.01 ms/token) |
| TP16 | **118 tok/s** (8.47 ms/token) |
| TP8 + DSpark speculative | 331 tok/s |
| TP16 + DSpark speculative | **370 tok/s** (3.14x) |

**These are GB300 NVL72 — NVIDIA Blackwell, not MI355X.** vLLM's own FAQ: *"Does vLLM support Kimi
K3 on AMD GPUs? Yes. **ROCm support ships at launch, with broader tuning on the roadmap.**"* and the
acknowledgements thank *"AMD for ROCm bring-up"* while thanking *"NVIDIA for the fused KDA decode,
KDA prefill, and Attention Residual kernels"*.

So on MI355X:
- **vLLM has published no K3 number.** AMD has published no K3 number. **Nobody has.**
- The kernels that produce 111 tok/s are named as NVIDIA contributions. The ROCm path is bring-up.

That is the same shape as GLM-5.2, where vLLM's gfx950 run has no tuned AITER config and needs ~57
min of JIT — and it means **any K3 number we produce on MI355X should be compared against a
vLLM-on-MI355X number we measure ourselves**, not against 111 tok/s on Blackwell. Quoting their
Blackwell figure as "the bar" would be dishonest in both directions.

Recommended config is **8x MI355X or 8x B300, TP8**, with `--enable-prefix-caching` in the
quick-start command.

## 2. Architecture — confirms our reading, and sharpens two things

Everything our agents derived from the checkpoint holds. Two refinements:

- **Block AttnRes attends over "up to eight cached block representations plus the current
  within-block residual"** — so **<=9 sources**, exactly our spec's number, and vLLM implements it as
  an **online softmax across model DEPTH rather than sequence position**, fused into one kernel with
  the residual update at the input and optional RMSNorm on the output. That is a useful shape for
  our AttnRes work: it is FlashAttention's algorithm on a different axis.
- **"A single layer's KDA state is roughly equivalent to the MLA cache for a few thousand tokens."**
  Consistent with our 6.5625 MiB/layer/seq and with AMD's table (69 KDA layers = 0.054 GB of state
  vs 24 MLA layers = 14.496 GB at 1M).

## 3. FIVE THINGS WE CAN USE — three of them on GLM-5.2 *today*

### 3a. LatentMoE tail fusion — applies to GLM-5.2 now, and it is a COLLECTIVE restructuring
> "At the end of LatentMoE, the reduced activation from routed experts must be normalized with
> RMSNorm and up-projected before it is added to the shared-expert output. In the normal TP case,
> this requires **two all-reduces** ... vLLM instead performs **reduce-scatter on the shared experts
> and keeps all-reduce on the routed experts** because their activations need to be normalized. The
> replicated routed-expert activation then performs matrix multiplication with the up-projection in
> a **column-parallel** fashion and is added elementwise to the already-sharded shared-expert
> output. Finally, the results are **all-gathered** onto each rank using broadcast."
> — **~20% latency reduction in that step, ~7-8% end-to-end.**

**This is directly relevant to §6e-0 and to the 48% gate-stall work.** GLM-5.2 also pays two
collectives per layer (156 total, all on the critical path), and we measured the whole set at 3.84
ms. A restructuring that removes one of the two, and replaces a replicated up-projection with a
column-parallel one, is exactly the class of change we have not tried — we have only tried making
the *existing* collectives cheaper.

### 3b. skinnyGEMM — INDEPENDENT CONFIRMATION of plow's GEMV design
> "we replace generic BF16 GEMM ... with our own **skinnyGEMM**. Generic cuBLAS kernels do not
> achieve the best performance here because they are optimized for more general shapes. In the
> kernel, we **bypass shared-memory data staging, load activations and weights directly into
> registers, and use CUDA Core FMA instructions** ... This avoids the heavy TMA and Tensor Core setup
> phase." — 8-100% kernel speedup, **~10% end-to-end in small-batch**.

That is plow's GEMV thesis, arrived at independently. Our `Gemv` family already runs at **83-106% of
the 6200 GB/s ceiling** and `lm_head` at 94-106%, so this is a place where plow is *already* doing
the right thing — worth recording so nobody "improves" it toward a tensor-core path.

### 3c. Fused KDA decode — a DIFFERENT choice from ours, and worth understanding before we judge it
> "vLLM fuses the post-projection decode path — from the causal convolutions through gated RMSNorm —
> into **a single specialized CUDA kernel**. The kernel updates the convolution and recurrent states
> in place and writes the normalized output directly, avoiding intermediate tensors, repeated state
> traffic, and per-operation launch overhead."

**We deliberately did the opposite**: `docs/kimi-k3-kda.md` decomposes KDA into **14 packets**,
because a monolithic op is how `Mamba2Scan` died and because the register objection dissolves once
the state is a declared HBM tensor. Our decomposition measured **zero extra VGPRs, 256/256 blocks,
100% occupancy**, and the one-layer gate passes.

The two are not obviously in conflict: vLLM pays *per-operation launch overhead* because each op is
a CUDA kernel launch, whereas in plow a packet is not a launch — the whole decode is **one dispatch**
and packets are counter-gated work items. **The fusion argument that motivates their kernel is an
argument plow's architecture already answers.** Worth stating explicitly rather than assuming we are
behind. What we should still steal is the *in-place state update* and *no intermediate tensors*.

### 3d. FlashKDA is open source and is the reference to check numerics against
> "Moonshot AI first released **FlashKDA**, a high-performance CUTLASS implementation of KDA ...
> Shikhar Mishra then optimized the kernels for H100 and published **Flash-Flash-KDA**."

Our KDA gate currently checks against `fla.ops.kda`. FlashKDA is a second, independent
implementation and a better oracle for the prefill scan we have not built yet.

### 3e. Prefix caching is ON in their quick-start — which settles §21
`--enable-prefix-caching` appears in the recommended command and there is an FAQ entry for it. Our
harness never passes `--no-enable-prefix-caching`, so **every plow-vs-vLLM number so far had vLLM
caching and plow not**. Bounded at ~2-3% for `random` 1024-token prompts (only the chat template is
shared), but it should be disabled for a clean comparison, or matched.

## 4. Accuracy bar, if we want one

vLLM validated K3 through a served OpenAI endpoint: **GSM8K 0.976, GPQA-Diamond 0.939, OCRBench
0.889, MMMU Pro Vision 0.818**. AMD's ATOM post claims GSM8K 5-shot over all 1,319 samples at 16K
length. Their caveat is worth keeping: *"Kimi K3 thinks a lot before it answers. A low score is more
often a truncated answer than a wrong one."*

## 5. What this does NOT give us

- No MI355X numbers, from anyone.
- The KDA prefill scan, PD disaggregation, speculative decoding (DSpark), and hybrid prefix caching
  over recurrent state are all things vLLM has and plow does not. The **hybrid prefix caching** work
  is the deepest: KDA state is updated in place, so a snapshot must be copied at a chosen boundary,
  and they added interval-based and Marconi-style ("cache on the second hit") retention policies.
  plow has no prefix caching on AMD at all.
