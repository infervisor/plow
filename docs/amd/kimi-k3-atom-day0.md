# AMD's Kimi-K3 Day-0 post — what it gives us, and what it does not

Source: <https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-k3-on-amd-instinct-gpus.html>
"Day 0 Kimi-K3 Inference Deployment with ATOM on AMD Instinct MI355X GPUs", fetched 2026-07-28.

## 1. THERE IS NO PERFORMANCE NUMBER TO BEAT. Read this before planning against it.

AMD states it explicitly:

> "This post does not make claims about throughput, time to first token (TTFT), time per output
> token (TPOT), or kernel efficiency. HBM optimization, TP8 collectives, MXFP4 Grouped MoE, KDA
> prefill and decode, MLA, and 1M-token context optimization will be addressed in dedicated
> performance-tuning posts."

and

> "The goal is not to pursue peak performance, but to answer three practical Day 0 questions: why
> the Kimi-K3 weights fit on these GPUs, how the weights are distributed under TP8, and how to
> bring up the model quickly with ATOM and run a minimal correctness check."

So unlike Kimi-K2.7 (where AMD published 5,369.6 tok/s/GPU @ conc 128 and 116.4 tok/s/user @ conc 4),
**there is no published K3 figure.** This changes the framing: K3 is not a catch-up target, it is an
open one. The only validation AMD claims is **GSM8K 5-shot, all 1,319 samples, MI355X TP8, 16K max
model length** — a correctness bar, and a reasonable one for us to aim at first.

## 2. It independently confirms four things our agents found from the checkpoint

Worth recording because they were derived here from config/tensors alone, and now have a second source:

| our finding | AMD's wording |
|---|---|
| 93 layers = 69 KDA + 24 MLA | "93 layers: 69 KDA layers and 24 Gated MLA layers" |
| tail is `KKK MM`, **not** a clean 3:1 motif | "interleaved KDA x 3 -> MLA x 1 pattern, **with one additional MLA layer at the end**" |
| 497,220 tensors | "safetensors headers of 497,220 tensors" |
| routed experts run at 3584, not hidden 7168 | "Stable LatentMoE first projects the 7168-dimensional hidden state down to 3584 dimensions before running the expert computation" |
| `attn_res_block_size = 12` | "Stores one block residual every 12 layers" |

The layer-pattern one matters most: a naive `i % 4 == 3` rule gets the last block wrong, and two
independent derivations now say so.

## 3. New information we did not have

**Official names.** It is **Gated** MLA (not plain MLA), **Stable LatentMoE**, and **AttnRes** —
worth using in code comments so future readers can find AMD's material.

**AMD's TP8 placement rules** (directly comparable to `crates/plowrt/src/asset/shard.rs`):
- attention heads sharded across ranks;
- Dense MLP + Shared Expert gate/up **column** parallel, down **row** parallel; routed expert
  w1/w2/w3 likewise;
- **"Every rank retains all 896 expert IDs; TP shards each expert's matrices rather than
  partitioning the expert IDs."** — i.e. AMD runs **TP, not EP**, for the experts. Note plow has an
  EP mode (`GLM_EP=1`, whole experts per rank) that is still unmeasured; this is a data point that
  the obvious production choice is TP-sharded experts, not expert-partitioned.
- **replicated**: MLA `q_a`/`kv_a`, KDA `f_a`, LatentMoE down/up, Norm, router, AttnRes score
  projections;
- token embedding and LM head sharded along **vocab** — which is exactly the `GLM_SHARD_HEAD=1` +
  `XArgmaxFin` work we just landed for GLM.
- text-only service does **not** load `vision_tower` / `mm_projector`.

**Weight distribution at TP8** (their table; full checkpoint 1.5609 TB, 2.78T params):

| category | full | TP8 per GPU |
|---|--:|--:|
| Routed Expert packed values + scales | 1446.456 GB | 180.807 GB |
| KDA Attention GEMM | 61.214 GB | 7.763 GB |
| Shared Expert | 24.310 GB | 3.039 GB |
| MLA Attention GEMM | 11.145 GB | 2.029 GB |
| Dense MLP | 1.453 GB | 0.182 GB |

Totals: **190.974 GiB weights/GPU**, +14.427 GiB for a 1M-token context = **205.401 GiB of 288 GiB
(71.3%)**, leaving ~82.6 GiB for everything they did not model.

**Runtime state at 1M tokens, TP8** — the number that makes the hybrid worth having:

| state | formula | layers | TP8/GPU |
|---|---|--:|--:|
| MLA latent KV | `1048576 x (512+64) x 2 B/layer` | 24 | 14.496 GB |
| KDA SSM state | `(96/TP) x 128 x 128 x 2 B/layer` | 69 | 0.054 GB |
| KDA conv state | `3 x (12288/TP) x (4-1) x 2 B/layer` | 69 | 0.002 GB |
| AttnRes 8K chunk | `8192 x ceil(93/12) x 7168 x 2 B` | 93 | 0.940 GB |

**69 layers of KDA cost 0.054 GB; 24 layers of MLA cost 14.496 GB.** That is the whole argument for
the architecture, and it matches our own `docs/kimi-k3-kda.md` conclusion (fixed-size state, 3.81x
better than all-MLA at 1M) from an independent direction.

**Their software stack is ATOM**, not vLLM or SGLang, with:
```
export AITER_USE_GROUPED_GEMM=0
export AITER_FLYDSL_FORCE=1
export AITER_FORCE_GFX1250=0
… --kv_cache_dtype fp8 -tp 8
```
`AITER_USE_GROUPED_GEMM=0` is worth noting: AMD turns AITER's grouped GEMM **off** for K3 day-0.

## 4. Two internal inconsistencies in their article — do not copy either blindly

1. The prose says "**FP32** KDA SSM states" but the formula uses **2 bytes** per element
   (`... x 128 x 128 x 2 bytes`), which is bf16. Our `docs/kimi-k3-kda.md` specifies **f32** state.
   The dtype changes the state size by 2x and is worth settling from the reference implementation
   (`fla.ops.kda`) rather than from either document.
2. The prose says "an **FP8** latent KV cache" but the MLA latent KV row also uses **2 bytes** and is
   labelled BF16 in the same table.

Neither affects our design; both are reasons to trust the checkpoint and the reference code over
prose, which is the same discipline that settled GLM's `qk_rope_head_dim` and rope convention.

## 5. What this changes for us

- **No K3 performance target exists yet.** Getting K3 running with real numbers would put us ahead of
  AMD's own published position, not behind it. Their follow-up posts are the eventual bar.
- **GSM8K 5-shot / 1319 samples / 16K length is the correctness bar to aim at**, and it is concrete.
- **Their TP8 placement rules are a free cross-check** for our shard classifier once K3 emits.
- **Experts are TP-sharded, not EP-partitioned**, in the one production config we can see.
