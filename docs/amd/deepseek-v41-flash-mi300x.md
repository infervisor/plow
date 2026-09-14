# DeepSeek-V4.1-Flash on MI300X — bringup state

Architecture delta, checkpoint contract, and the stage-by-stage gap for
`deepseek-ai/DeepSeek-V4.1-Flash`, which plow can now PARSE but not build.

Companion to [`deepseek-v4-mi300x.md`](deepseek-v4-mi300x.md), which covers
`DeepSeek-V4-Flash-0731`. The two are different architectures; §1 is the list
of ways in which reusing the V4 work would produce a plausible wrong model.

**Sources.** Every structural claim below is from the shipped checkpoint or
its reference implementation, not from the model card:

```
/workspace/models/DeepSeek-V4.1-Flash/
├── config.json                      served HF config
├── model.safetensors.index.json     96 085 tensors, 48 shards, 475.3 GiB
└── inference/{model.py,kernel.py,convert.py,engram.py,generate.py,config.json}
```

plow citations are repo-relative.

---

## 0. What the checkpoint actually contains

`model.safetensors.index.json`, canonicalized over layer/expert indices. The
counts are the contract — several of them are the only way to tell V4.1's
layer roles apart, because `config.json` alone is ambiguous.

| tensor | n | what the count proves |
|---|---|---|
| `layers.{L}.attn.wq_a.weight` | 40 | 40 backbone layers |
| `layers.{L}.attn.compressor.wkv.weight` | **4** | only 4 layers compress — `kv_source_layer_ids` |
| `layers.{L}.attn.compressor.wgate.weight` | **3** | the ratio-1 compressor has no softmax gate |
| `layers.{L}.attn.indexer.wq_b.weight` | **8** | `index_source_layer_ids` |
| `layers.{L}.attn.indexer.wk.weight` | **4** | only KV sources OWN index keys |
| `layers.{L}.engram.embed.weight` | 2 | `engram_layer_ids = [1, 14]` |
| `layers.{L}.ffn.experts.{E}.w{1,2,3}` | 15 360 | 40 x 384 experts |
| `mtp.{M}.ffn.experts.{E}.w{1,2,3}` | 384 | 3 x 128 — DSpark has its OWN narrower MoE |
| `mtp.{M}.markov_head.*`, `confidence_head.*`, `main_proj.*` | 1 each | only ONE of the 3 MTP blocks carries them |

Note the naming: there is **no `model.` prefix** (`layers.0…`, `embed.weight`,
`norm.weight`, `head.weight`), expert projections are `w1`/`w3`/`w2` (gate/up/
down), and the vision tower is `vision.*` with an `aligner.*` projector.

## 1. Why V4.1 is not V4

Each of these silently produces a plausible wrong model if V4's handling is
reused. This is the reason `DeepSeekV41Config` is a separate struct rather
than a relaxation of `DeepSeekV4Config`.

1. **CSA2 SHARES one KV cache.** In V4, a nonzero `compress_ratios[l]` means
   layer `l` runs its own compressor. In V4.1 it names the cache layer `l`
   *reads*. Only `kv_source_layer_ids = [2, 8, 14, 20]` compress; the other 36
   layers read that cache. Reading `compress_ratios` V4's way expects 38
   compressors and finds 4 — `inference/model.py:619` says it outright ("does
   not mean the layer compresses its own KV").
2. **`compress_ratio == 1` is a MODE, not an error.** V4's validator rejects
   it ("use 0 to mean no compressor"). All 20 decoder layers run at ratio 1 —
   a per-token projection with no softmax pooling
   (`inference/model.py:446,458`), which is exactly why there are 4
   `compressor.wkv` but 3 `compressor.wgate`. V4's validator would reject the
   released checkpoint.
3. **The indexer is two-level.** 8 indexers, of which only the 4 KV sources
   own `wk`/`k_norm` (`inference/model.py:517`). From
   `candidate_source_layer_id = 20` onward an indexer scores only inside the
   2048 candidate blocks of 8 positions that layer selected.
4. **Engram conditional memory.** Layers 1 and 14 carry n-gram lookup tables
   of 384 006 168 and 384 016 682 fp8 rows — ~196 B parameters, the bulk of
   the checkpoint. Nothing in V4 resembles it.
5. **mHC is single-pass.** Each sublayer's `hc_mixes` produces the mix used by
   the *next* sublayer, not its own (`inference/model.py:968-995`). A V4-style
   same-sublayer mix is a different function.
6. **Block-FP8 at a `[32,32]` grid**, against V4's `[128,128]` — 16x the
   scales. `quantization_config` also moves `expert_dtype` inside and drops
   `fmt`, so neither struct deserializes the other's document.
7. **A causal encoder/decoder split.** Layers 0-19 are the encoder (ratio 2),
   20-39 the decoder (ratio 1); the decoder's global KV is projected from
   encoder hidden states rather than produced per layer.

## 2. Geometry

From `config.json` `text_config`, cross-checked against `inference/config.json`.

| | |
|---|---|
| layers / hidden / heads | 40 / 5120 / 64 |
| head_dim / qk_rope | 512 / 64 (nope 448) |
| num_key_value_heads | **1** — the 512-wide latent is REPLICATED under TP, not sharded |
| q_lora_rank | 1280 |
| o_groups x o_lora_rank | 8 x 1024 — caps clean TP at 8 |
| MoE | 384 routed + 1 shared, top-6, inter 2304, sqrtsoftplus / noaux_tc |
| sliding_window | 128 (SWA Bounded Replay) |
| index | 32 heads x 128, top-512 |
| rope | theta 10000, YaRN factor 16 over 65536 → 1 048 576; compressor theta 160000 |
| rms_norm_eps | **1e-20** |
| quant | block-FP8 e4m3 `[32,32]` ue8m0 scales; routed experts MXFP4 |
| DSpark | 3 MTP blocks, block size 5, markov rank 256, own 128-expert top-3 MoE |
| vision | 32-layer ViT, 1024 wide, patch 14, downsample 3 |

`compress_ratios` has **43** entries = 40 layers + 3 DSpark blocks.
`num_nextn_predict_layers` (3) counts speculative ITERATIONS and is not what
reconciles the length — the same trap as V4.

## 3. Stage state against `docs/bringup`

| Stage | Gate | State |
|---|---|---|
| 1. nn-graph | graph builds, shapes infer | **partial** — config parses and validates (`crates/nn-graph/src/models/config/deepseek_v41.rs`, 15 tests incl. 7 negative controls). No builder: `build_graph` refuses with `unimplemented()`. |
| 2. egglog rewrite | saturation, fired rules sound | not started (needs a graph) |
| 3. lean verify | checkpoints A-G certify | not started |
| 4. kernel tuning | hot kernels at roofline | not started |
| 5. single-block sweep | block matches oracle | not started |
| 6. runtime opt | TTFT/TPOT at target | not started |
| 7. perf campaign | battery + written results | not started |

### What Stage 1b needs

The builder is blocked on IR, not on plumbing. Reusing `dsa_indexer` /
`dsa_attention` covers the single-level V3.2-style index but not CSA2's shared
cache or the candidate-block level. New ops needed, in dependency order:

1. **mHC** — the residual stream is `[B, S, hc_mult, D]`, mixed by a
   Sinkhorn-normalized matrix. Expressible with existing primitives only by
   unrolling 20 Sinkhorn iterations x 2 sublayers x 40 layers (~3200 extra
   nodes), which no kernel would fuse. Precedent for a dedicated op:
   `Op::BlockResidual` (Kimi-K3's AttnRes).
2. **CSA2 compressor + shared cache read** — one cache written by 4 layers,
   read by 40, at two ratios.
3. **Two-level indexer** — candidate-block selection then in-block scoring.
4. **Engram** — a 384M-row fp8 gathered lookup with its own q/k weighting.

DSpark is optional for a first bringup: `generate.py` never calls
`forward_spec`, so base output is bit-exact without it.

## 4. Why there is no V4.1 reference number yet

Nothing on this box can serve V4.1:

| runtime | V4.1 | evidence |
|---|---|---|
| vLLM 0.28.0+rocm723 (installed) | no | `DeepseekV41ForCausalLM in ModelRegistry.get_supported_archs()` → `False` (`DeepseekV4ForCausalLM` → `True`) |
| vLLM 0.29.0 | no | its `registry.py` maps `DeepseekV4ForCausalLM` only; PyPI wheels are CUDA anyway |
| vLLM >= 0.30 | yes | container-only per vLLM's own recipe; this box has no container runtime |
| ROCm/ATOM | no | `atom/models/deepseek_v4.py` has no `engram` / `kv_source` / V4.1 handling |

So [`scripts/dsv4_vllm_8k.sh`](../../scripts/dsv4_vllm_8k.sh) measures
**V4-Flash-0731** at 8k as the closest runnable reference — the previous
generation standing in for the target, not a V4.1 result. Reading it as a V4.1
number would be wrong twice over: different parameter count, and V4.1's whole
point is a ~4x smaller KV cache (~890 B/token), which is a serving-shape
change that a V4 measurement cannot show.

Unblocking a real V4.1 number needs a container runtime plus
`vllm/vllm-openai-rocm:nightly`, or a from-source ROCm build of vLLM >= 0.30.

---

## 5. Kernel extraction: what V4.1 needs against what plow has

The V4 campaign left plow with most of the primitives. What follows maps each
piece of the V4.1 attention/FFN block to an existing `DevOp` or marks it new.
"Exists" means the op is in `crates/packet/src/dev.rs` with a slot contract in
`slots.rs`; it does **not** mean an emitter reaches it (none does — see §3).

### 5.1 Already covered

| V4.1 piece | reference | plow op | note |
|---|---|---|---|
| mHC Sinkhorn split | `model.py:948` | `HyperConnPre` | computes pre/post/comb from `mixes` and applies pre. Contract change needed — §6.1 |
| mHC expand + residual mix | `model.py:962` | `HyperConnPost` | `t=[new_residual, x_out, residual, post_mix, comb_mix]` — matches `hc_post` |
| attention sink | `model.py:639` | `FlashMerge` `t3=sinks` | already an unscaled per-head logit joining the denominator. V4.1's is f32, plow's slot is bf16 |
| indexer score over pooled K | `model.py:527` | `IndexScoreKpool` | `t=[Score, Qfp8, Qscale, Kfp8, Kscale, W, kv_len]`, `W` = `weights_proj` |
| indexer top-k select | — | `IndexSelect` / `IndexSelectPf` | |
| indexer q quant | `kernel.py:41` | `DsaQQuant` | per-row Hadamard + fp8 |
| sparse gathered attention | `kernel.py:311` | `FlashGatherPrefill` / `FlashGatherDecode` | operand shape differs — §5.2 |
| MXFP4 routed experts | `model.py:830` | `MoeGluMx` / `MoeDownMx` (+`Pf`) | w4a16 arms; V4.1 is w4a8, same gap V4 records |
| clamped SwiGLU | `model.py:841` | `Glu` `f1=limit` | `swiglu_limit` 10.0 |
| sqrtsoftplus / noaux_tc routing | `model.py:809` | `MoeRouterTopk` + `MoeAlignPf` | |
| block-FP8 projections | — | `GemmFp8Blk` / `GemvFp8Blk` / `GemmBlkPf` | grid is a packet immediate, so `[32,32]` needs no new body |
| learned-pool KV compressor | `model.py:429` | `d_v4_compress` (`runtime/amd/op_compress.h`) | per-channel softmax pool + learned RMSNorm + post-norm RoPE. Epilogue differs — §5.2 |

### 5.2 New kernel work

1. **FP4 KV cache at group 16 with E4M3 scales.** `_compress_kv` stores the
   latent as real fp4 via `fp4_act_quant(latent, 16, scale_dtype=e4m3)`
   (`model.py:758`). Every fp4 path in plow is OCP MXFP4 — group **32**, scale
   **E8M0** (`interp.hip:2521,2530,2545`), and `op_compress.h`'s epilogue
   fake-quantizes in blocks of 64 with a power-of-two scale and round-trips
   back to bf16. V4.1 wants a different group, a different scale type, and a
   real fp4 store that the attention kernel then reads. This is the single
   biggest new kernel item, and it is what buys the ~890 B/token cache.
2. **Ratio-1 compressor.** No softmax gate, no fp32 promotion — just
   `norm(wkv(x))` (`model.py:459`). Not a new kernel body: a Gemm + RmsNorm,
   or a `coff`-style mode flag on the existing compressor. But 20 of 40 layers
   take this path, so it must not be emitted as a degenerate ratio-1 pool.
3. **Two-level candidate selection.** `IndexUnionPf` unions top-k index sets,
   but nothing selects `candidate_topk_blocks=2048` blocks of
   `candidate_block_size=8` at layer 20 and restricts later indexers to them.
4. **Engram lookup.** A gather over a 384M-row fp8 table with per-row scales
   (`ParallelEngramEmbedding`, `model.py:296`), then an n-gram-keyed q/k
   weighting into `dim * (hc_mult + 1)`. `RowGather` covers the gather shape
   only; the hashing and the weighted combine are new.
5. **Grouped output LoRA.** `wo_a` is block-diagonal over `o_groups=8`, done as
   an einsum because a plain Linear is wrong (`model.py:786`). A grouped GEMM;
   the MoE group ops are the closest existing shape but are expert-indexed.
6. **`sparse_attn` operand shape.** `FlashGather*` is MLA-shaped
   (`Qabs`/`Qrope`/`Ckv`/`Krope`). V4.1 concatenates window KV and compressed
   KV into ONE cache and one index list (`model.py:781-783`), over a flat
   512-wide latent with `num_key_value_heads=1`. Either a new arm or a
   re-specification of the existing one.
7. **Inverse RoPE on O.** `apply_rotary_emb(o[..., -rd:], freqs_cis, True)`
   (`model.py:783`) — the V4 campaign reports a kernel for this; it needs a
   packet slot on the V4.1 path.

## 6. Pipelining changes

These are the changes that are NOT kernel bodies — they are dataflow and
scheduling, and each breaks an assumption the current emitter and runtime
make.

### 6.1 mHC is single-pass: the mix crosses a sublayer boundary

`HyperConnPre` today computes the mixes and immediately applies `pre` to
produce `layer_input` — correct for V4, where a sublayer's mix is its own. In
V4.1 each sublayer's `hc_mixes` produces the `pre` used by the **next**
sublayer, and `Block.forward` returns `ffn_pre` for the next BLOCK
(`model.py:968-995`):

```
attn_pre, attn_post, attn_comb = hc_mixes(x, hc_attn_*)   # for the FFN
x = hc_pre(x, pre_mix)                                     # from the PREVIOUS block
... attention ...
ffn_pre, ffn_post, ffn_comb = hc_mixes(x, hc_ffn_*)        # for the NEXT block
x = hc_pre(x, attn_pre)                                    # from THIS block's attention
return x, ffn_pre
```

The math is unchanged. What changes is the contract: `HyperConnPre` must take
an incoming `pre_mix` tensor for `layer_input` and emit its own `pre_mix` as
an extra output. That makes `pre_mix` a **loop-carried value across layers** —
the graph gains an edge that the per-layer block structure does not currently
carry, and layer 0 needs a seeded initial `pre_mix`.

### 6.2 CSA2 publishes state across layers

`SharedAttentionRuntime` (`model.py:1166`) is a mutable object carrying
`compress_kv` and `topk_idxs`. A `kv_source` layer writes it; every later
layer reads it until the next source overwrites it. Concretely, with
`kv_source_layer_ids = [2, 8, 14, 20]`:

* layers 0-1 have `compress_ratio = 0` and run **sliding window only** — no
  compressed read at all;
* layer 2 compresses and publishes; layers 3-7 read layer 2's cache;
* layer 8 republishes, and so on; layers 21-39 all read layer 20's.

Two consequences. First, the KV cache is **not per-layer**: there are 4
compressed caches for 40 layers, plus a 128-entry window ring per layer. Any
allocator that sizes KV as `layers x ctx x width` over-allocates by ~10x and,
worse, gives each layer its own buffer so the sharing never happens. Second,
`topk_idxs` is published by the 8 `index_source` layers and reused by the
layers between them (`_compress_topk_idxs` returns `shared_attn.topk_idxs`
unchanged when `not self.is_index_source`) — so 32 of 40 layers do **no**
index work at all. An emitter that runs an indexer per layer is doing 5x the
selection work and will not match the reference.

### 6.3 The compressed cache grows at a variable rate

During decode the compressor returns `None` until a group completes
(`model.py:477`): a ratio-2 layer appends a compressed entry every 2 steps,
ratio-1 every step. So the compressed cache length is
`(start_pos + seqlen) // ratio`, and the partial group lives in
`kv_state`/`score_state` across steps. The decode packet program is re-emitted
per step in plow, so "is this step a pool boundary" is answerable — this is
exactly the case `op_compress.h`'s decode mode already handles — but the
**cache length is now layer-group-dependent**, and the attention packet's
`kv_len` must come from the publishing layer, not from the global position.

### 6.4 SWA Bounded Replay changes what prefix caching can reuse

The window KV is a 128-entry **ring buffer** written modulo `win`
(`model.py:718`), not an append-only cache. The model card's "SWA Bounded
Replay reconstructs missing SWA KV states by replaying only the most recent
n_win tokens" means a resumed sequence does not need its full window history
persisted — it can be rebuilt from the last 128 tokens. That is a serving-shape
win (it is where the 1/8 persistent footprint comes from) but it means the
prefix cache stores compressed entries plus a replay tail, not per-layer window
KV. `crates/plowrt/src/memory/prefix.rs` assumes the latter shape.

### 6.5 The encoder/decoder split is a static partition, not a new control flow

Layers 0-19 run at ratio 2, layers 20-39 at ratio 1, and layer 20 is both the
last KV source and the candidate source. Nothing about this needs dynamic
control flow — every mode is fixed per layer at compile time by
`compress_ratios`, `kv_source_layer_ids` and `index_source_layer_ids`. So the
CED structure costs the emitter a per-layer role table, not a scheduler
change. The scheduler changes are §6.2-6.4.

### 6.6 Order within the block is load-bearing

`_compress_kv` runs the indexer **before** writing the cache, because the
indexer needs the pre-RoPE latent and the write applies RoPE in place
(`model.py:748-750`, and the reference comments the read-after-write at
`model.py:762`). An emitter that reorders the compressor's cache write ahead
of the indexer reads roped keys and silently produces wrong indices.

---

## 7. Object contract (read from the shards, not the reference)

Safetensors headers of the downloaded shards. Two entries contradict
`inference/model.py`, and in both cases the SHARD wins — `convert.py`
dequantizes on the way to the reference implementation, so reading dtypes off
`model.py` would size the weights wrong.

| tensor | dtype | shape | note |
|---|---|---|---|
| `attn.wq_a.weight` / `.scale` | F8_E4M3 / F8_E8M0 | `[1280, 5120]` / `[40, 160]` | 40 x 160 = ceil(1280/32) x ceil(5120/32) — the `[32,32]` grid, confirmed |
| `attn.wq_b.weight` / `.scale` | F8_E4M3 / F8_E8M0 | `[32768, 1280]` / `[1024, 40]` | 32768 = 64 heads x 512 |
| `attn.wkv.weight` / `.scale` | F8_E4M3 / F8_E8M0 | `[512, 5120]` / `[16, 160]` | the single 512-wide KV latent |
| `attn.wo_a.weight` / `.scale` | **F8_E4M3** / F8_E8M0 | `[8192, 4096]` / `[256, 128]` | `model.py:645` declares `dtype=bfloat16`; the shard is fp8. 8192 = `o_groups * o_lora_rank`, 4096 = `o_group_in_features` |
| `attn.wo_b.weight` / `.scale` | F8_E4M3 / F8_E8M0 | `[5120, 8192]` / `[160, 256]` | |
| `ffn.experts.{E}.w1.weight` / `.scale` | **I8** / F8_E8M0 | `[2304, 2560]` / `[2304, 160]` | 2560 = 5120/2, two fp4 nibbles per byte. 160 = 5120/32 — MXFP4 group 32, E8M0 |
| `ffn.experts.{E}.w2.weight` / `.scale` | I8 / F8_E8M0 | `[5120, 1152]` / `[5120, 72]` | 72 = 2304/32 |
| `ffn.shared_experts.w*` | F8_E4M3 / F8_E8M0 | `[2304, 5120]` / `[72, 160]` | the shared expert is block-FP8, NOT fp4 |
| `attn.compressor.wkv.weight` | **BF16** | `[512, 5120]` | `model.py:446` declares f32 at ratio > 1; the shard is bf16 |
| `attn.compressor.wgate.weight` | BF16 | `[512, 5120]` | |
| `attn.indexer.wq_b.weight` / `.scale` | F8_E4M3 / F8_E8M0 | `[4096, 1280]` / `[128, 40]` | 4096 = 32 index heads x 128 |
| `attn.indexer.wk.weight` | BF16 | `[128, 512]` | index key from the 512-wide latent |
| `attn.attn_sink` | **F32** | `[64]` | one per head. plow's `FlashMerge` `t3=sinks` slot is bf16 |
| `hc_attn_fn` / `hc_ffn_fn` | F32 | `[24, 20480]` | 24 = `(2 + hc_mult) * hc_mult`, 20480 = `hc_mult * hidden` |
| `hc_attn_base` / `hc_attn_scale` | F32 | `[24]` / `[3]` | |
| `ffn.gate.weight` | BF16 | `[384, 5120]` | |
| `ffn.gate.bias` / `.bias_vl` | F32 | `[384]` | `bias_vl` is the image-span routing bias |
| `embed.weight` | BF16 | `[129280, 5120]` | |
| `vision.blocks.{B}.mlp.w1.weight` | BF16 | `[5632, 1024]` | 5632 = 2 x 2816 — fused gate+up |

Two encodings therefore live in one layer, as in V4: routed experts are
nibble-packed fp4 with an E8M0 scale per 32 along K, while every projection
and the shared expert are block-FP8 e4m3 on a `[32,32]` ue8m0 grid. A single
`projection_weight_dtype` cannot describe the block.

Note this is the WEIGHT encoding. The compressed KV *cache* is a different fp4
again — group 16 with E4M3 scales, written at runtime (§5.2) — and the two
must not be conflated.
