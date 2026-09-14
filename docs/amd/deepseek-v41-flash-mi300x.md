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

### 5.3 ASM audit, done up front

Run before any kernel work, against the 1295 prebuilt gfx942 objects in
`build-tile/hsaco-bm256c5/` -- so this is what the compiler actually emitted,
not what the headers imply. `scripts/asm_audit.py` in report mode; the arch in
each object's ELF header is gfx942, matching `scripts/asm_expect_gfx942.json`.

**The MXFP4 expert path does not exist in any servable object.** This is the
finding that matters, because the roofline puts 49.4% of V4.1's prefill FLOPs
through routed MXFP4 experts.

* There is no MXFP4 grouped-MoE kernel anywhere. Every grouped expert kernel
  is fp8 or gemma: `d_moe_group_glu_fp8_blk`, `d_moe_group_down_fp8_blk`,
  `d_moe_group_{glu,down}_gemma_pf`. The grouped/batched form prefill needs
  has no fp4 variant at all.
* The mxfp4 bodies that do exist are in `test_kernels.elf` -- 14 symbols,
  a test object, not something served. Of the shipped objects only the four
  `interp_decode*_k3*` carry a single mxfp4 symbol, and it is
  `d_gemm_mxfp4_k` at **59 instructions with zero MFMA**: a dispatch shell,
  not a GEMM body. No prefill object defines any mxfp4 kernel.

**Where fp4 lands on gfx942, when it lands at all.** `gemm_mxfp4_c2/c3/c4`
are real bodies, and every one of them issues `v_mfma_f32_32x32x8_bf16`:

| kernel | instrs | MFMA | burst | wait-stalled | spill |
|---|---|---|---|---|---|
| `gemm_mxfp4_c2` | 2129 | 32 bf16 | 7 | 8/32 | 0 |
| `gemm_mxfp4_c3` | 1204 | 16 bf16 | 4 | 4/16 | 0 |
| `gemm_mxfp4_c4` | 837 | 8 bf16 | 2 | 4/8 | 0 |
| `d_gemv_mxfp4_k` | 12355 | **0** | - | - | 0 |
| `d_gemv_glu_mxfp4_k` | 1193 | **0** | - | - | 0 |

So fp4 weights are dequantized and fed to the **bf16** matrix engine, never
fp8 -- which is the best gfx942 can do, since it has no fp4 MFMA at all
(`hwspec` MI300X `mma.fp4: None`). `d_gemv_mxfp4_k` is the decode shape and
uses no matrix engine whatever: 12355 instructions, **1579 `cvt`** and 8560
VALU, i.e. it is a dequantization kernel with a dot product attached.

This changes the baseline. Pricing the expert FLOPs at the bf16 rate they
actually issue at, rather than the fp8 peak, moves the 8k floor from
**24-55 ms to 39-77 ms** (`scripts/dsv41_roofline.py` prints both). 300 ms
survives it -- 3.9x-7.8x headroom -- but half the model's arithmetic is
running at half the peak the part advertises, and closing that is the largest
single performance item in the expert path.

**The grouped-MoE path never reaches the fp8 matrix engine.** Checked because
a 70 ms target depends on it. Every grouped-MoE kernel in the tree that issues
MFMA at all issues `v_mfma_f32_32x32x8_bf16` -- not one uses
`v_mfma_f32_32x32x16_fp8_fp8`, which gfx942 does have and
`interp_prefill_fp8.elf` does use elsewhere:

| kernel | instrs | MFMA | burst | stalled | spill |
|---|---|---|---|---|---|
| `d_moe_group_glu_pf` | 9210 | 32 bf16 | 3 | 16/32 | **112** |
| `d_moe_group_down_pf` | 5274 | 32 bf16 | 3 | 16/32 | **62** |
| `d_moe_group_glu_gemma_pf` | 1937 | 16 bf16 | 3 | 8/16 | 16 |
| `d_moe_group_down_gemma_pf` | 986 | 16 bf16 | 3 | 8/16 | 16 |
| `d_moe_group_glu_gemma_pf_w8a8` | 2831 | 16 bf16 | 3 | 8/16 | 16 |
| `d_moe_group_glu_fp8_blk` | 801 | **0** | - | - | 2 |
| `d_moe_expert_down_fp8_blk` | 2512 | **0** | - | - | 0 |

Two things follow. The `_fp8_blk` kernels have no MFMA whatever -- "fp8" there
names the weight storage format, not the matrix instruction, and those bodies
are VALU. And the `w8a8` variant issuing bf16 MFMA with 8-bit weights AND
8-bit activations shows this is not a missing-dtype plumbing problem that a
flag fixes: no grouped-MoE kernel here has ever targeted the fp8 MFMA.

So "get the experts onto fp8" is new kernel work, not tuning. Combined with
5.3's finding that no MXFP4 grouped-MoE kernel exists at all, the expert path
that carries 49.4% of V4.1's prefill FLOPs has to be written from scratch to
hit a target below ~120 ms.

Caveat on reading the table: MFMA counts are STATIC disassembly counts, so a
low MFMA-to-instruction ratio is not directly an achieved-FLOP number -- the
MFMAs sit in loop bodies with unknown trip counts. What the static form does
settle without inference is the instruction SELECTION (bf16, never fp8), the
spill traffic, and the in-body pipelining (longest burst 3, half the MFMAs
issuing straight after an `s_waitcnt`).

**What IS shipped and healthy.** The mHC, DSA-pool and indexer kernels are
real and present, but only in the `_k3` objects (`interp_{prefill,decode}_*_k3*`) --
not, as the unconditional `#include` in `runtime/amd/interp.hip` suggests,
in every object:

| kernel | instrs | MFMA | spill | note |
|---|---|---|---|---|
| `d_hyperconn_pre` | 1270 | 0 | **2** | 775 VALU, 23 ds_read / 14 ds_write |
| `d_hyperconn_post` | 301 | 0 | 0 | clean |
| `d_dsa_pool_compress` | 575 | 0 | 0 | 16 barriers |
| `d_dsa_pool_expand` | 130 | 0 | 0 | clean |
| `d_dsa_pool_stash` | 60 | 0 | 0 | clean |
| `d_dsa_q_quant` | 333 | 0 | 0 | 16 barriers |
| `d_index_score_kpool` | 496 | 0 | 0 | 25 ds_write |
| `d_index_score_pf_row` | 311 | 16 bf16 | 0 | burst 2, **8/16 wait-stalled** |

Three things to carry into Stage 4:

1. `d_hyperconn_pre` **spills** (2 scratch ops). `asm_audit.py`'s contract
   class 2 refuses on spill, so this kernel fails the contract as built. The
   mHC mix is GEMM-shaped -- [24, 20480], 0.64 TFLOP at 8k -- and runs
   entirely in VALU with no matrix engine.
2. `d_index_score_pf_row` has a longest MFMA burst of 2 and half its MFMAs
   issuing straight after an `s_waitcnt`. That is the quadratic term in the
   model; it is small at 8k (1.79 TFLOP) but it is the term that grows.
3. The `_k3` prefill object contains **only** `v_mfma_f32_32x32x8_bf16` --
   no fp8 MFMA at all, where `interp_prefill_fp8.elf` does carry
   `v_mfma_f32_32x32x16_fp8_fp8`. The object that holds V4.1's new kernels is
   the one with no fp8 matrix path.

Also `d_moe_group_{glu,down}_gemma_pf` both spill 16 and run 0.8%/1.6% MFMA
density at burst 3, which is the shape of the prefill expert kernel V4.1 would
inherit.

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

---

## 8. Stage 2: egglog rules

### Where egglog sits relative to devgen

devgen depends on `rewrite` only as a **dev-dependency**
(`crates/devgen/Cargo.toml`). It never runs saturation. `plowc` does, extracts
the fused graph, and hands devgen a `fused kind -> anchor weight name` map;
`devgen/src/rewrite_lower.rs` looks a site up by the tensor handle it would
fuse on. With `PLOW_EMIT_REWRITE` off — or on with no sites — every query
answers `None` and devgen uses its hand fusions. So there is no separate
"devgen rule set" to extend: the rules live in
`crates/rewrite/src/egl/rules.egg`, and what devgen adds is a LOWERING for
each fused kind it can consume.

### The first problem: devgen consumes only residual+norm fusions

`Lowering` in `rewrite_lower.rs` has exactly two variants:

| devgen lowering | fused kinds it accepts |
|---|---|
| `AddNorm` | `FusedResidualNorm`, `FusedResidual3Norm` |
| `NormResidualNorm` | `FusedNormResidualNorm`, `FusedNormResidualScaleNorm` |

All four are residual-**add**-plus-norm. V4.1 has no plain residual add: both
sublayers end in `hc_post`, which is `post * x + sum(comb * residual)`
(`model.py:962`). So **none of the existing rewrite-driven lowerings can fire
on a V4.1 block**, and the rules that produce them (`residual-rmsnorm-fuse`,
`residual3-rmsnorm-fuse`, `norm-residual-rmsnorm-fuse`, ...) have no match.
This is not a gap to patch around — it is the same observation as §6.1 seen
from the rewrite side, and it means Stage 2 for V4.1 is mostly NEW rules
rather than reused ones.

The rules that DO still fire on V4.1 are the ones that do not touch the
residual: `rmsnorm-linear-fuse` matches `wq_b(q_norm(...))`, and
`linear-act-fuse` / `gated-mlp-fuse` match the expert SwiGLU.

### The second problem: the router cannot be represented

`nn_graph::op::MoeScoring` has two variants, `Softmax` and `Sigmoid`. V4.1 —
and V4 — route with **`sqrtsoftplus`**, which is not one of them. Worse,
`Nn::moe_router_noaux` hard-codes `scoring: MoeScoring::Sigmoid`
(`builder.rs:508`), and `lower.rs`'s `MoeRouterNoAux` arm pattern-matches that
variant explicitly. `schema.egg` states the assumption outright: "Scoring is
sigmoid by definition."

So before any V4.1 graph can saturate:

1. add `MoeScoring::SqrtSoftplus`;
2. give `moe_router_noaux` the scoring as a parameter instead of a constant;
3. carry it into the egglog term.

Step 3 matters for more than fidelity. A grouped router that does NOT match
the `Sigmoid` arm falls through to `MoeRouterGrouped`, which passes only
`e(0)` and `e(1)` — the correction-bias leaf `e(2)` is **dropped**, and with
it `ffn.gate.bias` from the weight manifest. The same class of bug the schema
comments record for Kimi-K3 ("`Opaque` ... would have dropped every weight but
the first").

### New constructors Stage 2 needs

`lower.rs`'s match over `Op` is **exhaustive** — there is no catch-all arm, and
`Opaque` is declared in `schema.egg` but never emitted. That is a good
property: adding an IR op for V4.1 will not compile until `lower.rs` handles
it, so the egglog schema cannot silently fall behind the IR. Each new op needs
a constructor that names every weight leaf:

| V4.1 piece | weight leaves that must not be dropped |
|---|---|
| single-pass mHC | `hc_attn_fn`, `hc_attn_base`, `hc_attn_scale` (x2, attn + ffn) |
| CSA2 compressor | `compressor.wkv`, `compressor.wgate` (ratio > 1 only), `compressor.norm` |
| Engram | `engram.embed` (+ `scale`), `engram.q_weight`, `engram.k_weight`, `engram.wkv` |
| grouped output LoRA | `wo_a`, `wo_b` |

`DsaIndexer` already exists with the right leaves (`wq_b`, `wk`, `k_norm_w`,
`k_norm_b`, `weights_proj`) but its arity assumes the layer OWNS its index
keys; the 4 V4.1 indexers that read published keys have no `wk`/`k_norm`.

### Candidate new rules

Worth trying once a graph exists, in rough value order. Each needs a `; rule:`
annotation and an entry in `Plow.Rewrite.soundRules` with a proof
(`lean-plow/Plow/Rewrite.lean:397`) before it may fire.

1. `hcpost-hcmixes-fuse` — `hc_post` feeding the next sublayer's `hc_mixes`
   reads the stream it just wrote; fusing them saves a `[B,S,hc*D]` round trip
   per sublayer, 80 per token.
2. `hcpre-rmsnorm-fuse` — `hc_pre` is immediately followed by `attn_norm` /
   `ffn_norm` in every block (`model.py:975,992`); the collapse and the norm
   are one pass over the same row.
3. `compressor-norm-rope-fuse` — the compressor's pool, learned RMSNorm and
   post-norm RoPE are already one kernel in `op_compress.h`; the rule makes
   the emitter reach it.
4. `kvnorm-quant-fuse` — `_window_kv` does `kv_norm(wkv(x))` then RoPE then
   `act_quant` in place (`model.py:705-707`). `RmsNorm`'s `t3/t4` fused w8a8
   quant slot already exists for exactly this shape.

---

## 9. Target: 8k context, 300 ms

Campaign goal, set 2026-09-14: serve V4.1-Flash end to end at 8192 input
tokens within **300 ms**, read as TTFT, on 8x MI300X (gfx942).

### Roofline: the measured baseline for 8k prefill

Computed by `scripts/dsv41_roofline.py`, which reads the checkpoint's own
`config.json` and safetensors headers -- so every shape and dtype below is the
checkpoint's, not an estimate -- and `crates/hwspec` MI300X, which carries a
MEASURED 4091.9 GB/s rather than the 5325 datasheet peak. A roofline drawn
against the datasheet understates every kernel here by ~30%.

`build-dsv41-ref/roofline-8k.json` holds the full breakdown.

| 8192-token prefill, 8x MI300X | |
|---|---|
| total | **281.6 TFLOP**, **377.9 GB** HBM, **14.8 GB** per-GPU xGMI |
| arithmetic intensity | 745 FLOP/byte vs machine balance 639 |
| bound | **compute, but only by 1.17x** -- this sits on the knee |
| fp8 dense peak, 8 GPUs | 20 919 TFLOP/s (2614.9 each) |
| HBM, 8 GPUs, measured | 32.74 TB/s |
| compute floor @100% fp8 | 13.5 ms (unreachable) |
| compute floor @55% / @35% | 24.5 ms / 38.5 ms |
| HBM floor | 11.5 ms |
| collectives, if fully exposed | 16.5 ms |
| **baseline 8k TTFT floor** | **24 - 55 ms** |
| headroom to the 300 ms target | **5.5x - 12.3x** |

Where the FLOPs are: routed experts 49.4%, attention projections 29.5%,
sparse attention core 9.4%, shared expert 8.2%, Engram 1.8%, everything else
under 1% each. Where the bytes are: routed expert weights 76.4%, the residual
stream 21.3%, all weights besides the experts 2.3%.

Cross-check: active params per token are 15.05 B (126.6 M attention + 35.4 M
shared + 6 x 35.4 M routed per layer, x40), and 2 x 15.05 B x 8192 = 246.6
TFLOP, which is the 281.6 total less attention core, Engram and the indexer.
**The model card's "8 B active" does not describe this checkpoint**; the
attention projections alone are 126.6 M/layer because `wq_b` is
[32768, 1280] -- 64 heads at head_dim 512.

### Three structural facts the config does not show

**1. Prefill attention is LINEAR in T, not quadratic.** `Attention.forward`
concatenates a 128-token sliding window with `index_topk` = 512 compressed
positions and makes one `sparse_attn` call, so every query attends exactly
640 keys at any sequence length (128 on layers 0-1, which have ratio 0 and no
compressed stream). The only quadratic term left in the model is the
indexer's own score einsum, and at 8k that is 1.79 TFLOP -- 0.6% of the
total. Consequence: **8k is not a hard context for this model, and the 300 ms
target does not get easier by shortening it.** The risk that replaces
quadratic cost is the gather: 640 scattered KV rows per query is between
4 MB/layer (perfect reuse) and 2.7 GB/layer (none), a 650x spread that no
roofline can settle and that the Stage-5 block sweep must measure.

**2. MI300X has no fp4 matrix engine.** `hwspec` MI300X reports
`mma.fp4: None`, and `scripts/asm_expect_gfx942.json` forbids
`v_mfma_f32_32x32x64_f8f6f4` for exactly that reason. The routed experts are
MXFP4 (group 32, E8M0 -- confirmed from the headers: `w1` is I8 [2304, 2560]
for a logical [2304, 5120], two values per byte, with scale [2304, 160]).
So on this part **fp4 buys memory, not FLOPs**: the weights must be
dequantized and issued at the fp8 rate. That is already priced in above --
the 49.4% expert share is computed at the fp8 peak -- but it means the
dequant is on the critical path of half the model's arithmetic, and a
dequant that lands in VALU rather than fused into the MFMA feed is the single
largest performance risk in the expert path.

**3. The collectives are first-order, and the indexer's is the surprise.**
With expert parallelism (48 of 384 experts per rank) and row-parallel `wo_b`,
each rank moves 14.8 GB over xGMI per prefill: 5.87 GB for `wo_b`, 5.87 GB
for the MoE output, and **3.05 GB for the indexer score all-reduce**. That
last one is [S, S/ratio] fp32 -- 268 MB on a single ratio-1 layer, larger
than any weight tensor in the model -- because the 32 index heads are sharded
and the score must be summed before the top-k. At 896 GB/s that is 16.5 ms if
nothing overlaps, comparable to the entire compute floor. Overlapping it is
not optional.

### The target is not the hard part

**300 ms is 5.5x-12.3x the roofline floor.** That is a comfortable target for
a mature stack -- the V4-Flash-0731 recipe reports 7.9-8.5 K tok/s prefill on
a SINGLE MI300X, ~1.0 s for 8k, so ~128 ms on 8 GPUs at perfect scaling. The
risk in this campaign is therefore not the number. It is that six pieces
between `config.json` and a served token do not exist yet.

### What stands between here and it

In dependency order. Items 1-3 are prerequisites for any measurement at all.

| # | Piece | Where |
|---|---|---|
| 1 | `MoeScoring::SqrtSoftplus` + a scoring-parameterised `moe_router_noaux` | §8 — the IR cannot express V4.1's router today |
| 2 | IR ops for single-pass mHC, the CSA2 compressor and Engram, each naming every weight leaf | §8 |
| 3 | the `deepseek_v41` graph builder | §3 |
| 4 | the fp4 group-16 / E4M3 KV-cache kernel | §5.2 — the one genuinely new kernel body |
| 5 | runtime: 4 shared compressed caches for 40 layers, not 40; window ring + Bounded Replay in the prefix cache | §6.2, §6.4 |
| 6 | Stage-2 rules + `soundRules` proofs, then stages 4-7 | §8 |

DSpark and the vision tower are out of scope for the first number: the
reference `generate.py` never calls `forward_spec`, so base output is
bit-exact without DSpark, and an 8k TEXT prefill does not touch the ViT.

### Measurement caveat that will outlive this doc

No V4.1 reference exists on this box (§4), so "300 ms" cannot yet be checked
against a second implementation. Until a container runtime or a source-built
vLLM >= 0.30 lands, a plow number at 8k is unfalsified rather than validated,
and should be reported that way.

---

## 10. Correction: the serving path is devgen, not the nn-graph builder

§3 lists the bringup stages against `nn-graph`, which is right for the
documented harness and wrong as a description of what serves a token. Checked
rather than assumed:

* `crates/devgen/Cargo.toml` has **no `nn-graph` dependency**. devgen reads
  `model_type` out of the checkpoint's `config.json` itself
  (`devgen/src/lib.rs:7412`) and emits a packet program from it.
* `rewrite` is a devgen **dev-dependency**, so saturation never runs in the
  emit path (§8).
* GLM-5.3 and Kimi-K3 are emitted this way. `config/mod.rs` says so for K3:
  "emitted by `devgen`, which does not go through this crate at all".

So the served path is `plowc --hf-dir <ckpt> --emit devblob` -> devgen ->
`model.pkt` -> `plowrt serve`. nn-graph + rewrite is the analysis/JIT side: it
is what Stages 1-3 gate on, and it is where the weight manifest, the shape
contract and the fusion decisions come from, but a V4.1 graph that builds does
NOT by itself serve anything.

### What that means for the 8k/300 ms goal

devgen's DeepSeek support is `MlaArch::DeepSeek` (`devgen/src/mla.rs:10164`),
which covers V2/V3-style MLA + MoE and is shared with the GLM and Kimi paths.
V4.1's block is not that shape — a 512-wide MQA latent, a grouped output LoRA,
CSA2's shared cache, mHC instead of a residual add. `mla.rs` is ~10 k lines for
the MLA family it already serves, which is the honest scale of a V4.1 emitter.

The §9 item list therefore needs one more entry, and it is the largest:

| # | Piece | Scale |
|---|---|---|
| 7 | a devgen emitter for `deepseek_v41` | the dominant item; compare `mla.rs` at ~10 k lines for the MLA family |

Items 1-6 remain worth doing in order — the IR ops and the shape contract are
what the emitter is written against, and the object contract in §7 is what it
binds. But "run V4.1 end to end" is gated on item 7, and no amount of
nn-graph work reaches a served token without it.

---

## 11. Two things the reference settles

### The CED split does not skip layers

`Transformer.forward` (`model.py:1241`) runs **all 40 layers in one loop**.
There is no encoder/decoder branch, no early exit and no layer skipping: the
"20-layer causal encoder + 20-layer decoder" split is about which KV cache each
layer READS (`compress_ratio` 2 vs 1, §1), not about how much compute a token
does.

So §9's 15.1 B activated parameters per token is what the shipped code
executes, and the model card's "8 B (prefill)" does not reconcile with it from
the reference. Note 15.1 B = 6.6 B dense + 8.5 B routed, and 8.5 B is within
rounding of the card's 8 B — the card is plausibly counting the routed experts
alone. Either way the roofline in §9 used the LARGER number, so the floor it
gives is conservative and the 300 ms target has at least the headroom stated.

### The residual stream is 4x hidden, everywhere

`h = h.unsqueeze(2).repeat(1, 1, hc_mult, 1)` right after the embedding
(`model.py:1257`), and it stays that way to the final `hc_pre`. Every layer
therefore carries `[B, S, 4, 5120]` of activation, not `[B, S, 5120]`, and
each sublayer's `hc_mixes` reads it flattened at 20480 wide. The WEIGHT cost is
trivial (`hc_*_fn` is 24 x 20480, ~0.5 M x 2 x 40 = 39 M params) but the
activation traffic and the LDS/register pressure are 4x what a normal decoder
block plans for. At 8k that is 8192 x 4 x 5120 x 2 B = 336 MiB per layer of
residual alone.

`make_identity_pre_mix(h, hc_mult)` seeds layer 0's `pre_mix` before the loop —
confirming that `pre_mix` is genuinely loop-carried across layers (§6.1) and
that the first layer needs an explicit identity rather than reading a previous
sublayer's mix.
