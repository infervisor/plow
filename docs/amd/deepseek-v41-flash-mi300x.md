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
