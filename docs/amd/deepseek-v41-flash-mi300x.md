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

### 4.1 Re-checked against a current vLLM tree, and corrected

Checked again on 2026-09-14 against both the installed vLLM 0.28.0 and the
source checkout at `/tmp/vllm` (HEAD 2026-09-09, post-0.28 dev). Neither
registers `DeepseekV41ForCausalLM`; both register `DeepseekV4ForCausalLM` out
of `vllm.models.deepseek_v4`.

**A correction to what this doc said earlier.** vLLM's V4 module is much
closer to V4.1 than section 1 implied. It already carries `hc_mult` and
hyper-connections, `sqrtsoftplus` routing, and `o_groups` / `o_lora_rank`;
`quant_config.py` handles `expert_dtype="fp4"` MXFP4 experts with **ue8m0**
(e8m0fnu) scales and has an `amd/` path. Those were listed here as V4.1-only
gaps. They are not.

The real blockers are narrower, and both are load-fatal. Reproduce with
`scripts/dsv41_vllm_compat.py`:

| | |
|---|---|
| compressors vLLM would build | 18 (layers 2-19) |
| compressors the checkpoint has | **4** (2, 8, 14, 20) |
| weights vLLM demands that do not exist | **15 layers** |
| checkpoint compressor vLLM never instantiates | layer 20 |
| Engram | 12 tensors, layers 1 and 14, ~196 B params, **no support anywhere in vLLM** |

The compressor mismatch is section 1's V4.1-vs-V4 semantics, seen from the
serving side: `attention.py:227` takes `compress_ratio = max(1,
compress_ratios[layer_id])` and builds a compressor wherever that exceeds 1,
which is the V4 reading. V4.1 uses `kv_source_layer_ids` to say who OWNS a
compressor and lets `compress_ratios[l]` name the cache layer `l` reads.

So vLLM is not a route to a V4.1 number, and an architecture alias would not
make it one. Two things it is still worth reading for:

* its ROCm MXFP4 + ue8m0 path (`Mxfp4MoEMethod`, `models/deepseek_v4/amd/`)
  remains a useful second opinion on a path carrying 49.4% of V4.1's prefill
  FLOPs -- though NOT, as this bullet first said, a stand-in for something plow
  lacks: plow's own gfx942 MXFP4 grouped-MoE body exists and is measured (5.6);
* V4-Flash-0731 DOES run on this vLLM, so it remains available as a measured
  comparison point on the same 8 cards.

### 4.2 Porting the shipped reference to ROCm: how far it gets

The reference is written for NVIDIA. Running it on MI300X took six fixes to
the `nix develop` / system-ROCm toolchain seam (see the commit log), then hit
two real gaps in tilelang's ROCm backend. Status as of 2026-09-14:

| stage | state |
|---|---|
| model builds across TP8 | works, every attempt |
| 501 GB MP8 checkpoint loads | works, every attempt |
| mHC Sinkhorn kernel (`hc_split_sinkhorn`) | compiles and runs |
| UE8M0 block scales | fixed here -- `scripts/dsv41_tl_hip_ue8m0.py` |
| fp8 GEMM | **blocked, and not by a missing alias** |

The fp8 blocker is worth stating carefully, because the cheap fix is a trap.
tilelang raises `KeyError: dtype('float8_e4m3')` from
`rocm/intrinsics/mfma_macro_generator.py`, whose MFMA-prefix map carries only
`float8_e4m3fn` and `float8_e4m3fnuz` while `kernel.py:14` spells it
`float8_e4m3`. Adding the key makes it compile. It would also be WRONG:

* the checkpoint stores OCP e4m3 (`torch.float8_e4m3fn`);
* CDNA3 reads fp8 MFMA operands as **FNUZ** -- tilelang's own header picks
  `TILELANG_FP8_E4M3_VARIANT_FNUZ` for gfx940/941/942, and
  `v_mfma_f32_32x32x16_fp8_fp8` assembles for gfx942 as the FNUZ form;
* OCP bias is 7 and FNUZ bias is 8, so feeding OCP bytes to that instruction
  is a factor-of-two error on every fp8 weight, silently.

ROCm 7.14 does convert OCP on gfx942 (`__hip_cvt_float_to_fp8(..., __HIP_E4M3)`
compiles, verified), but conversion is not what the GEMM does: it consumes the
stored bytes directly. So a correct fp8 path on this part needs either an
OCP->FNUZ re-encode of the weights (which loses the top binade, since the
exponent ranges differ by 2x) or an in-kernel dequantize.

The resolution is the same one gfx950 offers throughout: it has native OCP fp8
AND an fp4 matrix engine, so the reference's own dtypes land there without
re-encoding. On gfx942 the fp8 side still needs the conversion layer above.
(This paragraph used to cite 5.3's MXFP4 finding as the same shape. It is not:
that finding was retracted in 5.6 -- the fp4 side of gfx942 is covered.)

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
| block-FP8 projections | — | **`GemmFp8Mx` (184)**, new; NOT `GemmFp8Blk` | ~~grid is a packet immediate, so `[32,32]` needs no new body~~ **WRONG, corrected 2026-09-14 — see item 5b.** The scale ELEMENT TYPE differs (ue8m0 bytes vs f32), and at a 32-block the promotion lands inside the MFMA burst instead of between k-tiles. Needed a new arm and a new opcode |
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

   **This does NOT gate a prefill emit (2026-09-14).** As written the item reads
   as blocking item 3 of section 5.6's critical path, and it does not.
   Block-diagonal over 8 groups *is* 8 ordinary GEMMs -- `wo_a` is `[8192, 4096]`
   = 8 stacked `[o_lora_rank, heads/o_groups * head_dim]` = `[1024, 4096]`
   blocks, so group `g` is `[T, 4096] x [4096, 1024]` and nothing fuses them but
   dispatch count. At T=8192 ONE of those fills 304 CUs. The fused kernel earns
   its keep at **decode**, where T=1 turns the same work into 8 tiny GEMVs and
   the 8 dispatches are the entire cost. So: emit 8 GEMMs, measure, and write the
   grouped kernel against the decode number rather than ahead of it.

   Worth having sized while looking, because it is bigger than this doc has been
   treating it: `wo_a` is 549.8 GFLOP/layer and `wo_b` 687.2, so together they
   are **49.5 TFLOP of the 281.6 TFLOP 8k prefill -- 17.6%**. `o_groups *
   o_lora_rank` = 8192 is wider than `hidden` itself, and the output projection
   is consequently a first-order term in section 9's budget, not a detail.

5b. **Block-FP8 at `[32, 32]` with E8M0 scales — THE REAL GATE ON THE
   PROJECTIONS, found 2026-09-14.** Every block-FP8 kernel in plow assumes a
   `[128, 128]` f32 scale grid. V4.1's is `[32, 32]` and its scales are **ue8m0
   bytes, not f32**. Both halves are checkable from the shards: `attn.wq_a.weight`
   is `[1280, 5120]` and its `.scale` is `F8_E8M0 [40, 160]` = `1280/32` x
   `5120/32`.

   This is not a corner. It is **112.0 TFLOP, 39.8% of the 8k prefill** — every
   projection in the model:

   | affected | TFLOP | note |
   |---|---|---|
   | `attn_proj` | 82.98 | `wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b` |
   | `shared_expert` | 23.19 | the shared expert is block-FP8, not fp4 |
   | `engram_proj` | 5.15 | `engram.wkv`, scale `[800, 192]` |
   | `indexer_proj` | 0.71 | `indexer.wq_b`, scale `[128, 40]` |
   | **total** | **112.03** | **39.8%** |

   The routed experts are NOT affected — they are MXFP4, a different fetch path.

   Both numbers are load-bearing in the kernel, not cosmetic. `d_gemm_t`'s
   block-FP8 promotion reads `bsblk[nsblk[j] * KB + kb]` with
   `KB = (K + 127) >> 7` and `nsblk = n >> 7`, and it fires on
   `(kt & 1) == 1` because — as its own comment says — "a 128-element K scale
   block is exactly two BK=64 tiles, so the boundary always falls **between**
   k-tiles, outside every MFMA burst and outside the ping-pong's barrier pairs".
   At a 32-element block that property is GONE: one BK=64 tile spans two scale
   blocks, so the promotion lands *inside* the MFMA burst. `op_gemm_common.h`
   states the convention is shared by ops 44, 47 and 85/86 — "one convention,
   three kernels" — so this is a family-wide assumption, not one kernel's.

   devgen already refuses it, loudly and for the right reason
   (`mla_ckpt_enc`): "the 128 is not a parameter anywhere in this emitter —
   `div_ceil(128)` is written into every scale-grid size". So V4.1 stops here
   whatever else is built, and no amount of emit plumbing gets past it.

   **That refusal was true and badly incomplete, corrected 2026-09-14.** It named
   the block size, which is the *easiest* of the three things in the way, and a
   reader who took it at face value would change a 128 to a 32 and find the other
   two waiting. V4.1's `quantization_config` reads, in full:

   ```json
   {"quant_method": "fp8", "activation_scheme": "dynamic",
    "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
   ```

   `mla_ckpt_enc` read *neither* of the last two fields. What they mean:

   1. **`div_ceil(128)` -> `div_ceil(32)`** — 16x the grid elements. The stated gap,
      and the mechanical one.
   2. **`scale_fmt: ue8m0`** — the grid's entries are **bytes**, where op 107's are
      **f32**. This is why op 184 is a separate opcode rather than a flag on 107:
      binding a V4.1 grid to 107 faults nowhere. It rescales every output by a
      number read out of the wrong type at the wrong stride, and the model merely
      gets worse — the silent-corruption shape, not a crash.
   3. **`expert_dtype: fp4` while the dense projections are block-FP8** — so this
      checkpoint is **MIXED**, and that breaks an invariant rather than a constant.
      `MoeEnc` carries ONE encoding for a whole run; `mla_moe_enc_env` says so in
      as many words, refusing `PLOW_MXFP4` and `PLOW_FP8` together because "a run
      is ALL-mxfp4 or ALL-fp8 or ALL-bf16; pick one". V4.1 needs both at once. That
      is a change to what the enum MEANS, not another variant on it, and it is the
      largest of the three.

   The refusal now names all three, keyed on `[32,32]` **and** `scale_fmt ue8m0`
   together so a hypothetical f32-scaled 32-block still takes the generic path
   (`ckpt_quant_tests`, two tests). Capability tag:
   `ckpt_quant_fp8_blk32_ue8m0_mixed_fp4_experts`.

   **The encouraging half: plow already reads exactly V4.1's scale convention,
   just on the wrong operand.** The MXFP4 B-fetch takes "E8M0 scale rows
   `wscale` (K/32 bytes/row)" — group 32, E8M0, byte-wide, which is V4.1's
   convention precisely. So the work is to apply the fp4 path's scale handling
   to an e4m3 weight rather than to invent a scheme: `GM_BLK_BK` is already a
   `#define` and setting it to 32 makes one k-tile exactly one scale block
   (promotion every tile, `kb = kt`), at the cost of halving the K step. That
   trade needs measuring, not assuming, and it is the FIRST thing a GPU lease
   should be spent on — it gates 40% of the budget and it is a kernel question,
   answerable on one card without any of the emit.

   The two alternatives are worse and should be named so they are not
   rediscovered: requantizing to `[128, 128]` at load is LOSSY (one scale
   replacing 16 changes the numerics of every projection), and dequantizing to
   bf16 doubles the projection weight bytes and gives up the fp8 fetch rate.

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

So "get the experts onto fp8" is new kernel work, not tuning -- which is what
puts a sub-120 ms target out of reach without it.

The MXFP4 half of this finding was RETRACTED on 2026-09-14; see 5.6. The audit
reads built objects, and the gfx942 `_k3_moe_a4w4` variant was not among the
blobs on disk -- but `d_moe_group_pf_a4w4`'s CDNA3 arm (`op_moe.h:4308`) is in
the tree, builds for gfx942 without spilling, and benches at ~71% of the part's
bf16 MFMA peak. Nothing below is retracted: no grouped-MoE kernel here targets
the fp8 MFMA, and that is still true of the MXFP4 arm, which uses bf16 MFMA by
construction.

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

### 5.5 What a 70 ms target demands of the expert kernel

Priced by `scripts/dsv41_roofline.py --target-ms 70`. A 70 ms TTFT needs
**39.7%** of matrix peak as built with collectives exposed -- above the 35%
low end of plow's own gfx942 GEMM band -- 30.3% if the collectives overlap,
and 19.2% if the experts also reach fp8. So unlike 300 ms, 70 ms has no
slack: the fp8 expert path and collective overlap both have to land.

gfx942 is not the obstacle. The kernels that DO reach
`v_mfma_f32_32x32x16_fp8_fp8` are well pipelined, and they live only in
`interp_prefill_fp8.elf`:

| kernel | MFMA | burst | wait-stalled | spill |
|---|---|---|---|---|
| `d_gemm_fp8_t<192,256,64,2,4>` | 48 fp8 | **23** | 4/48 | 230 |
| `d_gemm_glu_fp8` | 48 fp8 | **23** | 4/48 | 247 |
| `plow_exec` | 56 of 156 fp8 | 15 | 38/156 | 153 |
| -- versus grouped MoE -- | bf16 only | 3 | 16/32 | 16-112 |

Burst 23 at 8% wait-stalled is what this part does when a kernel is written
for it; burst 3 at 50% is what the grouped-MoE kernels do. The gap is the
kernel, not the hardware. Note the trade the good kernels make: deep MFMA
bursts need accumulator VGPRs, and both spill >200. A 70 ms expert kernel
will have to buy its burst depth the same way and carry an explicit spill
budget, because `asm_audit.py` contract class 2 refuses on spill.

**The tile does not fit the problem.** The well-pipelined template is tiled
BM=192, but the expert GEMM at 8k is M=128: 8192 tokens x top-6 / 384
experts = 128 tokens per expert on average. A 192-row tile on a 128-row
problem wastes a third of the tile before anything else goes wrong, and the
per-expert M is a distribution, not a constant -- routing skew makes some
experts much smaller. So the expert kernel cannot simply reuse
`d_gemm_fp8_t`'s shape; it needs a smaller BM, or token grouping that packs
several experts' rows into one tile.

That is the single largest open design question for a sub-120 ms number, and
it is orthogonal to the fp4 dequant: even a perfect MXFP4 unpack feeding a
BM=192 tile leaves a third of the matrix engine idle on the average expert.

### 5.6 The plow path to 90 ms, and the one thing that gates it

Target moved to 90 ms on 2026-09-14. `scripts/dsv41_roofline.py --target-ms 90`
puts that at **28.9%** of matrix peak as built with collectives exposed, 23.6%
if they overlap -- inside plow's own 35-55% gfx942 GEMM band. So unlike 70 ms,
90 ms needs no fp8 expert kernel and no change of part.

**Correction, 2026-09-14: the expert kernel is NOT the gate. It already
exists.** An earlier revision of this section specified a new `w4a16` arm on
`d_moe_group_pf_t` and called it the one item standing between here and 90 ms.
That was wrong, and it was wrong because section 5.3 audited the *built*
objects rather than the source: the gfx942 `_k3_moe_a4w4` variant was not among
the blobs on disk, so the audit reported no MXFP4 grouped-MoE kernel anywhere.
The body is in the tree and has been for some time.

`runtime/amd/op_moe.h:4308` is `#else /* !PLOW_HAS_MX_MMA`, the CDNA3 arm of
`d_moe_group_pf_a4w4`. Its own header states the design, which is the design
the retracted spec re-derived:

* fp4 x 2^e8m0 is EXACT in bf16 (<= 3 significant bits, power-of-two scale), so
  it dequantizes both operands to bf16 and feeds the ordinary bf16 MFMA --
  differing from CDNA4's scaled-MFMA path only in accumulation GROUPING;
* the dequant lives at **commit**, not at fragment-read time, because a B
  fragment is re-read by `MPF4_WMc` waves and an A fragment by `MPF4_WNc`, so
  read-time dequant would run 2-4x per element while commit-time runs once --
  and lands in the staging phase, whose latency the in-flight global reads
  already cover. The MFMA block is then pure `ds_read_b128` + MFMA with no
  scale traffic at all;
* the tile is BK=64 single-buffered (40,960 B) because bf16 staging doubles the
  tile bytes and CDNA3 halves the arena to 64 KiB -- the CDNA4 plan of
  BK=128 double-buffered fp4 is over budget before the first bf16 byte;
* the swizzle is a 3-bit XOR over 16-byte chunks, chosen by measurement:
  1-bit 737 GB/s, 2-bit 928, **3-bit 963 GB/s** weight stream.

The GLU op on this arm **is** w4a16 in the strict sense, and the header says so
as a deliberate divergence from CDNA4: "the GLU A side stages the gathered bf16
activation RAW instead of quantizing it to fp4 first -- CDNA4 quantizes because
its matrix core demands fp4 operands; this one does not, so the
A-quantization error term simply does not exist here."

Its measured number is the one that settles the budget question. On its own
bench (`moe_prefill_a4w4_cdna3_test.hip`, `MPA4C3_BENCH=1`, H=3584, IM=3072,
4096 rows, grid 304) the GLU runs **232 TF/s, ~71% of this part's bf16 MFMA
peak**. The 90 ms target needs 28.9%. The expert GEMM is not close to being the
constraint.

The build axis is wired already: `scripts/build_gfx942.sh:615` and
`runtime/CMakeLists.txt:1350` build the `_k3_moe_a4w4` gfx942 variant with
`-DPLOW_MOE_PF_A4W4=1`, and the flag stopped demanding `PLOW_HAS_MX_MMA` when
the simulated arm landed. Verified here by building it: the gfx942 object
compiles, both ops emit `v_mfma_f32_32x32x8_bf16`, and **neither spills**.

One real divergence from the reference remains, and it is an accuracy question
rather than a missing capability. On this path the GLU epilogue is a fused
bridge that writes `fu` as MXFP4 + E8M0, so the DOWN op's A side is
bridge-quantized fp4 where the reference keeps bf16 activations. Cross-arch the
contract is deliberate -- an emitted packet cannot tell which arch ran it -- but
against the reference it is an extra lossy step on every routed expert. It
needs a numerics comparison, not a new kernel.

**2. The real critical path, audited op by op.** With the expert kernel struck
off, what remains is emit-side plus two runtime holes. This list is read from
the source, not from built blobs -- the mistake that produced item 1.

The runtime is further along than this doc has been giving it credit for. Six
V4-specific pieces are already written, each with a gfx942 test:

| piece | kernel | dispatchable from `interp.hip`? |
|---|---|---|
| mHC | `op_hyperconn.h` `d_hyperconn_pre` / `_post` | **yes** -- `PLOW_DOP_HYPER_CONN_PRE/POST` |
| DSA indexer | `op_attention_common.h` `[DSV4-IDX]` | **yes** -- ops 58 / 117 / 134 |
| MXFP4 experts | `op_moe.h` `d_moe_group_pf_a4w4` CDNA3 arm | **yes** -- `enc = 2` on ops 85/86 |
| clamped SwiGLU | `PLOW_MOE_ACT_SWIGLU_CLAMP` `[DSV4-ACT]` | **yes** -- `act = 4` |
| router | `op_moe.h` `[DSV4-ROUTE]` | **yes** |
| nope MLA prefill | `[DSV4-MLA-PF]` | **yes** |

Two are written and tested but **have no opcode and no interpreter case**, so
no packet can reach them:

* `op_compress.h` `d_compress_pool` -- the CSA2 learned-pooling KV compressor
  (`runtime/tests/compress_pool_gfx942_test.hip`, `[DSV4-COMPRESS]`). Called
  today only from that test and from `runtime/bench/amd/kx/exp_quant_scale.hip`.
* `op_compress.h` `d_rope_inverse_o` -- the conjugate rotation on the attention
  output's last `rd` dims (`runtime/tests/dsv4_ops_gfx942_test.hip`,
  `[DSV4-IROPE]`). Same: test-only.

**Both are wired as of 2026-09-14** -- `PLOW_DOP_COMPRESS_POOL = 180` and
`PLOW_DOP_ROPE_INVERSE_O = 181`, behind a `PLOW_DSV4_CSA2` build axis that
defaults off, with a `plow_dsv4_csa2_arm` marker the loader requires before it
will accept a CSA2 packet. With the axis off the gfx942 interpreter's `.text` is
byte-identical to what it was before; with it on the object grows ~20 KB.

`d_rope_inverse_o` gained an optional `pos` tensor in the process. It took
`pos0` as an immediate, which cannot work for decode -- the packet stream is
built once and replayed -- and every other rope op in the tree reads the
device-resident counter instead (`d_headnorm_rope`'s `t5`, `d_compress_pool`'s
own `pos`). Verified on gfx942 hardware: `dsv4_ops_gfx942_test` passes with 0
failures after the change, including the inverse rotation at `pos0 = 0` and
`pos0 = 1000` (worst relative error 3.9e-03 against the host reference, which
is bf16 rounding), alongside the V4 router's sqrtsoftplus and hash arms, the
mHC head gate, and the clamped SwiGLU at act code 4.

`compress_pool_gfx942_test` passes on gfx942 as well, 0 failures across all
eleven cases and **exact** (0.00e+00 relative error) on every one: both template
arms (the fp8 path and the `ROTATE` Hadamard-rotated fp4 path), both the
prefill form and the decode form, at `out_base` 0/2/3/5, and -- the case that
matters most for op 180's contract -- the three decode calls at a position that
is NOT a `ratio` boundary, each of which correctly writes nothing at all. That
gate is why `t7 = pos` is the decode form rather than a patched immediate.

One trap worth recording, because it cost a build cycle and would have shipped
silently: the first placement of the two `case` arms was inside `interp.hip`'s
`#if PLOW_K3` region, so they compiled to nothing in a non-K3 object. Nothing
failed -- the object simply did not grow. What caught it was diffing the armed
object's `.text` against the unarmed one and finding them identical, which is
the same check that proves the axis-off path unchanged. Run both directions.

**Engram, decomposed (2026-09-14).** Read from the reference rather than guessed
at: `inference/engram.py` for the hash and `inference/model.py:296-366` for the
rest. It is three stages, and only one of them wants an opcode.

1. **Hash.** Each position is hashed with the `max_ngram_size - 1` tokens before
   it, once per (n-gram size, head) pair, into that pair's own prime-sized
   bucket range -- `(max_ngram_size - 1) * n_heads` = 24 ids per token. This is
   integer work over TOKEN IDS ALONE: it reads no activation, and its primes,
   multipliers and compressed-token map are fixed at load time from the
   tokenizer. So it is host work arriving as a tensor, like `pos` or
   `row_token`, and there is no hash kernel. It cannot be folded into the embed
   either -- the lookback stops at a dead token, so a position's ids depend on
   the mask, not only on its own id.
2. **Embed + project.** The 24 ids fetch 24 fp8 rows of `head_dim`, dequantized
   by block scale, flattened to 6144 and run through `wkv` -- an ordinary fp8
   block-scale GEMM to `dim * (hc_mult + 1)` = 25600, which is exactly the
   `engram.wkv.weight` `[25600, 6144]` in the shards. The table is sharded over
   its rows with an all-reduce.
3. **Gate + mix.** A per-(token, hc copy) normalized dot of the residual stream
   against the key, through a SIGNED sqrt before the sigmoid, with the shared
   value added into every hc copy under that gate.

Stages (2) and (3) are now opcodes -- **183 `PLOW_DOP_ENGRAM_EMBED`** and
**182 `PLOW_DOP_ENGRAM_GATE`**, both in `runtime/amd/op_engram.h`, behind one
`PLOW_DSV41_ENGRAM` axis that defaults off with a `plow_dsv41_engram_arm`
marker the loader requires. Axis off leaves the gfx942 interpreter's `.text`
byte-identical. The `wkv` projection between them needs no opcode: it is an
ordinary fp8 block-scale GEMM.

Two decode facts about the table read, either of which silently rescales every
lookup: the elements are **OCP** e4m3 and `dequant_fp8` is the OCP decode (the
FNUZ hazard §4.2 records applies to gfx942's fp8 MATRIX CORE, which this op
does not touch), and the scale is **ue8m0 at a 32-element block** --
`weight_block_size` is `[32, 32]` in V4.1's `quantization_config`, not V4's
`[128, 128]`, which is why `engram.embed.scale` is `[rows, 8]` for a 256-wide
row. The shard mask is compared as SIGNED: an unsigned `id - vocab_start` turns
an id below the window into a huge in-range-looking index and reads past the
shard, so the test plants ids on both sides and at both boundaries.

Three things in the gate that look like details and are not, each recorded
because getting one wrong compiles, runs, and is a different model:

* the normalization is per (token, hc copy) over `dim`, **not** jointly over
  the copies;
* the signed sqrt before the sigmoid is what shapes the gate's response, and
  `clamp_min` sits INSIDE it on the magnitude -- a dot of zero gives
  `sqrt(1e-6)` carrying zero's sign, not zero;
* `token_mask = 0` shuts the GATE, so the position passes through untouched. It
  does not add a zero value. That distinction is the whole reason image spans
  are masked, and the test asserts the masked token comes back bit-identical
  rather than merely close.

**Engram's host side is DONE as of 2026-09-14**, in
`crates/plowrt/src/text/engram.rs`: the compressed-token map, the n-gram hash,
and the per-step cache that lets a lookback cross the prefill/decode split. The
hash tables it uses are `DeepSeekV41Config::engram_hash_tables` -- see item 2 of
the ordered path below for why the primes are derived and the multipliers are
not.

The token map is a PORT rather than a reimplementation, and that is worth
stating because it looks like the opposite. `build_compressed_token_map` runs
every token id through a `tokenizers` normalizer sequence -- NFKC, NFD,
StripAccents, Lowercase, two regex Replaces around a private-use sentinel, Strip
-- and collapses ids whose normalized text agrees, which is what makes `" The"`,
`"the"` and `"THE"` hash alike. Python's `tokenizers` **is** the Rust crate
plowrt already vendors behind the `hf-tokenizer` feature, and every one of those
normalizers exists there under the same name, so the sequence transcribes. The
one trap is the sentinel: a token that is exactly one space must survive `Strip`,
which is why the reference swaps it for `\ue000` and back.

It verifies against the CONFIG rather than against itself. The map produces
exactly **99 092** compressed ids, which is `engram_compressed_vocab_size` on the
nose -- and that number is not a bound, it is what every hash multiplier was
drawn against, so a map that differed would silently rehash the whole table
rather than fail. `EngramHasher::new` refuses the mismatch for that reason. The
other invariant: a partial UTF-8 byte token (one whose decode contains `\ufffd`)
is keyed by its RAW token string, since there is nothing there to normalize --
normalizing the replacement character merges every such token into one.

The hash itself is pinned against an independent Python transcription of
`inference/engram.py` run on the released config: two implementations of one
spec, rather than one checked against itself.

**The lookback LATCHES, and the window edge is worth pinning.** A dead token
blocks lookback THROUGH itself, and because the n-grams have different reaches it
blocks the long ones further along the sequence than the short ones. With
`max_ngram = 4`, position `p` sees positions `p` through `p-3`. So for a dead
token at position 2: at position 4 the 2-gram is clear of it while the 3- and
4-grams are not, and at position 5 only the 4-gram still reaches it. A lookback
that stopped at the masked position itself, or one that failed to latch after the
first block, reproduces the short n-grams correctly and gets the long ones wrong
-- which is why the tests assert on all three reaches rather than on the masked
position alone.

**Engram is a CAPACITY problem, not a bandwidth one.** From the checkpoint:

| field | value |
|---|---|
| `engram_layer_ids` | `[1, 14]` |
| `engram_num_embeddings` | `[384 006 168, 384 016 682]` |
| `engram_head_dim` | 256 |
| `engram_n_heads` | 8 |
| `engram_max_ngram_size` | 4 |
| `engram_vocab_size` | 16 000 000 (compressed: 99 092) |

384 M rows x 256 fp8 is **98.3 GB per layer, 196.6 GB for the two** -- about 41%
of the 475.3 GiB checkpoint, which is what "the checkpoint's largest tensors by
far" means concretely. It is the single biggest term in the memory plan and
nothing else in this doc had been accounting for it.

**Correct that to 202.8 GB (2026-09-14).** The 196.6 GB is the tables ALONE. Their
ue8m0 scales add **6.1 GB** on top, because `weight_block_size` is `[32, 32]` and
the rows are 256 wide: 8 scale bytes ride with every 256 weight bytes, a 3.1%
surcharge applied to the largest tensors in the checkpoint. It still fits 8x192 GB
(25.4 GB/rank sharded), but a capacity plan drawn from the weight figure is 6 GB
short before anything else is allocated. `engram_tables_dominate_the_weight_budget`
(`crates/devgen/src/mla/dsv41_tests.rs`) asserts both halves and the 1/32 ratio
between them, so the surcharge cannot quietly go missing again.

The TRAFFIC is negligible and does not move section 9's roofline. A token reads
`n_heads` rows of `head_dim` fp8 = 2 KB; at 8k tokens over two layers that is
33.5 MB of gathers total, and sharded by head each rank touches an eighth of it.
The reads are random into a 98 GB table, so they will not prefetch -- but 4 MB
of scattered reads per rank is not a millisecond-scale term against 377.9 GB of
HBM traffic. Build it for correctness and for the capacity budget; do not expect
it to show up in the prefill time.

On the emit side there is no `deepseek_v4` or `deepseek_v41` arm in
`crates/devgen/src/lib.rs` -- it routes `glm5_next`, `glm_moe_dsa`, `kimi_k3`
and `kimi_k2`/`deepseek_v2`/`deepseek_v3`, and the DeepSeek arm wires only
`--block` (a full-model emit panics). Stage 1a/1b live in `nn-graph`, which
section 10 already established is NOT the serving path.

**But mHC is a drop-in, not new work, and that was worth checking rather than
assuming.** `glm53_emit_full` (`mla.rs`, reached from the `glm5_next` route)
already emits full prefill buckets AND decode rungs with hyper-connections --
the `glm_main` analogue item 5 below asks for. Its `emit_glm53_hc_pre` /
`_hc_post` emit ops 128/129 with the constants already at V4.1's values, and
`declare_glm53` names the weights with V4.1's own checkpoint spellings.
Checked against the shards:

| devgen declares | V4.1 checkpoint | shape | agrees |
|---|---|---|---|
| `hc_attn_fn` (`MIX * N * width`) | `layers.N.hc_attn_fn` | `[24, 20480]` | MIX=24, N=4, width=5120 |
| `hc_attn_base` (`MIX`) | `layers.N.hc_attn_base` | `[24]` | yes |
| `hc_attn_scale` (`3`) | `layers.N.hc_attn_scale` | `[3]` | yes |
| `hc_ffn_*` | `layers.N.hc_ffn_*` | same | yes |
| `HyperConnPre.i1 = 4` | `hc_mult` | 4 | yes |
| `HyperConnPre.i3 = 20` | `hc_sinkhorn_iters` | 20 | yes |
| `GemvF32.i1 = 24`, `i2 = 4 * hidden` | -- | 20480 | yes |

GLM-5.3-Flash and V4.1 share the mHC scheme down to the tensor names, which is
also why `op_hyperconn.h` carries the `[DSV4-MHC]` tag. So "mHC emit" is a
binding exercise, not an implementation.

What IS genuinely new on the emit side: the CSA2 chain (ops 180/181, which had
no opcodes until now), the two-level indexer split (`index_source_layer_ids` has
8 layers, `kv_source_layer_ids` 4, so 8 `indexer.wq_b` against 4 `indexer.wk`),
Engram, and the V4.1 config/tensor binding itself. The compressor's operands are
in the checkpoint at the shapes op 180 wants -- `attn.compressor.wkv` and
`.wgate` both `[512, 5120]`, `.norm` `[512]` -- so that binding is mechanical
too once the emit site exists.

**One decision to make before item 3, recorded so it is not re-derived.** V4.1's
config is already parsed and validated once, by
`crates/nn-graph/src/models/config/deepseek_v41.rs` (Stage 1a) -- compress-ratio
reconciliation, the two-level index split, the Engram field cross-checks, all of
it, with a test against the released shards
(`crates/nn-graph/tests/deepseek_v41_official.rs`). devgen cannot see any of
that: it depends on `costmodel`, `packet`, `kernelcaps`, `tunedb`, `hwspec` and
`plow-asset`, and NOT on `nn-graph`. So the emit path either

* gains a `devgen -> nn-graph` dependency and reuses the verified parser, which
  makes Stage 1a serving-relevant after all and is the only way the two cannot
  drift; or
* grows a second V4.1 reader beside `cfg_glm`, matching how every other family
  on the devgen side reads its own checkpoint, at the cost of two parsers for
  one config -- and this config's whole difficulty is in fields that are easy to
  read and easy to misread.

**Settled: reuse the parser.** The supposed cost of the first option is not
real -- `costmodel` already depends on `nn-graph`, and devgen depends on
`costmodel`, so nn-graph was ALREADY in devgen's transitive tree. The direct
edge adds no crate, and nn-graph's own dependencies are `smallvec` and
`thiserror` plus optional serde. A second reader, by contrast, would drift on
exactly the fields whose difficulty is the whole problem -- `compress_ratios`
and `kv_source_layer_ids` disagree by construction (section 5.1).

So `devgen`'s `deepseek_v41` refusal now reports the checkpoint's own VALIDATED
geometry, read through `nn_graph::models::config`: 40 layers, 4 kv_source layers
`[2, 8, 14, 20]` whose compressed cache all of them read, 8 indexer layers, and
Engram at `[1, 14]`. A test asserts those numbers come from the released shards
rather than a hardcoded copy.

Two things that made this less obvious than it looks, both worth knowing before
the emit path reads a config again:

* `ModelConfig::from_json` deliberately REFUSES the released checkpoint. It is
  the multimodal wrapper, its ViT and aligner are unmodeled, and the parser
  points at the text-generation frontend rather than silently dropping the
  vision half. The text tower is the `text_config` sub-object.
* That sub-object does not parse on its own either: `quantization_config` and
  `dtype` live at the wrapper's TOP level. `sub_config` is the merge that lifts
  them down, and it is now `pub` for that reason -- every consumer that wants a
  refused wrapper's text tower needs exactly it, and copying the ten lines into
  each one is how the rule drifts.

**The ordering changed on 2026-09-14, and the new item 0 is not an emit item.**
V4.1's block-FP8 grid is `[32, 32]` with ue8m0 scales; every block-FP8 kernel in
plow assumes `[128, 128]` with f32 scales, and `mla_ckpt_enc` refuses anything
else outright. That gates **39.8% of the 8k prefill** -- every projection in the
model -- and no amount of emit plumbing reaches it, because the refusal fires at
checkpoint-quant time. See item 5b of section 5.2 for the mechanism and for why
it is more tractable than it first looks (the MXFP4 fetch path already reads
group-32 E8M0 scale rows; `GM_BLK_BK` is already a `#define`).

It is also the item best suited to the scarce resource. It is a KERNEL question
answerable on ONE card with no emit at all, where items 3-5 need a whole model
and eight. When a lease comes free, spend it here first.

So the ordered critical path to an 8k/90 ms number is:

0. **block-FP8 at `[32, 32]` with E8M0 scales** -- gates 112.0 TFLOP of 281.6.
   **The arm is WRITTEN (`d_gemm_t<WFP8MX>` / `d_gemm_fp8_mx`) and verified
   offline; its NUMERICS ARE UNVERIFIED** because no GPU came free. It runs at
   BK=32, where one k-tile is exactly one scale block, so the promotion is every
   tile and still lands outside the MFMA burst. Offline it compiles, issues
   `v_mfma_f32_32x32x8_bf16`, reads the scale with `global_load_ubyte` (not
   `flat_`), does not spill, and costs **136 VGPR at occupancy 3** against the
   `[128, 128]` sibling's 160 VGPR at occupancy 2 -- BK=32 halves the staging
   fragments, so it is *cheaper* in registers than the arm it parallels and the
   halved K step is at least partly bought back. The interpreter's `.text` is
   byte-identical to the commit before it, which is the check that matters for a
   new template parameter on a `d_gemm_t` every GEMM in the tree instantiates.
   `runtime/tests/dsv41_blockfp8_gfx942_test.hip` is built and queued: it checks
   the real projection shapes plus ragged tails and an `N = 33` case against a
   reference that decodes e4m3 and ue8m0 from first principles. Remaining: run
   it, then measure BK=32 against a BK=64 variant that promotes twice per tile;

1. ~~opcodes + dispatch for `d_compress_pool` and `d_rope_inverse_o`~~ -- DONE
   (ops 180/181, verified on gfx942 hardware);
2. ~~Engram: kernel, opcode, test~~ -- DONE for the device side (ops 182/183),
   and the **hash tables are now done too**. The prime bucket ranges are
   DERIVED (`DeepSeekV41Config::engram_hash_tables`) by a trial-division walk
   that mirrors the reference's global `seen` set, and the derivation proves
   itself: the ranges are laid end to end, so each layer's primes must sum to
   its `engram_num_embeddings`, and both do exactly (384 006 168 and
   384 016 682). `validate()` asserts it, so a drifting walk cannot reach a
   lookup. The MULTIPLIERS are constants and deliberately so -- they come from
   numpy's PCG64 via `np.random.default_rng(10007 * layer_id)`, reproducing
   that bit-exactly in Rust fails SILENTLY (every hash merely comes out wrong),
   and the draw is eight numbers. Any other V4.1 checkpoint gets `None` rather
   than a guess. What remains of stage 1's host side is the compressed-token map
   -- a direct PORT, not a reimplementation, since Python `tokenizers` *is* the
   Rust crate plowrt already vendors behind `hf-tokenizer` and the normalizer
   sequence (NFKC / NFD / StripAccents / Lowercase / Replace / Strip) exists
   there under the same names -- plus the per-step cache that lets an n-gram
   look back across the prefill/decode split. **Both landed on 2026-09-14**
   (`crates/plowrt/src/text/engram.rs`), so item 2 is complete but for the
   gfx942 hardware run of ops 182/183, which is still queued behind a GPU
   lease;
3. a `deepseek_v41` claim in `devgen::run_verified` with `--block` emit, the
   pattern every family since M3 has started from -- V4.1's config/tensor
   binding is the substance here, since `hc_mult = 4` puts mHC on EVERY layer
   and there is no mHC-free block to extract. **The binding half has landed**
   (`crates/devgen/src/mla/dsv41.rs`): the geometry resolves through nn-graph's
   verified parser, and every layer-0 attention shape is cross-checked against
   the shards, which is how the absorbed-MLA fact in section 7 and the output-LoRA
   correction in section 5.2 were found. A tensor's ABSENCE is never evidence
   there -- a download may be partial -- so only a contradiction is reported.

   The TENSOR contract has landed with it: `dsv41_layer_tensors` gives the
   `(name, bytes)` list for every weight a layer carries, and
   `dsv41_layer_tensors_match_the_shards` checks it against all 96 085 shard
   tensors in BOTH directions -- nothing declared the shards lack, nothing in
   the shards left unbound. The second direction is not symmetry for its own
   sake: dropping `ffn.gate.bias_vl` (the image-span routing bias, no V4
   analogue) passes the first check perfectly. It found two errors on its first
   run, including an `engram.wkv` input sized `n_heads * head_dim` = 2048 where
   this very section already said 6144 -- the prose was right and the code was
   wrong, which is the argument for writing the check before the emit rather
   than after it. What remains of item 3 is the emit itself;
4. CSA2 emit (ops 180/181) and the two-level indexer in the block path. mHC
   comes along with the `glm53` emit, per the table above;
5. full-model emit -- fork `glm53_emit_full` rather than writing one, since it
   already does prefill buckets plus decode rungs with the mHC wiring V4.1
   needs.

Only after 5 does an end-to-end number exist to measure against 90 ms. The
roofline (section 9) says the target is reachable at 28.9% of matrix peak and
the expert GEMM already runs at ~71%; nothing found so far contradicts that.
What stands in the way is emit plumbing and two-and-a-bit missing ops, not
arithmetic.

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

**There is no `kv_b` tensor, and that is the shape of the attention.** `wkv` is
`[512, 5120]` and nothing in the 48 shards splits it -- so V4.1's MLA is FULLY
ABSORBED: one 512-wide latent per token, shared by all 64 heads
(`num_key_value_heads = 1`), serving as both K and V. `head_dim = 512` is
therefore the latent width AND the full per-head q width (`wq_b` is
`[64 * 512, 1280]`), the softmax scale is `head_dim ** -0.5` over the whole 512
rather than over the 64 rotated dims, and the latent is REPLICATED under TP
rather than sharded. The config does not say any of this; reading the 512 as V4's
split latent emits a blob that loads and runs. `devgen`'s `mla::dsv41` now
cross-checks every one of these layer-0 shapes against the shards and refuses
loudly on a contradiction -- an absent tensor proves nothing (a download may be
partial), so only a disagreement is reported.

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

### `costmodel` cannot answer "how many milliseconds", and it was worth finding out

An obvious-looking shortcut to a 90 ms answer without a GPU: cost every V4.1 GEMM
shape through `crates/costmodel`, the model devgen already uses to pick tiles,
convert `Cycles` to time, and sum. It does not work, and the reason is worth
recording so the next person does not spend the afternoon on it.

`costmodel::Cycles` is a **tile-RANKING metric, not a wall-clock prediction**.
`cost::gemm_cycles` computes `steps = out_tiles * k_iters` and multiplies by the
per-tile cost with **no division by `sm_count`** -- the chip's parallelism is
deliberately absent from the compute term, because ranking two tiles for the same
GEMM does not need it. The memory term is per-SM by construction too:
`dma_cycles` divides bandwidth by `spec.sm_count` (`cost.rs:73`). `sm_count`
enters only through `wave_tail_penalty`, which is a quantization correction, not
a throughput divisor.

Taken as wall-clock the numbers are absurd -- 3.3 s per layer, 130 s for the
model -- and, tellingly, dividing by the 304 CUs does NOT rescue them either
(still ~428 ms for the model against a 281.6 TFLOP roofline floor of 24-55 ms).
There is no scaling factor that converts this metric into time, which is the
clearest possible sign it is not a time.

This is the same shape of error as section 5.6's retracted w4a16 item: building on
a tool read wrong rather than on a measurement. The honest position stands --
**the 8k number comes from hardware, and nothing else substitutes for it.** What
`costmodel` legitimately answers is which of two tiles to pick for a given shape,
and devgen already asks it that.

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
