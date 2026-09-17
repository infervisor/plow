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
and eight.

**On getting a card at all, corrected 2026-09-15.** A long-lived `gpuq-runner`
holds a `gpulease -n 8` on the whole box and schedules everyone's work from a
FIFO -- it is not squatting, it IS the scheduler. Raw `gpulease` requests can
never win against it and simply time out; this cost most of a day of "the GPUs
are blocked" before anyone read what the holder was. Submit instead:

```
/root/.claude/jobs/c08d1232/tmp/gpuq/submit.sh <label> <ngpu> <command...>
```

`ngpu=1` is auto-tagged *quick* and sorts ahead of full campaigns, so a kernel
harness starts within minutes. One trap: the runner launches every job as
`nix develop --command <argv>`, so a binary built with the SYSTEM toolchain loads
the dev shell's `libstdc++` against the older system glibc and dies in the loader
with `GLIBC_2.38 not found` -- `rc=1` and no kernel output, which reads as a test
failure and is not one. Build test binaries inside `nix develop`; its ROCm is
7.14, the same version as the lab tree.

So the ordered critical path to an 8k/90 ms number is:

0. ~~**block-FP8 at `[32, 32]` with E8M0 scales**~~ -- **DONE, VERIFIED ON
   gfx942 HARDWARE 2026-09-15.** Gates 112.0 TFLOP of 281.6. The arm
   (`d_gemm_t<WFP8MX>` / `d_gemm_fp8_mx`), its opcode (184) and the N-row clamp
   all landed, and `dsv41_blockfp8_gfx942_test` passes **12 of 12** cases on a
   real card: every V4.1 projection shape plus ragged M/N and `N = 33`. Worst
   relative error across all twelve is **3.4-3.8e-03**, which is bf16's own
   relative epsilon (2^-8 = 3.9e-03) -- i.e. the OUTPUT DTYPE's rounding, not
   error in the arm. That distinction is the point: a misread scale is wrong by a
   POWER OF TWO (the ue8m0 exponent), not by a fraction of an ulp, so this
   residual is positive evidence the scale path is right rather than merely
   "close enough". `N = 576` -- the `attn.wkv` shape that carried the
   out-of-bounds read -- passes with the clamped binary.

   The historical note below is kept because the offline checks are what made the
   hardware run a formality rather than a discovery. It runs at
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
   reference that decodes e4m3 and ue8m0 from first principles.

   **The ISA half has landed too** -- `PLOW_DOP_GEMM_FP8_MX` (184), behind
   `PLOW_DSV41_BLKFP8`, checked in both directions: axis-off leaves the
   interpreter's `.text` byte-identical to the commit before it (the dispatch
   case sits in the hot switch, where an extra case can perturb every other op's
   codegen, so "it compiled" would not have caught it), and axis-on grows it by
   5 632 bytes and exports `plow_dsv41_blkfp8_arm` -- without which a misspelled
   `#if` leaves the arm dead and the build green. A SEPARATE opcode from 107 and
   not a flag on it, because the scale operand TYPE differs (ue8m0 bytes against
   f32) and binding the wrong one faults nowhere. Ordering was the point: the
   encoder must not accept `[32,32]` until an opcode exists that reads it, or it
   emits 107 against a V4.1 grid and silently rescales every output.

   **A real bug in the arm, found on the HOST on 2026-09-14 and fixed.** The
   scale-row index `nsblk[j]` carries no lane component -- one row per 32-column
   fragment -- so it is only right if a fragment's base is 32-aligned. It is
   (`MFMA_N` is 32 and `static_assert(BN % (WN * MFMA_N) == 0)` forces `BN/WN` to
   be a multiple of 32), and an exhaustive host check confirms every valid column
   takes the correct scale. But the ROW COUNT was unguarded:

   > At a 128-wide N block one `BN=128` tile is exactly ONE block row, so the
   > index can never exceed `ceil(N/128)` -- which is why the `[128,128]` sibling
   > needs no guard. At 32 a `BN=128` tile spans FOUR block rows, so when `N` is
   > not a multiple of `BN` the tile's upper j-groups address rows past
   > `ceil(N/32)`.

   V4.1's `attn.wkv` makes that concrete rather than hypothetical: **N = 576**
   (512 latent + 64 rope) is not a multiple of 128, so the last tile addressed
   rows 18 and 19 of an 18-row grid -- **320 bytes past the end of the scale
   array**. Those fragments own no live column and their output is discarded at
   the guarded store, but the promotion reads the byte before anything is
   discarded. `op_gemm_common.h` now clamps the row exactly as the K axis was
   already clamped; the default `.text` is unchanged (the clamp is inside
   `if constexpr (WFP8MX)`) and the armed one grew 256 bytes.

   **Why the GPU test would not have found it.** It already covers `N = 576`,
   `N = 130` and `N = 33`, so it exercised the overread on every run -- but
   `hipMalloc` pads to page granularity, so 320 bytes past a buffer end would
   almost certainly not fault. It would have PASSED with the bug latent. An
   out-of-bounds read is not reliably observable from the output of the kernel
   doing it, which is the general lesson: the scale indexing is integer
   arithmetic and belongs in a test that needs no card
   (`runtime/tests/dsv41_mx_index_test.cpp`, 294 836 assertions over the real
   projection shapes at both wave grids, built and run by
   `scripts/build_dsv41_blockfp8.sh`). Removing the clamp makes it fail, which is
   the check that it is a test at all.

   Remaining: run the numerics, then measure BK=32 against a BK=64 variant that
   promotes twice per tile;

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
   (`crates/plowrt/src/text/engram.rs`). **The gfx942 hardware run landed
   2026-09-15 and item 2 is now COMPLETE**: `dsv4_ops_gfx942_test` passes 16 of
   16 -- engram gate+mix (182) at three shapes with the masked token provably
   untouched, the engram gathered table read (183) at `blk = 32` with every
   masked element zero over 35 798 non-zero reads, inverse RoPE (181), clamped
   SwiGLU, and the SHIPPED router arms unmoved at worst gate rel ~1e-07, which is
   the check that none of this disturbed what already works;
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

   **What "fork it" actually costs, measured 2026-09-14 rather than assumed.**
   `glm53_emit_full` itself is only 82 lines, which is what made this look like a
   one-line item. It is a driver; the work is in what it threads together:

   | callee | lines | for V4.1 |
   |---|--:|---|
   | `declare_glm_rows_batched` | **1 182** | a REWRITE, not a copy -- V4.1's tensors differ throughout. Guided by `dsv41_layer_tensors`, whose contract is already pinned against all 96 085 shard tensors, so the target is specified even though the code is not written |
   | `emit_glm53_program` | 168 | a driver too; the per-layer emitters under it are the real bulk |
   | `declare_glm53` | 66 | shared scratch |
   | `glm_emit_block` (item 3's `--block` sibling) | 61 | the smaller entry point, and the right one to land first |

   So item 5 is low thousands of lines, not a fork in the cheap sense. It is
   still the right shape -- the structure and the mHC wiring carry over -- but it
   should be planned as a multi-session build.

   ~~**And it is blocked on something that is NOT plumbing.**~~ **RESOLVED
   2026-09-15.** `MoeEnc` carries ONE weight encoding for a whole run, threaded as
   `enc` through `declare_glm_rows_batched` and `emit_glm53_program` alike, and
   `mla_moe_enc_env` enforces that outright ("a run is ALL-mxfp4 or ALL-fp8 or
   ALL-bf16; pick one"). V4.1 is **mixed** -- block-FP8 `[32,32]` dense
   projections, fp4 routed experts, stated by its own `expert_dtype: "fp4"`.

   The fix was **not** a new `MoeEnc` variant, and that is the part worth keeping.
   `MoeEnc` travels in an `i[]` slot on the grouped expert ops and the kernel
   BRANCHES on it; a dense GEMM has no such slot and needs none, because the
   OPCODE is the encoding -- 107 reads an f32 `[128,128]` grid, 184 a ue8m0
   `[32,32]` one. A variant would have added a wire value no kernel reads and
   invited exactly the substitution op 184 exists to prevent. So the dense
   encoding is its own type, off the wire:

   ```rust
   enum DenseEnc { Bf16, Fp8Blk128, Fp8Mx32 }
   struct CkptEnc { expert: MoeEnc, dense: DenseEnc }   // + is_uniform()
   ```

   and the PARSE was split from the COLLAPSE. `mla_ckpt_enc_full` returns what the
   checkpoint SAYS and now succeeds for V4.1 -- a fact, not a capability --
   while `mla_ckpt_enc` returns the single value today's emitters thread and
   REFUSES when collapsing would be a lie (`emit_mixed_dense_expert_encoding`).
   V4.1 used to die inside the parse, which meant the fork could not get a typed
   fact out of it; now it is refused one level up, at the assumption that actually
   fails. None of `MoeEnc`'s 239 references moved and the uniform families are
   unchanged.

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

## 12. First hardware numbers: layer 0 runs at 1k and 8k

THE RUNG RUNS END TO END. `plowc --block 0` at TP8 on 8x MI300X, driven by
`plowrt/examples/rung_run`, returns `rc=0` with every probed activation finite
and no NaN or Inf at both bucket widths:

| T | exit range | NaN/Inf | median layer time |
|---|---|---|---|
| 1024 | +-3.11 | 0 / 0 | 159 ms |
| 8192 | +-3.53 | 0 / 0 | 1697 ms |

This is ONE layer on a synthetic input. There is no parity check against a
reference here -- `rung_run`'s entry is a seeded activation, so the numbers are
meaningless as text. What they establish is that every arm the packet names
exists, nothing faults, and the layer is numerically stable.

Two defects stood between the emit and these numbers, both fixed:

  * The grouped MXFP4 GEMM read a whole k-tile when K was 2.25 of them. V4.1 is
    the first shape whose per-rank `I_moe` (2304/8 = 288) is not a multiple of
    the staged K tile, so the DOWN GEMM built a partial last tile and read 48 B
    past every weight row -- off the end of the packed expert slab on the last
    expert's down projection. Latent until the router was fixed, because until
    then every token routed to experts 0-5 and expert 383 was never dispatched.
    It was also wrong math, not only a bad read: the tail staged the next row's
    bytes and multiplied them in.

  * `scripts/build_gfx942.sh` defined `PLOW_MLA_PF_NOPE_ARM` where the marker
    plowrt checks is guarded by `PLOW_MLA_PF2_NOPE_ARM`, and the header's shim
    only maps pf2 -> pf. The flash object compiled the zero-rope arm and could
    not prove it. Invisible at T=1024, which keeps its MLA segment in the
    prefill object; at T=8192 the segment routes to the four-wave flash object.

### 12.1 The performance gap is ~870x and NOTHING I removed changed it

Against a per-layer roofline near 1.95 ms (7.04 TFLOP/layer, 8 GPUs, 35% of
matrix peak), 1697 ms is ~870x off. Forty layers at this cost is 68 s against
the campaign's 90 ms target for the whole model. THE TARGET IS NOT CLOSE.

CORRECTION. An earlier revision of this section blamed the fp4 simulation --
`PLOW_HAS_MX_MMA` is `PLOW_CDNA4`, gfx942 is CDNA3, so the MXFP4 experts run
`d_moe_group_pf_a4w4`'s dequantize-to-bf16 arm. That fact is true and the
inference from it was wrong, and the arithmetic should have caught it first:
the MoE is ~532 GFLOP per rank per layer, so 1697 ms is 0.024% of matrix peak.
A simulated dequant path costs a factor of a few, not four thousand.

WHAT THE MEASUREMENTS SAY. Every one of these changes ONE variable and leaves
every weight shape alone, so the same checkpoint loads unchanged:

| variant | what it removes | median at T=8192 |
|---|---|---|
| baseline | -- | 1733 ms |
| `num_experts_per_tok` 6 -> 1 | 6x the routed-expert work | 1699 ms |
| `sliding_window` 128 -> 32 | 4x the attention work | 1701 ms |
| `hc_sinkhorn_iters` 20 -> 2 | 10x the mHC iteration | 1697 ms |
| `PLOW_MOE_TILE_BINSEARCH=1` | the O(n_exp) tile->expert walk | 1695 ms |
| `PLOW_XR_NOWAIT=1` | BOTH collective rendezvous waits | 1697 ms |

Every row is within 2% of the baseline. The top-k row is the strongest of them:
it provably removes 5/6 of the gathered rows the grouped GEMMs process, and it
buys nothing. `PLOW_XR_NOWAIT` is the ceiling instrument and its output is
garbage by construction (126 M NaN), which confirms it took effect -- deleting
the collectives' synchronization entirely is also free.

Two rows are WEAKER EVIDENCE than they look and are recorded that way: if
`FlashMlaPrefill` ignores the window bits in `i[3]`, or `HyperConnPre` does not
read the iteration count from the packet, then those runs never reduced any work
and prove nothing. Only the top-k and NOWAIT rows are load-bearing.

NOT A TIMEOUT, now actually checked. `interp.hip` passes the collectives a null
status word and a 1 s `PLOW_XCTR_DEADLINE_TICKS`; on timeout the op returns
WITHOUT reducing and `out` keeps its previous value, silently and finitely. At
1.7 s per layer that was a live hypothesis and it would also have invalidated
the correctness claim above. `run_rung` now calls `TpGroup::audit_xctr` after
its drain, as `prefill_chunk` and the decode path always have, and the audit
PASSES at both T=1024 and T=8192: every cross-GPU gate reads its expected
arrival count. The reduces really do happen.

SO IT IS NOT: routed-expert compute, the tile map, collective synchronization,
collective timeouts, and -- weakly -- attention or the mHC iteration. The cost
is linear in T (8x the context costs 10.7x the time; 155 us/token at 1k against
207 us/token at 8k) with no fixed component, and immune to removing parallel
work. That combination is the signature of SERIALIZATION rather than throughput:
work proportional to T being done by a grid that is not covering it. No
bandwidth argument reaches 1.7 s either -- the whole arena is ~2.5 GB and the
layer would have to touch it at 4 GB/s against 5.3 TB/s of HBM.

WHAT WOULD SETTLE IT. Two instruments, neither available here yet:
  * CU scaling. Emitting at `--n-cu 76` and running on the 304-CU device is
    refused by the runtime's n_cu guard, and `PLOW_OVERSUB=1` covers the
    oversubscribed direction, not this one. If a quarter of the CUs costs the
    same, the grid is not the parallel agent it is assumed to be.
  * Per-op device timing. The interpreter is a megakernel, so one launch covers
    34 ops and a kernel profiler reports one number. `PLOW_DSV41_OPS=<n>` was
    added here to emit prefixes and difference their run times, but it only
    reaches prefixes that still contain a collective: `DevBlob::parse` recovers
    `slot_bytes` as `max(i[2])`, and under the two-slot design only the FINAL
    reduce carries a non-zero one, so any cut before op 32 loads as
    `slot_bytes = 0` and is refused. Cutting usefully needs that recovery to
    stop being a max over the stream.

### 12.3 Where the 1.7 s actually is: a per-op profile

> **RETRACTED IN §12.4.** The conclusion below -- that the shortfall is UNIFORM across every op,
> and that no single stage explains it -- is WRONG, and so is the claim in its last paragraph that
> attributing time inside the megakernel "needs a device-side timer inside the interpreter loop,
> which this tree does not have". The tree has one. With it, 97.6% of the layer turned out to be a
> single op. The prefix table below is still accurate as far as it goes; it was just too coarse to
> see that prefix 14 and prefix 23 each contain exactly one `GemvF32`. §12.4 has the measurement.


The interpreter is a MEGAKERNEL -- `--segs 1` times the same as uncapped, so the whole 34-op
layer is ONE cooperative launch and no kernel profiler can attribute inside it. `PLOW_DSV41_OPS`
emits PREFIXES of the layer; differencing their run times is the profiler. T=8192, TP8:

| prefix | what it adds | median | delta |
|---|---|---|---|
| 14 | mHC pre + attention + o-proj + **reduce #1** | 850 ms | 850 |
| 23 | + mHC post/pre + shared expert + **reduce #2** | 1700 ms | +850 |
| 25 | + router GEMM + top-k | 1696 ms | ~0 |
| 29 | + the four align packets | 1696 ms | ~0 |
| 30 | + grouped GLU | 1763 ms | ~0 (noise) |
| 31 | + grouped DOWN | 1705 ms | ~0 (noise) |
| 34 | + combine + **reduce #3** + mHC post | 1917 ms | +212 |

THE WHOLE MoE IS FREE. Router, align, the grouped GLU and the grouped DOWN together move the
number by less than the run-to-run spread, which is the same answer the top-k experiment gave
from the other direction. Everything is in ops 0-22.

AND IT IS NOT THE COLLECTIVES EITHER, though they sit in both expensive segments:

  * `PLOW_XR_NOWAIT=1` deletes both two-shot rendezvous waits: 1697 ms (output garbage, which is
    how we know the arm changed).
  * `PLOW_XR_SCHED=twoshot` against the default `aiter` -- the same data movement on a different
    schedule, 1002 us vs 728 us apart in the reference bench -- measures 1687.7 vs 1687.2 ms.
  * `TpGroup::audit_xctr` passes, so no reduce is timing out and silently skipping.

A collective whose schedule, synchronization and completion all fail to move the number is not
where the time is; it is merely what sits at the segment boundaries.

WHAT IS LEFT IS THE mHC. Ops 0, 1, 14, 15, 16 and 33 -- `GemvF32`, `HyperConnPre`,
`HyperConnPost` -- are the only things in ops 0-22 not already excluded, and they are the ops
this model added. They stream `act.hc_residual_a`, which at `hc_mult=4`, T=8192 is 335 MB, about
2.8 GB of traffic per layer per rank between them. That should cost ~0.5 ms.

THE SHORTFALL IS UNIFORM, WHICH IS THE REAL CLUE. Summing the layer's traffic (mHC 2.8 GB, the
expert weights, `act.part` written and read at 1 GB each) gives ~7 GB against 1.7 s: **~4 GB/s
where the HBM does 5,300 GB/s**, and the same ~5 GB/s falls out of the T=1024 point. Every op is
slow by about the same factor, which is why removing any ONE stage's work changes nothing. Ruled
out as the cause of that: the activation arena is `vram_pool`, i.e. ordinary coarse-grained VRAM
(`device/hsa.rs:1443`), not the fine-grained host-coherent path, so this is not a placement bug.

NEXT, and stated as hypotheses rather than findings: the megakernel's per-op cross-workgroup
synchronization at 304 workgroups, or a specific pathology in the three mHC kernels. Separating
those needs a device-side timer inside the interpreter loop, which this tree does not have.

### 12.4 It was one op, and the layer is now 26 ms

`PLOW_TRACE_RAW` allocates a `PlowTraceRec[n_stream]`; the interpreter already stamps
`s_memrealtime` at arrive/ready/end for every (workgroup, packet); `AmdEngine::trace_write` dumps
it and `scripts/k3_trace_report.py` reduces it. The prefill path was wired for this deliberately.
`rung_run` simply had no way to ask -- it does now, with `--trace <path>`. The first trace, T=8192:

| op | packets | body | share |
|---|---|---|---|
| `GEMV_F32` | 2 | 1 654 174 us | **97.6%** |
| `XREDUCE2` | 3 | 18 116 us | 1.1% |
| `HYPER_CONN_PRE` | 2 | 6 446 us | 0.4% |
| everything else | 27 | 15 108 us | 0.9% |

Not uniform at all. The ~4 GB/s figure in §12.3 was a whole-layer traffic estimate divided by a
wall time that ONE op owned; there was never a shortfall spread across the layer to explain.

**The bug.** `d_gemv_f32` parallelises over N alone:

    for (m = 0; m < M; m++)
        for (n = slice * PLOW_WAVES + wave; n < N; n += nblk * PLOW_WAVES)

Its own header claims it is "grid-strided over N ... and over M for a batched/prefill caller", but
the M loop has no stride. The op was written for GLM-5.3's DSA indexer at M=1..3, N=32. V4.1's mHC
dispatches the SAME opcode at **M=8192, N=24, K=20480**, where the n-stride is 2432 and waves
24..2431 get nothing: three of 304 workgroups computed the entire product while 301 fell through.

Three fixes, each measured on 8x MI300X at T=8192 / TP8, each leaving the output bit-identical:

| change | median | vs prev |
|---|---|---|
| baseline | 1698.9 ms | |
| `d_gemv_f32` wide-M arm (block owns a row, wave owns a column) | 38.2 ms | 44.5x |
| four columns per pass over `x` instead of one | 30.3 ms | 1.26x |
| mHC Sinkhorn's 4x4 in lane-0 registers, not LDS | 26.0 ms | 1.16x |

**65x cumulative, and no bit moved** -- every arm reports exit `min -2.68750 / max 3.53125 /
mean -0.000708 / 0 NaN / 0 Inf`. That is by construction, not luck: an output is still one wave's
`wave_sum` over the same lane-strided K in the same order, and the Sinkhorn copy is generated from
the shipped block by substitution rather than retyped.

The Sinkhorn cost was priced by ablation BEFORE it was fixed: `hc_sinkhorn_iters` 20 -> 2, same
weights and objects, only the packet operand changing, moved the layer 30.34 -> 25.49 ms. The
register fix recovered 4.27 of those 4.85 ms. (The same ablation at 1.7 s read as a null result --
a 5 ms effect is invisible under a 1654 ms op. Several of §12.1's null results are that kind of
null, and are worth re-running now that the layer is 26 ms.)

**Where the 26 ms sits now**, per the trace:

| op | packets | body | note |
|---|---|---|---|
| `GEMV_F32` | 2 | 6 268 us | 8 waves each sweep the whole 40 KB row -> 2.7 GB of x traffic |
| `HYPER_CONN_POST` | 2 | 3 477 us | |
| `MOE_GROUP_DOWN_PF` | 1 | 2 982 us | |
| `GEMM_FP8_MX` | 8 | 2 524 us | |
| `HYPER_CONN_PRE` | 2 | 2 396 us | |
| `MOE_GROUP_GLU_PF` | 1 | 2 149 us | |
| `XREDUCE2` | 3 | 1 873 us | |
| `FLASH_MLA_PREFILL` | 1 | 1 492 us | |
| rest | 14 | 2 167 us | |

The mHC trio -- `GemvF32`, `HyperConnPre`, `HyperConnPost` -- is 12.1 of 25.3 ms, **48%**, and it
is this model's own addition: `hc_mult=4` makes the residual stream 335 MB at 8k and every mHC op
streams it. The next structural cut is that `GemvF32` is really an 8192x24x20480 GEMM being run as
196 608 independent wave-level dot products with no register reuse; a tiled GEMM would read `x` and
`W` once each. That is an emit change (pick a GEMM opcode when `rows` is large), not a kernel one.

### 12.6 Two more cuts, and one that did not work

`HyperConnPost` computes `new_residual[j][d] = sum_i comb_mix[i][j] * residual[i][d] +
post_mix[j] * x_out[d]` with the `rrow` load INSIDE the j loop, so each of the n output streams
re-read all n input streams: 16 loads to produce 4 outputs at n=4, i.e. 1.34 GB of reads against
a 335 MB tensor. Hoisting the n reads into registers ahead of the j loop took the op from 1738 to
**314 us per packet (5.5x)** and the layer from 26.27 to **23.27 ms** -- more than the 4x the read
count predicts, because the re-reads were also serialising the j loop behind a load.

**NEGATIVE RESULT, recorded so it is not re-tried.** `GemvF32` reads W -- 24 rows of K floats,
2 MB -- once per output ROW, so a block re-reads it for all 27 rows it owns, ~51 MB out of L2.
Carrying TWO rows against the same four `w` loads halves that and gives eight independent FMA
chains instead of four. It is SLOWER: 23.26 -> 23.61 ms, and `GemvF32` itself 3148 -> 3290 us per
packet. Register pressure did not move (256 VGPR, 122 spills, identical), so the extra state was
not the cost; W simply is not the binding constraint. Reverted.

That matters for what to try next. `GemvF32`'s x traffic is 2.7 GB (eight waves each sweep the
whole 40 KB row) = ~510 us at HBM speed, and its W traffic is now shown not to bind, yet the op
costs 3148 us. So it is neither x- nor W-bandwidth-bound: it is latency-bound, 320 loop iterations
of dependent loads at the megakernel's fixed 2 waves/SIMD with nothing to hide them behind. More
register tiling will not fix that; a tiled GEMM that stages x and W in LDS, or an emit that picks
a real GEMM opcode for this shape, is the actual fix.

**Layer at 23.3 ms**, `GemvF32` 6.3 ms (27%), routed MoE ~5.7 ms, dense+shared 2.7 ms, collectives
2.0 ms, attention 1.8 ms, `HyperConnPre` 2.4 ms, `HyperConnPost` 0.6 ms. 40 x 23.3 = 932 ms
against 90 ms: **10.4x**.

### 12.5 The nulls re-run, and where the 26 ms goes

Two of §12.1's null results, re-run against the 26 ms layer. Both were real all along; a 5 ms
effect simply cannot be seen under a 1654 ms op.

| ablation | at 1.7 s | at 26 ms | what it prices |
|---|---|---|---|
| `hc_sinkhorn_iters` 20 -> 2 | 1697 vs 1733 ms (null) | 30.34 -> 25.49 ms | the serial mHC Sinkhorn |
| routed top-k 6 -> 1 | 1699 vs 1733 ms (null) | 26.06 -> 21.27 ms | the grouped MXFP4 experts |

Both ablations demonstrably took: the exit moves (`min -2.45312` and `min -2.75000` against the
real `-2.68750`), which is the check that separates "the knob did nothing" from "the knob was not
read". **Any null result recorded before §12.4 should be treated as unmeasured, not as negative.**

Top-1 saving 4.79 ms puts the routed MoE at about **5.7 ms per layer** at top-6, which scales with
routed work as a compute-bound op should. The arithmetic floor is 435 GFLOP/rank/layer against
MI300X's ~1307 TFLOPS bf16, i.e. **333 us** -- so the grouped A4W4 GEMM is running at roughly **6%
of peak**. That is the CDNA3 arm, which dequantizes fp4 to bf16 in staging because `PLOW_HAS_MX_MMA`
is `PLOW_CDNA4`; gfx942 has no native MX MMA. This is the first measurement that actually supports
attributing cost to the fp4 simulation -- §12.1 retracted an earlier attribution that was made
without one, and that retraction was correct at the time.

**The 26 ms layer, by subsystem:**

| subsystem | cost | share |
|---|---|---|
| mHC (`GemvF32` + `HyperConnPre` + `HyperConnPost`) | 12.1 ms | 48% |
| routed MoE (grouped GLU + DOWN + combine) | 5.7 ms | 22% |
| shared expert + dense projections (`GEMM_FP8_MX`, `GEMM_MED`) | 2.7 ms | 11% |
| collectives (`XREDUCE2` x3) | 1.9 ms | 7% |
| attention (`FLASH_MLA_PREFILL` + `FLASH_MERGE`) | 1.8 ms | 7% |
| norms, RoPE, router, align | 1.1 ms | 4% |

Reaching 90 ms for 40 layers needs 2.25 ms per layer, so it is not one more op: it needs the mHC
restructured (a tiled GEMM for `GemvF32`, and `HyperConnPost`'s 335 MB write), the A4W4 arm lifted
off 6% of peak, and the whole-model emit that §12.2 lists. None of those is small.

**The gap is now 11.6x, not 870x.** 40 layers x 26.0 ms = 1.04 s against 90 ms. That over-counts a
whole-model emit, which overlaps seams this measures in isolation, and it under-counts everything
in §12.2 that is still not emitted.

### 12.8 Occupancy is capped by LDS, not by the register budget

§12.6 left `GemvF32` latency-bound: 320 iterations of dependent loads at 2 waves/SIMD with nothing
to hide them behind. The obvious lever is `PLOW_WPE`, which `interp.hip` documents as exactly this
measurement -- "Raising this forces the allocator under 128 VGPRs; it will spill, and whether the
extra latency hiding outruns the spill is the measurement." The decode arms already set it; prefill
never did, so it took the default `PLOW_WAVES/4 = 2`. Wiring it through to `AX_PREFILL` (default
unset, so an ordinary build is unchanged):

| waves/EU | VGPR | spills | private seg | median |
|---|---|---|---|---|
| 2 (default) | 256 | 122 | 1124 B | **23.25 ms** |
| 3 | 216 | 0 | 1816 B | 24.31 ms |
| 4 | 176 | 0 | 2632 B | 24.43 ms |

**Both are slower, and the spills going to ZERO did not help.** The reason is that
`waves_per_eu` never bought any occupancy: `group_segment_fixed_size` is **64 720 B** of the CU's
64 KiB LDS, so exactly ONE workgroup is resident per CU no matter what the register allocator
does. Eight waves over four SIMDs is 2 waves/SIMD, and that is fixed by the arena. All the hint
changed was the register budget, and the extra scratch traffic cost ~4%.

**So the lever for this megakernel's occupancy is LDS, not registers.** Getting two workgroups per
CU (4 waves/SIMD, and real latency hiding for every latency-bound op in the layer, not just
`GemvF32`) needs the prefill arena under 32 KiB. `op_gemm_common.h` puts the smallest tile the
fused-GLU `SN==2` assert allows at 8 waves at 128x256x32 = 30 720 B, which would fit twice over --
so the 64 720 B is the GEMM arena plus the MoE staging buffers, and the question is whether the
prefill arena can be sized separately the way `PLOW_DEC_ARENA_HALVES` already sizes decode's.
That is the next experiment worth running, and it is a bigger one than anything in §12.4-12.6.

What sets the 64 720 B is NOT the attention tile, which was the obvious guess: every prefill
object measures the same arena, including `interp_prefill.elf`, which has no MLA arm at all.

| object | LDS |
|---|---|
| `interp_prefill` (no MLA) | 64 720 B |
| `interp_prefill_mla` | 64 720 B |
| `interp_prefill_mla_moe` | 64 720 B |
| `interp_prefill_fp8kv_k3_moe_a4w4` (the one that runs) | 64 720 B |
| `interp_flash` (4 waves) | 58 368 B |

`raw` is `max(PLOW_GM_ARENA, FA_LDS_HALVES(512), PLOW_MLA_PF_HALVES)`, and the GEMM arena is what
wins. **It cannot be shrunk from the build**: the packet PINS the tile, and the build refuses a
mismatched object --

    FAIL: packet requires GM_BM=192 but GM_BM=128 is set in the environment

which is the right refusal (a smaller arena against a packet emitted for a 192-row tile would
read past the allocation, the exact failure mode `op_moe.h` records as having shipped once
already). So the arena is an EMIT-side choice: `pick_tile` picks `GM_BM=192`, and moving it means
changing what the emitter selects for this shape, then rebuilding both sides together. There is no
emit knob for it today, and `pick_tile` is shared with every other model, so this is a real change
with a real blast radius rather than a flag flip.

And it is a TRADE, not a free win: a smaller tile buys a second workgroup per CU (4 waves/SIMD,
which is what the latency-bound ops want) and costs GEMM efficiency on the ops that are already
compute-bound — the shared expert and the projections, 2.7 ms of the layer. Worth measuring, not
worth assuming.

### 12.7 Multi-layer: `--block l..r` chains

> **The "31 of 40 layers emit" claim in this section is RETRACTED by §12.9.** The parts table was
> reporting the CSA2 WRITE side only, so 38 reader layers were marked complete while the emit gave
> them sliding-window attention and nothing else. The true count is **1 of 40**. The chaining work
> and the 5-layer timing below both stand; what they measure does not include the compressed-KV
> read those layers owe.

**`--block 0..39` used to silently emit layer 0.** The spec was parsed with
`block_spec.split("..").next()` -- the left end alone -- so a whole-model request produced a blob
**byte-identical** to `--block 0` (md5 `3344f59e...` both ways) with no diagnostic. That is the
silently-wrong artifact this emitter refuses everywhere else, and it is also why §12's "40 layers
at this cost" was arithmetic rather than a measurement.

A range now means `l..=r`, every layer in it is plan-checked before anything is written, and a
malformed range is refused rather than narrowed. The chain itself was a small change because the
pieces were already there: `Dsv41Weights::per_layer` is indexed by layer id, `declare_dsv41_weights`
already took a slice, and a layer's output tensor IS its input tensor (`act.hc_residual_a`, because
`ri` flips twice per layer), so layers need no copy between them -- each layer's last op is simply
the next layer's dependency.

**The chain runs.** Layers 3..7, one program, 170 ops, T=8192, TP8:

| | median | per layer |
|---|---|---|
| 1 layer (layer 0) | 23.28 ms | 23.28 ms |
| 5 layers (3..7) | 115.62 ms | **23.12 ms** |

So per-layer x N was a sound model after all: layers compose with no measurable seam cost, and the
40-layer projection is **925 ms**, not a guess. The exit stays finite (0 NaN, 0 Inf) though its
range grows from +-2.69 to +-410 over five layers -- expected from `PLOW_HC_POST_MULT = 2.0`
compounding on a synthetic seed with no embedding to anchor it, and NOT evidence of correctness
either way. There is still no parity check.

**What is actually missing for end to end**, which is much narrower than §12.2's list suggested.
Running the planner over all 40 layers: **31 emit today**. The 9 that refuse are layers
**1, 2, 8, 14, 20, 24, 28, 32, 36** -- exactly `kv_source_layer_ids` [2, 8, 14, 20] union the
indexer layers [2, 8, 14, 20, 24, 28, 32, 36] union Engram [1, 14]. Each is missing 2 of its 11
parts, and the refusal says the rest is done:

    layer 2: 9 of 11 parts are done.
      not emitted:
        - csa2 compressor (ops 180/181)
        - indexer queries (two-level)

with, in its own words, *"The kernels are NOT the gap: ops 180/181, 182/183 and 184 all exist and
pass on gfx942. What is missing is the emit around them."* The eight-subsystem list quoted earlier
in this document is the **model-level** refusal from the nn-graph path, which is not the path that
serves; the rung path's gap is the CSA2 compressor emit, the two-level indexer emit, and Engram's
emit for layer 1.

### 12.9 The parts table was reporting the write side of CSA2 only

Chasing "which layers emit" turned up a hole in the refusal itself, which matters more than the
count it produced.

`kv_source_layer_ids` says who **writes** the shared compressed cache -- 4 layers.
`compress_ratios[l]` says which cache layer `l` **reads**, and `0` is the only value meaning
"sliding window only". The config module says so in its own header -- *"In V4 a nonzero
`compress_ratios[l]` meant layer `l` runs its own compressor. In V4.1 it does not: only
`kv_source_layer_ids` compress, and every other layer reads that cache"* -- and `Dsv41Cfg` repeats
it on the field: *"Layers owning a CSA2 compressor. EVERY other layer READS their cache."* The
module header even warns the two *"disagree by construction"* and are *"easy to MISREAD"*.

`dsv41_layer_parts` consulted the writer list alone. So every layer that merely READS the cache was
reported **complete**, while `emit_dsv41_attn_core` dispatched `FlashMlaPrefill` with
`d.i[5] = KV_MASK_NONE` and nothing but the 128-token window. `compress_ratios` for this checkpoint
is `[0, 0, 2 x18, 1 x20, 0, 0, 0]` (the trailing three are DSpark blocks, not layers), so **38 of
40 layers** were attending over a fraction of the keys they owe and being certified done for it.

That is the exact "loads, runs, and produces fluent-looking garbage" outcome this emitter's refusal
exists to prevent -- produced *by* the refusal, which is the worst place for it. The table now asks
`attn_kind(l)`, an accessor that already existed and that nothing consulted:

| | before | after |
|---|---|---|
| layers that emit | 31 of 40 | **1 of 40** (layer 0) |

Layers 0 and 1 are the only genuinely window-only ones; layer 1 also needs Engram. A test pins the
reader side by layer id and pins the count at 1, so this cannot silently reopen.

**What this costs the numbers above.** §12.7's 5-layer chain (23.12 ms/layer) is still a real
measurement of the ops that were in it, and the chaining machinery is still right -- but those
layers were missing their compressed-KV attention, so **23.1 ms/layer is a floor, not the layer
cost**, and the 925 ms projection is an under-estimate of the real model. It also means that
packet can no longer be emitted, which is correct.

**And it moves the end-to-end gap.** It is not "9 layers need two emits" -- it is: 38 layers need
the compressed-KV attention read, 4 need the compressor that writes it, 8 need indexer queries,
2 need Engram. The kernels for the write side exist (ops 180/181, 182/183, 184 all pass on gfx942);
the read side additionally needs the top-k gather wired into the attention core, for which
`FlashGatherPrefill` (op 55) is the intended kernel and is already built and dispatched.

### 12.10 What the remaining 39 layers actually need, from the reference

The checkpoint ships its own inference code at `inference/model.py`, which is what the `model.py:NNN`
citations throughout this tree point at. This section is read off it, not inferred, so the next
session does not re-derive it. Line numbers are that file.

**The shape of it (`class Attention`, 613).** Its own docstring: *"Latent attention over two KV
sources at once, concatenated into one `sparse_attn` call: a sliding window of raw KV, plus -- when
compress_ratio > 0 -- `index_topk` compressed positions reaching further back."* The forward is

    kv, topk_idxs = self._window_kv(...)                 # window KV + window INDICES
    if self.compress_ratio:
        compress_kv, compress_idxs = self._compress_kv(...)
        kv         = torch.cat([kv, compress_kv], dim=1)
        topk_idxs  = torch.cat([topk_idxs, compress_idxs], dim=-1)
    o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
    apply_rotary_emb(o[..., -rd:], freqs_cis, True)      # INVERSE rope -> op 181

so it is ONE gathered attention over a concatenation, not two attentions merged. Note the window is
expressed as indices too (`get_window_topk_idxs`), which makes plow's current `FlashMlaPrefill` +
window-mask form a *specialization valid only at ratio 0*. `FlashGatherPrefill` (op 55) takes the
`[b, m, topk] int32` table in `t7` and `top_k` in `i6`, which is exactly this.

**1. The rope tables are wrong for 38 layers, independently of everything else** (658-666):

    if self.compress_ratio: original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
    else:                   original_seq_len, rope_theta = 0, args.rope_theta   # disable YaRN

Ratio-0 layers (0 and 1 only) use base `rope_theta` 10000 with YaRN OFF. **Every other layer uses
YaRN at `compress_rope_theta` = 160 000**, `original_seq_len` 65 536, factor 16, beta_fast 32,
beta_slow 1. `emit_dsv41_block` currently builds the ratio-0 table (`RopeScale::None`, base theta)
and hands it to whatever layer it is emitting.

**2. The compressor, 4 layers (`class Compressor`, 429).** Ratio 1 (layer 20) is `norm(wkv(x))` --
no gate, bf16 weights. Ratio 2 (layers 2, 8, 14) promotes `wkv`/`wgate` to fp32 and pools
`kv = (kv * score.softmax(dim=2)).sum(dim=2)` over the ratio axis, then `norm(kv.to(bf16))`. It
returns the latent **before RoPE** -- the indexer needs it unrotated -- and `Attention` then ropes
at position `j * ratio` and fake-quants.

Against op 180, which implements **V4's** compressor, three differences: V4.1 has **no `ape`** (no
learned position-in-block gate bias), **no overlap** (`coff = 1`; V4's ratio-4 form draws 8 slots
through a `2*d` projection), and the fake-quant's per-block scale is **E4M3, not a power of two**
(`fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`, 672). `qblk` already accepts
16 and `ROTATE` already selects e2m1, so the only *kernel* work is a third scale arm beside
`plow_round_scale`. Everything else is emit.

**3. The indexer, 8 layers (`class Indexer`, 488).** q from `wq_b(qr)` at 32 heads x 128, roped;
k from the compressor latent via `wk` + `k_norm`, which only the 4 kv_source layers can produce
(the other 4 read that layer's `k_cache`); scores rectified and combined per token by
`weights_proj`. Reachability: position `j` becomes visible once the query has passed its last
token, `compress_lens = arange(1, T+1) // ratio`. Then the two levels:

  * Layer 20 (`candidate_source_layer_id`) publishes `select_candidate_blocks` (585): score each
    block of 8 by its **best** position, pin the block holding this query's newest position with
    `+inf` so a partly-filled recent block cannot be outscored by an older full one, take the top
    2048, and expand back to a per-position bool mask.
  * Layers after 20 mask their own scores with that shared mask before their own top-k.

Then `topk = min(512, end_pos // ratio)`, taken by score, **re-sorted into position order**, with
unreachable entries `-1` and the rest shifted by `offset` (the window KV length, so the indices
address the concatenated `kv`).

**4. Cross-layer shared state, and it is what keeps this tractable.** `shared_attn.compress_kv`,
`shared_attn.topk_idxs` and `shared_attn.candidates` are module-level and carried across layers.
A layer that is not an index source **reuses the last indexer's `topk_idxs` verbatim** (`_compress_topk_idxs`,
722). So the index table is computed **8 times, not 38** -- the other 30 layers only need the
gather. In plow that is three arena tensors declared once for the whole chain, which is what the
`--block l..r` chaining added in §12.7 makes expressible.

**Engram** (layers 1 and 14) is separate and its opcode (182) is already specified and hardware-verified.

None of this is emitted today. It is a multi-piece build -- one small kernel arm, four emit paths,
and correct rope tables -- and the honest blocker on signing any of it off is that there is still no
parity harness: `rung_run` feeds a seeded synthetic, so "finite and stable" is the only bar
available, and it is not one this work can be checked against.

### 12.11 There IS a parity harness, and §12.10's spec now passes it

I said four times that nothing here could be checked against a reference, because serving V4.1
needs vLLM >= 0.30 and this box has no container runtime. That was true about vLLM and wrong about
parity. The tree's own method for this is an oracle on tiny synthetic shapes -- it is how
`op_hyperconn.h` was signed off (*"Verified against a numerical oracle (tiny synthetic shapes,
hand-checkable) BEFORE this file was written"*) -- and that needs torch, not vLLM.

The installed torch is a ROCm 6.3 build whose `libamdhip64.so.6` is missing, and `/opt/rocm-6.3.1`
turns out to be a hipblaslt-only stub. But a CSA2 oracle needs no GPU at all: `pip install torch
--index-url .../whl/cpu` into `/workspace/oracle-venv/site` gives a working CPU torch 2.14, and the
reference module (`inference/model.py`) is plain PyTorch with no tilelang on the paths that matter.

`scripts/dsv41_csa2_oracle.py` runs the reference `Compressor` against a transcription of what
op 180 computes, and turns §12.10's three READ claims into measured ones:

| check | result |
|---|---|
| op 180 with `ape = 0`, `coff = 1` vs the reference compressor | **max err 0.000e+00** |
| the same with a nonzero `ape` | max err 1.65 |
| ratio-1 (layer 20) as a plain projection + norm | **max err 0.000e+00** |
| fake-quant with an E4M3 scale vs a power-of-two scale, both fp4 at block 16 | max err 0.875, **15.3% of latent amax** |
| quant span `d` vs `d - rd` (rope tail left bf16) | max err 0.406 |

So: op 180's pooling, its per-channel slot softmax and its post-norm ARE V4.1's compressor once
`ape` is omitted and the overlap transform is off -- exactly, not approximately. `ape` is not
zero-equivalent, so it has to be left out rather than passed as zeros-by-accident. And the scale
format really is the one kernel difference, worth 15% of the latent's amax rather than a rounding
detail that would wash out.

That also fixes the tolerance question a hardware test has to answer: the epilogue is lossy
(fake-quant vs the unquantized latent is 0.75 here), so a test asserts against the QUANTIZED
reference, not the pre-quant latent.

**Three kernel differences, not one.** §12.10 named the scale format. Reading op 180's body against
the reference, with the oracle to price each one, there are three, and all three are in the
epilogue rather than the pooling:

  1. **Scale format.** `plow_round_scale` rounds to a power of two (V4); V4.1 wants E4M3.
     Worth 15.3% of the latent's amax.
  2. **Quant span.** op 180's attention arm quantizes `[0, d - rd)` -- *"rope dims stay bf16 for
     positional precision"*, citing V4's `model.py:510`. V4.1 quantizes the WHOLE latent
     (`fp4_act_quant(latent, 16, True, ...)`, no slice, `model.py:760`), and `_window_kv`'s own
     docstring says the window K is *"quantized over the whole post-RoPE vector, RoPE tail
     included"*. Worth 0.406 here.
  3. **No tap between norm and rope.** op 180's body is pool -> norm -> rope -> quant -> store with
     no exit in the middle. V4.1's `Compressor.forward` **returns the latent before RoPE** because
     the indexer consumes the unrotated form, and only then does `Attention` rope and quantize it
     (`_compress_kv`, 738-762). So the 4 kv_source layers cannot get their index keys out of op 180
     as it stands -- it needs a stop-after-norm mode, or to be split in two.

(1) and (2) are small, local and now covered by the oracle. (3) is a factorization change, and it
is the one that would have been found late and painfully: everything upstream of it matches
exactly, so an emit built on op 180 would work right up until the indexer needed its input.

**What this changes.** "Nothing can be signed off on numerics" was the stated reason for not
building the emit. It is no longer true for the compressor, and the same harness extends to the
indexer and the gather -- both are plain PyTorch in the reference. The bar now exists; what remains
is the work in §12.10 against it, plus these three.

### 12.12 The mHC is CROSS-SUBLAYER, and layer 0 was wrong too

The three §12.11 differences are all in the CSA2 compressor, a part the tree already marked
`Todo`. This one is in a part it marked **Done**, and the part that dominates the 23.3 ms:
`HyperConnPre` + `GemvF32` + `HyperConnPost` are 9320 us of it, 48%.

`Block.forward` (`inference/model.py:965-996`) takes `pre_mix` as an **argument**:

```python
residual = x
attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, ...)
x = self.hc_pre(x, pre_mix)        # <- the PREVIOUS block's ffn_pre
x = self.attn_norm(x); x = self.attn(...)
x = self.hc_post(x, residual, attn_post, attn_comb)   # post/comb ARE same-sublayer
residual = x
ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, ...)
x = self.hc_pre(x, attn_pre)       # <- THIS block's attention pre
...
return x, ffn_pre                  # <- to the NEXT block
```

The class docstring states it outright -- *"The coefficients a sublayer computes are used by the
\*next\* one"* -- and `Transformer.forward` (1260-1268) seeds the chain with
`make_identity_pre_mix`, a **one-hot** on copy 0, then spends the dangling `ffn_pre` on one final
`hc_pre` after the last layer.

GLM-5.3's hyper-connection does not do this: `d_hyperconn_pre` derives `pre` and consumes it in the
same call (`op_hyperconn.h`, lane-0 block then reduction 2), and `MhcHandles` had no field for an
incoming `pre_mix`. `emit_dsv41_mhc_pre` binds V4.1 onto that op, so V4.1 inherited GLM's ordering.
**Nothing in the shapes distinguishes them** -- both derive `pre` from `mixes[0..hc_mult)`, both
collapse the same residual with an `hc_mult`-vector -- which is why it survived the parts table,
the topological-program test and the activation read/write test.

`scripts/dsv41_mhc_oracle.py` prices it instead of asserting it. Same weights, same input, f64,
only the ordering varies:

| check | result |
|---|---|
| **layer 0 alone**, V4.1 ordering vs GLM ordering | **51.9% relative** |
| plow's layer-0 attention input vs `residual[0]` (what V4.1 feeds it) | 140.6% relative |
| 3 layers chained | 36.6% relative |
| `post`/`comb` under the two orderings | **0.000e+00** -- same-sublayer in both |
| bump layer 0's `hc_ffn_base[pre]`: residual moves? | 0.000e+00 (no) |
| ... and layer 1's attention input moves? | 99.2% (yes) -- a real cross-layer dependency |
| the `[2,T,hc]` ping-pong half arithmetic vs the reference chain | **0.000e+00** |

The last pair is the one that rules out a renaming: perturbing only the `pre` rows of layer 0's FFN
mix leaves the residual stream bit-identical and still moves layer 1's attention input, because
that input is gated by coefficients layer 0's FFN produced.

**The fix.** `pre_mode` on op 128, three arms: `PRE_OWN` (GLM, unchanged), `PRE_SEED` (V4.1's first
sublayer, the one-hot) and `PRE_DEFER` (read the previous sublayer's `pre`). In devgen it is
`MhcPre::{Own, Seed, Deferred}` -- an enum rather than a bool, so a new caller has to say which
model it is. The gates travel in ONE `[2, T, hc_mult]` f32 tensor, `act.hc_pre_pair`, because
`HyperConnPre` had already spent 7 of the descriptor's 8 tensor slots; sublayer `k` reads half
`k % 2` and publishes into the other, so consecutive sublayers alternate and a call's read and
write never touch the same half. `the_mhc_pre_gate_comes_from_the_previous_sublayer` pins the mode
and half sequence of a 2-layer chain (`[1,2,2,2]`, `[0,1,0,1]`), so reverting to `MhcPre::Own` is
now a test failure rather than a 52% numerical error.

**What this changes about the count.** §12.9 corrected "31 of 40 layers" to "1 of 40". For the
duration between that correction and this fix, the honest count was **0 of 40**: layer 0 emitted,
but with the wrong mHC ordering, so the 23.3 ms in §12.4-12.8 was measured on a graph that is not
the model. The *timing* stands -- the fix adds one f32 store and one f32 load per token per
sublayer against 5120-wide bf16 traffic, so it is noise -- but the numerics of every run before
this commit were not V4.1's.

**What is NOT verified.** The oracle checks the SCHEME -- including the half arithmetic, replayed
through a `[2, T, hc_mult]` buffer indexed exactly as `d_hyperconn_pre` indexes it, which
reproduces the reference chain to 0.000e+00 in f64. It does not check the HIP. The gfx942 object
compiles (all 53 rows, `interp_prefill_fp8kv_k3_moe_a4w4` among them), but the `PRE_SEED` and
`PRE_DEFER` arms have not run on hardware: `hyperconn_sinkhorn_gfx942_test.hip` exercises only
`PRE_OWN` through the defaulted parameters, and the standalone HIP test link is broken on this box
(`arc4random@GLIBC_2.36`, nix ROCm lld vs system glibc -- §4.2). A `rung_run` timing pass would
exercise the arms but cannot check their numerics, since there is still no full-layer parity
harness.

**Still missing, and not per-layer.** The model's tail: after the last block, `Transformer.forward`
spends the dangling `ffn_pre` on a final `hc_pre` (`model.py:1268`). A block emit's output is the
residual stream, so nothing is wrong today, but a whole-model emit owes that collapse.

**The pattern, for the third time.** §12.9 (the CSA2 read side), §12.11 item 3 (the tap between
norm and rope) and this one are all the same shape: a V4.1 subsystem that is *structurally* a
rearrangement of a shipped GLM/V4 one, with identical tensor shapes, bound onto the shipped op and
marked Done. Shape agreement is not evidence. The next candidates to check the same way are the
indexer's two-level selection and the Engram gate.

### 12.13 The mHC fix on hardware, and 2.8 ms out of the mHC's own GEMV

The §12.12 fix is not a paper change: the packet was re-emitted, three objects built against it,
and all three run on 8x MI300X at T=8192, TP8.

**The fix is free.** Arm 0 -- the same `d_gemv_f32` the 23.3 ms was measured with, now with
`pre_mode` wired -- lands at **23272 us median**, against 23272 us before. One f32 store and one
f32 load per token per sublayer against 5120-wide bf16 traffic is noise, as predicted. The exit is
finite and stable: `min -2.73438  max 3.53125  mean -0.000708  zero 0  NaN 0  Inf 0`.

**And the arms are bit-identical, demonstrated rather than argued.** All three objects produce the
exit above to the last printed digit. That is the check that matters for a knob that reassigns
which wave owns which `(m, n)`.

#### `GemvF32` was reading `x` eight times

§12.4 found this op and cut it 827 -> 3156 us per packet. It was still the largest single item in
the trace, and 3156 us for a `[8192, 24] = [8192, 20480] x [20480, 24]` GEMV is ~50x off what its
335 MB of `x` should cost. The reason is in the arm §12.4 wrote: a BLOCK owns a row and its eight
waves each take four column slots, so **each of the eight waves reads the whole row** -- 2.7 GB of
`x` traffic where the tensor is 335 MB, held together only by L1.

Five candidate arms, built and measured because reading cannot rank them -- and two of the five
came out backwards from the reading:

| arm | who owns what | `GemvF32` (2 pk) | layer median |
|---|---|---|---|
| 0 (was shipped) | block per row, 4 column slots per wave, one aliased at N=24 | 6311 us | 23272 us |
| 1 | the same, alias removed (3 contiguous columns per wave) | 6300 us | 23284 us |
| 2 | wave per row, N swept in groups of 8 | 3429 us | 20465 us |
| 3 | wave per row, all 24 columns in registers -- `x` read ONCE | 8136 us | 25207 us |
| **4 (now default)** | **flattened `(row, column group)` index** | **3362 us** | **20416 us** |

**Arm 1 is the negative result.** Removing the aliased slot deletes a quarter of the FMAs and a
quarter of the W reads and changes nothing (6311 -> 6300 us, inside run-to-run noise). So neither
the VALU nor W was the constraint -- the alias was already free, because W is 1.97 MB and lives in
L2.

**Arm 2 is worth 2.8 ms on the layer**, 12%. A wave owning its own row re-reads that row once per
column group -- three passes at N=24 rather than eight -- trading inter-wave spatial reuse for
intra-wave temporal reuse. Every other op in the trace is unchanged within noise, which is what
makes this attributable rather than a whole-layer wobble:

| op | arm 0 | arm 2 |
|---|---|---|
| GEMV_F32 | 6311 | **3429** |
| MOE_GROUP_DOWN_PF | 3002 | 3038 |
| GEMM_FP8_MX (8 pk) | 2510 | 2512 |
| HYPER_CONN_PRE | 2474 | 2509 |
| MOE_GROUP_GLU_PF | 2152 | 2143 |
| XREDUCE2 (3 pk) | 1916 | 2071 |
| FLASH_MLA_PREFILL | 1462 | 1404 |
| **body total** | **22603** | **19869** |

**Arm 3 is the instructive failure.** It makes the largest traffic cut available -- every column in
registers, so `x` is read once per row rather than three times -- and is the WORST of the five, 30%
slower than the arm it was meant to beat and slower than the one that shipped. 24 live floats per
lane on a kernel already at the 256-VGPR cap with 122 spills do not fit, the accumulators go to
scratch, and the spill traffic costs more than the `x` re-reads it saves. The trace says so
directly: the straggler goes 460 -> 917 us per packet. The object's reported VGPR/spill/LDS numbers
are IDENTICAL to arm 2's (256 / 122 / 64720) -- the global allocation was already at the cap, so
the extra pressure shows up only as more scratch traffic inside this one op and not in the summary
the build prints. Reading the build log would have cleared this arm.

**Arm 4's rationale was right and its payoff was small.** Arm 2 gives a wave whole rows, and 8192
rows over 304 x 8 = 2432 waves is 3.37 each, so the waves that get 4 set the op's time -- 18.7% of
the machine idle at the tail, against 3.7% for arm 0's 8192/304 = 26.9 rows per block. Flattening
`(row, column group)` into one index over 8192 x 3 = 24576 items is 10.1 per wave, and the
straggler does halve exactly as predicted, 460 -> 201 us. But the op only gains 2% (3429 -> 3362),
so the ragged tail was never the main cost either. Arm 4 is the default because it is the best
measured and its tail behaviour should compound less in a long chain, not because 2% matters.

**Where the remaining 3362 us is: still unexplained.** It is ~50x off the 63 us that 335 MB of `x`
at HBM bandwidth would cost. Three hypotheses have now been tested and priced -- wasted FMA/W work
(arm 1: free), `x` re-read count (arms 2/3: real but bounded by register pressure), tail raggedness
(arm 4: real but 2%) -- and together they account for a 1.9x cut, not a 50x one. The occupancy cap
in §12.8 (LDS pins this kernel at 1 workgroup per CU, 2 waves per SIMD) is the remaining suspect
and is not addressable from inside this op.

**What this does NOT change.** 20.4 ms/layer x 40 is ~817 ms against a 90 ms target, still ~9x off,
and it is a floor measured on a layer that is missing its compressed-KV attention entirely. The
performance half of the goal is not close, and this is an honest 12%, not a step toward it.

### 12.14 Layer 1 emits: Engram, and a defect no parts table can see

§12.9 corrected the emittable count to 1 of 40. It is now **2 of 40**. Layer 1 is the cheapest
remaining layer -- its `compress_ratio` is 0, so it needs no compressed-KV attention, and Engram
was the only thing it was missing.

**The kernels were already right.** `scripts/dsv41_engram_oracle.py` transcribes
`ParallelEngramEmbedding.forward` and `Engram.forward` from the reference, and `d_engram_embed` /
`d_engram_gate` from `op_engram.h`, separately, then runs them against each other:

| check | result |
|---|---|
| op 183 vs `ParallelEngramEmbedding`, all ids in shard | **0.000e+00** |
| ... with ids below AND above the shard | 0.000e+00, out-of-shard exactly 0 |
| op 182 vs `Engram.forward` | **2.220e-16** |
| ... with a `token_mask`, and the masked token untouched | 2.220e-16, 0.000e+00 |
| signed-sqrt gate vs a plain `sigmoid(dot)` | 0.104 at `dot = -3.36` |
| joint normalization vs per-`(token, hc copy)` | 47.8% relative |

The last two are `op_engram.h`'s own "three things that look like details and are not", priced
rather than asserted: the gate at `dot = 0` is 0.50025 (the clamp, square-rooted), not 0.5 and not
0. The checkpoint corroborates the rest -- `q_weight`/`k_weight` are BF16 `[4, 5120]`, which is what
the kernel reads them as, and `embed.scale [rows, 8]` and `wkv.scale [800, 192]` are both the
`blk=32` the ops assume, not V4's 128.

**So the gap was the emit**, four ops: op 183 gathers 24 rows of 256 fp8 per token, an `XReduce`
sums the row-split shards, op 184 projects `[T][6144] -> [T][25600]`, and op 182 gates and mixes in
place. The all-reduce is the reference's own `dist.all_reduce` (`model.py:323`) -- the table is
98.31 GB and two of them do not fit in 192 GB, so row-splitting is forced and a rank's gather is a
partial by construction.

Three things could not be written as constants, and each was settled from the reference rather
than chosen: `vocab_start` (one program serves eight ranks, so `i4 = TENSOR_NONE_I` now means
"derive `rank * i5`" -- the interpreter has `prog.rank`); the peer slot (the gather is a partial,
so it must land in the peer region, and with no third slot it reuses attention's, safely, because
Engram runs at the top of the layer); and the split rule for `engram.wkv`/`q_weight`/`k_weight`,
which is Replicated because `self.wkv` is the plain `Linear` and not `ColumnParallelLinear`
(`model.py:345`). A note in the split table had said that last one "follows from how the row gather
exchanges between ranks, and that is not designed yet". It does not -- the exchange is the embed's
own all-reduce, and it happens before `wkv` ever runs.

#### The defect: a packet invariant that spans two ops

The layer-1 packet emitted, built 53 objects, reached the queue -- and the loader refused it:

```
TP  n_gpu=8 hidden=6144 slot_bytes=83886080        83886080 % 12288 = 8192
```

`DevBlob::parse` recovers exactly two numbers from a packet's collectives, `hidden` as
`max(i[0] / program.t)` and `slot_bytes` as `max(i[2])` (the byte offset of partial slot B), and
`AmdTpGroup::load` divides one by the other to get `max_tokens`, refusing the packet unless it is
exact. Every collective in V4.1 is 5120 wide except **one** -- Engram's all-reduce of the gather,
which is `n_cols * head_dim` = 6144. That single op raised the recovered `hidden` while the slot
offset stayed `t * c.hidden * 2`.

The fix is that the slot **unit** is the widest per-token message in the packet, not the model's
hidden size, and it has to be conditional on the chain containing an engram layer -- widening it
unconditionally breaks every other packet the same way, since their recovered `hidden` stays 5120
and a 6144-based offset is not a multiple of 10240 either. Layer 1 now reads `slot_bytes =
100663296`, and `100663296 / 12288 = 8192` exactly.

**This is a different class from §12.9 through §12.13.** Those four were MODEL structure that the
tensor shapes agreed about. This is PACKET structure that the shapes agreed about: the emit was
locally correct at every single op, and what broke was an invariant spanning two ops that neither
op owns. A parts table cannot see it, and neither can a reading of either op --
`the_peer_slot_unit_divides_the_widest_collective` recovers both numbers the way `DevBlob::parse`
does and asserts they divide, for every emittable layer.

Two smaller ones behind it, and both were caught by a refusal rather than by a wrong number:

  * `build_gfx942.sh` never passed `-DPLOW_DSV41_ENGRAM=1`, although the C side, the packet's
    `build.json` `requires` list and the knob registration all had it. plowrt refused the object
    **by name** -- `plow_dsv41_engram_arm` absent, *"the AMD dispatch's `default:` does not trap,
    so those ops would write nothing and the prefill would complete with garbage"*.
  * `plowrt`'s `shard_of` is a SECOND name table, separate from devgen's `dsv41_shard_of` (plowrt
    cannot depend on devgen), and it had no Engram rows -- so the 98.31 GB table fell through to
    `Replicated` and the load died on the size check: *"replicated but the checkpoint has
    98305579008 B and the blob declares 12288197376 B"*. The file's own header says this is exactly
    how the V4.1 block was found the first time. `engram.embed.` is Column; `engram.wkv.`,
    `engram.q_weight` and `engram.k_weight` are Replicated, and they do not collide with
    `attn.wkv.`.

#### Layer 1 on hardware

```
layer 0 (no engram)  median 20470 us   min -2.73438  max 3.53125  mean -0.000708  NaN 0  Inf 0
layer 1 (ENGRAM)     median 33847 us   min -6.68750  max 6.56250  mean -0.000195  NaN 0  Inf 0
```

Finite and stable, and the wider range is the Engram value being added to the stream rather than
noise. (The first iteration is 236 ms: paging in a 12.29 GB table slice per rank. The median is
over the warm ones.)

**Engram costs 13.4 ms, and 11.2 ms of it is one op.** From the trace, against layer 0:

| op | layer 0 | layer 1 | note |
|---|---|---|---|
| GEMM_FP8_MX | 2512 us (8 pk) | **13734 us (9 pk)** | the 9th packet is `wkv` |
| XREDUCE2 | 2071 us (3 pk) | 2934 us (4 pk) | the 4th is the gather's all-reduce |
| ENGRAM_GATE | -- | 957 us | |
| ENGRAM_EMBED | -- | 169 us | the gather itself is nearly free |
| body total | 19869 us | 33129 us | |

The gather -- 24 random rows of a 98 GB table per token -- costs 169 us, which is the result worth
keeping: the sharded random-access read is not the problem. `wkv` is, and the reason is placement.
It is `[8192, 6144] x [25600, 6144]^T` = 2.58 TFLOP, and because `self.wkv` is the plain `Linear`
the emit replicates it, so **all eight ranks compute the same 2.58 TFLOP**. That is faithful to the
reference, which is describing arithmetic and not a placement, and it is 8x redundant work on the
largest op in the layer. Column-parallel over the 25600 output is the obvious alternative and is
not obviously better -- the gate needs every column, so it would trade the redundancy for an
all-gather of `[T][25600]` bf16, 419 MB -- but a placement that splits `hidden` instead, reduces
only the `[T][hc_mult]` gates and leaves the mix elementwise, would avoid both. Not attempted.

**This does not change the projection much**, because only layers 1 and 14 carry an Engram: 38
layers at 20.4 ms plus 2 at 33.8 ms is ~843 ms, against 90 ms.

### 12.15 Two defects in the layers that already run, and the compressor's split

Going after the 38 reader layers meant reading `Attention.forward` end to end rather than the
compressed branch alone. That turned up two defects in the part marked Done -- the attention core
of layers 0 and 1, the two layers that emit, load and produce finite output today. Both are the
campaign's recurring class: **the tensor shapes agree and the structure does not.**

#### The RoPE pairing

`emit_dsv41_attn_core` rotates `q` and the shared latent with [`DevOp::QwenHeadNormRope`] (op 142),
whose body pairs element `i` with element `i + rotary/2` -- the half-split (NeoX) convention every
Qwen packet wants. V4.1's `apply_rotary_emb` says otherwise in its first line:

> Rotate `x` in place, **taking adjacent element pairs as complex numbers**.

`view_as_complex(x.float().unflatten(-1, (-1, 2)))` (`model.py:392-397`) is the interleaved (GPT-J)
convention. Same tensor, same `[ctx][rd/2]` table, same table CONTENTS -- a different operator.

The interesting question was whether it cancels. Both `q` and the latent go through the same op, and
a rotation applied in a permuted basis is conjugate to the right one, so a consistent relabelling
would be free. It is not a relabelling: half-split gives the pair `(i, i+32)` the angle of frequency
`i` and interleaved gives the pair `(2m, 2m+1)` the angle of frequency `m`, so the two assign
DIFFERENT angles to the same channels. `scripts/dsv41_rope_oracle.py` asks the attention score
rather than the vectors:

| check | result |
|---|---|
| `q` after interleaved vs half-split rope | 6.63 |
| **attention score, both sides consistently wrong** | **31.3, 41.2% of the max score** |
| `irope(rope(q)) == q` | 4.8e-07 |
| attention output with vs without op 181 | 7.05 |

Op 142 now takes the pairing in **bit 31 of `i2`** -- it rides the width because the op has no free
`i` slot -- and the interleaved arm's partner is `lane ^ 1` with both lanes of a pair reading table
entry `lane >> 1`. It also reads the tables as f32 rather than rounding them to bf16, because the
reference multiplies by a complex64 `freqs_cis` and rounds only the result; the bf16 round in the
half-split arm is vLLM's Qwen cache convention and does not belong to DeepSeek.

#### The inverse RoPE was never emitted

`Attention.forward` ends:

```python
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
apply_rotary_emb(o[..., -rd:], freqs_cis, True)     # model.py:781
```

on EVERY layer, window-only ones included. The emit stopped at `FlashMerge`. Op 181
(`RopeInverseO`) has existed since the CSA2 wiring and had **no emit site anywhere in the tree** --
`grep RopeInverseO crates/devgen` returned nothing. Its own header calls it "STRUCTURAL, not
cosmetic": `o` mixes cached rows each rotated by its own position, and de-rotating by the QUERY's
position is what leaves the position-independent latent the fixed `wo_a` was trained against.

A layer without it runs, stays finite, and is wrong -- which is exactly what §12.9's refusal exists
to prevent and exactly what the parts table said was Done. The row now reads "interior rope +
windowed absorbed MLA + sink merge + **inverse rope**", and the test asserts op 181 is the core's
LAST instruction, after the merge, because `o` is what carries the rotation and the partials are
not.

#### The compressor: V4.1 splits what V4 fused

§12.11 found three differences in op 180's epilogue and called the third a factorization change.
It is, and the reference states the reason in the class docstring:

> Returns the latent before RoPE ... **Pre-RoPE is deliberate: the indexer needs the unrotated
> form, so Attention rotates afterwards.**

So `Compressor.forward` is pool -> norm and stops; `_compress_kv` ropes and fake-quantizes into the
cache only after the indexer has turned that latent into index keys. Op 180 gained `i7 = 2`, an arm
that stops after the norm, and the tail became **op 185 `CompressRopeQuant`**, which serves both
consumers of the latent because they differ in nothing else:

| consumer | reference | `i3` (qblk) | `i6` (qmode) |
|---|---|---|---|
| compressed KV | `fp4_act_quant(latent, 16, True, scale_dtype=e4m3)` | 16 | fp4 e2m1, **E4M3 scale** |
| index keys | `fp4_act_quant(k, fp4_block_size, True)` | 32 | fp4 e2m1, E8M0 scale |

`t0` and `t1` are separate buffers rather than the reference's in-place rope: the indexer reads
`src`, and an in-place rope is a write-after-read the packet order does not express. That costs one
`[n_rows][d]` bf16 buffer -- 8.4 MB at ratio 1 and 8k.

`cmp_fake_quant_block` now takes the mode at runtime, because the three call sites differ in value
format, scale format AND amax floor together, and the floor is chosen rather than conservative:
`6 * 2**-9` makes `amax / 6` exactly `2**-9`, e4m3's smallest subnormal, so an all-zero block gets
the smallest NONZERO scale. The E8M0 branch's `6 * 2**-126` rounds to zero in e4m3 and the dequant
would divide by it. `scripts/dsv41_csa2_oracle.py` grew four checks for this:

| check | result |
|---|---|
| op 180 (stop after norm) + op 185 vs `_compress_kv` | **max err 0.000e+00** |
| all-zero block's scale at the E4M3 floor / at the E8M0 floor | 2**-9 / **0** |
| index-key settings vs KV settings on the same row | 0.625 |
| the pre-RoPE tap vs the cache row it becomes | 7.52 |

Nine checks, all passing. What op 185 does NOT yet have is an emit site: the compressor emit is the
next piece, and it is one of the three Todo rows layers 2-39 still carry, beside the two-level
indexer and the gathered read.

### 12.16 All 40 layers emit: the indexer, and the read side of CSA2

Three Todo rows closed, and the count goes **2 of 40 to 40 of 40**. None of the three needed a new
algorithm; each needed one handle added to an op the tree already had, and the reading that says
which handle.

#### The selector op 55 was waiting for

Op 55's own doc names the blocker: *"What kept this without an emit site is the SELECTOR, not the
flash: a learned top-k needs T-row `IndexScore`/`IndexSelect`, which is a real design problem."*
Ops 117/118 ARE that pair — built for GLM-5.3's lightning indexer — and V4.1's indexer is the same
computation at the same `index_head_dim` 128 and the same 32 heads.
`scripts/dsv41_indexer_oracle.py` puts op 117 at **1.1e-07 relative** against `Indexer.forward`'s
own expression.

Three differences, and only one needed a kernel change:

  1. **The columns are POOLS.** V4.1's index keys are compressed entries, so a query reaches
     `(t + 1) / ratio` of them, not `t + 1` — `compress_lens = arange(1, seqlen+1) // ratio`
     (`model.py:564`). Op 118 already had `pool_size` at `i3`; op 117 did not, and now takes it at
     `i4` under the same zero-means-1 rule, so no existing blob changes. The two MUST agree, and a
     test holds them to the same number: a score written over a range the select never reads is
     merely wasted, but a select ranking columns the score never wrote reads the arena.
  2. **The indexer is REPLICATED**, and that is the tree's own standing recommendation rather than
     this emit's preference. `d_index_score_pf_row` says so in a `static_assert`:
     > `HIc % 32 == 0` — "the 32x32 MFMA A-tile is 32 index heads; HIc must be a multiple of it
     > (replicate a sharded indexer instead)" … "NOT generalized below 32: HIc = 16 or 8 (V4 at
     > TP4/TP8 with the indexer SHARDED) would leave half or three quarters of the A tile idle."
     > `[DSV4-IDX]`
     Column-parallel would give a rank 4 of 32 heads. The alternative — shard and all-reduce
     `index_score` — is a `[T][compress_len]` f32 reduce: **268 MB per layer** at 8k and ratio 1,
     on 8 layers. Replicating costs 5.2 MB of `wq_b` and 328 KB of `weights_proj` per rank and no
     collective at all. Both shard tables now say `Replicated`.
  3. **The two-level candidate stage is SKIPPED**, and that is a statement about 8k, not about the
     model. `candidate_topk_blocks` 2048 × `candidate_block_size` 8 covers 16384 compressed
     positions; 8k gives 8192 at ratio 1 and 4096 at ratio 2. Level one therefore keeps every
     reachable block, and — the part worth checking rather than asserting — the mask level two
     applies is a block-rounded **superset** of the reachability mask `Indexer.forward` has
     *already* applied, so it removes nothing. The oracle's first form of this check claimed the
     mask EQUALS reachability and failed with 68 differences; the true statement is the superset
     one. `emit_dsv41_indexer` asserts the block count rather than assuming it, so a longer context
     trips an assert instead of quietly over-selecting.

#### The read side: two partials, one softmax, one sink

`Attention.forward` runs **one** `sparse_attn` over `cat([window_kv, compress_kv])` with
`cat([window_idxs, compress_idxs])`, and `sparse_attn_kernel_` adds `attn_sink` to the denominator
once, after the last block. The emit produces that without materialising either concatenation:

| | op | writes |
|---|---|---|
| window | `FlashMlaPrefill` (51), 128-token window | partial 0 |
| compressed | `FlashGatherPrefill` (55), the CSA2 cache + the selection | partial 1 |
| | `FlashMerge` (13) at `nsplit = 2`, `t3 = attn_sink` | `O` |

Both flashes needed a **write-only output split**, `i7`: low 8 bits the partial index this call
fills, next 8 the total. It does NOT divide the KV range — that is the merge's `nsplit`, which this
then matches — it says where a call's partials LIVE, so two flashes over two DIFFERENT caches fill
disjoint halves of one `(Opart, mlpart)` pair. Inside `d_flash_mla_decode` it is two indices and a
`ONS`; `out_nsplit = 0` is the shipped layout and every existing emission keeps it.

The alternative was the reference's literal `torch.cat`: copy the SHARED compressed cache into a
per-layer buffer beside that layer's window K (8.4 MB per layer), and rewrite the index table by
`+window_len`. The split avoids both, and it avoids the question of what a shared cache is doing
inside a per-layer buffer.

Not everything got cheaper: **38 of 40 layers now run a second flash**, and 8 of them run an indexer
whose score is `[T][8192]` f32. Nothing here is measured yet — the emit is the claim, the hardware
run is the next one.

#### The compressor's tail, and one more missing build line

`PLOW_DSV4_CSA2` was never passed by `build_gfx942.sh` — the C side, the ISA, the dispatch and
`manifest.rs`'s `requires` all named it and nothing turned it into a `-D`. That is the **same
defect, in the same file, as §12.14's Engram line**, and this time it was found before it cost a
queue slot. It matters more now than it did: op 181 makes CSA2 reach every V4.1 layer rather than
only the four with a compressor, so an object without it refuses a layer-0 packet that used to
load.

#### What the count means, and what it does not

40 of 40 layers EMIT. That is the first half of the campaign goal and it is not the second: nothing
in this section has run on a GPU, and the last measured projection was ~843 ms against 90 ms with
38 layers doing strictly less work than they now do. The parts table is also per-layer, so it still
says nothing about the model TAIL — `Transformer.forward:1268` spends the dangling `ffn_pre` on one
final `hc_pre` — which a whole-model emit owes and a block emit cannot show.

A rung is also allowed to be a chain that reads shared CSA2 state nothing in it wrote: `--block 3`
attends over the cache layer 2 publishes. The emit now says so on stderr rather than leaving it to
be discovered, and `every_activation_read_is_written_by_something` runs over five chain shapes —
`[0]`, `[1]`, `[2,3]`, `[20]`, `[20,24]` — because no two of them read the same set.

### 12.17 The first compressed layer runs, and the gather trusted its selector's pad

Every compressed layer faulted. `--block 2`, `--block 2..4`, single-layer or chained: the packet
loaded (`weights_bound=true`), ran, and died with

    HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION

The `--segs` bisection is layer-granular — one cooperative launch per layer — so it could only
report "layer 2". Splitting this session's changes into a layer-0 packet (the always-on ones: op
142's interleaved rope bit, op 181, the two rope tables) and a layer-2 packet (those plus the
compressor, the indexer and the gathered read) narrowed it further: **layer 0 passed, layer 2
faulted**, so the fault was in the compressed path and not in anything the window-only layers had
started doing.

#### It was not in the new code

`d_index_select_pf` — op 118, shipped, GLM's selector — documents its own padding rule:

> Rows with `len <= top_k` emit the identity and pad with **-1**.

and `d_flash_mla_decode`'s GATHER arm read that table as

```c
const unsigned row = kv < hi ? (unsigned)ibase[kv] : 0u;
```

with no validity test. `(unsigned)(-1)` is `0xFFFFFFFF`, and `cbase + row * DK` is 4.4 TB past the
latent base — not a wrong number, an address outside the aperture.

What makes this reachable rather than theoretical is the **bound the gather actually walks**:

```c
const unsigned tk_live = GATHER ? (top_k < len ? top_k : len) : 0u;
```

`len` is `kv_len[b]`, the CHUNK-end token count — 8192 — not this query row's own length. So
`tk_live` is 512 for every row, while the number of slots op 118 actually filled for row `t` is
`min(top_k, (q_pos0 + t + 1) / pool_size)`. At `index_topk` 512 and pool ratio 2, **every query row
below 1023 has padded slots inside `[lo, hi)`**. A quarter of the rows in the first chunk walk off
the end of the cache.

This is why the window-only layers never showed it: they do not gather. It is the seventh instance
of the campaign's recurring class — the gather and the selector agree on the table's SHAPE and
disagree on what a slot MEANS — and the first where both halves were already in the tree and
already shipping.

#### Two phases, two different fixes

The score phase can branch, so it drops the slot outright:

```c
const int sel = GATHER ? (kv < hi ? ibase[kv] : -1) : 0;
const bool keep = GATHER ? (kv < hi && sel >= 0) : ...;
```

That is also the right arithmetic, not just a guard: the pad stands for a key that does not exist,
so it must not enter the softmax.

The PV phase cannot branch — it is the vectorized column read, `VU` rows in flight per thread, and
a per-slot test there costs the unroll. It does not need one. The slot's softmax weight is already
zero (its score was `-inf`), so clamping the ROW keeps the contribution exactly zero and the
address in range:

```c
__device__ __forceinline__ size_t fa_gather_row(int sel) {
    return (size_t)(unsigned)(sel < 0 ? 0 : sel);
}
```

The head-packed MFMA gather arm (`d_flash_mla_decode_mfma`, non-default but live) had the identical
cast in both of its row reads and gets the same clamp.

#### Layer 2 on hardware

| | median | exit |
|---|---|---|
| layer 0 — window only | 20,707 us | min -2.875 max 3.781 NaN 0 Inf 0 |
| layer 2 — compressor + indexer + gathered read | **36,380 us** | min -8.75 max 34.0 NaN 0 Inf 0 |

The compressed path costs **15.7 ms per layer** on top of the window flash, unoptimized and not yet
attributed between the compressor (op 180 + op 185), the indexer (ops 117/118, replicated across all
8 ranks) and the second flash. The indexer is the suspect: its score array is `[T][4096]` f32 and
§12.16 chose to replicate rather than shard it, with row-splitting the queries recorded as the next
move. That attribution is the next measurement, not a conclusion.

This is the first V4.1 layer to run its own attention as the reference defines it. It is not
evidence of correctness — the input is still a seeded synthetic and there is still no reference
parity harness — only that the shapes, the caches and the selection survive a real launch.

#### Where the 15.7 ms went, and it was not where §12.16 guessed

`--block 2..4` — layer 2 writes the cache and publishes the selection, 3 and 4 read both — runs in
**82.4 ms for three layers**, and its trace answers the attribution question §12.16 left open:

| op | calls | body / call |
|---|---|---|
| `COMPRESS_ROPE_QUANT` (185) | 3 | **3,231 us** |
| `FLASH_GATHER_PREFILL` (55) | 3 | 3,317 us |
| `INDEX_SELECT_PF` (118) | 1 | 1,152 us |
| `INDEX_SCORE_PF` (117) | 1 | 504 us |
| `COMPRESS_POOL` (180) | 1 | 61 us |

**The indexer is cheap.** §12.16 named it the suspect and recorded "row-split the queries instead"
as the next move; 1.7 ms of a 27 ms layer says that lever is not worth pulling, and the decision to
replicate rather than shard stands on its own measurement now instead of on the `static_assert` that
motivated it.

Op 185 was the cost, at **80x its own roofline** — it moves 67 MB and took 3.2 ms. The shape of the
error is the one a grid-strided megakernel op makes most easily: it claimed a whole WORKGROUP per
row and then found nothing for the workgroup to do. The indexer's queries are `d`=128 at `qblk`=32,
so the quant loop

```c
for (unsigned b = threadIdx.x; b < d / qblk; b += PLOW_THREADS)
```

ran **four** of 512 lanes, and the rope and staging loops before it ran 64 and 128. Three
`__syncthreads()` and four dependent global round trips, 862 times per workgroup at 8k — 3.75 us per
row, which is latency, not work.

Nothing about the op needs a workgroup. A quant block's scale depends on its own `qblk` channels and
on nothing else, and an interleaved (GPT-J) rope pair is two ADJACENT channels, so one thread can
own a block end to end: `cmp_rope_at` computes a channel's roped value from the two bf16 its pair
needs, and the kernel became a flat grid stride over (row, block) with no LDS and no barrier. The
block's channels are read twice — once for the amax, once for the round trip — which is one HBM trip
and one L1 hit, and cheaper than any buffer that would fit.

**9,692 us to 820 us, 11.8x**, and the three-layer chain from 82.4 ms to **73.3 ms**. The exit's min
and max are unchanged (-42.5, 316.0); the mean moves in its seventh digit, which is op 118 emitting
a selected row "in arbitrary order (the union build is order-blind)" and the gather therefore summing
it in a different order run to run — not the rewrite, which is value-identical by construction.

#### What 73.3 ms says about 90 ms

24.4 ms per layer, so a 40-layer model is ~977 ms against a 90 ms target. The gap is **not** in the
V4.1 machinery this campaign added. Per layer, from the same trace:

| op | per layer | what it is |
|---|---|---|
| `GEMV_F32` | 3.33 ms | the mHC's own GEMV — §12.13's open item |
| `FLASH_GATHER_PREFILL` | 3.29 ms | the compressed read (512 slots vs the window's 128) |
| `MOE_GROUP_DOWN_PF` | 2.76 ms | |
| `GEMM_FP8_MX` | 2.67 ms | 25 calls |
| `HYPER_CONN_PRE` | 2.52 ms | |
| `XREDUCE2` | 2.05 ms | 674 of its 684 us per packet is STRAGGLER — TP imbalance, not work |
| `MOE_GROUP_GLU_PF` | 1.94 ms | |
| `FLASH_MLA_PREFILL` | 1.42 ms | the 128-token window |

Six of the eight predate this session. The gathered flash is the only new line in the top half, and
at 4x the window's KV for 2.3x its time it is already sublinear — it is the model's design cost, not
a defect. The 90 ms target is a GEMM/MoE/collective problem, and the first two numbers to chase are
`GEMV_F32` and `XREDUCE2`'s straggler, neither of which is about V4.1 at all.

### 12.18 All 40 layers RUN: 992 ms against 90 ms, and where every millisecond is

`--block 0..39` — **40 layers, 1,513 prefill ops**, a 20.4 MB packet — loads on 8x MI300X and runs.

```
loaded in 306.1 s: tp=8, weights_bound=true
exit: 167772160 elems  min -6784.0  max 6752.0  mean 0.149356  zero 0  NaN 0  Inf 0
layer time over 3 iters: min 982387.2 us  median 991968.6 us  max 1006351.4 us
```

This is the first half of the campaign goal. It is not the second: **992 ms against a 90 ms
target**, 11x.

#### The census is the correctness evidence, such as it is

The trace's per-op call counts are the one structural check available without a parity harness,
and every one of them is a number the config predicts and a wrong emit would miss:

| op | calls | why that number |
|---|---|---|
| `COMPRESS_POOL` (180) | **3** | 4 kv_source layers, but layer 20 is ratio 1 — a plain projection with no softmax gate, so it pools nothing |
| `COMPRESS_ROPE_QUANT` (185) | **16** | 4 compressed-KV + 4 index-key (only the kv_source layers own keys) + 8 index-query |
| `INDEX_SCORE_PF` / `INDEX_SELECT_PF` | **8** each | `index_source_layer_ids` = [2, 8, 14, 20, 24, 28, 32, 36] |
| `FLASH_GATHER_PREFILL` (55) | **38** | every layer whose `compress_ratios` entry is non-zero — all but 0 and 1 |
| `FLASH_MLA_PREFILL` (51) | **40** | the 128-token window, every layer |
| `ROPE_INVERSE_O` (181) | **40** | `Attention.forward` runs it unconditionally (model.py:781) |
| `ENGRAM_GATE` / `ENGRAM_EMBED` | **2** each | `engram_layer_ids` = [1, 14] |

None of this says the numbers are right — the input is still a seeded synthetic, there is still no
reference parity harness, and an exit spanning +-6.8e3 after 40 layers with no final norm is
plausible rather than checked. It says the SHAPE of the model that ran is the shape of V4.1.

#### The 90 ms budget, itemized

980.4 ms of body across 1,513 packets:

| op | total | calls | per call | share |
|---|---|---|---|---|
| `GEMV_F32` | 134.8 ms | 80 | 1.68 ms | 13.7% |
| `GEMM_FP8_MX` | 129.9 ms | 330 | 0.39 ms | 13.2% |
| `FLASH_GATHER_PREFILL` | 126.3 ms | 38 | 3.32 ms | 12.9% |
| `MOE_GROUP_DOWN_PF` | 109.5 ms | 40 | 2.73 ms | 11.2% |
| `HYPER_CONN_PRE` | 102.8 ms | 80 | 1.28 ms | 10.5% |
| `XREDUCE2` | 86.1 ms | 122 | 0.70 ms | 8.8% |
| `MOE_GROUP_GLU_PF` | 77.0 ms | 40 | 1.92 ms | 7.9% |
| `FLASH_MLA_PREFILL` | 57.8 ms | 40 | 1.44 ms | 5.9% |
| `MOE_COMBINE_PF` | 34.5 ms | 40 | 0.86 ms | 3.5% |
| everything else | 121.7 ms | 903 | | 12.4% |

**Everything V4.1-specific is in the noise.** The compressor, the indexer and the inverse rope
together are 27.6 ms — 2.8% — and after §12.17's rewrite op 185 is 6.6 ms of that. The gathered
flash is the one new line that matters at 126 ms, and it is the model's design cost: 512 selected
pools against the window's 128, for 2.3x the window flash's time.

So the 90 ms target is a **GEMM, MoE and collective** problem, and three lines are worth naming:

* **`GEMV_F32`, 134.8 ms.** Two calls per layer at 1.68 ms each, and §12.13 already took 2.8 ms out
  of this same mHC GEMV once. It is the largest single line in the model and it is a GEMV.
* **`XREDUCE2`, 86.1 ms.** 690 of its 699 us per packet is STRAGGLER — the spread between the first
  and last workgroup to finish. That is TP imbalance, not arithmetic, so it is the cheapest 80 ms
  on the list if the imbalance has a cause rather than a cost.
* **`HYPER_CONN_PRE`, 102.8 ms** at 1.28 ms x 80 against `HYPER_CONN_POST`'s 0.32 ms x 80. The two
  halves of the same mHC differ by 4x and nothing about their shapes says they should.

Getting to 90 ms means roughly an 11x, which no single one of these delivers. It is a campaign, and
this section is its baseline rather than its conclusion.

### 12.19 The mHC GEMV was never bandwidth-bound, it was issue-bound

`GEMV_F32` was the largest single line in §12.18 at 134.8 ms, and its own header records five
ownership maps measured against each other — block-per-row, wave-per-row, all-24-in-registers,
flattened — spanning a 2.4x range and all landing 25-50x off the 63 us that 335 MB of `x` at HBM
bandwidth would cost. The header's closing line is "the op is not done; it is measured."

Five maps all missing by the same order of magnitude is the signal. **None of them changed the
instruction mix.** Every arm walks K as

```c
for (unsigned k = lane; k < K; k += PLOW_WAVE)
```

which is ONE element per lane per step: a 2-byte `x` read and CG 4-byte `W` reads, so a group of 8
columns costs **nine load instructions per eight FMAs**. At that ratio the kernel is issue-bound
long before it is bandwidth-bound, and rearranging WHICH wave owns (m, n) cannot help — which is
precisely what the five measurements say, in retrospect.

Arm 5 takes eight k per lane: one `bf16v8` of `x` and CG pairs of `f32x4` of `W`, **17 vector loads
per 8 k-steps against 72 scalar ones** — the same bytes, 4.2x fewer instructions.

**134.8 ms to 68.9 ms, 1.96x**, and the straggler with it, 193 to 76 us per packet. The whole model
goes 992 to 921 ms.

It is the first arm that is NOT bit-identical to the others: a lane now folds eight consecutive k
before the next stride, so `wave_sum` sees a differently-associated f32 sum of 20480 terms. On the
40-layer chain the exit extremes move visibly (-6784/6752 to -9344/18304) while the mean barely
does (0.1494 to 0.1518) — which is what a chaotic 40-deep residual stream does to any perturbation,
and NOT something to accept on that argument alone. **Layer 0 alone settles it**: one layer, nothing
downstream to amplify, and the exit is identical to the arm-4 baseline to every printed digit —
min -2.875, max 3.78125, mean -0.000706 — at 18,779 us against 20,707 us, a 1.93 ms saving that is
exactly the layer's two GEMV calls.

At 68.9 ms this is still ~9x off roofline and the op is still not done.

#### `HYPER_CONN_PRE` is the same size and a different problem

102.8 ms, 1.29 ms per call, against `HYPER_CONN_POST`'s 0.32 ms — and the two move almost the same
bytes. POST reads the residual once, `x_out` once and writes the residual: 754 MB, 215 us at
roofline, measured 319 us. **POST is at 1.5x roofline.** PRE reads the residual TWICE (the
sum-of-squares, then the collapse) and writes `layer_input`: also ~754 MB, also 215 us at roofline,
measured 1,285 us. **6x.**

Same memory, same access pattern, 4x apart — so PRE's gap is not its loads. It is the **serial
section**: one workgroup owns one token, lane 0 runs the ~1,300 dependent scalar ops of the 20
Sinkhorn iterations while 511 lanes idle, and a block does that 27 times (8192 tokens / 304 blocks)
with a `block_sum` and two `__syncthreads()` between each. The op's own comment already names why
nothing hides it — LDS caps the interpreter at one workgroup per CU.

The fix this points at is a WAVE per token rather than a workgroup: reduction 1 becomes a
`wave_sum` over 320 elements per lane, reduction 2 gives each lane 80 of `hidden`, the per-wave
`logits`/`comb` strips are 320 floats total, and eight serial sections then run concurrently
instead of one — with no barrier at all. Gated on `T >= nblk && n == 4`, as the register-resident
4x4 above it already is, so decode and the `head_only` and `n != 4` paths keep the shipped block.
Not attempted here.

### 12.20 `HyperConnPre` was not slow, it was SERIAL

§12.19 left this one diagnosed and not attempted. The diagnosis held exactly, and it is worth
stating why it was checkable in advance rather than by trying things.

`HYPER_CONN_PRE` and `HYPER_CONN_POST` are the two halves of the same mHC and move almost the same
bytes: POST reads the residual once, `x_out` once and writes the residual; PRE reads the residual
TWICE — once for the sum of squares, once for the collapse — and writes `layer_input`. Both about
754 MB, both 215 us at roofline. POST measured 319 us. **POST is at 1.5x roofline; PRE was at 6x.**

Same memory, same access pattern, 4x apart. That is not a bandwidth result, and it says so before
any experiment: the only thing PRE has that POST does not is the serial section. One workgroup owns
one token, lane 0 runs the ~1,300 dependent scalar ops of 20 Sinkhorn iterations while 511 lanes
idle, and a block does that 27 times (8192 tokens over 304 blocks) with a `block_sum` and two
`__syncthreads()` between each. Nothing hides it, because LDS caps the interpreter at one workgroup
per CU — which the op's own comment already said.

A **wave** per token runs eight of those concurrently and needs no barrier at all:

| | workgroup per token | wave per token |
|---|---|---|
| sum of squares | 512 lanes, 40 elements each, `block_sum` | 64 lanes, 320 each (8-wide loads), `wave_sum` |
| the gate | LDS + `__syncthreads()` | `__shfl` from lane 0 |
| the collapse | 512 lanes over `hidden` | 64 lanes, 80 of `hidden` each |
| serial sections in flight per CU | 1 | 8 |

**102.8 ms to 25.7 ms, 4.0x**, straggler 102 to 83 us. PRE now costs **321 us per call against
POST's 323** — the parity the diagnosis predicted, and a better sign that the op is done than the
speedup is. The whole model goes 921 to **854 ms**.

The 4x4 Sinkhorn is lifted into `hc_sinkhorn4` so the new arm and the shipped `n == 4` branch share
one copy. It is the most error-prone block in the file and two copies of it is not a trade worth
making.

Like §12.19's GEMV arm it is **not bit-identical** — `inv` is now a 64-lane reduction of
320-element partials where it was a 512-lane reduction of 40-element ones — so it sits behind
`PLOW_HC_WAVE_TOKEN` (default 1) rather than being taken silently. GLM-5.3's mHC is `n = 4` as
well, and a V4.1 change should not quietly reassociate another model's reduction. Layer 0 alone
settles the numerics the same way §12.19 did: the same exit to every printed digit — min -2.875,
max 3.78125, mean -0.000706 — at 17,018 us.

#### Layer 0 across this session

| | layer 0 median | whole model |
|---|---|---|
| session start | 20,707 us | — |
| + §12.19 GEMV arm 5 | 18,779 us | 992 -> 921 ms |
| + §12.20 wave per token | **17,018 us** | 921 -> **854 ms** |

Two ops, 138 ms off the model, and neither was a V4.1 op. The pattern both share is the one worth
carrying forward: **a megakernel op that is 5-50x off its own roofline is almost never short of
bandwidth.** It is issuing one element per lane per step (§12.19) or running a serial section with
nothing resident to hide it (§12.20), and both are visible by reading the loop and comparing
against a sibling op that does the same traffic.

### 12.21 The largest GEMM's tile could not be shown to take, and it is already the finest legal one

`GEMM_FP8_MX` (op 184) is 123.8 ms — the largest GEMM line in the model and second-largest line
overall. Every build this campaign has run ended with

```
FAIL  geom_contract.h: GM_MX_BM is a tunable in runtime/amd but has no PLOW_GEOM_MARK line,
      so a -DGM_MX_BM could not be shown to have taken
FAIL  geom_contract.h: GM_MX_BN ...
```

and every one of those builds was waved through as "pre-existing, not mine". It was not noise. Its
own file says what the rule is for: *"A knob added without a marker is therefore a build failure"* —
because without the marker nothing proves a `-D` reached the object. **The tile knobs of the
model's biggest GEMM were untunable, and the audit had been saying so all along.** Two
`PLOW_GEOM_MARK` lines fix it, and the build goes **rc=1 to rc=0** — the first clean audit of this
campaign.

#### What the now-measurable knob measures

| tile | `GEMM_FP8_MX` | straggler/pk | whole model |
|---|---|---|---|
| 128 x 128 (default) | **123.8 ms** | 164 us | **854 ms** |
| 256 x 128 | 138.9 ms | 279 us | 865 ms |
| 128 x 64, 64 x 128 | *will not compile* | | |

The default wins, and the two that would have helped do not exist. `d_gemm_t`'s own static asserts
say why:

* `BN % (WN * MFMA_N) == 0` with `WN=4`, `MFMA_N=32` forces **BN % 128 == 0**.
* `APT % 8 == 0` ("tile must stage in 16-byte units") with `APT = BM*BK/512` at **BK=32** forces
  **BM % 128 == 0**.

So the entire legal tile space here is {128, 256} x {128, 256}, and 128 x 128 is already the
FINEST member of it. That matters because of what the straggler is: 164 us of a 375 us body is a
**44% tail**, and at N=1280 a 128 x 128 tile gives (8192/128) x (1280/128) = 640 tiles over 304 CUs
= 2.1 waves, whose last wave is ~10% occupied. The arithmetic and the measurement agree, which
means the tail is not a tuning miss — **it is structural, and no legal tile fixes it.**

The route out is the one the kernel's own header already named and had no number for:

> BK=32 IS THE POINT, NOT AN ARBITRARY TILE. ... The cost is a halved K step, i.e. twice the LDS
> traffic per MFMA, and it is UNMEASURED -- a BK=64 variant promoting twice per tile is the
> optimization to try once there is a number to beat.

There is now a number to beat (123.8 ms), and a second reason to want BK=64 that the note does not
mention: at BK=64, `APT = BM*64/512` makes **BM=64 stage legally**, so BK=64 is not just half the
LDS traffic — it is the only way to reach the finer tile that would close the 44% tail. Both wins
come from the same change. Not attempted here; it is a real change inside a heavily-templated
`d_gemm_t`.

#### And the tuning database cannot help, for a nameable reason

The emit warns on every build:

> `tunedb amd/gfx942/mi300x: 4961 record(s) skipped as STALE ... NO usable records remain, so tile
> selection fell back to the analytical model` — `pick_tile` reports tier `portable`, "what it
> reports when no campaign has ever run."

`plowc tune gemm --shapes auto` derives the compiler's demand by **running a real emit**, so it
takes the whole-model path and hits `deepseek_v41: no device emit yet` — it never sees `--block`.
V4.1 is therefore in exactly the category the CLI documents an escape for ("the models `auto`
cannot yet reach (Kimi-K3 has no full-model emit, so its demand cannot be observed)"), and a
hand-written `--shapes <FILE>` is the way in. Also worth recording: the tuning digest includes the
INTERPRETER, so every kernel change in §12.19-§12.21 invalidates the database again. **A tuning
campaign has to be the last thing done, not the first.**

### 12.22 The shared expert was counted EIGHT times, and it bought 40 all-reduces to do it

Chasing the 90 ms target into the collectives turned up a correctness bug, which is the seventh
instance of this campaign's recurring class and by a wide margin the largest.

**The count that started it.** The 40-layer trace has **122** `XREDUCE2` calls. TP8 needs two per
layer — one after the attention output projection, one after the MoE — plus Engram's two at layers
1 and 14. That is 82. There were forty too many, one per layer, and they were the shared expert's.

**Why a third one is not just waste.** `d_moe_combine_pf` computes

    out[t] = residual[t] + shared[t] + SUM_slot part[t*k + slot]

and the band all-reduce that FOLLOWS it sums `out` across all `tp` ranks. So whatever is handed to
`shared` gets summed `tp` times. GLM settles what that operand means by construction: its own body
emits the shared-expert down projection straight into `n.shared` with nothing between that GEMM and
the combine, so `n.shared` is the row-parallel **PARTIAL**.

V4.1 cannot reuse that body — its shared expert is block-FP8 while its routed experts are MXFP4, so
the caller emits the GEMM itself and passes the result back. It passed `sh_out`: the buffer its own
struct documents as *"Shared-expert output after the cross-rank sum"*. Every rank therefore added
the FULL shared expert before a sum over eight of them.

`scripts/dsv41_moe_tp_oracle.py` prices it, and the third check is the one that names the defect
rather than just detecting it:

```
  PASS  the row-parallel split reproduces the shared expert exactly   max|err| = 9.537e-07
  PASS  handing the combine the PARTIAL reproduces the reference      max|err| = 7.153e-07
  PASS  handing it the REDUCED buffer does not                        max|err| = 2.134e+01
  PASS  and the error is exactly (tp - 1) copies of the shared expert max|err| = 7.629e-06
  PASS  it is not a rounding detail        relative error 6.595 of the layer's FFN output
  PASS  every buffer is the same shape either way, and both are finite
```

**659% relative error on every layer's FFN output**, with every buffer the right shape and every
value finite. No shape check, no NaN check and no "it runs" can see it — which is the whole point
of the class, and the reason the fix had to be an oracle and not an inspection.

**A test was pinning it.** `the_shared_experts_partial_lands_in_a_peer_slot_and_is_summed` asserted
that the shared partial IS separately reduced. Its reasoning was sound as far as it went — a
partial must live in a peer slot, because `d_xreduce` sums `peer_scratch[r] + slot` and never reads
`out` — but it did not ask WHO does the summing, and the answer is the combine's own band reduce.
The test is now
`the_shared_experts_partial_is_consumed_by_the_combine_not_reduced_twice` and pins the operand
(`t2 == PEER_SLOT_SHARED`) and the collective count (two per layer, not three). The peer-slot
residency stands; the combine reads slot 0 LOCALLY, since a rank only ever reads its own partial.

**On hardware**, 40 fewer collectives and a different answer:

| | before | after |
|---|---|---|
| prefill ops | 1,513 | **1,473** |
| `XREDUCE2` | 122 calls, 87.2 ms | **82 calls, 65.1 ms** |
| whole model | 852.6 ms | **841.6 ms** |
| exit mean | 0.1499 | **0.0496** |

The mean moving by 3x is the confirmation that matters. Deleting a genuinely redundant collective
changes the timing and nothing else; this changed the answer, which is what a `tp`-times-too-large
shared expert would do.

### 12.23 The MoE 86->87 fusion was unreachable at top-6, and is a LOSS once reached

`PLOW_MOE_PF_DET` fuses the grouped MoE's DOWN epilogue into its combine: op 86 accumulates
`rint(gate*value * 2^32)` into a `[T, H]` f64 accumulator with an order-independent atomic, and op
87 then reads ONE contiguous stream instead of `k` slot streams at `H*4` stride. GLM measured
**-980 us/layer** for it and it is DEFAULT ON in the gfx942 object build.

**V4.1 could not use it, and nothing said so.** The emit gated on

```rust
if moe_pf_det() && tk != 0 && tk.is_power_of_two() && tk <= 16 { MoePfFuse::Det }
```

because the epilogue recovers the token as `pidx >> log2(k)` from `row_partidx[row] == token*k +
slot`. **V4.1 routes top-6.** So the arm was compiled into every object this campaign built, cost
its LDS and its marker symbol, and silently never fired — the emit fell through to `MoePfFuse::None`
with no diagnostic.

#### Reaching it costs nothing, because the table already exists

A division in the innermost epilogue loop is not the answer. But `d_moe_align_pf` already writes
`row_token[pos] = s / k` — one host-side division per row — over the same padded range, with the
same `PLOW_EXPERT_UNUSED` sentinel in the padding. In the DET arm the `row_partidx` operand is used
for *nothing but* deriving that token, so the emit can bind `row_token` in its slot and the
epilogue does **no arithmetic at all**:

* `det_ksh` **1..5** = `log2(k)+1`, the shipped encoding, t6 is `row_partidx`, one shift.
* `det_ksh` **>= 32** = `32+k`, t6 is `row_token`, `pidx` IS the token.

The split keeps every existing blob byte-identical — a GLM blob still emits 1..5 and still takes
the shift — and needs no new operand slot, which matters because op 86 has all eight in use.

#### And then it loses

| op | DET off | DET on |
|---|---|---|
| `MOE_COMBINE_PF` (87) | 34.0 ms | **12.2 ms** (-21.8) |
| `MOE_ROUTER_TOPK_PF` (83) | 14.2 ms | 17.5 ms (+3.3, the accumulator zeroing prologue) |
| `MOE_GROUP_DOWN_PF` (86) | 117.4 ms | **153.1 ms** (+35.7) |
| whole model | **841.6 ms** | 855.8 ms |

The read side does exactly what GLM's note promises — op 87 goes 2.8x faster on one contiguous
stream. The write side loses more. GLM's own note says "the atomic itself buys nothing — it COSTS
44.8 us/layer"; at V4.1's shape it costs **893 us/layer**, because `k=6` contributions per token
land as f64 atomics into a `[8192, 5120]` accumulator — 336 MB, six times over — where the shipped
scatter writes `[T*k, H]` f32 once and streams it.

So the arm stays OFF for V4.1. What was worth doing is making it REACHABLE: "this optimization does
not apply to your model" and "this optimization silently does not apply to your model because your
top-k is not a power of two" are different states, and only the first is a measurement. The
generalization is kept, default off, with the number attached.

#### Three knobs measured and rejected, which is also a result

| axis | what it targets | verdict at V4.1's shape |
|---|---|---|
| `GM_MX_BM/BN` 256x128 | op 184's tile | **worse**: 123.8 -> 138.9 ms, straggler 164 -> 279 us |
| `PLOW_MOE_PF_DET` | fuse op 86 -> 87 | **worse**: 841.6 -> 855.8 ms (87 wins 21.8, 86 loses 35.7) |
| `PLOW_MOE_PF_GH=2` | hoist + pipeline the MoE A-gather index | **noise**: 839.0 -> 837.2 ms, inside the +-8 ms spread; GLU unchanged at 78.4 ms |

`PLOW_COMBINE_VEC` was already default ON, so op 87's 8-wide arm was not a lever either. The
shipped defaults are, on this model, the right ones everywhere they were tested — which is worth
knowing before writing a kernel, and is why the audit fix in §12.21 mattered more than any tile it
went on to reject.

### 12.24 The largest GEMM's k-tile was held at 32 by a constraint that was never about BK

§12.21 closed with op 184 at 128x128 and the note that this is the finest legal tile, because
`APT = BM*BK/512` needs `BM >= 128` once `BK == 32`. It also recorded, from `d_gemm_fp8_mx`'s own
header comment, why BK could not move:

> BK=32 IS THE POINT, NOT AN ARBITRARY TILE. The promotion must land on a k-tile boundary [...] and
> a 32-element K block does not divide BK=64. [...] it would land mid-burst.

**The premise is false, and the kernel already contained the counter-example.** A k-tile is not one
MFMA burst. `d_gemm_t` splits every tile into `NSL = BK / GM_SLICE` MEM/MFMA cluster pairs, and
`GM_SLICE` is **16**. The burst is per cluster. So the real constraint on the 32-element MX scale
block is `32 % SLICE == 0` — which BK does not enter at all. At `BK=64, SLICE=16` the scale
boundary falls between cluster 1 and cluster 2, and again at the end of the tile: both are cluster
boundaries, outside every burst, which is the one property the promotion ever needed. The sibling
`WFP8BLK` arm's own comment says as much about its 128-element block ("exactly two BK=64 tiles"),
and nobody asked what a *cluster* was.

#### Moving the promotion is bit-identical at BK=32, and that was checked rather than argued

`GM_MX_PROMOTE(sl)` drains `acc` into `accf` when `((sl+1)*SLICE) % 32 == 0`, with
`kb = kt*(BK/32) + (sl+1)*SLICE/32 - 1`. Every factor is a compile-time constant, so off the
boundary it folds to nothing. At `BK=32` the condition is true exactly once — on the last cluster,
with `kb == kt` — so it is the tile-level site it replaced, spelled differently.

That was verified by building the same emit twice, once per header, and diffing disassembly:

* 137,122 instruction lines each, **zero opcode or register differences**;
* 11 differing lines, all `s_add_u32 sX, sX, <imm>` PC-relative fixups, all by exactly `+0x40`;
* `llvm-readelf -s` accounts for it — the new object carries one extra symbol,
  `plow_geom_GM_MX_BK`, which shifts the data section by 64 B.

Object BYTES differ across all 54 objects either way, because the build embeds its own
`PLOW_HSACO_CONFIG` path; byte comparison is not the tool for this question and disassembly is.

#### And then BK=64 wins, while the tile it unlocks loses

| op 184 tile | body (330 pkts) | straggler/pk | whole model |
|---|---|---|---|
| 128x128, BK=32 | 124.1 ms | 164 us | 839.0 ms |
| **128x128, BK=64** | **118.5 ms** | 157 us | **827.2 ms** |
| 64x128, BK=64 | 146.4 ms | **117 us** | 860.0 ms |

BK=64 takes **5.6 ms off op 184 (-4.5%)** while its neighbours move under 1% (GEMV_F32 -0.3%,
XREDUCE2 +0.8%, FLASH_MLA_PREFILL -0.9%) — the win is attributable to the op it targets, which is
the test the GH=2 knob in §12.23 failed. Confirmed twice at 827.2 ms, once via the env knob and
once with 64 as the committed header default (827,152 and 827,242 us).

**The 64-row tile is the interesting negative.** BK=64 is what makes it stage legally at all, and
it does exactly what §12.21 predicted: the ~44% structural tail collapses, 157 -> 117 us/pk. It
also costs 28 ms of body to do it. The tail was never the cost. Halving the rows halves the MFMA
work each pass over B amortises, and op 184's B is the whole weight — so the fix for a straggler
was a 24% slower kernel. A tail is worth closing only when the work that hides it is free, and
here it is the work.

### 12.25 The MoE GEMMs do almost no GEMM, and two of the three ranked levers are dead

§12.18 ranked the remaining work by op cost. `MOE_GROUP_DOWN_PF` (112.9 ms) and
`MOE_GROUP_GLU_PF` (74.7 ms) are 23% of the run between them, and the ranked fixes were (2)
vectorize the DOWN epilogue, whose own note records `global_store_short`, one element per store,
and (3) `PLOW_MOE_PF_A8`, fp8 A rows to halve LDS traffic and double the MFMA rate.

Both were ranked from a disassembly reading. Neither survives a measurement.

#### Instrument 1: cap the k-loop (`PLOW_MOE_PF_ABL=1`, already in the tree)

`NT = NT_full > 1 ? 1 : NT_full` computes a 1/NT slice of each dot product — wrong output, but
every fixed cost (staging, routing, gather/scatter, epilogue, collectives) is still paid. At
V4.1's shape DOWN runs NT=5 k-tiles (K = moe_inter/TP = 288, MPF_BK=64) and GLU runs NT=80
(K = 5120). So this removes **80% of DOWN's and 98.75% of GLU's MFMA work**:

| op | full | k-loop capped at 1 tile | delta |
|---|---|---|---|
| `MOE_GROUP_DOWN_PF` | 112,875 us | 112,604 us | **-0.24%** |
| `MOE_GROUP_GLU_PF` | 74,745 us | 73,128 us | -2.2% |
| whole model | 827.2 ms | 826.2 ms | -1.0 ms |

Deleting 98.75% of an op's multiply-accumulate makes it 2.2% faster. **The grouped MoE GEMMs are
not FLOP-bound, not weight-bandwidth-bound, and not k-loop-bound at all** — capping NT also cuts
the weight stream 5x on DOWN and 80x on GLU, and that moves nothing either. Lever (3) is dead on
arrival: fp8 A rows double a matrix rate that is already free.

(The ablation is confirmed to engage: the object is 505 instructions smaller.)

#### Instrument 2: cap the store (`PLOW_MOE_PF_EPIABL=1`, added here)

Lever (2) said the cost was the store SHAPE — one `global_store` per output element, because a
lane's 16 accumulator elements are 16 different ROWS at one column, so consecutive elements are
`N` apart. The fix would be an LDS transpose so a lane writes contiguous `N`. Before building
that, the same ablation asks the same question: issue **1 of every 16** stores, keeping the MFMA,
the row metadata, the gate multiply and the address arithmetic.

| op | full | 1/16 of the stores | delta |
|---|---|---|---|
| `MOE_GROUP_DOWN_PF` | 112,875 us | 116,463 us | **+3.2%** |

Removing 94% of the stores makes the op *slower*. Lever (2) is dead too. This is consistent with
the one prior datum: the `part16` arm halved the same stream's bytes and measured ~0%, and its
note blamed the shape. The shape is not it either — **the scatter, in bytes and in instruction
count, is not what op 86 is spending its time on.**

#### What this leaves

Of §12.18's ranked list, (1) op 184's tile is done and paid (§12.24), and (2) and (3) are now
falsified by measurement rather than deferred. The 187.6 ms in the two MoE ops is in neither the
math, nor the weight stream, nor the output stream. It is in what is left: the per-tile staging
and barriers, the A-gather indirection, and the routing/alignment metadata — the fixed cost of
**27,648 output tiles per layer per rank**. That is a different kind of fix from every one
attempted in this campaign: not a wider load, but fewer, larger tiles. It is also the reading most
consistent with §12.23, where the DET fusion's f64 atomic cost 893 us/layer against GLM's 44.8 —
V4.1's MoE is dominated by per-tile and per-row overheads that scale with its 384 experts and
top-6, not by anything that scales with FLOPs or bytes.

**The method is the transferable part.** Three of this campaign's wins came from reading
disassembly and finding narrow issue (§12.19, §12.20, §12.17's op 185). The same reading of op 86
produced two confident, specific, wrong predictions. A ceiling instrument that deletes the
suspected work and re-measures costs one build and one run, and it is the only thing that
separates "this code looks expensive" from "this code is expensive".

### 12.26 Fewer, larger MoE tiles: -72 ms, and the emit had been asking for it all along

§12.25 ended with the only reading left: the 187.6 ms in the two MoE ops is the fixed cost of
**27,648 output tiles per layer per rank**, and the fix is fewer, larger tiles rather than a wider
load. `scripts/build_gfx942.sh` already carried that hypothesis, written for a different model:

> TP8 is what makes its overhead PER-TILE rather than per-byte [...] Raising BM halves the tile
> count and so halves the fixed cost that k-loop is too short to amortize.

and an `MPF_BM` escape hatch to test it. It also carried the reason the test is not free: both
DOWN metadata hoists `#error` unless `MPF_BM == PLOW_WAVE == 64`, so a BM=128 arm must be compared
against a BM=64 arm **with the hoists off too**, or the measurement is the hoist and not the tile.
(V4.1's routed experts are MXFP4, so its DOWN runs `d_moe_group_pf_a4w4` and the live hoist is
`PLOW_MOE_PF_EPI_SIB`, which the K3 A4W4 row forces on independently of `PLOW_MOE_PF_EPI`. Both
had to come off.)

| | DOWN | GLU | whole model |
|---|---|---|---|
| BM=64, hoists on (shipped) | 112.9 ms | 74.7 ms | 827.2 ms |
| BM=64, hoists off (control) | 119.4 ms | 74.0 ms | 833.8 ms |
| **BM=128, hoists off** | **72.0 ms** | **49.7 ms** | **755.4 ms** |

**DOWN -39.7%, GLU -32.9%, the model -71.8 ms** against what was shipped. The metadata hoist is
worth ~6.6 ms and cannot ride the larger tile; 78.5 ms beats 6.6 ms, so V4.1 takes the tile.
Confirmed three times at 755.4 / 755.6 / 755.9 ms.

This is the first fix of the campaign that is not "issue wider loads". Every earlier win (§12.17
op 185, §12.19 GEMV, §12.20 HyperConnPre) was one lane-serial inner loop made vector. This one
does not touch an instruction: it changes how much work a tile amortises its fixed cost over. It
was reachable only because the two ablations in §12.25 ruled out the math and the stores first —
the disassembly reading pointed at the stores, twice, with confidence, and was wrong.

#### The emit had already been sized for it

`crates/devgen/src/mla.rs:1918` declares `MPF_BM = 128` host-side, with a comment saying the align
op pads each expert to the OBJECT's tile height so the buffer bound must cover the largest
variant. So the packet's gathered-row arrays were **already** allocated at
`T*k + n_exp*(MPF_BM-1)` for BM=128, and the prefill objects had been tiling at 64 inside them.
Raising the object's tile does not enlarge any allocation; it stops under-using one. That is why
this is a build-flag change and not an emit change.

The default is scoped to `PLOW_PREFILL_DSV41=1` so no other model's objects move, and an explicit
`MPF_BM` from the caller still wins.

#### Unresolved: the exit magnitude is wider at BM=128

`rung_run` feeds a seeded synthetic and reports only `min/max/mean/NaN/Inf`. Across runs the exit
mean is stable (0.054-0.061) and there is never a NaN or an Inf, but the magnitude range at BM=128
is visibly wider than at BM=64: every BM=64 run of this campaign landed within +-8,160, while
BM=128 gave 1,080 / 73,728 / 137,216 on three runs of identical objects. The harness is not
run-to-run deterministic (the MoE routing and its atomics reassociate), and a different tile
changes the f32 summation order in a way 40 layers will amplify, so this is consistent with
numerical wander. **It is also exactly what a padding bug would look like, and this campaign has
no reference parity harness to tell the two apart** -- see §12.2. The buffer itself is not the
suspect: the emit sizes it for 128 deliberately. Recorded here rather than resolved.

### 12.27 The largest op runs a scalar body ON PURPOSE, and its MFMA replacement is built but unwired

After §12.24 and §12.26, `FLASH_GATHER_PREFILL` is the largest single op in the run: **125.1 ms**,
3,293 us per packet over 38 packets. Its arithmetic is about 73 GFLOP per layer per rank
(QK over top-512 selected keys, then PV), so it is running at roughly **22 TFLOP/s — under 2% of
this part's bf16 peak.** That is the 5-50x-off-roofline signature that paid three times in this
campaign.

It is not an oversight. `interp.hip` says so at the dispatch:

> Dense only. The gathered arm above keeps the SCALAR body: its top_k set is per QUERY, so a tile
> of query rows shares no KV range to stage.

Op 55's index table is one top-512 row per query token, so 64 queries in a tile select 64
different key sets, and there is nothing to stage in LDS or feed an MFMA fragment with. Given that
operand shape the scalar body is the correct kernel. **The fix is to change the operand, not the
kernel** — and plow already contains the machinery for it.

#### The DSA sparse-prefill chain exists, and one of its four pieces is missing

`op_attention_common.h` §[GLM52-DSA-PF] documents a four-piece chain built for exactly this:

* **op 117** `d_index_score_pf` — the T-row lightning-indexer score, **MFMA**, one
  (query, 32-position) subtile per wave. Its `pool_size` arm already implements
  **DeepSeek-V4.1's own** `compress_lens = arange(1, seqlen+1) // ratio` (`model.py:564`), floor
  rule and all, and it `static_assert`s `index_n_heads == 32` — V4.1's value.
* **op 118** — per-row EXACT top-k, the op-59 radix run inside one workgroup.
* **op 119** — the **per-64-query-tile UNION table**, whose stated purpose is that the gathered
  flash "stages each tile's selected KV rows ONCE and masks per query, instead of vLLM's
  per-token random gather with zero reuse".
* **the `GATHER=true` arm of `d_flash_mla_prefill_v2`** — the MFMA gathered flash that consumes it.

V4.1 already emits **117 and 118**: they are the `INDEX_SCORE_PF` (7.0 ms) and `INDEX_SELECT_PF`
(11.0 ms) rows in the trace, at n=8, one per `index_source_layer_id`. It does **not** emit 119, so
its selection stays per-query and its attention necessarily takes the scalar op-55 body. V4.1 is
the "per-token random gather with zero reuse" the union table was written to avoid.

#### What it would actually take, stated honestly

This was ranked in §12.18 as "MFMA gathered flash" and described as substantial. Reading the tree
rather than the ranking, it is substantial in a different place than assumed:

* The kernel arm **is written and does compile for gfx942** — `test_kernels.hip:1065` instantiates
  `d_flash_mla_prefill_v2<512, 0, GATHER=true, false>` as `mla_flash_prefill_v2_nope_gather_512`,
  and it is in the shipped object's test set.
* It has **no interpreter dispatch at all.** Every `d_flash_mla_prefill_v2` call site in
  `interp.hip` passes `GATHER=false` (lines 2814, 2863), and the non-FP8 site
  `__builtin_trap()`s when `t[7]` is set. devgen *can* emit `t[7] = n.iuni` ("its presence selects
  the GATHER arm", `mla.rs:6873`) for a non-fp8 sparse V2 prefill; nothing consumes it. That
  pairing is refused at load by `plowrt`'s `check_dsa_pf_arm`, so this is a guarded gap and not a
  live defect — the interpreter trap is, in its own words, "the belt to that braces".
* V4.1's layer is ONE 8-wave cooperative launch, and the V2 arm is **hard-wired to four waves** --
  not merely housed in the 4-wave flash object. Inside `d_flash_mla_prefill_v2`:
  `constexpr unsigned WG = 256; /* V2 is always a four-wave kernel */`, `BQ = 4 * RW`, and the
  LDS layout allocates exactly four per-wave P strips (`Pw0 + 4 * RW * BKV` is where the
  membership-mask array starts). Waves 4-7 of an 8-wave workgroup would index `Pw0 + wave*RW*BKV`
  straight into `Msm`. The existing test kernel agrees: `__launch_bounds__(256, 1)`.
  So routing V4.1's attention here means a mid-layer segment boundary into the 4-wave object, or
  reworking RW/BQ and the LDS layout for eight waves, or parking waves 4-7 -- which halves the
  occupancy of the op being sped up. This is the real cost of the lever and it was measured from
  the source, not assumed: an earlier reading of this section claimed the 8-wave default plus the
  compiled test kernel made the constraint go away. It does not. `PLOW_WG_WAVES` does default to 8
  and `test_kernels` takes no axes, but that kernel sets its own 256-thread launch bound.

So the work is: emit op 119 for V4.1, write the V2 GATHER dispatch, resolve the wave-count
topology, and re-validate numerics — against a top-k *union* whose size is data-dependent
(`glm_dsa_pf_cap` bounds a tile's union at `min(64*top_k, ctx)`, so the staging cost depends on
how much adjacent queries actually overlap, which is unmeasured at V4.1's shape). **None of that
was attempted here, and none of it is validated.** It is recorded because it is the only lever
left with the right order of magnitude: 125 ms of scalar attention against an MFMA arm that is
already written.

#### The honest state of the target

755 ms against 90 ms is 8.4x. §12.26's arithmetic still holds: ~600 GFLOP per layer per rank means
90 ms is ~267 TFLOP/s sustained, about 20% of this part's bf16 peak, and the pipeline is at roughly
2-4%. The two wins in this session came from finding work that was being done at the wrong
granularity, not from making the hardware go faster, and there is a limited supply of that. Closing
8.4x needs the attention path on the matrix pipe (above), and then the same question asked of
`GEMV_F32` (69 ms for a GEMM-shaped mHC projection), `XREDUCE2` (62 ms, of which 782 of 791 us per
packet is STRAGGLER — collective imbalance, not throughput) and `MOE_COMBINE_PF`. Each is a
separate campaign.

### 12.28 One op in thirteen fails to fill the machine, and it is the collective

§12.25's lesson was to ablate before building. There is a cheaper instrument still: the trace
already on disk. `rung_run --trace` writes one `PlowTraceRec` **per (workgroup, packet)** —
406,872 records for a 40-layer run — and `k3_trace_report.py` folds them to per-packet envelopes.
Folding them the other way, per workgroup, answers a question no per-op total can: **of the 304
workgroups the megakernel launches for every packet, how many actually do work?**

An op that confines itself to a subset serializes — its wall time is the subset's while the rest of
the machine idles. Define `busy` as workgroups whose `t_end - t_ready` clears 5% of that packet's
max, `aggregate` as the sum of every workgroup's busy time, `ideal304 = aggregate / 304` (the wall
time if the same work filled the machine evenly), and `serial = max / ideal304`.

| op | pkts | busy/304 | max us | aggregate | ideal304 | **serial** |
|---|---|---|---|---|---|---|
| FLASH_GATHER_PREFILL | 38 | 304 | 3309.6 | 37.4 s | 123.0 ms | 1.02 |
| GEMM_FP8_MX | 330 | 286 | 360.5 | 30.1 s | 99.0 ms | 1.20 |
| MOE_GROUP_DOWN_PF | 40 | 304 | 1799.7 | 21.2 s | 69.6 ms | 1.03 |
| GEMV_F32 | 80 | 304 | 865.9 | 19.5 s | 64.2 ms | 1.08 |
| **XREDUCE2** | 82 | **24** | 787.4 | 1.9 s | **6.3 ms** | **10.27** |
| FLASH_MLA_PREFILL | 40 | 304 | 1527.7 | 18.0 s | 59.4 ms | 1.03 |
| MOE_GROUP_GLU_PF | 40 | 304 | 1242.1 | 13.7 s | 45.2 ms | 1.10 |
| MOE_COMBINE_PF | 40 | 304 | 864.1 | 10.2 s | 33.6 ms | 1.03 |
| HYPER_CONN_PRE | 80 | 304 | 325.6 | 6.7 s | 22.0 ms | 1.18 |
| HYPER_CONN_POST | 80 | 304 | 321.9 | 7.5 s | 24.8 ms | 1.04 |
| FLASH_MERGE | 40 | 304 | 625.7 | 7.1 s | 23.4 ms | 1.07 |
| MOE_ROUTER_TOPK_PF | 40 | 304 | 358.0 | 4.2 s | 13.9 ms | 1.03 |
| RMSNORM | 125 | 304 | 93.4 | 3.1 s | 10.2 ms | 1.14 |

The sweep covers EVERY op in the trace, not just these thirteen. The only other sub-linear
schedule is `MOE_ALIGN_PF` (op 84), which launches 48 workgroups rather than 304 and scores
serial 7.80 -- by construction, since part of it is a single-workgroup prefix, and its entire
cost is 4.5 ms. Everything else lands between 1.02 and 2.51, the outliers being small ops
(op 117 at 1.84, op 14 at 2.51) whose totals are single-digit ms. The audit is
`scripts/dsv41_wg_audit.py`.

**Twelve of thirteen fill the machine.** Every compute op runs all 304 workgroups within 1.02-1.20x
of a perfectly even split — they are well balanced and the grid-stride loops work. That is a useful
NEGATIVE result for the whole remaining campaign: the pipeline's 2-4% of peak is genuine per-CU
inefficiency, not idle CUs, so "spread it wider" is not available as a fix anywhere except one
place.

#### XREDUCE2 is 64.5 ms of wall time for 6.3 ms of machine-filling work

Exactly **24 of 304** workgroups do the work, and the 24 are near-perfectly balanced — on one
packet, 649 / 648 / 648 / 648 / 647 / 647 / 647 / 647 us, a spread under 0.5%, on CUs 1-23. The
other 280 sit for 12-16 us and exit. So this is not imbalance and not a straggler in the usual
sense; `k3_trace_report`'s `strag/pk` of 782 out of 791 us is measuring a **deliberately narrow
schedule inside a 304-wide packet**, and reading it as a tail would have sent the fix in the
wrong direction.

The 24 is `PLOW_XR_SCHED_NWG`, and `scripts/build_gfx942.sh` states where it came from:

> MEASURED, 8x MI300X: 8192x6144 two-shot 1002 -> 728 us isolated [...] the other workgroups only
> arrive.

**8192x6144 is GLM's hidden size. V4.1's is 5120**, its collective count and surrounding megakernel
differ, and the knob is env-exposed and bit-identical by its own contract (same per-element
`r = 0..7` f32 sum; only the partition moves). This is the same cheap A/B class that paid in
§12.24 and §12.26, against **64.5 ms** — the largest single quantified inefficiency left in the
run, and the only one that is a knob rather than a kernel.

Objects at `PLOW_XR_SCHED_NWG` = 48, 64 and 96 all build clean and pass the contract audit
(`/workspace/xrnwg{48,64,96}`). **They are NOT yet measured** — the A/B is one run each and the
machine was occupied by another job's `plowrt serve` throughout. The honest bound: if the reduce is
compute-limited it scales toward `ideal304` and ~58 ms comes back; if it is XGMI-limited at 24
workgroups, nothing does. 226 GB/s of fabric traffic per packet at the measured rate suggests
headroom, but suggesting is not measuring, and the two ceiling instruments in §12.25 are exactly
why that distinction is now written down instead of assumed.

### 12.29 Single-block is the right harness, and it kills two of my own hypotheses

Everything in §12.17-§12.28 was measured on the 40-layer run: a 54-second load, ~135 GB per GPU,
and a 6-minute object build per variant. Emitting ONE block (`--block 2`, the common case --
`compress_ratio` 2 so it carries the gathered attention, plus the MoE and the mHC) changes the
economics completely:

| | 40 layers | one block |
|---|---|---|
| load | 54 s | **12.8 s** |
| VRAM | ~135 GB/GPU | fits in 13 GB free |
| object build per variant | ~6 min | **18 s** (`PLOW_ROWS_ONLY=interp_prefill_mla_moe`) |
| exit range | -40,192 .. 137,216 (wanders) | **-1.37 .. 3.64, mean -0.0007** |

The last row matters as much as the speed. One layer does not amplify a seeded synthetic the way
forty do, so the exit is a usable equality check — every bit-identical knob below produced
`min -1.36719 max 3.64062 mean -0.000712`, to the last printed digit, which is the parity evidence
the whole-model run could never give (§12.26's open question). Two gotchas: the object stamps
`PLOW_PACKET_HASH` from the `plow_config.h` in `PLOW_HSACO_CONFIG`, so a single-block packet needs
objects built against ITS config; and the loader opens objects for buckets the run never executes,
so a one-row build has to be back-filled (`cp -n`) AFTER the contract audit, not before.

#### GEMV_F32: -37.8%, and the W-volume diagnosis was right

`PLOW_GEMV_F32_MR` (§12.28's arm 6) blocks the wide-M GEMV over rows so each f32 W load feeds MR
fmas instead of one. `d_gemv_f32` body over the layer's two packets, median of 5:

| MR | op body | layer |
|---|---|---|
| 1 (== arm 5) | 1688 us | 20458 us |
| 2 | 1476 us | 20133 us |
| **4** | **1051 us** | **19771 us** |
| 8 | 1009 us | 19812 us |

**-37.8% on the op**, and the layer moves 640 us against an op delta of 637 -- attributable, not
drift. Confirmed at the committed default: layer 19818 us, `GEMV_F32` 1045 us, same exit digits.
MR=8 buys another 42 us on the op and gives 41 back on the layer, exactly the spill its scratch
count predicted (2642 -> 2901 scratch ops), so the default is 4.

#### Two hypotheses of mine, falsified by measurement

**§12.28's collective lever is not there.** I argued `XREDUCE2` spends 64.5 ms of wall time on
6.3 ms of machine-filling work because only 24 of 304 workgroups run it, and that
`PLOW_XR_SCHED_NWG` -- tuned at GLM's 6144 hidden, not V4.1's 5120 -- was the largest remaining
knob. Raising it makes the op WORSE:

| NWG | 24 (default) | 48 | 64 | 96 | 152 |
|---|---|---|---|---|---|
| `XREDUCE2` body | **1439 us** | 1467 | 1622 | 1491 | 1723 |

The collective is genuinely fabric-bound at 24 workgroups. The 280 idle CUs are not waste, they
are what a 24-workgroup rendezvous costs, and `serial = 10.27` measured a schedule that is correct
rather than a lever. The occupancy audit was still worth having -- it is what established that
*every other* op fills the machine -- but its one actionable conclusion was wrong.

**§12.27's union table would buy ~3%.** `PLOW_FA_GATHER_ABL=1` collapses every gathered row to a
fixed 64-row window: same loads, same scores, same softmax, same PV, index still computed and read,
only the address tamed. `FLASH_GATHER_PREFILL` goes **3278 -> 3187 us, -2.8%**. So the random
scatter across top-512 cache rows is worth 91 us of a 3.3 ms op. Emitting op 119, writing a V2
GATHER dispatch and resolving the 4-wave/8-wave topology -- the substantial work §12.27 scoped --
would chase 3%. **The op's cost is its SCALAR MATH, not its locality**, and the only fix that
matters is the matrix pipe.

That is two expensive plans deleted by two cheap instruments, on top of §12.25's two. The score for
this campaign is now four confident architectural readings falsified by ablation, against three
wins found the same way. **The instrument is not a formality.**

### 12.2 What is still not demonstrated

  * ~~ONE layer, not 40.~~ **Superseded by §12.18**: all 40 layers emit and run as
    one chain, 1,513 ops, 992 ms median. What is still missing from a WHOLE MODEL
    is the tail, not the layers — no embedding lookup, no final norm, no lm_head,
    no DSpark — because those go through the nn-graph builder, which is still
    unimplemented and is not on the devgen serving path.
  * No reference parity. The input is synthetic; correctness so far is
    "finite and stable" plus an op census that matches the config (§12.18), not
    "right". This is now the largest single gap.
  * **~581 ms against 90 ms** as of §12.50 (chain-sum 608.3 ms less a measured 27.3 ms seam offset)
  * **90 ms is 0.72x the ROOFLINE FLOOR of this model on this hardware — see §12.56 and §12.60. It is not reachable.** (The floor is 125 ms, not the 91 ms §12.56 published: that figure priced FLASH_GATHER_PREFILL against MFMA peak, and §12.59 measured that the matrix core cannot take that op.) (§12.29-§12.42 took a further 141 ms off the 755 below;
    §12.19, §12.20, §12.22, §12.24 and §12.26 took 237 ms off §12.18's 992,
    neither in a V4.1 op; §12.21 showed the next 124 ms line is already on its
    finest legal tile). Closing the remaining 9.5x is not a list of point fixes:
    it needs the whole GEMM/MoE/collective pipeline at MFMA efficiency, which is
    a campaign. §12.18 itemizes the baseline: the V4.1-specific machinery is
    2.8% of the time, and the gap is in `GEMV_F32` (134.8 ms), `GEMM_FP8_MX`
    (129.9), `MOE_GROUP_DOWN_PF` (109.5), `HYPER_CONN_PRE` (102.8) and
    `XREDUCE2` (86.1, of which 99% is straggler).

### 12.30 The head-packed MFMA gathered flash is correct, and structurally wrong at TP8

§12.29 closed with the one claim the gather ablation left standing: `FLASH_GATHER_PREFILL`
(3.25 ms, the layer's largest op) is bound by its SCALAR ARITHMETIC, and only the matrix pipe
addresses that. The tree already held a validated matrix body —
`d_flash_mla_decode_mfma` (op_attention_common.h), "CORRECT (mla_test PASS incl n_head=64
dense+gather) but NON-DEFAULT" — and its rejection note is decode-specific ("filling 256 CUs
needs nsplit~256"), which a prefill with 8192 query work items does not hit.

It is also the ONLY matrix body a gathered prefill can take. `d_flash_mla_prefill_mfma` tiles
QUERY ROWS into M and stages one K tile for the tile, and a gathered prefill's rows share no KV
range — the interpreter says so at the dispatch. The head-packed arm puts HEADS in M, and all of
a token's heads share that token's top-k set. So one work item = one query token, and the port is
a query axis: `n_work = n_batch * n_tok * nsplit`, `ibase = idx + (b*n_tok + t)*top_k`,
`qrow = (b*n_tok + t)*n_head`, plus the two masks the decode arm had no reason to carry —
`tk_live = min(top_k, len)` and the `-1` PAD `d_index_select_pf` writes for a query with fewer
than top_k candidates (`fa_gather_row` maps PAD to row 0, so unmasked it gives row 0 real weight).

**It is correct.** `PLOW_FA_GATHER_MFMA=1` on the single block reproduces the exit to the last
printed digit: min -1.36719, max 3.64062, mean -0.000712, the same digits every bit-identical
knob in §12.29 produced. Which is the whole value of the port, and is not the same as being fast.

    FLASH_GATHER_PREFILL body, layer 2, T=8192, TP8, median of 5:

      scalar (default)                      3267 us
      MFMA, first build                    17064 us     5.2x SLOWER
      MFMA, oacc index made compile-time    7604 us     2.3x slower

The first build's `oacc[t]` was indexed by the RUNTIME `ndt`. It is a register array, so a runtime
index spills the whole accumulator to scratch — object scratch ops 2641 -> 2869. Walking the
constant `MAXNDT` and predicating the excess puts it back in registers (2637, below the control)
and recovers 2.2x. That is the only free win in this shape.

**A three-way ceiling instrument then kills it.** `PLOW_FA_GMFMA_ABL` deletes one term at a time:

      full                                  7686 us
      1  QK contracts 1/8 of its k-steps    6191 us    -1495
      2  PV deleted entirely                7628 us      -58
      3  LDS staging loop deleted           4752 us    -2933

Staging 2933, QK 1700, PV nothing — and a **remainder of ~3050 us that none of the three touches**.
With the gathered latent read free AND the score contraction free, the floor is still the scalar
arm's measured 3267. There is no tuning inside this shape that wins.

The remainder is the online-softmax scaffolding, and the reason is LDS arithmetic. `FA_DEC_TILE`
is `PLOW_THREADS` = 512, so the scalar arm walks a query's whole 512-slot gathered set in ONE
pass: one max, one sum, one correction. The MFMA arm stages `Ksm[BKV][DK+pad]`, and at DK=512
that caps BKV at 32 — 16 passes per query, each with three `__syncthreads()`, sixteen `FA_EXP`
and thirty-two half-wave reductions. **Sixteen times the softmax scaffolding, and at TP8
`n_head = 64/8 = 8`, so only 8 of the 32 MFMA M-rows are live while it is paid.** Raising BKV to
64 needs `Ksm` at 66.5 KB against a 64,720 B arena. The shape is not reachable.

So the head-packed body wants `n_head >= 32` and V4.1 at TP8 has 8. Kept at
`PLOW_FA_GATHER_MFMA=0`: correct, measured, and the validated foundation for the one shape that
is left.

**That shape is §12.27's union table, re-motivated.** §12.29 dismissed it on the gather ablation
(-2.8%), and that was the wrong reading of the right number: `PLOW_FA_GATHER_ABL` prices the
SCATTER, and the union table's value is not locality at all. It is that a per-64-query union set
lets the QUERY-ROW-tiled MFMA arm run — 64 query rows fill two M-tiles completely, and the
softmax scaffolding is paid once per 64 queries instead of once per query, which is exactly the
~3050 us this section could not delete. The cost is arithmetic: each query attends |U| positions
and masks down to its own 512, so the design pays |U|/512 and needs the union of 64 adjacent
queries' top-512 sets over layer 2's 4096-row compressed cache to come in under ~2048. That
number is measurable from the shipped `idx[]` before any kernel is written, and measuring it is
the next step rather than building it.

### 12.31 The union set is half the cache, and the cheap lever on the same op was GF

§12.30 ended by naming the union table as the one shape left and the union SIZE as the thing to
measure before building it. `rung_run --dump act.index_idx=<path>` (new; `--probe` reports f32
stats, which say nothing about an i32 selection table) writes layer 2's shipped `[8192][512]`
table, and it is 93.75% live — every query past the warm-up selects a full 512 of the 4096-row
compressed cache.

    tile of adjacent queries     16      32      64     128     256
    mean |union|               1877    2017    2059    2078    2110
    work vs the sparse ideal   3.91x   4.20x   4.29x   4.33x   4.40x

**At the 64-query tile the union is 2059 rows — half the whole cache — for 4.29x the arithmetic.**
The selection is nothing like as clustered as a union design needs. A query-row-tiled MFMA arm
over that union has to beat the scalar body by more than 4.29x before it returns anything, and it
would need a new device op (union plus per-query membership bitmap), a new kernel, and 32 KB of
LDS for the bitmaps on top of a 33 KB K tile against a 64,720 B arena. The curve is also flat:
halving the tile to 32 buys only 2%, so there is no tile size that makes it comfortable. §12.27's
plan is now measured rather than argued, and it is marginal.

**The cheap lever on the same op was GF.** `d_flash_gather_prefill` re-streams the gathered latent
once per HEAD-GROUP, `n_grp = n_head / GF`, and V4.1 at TP8 has `n_head = 64/8 = 8` against a
hard-wired GF=4 — so it read the top-512 set twice per query for no reason. (The dispatch's `gf`
cannot say otherwise: on op 55 `i[7]` carries the output split, so `gf = in->i[7]` is 513 and only
the final GF=4 arm was ever reachable.)

    FLASH_GATHER_PREFILL body, layer 2, T=8192, TP8, median of 5:

      GF=2  (4 groups)   4416 us      layer 20799 us
      GF=4  (2 groups)   3276 us      layer 19759 us   <- was the default
      GF=8  (1 group)    3002 us      layer 19435 us   <- new default

Exits identical at every GF. -8.4% on the op and -324 us on the layer, against an op delta of 274
— attributable. The curve's shape is the same finding §12.30's ablation made: 2->4 saves 1140 us
and 4->8 saves only 274, so the latent re-read was never the dominant term. It was simply the one
term that cost nothing to delete.

Defaulted under `PLOW_PREFILL_DSV41` (build_gfx942.sh), not in the header: GF must DIVIDE n_head,
and `n_grp = n_head / GF` silently does no work at zero. The dispatch falls back to GF=4 when 8
does not divide, so a different TP cannot hit that.

Layer 2 at the committed defaults is now **19,435 us**, from 20,458 at the start of §12.29.

### 12.32 The dense window pass was on the scalar body because of a missing output split: -68.7%

§12.31's per-packet fold put `FLASH_MLA_PREFILL` second at 1380 us — and it is the DENSE pass over
a **128-wide sliding window**, 1.05M query-kv pairs against the gathered pass's 3.93M. Per pair
that is 1.30 ns against the gathered arm's 0.76: the dense, contiguous, tileable half of the
attention was running 1.7x WORSE per pair than the scattered half.

It was on the scalar body, and the dispatch said why:

    /* The MFMA arm has no output-split form, and a split packet taking it would write its
     * partials at the one-partial stride -- on top of the other half. `nope` already excludes
     * DeepSeek-V4.1, but the exclusion should be the SPLIT's, not a geometry coincidence. */
    if (!nope && !ons && mla_pf_tiled_fills(...))

V4.1 emits `i[7] = (2<<8)|0` on op 51, so `ons` is 2 and the shipped, validated, 2.8x-faster tiled
MFMA prefill was excluded — twice over, since `nope` (DR=0) excluded it as well.

Neither exclusion is structural. `d_flash_mla_prefill_mfma` already takes `n_tok`, `n_head` and
`window`, already bounds its KV range workgroup-uniformly against the tile's last causal horizon,
and — unlike the decode arm §12.30 ported — its `NDT` and `CPW` are compile-time, so its
accumulators never left registers. Every rope path is inside a loop bounded by `DR` or `NK_ROPE`,
both zero at DR=0. **The whole change is `out_nsplit`/`out_sp0` on the epilogue's two writes**
(`oh` becomes `(... )*ONS + out_sp0`), a `<512, 0>` instantiation, and deleting the guard.

    FLASH_MLA_PREFILL body, layer 2, T=8192, TP8, median of 5:

      scalar (PLOW_MLA_PF_MFMA_SPLIT=0)   1380 us     layer 19463 us
      tiled MFMA (default 1)               431 us     layer 18559 us

**-68.7% on the op**, and the layer moves 904 us against an op delta of 948 — attributable. Exits
identical to the last printed digit. The fill heuristic `mla_pf_tiled_fills` still guards the
short-chunk regression it was built for; nothing about it changes.

This is the largest single-op win of the campaign, and it was not a kernel to write. It was a
capability the kernel lacked by four lines, blocking a body that had been shipped and measured on
another model for months. §12.30 spent a full port and three ceiling builds to learn the
head-packed arm cannot win at TP8; this one was the SAME question — "which arm is this op on?" —
asked of the op next to it.

Layer 2 at the committed defaults: **18,556 us**, from 20,458 at the start of §12.29.

    FLASH_GATHER_PREFILL   3069      (scalar; §12.30 shows its floor is the scaffolding)
    GEMM_FP8_MX            2731  x9
    XREDUCE2               1352  x2  (fabric-bound at NWG 24, §12.29)
    INDEX_SELECT_PF        1303
    MOE_GROUP_DOWN_PF      1240
    GEMV_F32               1046  x2
    MOE_GROUP_GLU_PF       1013
    COMPRESS_ROPE_QUANT     870  x3
    MOE_COMBINE_PF          864
    HYPER_CONN_PRE/POST     643/637  x2
    FLASH_MERGE             638
    FLASH_MLA_PREFILL       431

### 12.33 One lane walking 256 bins was 78% of the top-k selector, and the ablation that found it was invalid

`INDEX_SELECT_PF` (op 118) was fourth at 1286 us. It is an MSB-first radix top-k, one workgroup
per query row: each pass histograms the row into `SEL_NB` = 256 LDS bins, then finds the bin
holding the k-th key. That last step is a **serial descending walk on `tid == 0`**, one dependent
LDS read per bin, run once per pass per row — 27 rows per block, four passes.

    serial walk (PLOW_IDXSEL_SCAN=0)   1286 us   strag 288   layer 18488 us
    wave-parallel scan (default 1)      278 us   strag  39   layer 17470 us

**-78.4%**, and the layer moves 1018 us against an op delta of 1009. Exits identical to the last
printed digit, which an integer suffix sum reassociated across lanes must give. The straggler
collapses with it (288 -> 39): a one-lane walk whose length is data-dependent is exactly what
makes workgroups finish at different times.

Lane `l` owns the l-th group of `SEL_NB/PLOW_WAVE` = 4 bins counting DOWN from the top, so an
exclusive `__shfl_up` scan over lanes IS `acc` at that group's first bin; the single lane whose
running total crosses `k_rem` then walks 4 bins instead of 256. The last lane reproduces the
serial walk's fall-through (total below `k_rem`: `dsel` 0, `acc` = the total, `bnd` 0) so the
degenerate case stays byte-identical as well.

**The ablation that led here was invalid, and that is the part worth keeping.**
`PLOW_IDXSEL_ABL=1` deleted the histogram `atomicAdd` to price bin contention, and the op got
SLOWER — 1269 -> 3423 us. A zero histogram never satisfies the boundary test, so `k_rem` never
falls, `FAST_EXIT` never fires, and the prefix filter never narrows: the instrument changed the
PASS COUNT from four to eight and left every pass unfiltered. **A ceiling instrument that deletes
a term feeding a data-dependent exit is not a ceiling.** Every instrument that worked in this
campaign deleted work whose AMOUNT was fixed by the packet — a k-loop tile count, a store, an
address computation. This one deleted a term the control flow reads back.

It was still informative by accident: 428 us per pass with no atomics at all pointed at the scan
rather than at the atomics, which is what got measured next. But it went into the tree as a trap
and came straight back out, along with `PLOW_IDXSEL_ABL=2` (the emit pass's slot atomic), which
measured null — 1269 vs 1286 us, so stream compaction through one LDS counter costs nothing here.

The same serial walk exists in `d_index_select_coop`, the decode twin. Not measured; this campaign
is prefill.

Layer 2 at the committed defaults: **17,477 us**, from 20,458 at the start of §12.29.

### 12.34 The 8-wide MoE combine was gated on k == 1, and V4.1 is top-6 plus a shared tail

`d_moe_combine_pf` (op 87) has had an 8-wide arm since `PLOW_COMBINE_VEC`, and
`build_gfx942.sh` turns it on by default — but it fires only at `k == 1`, because it walks `part`
as one flat `[T*H]` stream. V4.1 emits `k = 7` (top-6 routed plus the shared fold), so every V4.1
blob fell through to the scalar loop the arm's own comment describes: *"ONE 2 B or 4 B load per
operand per iteration and waits on it"*.

The gate is not structural. With `k` slots the element index splits into `(token, h)` and a slot's
row is `part[(tok*k + s)*H + h]` — still eight CONTIGUOUS elements in `h`, so the load widens the
same way. `H` is a multiple of 8, so a group never straddles a token and one divide covers all
eight; the slot loop is unrolled by four to keep loads in flight.

    MOE_COMBINE_PF body, layer 2, T=8192, TP8, median of 5:

      scalar (PLOW_COMBINE_VEC=0)   865 us   strag 44   layer 17454 us
      8-wide (default 1)            334 us   strag 31   layer 16868 us

**-61.4%**, layer -586 us against an op delta of 531. Same operands in the same order (residual,
shared, slots 0..k-1), f32 accumulate, rounded once — bit-identical to the scalar loop, and the
exits agree to the last printed digit. The `k == 1` arm is untouched, so no existing model moves.

**Three of this session's four wins are the same finding.** §12.32's dense prefill, §12.33's
selector scan and this one were not kernels to write: they were a fast body that a gate excluded
(`!ons`), a fast decomposition nobody had applied (the boundary scan), and a fast arm gated on a
geometry V4.1 does not have (`k == 1`). The question that found all three is *"which arm is this
op actually on, and why not the other one?"* — and it is cheaper than every ablation in §12.30.

Layer 2 at the committed defaults: **16,920 us**, from 20,458 at the start of §12.29.

### 12.35 Every workgroup got the same span for the whole packet, because 4 divides 304

`INDEX_SCORE_PF` was 503 us of which **428 was straggler** — 85%, the worst ratio outside the
collective. `d_index_score_pf_row` decomposes into `(pack, span)` items and walked them
SPAN-fastest: `p = w / n_span, sp = w % n_span`. At the shipped geometry `n_span` is
`len / IDXPF_SPAN` = 4096/1024 = 4, and the grid-stride step is `nblk` = 304. **4 divides 304**,
so `w % n_span` is invariant along a workgroup's stride — every workgroup processed exactly ONE
span value for the entire packet.

The spans are not equal work. A pack only reaches the spans below its causal end, so span 0 has
work for all 1024 packs and span 3 for only the last quarter. One workgroup in four did four times
its share while the rest waited at the packet's release signal.

    INDEX_SCORE_PF body, layer 2, T=8192, TP8, median of 5:

      span-fastest (PLOW_IDXPF_PACKFAST=0)   503 us   strag 428   layer 16928 us
      pack-fastest (default 1)               330 us   strag  65   layer 16754 us

`p = w % n_pack, sp = w / n_pack`. -34.4% on the op and **-85% on the straggler**, which is the
number that identifies the cause: the item -> data mapping is untouched, only which workgroup
takes which item, so nothing but balance can have moved. Exits identical.

This is a class of bug, not an instance: any grid-stride loop that decomposes `w` with a modulus
that shares a factor with `nblk` gives each workgroup a biased slice of the work. It is invisible
in a total-time profile and obvious in the straggler column.

Layer 2 at the committed defaults: **16,784 us**, from 20,458 at the start of §12.29 — **-18.0%**
across §12.29-§12.35, all of it verified by an exit that never moved a printed digit.

### 12.36 End to end: 645 ms, from 755

§12.29-§12.35 were all measured on the single block, so the model run is the check that the wins
are real and that nothing about the whole-model emit undoes them. All 40 layers, 8k context, TP8,
`all40.pkt` against objects rebuilt from this tree at the committed defaults:

    run 1   min 636.7   median 645.5   max 777.5 ms   (5 iters)
    run 2   min 640.3   median 644.6   max 740.6 ms   (5 iters)

**645 ms, from 755 at §12.29** — -14.6%. The single block predicted 619 ms
(755 x 16,784/20,458); the model lands 4% above that, which is what the 38 layers that are not
layer 2 cost: two are window-only (no gathered pass, so §12.31's GF and §12.30's floor do not
apply to them) and the compress ratios differ, so layer 2 is not a uniform sample.

**90 ms is not met. 645 ms is 7.2x the target**, and the honest read of §12.29-§12.36 is that
nothing in the remaining profile closes a 7x. The five wins here came to 3.7 ms of a 16.8 ms
layer, and four of the five were gates rather than kernels — a class that is now largely
exhausted for the ops that matter. What is left is FLASH_GATHER_PREFILL at 3.0 ms with a measured
floor at its own scalar body (§12.30), the MoE pair at 2.3 ms already worked in §12.25-§12.26,
and a collective that is fabric-bound by design (§12.28). Reaching 90 ms is a pipeline campaign,
not a knob list, and §12.18's itemization still stands.

ONE OPEN ITEM, AND IT IS NOT NEW. The 40-layer exit is NOT reproducible run to run: -7424/6720
mean 0.0586 against -6432/30208 mean 0.0605. The single block reproduces to the last printed digit
across every build in §12.29-§12.35, so this is amplification over 40 layers of something small,
not a wrong arm — which is exactly why §12.29 built the single-block harness in the first place.
It is not evidence that anything here is wrong, and it is not evidence that nothing is; it is an
unmeasured question that needs a deterministic 40-layer reference to answer.

### 12.37 Staging W in LDS cut the traffic 8x and cost 6x, because the barriers are not free

`GEMV_F32` is the mHC mixing projection, M=8192 N=24 K=20480, and §12.29's arm 6 established the
binding term: a wave owns MR rows, so the W read volume is `M*N*K*4/MR` = 4.0 GB per packet
against 1.97 MB resident. MR is the only divisor and it is capped by the 256-VGPR wall.

LDS lifts that cap in principle. Let all `PLOW_WAVES` waves of a workgroup take the SAME column
group and DIFFERENT row blocks, stage `Wsm[CG][KC]` once, and the volume divides by
`PLOW_WAVES * MR` = 32 instead of 4 — 4.0 GB to 503 MB for 16 KB of LDS.

    arm 6  wave-local W from L2        1060 us   layer 16789 us
    arm 7  workgroup-shared W in LDS   6315 us   layer 22079 us

**6x worse.** Exits identical, so the arithmetic was right and the structure was wrong. The
barriers are the cost, not the LDS. Arm 6 has NO barriers, so its eight waves run fully out of
step and each one's x loads hide the others'; arm 7 puts two barriers around every k-chunk, which
pins all eight waves to the same chunk and exposes the x latency 40 times per work item — and the
megakernel's LDS budget allows exactly ONE workgroup per CU, so there is no second workgroup to
hide it with.

That is the fifth architectural reading this campaign has had falsified by building it, and the
second on this op (arm 3 lost the same way: the largest traffic cut available was the worst arm).
**Inside a megakernel at one workgroup per CU, a barrier is not a synchronization primitive, it is
a serialization of the only latency-hiding the kernel has.** Reverted. W is still 4.0 GB and still
the binding term; closing it needs a decomposition that shares W without stepping the waves.

### 12.38 FLASH_MERGE gave a work item to 512 threads and asked each for one float: -77.1%

At V4.1's prefill geometry `n_bh` = T*n_head = 65536 against `nblk` = 304, so `d_flash_merge`'s
`dsplit` is 1, `dchunk` is D = 512, and the shipped loop hands one work item to the WHOLE
workgroup — 512 threads each doing exactly one `d`. That is one 4-byte load per split and one
2-byte store per thread per item, **ten bytes of traffic** against the item's index math and its
three transcendentals, with the megakernel's LDS budget leaving two waves per SIMD to hide the
latency. 648 us for 335 MB, against 63 us of HBM.

`D` is `8 * PLOW_WAVE`, so one WAVE covers a whole item at eight elements per lane: the loads
widen to two `f32x4`, the store becomes one `st_glob8`, and `PLOW_WAVES` items are in flight
instead of one.

    FLASH_MERGE body, layer 2, T=8192, TP8, median of 5:

      workgroup per item (PLOW_FMERGE_VEC=0)   648 us   strag 90   layer 16752 us
      wave per item, 8/lane (default 1)        149 us   strag 26   layer 16283 us

**-77.1%**, layer -469 us against an op delta of 499, and 149 us is 2.4x off the traffic floor
where 648 was 10x. Scratch moves 2720 -> 2735 ops: the four accumulator groups cost fifteen
instructions, not a spill.

The item SET each workgroup owns is unchanged — still `{slice + j*nblk}` — which
`flash_merge_map()` requires, because the fine dep gates a workgroup on exactly those items'
flash slices. Bit-identical: the same four-accumulator fold in the same order, the same
`(a0+a1)+(a2+a3)`, the same single `f2bf(acc * inv)`; only which lane owns a given `d` changes.

§12.37 and this section are the same measurement from opposite ends. Both ops were starved of
memory-level parallelism at one workgroup per CU. Widening the per-lane work fixed this one;
adding a barrier to share a tile destroyed the other. **At two waves per SIMD the only latency
hiding available is inside a lane's own instruction stream.**

Layer 2 at the committed defaults: **16,312 us**, from 20,458 at the start of §12.29.

### 12.39 The compressor roped every channel twice

`d_compress_rope_quant` quantizes a block against its own amax, so it needs each channel's roped
value twice: once to find the amax, once to quantize against the scale that implies. The shipped
body calls `cmp_rope_at` both times — per channel two bf16 loads, two f32 table loads, two fmas
and a bf16 round trip, done twice for one result.

A quant block is `qblk` floats and nothing else, so it fits in registers. Holding it there
(`qblk <= 32`; larger blocks keep the recompute) removes the second pass outright:

    COMPRESS_ROPE_QUANT body, layer 2, T=8192, TP8, median of 5, same run:

      recompute   878 us   strag 123
      cached      665 us   strag  77     -24.3%

Bit-identical — the same values in the same order through the same `fmaxf` fold and the same
`cmp_quant_rt` — and the exits agree to the last printed digit.

Layer 2 at the committed defaults: **16,138 us**, from 20,458 at the start of §12.29.

### 12.40 End to end after §12.37-§12.39: 620 ms

All 40 layers, 8k, TP8, `all40.pkt` against objects rebuilt from this tree at the committed
defaults, after the FLASH_MERGE widening and the compressor's cached rope:

    run 3   min 617.1   median 620.2   max 634.3 ms
    run 4   min 616.6   median 635.7   max 802.5 ms

**620 ms**, from 645 at §12.36 and **755 at the start of §12.29 — -18%.** The `min` is the stable
statistic (617.1 / 616.6); the median and max spread with machine contention, so the two runs
bracket 620-636 rather than agreeing on a point.

**90 ms is not met. 620 ms is 6.9x.**

The session's ten shipped changes came to 4.3 ms of a 20.5 ms layer, and the pattern is worth
stating because it is what the next campaign should start from. Six were a faster body the tree
already had, excluded by a gate that was not structural — `!ons` on the dense prefill arm,
`k == 1` on the 8-wide MoE combine, a serial 256-bin walk where a wave scan works, a span-fastest
work order that a `4 | 304` coincidence turned into a per-workgroup bias, a workgroup-per-item
merge asking 512 threads for one float each, and a compressor roping every channel twice. None
needed a new kernel. Three more were falsified by building them, including a traffic cut that was
right on paper and 6x slower on the machine (§12.37).

What is left does need new kernels. `FLASH_GATHER_PREFILL` is 2.96 ms against 53 us of bf16 MFMA
peak, and §12.30 measured that the one matrix body in the tree cannot beat its scalar arm at TP8
(`n_head` = 8 fills 8 of 32 M-rows; `DK` = 512 caps the KV tile at 32; the softmax scaffolding is
then paid sixteen times per query). The MoE pair is 2.24 ms against ~389 us at fp8 peak. Those two
are 5.2 ms of a 16.1 ms layer and both are architectural: the gathered attention needs a
decomposition that fills the M dimension — which at TP8 means sharding the heads differently, an
EMIT change, not a kernel one — and the MoE needs the grouped GEMM at MFMA efficiency, which
§12.25 already scoped.

Layer 2 at the committed defaults: **16,138 us**, from 20,458.

### 12.41 The hyper-connection streams walked `hidden` two bytes at a time

Both mHC ops step `hidden` one bf16 per lane per iteration: the pre op's layer-input sum is four
2-byte loads and one 2-byte store per element, the post op's n x n combine is five and four. Nine
of every ten instructions in those loops is a 2-byte memory op where a 16-byte one carries the
same bytes. `hidden` is a multiple of 8 everywhere these run, so the ragged remainder keeps the
scalar loop.

    layer 2, T=8192, TP8, median of 5, same run:

      HYPER_CONN_PRE    654 -> 580 us   -11.3%
      HYPER_CONN_POST   646 -> 537 us   -16.9%

-183 us on the pair. Bit-identical — the same products accumulated in the same `i` order through
the same `fmaf`, one `f2bf` per output — and the exits agree to the last printed digit.

Much smaller than §12.38's 77% on the same kind of fix, and the reason is worth keeping: these two
were never latency-starved the way `FLASH_MERGE` was. `FLASH_MERGE` had ten bytes of traffic per
thread per work item; these have `4 * hidden` bytes per token and already ran a wave per token, so
widening the load only removes issue slots. **The load-width fix pays in proportion to how little
the loop had to do between loads.**

### 12.42 Token-packing cannot rescue the head-packed gathered flash either

§12.30 measured that the head-packed MFMA body loses at TP8 because `n_head` = 8 fills 8 of 32
MFMA M-rows. The obvious escape is to pack four QUERY TOKENS alongside the eight heads to fill the
tile, attending over the union of their four selections. §12.31's dump answers whether that pays
without building it:

    query tile       2      3      4      6      8     12     16
    mean |union|   782   1005   1177   1419   1582   1773   1877
    work vs ideal  1.63x  2.09x  2.45x  2.96x  3.30x  3.69x  3.91x

Four tokens cost **2.45x the arithmetic** to gain 4x the M occupancy — a net 1.63x on the matrix
work. Carry that through §12.30's measured decomposition (staging 2933, QK 1700, PV ~0, remainder
~3050 us): the item count falls 4x but each item walks `1177/32` = 37 KV tiles instead of 16, so
staging becomes 2933 x 0.57 = 1672, the scaffolding 3050 x 0.58 = 1770, the QK 1700 x 0.58 = 986.
**About 4.4 ms, against the scalar arm's 2.96.** Two tokens is worse still (1.63x work for 2x
occupancy).

So the gathered attention cannot be fixed by any kernel change at TP8 — proven now by two
independent measurements, a ceiling instrument and a selection-table audit. What fixes it is
filling the M dimension with heads, which means each rank holding all 64 of them and sharding the
QUERY TOKENS instead: an emit and collective change (replicated q/o weights, an all-gather over
tokens in place of an all-reduce over heads), not a kernel one.

Layer 2 at the committed defaults: **15,948 us**, from 20,458 at the start of §12.29 — **-22.0%**.

### 12.43 End to end after §12.41: 614 ms

    min 610.7   median 614.5   max 763.1 ms   (5 iters, 40 layers, 8k, TP8)

**614 ms, from 755 at the start of §12.29 — -18.7%.** Layer 2 is 15,948 us from 20,458 (-22.0%).

**90 ms is not met. 614 ms is 6.8x.**

Twelve shipped changes across §12.29-§12.42, every one verified against a single-block exit that
never moved a printed digit, and four designs falsified by building them. The shipped set divides
cleanly:

  * **Seven were a gate, not a kernel** — a faster body the tree already had, excluded by a
    condition that was not structural: `!ons` (§12.32), `k == 1` (§12.34), a serial 256-bin walk
    (§12.33), a `4 | 304` work-order coincidence (§12.35), a workgroup-per-item merge (§12.38), a
    doubled rope (§12.39), a hard-wired GF (§12.31).
  * **Four were load width** — §12.38, §12.39, §12.41 and the GEMV MR, all the same shape: too few
    bytes moved per instruction at two waves per SIMD.
  * **One was a knob** (§12.31's GF).

None required a new kernel, and that is the finding: **at 8.4x off target the tree's own faster
paths were worth 18.7%, and they are now spent.** What remains is two ops and both are
architectural:

  * `FLASH_GATHER_PREFILL`, 2.94 ms against 53 us of bf16 MFMA peak. §12.30 and §12.42 measured,
    independently, that no kernel change reaches it at TP8 — the head-packed body's floor IS the
    scalar arm, and token-packing costs more union than it buys occupancy. The fix is to give each
    rank all 64 heads and shard the query TOKENS: replicated q/o weights and an all-gather in
    place of the head all-reduce. An emit and collective change.
  * The MoE pair, 2.24 ms against ~389 us at fp8 peak. §12.25 proved it is neither FLOPs nor
    bytes nor stores but the fixed cost of its output tiles; §12.26 halved that once. Going
    further needs the tile geometry AND the host buffer bound in `mla.rs` to move together, which
    is a packet change.

Those two are 5.2 ms of a 15.9 ms layer. Everything else is ~15 ops each 2-10x off its own floor,
which is §12.18's conclusion unchanged: reaching 90 ms is a pipeline campaign.

### 12.44 The router's fast selection arm was wired to the decode row only

`d_moe_router_topk` has three selection arms and a `_LOCAL` first pass that loads every key to
registers once so the `k` rounds never re-scan. `PLOW_MOE_ROUTER_SELECT` defaults to `PLOW_K3` —
zero for a DeepSeek build — and `AX_K3_ROUTER_LOCAL` is handed only to the K3 DECODE row, so a
V4.1 prefill object could take neither. `n_exp` is 384, well inside the `_LOCAL` arm's
`4 * PLOW_THREADS` bound.

    MOE_ROUTER_TOPK_PF body, layer 2, T=8192, TP8, median of 5:

      default (arm 0, all-pairs rank)   351 us
      arm 1 + LOCAL                     357 us
      arm 2 + LOCAL                     255 us    <- new default under PLOW_PREFILL_DSV41

**-27%.** All three arms pick the same keys in the same order, and the exits agree to the last
printed digit. Arm 2 is the one its own comment calls "the previous K3 arm, kept as the bench
control" — it wins here because `_LOCAL` removes the thing that made it slow.

That is the eighth gate this campaign, and the registry test caught the ninth mistake: both knobs
were ALREADY in `knob_spec.rs`, and adding them again failed `registry_is_well_formed` on
duplicate ids before the build could ship them.

Layer 2 at the committed defaults: **15,877 us**, from 20,458 at the start of §12.29 — **-22.4%**.

### 12.45 End to end, final for this campaign: 611 ms

    min 607.8   median 611.1   max 625.1 ms   (5 iters, 40 layers, 8k, TP8)

**611 ms, from 755 at the start of §12.29 — -19.1%**, on a tight spread (17 ms max-min, where the
first §12.36 runs spread 141). Layer 2 is 15,877 us from 20,458 (-22.4%).

**90 ms is not met. 611 ms is 6.8x, and this campaign does not close it.**

Fourteen shipped changes, every one verified against a single-block exit that never moved a
printed digit, and four designs falsified by building them. They fall into exactly two classes:

  * **Eight were a GATE, not a kernel** (§12.31-§12.35, §12.38, §12.39, §12.44): a faster body the
    tree already had, excluded by a condition that was not structural — `!ons`, `k == 1`, a
    serial 256-bin walk, a `4 | 304` work-order coincidence, a workgroup-per-item merge, a doubled
    rope, a hard-wired GF, a selection arm wired to the decode row.
  * **Four were LOAD WIDTH** (§12.38, §12.39, §12.41, and the GEMV MR of §12.29): too few bytes
    moved per instruction at two waves per SIMD.

Neither class has anything left in the ops that matter. The honest summary of §12.29-§12.45 is
that **the tree's own unreached fast paths were worth 19%**, and the remaining 6.8x is not of that
kind.

What is left, and why each is out of reach of a kernel change:

  * `FLASH_GATHER_PREFILL`, 2.99 ms against 53 us of bf16 MFMA peak — 56x. Measured twice,
    independently: the head-packed MFMA body's floor IS the scalar arm at TP8 (§12.30's ceiling
    instrument), and token-packing costs more union than it buys occupancy (§12.42's audit of the
    shipped `idx[]`). The M dimension has to be filled with HEADS, which means every rank holding
    all 64 and sharding the query TOKENS: replicated q/o weights and an all-gather in place of the
    head all-reduce. An emit and collective change.
  * The MoE pair, 2.24 ms against ~389 us — 5.8x. §12.25 proved it is neither FLOPs nor bytes nor
    stores but the fixed cost of its output tiles; §12.26 halved that once. Going further moves
    `MPF_BM`/`MPF_BN` AND the host buffer bound in `mla.rs:1918` together, which re-emits the
    packet.
  * `XREDUCE2`, 1.36 ms, fabric-bound at 24 of 304 CUs by design (§12.28, §12.29).

Those three are 6.6 ms of a 15.9 ms layer. The other ~15 ops are each 2-10x off their own floors,
and §12.18's itemization of why stands unchanged. Reaching 90 ms means the whole
GEMM/MoE/attention/collective pipeline at MFMA efficiency — a campaign, and the two entries above
are where it starts.

### 12.46 The MoE tile had one more halving in it: MPF_BM 128 -> 192

§12.25 established that the grouped MoE prefill pair is bound by the FIXED PER-TILE cost of its
output tiles -- not its k-loop (capping it at one tile moved DOWN -0.24%) and not its scatter
(issuing 1 store in 16 moved it +3.2%) -- and §12.26 halved that cost once by taking BM from 64 to
128. The tile was then left alone for the rest of the campaign on the belief that 128 was the top.

It was not, and the reason is arithmetic nobody had done. TP8 puts ALL 385 experts (384 routed
plus the shared fold) on EVERY rank, so each expert gathers `T*k/385 ~ 149` rows. The align op
pads each expert's range up to a whole tile. So:

    BM=64    ceil(149/64) = 3 tiles   192 padded rows   29% waste
    BM=128   ceil(149/128) = 2 tiles  256 padded rows   72% waste
    BM=192   ceil(149/192) = 1 tile   192 padded rows   29% waste

**BM=192 cuts BOTH terms at once** -- half the tiles of 128 AND the padding back down to 64's
ratio. That is why it is not the usual tiles-versus-waste trade, and why the 64-vs-128 result did
not predict it. 256 does not exist: `(256+256)*64*2 = 65,536 B` against `plow_smem`'s 64,512,
where `(192+256)*64*2 = 57,344` fits with `MPF_DBUF` still 1. **192 is the ceiling and the
optimum.**

MEASURED, layer 2 at 8k/TP8, arms interleaved so thermal drift lands on both:

    BM=64     DOWN 2489.4   GLU 1574.0   pair 4063
    BM=128    DOWN 1275.8   GLU 1011.5   pair 2361 / 2354 / 2402 / 2399 / 2385
    BM=192    DOWN  971.2   GLU  613.4   pair 1609 / 1588 / 1571 / 1619     <- default

**-783 us on the pair, -32.9%**, over five control repeats and four treatment repeats with no
overlap between the arms. DOWN -26%, GLU -40%. Layer min-to-min 16,102 -> 15,374 us.

**The end-to-end number for this change is NOT yet measured.** The 40-layer run needs ~135 GB/GPU
and the co-tenant described below has held VRAM through eight retry attempts; allocations now fail
at 16 MB. The single block needs 13 GB and still runs, which is why the op-level result above is
solid and the model-level one is absent. -783 us/layer predicts roughly -31 ms of the 611, i.e.
~580 ms -- PREDICTED, not measured, and not to be quoted as a result until a 40-layer run lands.

Two things this cost, both worth recording:

**The A/B harness was lying, and had been.** The whole V4.1 default block sat behind
`[ -z "${MPF_BM:-}" ]`, so naming MPF_BM to sweep the tile ALSO silently dropped `PLOW_FA_GATHER_GF=8`,
both router knobs and both epilogue settings. Every MPF_BM A/B ever run was a five-variable
experiment reported as a tile result. The guard is now scoped to the assignment it belongs to and
each default is individually `:-`. The §12.26 numbers survive only because the EPI note happened to
force the one variable that mattered.

**The exit MEAN is TILE-dependent, and that is expected.** On a quiet machine each tile returns a
STABLE mean and they differ from each other: BM=64 -0.000643, BM=128 -0.000712/-0.000713, BM=192
-0.000736, while min/max hold to the printed digit (-1.36719 / 3.64062) at every tile. So this is
a summation-order difference, not drift -- the tiling fixes the order in which the grouped GEMM's
output tiles are produced, the combine is not on the deterministic arm (`moe_pf_det` off), and a
mean that is a near-cancellation of 167M values of magnitude ~1.4 moves in its fifth decimal for
an absolute rounding change of ~2e-5. **The 64 -> 128 transition already shipped a LARGER shift of
exactly this kind** (-0.000643 -> -0.000712) and was validated end to end at 755 ms. min/max is
the parity check; the mean is incidental, and earlier sections quote it only alongside them.

**A caveat on every parity reading taken late in this round.** A co-tenant on this host took most
of VRAM partway through (two KFD contexts in `/sys/class/kfd/kfd/proc` with no `/proc` entry --
another container). Under that pressure runs begin to OOM, and some that DO complete return a
visibly wrong exit -- the control at BM=128 returned `-1.34375 / 3.67188 / -0.000666` on one run
of five, and BM=192 returned `-1.38281 / 1.72656` on another. That is the failure the runtime
names itself elsewhere in this session: "a collective hit its deadline and returned WITHOUT
reducing". **It is environmental and it affects BOTH arms equally**, so it does not implicate the
tile -- but it means a parity check is only worth reading from a run that completes on a quiet
machine. The stable numbers above are from the earlier quiet window: counting only runs that
printed an exit, the first EIGHT consecutive ones -- across all three tiles, 64 and 128 and 192 --
held min/max at -1.36719 / 3.64062, and the deviations begin at the ninth and then appear on both
arms. (The interleaved A/B itself reported only the clock and the MoE pair, so it contributes
timing evidence and no parity evidence.)

Requires the `mla.rs` `MPF_BM` sizing bound raised to 192 and the packet RE-EMITTED: the align op
pads to the OBJECT's tile height, so an object whose tile exceeds the bound its packet was sized
from is an out-of-bounds device write with no symptom at low expert counts. The bound costs ~4 MB
of arena, not 40x that -- the gathered arrays are shared across layers, not per-layer.

### 12.47 What else was tried this round, and did not survive

**The fp8-MX GEMM tile is already at its optimum.** `GEMM_FP8_MX` is now the SECOND op of the layer
at 2765 us (18%), nine packets of 604 / 375 / 373 / 341 / 335 / 197 / 193 / 191 / 156 us -- the
five projections wq_a [8192,1280,5120], wq_b [.,4608,1280], wkv [.,576,5120], wo_a [.,1024,4096],
wo_b [.,5120,1024], ~406 GFLOP/layer against a ~311 us bf16-MFMA floor, so 8.9x off. `GM_MX_BK=64`
was already taken (the file's own note measured it). The untried axis was BN, because A is bf16 and
re-read once per n-tile, so A traffic (~3.2 of the 4.8 GB/layer) halves at BN=256. MEASURED:

    128x128 BK=64 (default)   2921 us   straggler 163 us/pk
    128x256 BK=64             5910 us   straggler 453 us/pk
    256x128 BK=64             5437 us   straggler 421 us/pk
    128x256 BK=32             so slow the collective hit its deadline and the run failed

Both wider tiles are ~2x WORSE and their stragglers nearly triple -- the register-spill signature
the file's own 128x128 note predicts ("the second promotion accumulator doubles the register
cost"). The traffic cut was real and bought nothing, the third time this campaign that a real
traffic cut has lost to the megakernel's one-workgroup-per-CU occupancy (GEMV arm 7, MoE arm 7,
and now this). Not committed; the default stands.

**The dense-GEMM tuning store is 100% stale and cannot currently be refilled for V4.1.**
`plowc tune status --gpu mi300x` reports 4961 records across 14 digests, EVERY one stale, so all
five dense tiles come from the analytical model at tier `portable`. The campaign's `--shapes auto`
derives its list from a full-model emit, which V4.1 does not have -- `tune gemm` panics on the
`deepseek_v41: no device emit yet` refusal. `PLOW_TUNE_DUMP=1` on the block emit gives the demand
by hand, and it is small: `4096x128x512`, `8192x32x5120`, `8192x384x5120`, `8192x512x5120`, plus
`TUNEDUMP_GEMV 8192x24x20480` for the mHC mix. Those are GEMM_MED + GEMM_SMALL = 553 us of a
15,778 us layer, so the ceiling on tuning them is ~0.5% of the layer. The `--shapes <FILE>` escape
hatch (the one Kimi-K3 uses) is the route if it is ever worth it. Recorded so the next reader does
not re-derive it.

**Straggler, not tile, is where the fp8-MX loss actually sits.** Three of the nine packets have a
straggler LARGER than their body (245/197, 184/193, 367/191 us): those are the small-N projections
where 304 CUs are handed 320 tiles and most of the second round is idle. `glm_glu_halves` -- the
disjoint-CU-set mechanism already in the tree -- is the obvious tool, but the only independent
pair here is (wq_a, wkv) at 604 and 197 us, and splitting 304 CUs between them costs more than the
197 it could hide. Left alone, with the arithmetic recorded.

### 12.48 Unrelated, found while trying to build a fallback: a FULL build of this tree fails

Not V4.1's, not fixed here, and recorded because the campaign's own habits hide it. Every build in
this campaign uses `PLOW_ROWS_ONLY`, so four rows have gone unbuilt for the whole of it:

    FAIL  interp_mixed     d_flash_merge has no arm for head_dim [128, 256, 64]
    FAIL  interp_mixed_gq          -- on AMD a missing arm does not trap, the dispatch
    FAIL  interp_tokbatch             falls through and WRITES NOTHING
    FAIL  interp_tokbatch_gq

**It is pre-existing, not this session's.** Built with `PLOW_FMERGE_VEC=0` -- backing out the
`d_flash_merge` vector arm added in §12.39 -- the four failures are byte-identical, so the new arm
is not the cause. The dispatch audit is doing its job; the gap is real and the audit is the only
reason it is visible rather than a silent zero-write.

The practical consequence for this campaign: a packet emitted at a NEW layer count cannot get a
matching object set, because the object stamps `PLOW_PACKET_HASH` from its config and the `cp -n`
back-fill the single-block flow uses only works when the donor objects were built against the SAME
config. That is what blocked the 20-layer fallback below.

### 12.49 The end-to-end number for §12.46 could not be taken: the box stopped being measurable

A co-tenant took the host partway through this round -- three KFD contexts in
`/sys/class/kfd/kfd/proc` with no `/proc` entry, i.e. other containers. The progression, in order:

  * the 40-layer run (~135 GB/GPU) OOMs on every one of eight retries over ~25 minutes, latterly
    failing at allocations as small as 16 MB;
  * the single block (13 GB) still LOADS, in 8.2 s;
  * but it then fails with `cross-GPU gate 1 on rank 0 reads 1824, expected 2432. Some rank never
    arrived -- a collective hit its deadline and returned WITHOUT reducing.`

That last one is the mechanism behind the wrong exits in §12.46's caveat, caught in the act: the
co-tenant starves a rank of CU time, the collective times out, and the run completes with a token
that is not the sum of the ranks' partials. **A run that completes is therefore not evidence that
it completed correctly**, which is why the parity window in §12.46 is stated as the first eight
runs and not as all of them.

Host RAM was never the constraint (2085 GB available); it is VRAM and CU time.

**A later probe, after the box partly recovered, confirms BM=192's parity on a run that provably
completed**: `exit: 167772160 elems  min -1.36719  max 3.64062  mean -0.000736  zero 0  NaN 0
Inf 0` -- the same min/max as every other tile and the same mean §12.46 records for BM=192, with no
collective failure. So the tile's exit is confirmed, and only the 40-layer CLOCK is outstanding;
that run still OOMs at an 80 MB allocation.

**So the last MEASURED end-to-end figure remains 611 ms (§12.45).** §12.46's -783 us/layer predicts
roughly -31 ms of that, i.e. ~580 ms. That is arithmetic on a single-block measurement, not a
model-level result, and it should not be quoted as one until a 40-layer run lands on a quiet box.

**90 ms is not met, and nothing in this round changes that.** 611 ms is 6.8x; 580 would be 6.4x.

### 12.50 All 40 layers measured after all, as four chains: 638.4 -> 608.3 ms

The whole-model run never got its ~135 GB/GPU back. But a 10-layer chain needs ~34-46, and the
`--block l..r` emit already chains, so all 40 layers were measured as FOUR chains -- 0..9, 10..19,
20..29, 30..39 -- with the two arms INTERLEAVED per chain so co-tenant pressure lands on both.

Two things had to be fixed to get there. §12.48's build gap blocks a matching object set at any new
layer count; `PLOW_ROWS_ONLY` takes a comma list, so every row except the two broken stems builds
(49 objects, rc=0). And the BM=128 control arm is only expressible at all because §12.46 scoped the
`[ -z "$MPF_BM" ]` guard -- before that, naming MPF_BM to build the control would have silently
dropped GF=8 and both router knobs, and the A/B would have measured five things.

MEASURED, all 40 layers, 8k, TP8, median of 5 iters per chain:

    chain      BM=128     BM=192      delta
    0..9       160.2      152.0       -8.2
    10..19     167.6      160.0       -7.6
    20..29     158.2      152.3       -5.9
    30..39     152.3      144.1       -8.2
    SUM        638.4 ms   608.3 ms   -30.1 ms      sum(min) 635.1 -> 605.1

**-30.1 ms, -4.7%, and all four chains move in the same direction** -- which is what makes it the
tile and not drift. It also lands on the prediction: §12.46's -783 us/layer over 40 layers predicts
-31.3 ms against -30.1 measured, so the single-block result scales.

**The seam offset, measured rather than asserted.** A chain sum over-counts: a whole-model emit
overlaps the three seams these measure in isolation. That over-count is now quantified, because the
BM=128 arm has a whole-model twin -- §12.45's 611.1 ms. So the four-chain sum costs
638.4 - 611.1 = **27.3 ms over 3 seams, ~9.1 ms each**. Applying the same offset to the BM=192 arm:

    whole-model equivalent at BM=192  ~=  608.3 - 27.3  =  ~581 ms

That last number is an estimate carrying one assumption -- that the seam cost is the same at both
tiles, which it should be since a seam is a layer boundary and not a MoE tile. The MEASURED
quantity is the 40-layer chain sum, 608.3 ms against the control's 638.4.

**90 ms is not met.** ~581 ms is **6.5x**, from 755 ms at the start of §12.29 -- **-23%** over the
campaign. The three terms that remain are the ones §12.45 named, and §12.47 closed two more doors
on them: the fp8-MX tile is at its optimum and the dense-GEMM tuning store cannot be refilled for
V4.1 until there is a full-model emit. Nothing kernel-level of this size is left; the next 6.5x is
attention re-sharding, MoE packet geometry, and the collective.

### 12.51 op 184's tile is optimal on both axes, and per-shape tiling is not worth building

§12.47 measured the BN axis and found the default best. The BM axis is now measured too, at the
default BN=128, interleaved with the control and repeated:

    GM_MX_BM = 64      3415.2 / 3399.9 us     +21%
    GM_MX_BM = 128     2831.0 / 2795.7 us     <- default
    GM_MX_BM = 192     2899.0 / 2919.6 us     +3.4%
    GM_MX_BM = 256     5437.4 us (§12.47)     +93%

With §12.47's BN sweep (256 -> +102%) that is **four points on each axis, and 128x128x64 wins
both**. Exits held min/max at -1.36719 / 3.64062 on every arm.

**BM=192 was the interesting one and it is instructive that it lost.** It was predicted to help by
DISPATCH FILL: m-tiles are `ceil(8192/BM)`, so BM=128 gives `wkv` (N=576, 5 n-tiles) 320 tiles over
304 CUs -- 1.05 rounds, 53% fill, a whole second round for 16 tiles -- where BM=192 gives 215, one
round, 71%. That is real and it still lost 3.4%, because one tile serves NINE shapes: the same move
takes `wo_a` (8 n-tiles) from 512 tiles/84% to 344/57%, and SM goes 2 -> 3, adding a third
accumulator row to a kernel the file already measured at 148 VGPR.

**So per-shape tiling is bounded and the bound is small.** The obvious follow-up -- instantiate two
or three tiles and dispatch on shape -- is sized by this sweep: the shapes' optima are only 3.4%
apart, and the three underfilled packets are 197 + 193 + 191 us of a 2813 us total, so even PERFECT
fill on all of them buys well under 150 us/layer, ~6 ms end to end, for several template
instantiations and a dispatch rule. Recorded as measured-and-declined rather than left as an
open idea for the next reader to cost again.

`GEMM_FP8_MX` therefore stays ~9x off its ~311 us bf16-MFMA floor, and the gap is not the tile.

### 12.52 Where the layer stands, and what is genuinely left

At the committed defaults the single block is **15.06-15.17 ms** and every large term has now been
either improved or closed with a measurement:

    FLASH_GATHER_PREFILL  3026   4.50 GiB of gathered latent per rank at 1.60 TB/s = 30% of HBM
                                 peak. The cost is the GATHER, not the math -- which is why
                                 §12.30's ceiling instrument found the MFMA arm's floor IS the
                                 scalar arm. See §12.53: token sharding cuts the BYTES 8x.
    GEMM_FP8_MX           2813   tile optimal on both axes (§12.47, §12.51); BK=64 already taken.
    XREDUCE2              1384   already on all 304 CUs (`PLOW_XR_CUS` unset), and an all-reduce is
                                 a global sync -- nothing can run behind it. Needs a smaller
                                 payload or a pipelined consumer, not a knob.
    MOE pair              1575   -33% this round (§12.46), at the LDS ceiling (§12.50).
    GEMV_F32              1021   past its traffic knee: MR 4->8 halves W volume for 4%, so W
                                 traffic is no longer binding despite the arm-6 note.

**90 ms means 2.25 ms/layer against 15.1.** No kernel-level lever of that size exists in this
design: the three structural items are attention re-sharding (every rank holding all 64 heads and
sharding query TOKENS -- replicated q/o weights, an all-gather replacing the head all-reduce), MoE
packet geometry, and the collective. Each is an emit/collective change, and the first is the one
worth doing first because it is the largest single op.

### 12.53 CORRECTION twice over: the traffic arithmetic was wrong, and then the traffic ARGUMENT was wrong

**Correction 1, the arithmetic.** §12.52 as first committed said this op moves "48.3 GB of gathered
reads at ~16 TB/s, i.e. L2 rate". Both numbers were 10x out. The gathered latent is read once per
(token, key) for all of a rank's local heads -- what `PLOW_FA_GATHER_GF=8` buys at TP8 -- so it is

    T * topk * d_latent * 2 B  =  8192 * 512 * 576 * 2  =  4.50 GiB per rank

and 4.50 GiB / 3008 us = 1.60 TB/s, 30% of the 5.3 TB/s HBM peak.

**Correction 2, and it retracts a conclusion this document briefly carried.** From that arithmetic
I argued that attention re-sharding was re-opened on TRAFFIC grounds: V4.1's MLA is fully absorbed,
one 512-wide latent shared by all 64 heads, so head-sharding makes each of 8 ranks re-read the same
latent for its own 8 heads -- 4.50 GiB per rank where token-sharding would need 0.56 -- and at the
measured rate that predicted 3008 -> ~378 us, ~-106 ms end to end. **That is wrong, and the
instrument that refutes it was already in the tree.**

`PLOW_FA_GATHER_ABL=1` collapses every gathered row to a fixed 64-row window: same loads, same
scores, same softmax, same PV FMAs, only the ADDRESS is tamed, so full minus ablated is the
random-access/traffic cost and nothing else. Re-measured at this session's defaults, interleaved:

    full (kv-mpf192)   3008.0 / 2994.6 us
    ablated (64-row)   3009.8 / 3001.3 us      +0.1%, inside noise

**The gathered traffic costs nothing.** Making 4.50 GiB of scattered reads L1-resident does not
move the op, so bytes are not the binding term and cutting them 8x buys ~0. (The header note at
`d_flash_gather_prefill_mfma` recorded this years-equivalent ago as 3278 -> 3187, -2.8%; at GF=8 it
is now flat. I re-derived a model that measurement had already answered.)

**So attention re-sharding does not help this op, and BOTH of its rationales are now falsified** --
MFMA fill by §12.30's three-way ceiling instrument, traffic by this one. It is also worth being
explicit that the ALU work is IDENTICAL under either sharding: 8192 tokens x 8 heads and 1024
tokens x 64 heads are both 8192*8*512*1088 = 36.5 G MAC per rank. Re-sharding moves where the work
sits, not how much there is.

What the op actually is: **36.5 G MAC in 3008 us = 12.1 T MAC/s = 29.5% of the 40.9 T FMA/s scalar
vector peak**, on the vector ALU, irreducibly. The only lever left is getting that math onto the
matrix core for real -- and §12.30 measured the current head-packed MFMA body flooring at ~3050 us
even with staging AND QK free, so it is that BODY that is inadequate, not the idea. The header note
names the shape it would need: a fused persistent split-reduce with double-buffered per-CU MFMA.
That is a kernel rewrite, and it is now the only identified path on the layer's largest op.

**This removes the biggest item from the structural to-do list.** §12.45 and §12.50 named attention
re-sharding as the first thing to do next; on this evidence it is not, because it does not address
the term that binds. The MoE packet geometry and the collective remain.

### 12.54 Where this leaves 90 ms

The single block is 15.07-15.23 ms at the committed defaults, ~608 ms over 40 layers as measured in
§12.50 (~581 whole-model equivalent). 90 ms needs 2.25 ms/layer.

Every large term now has a measurement against it rather than an argument:

    FLASH_GATHER_PREFILL  3008   29.5% of SCALAR vector peak, ALU-bound; traffic proven free
                                 (§12.53) and MFMA fill proven not the lever (§12.30). Needs a
                                 better MFMA body, not a different sharding.
    GEMM_FP8_MX           2813   tile optimal on BOTH axes, 4 points each (§12.47, §12.51).
    XREDUCE2              1384   already all 304 CUs; an all-reduce is a global sync.
    MOE pair              1575   -33% this round, at the LDS ceiling (§12.46, §12.50).
    GEMV_F32              1021   past its traffic knee (MR 4->8 halves W volume for 4%).

None of these is one change away from 6.7x.

### 12.55 The occupancy wall is a REGISTER wall, and it is the megakernel itself

I have twice written in this document that the binding constraint is `plow_smem` at 64,720 B
forcing one workgroup per CU, and recommended shrinking the LDS union below 32,768 B so two
workgroups fit. **That is the wrong mechanism.** The build's own resource table says so:

    object                     vgpr agpr    lds  occ
    interp_prefill_mla_moe      256    0  64720    2

**256 VGPRs is the hard architectural ceiling**, and it pins the kernel to 2 waves/SIMD by itself:
gfx942 has a 512-register file per SIMD, so 512/256 = 2. With `PLOW_WG_WAVES=8` and 4 SIMDs per CU
that is 8 waves = exactly ONE 512-thread workgroup. Cutting LDS to 32 KB would change nothing at
all, because the registers would still allow only 2 waves/SIMD. Two workgroups per CU needs every
wave under 128 VGPRs -- half the current budget.

**And the cause is the megakernel design, not any one op.** Register allocation is per-KERNEL, and
`plow_exec` contains all 24 ops in one switch. So the most demanding op sets the register count,
and **every other op inherits its occupancy**. A 22-microsecond `GLU` runs at 2 waves/SIMD because
the grouped MoE GEMM in the same switch needs 256 registers.

That is the common cause behind every structural loss this campaign recorded, and they stop looking
like three coincidences:

  * GEMV arm 7 (LDS-staged W): traffic -8x, time +6x. "arm 6 has NO barriers, so its waves run
    fully out of step and each one's x loads hide the others'... there is no second workgroup to
    hide it with."
  * MoE arm 7, same shape, same outcome.
  * fp8-MX BN=256 / BM=256 (§12.47) and BM=192 (§12.51): every widening spills, because the
    accumulators are being added to a kernel already at 256.

Each was read at the time as a local result. They are one result: **the megakernel trades occupancy
for the fusion, and at 8k prefill the trade is losing.** Every op in the layer is running with the
latency hiding of two waves per SIMD, which is why so many of them sit at 2-10x off their own
floors with no local explanation.

**The bisect, and it narrows the target.** The row-suffixed objects already differ in which ops
they compile in, so a multi-row build log IS the attribution, free:

    interp_prefill                 256 vgpr        64720 lds   occ 2    <- BARE row
    interp_prefill_fp8             256             64720       2
    interp_prefill_fp8_mla         256             64720       2
    interp_prefill_fp8_mla_moe     256             64720       2
    interp_mla_small               256             60544       2
    interp_decode_fp8kv            255             64560       2
    interp_flash                   512 + 256 agpr  58368       1

**The BARE prefill row is already at 256.** Adding the fp8, MLA and MoE ops on top changes nothing
-- so the ceiling is NOT set by the grouped MoE GEMM or the MLA bodies, which is what I assumed
when I first wrote this section. It is set by whatever is already in the base prefill op set (the
dense `d_gemm_t` at GM_BM=192/GM_BN=256 is the obvious suspect: its accumulators alone are
192*256/512 = 96 VGPRs per thread).

It is also invariant to every knob this campaign owns. Same 256 vgpr, same 64,720 lds, same occ 2
at MPF_BM=64 and 192, at GM_MX_BM=64 and 128, and under the gather ablation:

    kv-mpf192   256  64720  2        kv-mxbm64   256  64720  2
    kv-mpf64    256  64720  2        kv-gabl     256  64720  2

So the occupancy is a property of the megakernel's BASE op set, and nothing at the tile level
reaches it. Note also that both ceilings bind at once -- 256 > 128 VGPR AND 64,720 > 32,768 B LDS
-- so a fix has to clear both.

**The tile that fills the LDS is a hardcoded arch constant, and it is deliberate.**
`hwspec/src/isa.rs` gives gfx942 a prefill `gemm_tile` of **192x256x64** with the comment
"SINGLE-buffered 192x256x64 (64,512 B, fits)" -- i.e. the geometry is chosen to CONSUME the whole
64 KiB LDS. Its accumulators are 192*256/512 = 96 VGPRs per thread. It is also not sweepable the
way `MPF_BM` and `GM_MX_BM` are: `GM_BM=192 GM_BN=256` rides in
`PLOW_PACKET_OBJECT_REQUIRES`, the loader refuses an object whose tile disagrees with its packet,
and `PLOW_HSACO_EXTRA_DEFINES` explicitly rejects `-DGM_BM` for that reason. Which is why no
section of this document has ever swept it.

That choice maximises the efficiency of ONE class of op and sets the occupancy of all 24. The
dense bf16 GEMMs it serves are `GEMM_MED` 449 us + `GEMM_SMALL` 93 us = **542 us of a 15.1 ms
layer** -- 3.6%. The other 96.4% pays for their tile.

**Neither ceiling is actually reached by that tile alone, though.** The GEMM arena is
(192+256)*64*2 = 57,344 B against the object's 64,720, and the bare `interp_prefill` row -- no MLA,
no MoE -- already reports both 64,720 and 256. On the LDS side the driver is the FLASH-PREFILL
tiles, exactly as `interp.hip`'s own note describes for the decode bucket ("leaving the MEMBER in
still sized the union at 58,368 B, which is what pinned the decode object to one workgroup per
CU"). So a fix has to clear the flash tile AND the GEMM tile AND whatever sets 256 registers.

**And the GEMM tile cannot be the fix on EITHER axis, which is worth stating before someone tries
it.** Two facts settle it without a run:

  * **LDS:** the GEMM arena at 192x256x64 is (192+256)*64*2 = **57,344 B, already BELOW the
    object's 64,720 B union**. Shrinking the tile cannot reduce the union by one byte; the driver
    is the flash-prefill tile. (`interp.hip` says the same thing about the decode bucket: gating
    the `fa` member out still left 58,368 B.)
  * **Registers:** `op_gemm_common.h`'s own tile note already records the measurement --
    "removing flash from the object (the **128/128 register split is the op union, not flash**)".
    The 256 is the union over all op bodies, so there is no single op to cut.

I confirmed the mechanism the hard way: `GM_BM` is ARCH-defaulted in `op_gemm_gfx942.h` (192,
`#ifndef`-guarded), while `hwspec/src/isa.rs`'s `gemm_tile` only DECLARES the value that lands in
`PLOW_PACKET_OBJECT_REQUIRES`. Patching the hwspec constant alone re-emitted a packet requiring
GM_BM=128 while the object still compiled at 192 -- and the object came out BYTE-IDENTICAL (same
256 vgpr, same 64,720 lds, same 3441/3228/1940/288 ISA counts), which is the tell. The two must
move together, and per the above, moving them buys nothing on either ceiling.

**What follows for a next campaign.** Not "shrink the LDS", not "shrink the GEMM tile", and not
"split off the MoE": bisect
within the BASE op set to find what needs 256, then either cut that op's demand or move the light
streaming ops (RMSNORM, GLU, the hyper-connection pair, COMPRESS_*) into a second, low-register
kernel where they can run at 4 waves/SIMD. Those ops are ~2.4 ms of the 15.1 ms layer and are
paying full occupancy cost for registers they do not use. The resource table is printed by every
build, so each step of that bisect is one 18-second row build and a `grep`.

The campaign's honest position is that the tree's own unreached fast paths were worth
755 -> ~581 ms (-23%), and the remainder is a pipeline-wide MFMA-efficiency problem whose root is
one number in a build log that nobody had read.

### 12.56 90 ms is below the floor, and here is the arithmetic

The target has been carried for the whole campaign without anyone checking it against a roofline.
`scripts/dsv41_roofline.py` does that: each op's own floor, taken GENEROUSLY -- compute at full
MFMA peak (653 T MAC/s bf16, 1306 fp8), traffic at full HBM peak (5.3 TB/s), collectives at full
xGMI (400 GB/s/GPU) -- whichever binds. No op can beat its own floor, so the SUM is a hard lower
bound.

    op                        measured     floor   x off
    FLASH_GATHER_PREFILL         3008        56    53.8
    GEMM_FP8_MX                  2813       312     9.0
    MOE pair (GLU+DOWN)          1575       194     8.1
    XREDUCE2                     1384       734     1.9
    GEMV_F32                     1021        98    10.4
    HYPER_CONN PRE+POST          1158       285     4.1
    COMPRESS_ROPE_QUANT           674        11    63.1
    RMSNORM                       333       127     2.6
    MOE_COMBINE_PF                318       237     1.3
    FLASH_MLA_PREFILL             423       228     1.9
    ------------------------------------------------
    SUM (these ten)             12707      2282     5.6
    other ~10 ops                2393         --
    LAYER                       15100

    40-layer measured   604 ms
    40-layer FLOOR       91 ms      <- ten ops only; the other ten counted as FREE
    target               90 ms
    target / floor      0.99x

**90 ms is 0.99x the floor.** Reaching it would need every one of those ten ops at 100% of
theoretical peak SIMULTANEOUSLY -- and would still miss, because the other ten ops (2.4 ms/layer
measured today) are counted at zero here and their floors are not zero. This is not an optimisation
gap; the target is below what the algorithm and the hardware allow.

Two lines carry most of it:

  * **`XREDUCE2`'s floor alone is 734 us/layer = 29 ms over 40 layers -- a THIRD of the entire
    90 ms budget** -- and it is already within 1.9x of that floor. Two all-reduces of `[8192, 5120]`
    bf16 per layer at TP8 is 147 MB per GPU per reduce after the ring factor; at full xGMI that is
    what it costs. No kernel work touches it. Only a different parallelism (less frequent
    reduction, or a sharding that reduces less) would.
  * `MOE_COMBINE_PF` (1.3x) and `FLASH_MLA_PREFILL` (1.9x) are likewise nearly done.

So the honest statement of where the campaign ended: **604 ms measured against a 91 ms floor is
6.6x, and the reachable number is neither 604 nor 90.** A strong implementation runs its ops at
roughly half of peak; that puts this model at **~180-250 ms on 8x MI300X at 8k**, and the work to
get there is the pipeline-wide MFMA-efficiency problem of §12.55 -- splitting the interpreter so
ops stop inheriting each other's register union -- not any remaining tile or arm.

If 90 ms is a hard requirement, it needs a change of premise, not of code: fewer layers, a shorter
context, more GPUs (the collective floor falls with a smaller per-GPU payload only if the reduction
itself shrinks), or a different attention/MoE algorithm. The number as stated is not on this curve.

**One concrete pointer left for whoever picks this up**, because the roofline names it and nothing
in this campaign touched it. `COMPRESS_ROPE_QUANT` has the WORST ratio in the layer -- ~40x -- and
it is not one op but three, with wildly different sizes. Per packet:

    #10  COMPRESS_ROPE_QUANT    53.9 us     the KV cache compressor, T/ratio rows x 576
    #13  COMPRESS_ROPE_QUANT    76.3 us     the index KEYS, n_pool rows x 128
    #15  COMPRESS_ROPE_QUANT   508.1 us     the index QUERIES, 8192 x 32 heads x 128   <-- 80%

Packet 15 alone is 508 us moving ~84 MB (67 MB of bf16 in, ~17 MB of fp4 + E8M0 out) = **165 GB/s,
3% of HBM**, against a ~16 us floor. It is a pure streaming rope-then-quantise over
`8192 * 32 * 128 = 33.6M` elements with `DSV41_IDX_QBLK = 16`, so one thread owns a whole 16-element
quant block and the work is ~23 cycles/element. The §12.38/§12.41 class of fix -- widen the lane to
8 elements and vectorise the load -- is exactly what it has never had; the register-caching pass in
§12.40 cut the pair 878 -> 665 us and left the ACCESS WIDTH alone. Worth ~350 us/layer, ~14 ms end
to end, and it is ordinary kernel work rather than an architectural change.

### 12.57 The compressor's access width, taken: 667 -> 567 us

§12.56 named `COMPRESS_ROPE_QUANT` as the worst ratio in the layer and pointed at the load width.
Taken, and it is the fifth win of the LOAD-WIDTH class.

The register arm from §12.40 still read CHANNEL BY CHANNEL, and `cmp_rope_at` touches TWO channels
per call -- its own and the rotary partner -- so a `qblk=16` block cost **32 scalar bf16 loads over
32 CONTIGUOUS bytes**. The partner is always inside the same 8-element vector: RoPE here is
interleaved (GPT-J), so `c_rope0 + 2m` pairs with `c_rope0 + 2m + 1`, and a pair straddles an
8-aligned boundary only if `c_rope0 + 2m == 7 (mod 8)` -- odd, hence impossible whenever `c_rope0`
is even. So one `ld_glob8` serves both reads and 32 loads become 2.

`PLOW_CMP_VEC8` (default 1), guarded on `qblk % 8 == 0 && c_rope0 % 2 == 0`. MEASURED, interleaved
with its control:

    PLOW_CMP_VEC8=0    669.0 / 664.4 us
    PLOW_CMP_VEC8=1    557.3 / 576.2 us      -15%

and per packet the one that matters moves with it:

    #15 (index queries, 8192 x 32 x 128)    508.1 -> 436.6 us

**Bit-identical, and exits confirm it**: min/max hold at -1.36719 / 3.64062 on every run of both
arms. The same two bf16 values per channel go through the same f32 fma pair, the same `f2bf` round
trip and the same `fmaxf` fold, in the same order; only the loads changed.

**It is -100 us/layer, ~-4 ms end to end -- well short of the ~350 us the load count predicted**,
and the reason is visible in what is left: the `cosb`/`sinb` table gathers are still one scalar
load per PAIR, and the `f2bf` round trip per channel is unchanged. The op is now ~34x off its
floor instead of ~40x. The remaining width is in those tables, not in the source reads.
## 12.58 The rope tables, and then the occupancy question answered for good

12.57 said the width left in `d_compress_rope_quant` was the `cosb`/`sinb` tables, still one
scalar f32 load each per PAIR. Taken: an 8-group lying wholly inside the rope region covers
exactly four pairs at CONSECUTIVE `m`, so each table wants one 16 B load, not four 4 B ones.
`PLOW_CMP_TAB4` (default 1), guarded on `c_rope0 % 8 == 0` (no group straddles the rope boundary,
and `m0` is a multiple of 4) and `(rd/2) % 4 == 0` (so `tb` is too). MEASURED, interleaved:

    PLOW_CMP_TAB4=0    561.5 / 570.5 us
    PLOW_CMP_TAB4=1    555.0 / 546.1 us      -2.7%

Bit-identical; exits -1.36719 / 3.64062 on every run of both arms.

**-2.7% where the load count predicted far more, and that is the useful result.** At 8k the
indexer's query call is ~32k items x 8 blocks = 262k work items against 304 CU x 512 threads =
156k threads: 1.7 iterations per thread, two dependent memory round trips, at occupancy 2 with no
third wave to hide either. The remaining 34x is LATENCY UNDER THE OCCUPANCY WALL, not width. So
the wall is the only thing left to attack -- and it is now settled.

### The bisect: what the 256 VGPRs and the 64,720 B of LDS actually are

Fourteen single-row builds of `interp_prefill_mla_moe`, each knocking out one op family through
the existing `PLOW_DECODE_INVENTORY_PRUNE` / `PLOW_HAS_*` machinery. **VGPR and LDS did not move
once**: 256 / 0 / 64,720 in every arm, including the one with eight ops deleted. Only the spill
count moved -- 122 baseline, 92 without flash, and **10 without RMSNORM alone**, which is not the
multi-row arm (`PLOW_RN_ROWS` 1/2/4 all report 122).

So 256 is not an op's demand. It is the ARCH CAP the allocator is handed once LDS has fixed
occupancy, and the causality runs the other way from what 12.53 recorded:

    LDS 64,720 B  ->  1 workgroup/CU (65,536 B per CU)
                  ->  8 waves / 4 SIMDs = 2 waves/SIMD
                  ->  512-register file / 2 = 256 VGPR, and 122 of them spill

**12.53's "register wall, not an LDS wall" was backwards, and 12.53's GEMM arena figure was wrong
too.** `GM_LDS_HALVES_T(192,256,64)` is `(192+256) * (64+8)` = 32,256 halves = **64,512 B**, not
57,344 -- the earlier arithmetic dropped `GM_STRIDE`'s +8 bank-conflict pad, which is 7,168 B of
it. 64,512 plus 208 B of interpreter scalars is exactly the 64,720 the object reports.

### Occupancy 2 cannot be escaped, and the reason is the model, not the code

2 workgroups/CU needs the WHOLE `plow_smem` union under 32,768 B. Its three members, evaluated:

    GEMM tile (192,256,64)   (192+256) * (64+8)                  = 32,256 h = 64,512 B
    FLASH_HD512              32*(512+8) + 32*(128+8) + 8*32*32   = 29,184 h = 58,368 B
    MLA prefill (KSPLIT=2)   Qsm + Ksm + Psm + corr              = 30,272 h = 60,544 B

All three exceed 32,768 B, so shrinking any one of them changes nothing -- which is exactly what
the builds show (`PLOW_MLA_PF_KSPLIT` 2 vs 4: LDS 64,720 both ways; the GEMM tile cannot be cut
at all without re-emitting, since the packet stamps `GM_BM=192` into
`PLOW_PACKET_OBJECT_REQUIRES`).

And the MLA-prefill member cannot be cut to fit **by arithmetic, not by measurement**:

    Qsm alone = BQ * (DK + DR + FA_PAD) halves = 32 * (512 + 64 + 8) * 2 B = 37,376 B

BQ=32 is one MFMA M-tile, the minimum. DK=512 is V4.1's latent head dim. **The query tile alone is
37,376 B against a 32,768 B budget, before K, P or the correction strip are placed.** No tile, no
split, no knob reaches it.

**Occupancy 2 is a property of a 512-wide latent head on a 64 KiB-LDS part.** It is not a plow
choice and no kernel change in this repo can move it. That closes the last structural lever named
in 12.55/12.57 for the single-kernel interpreter: the only remaining escape is to stop putting
these ops in the same kernel as the GEMM and the flash -- a second, low-LDS kernel for the light
streaming ops (RMSNORM, GLU, the hyper-connection pair, the compressors: ~2.4 ms of a 15.1 ms
layer). That changes the packet/object contract for EVERY model, not just V4.1, so it is a scope
decision and not one to take unasked. Even at a 4x speedup on those ops it is ~1.8 ms/layer = ~72
ms over 40 layers: 608 -> ~536 ms.

**The target is unchanged by any of this. 90 ms is 0.99x the roofline floor (12.56).**

### 12.58a Correction: the stores were still narrow, and they were worth 10x the tables

12.58 read the table lane's -2.7% as "the remaining 34x is LATENCY under the occupancy wall, not
width". That was wrong. The compressor's OUTPUT was still 16 scalar 2-byte stores over 32
CONTIGUOUS bytes, under the same alignment the vector lane already guarantees. Widening it:

    parent (reads wide, store scalar)   545.0 / 544.8 us    layer 15237.9 / 14947.2
    store 16 B wide                     401.6 / 394.3 us    layer 14872.1 / 14928.9   -27%

Bit-identical, exits -1.36719 / 3.64062 on every run of both arms. No new knob -- `PLOW_CMP_VEC8`
already gates the lane, and the parent object is the control.

Across the three load/store-width commits the op is **666.7 -> 398.0 us, -40%**, ~-150 us/layer.

What 12.58 says about OCCUPANCY is unaffected and stands: Qsm alone is 37,376 B against the
32,768 B a second workgroup would need, so occupancy 2 is a property of DK=512 on a 64 KiB part.

## 12.59 The layer's biggest op had an unranked MFMA body. Ranked: it loses by 2.6x

`FLASH_GATHER_PREFILL` is 2955 us of a 14,900 us layer -- 20%, the single largest op -- and
`op_attention_common.h` carried a written-but-never-measured alternative next to it,
`d_flash_gather_prefill_mfma`, marked "opt-in until hardware ranks it". Ranked, interleaved with
its control, two pairs:

    PLOW_FA_GATHER_MFMA=0   2955.1 / 2980.9 us     layer 14914.5 / 14856.3
    PLOW_FA_GATHER_MFMA=1   7729.2 / 7632.0 us     layer 19403.9 / 19448.5

Exits IDENTICAL (-1.36719 / 3.64062) on every run of both arms, so the body is CORRECT; it is the
wrong shape. It is `d_flash_mla_decode_mfma<GATHER=true>`, whose work item is ONE QUERY TOKEN: at
TP8 that puts n_head=8 on the MFMA M-dimension against a 16-wide tile and stages each token's own
top_k set with no reuse across tokens, where the scalar body groups GF=8 heads and streams the
gathered latent once. Recorded in the knob's comment so it is not re-ranked unchanged.

With `PLOW_FA_GATHER_ABL` (12.4x: traffic, +0.1%) and the re-sharding study (12.5x: identical ALU
work either way) this closes the third and last cheap hypothesis about the layer's biggest op. It
is ALU-bound at ~29.5% of scalar vector peak and the matrix core cannot take the work at this
shape.

### Where the layer stands after 12.58a

    FLASH_GATHER_PREFILL   2955    ALU-bound; traffic, re-sharding and MFMA all falsified
    GEMM_FP8_MX            2788    tile sized on both axes, all four points worse
    XREDUCE2               1673    98.7% straggler -- collective wait, not work
    GEMV_F32               1060    8 arms measured; register-bound at MR=4, LDS share lost 6x
    MOE_GROUP_DOWN_PF       952    MPF_BM 128 -> 192 already taken (-32.9% on the pair)
    MOE_GROUP_GLU_PF        610
    HYPER_CONN_PRE/POST    1090    already fully vectorised (PLOW_HC_VEC8)
    COMPRESS_ROPE_QUANT     378    666.7 -> 398.0 over three commits, -40%

Every line above is either measured-and-closed or measured-and-taken. **90 ms remains 0.99x the
roofline floor (12.56), and the gap is not in any one op.**

### 12.59a The gather's randomness is free; its bytes are not

12.59 closed the MFMA hypothesis for `FLASH_GATHER_PREFILL`. One lever on that op had still never
been swept: `PLOW_FA_GATHER_GF`, which trades ARCH REGISTERS against TRAFFIC. GF=8 carries 64 f32
accumulators (`oacc[8][8]`) on a kernel already at the 256-VGPR cap and reads the gathered latent
ONCE; halving GF halves the accumulators and doubles the re-streaming. Given 12.53's spill count
and the ABL result that the op is not scatter-bound, the smaller GF looked like it should win.

It does not. Two interleaved passes, op body and layer median:

    GF=8   2968.8 / 2925.2 us    layer 14888.5 / 14988.1   <- V4.1 default, unchanged
    GF=4   3291.1 / 3302.8 us    layer 15079.9 / 15338.4    +12%
    GF=2   4417.8 / 4413.9 us    layer 16137.4 / 16143.6    +50%

Exits identical at every GF. Monotone, so register pressure is NOT what bounds this op.

**Read with `PLOW_FA_GATHER_ABL` (+0.1% with every address tamed to a 64-row window), the pair says
what neither says alone: the op is insensitive to scatter LOCALITY and sensitive to VOLUME.** The
default was already right, and this is the fourth and last hypothesis about the layer's biggest op
to be measured and closed.

## 12.60 The floor was wrong, and it was too LOW: 125 ms, not 91 ms

12.56 summed generous per-op floors to 2282 us/layer = 91 ms for 40 layers and concluded 90 ms is
0.99x of it. One line of that sum was wrong, and 12.59 is what exposes it.

`scripts/dsv41_roofline.py` priced `FLASH_GATHER_PREFILL` -- the layer's biggest op, 2955 us -- at
`mac / MFMA_BF16` = 56 us, i.e. against the MATRIX CORE. **But 12.59 ranked the written MFMA body
on hardware and it is 2.6x SLOWER**, because its work item is one query token and TP8 leaves
n_head=8 on a 16-wide MFMA M-dimension. The shipped body does every score and every PV MAC on the
VECTOR ALU by construction. Pricing it against an engine it provably cannot use made its floor
53x too generous.

Corrected to `mac / VALU_F32`:

    FLASH_GATHER_PREFILL   floor 56 -> 892 us     x off  52.9 -> 3.3
    SUM (these ops)        floor 2282 -> 3118 us  x off   5.5 -> 4.0
    40-layer FLOOR         91 -> 125 ms
    target / floor         0.99x -> 0.72x

Two things change. **The target moves further out of reach, not closer**: 90 ms is 0.72x a floor
that ignores ~10 other ops entirely, so the real bound is above 125 ms. And **the layer's biggest
op stops being an outlier** -- at 3.3x off its true floor it is one of the better-behaved ops in
the table, which is exactly why four successive hypotheses about it (traffic, re-sharding, MFMA,
GF) all measured null or negative. It was never mis-implemented; it was mis-priced.

The measured column was also refreshed from the 2026-09-16 trace (layer 15,100 -> 14,900 us;
COMPRESS_ROPE_QUANT 674 -> 378, 63.1x -> 35.4x off).

## 12.61 The dense-GEMM tuning store was 100% stale. Refilling it changed nothing, and here is why

Every V4.1 emit prints a warning nobody had acted on: `>>> tunedb: ALL 5 dense-GEMM tile(s) chosen
by the ANALYTICAL MODEL. This build is UNMEASURED`. `plowc tune status` confirmed it -- 4961
records, every one STALE against the probed digest `gfx942-295749b0e5a2b676`, so **zero** usable
measurements. Acted on.

`--shapes auto` cannot run for V4.1 (it derives demand from a full-model emit, which V4.1 has no
graph builder for), but `PLOW_TUNE_DUMP=1` on the `--block` emit observes the same demand:

    TUNEDUMP      8192  512  5120  None   MISS  (x2)
    TUNEDUMP      8192  384  5120  None   MISS
    TUNEDUMP      8192   32  5120  None   MISS
    TUNEDUMP      4096  128   512  None   MISS
    TUNEDUMP_GEMV 8192   24 20480  None   MISS  (x2, PLOW_DOP_GEMV_F32)

Campaign run on those five (`--campaign dsv41-prefill-8k`, 40 rows, 30 published). **The analytical
model's pick is the worst or near-worst tile for every shape:**

    shape              best tile      best ms    c5 192x256x64 (picked)   speedup
    8192x512x5120      128x128x64      0.1323            0.2544            1.92x
    8192x384x5120      128x128x64      0.1214            0.2533            2.09x
    8192x32x5120        64x128x128     0.0622            0.2240            3.60x
    4096x128x512        64x128x128     0.0185            0.0618            3.34x
    8192x24x20480       64x128x128     0.2079            0.7940            3.82x
    TOTAL                              0.5423            1.5875            2.93x

Ingested; the re-emit then prints `tunedb: all 5 dense-GEMM tile(s) chosen BY MEASUREMENT`, the
packet's geometry contract moves to `GM_BM=128 GM_BN=256`, and the rebuilt object's LDS drops
64,720 -> 60,752 B. **And the layer does not move:**

    base    14902.1 / 14882.5 us     GEMM_FP8_MX 2875.8 / 2806.3   GEMM_MED 486.6 / 472.6
    tuned   14942.4 / 14907.1 us     GEMM_FP8_MX 2839.7 / 2790.3   GEMM_MED 487.8 / 471.3

Exits identical. Two reasons, and both are worth keeping:

1. **The campaign covers the wrong ~4% of the layer.** The demand dump is all `quant=None`. The
   layer's big GEMM is `GEMM_FP8_MX` at 2815 us, on the MX path, which does not consult this store
   at all -- the dense rungs it does cover are `GEMM_MED` (480 us) + `GEMM_SMALL` (97 us), 577 us
   of 14,900. The MX tile was already hand-swept on both axes (12.4x: BN 256 +102%, BM 64 +21%,
   192 +3.4%, 256 +93%), and 192x256 won, so that axis was already measured and already optimal.
2. **A standalone tile ranking does not transfer into the megakernel.** `gemm_tile_sweep` runs the
   GEMM alone, at whatever occupancy its own registers and LDS allow. Inside `plow_exec` the same
   GEMM runs at occupancy 2 in a shared 60 KB arena (12.58), where the tile is not what bounds it.
   This is the same way GEMV_F32's arm 7 and arm 3 lost, and the same lesson.

**The store is now populated with real records rather than a stale-by-14-digests pile**, which is
worth having on its own -- but the emit-time warning, though accurate, is not the lever it looks
like on this model. Both GEMM axes are now closed by measurement.

## 12.62 All 40 layers, complete: 599.7 ms chain-sum / ~572 ms whole-model

The two Engram-bearing chains finally cleared the co-tenant's memory. All four, at the current
defaults (MPF_BM=192, PLOW_CMP_VEC8=1, PLOW_CMP_TAB4=1, and the compressor store lane of 12.58a):

    0__9     min 151095.8  median 151620.5 us
    10__19   min 156961.5  median 157115.3 us
    20__29   min 147912.9  median 148954.4 us
    30__39   min 141637.9  median 142023.9 us
    ----------------------------------------
    sum(min) 597.6 ms      sum(median) 599.7 ms
    whole-model equivalent (less the measured 27.3 ms seam offset):  ~572.4 ms

Against 12.50's 608.3 chain-sum / ~581 ms: **-8.6 ms**, which is what this session's compressor
work predicted (-150 us/layer on COMPRESS_ROPE_QUANT alone is -6 ms).

## 12.63 The MoE tile is not padding-bound, it is per-tile weight-reload bound

`MPF_BM` sets the height every expert's gathered rows are padded up to, and op_moe.h calls the
padding "pure MFMA waste". At T=8192 that padding is large -- 8192 tokens x top-6 plus shared =
57,344 real rows over 385 experts (~149 each), padded to 192 gives ~81,984 rows, **~43% of the MoE
GEMM computing zeros**. So the tile should want to be SMALL. Swept, op body, two passes:

    MPF_BM      DOWN      GLU      pair
      64       2412      1598     4010 us
     128       1350      1020     2370 us
     192        933       627     1560 us     <- shipped, and still falling steeply

**Monotone the other way, and that settles the mechanism.** If padded rows were the cost, BM=64 --
which pads an average expert to ~0% against BM=192's 50% -- would win. It is 2.6x WORSE. The cost
is the PER-TILE EXPERT-WEIGHT RELOAD: an expert spanning two tiles streams its weights twice, and
TP8 is what makes that bite, because `down`'s K is moe_intermediate/TP = 288, a k-loop far too
short to amortise the fixed per-tile cost. op_moe.h's own note says exactly this and stops at 192.

It stops there because of the arena, not the model: the single-buffered tile is
`(MPF_BM+MPF_BN)*MPF_BK*2` bytes, so at BN=256/BK=64, BM=256 needs 65,536 B against `plow_smem`'s
64,512. **BK=32 halves the tile and reopens BM to 512**, and per op_moe.h halving BK doubles the
k-passes while each expert weight byte still crosses HBM exactly once -- the stream the grouped
form exists to amortise is unchanged. `MPF_BK` is now plumbed to the prefill row (it had only ever
reached the decode row), and BM=384 and BM=512 at BK=32 both build.

MEASURED, and the tile is taken. Three passes at iters=10, MoE pair and layer min:

    MPF_BM/BK     MoE pair     layer min
    192 / 64       1583.7       14795.5 us
    384 / 32        998.4       14196.8
    512 / 32        868.4       14185.3     <- new default

Then end to end at the SHIPPED defaults, each arm being the packet its own code state emits with
the objects that state builds, three passes at iters=10:

    old (192/64)   MoE pair 1566.9 us   layer min 14721.3   median 14915.7
    new (512/32)   MoE pair  870.7 us   layer min 13983.2   median 14248.8
                             -44.4%           -738 us/layer, -5.0%

Exits identical (-1.36719 / 3.64062) on all six runs. -738 us/layer predicted ~-29.5 ms over 40
layers, and **the 40-layer re-measure confirms it**: all four chains re-emitted and rebuilt at the
new defaults give 144.1 + 151.0 + 141.6 + 132.3 = **568.9 ms chain-sum, ~541.6 ms whole-model**,
against 599.7 / ~572.4 -- **-30.8 ms**. `mla.rs`'s sizing bound goes to 512 with it, costing ~60 MB/rank of `fu_g` --
which is what buys the tile.

### 12.63a The collective's workgroup cap is already right for this shape

`PLOW_XR_SCHED_NWG` caps the aiter collective schedule at 24 workgroups (reduce-scatter at 8). That
default was measured on GLM at 8192x6144; V4.1's hidden is 5120, so it was unverified here, and
XREDUCE2 is 1673 us/layer -- 2.3x its XGMI floor and 67 ms over 40 layers. Swept, two passes,
layer min (the reliable statistic; see below):

    NWG=16   14397 us     +374 vs default
    NWG=24   14023 us     <- default
    NWG=32   14010 us     within noise
    NWG=48   14053 us     within noise

Exits identical at every cap. **The GLM-tuned cap transfers**: 16 is clearly worse, 32 and 48 buy
nothing. Null result, and the default stands.

Note on reading XREDUCE2's OWN number: it is not rankable across arms. Its straggler is 98.7% of
its body (12.59's table), i.e. almost all of it is waiting on the slowest peer, and the spread
WITHIN a single arm here is 377 us (NWG=24 measured 1668.5 and 1291.5 on two passes of the same
object). The layer min is what carries signal for this op.

## 12.64 Expert parallelism, and the two things it is not

EP runs on gfx942 now. `PLOW_MOE_PREFILL_EP` gives each rank 48 WHOLE experts at the full
`moe_inter` 2304 instead of all 384 sliced to 288, and 288 was §12.63's finding: `down`'s k-loop is
far too short to amortise the grouped GEMM's per-tile expert-weight reload.

Four paired samples, best of 6 iterations each, TP and EP back to back so co-tenancy hits both:

    arm                 layer min   MoE pair(rk0)   align   XREDUCE2(rk0)   exits
    TP (shipped)         14052 us        863 us    74.9 us      1543 us     -1.36719 / 3.64062 / -0.000761
    EP                   13872 us        293 us    66.3 us      1731 us     identical

The layer falls 180 us, 1.3%. THE MoE PAIR COLUMN IS RANK 0'S AND IT IS NOT THE MODEL'S -- see
below. EP's align is genuinely cheaper than TP's, because it histograms only the slots this rank
owns.

### Reading one rank's trace overstates EP, and by 4x

`rung_run` wrote rank 0 only. That is the wrong statistic for a change whose entire effect is to
make the ranks UNEQUAL, and a collective bills the slowest one. With PLOW_TRACE_ALLRANKS (now
honoured by `rung_run` too), the MoE pair per rank:

    rank        0     1     2     3     4     5     6     7    max    spread
    TP        868   848   878   881   841   865   862   848    881     1.05x
    EP        301   607   652   690   703   731   756   348    756     2.51x

Reproduced on a second pass to within ~20 us per rank, so the spread is structural, not noise.
Rank 0 -- the one rank anybody had been looking at -- is the FASTEST under EP and typical under TP,
which is exactly the shape that flatters EP most.

`moe + XREDUCE2` is very nearly constant across ranks (EP: 2080, 2172, 2113, 1832, 2103, 2122,
2039, 2089), because a rank that finishes the MoE early simply waits longer at the reduction. So
the pair and the collective are ONE block and only their sum is meaningful:

    moe + XREDUCE2, averaged over ranks:   TP ~2185 us   EP ~2090 us

That is the EP win: about 95 us of the block, and 100-180 us of the layer -- 4 to 7 ms over 40
layers. Not the 570 us/layer rank 0's pair column reads as.

### Balancing, re-tested on all eight ranks: still nothing, and slightly worse

`PLOW_MOE_EP_CUTS` now reaches the device. The align takes its window from `i[6]`/`i[7]` when the
host patched them, which works because the instruction stream is uploaded PER RANK, so those two
fields are one rank's alone -- the only way an uneven split can reach a kernel that can otherwise
derive nothing but the even one from its rank.

    arm        moe per rank                                 max   spread   moe+XR avg   layer
    ep-even    298 615 642 683 700 722 746 375               746   2.50x      2210 us   13928 us
    ep-bal     283 620 646 686 326 700 1093 381             1093   3.86x      2314 us   13925 us

The spread got WORSE, 2.50x -> 3.86x, and the block cost 104 us more. The layer did not move.

So the conclusion stands, and now on the right statistic. It also shows plainly WHY: per-rank MoE
time is not a function of tile count. Balancing moved rank 4 from 700 to 326 us and rank 6 from 746
to 1093 us while moving both of their tile counts TOWARDS equality. Whatever orders these ranks, it
is not the quantity the cut points equalise, and a calibrated split is not the lever for it.

`moe + XREDUCE2` stays ~2210-2314 us either way, which is the same observation from the other side:
the MoE op's measured body absorbs each rank's wait for what gates it, so only the block sum is
meaningful and the block is not moved by placement.

### It was never a hardware gap

Four things refused EP on gfx942 and three of them were not requirements. The blanket
`arch != "gfx950"` check; the lean stage-1 A4 quantization scratch; the `_moe2` shuffled
down-weight companions; and `moe_ep_stage2` itself. Only the last is genuinely CDNA4 --
`v_mfma_scale_f32_16x16x128_f8f6f4` -- and it is an ACCELERATION of a contract the interpreter
already implements, because `d_moe_group_down_pf` scatters `part[row_partidx[row]][H]` scaled by
`row_gate[row]` and skips a null weight base, which is exactly what stage-2 does.

Two more refusals were Kimi-K3 constants: `top_k == 16` hardcoded in the packet rewrite AND in the
runtime route builder. V4.1 routes top-6 and was rejected at both.

And `filter_align.hip` hardcoded the gathered-row tile height at 64 while the interpreter GEMM is
compiled MPF_BM=512. The consumer reads the align's tables back as
`rowoff[e] + (mt - tilep[e]) * MPF_BM`, so the mismatch does not degrade -- it addresses ANOTHER
EXPERT'S ROWS and multiplies the tile count by the ratio. The exits still passed; what caught it
was EP's GLU running 2.6x TP's at identical padded MAC work.

### It is not expert load imbalance

Routing is wildly skewed. Dumped from `act.moe_meta` (`--dump`), at 8k only 153 of 384 experts take
any row at all, one takes 5868 against a mean of 128, and the even split's per-rank TILE load runs
1.558x: [42, 29, 17, 45, 22, 34, 19, 23].

`PLOW_MOE_EP_CUTS` replaces the even split with calibrated cut points. UNEVEN CONTIGUOUS cuts
suffice -- the optimal contiguous partition of this histogram reaches 1.004x, as good as an
unrestricted assignment, so no expert is relabelled and the router, the align filter and the weight
binder keep agreeing by construction.

It buys NOTHING. Rank 0's tile count went 42 -> 29 and its MoE pair did not move (298 vs 296 us);
the layer got slightly worse (13914 vs 13854 us). Tile count is not what sets this op's duration:
the grouped GEMM's work items are M-tiles x N-tiles, and at EP's full `moe_inter` there are 18
N-tiles per expert, so 29..45 M-tiles become 520..810 items over 304 CUs -- two or three rounds
either way, with the ROUNDING and not the total deciding. Balancing moved a quantity the op was
never bound by, which is also why EP's MoE ops show an 88-94% straggler.

### The benchmark's routing is not the model's

`rung_run` says so in its own header: `act.x` is a seeded synthetic. A synthetic activation through
a trained router is what collapses 384 experts onto 153. Real traffic routes far more evenly, so
the imbalance measured here is an ARTEFACT and the EP give-back it causes is pessimistic. The MoE
pair win is not an artefact: it is the k-loop length, which does not depend on the input.

Read the -174 us/layer as a LOWER BOUND on EP, not an estimate of it.

### What is actually left

XREDUCE2 is 99% straggler in BOTH arms -- it is pure wait -- and EP adds 301 us per collective to
it. Since balancing is ruled out, what remains is the dispatch structure EP adds: 4 standalone
align launches between cooperative packets, residual 1.5 -> 100.9 us.

So the interpreter align now filters to this rank's expert window itself, off `i[5]` and the
program uniform's rank, reproducing `balanced_expert_range` on-device. EP then needs no specialist
object on gfx942 and adds no standalone dispatch at all. That was the cause:

                             specialist align   interpreter align
    packets (TP has 47)            43                  47
    MOE_ALIGN_PF in trace       absent              68.3 us
    residual                   100.9 us             51.6 us
    XREDUCE2 vs TP              +602 us             +152 us
    traced body vs TP           -148 us             -423 us

Four dispatches per layer, on 8 ranks, cost ~450 us of the 561 us the MoE pair had saved -- not
because the launches are slow but because each is a point where 8 ranks resynchronise, and the
next collective bills the spread. It is worth stating plainly: the MoE arithmetic was never the
problem with EP, and neither was expert placement. The SEGMENTATION was.

Cost of the device-side filter: nothing. The prefill row is unchanged at 256 VGPR / 64720 B LDS /
122 spills, because the window collapses into `< n_exp` guards the op already had.

`PLOW_MOE_EP_CUTS` is refused without the specialist align -- it is a host-side table and the
interpreter derives the even split on-device, so binding weights one way while filtering the other
would hand a rank rows for experts it does not hold.

## 12.65 The biggest op has an arm it never took: the V2 pack-8 gathered flash

FLASH_GATHER_PREFILL is ~2990 us/layer, the largest single op, and 12.59 left it "insensitive to
scatter locality, sensitive to volume" at 3.3x its VALU floor. Volume is the thing to attack, and
the tree already contains the arm that attacks it.

V4.1 emits op 55, whose t7 is a PER-QUERY top-k list, and `d_flash_gather_prefill` walks it as
`d_flash_mla_decode<512, 0, GF, GATHER=true>` -- scalar, one query at a time. The other arm is op
51 with t7 = a per-PACK UNION table (op 119, `IndexUnionPf`), taken by
`d_flash_mla_prefill_v2<512, 0, GATHER=true>`: it stages the pack's union ONCE and applies per-row
u64 membership masks, with all 8 per-rank heads riding the MFMA M dimension.

`d_flash_mla_prefill_v2<512, 0, GATHER=true>` IS ALREADY COMPILED, under PLOW_MLA_PF_NOPE_ARM.
DK=512/DR=0 is exactly V4.1's compressed pass. The arm gates on `n_head == 8`, and V4.1 at TP8 has
n_head == 8. What is missing is only the EMIT: dsv41 stops at ops 117/118 and never builds a union.

Measured on this packet's own selections (`--dump act.index_idx`, 8192 queries x top_k 512):

    selected entries        3932160   (480.0 live per query)
    distinct positions         4094   (the compressed sequence, not the token sequence)

    pack    per-query rows    union rows   reduction
      8           3932160       1619738      2.43x
     16           3932160        961120      4.09x
     32           3932160        516359      7.62x
     64           3932160        263502     14.92x

The 64-row column is a trap and the kernel says so: the per-64-query union was tried for GLM and
measured NET-NEGATIVE, because those unions reach 45-80% of the causal range and the indexer cost
more than the gather saved. GLM_DSA_PF_PACK is 8.

So the honest trade at pack 8 is 2.43x fewer latent row-reads for ~3.3x MORE score/PV math --
8 queries scored densely against a 1582-row union instead of exactly against 480 -- except that the
math moves from VALU to the matrix core, which is ~8x faster per MAC on this part. Net ~2x on the
op, if it behaves. That is ~1.5 ms/layer, 60 ms over 40 layers, and it is the largest single lever
left in the layer.

NOT YET BUILT. The emit work is: declare `iuni`/`iumask`, emit op 119 at pack 8, and swap the
gathered flash from op 55 to op 51 carrying the NoPE bit, t7 = union and i6 = cap -- then route the
segment to the 4-wave flash object with PLOW_MLA_PF_NOPE=1.

## 12.66 Built, and it beat the estimate: 14.09 -> 11.37 ms/layer

> **The absolute layer numbers in this section are measured on a MoE that was computing 17.9% of
> its routed rows** (12.67). The union DELTA is unaffected -- the MoE is identical on both arms and
> the change is entirely in attention -- and it re-measures at -2556 us on the fixed build, against
> -2720 here. The corrected pair is **17122 -> 14566 us/layer**.

12.65 predicted "net ~2x on the op, if it behaves". It behaves, and it is 4.2x. Paired back to
back at 8k, block 2, TP8, gfx942:

    arm                        layer us      gathered us     exits (min / max / mean)
    op 55, 8-wave routing     14086-14165      2969-2996     -1.36719  3.64062  -0.000761
    op 55, V2 routing         13859-13995      2934-2950     -1.36719  3.64062  -0.000761
    op 51 + union             11366-11376     ~620 + 79      -1.36719  3.67188  -0.000761

-19.3% on the whole layer, which is 563 -> 455 ms at 40 layers. The 2.43x fewer row-reads was the
conservative half of the estimate; the other half -- score/PV moving from VALU onto the matrix
core -- is evidently worth more than the 3.3x extra MAC count it costs.

The middle row is the point of the table. It is the same packet as the first with only the routing
changed, so it isolates the four-wave move from the union arm, and it reproduces the exits
EXACTLY. min and mean are then exact on the union arm too; max moves two bf16 ulps, which is the
P-rounding of a different row order. Nothing structural changed.

Three things the estimate did not mention, each of which cost a build/run cycle:

**The routing is not optional and not a knob.** An op 51 carrying t7 has a body on the four-wave
object ONLY -- `interp.hip` traps on it deliberately on the 8-wave one rather than attend the full
causal range of a model trained sparse. Emitting the union without `PLOW_MLA_PF_V2=1` produced
`HSA_STATUS_ERROR_EXCEPTION` with no address in it, which is an `s_trap` and looks exactly like a
memory fault. `devbuild` now derives the segmentation from the instruction, the way `split_mla`
already did.

**i2 is `ctx`, not `ctx / ratio`.** Op 119 derives its scan bound from `kv_len` -- a TOKEN position
-- while indexing `umask[slice * i2 + s]`. GLM reaches op 119 after `DsaPoolExpand` so its two
spaces agree; V4.1 selects COMPRESSED entries directly, and the narrower stride walks off the end
of the row. Also a hardware exception, and the one that had to be ruled out first: emitting op 119
while leaving the flash on op 55 ran clean at 80 us, which put the fault in the other op.

**The four-wave dense body could not write half a partial pair.** V4.1's attention IS a split --
window at partial 0, gathered at partial 1, one FlashMerge folding the sink once -- and
`d_flash_mla_prefill_v2` had no `out_nsplit`/`out_sp0`, so routing the packet there silently moved
the window rows. That is what the control row above was built to catch, and before the fix it read
-1.53125 / 4.50000 / -0.000849 on a packet with no union in it at all.

Op 119 itself costs 79 us/layer and is emitted once per PUBLICATION, not once per reader: the 38
reader layers share `ix.idx`, so they share its union too.

The gathered flash is no longer the largest op in the layer.

## 12.67 The MoE prefill was computing 17.9% of its routed rows, and the tuning sweep rewarded it

`d_moe_align_pf` pads each expert's gathered rows up to a whole tile and publishes
`tiles_e = ceil(count_e / MPF_BM)` with `rowoff[e] = tilep[e] * MPF_BM`. `d_moe_group_pf_a4w4`
walks that table as `rowbase = rowoff[e] + (mt - tilep[e]) * MPF4_BM`. **They are two different
knobs.** MPF4_BM is a fixed 64 -- it is the MFMA fragment map, static_asserted to 32 or 64, not a
tile budget -- and MPF_BM defaults to 64 everywhere in the tree except V4.1, which set 512.

Unequal, the body covers `tiles_e * 64` rows of an expert that has `count_e` of them. The rest are
never computed. Measured on `act.moe_fug` against the align's own metadata, 8k/TP8, block 2:

    tiles (tilep[E])     231
    live routed rows   49152     T*k = 8192 * 6
    rows written        8783     17.9%, equal to sum_e min(count_e, tiles_e * 64) to the row
    first gap                    expert 17, row 128: 1010 live rows, 2 tiles, 2 * 64 covered

At MPF_BM = MPF4_BM = 64 the same probe reads 884 tiles and 49152 of 49152.

**Nothing in the stack could see it.** The skipped rows are not written, so they keep whatever the
arena held -- no fault, no NaN, no shape check, and `part` is f32 so there is not even a layout
tell. The block exit did not move: min and max over a 335 MB residual are attention statistics and
a missing routed FFN does not reach them. Only the exit MEAN shifts, -0.000761 -> -0.000643, and
that lands inside the 1e-6 band build_gfx942.sh had already documented the MoE reduction as being
run-order noisy in. "Exits identical at every arm" was true and meant nothing.

**And it read as a 45% speedup.** This is the part worth carrying forward. Raising MPF_BM cuts the
tile count, and every tile covers 64 rows however tall the tile claims to be, so each step of the
sweep deleted arithmetic and got faster for it:

| arm | reported MoE pair | rows actually covered |
|---|---|---|
| 192 / BK=64 | 1583.7 us | 380 tiles x 64 = 24 320 |
| 384 / BK=32 | 998.4 us | 289 tiles x 64 = 18 496 |
| 512 / BK=32 | 868.4 us | 231 tiles x 64 = 14 784 |

Monotonic, with a clean rationale attached to it ("the pair is per-tile weight-reload bound"), and
every arm's exits identical. The true pair is **4049 us**.

The lesson is not "check your tiles". It is that a sweep over a knob that can delete work will
report the deletion as a win, and the only defence is a coverage probe that does not go through the
same knob -- here, the align's `count_e` against what the GEMM's output actually contains. An A/B
cannot find this, because both arms are wrong and the faster one looks better.

`op_moe.h` now static_asserts `MPF_BM == MPF4_BM` whenever the A4W4 body is compiled.

### Where this leaves the campaign

| | per layer | 40 layers |
|---|---|---|
| op 55 gathered, MoE fixed | 17 122 us | 685 ms |
| op 51 union, MoE fixed | **14 566 us** | **583 ms** |

583 ms, not the 455 ms 12.66 reported. The MoE pair is now the largest single line in the layer at
4049 us -- ahead of `GEMM_FP8_MX` at 2813 -- and it is genuinely 884 tiles of work rather than 231.
That makes MoE scheduling a live target again for the first time since 12.46, and on honest
arithmetic: 884 tiles x 3 n-tiles = 2652 GLU tiles over 304 CUs is 8.72 waves, so the 54% tail the
231-tile build showed is gone on its own.

## 12.68 EP re-measured on arithmetic that is actually happening: DOWN is 3.7x

> **CORRECTION (12.69): the `-430 us` layer figure below is a MIN, and EP's min is a lucky
> iteration.** On the median EP is +2.4 ms/layer WORSE than TP. The MoE numbers in this section are
> right and the k-loop argument stands; the layer verdict is inverted. Read 12.69.

12.64 priced EP at ~95-180 us/layer. That was measured against a MoE pair of 858 us which was
doing 17.9% of its work (12.67), so it was pricing EP against a baseline that barely existed. On
the fixed build the MoE pair is 4129 us and EP's case is completely different.

Interleaved, 8k/TP8, block 2, union arm, MoE fixed, two repeats each:

| | TP | EP | |
|---|---|---|---|
| `MOE_GROUP_DOWN_PF` | 2531 us | **688 us** | 3.68x |
| `MOE_GROUP_GLU_PF` | 1597 us | 1143 us | 1.40x |
| MoE pair | 4129 us | **1832 us** | -55.6% |
| layer | 14581 us | **14151 us** | -430 us |

Exits identical at -1.36719 / 3.67188 / -0.000643.

**DOWN is where EP pays, and the reason is its k-loop.** Under TP, `down`'s K is
`moe_intermediate / tp` = 2304/8 = 288 -- three MPF4_BK=128 tiles, with the per-tile cost (the
expert weight pointers, the LDS double-buffer fill, the `row_partidx`/`row_gate` epilogue) amortised
over almost nothing. EP shards the EXPERTS instead of the intermediate, so K is the full 2304 and
the same total MACs run through an 18-deep k-loop. GLU never had the problem -- its K is the hidden
5120 either way -- and its 1.40x is the ordinary benefit of 48 experts instead of 384.

That 65 TF/s DOWN was running at is the clearest instance of this campaign's recurring pattern
(12.20): an op 5-50x off its roofline is not short of bandwidth, it is short of work per unit of
overhead.

**The collective takes most of it back.** `XREDUCE2` goes 1374 -> 5833 us on rank 0's trace, so the
layer moves only -430 of the -2297. Per-rank attribution is not reliable here for the reason 12.64
gives -- under EP, rank 0 is the FASTEST rank and its trace reads the others' skew as its own wait
-- but the direction is not in doubt: all eight ranks' `moe pair + XREDUCE2` sums to a near-constant
7.6 ms under EP (spread +/-1%), against 5.5 ms under TP. The expert placement is uneven and the
collective bills the slowest rank. (12.71 retracts this sentence: the ranks are not
skewed.)

**Balancing still does not help, and now we know why it cannot.** The even contiguous split gives
tile loads `[183, 129, 28, 217, 63, 152, 35, 77]`, 1.96x imbalance; a DP over contiguous cuts gets
`[115, 111, 117, 121, 121, 121, 105, 73]`, 1.095x. Applied through `PLOW_MOE_EP_CUTS` the layer
reads 14011 us even against 14040 us balanced -- no change, exactly as 12.65's predecessor found on
the broken build and for the same reason: **per-rank MoE time is not a function of tile count.** The
1.96x tile imbalance and the 20x per-rank time spread are not the same quantity, so a cut list that
fixes the first does not touch the second.

(The routing here comes from `rung_run`'s seeded synthetic `act.x`, so these specific cuts are a
benchmark artefact and not a shippable default. The mechanism and the negative result are real.)

### Where the layer stands

    TP,  union arm, MoE fixed     14 566 us/layer     583 ms
    EP,  union arm, MoE fixed     14 151 us/layer     566 ms

EP is not enabled by default: it needs `PLOW_MOE_PREFILL_EP_MAX_EXTRA_BYTES` raised to 2 GB,
because the largest rank holds more experts than an even byte split assumes. -430 us/layer is worth
having but it is a config decision with a memory cost attached, so it stays opt-in until the
placement skew that feeds the collective is addressed -- which is the actual open problem, and it
is a scheduling problem, not a tiling one.

## 12.69 EP loses on the median, and the min is what made it look like a win

12.68 quoted `min` layer times, as every arm in this document has. For TP that is harmless -- its
min and median differ by 0.8% -- and for EP it inverts the result. Three interleaved repeats,
12 iterations each, 8k/TP8, block 2, union arm, MoE fixed:

| | min | **median** | max |
|---|---|---|---|
| TP | 14 653 - 14 736 us | **14 792 - 14 834 us** | 16 488 - 16 558 us |
| EP | 14 076 - 14 218 us | **17 142 - 17 341 us** | 17 633 - 17 758 us |

EP's min is ~570 us BETTER and its median is ~2 450 us WORSE, reproducibly, three times out of
three. TP's min-to-median gap is 0.8%; EP's is 22%. **That gap is the placement skew**, and a min
over twelve iterations is precisely the statistic that hides it -- it reports the one iteration
where the routing happened to land evenly.

The all-ranks traces said so plainly and I read the wrong column first: every rank's total body is
16 157 - 16 185 us under EP against 13 986 - 14 044 us under TP. There is no rank for which EP is
cheaper. The body agrees with the median, not the min.

### What the constant sum means

Per rank, `moe pair + XREDUCE2` is near-constant -- 7 524 to 7 720 us under EP (+/-1.3%), 5 295 to
5 568 us under TP (+/-2.5%) -- while the MoE half alone spreads 20x under EP (319 to 6 367 us).
That is the signature of two adjacent ops with a barrier between them: whatever a rank does not
spend computing its experts, it spends waiting at the collective, and the SUM is set by the
slowest rank's MoE plus the collective's own cost.

    EP:  slowest rank MoE 6367 + collective ~1250  =  ~7600
    TP:  slowest rank MoE 4171 + collective ~1270  =  ~5440

So EP's problem states exactly: **it halves the AVERAGE MoE (2205 us against TP's 4116) and raises
the MAXIMUM (6367 against 4171).** A barrier bills the maximum. The -55.6% on the pair is real and
it is measured on the average, which is not the quantity that reaches the layer.

### And the maximum is not predicted by tile count

Even-split per-rank tiles against per-rank MoE time:

    rank      0     1     2     3     4     5     6     7
    tiles   183   129    28   217    63   152    35    77
    MoE us 1778  2512   317  1428  6367  3052  1411   740

Rank 4 holds 63 tiles and is the SLOWEST at 6367 us; rank 3 holds 217 -- 3.4x the rows -- and runs
in 1428. The relationship is not weak, it is inverted. This is why `PLOW_MOE_EP_CUTS` cannot work:
it balances tiles, and tiles do not predict time. Balanced cuts take the tile imbalance from 1.96x
to 1.095x and move the layer by 0 us (14 011 even against 14 040 balanced).

**Whatever sets a rank's MoE time under EP, it is not the amount of gathered-row arithmetic it was
given.** 12.71 answers that: it is not a measurement of work at all, and there is no imbalance to
balance.

### Honest position

    TP, union arm, MoE fixed     median 14 810 us/layer     ~592 ms at 40 layers

EP stays off. It is not a 430 us win with a memory cost; it is a 2.4 ms/layer regression whose one
genuine result -- `down`'s k-loop going from 3 iterations to 18, worth 3.7x on that op -- is worth
capturing some other way. The obvious candidate is to give `down` a deeper k-loop WITHOUT sharding
experts across ranks, since the k-loop depth was the whole mechanism and EP's rank skew was only
the delivery vehicle.

## 12.70 What actually binds DOWN, priced on correct arithmetic

EP's 3.7x on `MOE_GROUP_DOWN_PF` was delivered by rank sharding, which costs more than it returns
(12.69). The mechanism underneath it is not about ranks at all, and it is visible by comparing the
two halves of the MoE pair, which run the SAME body over the SAME 56 576 padded rows:

| | K | N | k-tiles/tile (BK=128) | tiles | MACs | body | achieved |
|---|---|---|---|---|---|---|---|
| GLU | 5120 | 576 | **40** | 884 x 3 = 2 652 | 166.8 G | 1 597 us | 208 TF/s |
| DOWN | 288 | 5120 | **3** | 884 x 20 = 17 680 | 83.4 G | 2 536 us | **65 TF/s** |

DOWN does HALF the multiply-accumulates in 1.6x the time. The two differ in exactly one structural
way: GLU's tile amortises its prologue, staging double-buffer fill, barriers and epilogue over 40
k-iterations, and DOWN's over 3 -- and the third of those three is only a quarter live, since
288 = 2.25 x 128. **13x less amortisation, 3.2x less throughput.**

K is 288 because `moe_intermediate` is TP-sharded: 2304/8. That is the whole story, and it is why
EP fixes it (K becomes the full 2304, 18 k-tiles) without the fix having anything to do with expert
placement. The placement skew is the delivery vehicle, not the mechanism.

So the lever is to give DOWN a deeper k-loop, or fewer tiles, without sharding experts across
ranks. Three candidates, none of them free:

1. **A taller row tile for the DOWN arm only.** `MPF4_BM` is 64 and pinned there by the GLU arm's
   `static_assert(!GLU || MPF4_BM == 64 && MPF4_BN == MPF4_WNc * 64)`; the body is already
   templated on `GLU`, so a per-arm tile is a contained change. BM=128 halves the tile count.
   Costs double the accumulators on an object at the 256-VGPR cap -- which is how the A4W4
   `GEMV_F32` arm 3 lost (12.19), so it must be measured and not assumed.
2. **A wider N tile for DOWN.** `tnc = 5120 / MPF4_BN` = 20. BN=512 halves it. Same static_assert
   blocks it for the shared constant, same per-arm escape.
3. **The n-loop inside the row-tile,** so A and the row metadata are staged once per row-tile
   instead of 20 times. Smallest register cost of the three, smallest win: A is ~20% of a tile's
   staged bytes.

The 12.25 ceiling instruments that concluded "bound by neither its k-loop nor its scatter" were run
against the 231-tile build, i.e. on 17.9% of the rows (12.67), so they need redoing before any of
this is chosen. That is the first step, not the third.

## 12.71 There is no EP imbalance. Every rank is slower by the same 2.15 ms

> **WRONG, and corrected in 12.72.** Equal total body per rank is what a barrier at each end of
> the block GUARANTEES -- it is not evidence that the ranks did equal work. The zero correlations
> and the verified partition below stand; the headline does not. There IS an imbalance, it is
> genuine compute, and 12.72 localises it.

Three rounds of this campaign have tried to fix EP by balancing expert placement -- even cuts, a DP
over contiguous cuts, and a hunt for what predicts a rank's MoE time. The premise was wrong.

**Total body, per rank, one traced iteration, 8k/TP8, block 2, union arm, MoE fixed:**

    TP   13 986  14 032  14 037  14 044  14 043  14 044  13 986  14 039     spread 58 us  (0.4%)
    EP   16 175  16 169  16 172  16 174  16 184  16 157  16 159  16 165     spread 27 us  (0.17%)

Every EP rank is slower than every TP rank, by 2.15 ms, to within 0.17%. **There is no straggler.**
A balancing pass cannot help a distribution whose spread is already a sixth of a percent.

The 20x per-rank "MoE pair" spread that motivated all of it (319 to 6 367 us) is an attribution
artefact, and the constant sum is the tell: `moe pair + XREDUCE2` is 7 524 - 7 720 us on every rank
(+/-1.3%) while its two halves swing 20x and 6x in opposite directions. The trace is splitting one
fixed interval at a boundary that moves with arrival phase. That is also why the correlations came
out at exactly zero rather than merely weak:

    corr(per-rank tiles, per-rank MoE us)   -0.001
    corr(per-rank rows,  per-rank MoE us)   -0.003
    corr(used experts,   per-rank MoE us)   -0.171

Zero correlation with three different measures of work is not a hard scheduling problem. It is a
number that does not measure work.

**The partition itself was never in doubt, and is now verified.** Per-rank `act.moe_meta` under EP
(`--dump` now honours `PLOW_TRACE_ALLRANKS`, since rank 0's metadata says nothing about what rank 4
was asked to compute):

    rank      0     1     2     3     4     5     6     7    total
    rows  10595  7314   988 12815  3140  8826  1296  4178    49152
    tiles   183   129    28   217    63   152    35    77      884

884 tiles, exactly TP's single-rank total, each rank's window exactly its cut. The expert weights
are uniform too -- every expert's `w2.scale` on layer 2 is 368 640 E8M0 bytes with 0.00% zeros and
0.00% 0xFF, so no rank is running MFMAs against denormal or NaN scales.

### What this leaves

EP costs a uniform +2.15 ms/layer while making the MoE arithmetic 2.3 ms cheaper, so something in
the EP path costs ~4.4 ms that the TP path does not, on every rank equally. It is not placement, not
skew, not the tile map and not the weight values -- all four are now excluded by measurement. The
candidates left are structural and cheap to enumerate: the second `XREDUCE2`'s operand shape
(`i2 = 2 621 440, i3 = 2, i4 = 3` under TP -- unchecked under EP), the `MOE_COMBINE_PF` contract
when experts are whole rather than sliced, and the EP align's own four-dispatch segmentation, which
12.64 already priced at ~450 us at the next collective on the broken build.

**`PLOW_MOE_EP_CUTS` should be treated as answered in the negative.** It balances tiles; tiles are
not the problem; nothing is imbalanced. It stays in the tree as a diagnostic, not as a tuning knob.

## 12.72 The imbalance is real, it is all in one packet, and it is 9x per TILE

12.71 concluded "no imbalance" from equal per-rank total body. That inference is invalid: the block
is bracketed by collectives, so every rank necessarily leaves it at the same instant. Equal totals
are what the barrier does, not what the ranks did. Retracted.

The per-PACKET view, which is what should have been read, localises the whole regression to one
instruction. Rank 0, one traced iteration:

| inst | op | TP | EP |
|---|---|---|---|
| 28 | XREDUCE2 (post-attention) | 667.7 us (strag 657.4) | 649.5 us (strag 639.9) |
| 43 | MOE_GROUP_GLU_PF | 1568.3 | 1154.6 |
| 44 | MOE_GROUP_DOWN_PF | 2476.8 | 692.8 |
| 45 | MOE_COMBINE_PF | 330.2 | 314.7 |
| **46** | **XREDUCE2 (post-MoE)** | **729.7 (strag 720.5)** | **5117.2 (strag 5105.2)** |

The first collective is unchanged. **All of EP's +2.15 ms sits in the second one, at 99.8%
straggler** -- rank 0 finishes its own MoE in 1847 us and then waits 5117 us for a peer. The
arithmetic closes: 1847 + 5117 = 6964, and the slowest rank's MoE is 6367 + its own ~600 us of
collective.

An instruction diff of the two emitted programs shows only four instructions differ at all (op 84
gains the EP degree in `i5`; ops 85/86 take `i_moe` 288 -> 2304 and the `_ep` weight table; op 87
gains the routing table on `t4`), and **XREDUCE2 is byte-identical**. The collective is not doing
anything different. It is waiting.

### And the wait is genuine compute on the slow rank

Straggler separates the two cases, and it is unambiguous here:

    rank 2  GLU body  199.7 us, straggler 188.8  -- 94% spread: a few workgroups work, most exit
    rank 4  GLU body 3920.1 us, straggler 279.1  --  7% spread: nearly every workgroup is busy

Rank 4 is not waiting. It is computing, for 3.9 ms, on 63 row-tiles. Per GLU tile (`tnc` = 18 under
EP, 3 under TP):

    rank 2   504 tiles   199.7 us   0.40 us/tile
    TP      2652 tiles  1568.3 us   0.59 us/tile
    rank 4  1134 tiles  3920.1 us   **3.46 us/tile**

Rank 2 is FASTER per tile than TP. Rank 4 is 5.8x slower than TP and 8.7x slower than rank 2, on
tiles of identical shape running identical code. Every measure of how much work the rank was given
predicts a ratio of 2.25x.

### What is excluded, by measurement

* **The partition.** Per-rank `act.moe_meta`: 884 tiles total, exactly TP's, each rank's window
  exactly its cut.
* **Tile count, row count, used-expert count.** Correlate with per-rank MoE time at -0.001, -0.003
  and -0.171.
* **The weight values.** Every expert's `w2.scale` on layer 2 is 368 640 E8M0 bytes, 0.00% zero and
  0.00% 0xFF -- no rank is running MFMAs against denormal or NaN scales.
* **The collective's shape.** Byte-identical instruction; the post-attention one is unchanged.

So the open question is sharp and small: **why does one rank's A4W4 tile cost 3.46 us when another
rank's identically-shaped tile costs 0.40 us?** It is per-tile, it is compute, it follows the
expert index range rather than the rank (balanced cuts move the slow rank from 4 to 5, tracking
experts ~215-290), and it is reproducible to within 1% across runs.

That is the thing to answer next, and it is worth answering: it is the ONLY reason EP is not a
2.3 ms/layer win, and 12.70's k-loop argument says the same mechanism is worth having under TP too.
`PLOW_MOE_EP_CUTS` remains a dead end for the reason 12.71 gives -- it balances a quantity that does
not predict the cost.

## 12.73 The benchmark was feeding the layer its own output, and it is why 12.68-12.72 disagree

`rung_run` wrote `act.x` once and then launched the rung `--iters` times. The rung writes its result
back over `act.x`. So iteration 2 ran on iteration 1's output, iteration 30 on a value pushed
through the same layer thirty times, and the activations degenerate.

The tell is in arrival order, which none of 12.68-12.72 looked at. TP is flat; EP ramps:

```
TP  : 20814 15115 14872 14743 14608 14788 ... 14697   (flat from iter 3)
EP  : 15145 15144 14090 14254 16157 16899 ... 19087   (climbs, no plateau)
```

Not thermal: a 100 ms gap between iterations leaves the curve identical, and a fresh process starts
fast again on a GPU the previous process had just left hot. Iteration-count driven, so it is state,
and the state is the entry.

Restoring the entry every iteration (the upload is synchronous and sits outside the timer, ~47 ms of
it) removes the drift from both programs and changes what is being measured:

| op                  | degenerated | real entry |
|---------------------|------------:|-----------:|
| FLASH_MLA_PREFILL   |       767.1 |     1991.5 |
| MOE_GROUP_DOWN_PF   |      2428.6 |     2696.7 |
| MOE_GROUP_GLU_PF    |      1581.5 |     1724.4 |
| RMSNORM             |       321.5 |      455.7 |

Attention is 2.6x. The degenerated index picks fewer distinct KV blocks and the router lights fewer
experts, so the grouped GEMM bills fewer padded tiles. Every attention and MoE number in 12.x before
this section was measured on a cheaper model than the one being served.

## 12.74 With a real entry, a balanced EP beats TP by 1.83 ms/layer -- but see 12.77, the balance is an oracle

Medians over 20 iterations, interleaved, repeated (min within 1.5%):

| config                          | max tiles/rank | us/layer | 40 layers |
|---------------------------------|---------------:|---------:|----------:|
| TP8                             |              - |    16989 |    680 ms |
| EP, even 48/rank                |            217 |    17150 |    686 ms |
| EP, `0,72,144,216,228,240,300,342,384` | 254     |    17755 |    710 ms |
| EP, `0,36,88,149,181,215,241,313,384`  | 121     |    15164 |    607 ms |

The ordering is exactly the max-tiles ordering, 121 < 217 < 254. Tile balance was always the model;
12.71 could not see it because the benchmark had already flattened the histogram it balances.

Retractions this forces:

* 12.69's "EP loses 2.4 ms/layer on the median" -- that 2.4 ms was the drift, not EP.
* 12.71 and 12.72 both, and for the same reason. 12.72's "one rank's tile costs 9x another's" came
  from reading per-rank `MOE_GROUP_*` body time as compute. It is not: on the isolated-band run rank
  3 holds 26 tiles and spends 3336 us in GLU while rank 7 holds 40 tiles and spends 222 us. Fewer
  tiles, 15x the time. GLU body absorbs the wait, and GLU+DOWN+XREDUCE2 is conserved across ranks to
  within 4% -- which is what the barrier guarantees and therefore what 12.71 should never have read
  as balance either.
* The `ep_expert_range` comment in `crates/plowrt/src/exec/amd.rs`, which said balancing was
  measured and inert.

What stands: 12.67's MoE coverage bug (a correctness fix, verified by row count, not by timing), and
12.66's union arm (verified by exit values).

Caveat on the cut list: it is calibrated on the seeded synthetic entry's histogram, which lights 153
of 384 experts. It is a demonstration that the tile load is the lever, not a shippable constant. The
shippable form computes the assignment from the router's own histogram; that is the open item.

Honest position against the goal: 607 ms at 40 layers, against 200 ms. Attention is now the largest
single op at ~2.0 ms/layer and was under-measured 2.6x until this section.

## 12.75 The gathered attention was giving away a third of the machine to its own load balancer

With the entry restored (12.73) attention is the second-largest op, and two thirds of its packet was
straggler: per-workgroup durations on the gathered packet ran min 554 / median 1357 / max 1866 us.

`mla_pf_fold` pairs the causal ends -- 0, NQ-1, 1, NQ-2 -- so that a worker walking the range
SEQUENTIALLY averages cheap and expensive tiles. It sends even indices to the front half and odd to
the back. But the work assignment is a stride of `nblk`, and **nblk = 304 is even**, so every item
one workgroup draws has the same parity. Half the workgroups got none but front packs, half none but
back packs. The mean duration by CU octile showed it as a step, not a ramp: CUs 0-114 at 1330 us,
CUs 115-303 at 1030 us.

The plain index already interleaves under a stride -- workgroup s draws s, s+304, s+608, s+912, one
from each quarter -- so the fold was the opposite of an improvement. The dense arm keeps it: its
tiles are window-bounded and equal-cost, so it is inert there.

Second, the rounds are coarse: 1024 packs over 304 workgroups is 3.37 rounds and the packet pays 4.
The dense arm already had a causal KV-split for exactly this; the gather arm now splits its union
walk the same way (`PLOW_MLA_GATHER_SPLIT`, default 2).

| arm                | FLASH_MLA_PREFILL | straggler | FLASH_MERGE |    layer |
|--------------------|------------------:|----------:|------------:|---------:|
| fold, no split     |           2053 us |    682 us |      141 us | 15043 us |
| no fold, no split  |           1571    |    294    |      164    | 14653    |
| no fold, split 2   |           1441    |    179    |      216    | 14540    |

Three interleaved pairs each; exits identical to the digit throughout (min -1.36719, max 3.67188,
mean -0.000643), as they must be -- both changes move which workgroup does which work and nothing
else. Split 4 is a wash: the partials are `[T][nh_l][512]` f32, so each extra split writes and reads
another 134 MB and the merge gives back what the straggler saves.

Layer 14540 us, so 582 ms at 40 layers against the 200 ms goal.

Where the remaining 14.5 ms sits, on the real entry, EP with balanced cuts:

| op                                   |    us | share |
|--------------------------------------|------:|------:|
| GEMM_FP8_MX                          |  2805 |   19% |
| MoE (GLU+DOWN+combine+router+align)  |  2812 |   19% |
| XREDUCE2                             |  2066 |   14% |
| mHC (GEMV_F32 + HYPER_CONN pre/post) |  2169 |   15% |
| FLASH_MLA_PREFILL                    |  1441 |   10% |
| everything else                      |  3247 |   23% |

Two things stand out for the next pass. GEMM_FP8_MX still carries a 48% straggler, which is the same
class of bug just found here and has not been looked at. And mHC costs as much as attention: the
`GEMV_F32` alone is 1060 us for a `[8192, hc_mult*5120] x [.., 15]` reduction, which is bandwidth on
the f32 residual, not arithmetic.

## 12.76 Pipeline parallelism does not buy single-request prefill latency here

`Parallel::Pp`/`Dp`/`Ep` are rejected in `crates/plowc/src/lib.rs:2601` with a test asserting the
rejection, so none of this is wired. Before wiring it, the arithmetic, from 12.75's measured layer:

TP8 costs 14.54 ms/layer, of which XREDUCE2 is 2.07 ms. The other 12.47 ms is compute on a machine
split eight ways, so one GPU at FULL width is ~8x that, ~100 ms/layer. Under PP8 each GPU owns 5 of
the 40 layers, pays no per-layer collective, and runs full width: ~499 ms per stage.

For ONE request that is strictly worse, because only one stage is live at a time -- 8 x 499 ms. The
fix is micro-batching, and it does not recover the gap: splitting 8192 tokens into 8 chunks of 1024
gives ~62 ms per (stage, chunk) and a filled pipeline of 8 + 8 - 1 = 15 slots, so ~936 ms against
TP8's 582 ms. PP trades 83 ms of whole-model collective for a pipeline bubble and the loss of
parallel width, and the trade is bad at this shape.

PP is a THROUGHPUT structure: it raises tokens/s at fixed memory by keeping every stage busy with
DIFFERENT requests. It does not shorten one prefill. The same is already on record for context
parallelism -- the 2026-09-12 CP study finds CP does not beat TP8 at C20 -- and DCP in this tree is
decode-side KV sharding, which a prefill does not exercise.

So the 200 ms target is not reachable by changing the parallelism. At 14.54 ms/layer it needs
5.0 ms/layer, and the whole identified op-level backlog -- GEMM_FP8_MX's 48% straggler (~390 us),
`GEMV_F32`'s L2 wall (~600 us if it reached its own floor), the MoE's 6x-off-roofline grouped GEMM,
and the 2.07 ms collective -- comes to about 7.5 ms even if every item went to zero. The remaining
gap is arithmetic efficiency, not partitioning: the MoE pair alone bills 2154 us against a ~335 us
fp8 roofline for the same MACs.

## 12.77 EP's balance win is an oracle bound; TP is the right MoE parallelism at this shape

12.74 scored a hand-solved cut list against the histogram of the run that produced it. That is not
a schedule -- a request does not hand you its routing before it runs. The question is whether any
REALIZABLE static expert-to-rank binding gets near it.

**Not from the checkpoint.** The router is `top6(x @ gate.weight.T + gate.bias)`, both model
properties, and the trained bias is a real concentrator (its std is ~1.0x the token-signal std at
layer 2, and isotropic tokens light only 291/384 experts). But a Monte-Carlo through the real gate
predicts the observed per-expert counts with corr 0.132 (rank corr 0.423), and cuts solved on the
prediction score 228 max tiles against the even split's 217 -- no better.

**Not from relabeling.** `corr(expert index, tiles) = -0.046`, so the imbalance is not index
clustering that a permutation could break up. Round-robin ownership gets 198 against contiguous
217; three random permutations give 200, 159, 210. All still ~1.8x.

**Because the tail is the constraint.** The top 8 experts carry 68.6% of all rows, and the hottest
single expert is 92 tiles against a fair share of 110.5 -- 83% of one rank's entire budget by
itself. Unrestricted LPT with the histogram in hand reaches 1.005x; without it, one unlucky
placement of one expert blows the bound. No static binding absorbs that.

| assignment                          | max tiles/rank | imbalance |
|-------------------------------------|---------------:|----------:|
| contiguous even (ships today)       |            217 |    1.96x  |
| round-robin                         |            198 |    1.79x  |
| checkpoint-derived cuts             |            228 |    2.06x  |
| oracle contiguous cuts (12.74)      |            121 |    1.10x  |
| oracle LPT, unrestricted            |            111 |    1.005x |

So the realizable EP number is the even split's 17150 us, against TP8's 16989 us. **TP is balanced
by construction** -- every rank runs every expert at `moe_inter/8`, so routing skew cannot reach the
partition at all -- and it is what ships. EP buys full-width `down` (12.70's k-loop argument) and
gives it back to the tail.

This closes the EP question rather than leaving it open: the 1.83 ms in 12.74 is HEADROOM that only
dynamic expert placement could reach, and dynamic placement means moving 6 MB of weights per expert
per batch. The MoE scheduling wins that were real this campaign are 12.67's coverage bug -- the
grouped GEMM was computing 17.9% of its routed rows -- and the within-rank work distribution, which
is histogram-independent.

## 12.78 The shippable number is 662 ms, not 582

12.75's 14540 us was measured on the EP packet with 12.74's cut list. 12.77 retired that cut list as
an oracle, so the number it produced is not a configuration anyone can run. The shipping path is TP
(`moe_prefill_ep` is opt-in and off), and the attention rebalance had not been measured there.

Re-measured on TP, both sides carrying 12.67's MoE coverage fix, three interleaved pairs of 16:

| TP8, real entry                  |  median us | 40 layers |
|----------------------------------|-----------:|----------:|
| before the attention rebalance    |      16978 |    679 ms |
| after (no fold, gather split 2)   |      16539 |    662 ms |

Exits identical on both sides (min -1.36719, max 3.67188, mean -0.000643). -440 us/layer, which
tracks the -450 measured on the EP packet, as it should: the change is in the attention body and
does not know which MoE parallelism it is running under.

A NOTE ON THE BASELINE. The first attempt at this comparison read 13.6 ms for "before" and exit mean
-0.000761. That object set predates 12.67 and was computing 17.9% of its routed rows -- it is fast
because it is wrong. The exit mean is the tell, and it is worth keeping as the check: -0.000761 is
the broken MoE, -0.000643 the fixed one. Any V4.1 layer-2 number quoted against -0.000761 is a
number for a different model.

Session position against the 200 ms goal: 662 ms, so 3.3x. Attention is now 1441 us of the 16539,
and the backlog in 12.75 is unchanged.

## 12.79 Overlapping the TP seam: built, and the shape refuses it

12.76 argued pipeline parallelism across GPUs is the wrong structure for one prefill. The structure
that IS right for a TP collective is to pipeline it against its own producer, and the GLM prefill
seam already had it (`emit_xreduce_twoshot_band`). Wired for V4.1: `PLOW_DSV41_XR_BAND=K` splits the
`wo_b` GEMM into K row bands, each feeding its own two-shot, and `PLOW_DSV41_XR_BAND_CUS` gives the
band collectives a CU prefix so workgroups outside it claim the next band's GEMM. Op 184 gained
`a_row0`/`c_row0`.

| V4.1 layer 2, 8k, TP8, real entry | median us |
|-----------------------------------|----------:|
| unbanded (default)                |     16471 |
| K=2, band cus 152                 |     16549 |
| K=2, band cus 228                 |     16542 |
| K=2, full width                   |     16553 |
| K=4, band cus 152                 |     16798 |

Three interleaved folds of 16 iterations; run-to-run spread ~120 us. Everything but K=4 is inside
it, and K=4 is worse. Exits identical at every K, so the row-band decomposition is right -- it is
the overlap that does not pay.

The reason is shape, and it is measurable rather than a matter of opinion. This seam's producer is
ONE `[8192, hidden] x [orow]` GEMM of a few hundred us; its transfer is ~750 us. **There is less
compute to hide the transfer under than there is transfer**, so finer bands only shrink the thing
doing the hiding. The op that IS big enough is the MoE at 4.5 ms, and the MoE sits downstream of
this collective through the residual and the norm, so it cannot hide it.

What would work is sequence parallelism (`PLOW_GLM_SEQ_PAR` in the GLM path): the all-reduce becomes
reduce-scatter plus all-gather and the norm/residual between seams runs on `t/tp` rows. That halves
the seam traffic instead of trying to hide it, and it cuts the elementwise ops eightfold. It is not
wired for V4.1 and is the open item this section leaves.

## 12.80 Where the shipped number stands

The TP ladder, each step measured against its own contemporaneous control, real entry, MoE coverage
fix throughout:

| step                                   | us/layer | 40 layers |
|----------------------------------------|---------:|----------:|
| before this campaign's attention work   |    16975 |    679 ms |
| + gather fold removed (12.75)           |    16560 |    662 ms |
| + gathered union split 2 (12.75)        |    16471 |    659 ms |

12.78 attributed its 16539 to "no fold, gather split 2". It was fold-only: the packet it measured
was emitted by a `plowc` built before the split default changed, which the 6-byte `opart`/`mlpart`
sizing difference shows (nsplit 2 against 3). The split is worth a further ~90 us, in the same
direction as the -113 us it measured on the EP packet, and at the edge of the TP noise.

659 ms against the 200 ms goal, so 3.3x, and the gap is arithmetic efficiency: the MoE pair bills
2154 us against a ~335 us fp8 roofline for the same MACs.

## 12.81 Sequence parallelism is worth 2.55 ms/layer, and it is the last big structural win

12.79 ended by naming sequence parallelism as the seam structure that would pay where overlapping
does not. This prices it before building it.

The mHC and norm packets are PER-TOKEN work, and every rank runs all `t` rows of it -- eight GPUs
computing the same 8192 rows of the same residual stream. Under SP each rank keeps only its `t/tp`
band between the seams, the all-reduce becomes reduce-scatter plus all-gather (the same traffic a
two-shot all-reduce already moves), and that replicated work divides by `tp`.

`PLOW_DSV41_SP_ABL=1` gives those packets `t/tp` rows and changes nothing else. The answer is WRONG
by construction -- 7/8 of the rows are never written, so the residual is garbage from layer 1 -- but
the cost is the cost SP would reach. Three interleaved folds of 16 iterations, medians:

| packet            | replicated | banded | delta |
|-------------------|-----------:|-------:|------:|
| HYPER_CONN_POST   |     537 us | 114 us | -423  |
| HYPER_CONN_PRE    |     574    | 175    | -399  |
| GEMV_F32          |    1040    | 908    | -132  |
| RMSNORM           |     327    | 307    |  -20  |
| **layer**         |  **16362** | **13809** | **-2553** |

**-15.6%, or 102 ms over 40 layers**, taking 659 ms to ~557 ms.

Two things in that table are worth reading carefully. `GEMV_F32` barely moves: at `t/8` it is
occupancy-bound rather than work-bound, because a wave's `W` read is `CG * K` whatever its row count
and 256 row-blocks cannot fill 2432 waves -- the same wall arm 6 documents. And `RMSNORM` barely
moves because the FFN norms are not in the ablated set; banding those is additional.

What building it needs, and why it was priced rather than built here: a reduce-scatter/all-gather
pair around the mHC segment (both helpers exist and are generic -- `emit_xreduce_scatter`,
`emit_xall_gather`), band views of the residual (the runtime's `<base>@band<t>` binding is generic
too, `amd.rs:8917`), and a THIRD peer-slot region in the dsv41 layout. That last one is the
obstacle: `DevBlob::parse` recovers `slot_bytes` as `max(i[2])` over the collectives and the TP
loader validates it, so the slot map is a runtime contract, and getting it wrong yields a plausible
wrong answer rather than a fault -- which is the failure mode this campaign has already hit twice
(12.67, 12.73).

The ablation is default off and the unablated emit is byte-identical to the packet 12.80 measured.
