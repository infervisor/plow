# GLM-5.3-FP8 on MI300X — bringup and TP4 baseline

Measured 2026-09-07 on an 8x MI300X (gfx942, CDNA3, 304 CU, 192 GiB/card) host, ROCm 7.14.0
from the flake. **Read the caveats in §5 before quoting any number here.**

## 1. GLM-5.3 needs no new architecture support

`zai-org/GLM-5.3` is `model_type: glm_moe_dsa` — the same architecture as GLM-5.2-FP8, which
this tree already serves through `glm_main`. The evidence:

- `config.json` differs from GLM-5.2-FP8 in `transformers_version` alone (5.15.0 vs 5.12.0).
- The two checkpoints' tensor name sets are **equal**: 118,629 names each, `set(a) == set(b)`.
- Same 141 shards, same `total_size` (755,617,140,416 B). Only the VALUES differ, which matches
  the model card: same base model, gains from post-training.

So GLM-5.3 required **zero devgen changes**. Do not confuse it with **GLM-5.3-Flash**, which is
`model_type: glm5_next` (KDA + hyper-connections) and is a genuinely different emit path
(`mla::glm53_emit_full`).

## 2. Precision: the checkpoint is block-FP8, and the served blob keeps it

The weights are native **e4m3 with a `[128,128]` `weight_scale_inv` grid**
(`quantization_config.weight_block_size = [128,128]`, `activation_scheme: dynamic`).
`"dtype": "bfloat16"` in config.json is the COMPUTE dtype, not storage.

| stored as | tensors |
|---|---|
| F8_E4M3 + F32 scale grid | q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj, dense MLP (layers 0-2), all 256 routed experts x 78 layers, shared expert, DSA indexer wq_b/wk |
| BF16 | embed_tokens, lm_head, every layernorm, router `mlp.gate.weight`, indexer k_norm(+bias), indexer weights_proj |
| F32 | the scale grids, `mlp.gate.e_score_correction_bias` |

The emitted blob's manifest confirms FP8 survives compilation:

```json
"precision": {"weight_enc":"fp8","act_enc":"bf16","kv_enc":"bf16","expert_enc":"fp8blk"}
```

`act_enc: bf16` is not an omission — this is W8A16. MLA+MoE has no W8A8 profile in this tree
(`docs/flags-reference.md`), so activations stay bf16 by construction.

The host prep (`scripts/glm52_prep_lite.py`, reused unmodified because the names are identical)
dequantizes only what the MLA absorption needs — `derived.q_absorb`/`v_absorb`/`kv_a_latent`,
plus `q_a_proj`, `o_proj` and the shared expert — and symlinks the 141 raw shards so the routed
experts and dense MLP resolve **verbatim as block-fp8**. 40 GB of new data instead of a 700 GB
copy. `GLM_LINEAR_FP8=1` would additionally keep `o_proj` + shared expert in fp8; it needs
`glm52_prep_fp8_linear.py`'s `.weight_fp8` shards and is NOT in the numbers below.

## 3. Capacity on a 192 GiB card

Prepped checkpoint ~714 GiB.

| | GiB/rank, weights | verdict |
|---|---:|---|
| TP2 | 357.1 | impossible — 165 GiB over the card, weights alone |
| TP4 | 178.6 | fits; measured 181.75 GiB/rank live |
| TP8 | 89.3 | comfortable |

## 4. Reproducing

```bash
# prep (no GPU; nix's torch is broken here, so use the qualified vLLM interpreter)
VLLM_ROCM_LIB=/opt/rocm/core-7.14/lib build-gemma31/vllm-python \
  scripts/glm52_prep_lite.py --model /workspace/models/GLM-5.3-FP8 \
                             --out /workspace/models/GLM-5.3-plow-lite

# objects (no GPU)
nix develop --command bash -c 'PLOW_DECODE_BATCH=4 JOBS=24 \
  bash scripts/build_gfx942.sh /app/plow/build-glm53/hsaco'

# emit, serve, smoke, bench — every GPU process takes a gpulease
MAXCTX=10240 scripts/glm53_mi300x.sh emit 4
MAXCTX=10240 scripts/glm53_mi300x.sh serve 4 18930
scripts/glm53_mi300x.sh smoke 18930
IN_LENS="1024 4096" CONCS="1 4" scripts/glm53_mi300x.sh bench 4 18930 plow
```

Load: 4 ranks bound in 30.5 s, 181.75 GiB/rank at ~6.0 GiB/s. Engine banner
`gfx942 n_cu=304 batch=4 decode_rungs=[1,2,4] max_ctx=10240`.
Coherence: "The capital of France is Paris."; a 214-token free-form answer was accurate and
drift-free.

### Two defects this bringup found

1. **plowrt outside `nix develop` silently serves from the CPU.** The flake's ROCm 7.14 tree is
   what carries `libhsa-runtime64.so.1`; outside it the dlopen fails and plowrt logs
   `CPU reference backend active` and then answers correctly at fictional speed. The serve now
   runs inside nix (`scripts/glm53_serve_inner.sh`).
2. **`build_gfx942.sh` armed `PLOW_MOE_PF_DET` on prefill objects only.** Above
   `PLOW_DECODE_BATCH=1` the GLM decode program emits its MoE seam with the grouped PREFILL
   family, so op 86/87's deterministic arm is reached from the DECODE object. The load was
   refused with a packet/object mismatch. Fixed in the `PLOW_DECODE_BATCH>1` branch.

## 5. TP4 baseline — plow only, and NOT yet a comparison

128 output tokens, 8 prompts per cell, random dataset, prefix caching off, greedy
(the AMD engine samples argmax on device and ignores temperature).

| input | conc | output tok/s | median TTFT ms | median TPOT ms | median E2EL ms |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 23.78 | 383 | 38.70 | 5308 |
| 1024 | 4 | 38.93 | 840 | 95.06 | 13086 |
| 4096 | 1 | 18.31 | 1024 | 47.02 | 6992 |
| 4096 | 4 | 29.48 | 1698 | 122.23 | 17171 |

**Every one of these numbers carries three caveats, and none of them is cosmetic.**

1. **No vLLM arm yet.** All eight cards were leased throughout by a concurrent agent session
   running an unrelated Gemma-4-31B campaign, so the matched vLLM 0.28 reference — which needs
   four of them — has not been taken. Nothing here says anything about beating vLLM.
2. **The tuning store is fully stale.** 868 records, zero usable against build
   `gfx942-f390fbaa86c582ff`; both blobs report `tile_source: analytical`,
   `tile_measured: 0`. Prefill GEMM tiles come from the analytical model, so **the TTFT column
   is unmeasured-tile** and is the first thing to re-derive
   (`plowc tune gemm --obj build-glm53/hsaco`, which needs a quiet card and no live plowrt).
3. **Measured under contention.** Four to six foreign GPU processes were live on the other
   cards. TP4 collectives cross the fabric those processes also touch. gpulease's own rule is
   that a contended run invalidates timings. Re-measure on a quiet box.

**Known asymmetry for whenever the vLLM arm is taken:** vLLM serves this checkpoint with DSA
armed (`index_topk` 2048) while this blob is emitted `PLOW_GLM_DSA=0` and reads EVERY KV row.
Above ~2k context plow is doing strictly more attention work per token. A plow win under that
asymmetry is a real win; a plow loss is not by itself evidence about plow's kernels.

## 6. Open levers, in the order worth taking

1. **Measured GEMM tiles** (§5 caveat 2). Cheapest, and it gates the ranking of everything else.
2. **`PLOW_DEC_SQUEEZE=1` decode tier.** The validated WPE=3 register recut; on GLM-5.2 it was
   rung-1 TPOT −12.3%. Built here into `build-glm53/hsaco-squeeze` (its K3 rows fail to
   compile, which is fine — GLM loads `interp_decode_gq.elf`) and served via
   `PLOW_HSACO_LOWRUNG=<dir>:4`. A/B not yet taken.
3. **`GLM_LINEAR_FP8=1`.** Decode is HBM-bound — the emit's own lean oracle puts the TP4 decode
   step's floor at 3.0 ms against 16.0 GB touched per step — and this keeps ~5 GB/rank/token of
   `o_proj` + shared expert in fp8 instead of bf16.
4. **TP8**, once eight cards are free.
