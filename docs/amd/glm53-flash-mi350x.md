# GLM-5.3-Flash on 8x MI350X (gfx950) — bring-up and serving campaign (2026-10-12)

Recipe: `recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-16k-fp8.toml`. Campaign plan and
research notes: `plans/glm53-flash-dsv41-mi350x.md` (local). Baseline to beat: vLLM on the
`vllm/vllm-openai-rocm` nightly image (AITER, TP4/TP8), measured on this box with the same client.

## 1. What GLM-5.3-Flash is, for the emitter

`zai-org/GLM-5.3-Flash`, `model_type: glm5_next` (text tower `glm5_next_text` under `text_config`,
tensors under `model.language_model.`), 320B total / 18B active, 45 layers: 34 KDA linear-attention
layers (Kimi-Linear tensor names: `q/k/v_proj`, `*_conv1d`, `f_a/f_b_proj`, `g_a/g_b_proj`, `b_proj`,
`A_log`, `dt_bias`, `o_norm`) and 11 NoPE sparse-MLA layers at 3, 7, ..., 43 (`q_a/q_b`, `kv_a_proj_with_mqa`,
`kv_b` bf16, `o_proj`, kpool-4 indexer with `index_kpool_compress_{ape,gate}`, top-2048 tokens);
mHC residuals (4 streams, 20 Sinkhorn iterations, eps 1e-6, mean exit); 3 dense FFN layers then
288-expert / top-8 / 1-shared MoE with sigmoid noaux_tc routing, scale 2.5, **clamped SwiGLU
(`swiglu_limit` 10)**; vocab 154880; block-FP8 e4m3 [128,128] projections and experts, bf16 KDA
projections, with `quantization_config` at the ROOT of the wrapper config. A vision tower and an
MTP layer (`layers.45.*`) ship in the checkpoint and are ignored. It is not GLM-5.3 (`glm_moe_dsa`,
78 layers, no mHC): `docs/amd/glm53-mi300x.md` §1.

## 2. Emitter state and what this campaign changed

`mla::glm53_emit_full` (commit ba857841, 2026-09-02) emits the full model; the last bring-up
(98a369ed, 2026-09-04) reached a correct first token at TP4/TP8 on gfx950 with decode degenerating
afterwards. Changes made here before the first serve on this box:

| change | where | why |
|---|---|---|
| clamped SwiGLU on every GLU (`ACT_SWIGLU_CLAMP`, `f1` = limit) when the config carries `swiglu_limit` | `crates/devgen/src/mla.rs` (`glm_glu_act`, 13 pair-form sites; the 7 gate-only bf16/MXFP4 GEMM/GEMV GLU sites refuse), `crates/packet/src/dev.rs` | every FFN of every layer differed from the reference in the tail of `up`; the kernel arm existed for DeepSeek-V4 but no emitter set it |
| `quantization_config` read from the root when `text_config` has none | `mla_ckpt_enc` | the Flash config keeps it at the root, so the emit chose **bf16** expert ops (`MoeExpertGlu`, `GemvGlu`) for a block-FP8 checkpoint |
| `index_kpool` capability probe also reads `text_config` | `crates/devgen/src/lib.rs` | the `indexer_pooled` knob cap never registered for the nested config |
| mHC constants and pooled-indexer flags asserted against the config | `cfg_glm` | the emit hard-codes n=4 / 20 iterations / 1e-6 and always compresses + appends the tail |
| KDA carried state sized to the decode ladder (`dbatch` slots), as Kimi-K3 does | `declare_glm53` | one slot made every batched decode row share one recurrence |
| bf16 shared expert GLU split into two GEMM/GEMV halves + the pair-form clamped `Glu` when a limit is set (prefill, batched decode, ladder-1 decode) | shared-expert sites in `mla.rs` | the prep dequantises the shared expert to bf16, whose fused `GemmGlu`/`GemvGlu` epilogue is gate-only and has no clamp arm; `GLM_LINEAR_FP8` (fp8 shared expert) hung on gfx942 in the GLM-5.3 campaign, so it stays a later lever |

Test: `cargo test -p devgen --lib glm5_next_cfg_reads_root_quantization_and_clamped_swiglu`.

## 3. Build chain on this box

```bash
# CPU, inside `nix develop` (worktree build: cargo build --release -p plowc; -p plowrt --features hsa)
nix develop .#quantize --command python3 scripts/glm53_prep.py --model /home/lava/models/GLM-5.3-Flash \
    --out /home/lava/models/plow-glm53-flash-prepped                       # ~40 GB sidecar + symlinks
env PLOW_MLA_PREFILL=full:128,512,2048,8192 PLOW_DECODE_BATCH_LADDER=1 PLOW_GLM_DSA=0 \
    PLOW_VERIFY_BIN=lean-plow/.lake/build/bin/plow_verify \
    target/release/plowc --hf-dir /home/lava/models/plow-glm53-flash-prepped --emit devblob \
    --gpu MI350X --arch gfx950 --num-gpus 8 --n-cu 256 --max-ctx 16384 --out <pkt>
env PLOW_HSACO_CONFIG=<pkt>/plow_config.h PLOW_DECODE_BATCH=1 JOBS=32 scripts/build_gfx950.sh <obj>
# GPU, through the queue
python3 scripts/bench/gpuq.py submit glm53f-smoke 8 <serve + smoke script>
```

The packet from config + index alone (no weights) is 70 MB: 4 prefill buckets + 1 decode program,
7 s to emit; gfx950 objects build in about 1 minute (interp prefill/decode/flash + `_gq` twins).

## 4. Status log

- 10:41 UTC: first packet emitted from the config/index stub; objects built (bf16 expert ops —
  the root-quantization bug above, found by reading `plow_config.h`).
- 11:09 UTC: emitter changes committed (00244f7e) and pushed. Emit from the stub with the fixed
  plowc hit the shared-expert gate-only GLU refusal (fixed by the split above); emit needs
  `PLOW_VERIFY_BIN` (the worktree has no lean-plow build; the main checkout's is used).
- (to be continued: prep, serve, smoke, correctness gate, grid.)

## 5. Results

(pending: the strict 12-row `campaign.py report` tables go here with the comparison.csv link.)
