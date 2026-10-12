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
- 11:29 UTC: download complete (328 GB, 62 shards); `glm53_prep.py` wrote the 5.23 GB derived
  sidecar in under a minute; packet v2 from the prepped dir (8 s, `plow_config.h` identical to the
  stub packet's), objects 93 s, assets assembled; doctor: 1 FAIL (its vLLM client check wants
  `/app/.../vllm-python`; this box's client is the Docker wrapper) + 2 WARN (shared plowrt binary,
  no lean-qualification.json). Smoke queued (gpuq) behind the vLLM ISL-1024 grid.
- 11:35 UTC: vLLM TP8 server up after ~25 min of AITER JIT; first cell g1 (ISL 1024, c1) running
  at ~11 s per 128-token request.
- CPU roofline of packet v2 (`scripts/campaign/packet_roofline.py` / `op_roofline.py`, MI350X
  ceilings HBM 6200 GB/s, bf16 2300 / fp8 4600 TFLOPS, xGMI 1075 GB/s): see §6.
- 11:37 UTC: vLLM cell g1 (ISL 1024 / OSL 128, c1, TP8, greedy): TPOT 83.9 ms, TTFT 230 ms,
  11.8 tok/s. The server log shows the vendor path is in use (AITER 2-stage fused MoE per-1x128
  fp8 with the shared expert fused as expert 289, AITER mHC tilelang kernels, custom all-reduce,
  full CUDA graphs); `torch.compile` is refused for this model; no MTP (the comparison is
  single-token on both sides). This is the bar for c1, not the vendor's published 14 ms.
- Decode op-count experiments (CPU emits, decode program ops): base 1846; `PLOW_KDA_DECODE_FUSED[_ARM]`
  1778 (-68 = the 34 KDA chains fused); `PLOW_GLM_FUSE_B1+SEAM` 1835; `+GLM_FUSE_XRN` 1835 with 11
  XReduce folded; `PLOW_KDA_FB_FOLD` 1812; flat decode / gemm group / shard head: no change;
  `PLOW_GLM_ROUTED_W8A8` refused (needs `GLM_LINEAR_FP8`, and the MoeGluFp8Block128 kernel had no
  clamp) -> clamp added to the block128 GLU epilogue (`glu_limit` = f[1]) and the routed-W8A8
  packet (3 MoE ops/layer instead of 16) is being built.
- 11:40 UTC: vLLM ISL-1024 / OSL-128 grid complete (TP8, greedy, 2 repeats). The first c1 repeat
  (TPOT 83.9 ms) was AITER JIT warm-up for the M=1 shapes; the second is the number:

  | conc | TTFT mean ms | TPOT mean ms | out tok/s |
  |---:|---:|---:|---:|
  | 1 (r2) | 90 | 6.87 | 133 |
  | 8 | 209 | 9.2 | 741 |
  | 32 | 481 | 12.8 | 1974 |
  | 64 | 638 | 16.6 | 2978 |
  | 128 | 901 | 25.9 | 3900 |

  The grid is re-run with the JIT cache warm before it is used in a report (both repeats must
  agree for the strict renderer).
- 11:41 UTC: first plow smoke refused at load: the loader resolves this packet's prefill object
  as `interp_prefill_k3_moe_gq.elf` and its decode object as `interp_decode_k3(_gq).elf`
  (`exec/amd/object.rs` PrefillArm), names only `scripts/build_gfx942.sh` produced;
  `build_gfx950.sh` now builds them when the packet's `PLOW_PACKET_OBJECT_REQUIRES` carries
  `PLOW_K3=1` (same flags as its MLA+MoE prefill and decode rows, plus `-DPLOW_K3=1`).
- 11:49 UTC: second smoke refused at load: the flash object lacked the zero-rope V2 MLA prefill
  arm (`PLOW_MLA_PF2_NOPE_ARM=1` in the packet's requires; gfx942 spells it `PLOW_MLA_PF_NOPE=1`,
  gfx950 never set it). `build_gfx950.sh` now adds `-DPLOW_MLA_PF2_NOPE_ARM=1` to the flash
  objects when the packet requires it. Lesson: the gfx950 script keys only a few requirement
  flags off `PLOW_PACKET_OBJECT_REQUIRES`; each unmapped one costs a GPU round trip.
- (to be continued: smoke, correctness gate, grid.)

## 5. Results

(pending: the strict 12-row `campaign.py report` tables go here with the comparison.csv link.)

## 6. Roofline (CPU, before any plow GPU time)

Decode program (T=1, per rank, TP8), weight-stream floor from `packet_roofline.analyze`:

| ctx | logical operand bytes / rank-step | floor @ 6.2 TB/s | ops / step |
|---|---:|---:|---:|
| 1024 | 4.52 GB | 0.73 ms | 1854 |
| 8192 | 4.52 GB (+KV, excluded) | 0.73 ms | 1854 |

Where the bytes go: `lm_head` bf16 **replicated on every rank** 1.27 GB (28%); KDA bf16
projections (`GemvQkv` 34x + gates) 1.18 GB (26%); routed experts fp8 (8 of 288 per layer, 42
layers) 1.06 GB (23%); MLA fp8 projections ~0.1 GB; everything else < 5%. The excluded ops
(90 `XReduce`, 90 `HyperConnPre`/`Post`, 90 `GemvF32`, 34x3 KDA state ops, norms) are
latency-bound, not byte-bound: the gfx950 GLM-5.3 campaign measured ~20 us per op in the
persistent interpreter, so 1854 ops/step puts the **dispatch floor near 37 ms/token**, far above
the 0.73 ms byte floor and above vLLM's TPOT. Op count, not bandwidth, is the decode roofline.

Prefill (per rank, bucket T): `op_roofline.py` floor 2048 -> 10.8 ms, 8192 -> 34.7 ms; the 87
`XReduceTwoShot` collectives are 22-27% of the floor (fabric-bound, 67 MB each at T=8192), then
the grouped MoE GLU/down (priced at the MXFP4 contract; the packet streams fp8, so x2) and
`MoeCombinePf`. Levers, in roofline order: decode op-count fusions (`PLOW_KDA_DECODE_FUSED[_ARM]`,
`PLOW_GLM_FUSE_B1/SEAM`, `GLM_FUSE_XRN`, `PLOW_GLM_ROUTED_W8A8` = 3 ops/layer instead of 16,
`PLOW_GLM_MOE_FLAT_DECODE`, `GLM_SHARD_HEAD`), then prefill seq-par collectives
(`PLOW_GLM_SEQ_PAR`). Each is an op-count experiment first (CPU emit + disasm), then a smoke, then
a cell. Measured per-op times replace the 20 us assumption once the smoke serves.
