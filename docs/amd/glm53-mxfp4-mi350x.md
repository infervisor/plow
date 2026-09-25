# GLM-5.3 MXFP4 on 8x MI350X (gfx950) — TP8 kernel campaign (2026-09-24/25)

Goal: plow-native kernels for every GLM-5.3 op at the vLLM 0.29 dtype contract of the Quark
checkpoint (`/opt/models/GLM-5.3-MXFP4-AttnFP8-amd-4992911b`), prefill rungs 1k–16k and decode
rungs 1–128, TP8, priced against a per-op roofline. Every GPU job ran through `gpuq`/`gpulease`.

## Dtype contract (vLLM 0.29, compilation mode NONE)

| op | vLLM | plow (this recipe) |
|---|---|---|
| q_a/kv_a (fused), q_b, kv_b, o_proj | block-FP8 W8A8 (128x128 W, 1x128 dynamic A) | `GemmFp8Block128` (+ `QuantFp8Block128`, or fused into RmsNorm) |
| MLA prefill attention | expanded (kv_b GEMM + MHA flash), BF16 | expanded: kv_b GEMM + `d_mla_mha_pf` (`PLOW_GLM_MLA_MHA`) |
| MLA decode attention | absorbed, BF16 KV | absorbed `FlashMlaDecode` + merge, BF16 KV |
| routed + shared experts, dense MLP | MXFP4 A4W4 (1x32 E8M0) | A4W4 grouped (shared folded as expert 257 at prefill and batched decode) |
| router | BF16 in, FP32 out | `GemmF32` |
| all-reduce | BF16 | XReduceScatter/XAllGather (seq-par seams), BF16 |
| DSA sparse attention / FP8 indexer | on for T > 2048 | **off (dense)** — open gap |

## Recipe (best: packet `parity-sf-sp-v9`, objects `…-v9-hsaco-final`)

Checked in as `scripts/glm53_mxfp4_mi350x.sh overlay|emit|objects|assets|validate` (the env below).
Inputs outside the repo: Quark checkpoint, prepped dir from `scripts/glm53_prep_quark.py`, FP8-original
checkpoint + its derived MLA TP8 dir (`scripts/glm53_mxfp4_overlay.py` byte-checks kv_b/q_b against them).
Re-check 2026-09-25: `emit` reproduces v9 byte-identical (sha16 3b8b96fd6339691d); `overlay` reproduces
every sidecar byte-identical and every symlink target; `objects` reproduces every interpreter .co
byte-identical and every lean ELF with identical disassembly (a metadata note differs).

Emit (env):
`PLOW_GLM_QKVA_W8A8=1 PLOW_GLM_OPROJ_W8A8=1 PLOW_GLM_MLA_W8A8=1 PLOW_GLM_MLA_MHA=1
PLOW_GLM_MOE_SHARED_FOLD=1 PLOW_GLM_DECODE_SHARED_FOLD=1 PLOW_GLM_NORM_Q128=1 PLOW_GLM_QUANT_NARROW=1 PLOW_GLM_SEQ_PAR=1
PLOW_GLM_SEQ_PAR_PROJ=1 GLM_GROUP=1 PLOW_GLM_FUSE_B1=1 GLM_FUSE_XRN=1 PLOW_GLM_FUSE_ROPE=1
PLOW_GLM_DECODE_NORM_ROWS=1 GLM_SPINE_CUS=64 PLOW_GLM_DECODE_GLUE_CUS=1`
(`PLOW_GLM_FUSE_SEAM` is illegal with QKVA W8A8.)

Objects (`scripts/build_gfx950.sh`, env): `PLOW_MXFP4=1 PLOW_MOE_PREFILL=1 PLOW_MOE_PF_A4W4=1
PLOW_MLA_PF_TR16=1 PLOW_DECODE_BATCH=128 PLOW_GEMV_MM=8 PLOW_GEMV_WALK=1 PLOW_MOE_PF_DOWN_SWEEP=1
PLOW_MLA_FOLD_MFMA=1 PLOW_MOE_PF_A4W4_BK=256 PLOW_COMBINE_VEC=1 PLOW_MOE_ROUTER_PF_WAVE=1
PLOW_MLA_MHA=1 PLOW_GEMV_MFMA4=1 PLOW_XR_SCHED=aiter PLOW_XR_SCHED_NWG=40 PLOW_XR_SCHED_NWG_SRS=16
PLOW_FP8_BLK_DMA=1 PLOW_FP8_BLK_KW=1 PLOW_MERGE_UNROLL4=1 PLOW_GEMV_F32_COL=1
PLOW_MOE_STAGE1_PIPE=1 PLOW_MOE_GLU_KW=1 PLOW_MOE_DOWN_SWEEP_LINE=1 PLOW_MOE_ALIGN_WAVES=1`

Runtime: plowrt from this branch (MHA segment routing, MoE stage-1 token-gather route, ragged
QuantFp8Block128 row shrink); checkpoint overlay `/opt/models/plow-glm53-mxfp4-prepped-mla-mha-20260925`
(adds `kv_b_proj.weight_scale_inv`); `PLOW_PREFIX_CACHE=0` (the MHA arm traps on kv_len != rows,
i.e. prefix-cache hits / continuation chunks); `PLOW_AMD_DECODE_MIN_RUNG=1` for decode below 8 rows.

## Kernels by op and dimension (standalone, 1 GPU, TP8 per-rank shapes)

| kernel (knob) | op | before -> after (ms) | numerics |
|---|---|---|---|
| `d_mla_mha_pf` (MLA_MHA) | MLA prefill attention, 8 heads causal | 1k 0.127->0.063, 2k 0.251->0.107, 4k 0.848->0.197, 8k 3.07->0.42 (653 TF/s), 16k 11.75->1.70 | rel-L2 2e-3 vs reference MHA |
| stage-1 token-gather pipe (MOE_STAGE1_PIPE) | routed gate/up A4W4 prefill | 1k 0.356->0.199, 8k 1.176->0.477, 16k 2.227->0.844 | bit-exact |
| DOWN sweep (MOE_PF_DOWN_SWEEP) | routed down A4W4 | 2.3–5.5x, T1..16k | bit-exact |
| combine vector (COMBINE_VEC) | MoE combine | 4.4x | bit-exact |
| router wave (MOE_ROUTER_PF_WAVE) | top-k prefill | 8k 0.205->0.068, 16k 0.401->0.131 | tables identical |
| flat merge (packet i6) | FlashMerge nsplit=1 | 8k 0.190->0.042 | bit-exact |
| FP8 GEMM DMA (FP8_BLK_DMA) | block-FP8 prefill GEMMs 128x256 | qkv_a 8k 0.365->0.307, o_proj 0.292->0.244, q_b 0.114->0.097, kv_b 0.086->0.072 | exact (BM=128 only) |
| K-split decode GEMM (FP8_BLK_KW) | qkv_a decode | M1 0.043->0.010, M8 0.047->0.011, M64 0.069->0.031 | bit-exact M1/M8 |
| decode GLU kw (MOE_GLU_KW) | routed gate/up A4W4 decode | T1-4 0.088->0.057, T8 0.089->0.085 (random routing) | bit-exact |
| router col GEMV (GEMV_F32_COL=1) | router decode M1 | 0.015->0.007 | rel-L2 1e-7 |
| RmsNorm+Q128 (NORM_Q128) | norm + block-128 quant | one packet fewer per layer (decode input norm, prefill q_a norm) | same formula |
| xr sched (XR_SCHED=aiter, 40/16) | seq-par collectives | in-model -7.5% prefill | bit-exact |
| DOWN line stores (MOE_DOWN_SWEEP_LINE) | MoE DOWN part writes (128 B non-temporal lines) | 8k 0.73->0.48, 16k 1.33->0.84 | bit-exact |
| align all waves (MOE_ALIGN_WAVES) | MoeAlignPf phases 2/4 | phase 4 8k 103->45 us, 16k 201->70 us | bit-exact |

## In-model TP8 (warm medians, `amd-bench --prefill-sweep … --prefill-reps 5`, 2 alternating rounds)

| packet/objects | 1k | 2k | 4k | 8k | 16k (ms) |
|---|---|---|---|---|---|
| start (parity-sf-sp v1/v5) | 209 | 236–254 | 403 | 835 | 2048 |
| + flat merge, router wave (v2/v7) | 205 | 230 | 390 | 786–800 | 1972 |
| + MLA MHA form (v5/v8) | 172 | 208 | 308–323 | 538 | 1043–1053 |
| + XR sched 40 + GEMM DMA | 159 | ~194 | 280 | 483 | 929–942 |
| + MoE stage-1 pipe (best3) | 144 | 172 | 246 | 405–422 | 811–823 |
| + decode fold, kw GLU, NORM_Q128 (v8b/best8) | 142.7–144.1 | 170.8–170.9 | 245.9–246.5 | 421.1–421.7 | 823–824 |
| + QUANT_NARROW, DOWN line stores, align waves (v9/final) | 140.2–140.6 | 165.1–165.5 | 231.9–232.0 | 387.8–388.3 | 735.7–736.1 |

Decode (8 identical prompts, ctx 1k; v8b/best8, ms/step, 2 rounds): M1 41.3–45.9 ms/token,
M2 51.6–73.5, M4 51.1–54.0, M8 53.6–56.6, M16 61.1–64.6, M32 85.1–85.3, M64 120.2–120.4, M128 181.4
(start of campaign: M1 ~50–87, M8 87–103, M64 205). Batched M8 at ctx 8k: 58.7 ms/step.
v9 (+QUANT_NARROW) same-run vs v8b: M2 50.6 vs 55.6–57.0, M8 53.5 vs 56.2–56.4, M32 80.2–82.6 vs
82.6–85.0, M64 117.8 vs 120.5, M128 177–179 vs 181; prefill unchanged.
v9/final (same run as best8): M1 37.7 ms/token, M2 49.2, M4 49.5, M8 52.2, M16 63.1, M32 83.5, M64 115.9,
M128 177.9; greedy T1024/T8192 and oracle logits identical to best8.

Decode determinism: the q_b live-split and o_proj split8 selectors (kept for AITER-CK rounding parity)
accumulate with bf16 atomics, so near-tie greedy tokens can differ run to run (seen on a ragged
100-token prompt); exact-bucket T1024/T8192 greedy lists are stable across every arm. Levers: comboA decode knobs + GEMV_MFMA4 (-21% M64), K-split qkv_a GEMM +
merge unroll (-9%), decode shared fold (-6% M8, -16% M64), kw decode GLU (-5..-16%).

## vs vLLM 0.29 (same `vllm bench serve` client, random in8064/out128, temperature 0)

| | TTFT mean | TPOT mean | total tok/s |
|---|---|---|---|
| vLLM c8 | 15.57 s | 34.9 ms | 3268 |
| vLLM c1 (in8064) | 13.24 s | 19.5 ms | 521 |
| plow v8b/best8 c8 | 2.07 s | 75.1 ms | 5551 |
| plow v9/final c8 | 1.96 s | 76.4 ms | 5528 |
| plow v9/final c1 (in8064) | 0.69 s | 43.8 ms | 1310 |
| plow v8b/best8 c1 (in8064) | 0.43 s | 43.0 ms | 1392 |
| plow v8b/best8 c1 (in1024) | 0.32 s | 45.8 ms | 188 |

Logits vs vLLM oracle (last prompt position, T1024–T8192): top-1 match 4/4, KL(vLLM||plow)
8.5e-5 / 6e-6 / 2.7e-7 / 2.8e-8 (MHA form). Smoke "capital of France" -> " Paris".

## Negatives (do not re-try blind)
- MoE stage-1 wide tile (128 rows, A+B via LDS): bit-exact but 423 -> 501 us at T8192.
  DOWN global loads / plain float4 stores / 256 B lines / 9-slot combine loads: no gain or slower.
- Router small-N GEMM prefill: 0.6x. GLU full-K sweep prefill: slower. XR sched 24/8 (gfx942
  defaults): no gain on gfx950. `PLOW_MLA_DEC_MINPER=64`: decode M32..M128 slower. Router col GEMV at
  M<=16: M16 slower in-model. FP8 GEMM DMA on 64x128 tiles: mismatch (cause not found; excluded).
- Ragged W8A8 prefill was broken on every packet until the `QuantFp8Block128` row shrink (plowrt
  kvrow) — exact-bucket amd-bench runs could not see it.

## Open
- Decode TPOT 2.4x behind vLLM at c8: per-op floor (~20 us/op x ~25 ops/layer); routed decode GLU
  still ~7% of roof; decode MLA BMMs are wave-per-tile full-K.
- DSA sparse attention + FP8 indexer (vLLM on for T > 2048) not in this packet.
- FP8 prefill GEMM ~2x behind ck_tile; MoE DOWN/combine f32 partial round trip.
