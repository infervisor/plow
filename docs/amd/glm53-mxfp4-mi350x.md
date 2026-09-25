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
| DSA sparse attention / FP8 indexer | on for T > 2048 | opt-in `GLM_RECIPE=dsa` (BF16 indexer; see DSA section) |

## Recipe (best: packet `parity-sf-sp-v9`, objects `…-v9-hsaco-final`)

Checked in as `scripts/glm53_mxfp4_mi350x.sh overlay|emit|objects|assets|validate` (the env below).
Inputs outside the repo (downloads + checked-in prep): Quark checkpoint
`amd/GLM-5.3-MXFP4-AttnFP8` (rev 4992911b); prepped dir from `scripts/glm53_prep_quark.py`;
FP8-original `zai-org/GLM-5.3-FP8` (aca966e4) + its derived MLA TP8 dir from
`scripts/glm52_prep_fp8_linear.py --mla-tp 8`; `scripts/glm53_mxfp4_overlay.py` byte-checks kv_b/q_b
against them. Full chain on a fresh machine (inside `nix develop`, after `cargo build --release`):

    scripts/glm53_mxfp4_mi350x.sh overlay CKPT            # GLM_RECIPE=dsa adds --dsa sidecars
    scripts/glm53_mxfp4_mi350x.sh emit PKT                # plowc flags + emit env (GLM_RECIPE=dsa)
    scripts/glm53_mxfp4_mi350x.sh objects PKT OBJ         # build_gfx950.sh env -> .co / .elf
    scripts/glm53_mxfp4_mi350x.sh assets PKT CKPT ASSETS
    python3 scripts/bench/gpuq.py submit scripts/glm53_mxfp4_mi350x.sh validate PKT OBJ CKPT ASSETS OUT

plowc flags: `--hf-dir <prepped> --gpu mi350 --arch gfx950 --num-gpus 8 --n-cu 256 --max-ctx 16384
--batch 128 --seq 128,512,1024,2048,4096,8192,16384`. plowrt env: `PLOW_PREFIX_CACHE=0
PLOW_AMD_DECODE_MIN_RUNG=1 PLOW_TP_NO_AUDIT=0 PLOW_TP_AGREE_EVERY=1`.
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

## DSA sparse attention (`GLM_RECIPE=dsa`)

Emit adds `PLOW_GLM_DSA=topk PLOW_GLM_DSA_PF=1 PLOW_GLM_DSA_PF_SPAN=3 PLOW_GLM_FUSE_ROPE=0
PLOW_GLM_DSA_PF_B8=1`; buckets <= 2048 keep MHA (exact dense = top-k identity), >= 4096 run
indexer -> top-2048 -> union per 8-query pack -> `d_mla_sparse_pf` (absorbed form, what vLLM runs).
Objects add `PLOW_MLA_SPARSE=1 PLOW_DSA_SELECT_V2=1 PLOW_DSA_IDX_QPW=1 PLOW_HNR_ILP=1` and pick up
`PLOW_DSA_PF_ARM=1 PLOW_DSA_DECODE_BATCH=1` from the packet config. Overlay `--dsa` (indexer
wq_b/wk `weight_scale_inv`). Same plowrt + run env as dense. Packets: dsa-v1 fbdd65b0 (BF16 interp
chain), dsa-v4 cf2a668a, dsa-v5 3aaa4a53 (= `GLM_RECIPE=dsa` emit; objects dsa-v6 from 5cfb283f).
Status: correct and closer to vLLM than dense, but NOT yet faster than dense prefill at any rung
(dsa-v6 vs dense in the same job: 4k +42, 8k +67..74, 16k +140..149 ms).

| kernel (knob) | op | standalone (ms) | numerics |
|---|---|---|---|
| `d_mla_sparse_pf` (DSA_PF_B8 + MLA_SPARSE) | sparse MLA prefill, 64 rows = 8 q x 8 heads, DMA gather, bf16 latent out (no merge), q-rope folded, largest-first pack tickets | T8192 2.43 (V2 gather) -> 0.695, T16384 6.67 -> 1.753 (~560 TF/s on union pairs) | rel-L2 5e-4 vs double ref over each query's own selection |
| select v2 (DSA_SELECT_V2) | IndexSelectPf | 8k 1.70 -> 0.34, 16k 3.75 -> 0.94 | same set |
| select v3 (DSA_SELECT_V3) | IndexSelectPf, one wave per row | real scores 8k 0.301 -> 0.121, 16k 0.998 -> 0.353 | same set |
| select+union (DSA_SELECT_V3 + emit GLM_DSA_SEL_UNION) | IndexSelectPf writes IndexUnionPf's table | real 8k 0.469 -> 0.111, 16k 1.487 -> 0.301 (v2 + union -> fused) | union bytes identical |
| score QPW (DSA_IDX_QPW) | IndexScorePf (BF16) | 8k 0.875 -> 0.472, 16k 3.39 -> 1.77 | byte-identical |
| HNR ILP (HNR_ILP) | indexer q/k rope HD=128 | 8k 0.100 -> 0.060 | byte-identical |
| score FP8 (DSA_IDX_FP8, opt-in) | IndexScorePf via `dsa_score_fp8.h`, vLLM's FP8 recipe | vs QPW: 4k 0.148 -> 0.072, 8k 0.49 -> 0.24, 16k 1.71 -> 1.01 | bit-identical to vLLM's ops (T4096/8192) |

In-model TP8 prefill (ms, warm medians; dense v9 = 232 / 388 / 736 at 4k / 8k / 16k):

| build | 4k | 8k | 16k |
|---|---|---|---|
| dsa-v1 (V2 gather, BF16 interp indexer) | 340 | 632 | 1289 |
| dsa-v2 (+ sparse flash) | 292 | 503-506 | 992-1001 |
| dsa-v3 (+ select v2, score QPW) | 291 | 486 | 910 |
| dsa-v4 (+ q-rope fold) | 279 | 470 | 912 |
| dsa-v6 (+ pack tickets, HNR ILP; = current preset) | 274 | 463 | 884 |

Numerics (vLLM runs DSA): top-1 == vLLM oracle 4/4 on every build; T8192 logit cos dense 0.957,
dsa1 0.977, dsa2 0.980, dsa4 0.984 (top-10 overlap 9/10). Greedy T1024/T8192 identical to dense.
dsa-v4 T8192 trace (ms): sparse flash 82 (dense MHA 39), score 10, select 9, MlaBmm 11+8 (dense kv_b
8.5), indexer ropes 8, union 4, LayerNorm 4, wq_b 4, projections 5. T16384: flash 201 (MHA 153),
score 34, select 30, MlaBmm 19+15, union 11. In-model flash is ~1.5x its standalone time.
dsa-v4 detail (job ab-dsa4, packet cf2a668a = `GLM_RECIPE=dsa` emit before the ticket counter;
objects `PLOW_MLA_SPARSE=1 PLOW_DSA_SELECT_V2=1 PLOW_DSA_IDX_QPW=1`, plowrt-v5, overlay `--dsa`):

| | 4k | 8k | 16k (ms) |
|---|---|---|---|
| dsa-v4 sweep median | 279.2 | 469.6 | 911.8 |
| dense v9, same job | 231.5 | 388.5 | 736.1 |
| delta | +47.7 | +81.1 | +175.7 |

| logits vs vLLM oracle | T1024 | T2048 | T4096 | T8192 |
|---|---|---|---|---|
| top-1 | match | match | match | match |
| cos | 0.959 | 0.733 | 0.955 | 0.984 |
| KL(vLLM‖plow) | 8.5e-5 | 6.0e-6 | 6.9e-7 | 2.9e-8 |
| top-10 overlap | 6/10 | 5/10 | 8/10 | 9/10 |

Greedy 16 steps: T1024 [2615, 5383, 24417, 11, …], T8192 [16539, 264, 16148, 429, …] — identical to
dense. (T1024/T2048 rows are dense MHA buckets in every build, hence identical to dense.)
Decode (dsa-v1): ctx1k M1/M8/M64 51.1/67.5/220.7 vs dense 38.4/53.5/117.4 ms — not yet worked on.
The "16k last-8-rows" zeros were not a bug: prompt-16384.ids holds 16376 tokens (kvrow shrink).

Selection structure (real iidx_pf dump, 118-distinct-token prompt): union of B adjacent queries' top-2048
vs dense causal pairs — 8k: B1 0.43x, B2 0.49, B4 0.57, B8 0.66, B64 0.91, B128 0.99; 16k: 0.23, 0.28,
0.34, 0.42, 0.76, 0.93. 64-key block skipping saves nothing (0.96-1.00x): selections are scattered.
So only the absorbed per-pack gather cuts work; absorbed costs 2.125x the MACs/pair of the MHA form, so
at 8k ideal sparse ~= dense MHA and the win must come from kernel efficiency; 16k has ~2x headroom.
Kernel probes (T8192 standalone): no K gather 0.585 ms, no QK MFMA 0.695, no PV MFMA 0.687 vs 0.729 —
latency-chain bound at 1 wave/SIMD, not DMA- or MFMA-bound. Negatives: two QK accumulators (0.748),
optimistic 2-barrier softmax (0.86); FP8 score arm (failed its CPU reference of vLLM's recipe, not faster).
Open levers: FP8 indexer (vLLM parity + ~2x score), select/union fusion, in-model flash tail (1.5x
standalone), MlaBmm (13% of roof), fewer/lighter chunks (B=4 packs need a smaller LDS stage).
Half-pack split (keys only queries 0-3 or only 4-7 selected: 27% of union keys at 8k, 39% at 16k)
would cut B8 flash work to 0.86x / 0.80x. dsa-v6 in-model flash: 0.82-0.85 ms/layer at 8k (WG bodies
0.68-0.82 after tickets) vs dense MHA 0.50.
Break-even budget (attention-related ms, dense = MHA flash + kv_b + quant + k-rope): 8k dense ~55 vs
dsa-v6 ~120 (flash ~65, indexer ~42, absorbed MlaBmm ~19); 16k dense ~176 vs ~260. Beating dense needs
roughly 2x on the sparse flash AND 3-4x on the indexer chain; with the absorbed form's 2.125x
MACs/pair, the crossover against plow's MHA prefill is expected beyond 16k.

Crossover probe at max-ctx 32768 (`GLM_MAX_CTX=32768 GLM_SEQ=...,32768 GLM_BATCH=64`,
`PLOW_DECODE_BATCH_LADDER=1..64`, objects `PLOW_DECODE_BATCH=64`; 128 x 32k KV does not fit), 2 rounds:

| prefill (ms) | 8k | 16k | 32k |
|---|---|---|---|
| DSA (preset) | 462-464 | 882 | 1892-1896 |
| dense | 392-396 | 735-742 | 1674-1678 |
| DSA / dense | 1.18 | 1.20 | 1.13 |

16k -> 32k growth: dense +939 ms, DSA +1011 ms; subtracting the shared non-attention part (~+559),
attention grew dense ~176 -> ~556 (the MHA kernel gets more efficient at long T) vs DSA ~260 -> ~712
(indexer score/select/union are O(T^2) and grew ~4x; union waste grows with T). With the current BF16
indexer DSA is not expected to cross over by 64k; the indexer chain is the binding term at long context.
A 3-4x faster indexer (FP8 score per vLLM, fused/faster select) would put the crossover at ~24-32k.
32k correctness not checked (no oracle at 32k).

### Select v3 and fused select + union (plan steps 2-3, 2026-09-25)
`runtime/amd/dsa_select_v3.h`, test `runtime/tests/index_select_pf_v3_gfx950.hip` (1 GPU, 256 WGs).
- One WAVE per row, no workgroup barrier inside a row. v2 ranks a row with the whole workgroup and
  pays ~15 barrier / LDS round trips per row regardless of length (measured ~20k cycles/row fixed);
  here 8 waves rank 8 rows (fused: the 8 rows of one query pack) with wave-local LDS (15 KB each).
- Pass 1 streams the row (buffer loads, 4 x 16 B per lane in flight) into an 11-bit histogram of
  the fp32 key's top bits; with a previous row on the wave, into 2046 bins of 2^13 keys around that
  row's threshold instead (keys below the window uncounted, above aggregated; miss -> top-digit
  passes, window skipped for the next row). A threshold group > 1792 keys takes more streaming
  digits (never truncated). Pass 2 emits keys above the prefix and compacts the equal ones (<= 1792)
  into LDS; 8-bit digit passes, score bits first and index bits (lowest index) only on an exact tie,
  until the group is needed whole or <= 16 keys, ranked exactly.
- Real GLM rows (act.iscore_pf dumps, last indexer layer): scores ~ -55..-82, and up to 70% of a
  16k row sits in the threshold's 11-bit top-digit bin (37/53 sampled rows > 1792), but <= ~60 in a
  2^13-key bin. Without the window pass v3 was 0.598 ms at 16k real (0.361 on gaussian rows).
- Fused (`PLOW_GLM_DSA_SEL_UNION=1` emit): op 118 carries t3 = union table, i4 = cap, i5 = zero
  the pack tickets; op 119 carries i6 = 1 and is skipped by V3 objects (~20 us/layer gate). Objects
  without the arm ignore both marks and run select + union, so the packet is valid either way.
  Pack mask = one byte per position (LDS, 16384 positions); packs with a causal bound past that
  select into iidx_pf and build the union from it in 16384-position windows. iidx_pf is not
  written otherwise (under B8 only the union is read).

| standalone (us/layer) | v2 | v3 | v2 + union | fused |
|---|---|---|---|---|
| real T8192 | 301 | 121 | 469 | 111 |
| real T16376 | 998 | 353 | 1487 | 301 |
| gaussian T4096 / 8192 / 16384 | 78 / 247 / 760 | 30 / 128 / 337 | 136 / 439 / 1386 | 29 / 125 / 308 |
| tie/overflow mix T16384 | 1297 | 904 | 1874 | 1081 |

Every case: sets identical to v2 on every row and to a CPU exact top-k on sampled rows; union
table + ticket words byte-identical to d_index_union_pf over v2's selection (also offset / ragged
chunks, causal bounds past 16384).

In-model TP8 (job sel3-final; packet dsa-su 5217ad9f = `GLM_RECIPE=dsa` + `PLOW_GLM_DSA_SEL_UNION=1`,
objects + `PLOW_DSA_SELECT_V3=1` built before the window pass, i.e. real-score v3 0.598 / fused
0.641 ms at 16k; plowrt-v5; greedy T1024/T8192 identical, logits T1024..T8192 byte-identical to dsa6):

| prefill (ms) | 4k | 8k | 16k |
|---|---|---|---|
| dsa-v6 baseline r1 / r2 | 257.3 / 274.6 | 461.4 / (608.9 noisy) | 882.1 / 883.5 |
| fused select+union r1 / r2 | 271.1 / 271.0 | 453.3 / 453.9 | 858.3 / 859.3 |
| select v3 object only (dsa-v5 packet), 1 run | 256.2 | 458.3 | 853.8 |

4k reps are bimodal (~255 / ~272) in every arm. Trace (fused): IndexSelectPf 4.16 / 14.32 ms and
IndexUnionPf (skipped) 0.40 / 0.42 ms over 21 layers at 8k / 16k, vs 8.91 + 4.07 / 30.11 + 10.76.
The window pass (real-score fused 0.641 -> 0.301 ms at 16k) is not in these objects.

### Indexer: reference implementations and adoption plan (research, 2026-09-25; steps 1-4 implemented, see sections around)
- vLLM 0.29 ROCm path (pinned image): `rocm_fp8_mqa_logits` -> AITER gfx950 gluon
  `_gluon_fp8_mqa_logits_kernel` (1 program/query row, longest rows first, BLOCK_KV=32,
  `mfma_scaled` 32x32x64 e4m3 unscaled, relu -> head reduce -> x kscale, fp32 logits; -inf prefill) +
  `top_k_per_row_prefill` (512-thread block/row, 2048-bin histogram passes 11/11/11/10 bits, final
  bin insertion/CUB sort; ties by atomic order, so vLLM's own selection is nondeterministic on ties).
  Tuned AITER on MI355X: 1.43-1.76 PF/s causal (plow BF16 score: 0.62 PF/s).
- Exact numerics: k after LN + bf16 rope, UE8M0 scale `2^ceil(log2(max(1e-4, amax)/448))` per token
  (132 B/token cache); q per (token, head) group-128 UE8M0; `w' = ((w*q_scale)*128^-0.5)*32^-0.5`
  fp32; `logit = kscale[j] * sum_h w'[h]*relu(mfma_e4m3(q_h, k_j))`, per-lane serial head sum then
  lane^32 add (plow `rowq` order). OCP e4m3fn (not FNUZ). No Hadamard rotation in vLLM's GLM path.
  Likely causes of plow's failed FP8 arm: FNUZ/fmt operand bits, q_scale applied twice (already in w'),
  or mismatched A/B k-permutation for the x64 MFMA.
- DeepGEMM sm90 `fp8_mqa_logits` (KV as A, BLOCK_Q x heads as B, TMA + math warps, per-CTA [ks,ke]);
  TRT-LLM: radix 8-bit x 4 passes then sort below 2048 candidates (7.4x torch.topk), fused K quant/store
  +33-64%. SGLang: same logits kernels + `topk_transform`; pitfalls on GLM data: fp16-bit coarse bins
  (recall 0.53) and truncated overflow bins (wrong sets). Fully fused score+top-k (FusedIndexTopK):
  only +3-9% — the score round trip is not the dominant cost.
- Plan (projected indexer chain 8k 31 -> ~9 ms, 16k ~85 -> ~28 ms):
  1. DONE (`PLOW_DSA_IDX_FP8=1`, see below): 16k 1.71 -> 1.01 ms/layer, 8k 0.49 -> 0.24.
  2. Select: one fp32-key 11-12-bit histogram pass with the row in registers, compact threshold-bin
     candidates into LDS, refine only those; handle overflow bins exactly: ~2.5x.
  3. Fuse select + union per 8-query pack (LDS bitmask): union 10.8 -> ~1 ms at 16k.
  4. Fuse indexer prep: k LN -> rope -> fp8 quant/cache in one pass; q rope + round + quant + weight
     fold in one pass; wk + weights_proj as one N=160 GEMM: ~10 ms at 8k.
  Then re-measure the crossover; the sparse flash (2x needed for 8k/16k) remains the other half.

### FP8 indexer score (`PLOW_DSA_IDX_FP8=1`, object-only, 2026-09-25)
Enable: `GLM_RECIPE=dsa GLM_OBJ_EXTRA="PLOW_DSA_IDX_FP8=1" scripts/glm53_mxfp4_mi350x.sh objects <pkt> <out>`
(no packet change; wins over DSA_IDX_QPW). `runtime/amd/dsa_score_fp8.h`: quantizes the packet's bf16
q/k/w on the fly (k per slab as it is staged into LDS, q per row in registers), 2 rows/wave sharing each
K fragment, 128-key double-buffered fp8 slabs, subtile epilogue under the next subtile's MFMAs.
- Numerics = vLLM's ops, bit for bit: `runtime/tests/index_score_pf_fp8_gfx950.hip dump` + the pinned
  image's `per_token_group_quant_fp8` / `indexer_k_quant_and_cache` / `rocm_fp8_mqa_logits` on the same
  operands: 100% of causal logits bit-identical at T4096 and T8192. Findings that got it there:
  - the head sum must replay the gluon kernel's ISA order, not the source: that image's Triton lacks the
    folded reduction (NUM_CHAINS=0), and LLVM emits `s=w1*r1; s=fma(w0,r0,s); fma slots 2..9; then
    s += round(w*r)` for slots 10..15 (packed muls, not contracted). The 4-chain source order matched 44%.
  - the x64 MFMA is not exactly rounded (fixed-point-like 64-term sum), but it is permutation-invariant:
    the k-slot layout and scaled vs unscaled MFMA give identical results.
  - `v_cvt_scalef32_pk_fp8_bf16(x, s)` == `cvt_pk_fp8_f32(x / s)` (1M probes); `v_cvt_pk_fp8_f32` is
    OCP e4m3fn on gfx950, NaN (not saturating) above 448. UE8M0 of amax/448 is an exact bit formula.
  - a packed-u16 amax must fold its halves before the cross-lane max (the one real bug found).
- Standalone (1 GPU, 256 WG, us, vs QPW): T4096 148 -> 72, T8192 489-506 -> 237-249 (1.1 PF/s),
  T16384 1706 -> 1006-1014 (1.08 PF/s), len 8192/n 4096 328 -> 211-224. 16k target (0.8) not reached:
  the per-row-subtile VALU epilogue (16 relu + 22 packed ops per 2 rows) ~ the MFMA time, and K is
  re-quantized per 16-row pack. Tried: 64/256-key slabs, 2/8/16 spans per WG (no gain).
- In-model (dsa-v5 packet, same job, 2 alternating rounds, ms): 4k 257/257 vs dsa-v6 273/273,
  8k 463/458 vs 461/461, 16k 858/870 vs 882/883. Score op (trace, 21 layers): 8k 10.2 -> 6.6 ms,
  16k 34.5 -> 20.9 ms. Greedy T1024/T8192 identical; top-1 == vLLM oracle 4/4; T8192 cos 0.977
  (dsa-v6 0.984), KL 3.3e-8 (2.9e-8); T4096 cos 0.957 (0.955).

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
