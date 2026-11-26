# Nemotron-3.5-Super-VL NVFP4 on GB10: plowrt vs vLLM

No ratio is quoted until every row below is `match` or `plowrt wider` with a work item, and both
sides pass the same quality gate (`campaign.py gate`, `llm_fp32_ref`, c1 and c8).

## Block-vs-roofline, production packet (per bucket)

HARD RULE step 1 (agent-tools.md). Packet: recipe emit env (`campaign.py build`) + `PLOW_FP8_KV=1`
+ `PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32`, HEAD 653708754 + tree diff 444ea2f8, frozen
binaries (block_run cec269da, plowrt ed00aeba, step_bench d0d21cca, plowc 02b4fae2). Ceilings
calibrated on GB10 (`calibrate_ceilings.py`): 241 GB/s read, 91.4 BF16 / 177.4 FP8 TFLOP/s; W4A4
experts are priced at the FP8 ceiling. Floors are `block_roofline.py` / `op_roof.py` per
instruction. Timings: `block_run` uninstrumented objects (block time), `-DPLOW_NV_TRACE=2` objects for
per-op splits (`tl_attr` critical-path walk), `step_bench` on natural 1K prompts for the full
packet. Boxes: jg02 (b1 sweeps, full-packet plain/spec) and jg01 (same GB10 / driver 595.91.07:
L6..7 sweeps, all traces). Blocks: L1 = MoE, L6..7 = Mamba-2 + GQA (`--block` is half-open).

MoE floors use the L1 real-text router table. At 4-32 verify rows the measured MoE expert time is
**below** that floor (verify c8: 163 ms of experts vs 198 ms floor), so deeper layers route to
fewer distinct experts than L1. MoE floors at B >= 4 are therefore an upper bound. If the experts
run at the 95-101% they reach in the L1 block, the c8 verify touches ~305 experts per layer on
average, against 369 at L1. A per-layer capture through `PLOW_NV_DUMP_ACT` needs packed-prefill
metadata, and this packet has none ("compact terminal off: no packed prefill batch"). The
remaining route is `step_bench --max-inst <layer topk> --dump-tensors act.moe.table`, one model
load per layer. Mamba2Step in span verify is priced per row (state read + write per row);
the real span reads the state once, so that op's floor is also an upper bound.

| block (layer kind, rung, ctx) | block time | roofline floor | % of floor | delta vs current |
|---|---:|---:|---:|---:|
| L6..7 Mamba-2+GQA, decode B=1/2/4/8/16/32, 1K | 0.713 / 0.744 / 0.769 / 0.854 / 1.152 / 1.607 ms | 0.627 / 0.648 / 0.691 / 0.777 / 0.950 / 1.297 ms | 88 / 87 / 90 / 91 / 83 / 81% | - |
| L6..7, decode B=1/2/4/8/16/32, 32K | 0.780 / 0.872 / 1.038 / 1.404 / 2.238 / 3.747 ms | 0.694 / 0.783 / 0.961 / 1.316 / 2.029 / 3.455 ms | 89 / 90 / 93 / 94 / 91 / 92% | - |
| L6..7, decode B=1/2/4/8, 128K | 1.010 / 1.305 / 1.895 / 3.140 ms | 0.903 / 1.201 / 1.796 / 2.987 ms | 89 / 92 / 95 / 95% | - |
| L6..7, decode B=1/2/4/8, 255K | 1.265 / 1.864 / 2.919 / 5.204 ms | 1.179 / 1.753 / 2.900 / 5.195 ms | 93 / 94 / 99 / 100% | - |
| L6..7, prefill 128/1024/4096/8192 rows, prior 0 | 1.094 / 3.910 / 17.31 / 36.97 ms | 0.879 / 3.416 / 14.71 / 32.40 ms | 80 / 87 / 85 / 88% | - |
| L6..7, prefill 128/1024/4096/8192 rows, prior 32K | 3.013 / 14.14 / 57.49 / 118.2 ms | 1.624 / 9.431 / 38.77 / 80.52 ms | 54 / 67 / 67 / 68% | - |
| L1 MoE, decode B=1/2/4/8/16/32, 1K (block asset, see note) | 1.081 / 1.395 / 1.967 / 2.866 / 4.316 / 5.850 ms | 0.475 / 0.750 / 1.254 / 2.161 / 3.484 / 5.058 ms | 44 / 54 / 64 / 75 / 81 / 86% | - |
| L1 MoE, decode B=1/4, 1K, block-asset artefact removed (trace) | ~0.53 / ~1.37 ms | 0.475 / 1.254 ms | ~89 / ~92% | - |
| L1 MoE, prefill 128/1024/4096/8192 rows, prior 0 | 9.30 / 10.83 / 18.31 / 28.27 ms | 6.79 / 8.36 / 13.65 / 21.86 ms | 73 / 77 / 75 / 77% | - |
| full packet, plain decode B=1/2/4/8, 1K | 48.3 / 62.2 / 84.9 / 123.5 ms | 41.7 / 53.5 / 75.2 / 114.6 ms | 86 / 86 / 89 / 93% | - |
| full packet, plain decode B=1, 128K / 255K | 50.3 / 52.4 ms | 43.9 / 46.1 ms | 87 / 88% | - |
| full packet, span verify c1, 32K (`--spread` prompts) | 75.3 ms | <= 77.4 ms | >= 100% (bound) | - |
| full packet, span verify c1/c2/c4/c8 (4/8/16/32 rows), 1K | 73.0 / 99.3 / 133.5 / 189.6 ms | <= 75.2 / 114.6 / 173.7 / 249.0 ms | >= 100% (bound) | - |
| full packet, MTP draft k=3, B=1/8 | 9.67 / 15.6 ms | ~10.0 / ~14.3 ms (estimate, see note) | ~100 / ~92% | - |
| full packet, prefill 1K / 8K prompt (TTFT, v2 MoE) | 0.565 / 2.199 s | 0.443 / 1.768 s | 78 / 80% | - |
| full packet, prefill 128K / 255K prompt (TTFT, v2 MoE) | 49.2 / 129.3 s | 39.8 / 104 s | 81 / 81% | - |

Notes:
* **L1 MoE decode at B <= 4 is a block-asset artefact, not production behaviour.** The MoE-only
  block's rule-derived constants give `GV_MM_MAX=1`, drop `PLOW_NV_GEMV_MMA`/W8A16/W8A8 and set
  `PLOW_PACKET_DECODE_BATCH 0`. At load the runtime logs "decode ladder retains widest execution"
  with `gemv_mm_cap=1, gemv_weight_passes=32`, so every B runs the T=32 program and each BF16
  latent GEMV (4096x1024, 35 us floor) walks its weights 32 times. In the trace, fc1_latent_proj
  runs 321 us and holds SigmoidTopk off its blocks for 291 us; fc2_latent_proj runs 257 us. That is
  ~520 us, constant across B=1..32. The routed and shared experts run at 95-101% of floor at every
  B. The Mamba-only block (L6) is hit worse: 13.7 ms against a 0.47 ms floor, because its 76 MB FP8
  in_proj makes 32 passes. In the production packet (GV_MM_MAX 32, rung selection on), the
  latent GEMVs are at or above floor in situ (verify c1: 80 GEMVs, 1.67 ms on the critical path).
  Fix owner: block emitter.
* Draft floor = 3 x (lm_head 2.23 ms + MTP dense ~0.63 ms + MTP MoE 0.475 / 1.9 ms at B=1 / 8).
  The per-op trace puts lm_head at ~100% of floor (B=1) and 86% (B=8); lm_head is 64% of the
  B=1 draft.
* Long-prompt TTFT floor = 8192-row chunk floor without attention (1.72 s) x chunks + causal
  attention at each chunk's prefix. Attention is 49 of the 104 s floor at 255K. The `PLOW_PF_SEG_TIME`
  split (prefill section) puts attention at ~64 s of the 129 s measured (77% of floor), and the
  MoE/Mamba chunks at ~85%. The L6..7 prior-32K rows (54-68%) are below that standalone figure;
  the block's prior-path cost is not yet separated from FlashPrefill.
* `certify_kernel.py` on every JSON above: REJECTED. Only L6..7 decode, L1 decode B >= 16 and
  some prefill points pass; the per-op files fail on the latency-bound ops listed below.

Per-op, in situ (full packet, span verify at 1.1K; critical-path ms per verify launch):

| op (count) | c1 verify (4 rows) | floor | % | c8 verify (32 rows) | floor | % |
|---|---:|---:|---:|---:|---:|---:|
| routed+shared experts MoeUp/DownNvfp4 (160) | 36.88 | <= 46.53 | >= 100 | 163.13 | <= 197.51 | >= 100 |
| Mamba in/out_proj GemvFp8 (80) | 21.48 | 18.25 | 85 | 25.27 | 18.57 | 74 |
| Mamba2Step (40) | 2.20 | <= 2.81 | bound | 12.46 | <= 22.47 | bound |
| MoeRouterSigmoidTopk (40) | 2.39 | 0.02 | 1 | ~1.4 (body) | 0.02 | 1 |
| MoE latent Gemv fc1/fc2 (80) | 1.67 | 2.80 | >= 100 | 1.83 | 2.89 | >= 100 |
| attention q/k/v/o GemvFp8 (32) | 1.45 | 1.19 | 82 | 1.85 | 1.24 | 67 |
| QwenRmsNorm (92) | 0.98 | 0.03 | 3 | 1.33 | 0.33 | 25 |
| MoeRouterLogitsF32 (40) | 0.89 | 0.72 | 81 | ~1.25 (body) | 1.09 | 87 |
| lm_head GemvFp8 (1, off the walk) | 2.67 (body) | 2.23 | 84 | 3.07 (body) | 2.27 | 74 |
| launch span / not on any op | 75.03 / 2.7 | 75.21 | - | 220.6 / ~4 | 249.0 | - |

Ranked by share of device time x (1 - fraction of floor). Shares come from the served
decomposition: c1 = 84% decode tick (verify 88% / draft 12%), 13% prefill; c8 = 71% decode,
29% prefill. Long context = one 255K request, TTFT dominant.

| # | stage, bucket | op | % of floor | share of device time | est. saved at 90% |
|---|---|---|---:|---:|---:|
| 1 | long ctx, prefill 8192-row chunks at prefix up to 255K | FlashPrefill (causal hd128) | ~77% | ~47% | ~9 s per 255K request (of 129 s) |
| 2 | long ctx, prefill chunks (non-attention) | W4A4 MoE + Mamba chunk ops (`PLOW_PF_SEG_TIME` split) | ~85% | ~48% | ~4 s per 255K request |
| 3 | c8, prefill 128/1024/4096/8192 buckets | W4A4 MoE routed+shared (MoeUp/DownNvfp4Pf, Combine, Residual) | 73-77% | ~18% | ~1.1 s per 50 s c8 run (~2%) |
| 4 | c1, span verify 4 rows | Mamba in/out_proj GemvFp8 | 85% | ~21% | 1.2 ms per 85 ms tick |
| 5 | c1, prefill 128-row bucket (half of c1 frames) | W4A4 MoE experts (all 512 streamed at 128 rows) | 73% | ~11% | 70 ms per frame, ~1.9 s per 92 s run |
| 6 | c1, span verify 4 rows | dispatch gaps (span minus critical path minus head) | - | ~2.7% | up to 2.7 ms per tick (unexplained) |
| 7 | c1, span verify 4 rows | MoeRouterSigmoidTopk | 1% | ~2.4% | ~2.2 ms per tick (at 5 us per layer) |
| 8 | c8, span verify 32 rows | Mamba in/out_proj GemvFp8 | 74% | ~7.6% | 4.6 ms per 237 ms tick |

Next in rank: c8 prefill Mamba+attention (~85%, ~11%); c1 verify RMSNorm (0.98 ms, fusion);
c8 lm_head in verify and draft (74 / 86%). Not ranked: verify experts (c1 36%, c8 49% of device
time). The L1-table bound puts them at >= 100%, and the same kernels reach 95-101% in the L1
block. Decode at 128K-255K is not ranked either: full-packet B=1 runs at 87-88%, L6..7 at 89-100%
and FlashDecode at 93-96%.

Ops more than 2x from floor, with reason:

| op | where | measured vs floor | reason |
|---|---|---|---|
| MoeRouterSigmoidTopk | every decode / verify MoE layer | 19 us standalone, ~60 us in situ (rows=4) vs ~0.5 us | latency-bound: 512-way sigmoid + top-22 select on one launch slice. The 3x in-situ increase over standalone is unexplained |
| QwenRmsNorm | 92 per decode step | 10-12 us vs < 1 us | latency-bound: a single 4096-wide reduce per row. Fusion candidate (step 2) |
| Mamba2GatedNorm, Mamba2Conv | decode B=1 | 8.5 / 5.1 us vs 0.2 / 1.1 us | latency-bound at one row |
| MoeCombinePf | decode B=1 | 11.5 us vs 0.4 us | latency-bound at one row |
| FlashDecode, FlashMerge | decode at 1K | 10.6 / 4.2 us vs 2.2 / 1.7 us (B=1) | latency-bound at short context; 93-96% at 128K+ |
| latent Gemv fc1/fc2 | L1 MoE block only | 321 / 257 us vs 35 us | block-asset artefact (GV_MM_MAX=1, widest rung); in situ >= floor |
| L6 Mamba-only block decode | block asset only | 13.7 ms vs 0.47 ms | same artefact (32 FP8 in_proj passes) |

## Precision parity (agent-tools.md HARD RULE)

Peer: vLLM 0.34.0, torch 2.14.0+cu130, FlashInfer 0.7.3, the model card's serving flags. Sources
are the peer's `server.log` lines and the checkpoint's `hf_quant_config.json`
(`quant_algo MIXED_PRECISION`, `kv_cache_quant_algo FP8`).

| Model | Stage | Peer | plowrt | Status |
|---|---|---|---|---|
| Nemotron-3.5-Super | routed experts (NVFP4 g16) | W4A4, `Using FLASHINFER_CUTLASS NvFp4 MoE backend` | prefill W4A4 (block-scaled mma, dynamic e4m3 sf, global scale max `input_scale`); decode W4A16 | prefill match; decode plowrt wider |
| | shared expert (NVFP4 g16) | W4A4, `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM` | prefill W4A4; decode W4A16 | prefill match; decode plowrt wider |
| | in/out/q/k/v/o, lm_head (FP8 per-tensor) | static per-tensor activation scale (`input_scale`) | prefill cuBLASLt FP8 with static `input_scale`; decode W8A16 GemvFp8 | prefill match; decode plowrt wider |
| | KV cache | FP8 e4m3 (`kv_cache_dtype=torch.float8_e4m3fn`, XQA decode), `kv_cache.py:158 Using KV cache scaling factor 1.0` (no backbone k/v scales in the checkpoint); store `sat_e4m3(x / 1.0)`, readers dequantize to bf16 (bf16 Q, bf16 MMA, P bf16) | `PLOW_FP8_KV=1`: e4m3 at unit scale (`QwenHeadNormRope` saturating RNE store), decode GQA-16 mma item and FA2 prefill dequantize exactly to bf16, P bf16, f32 accumulation | match (bf16 packets without the flag: plowrt wider) |
| | Mamba SSM state | fp16 with stochastic rounding (`--mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 --mamba-backend flashinfer`): decode FlashInfer SSU, software SR on sm_12x (Philox-4x32-5, 13 random bits, per-launch seed); prefill final state round-to-nearest | fp16: decode the same SR operator (Philox-4x32-5 per 4 elements, per-step seed, per-layer key), prefill final state round-to-nearest | match |
| | Mamba conv history | bf16 (model dtype) | bf16 | match |
| | Mamba chunk scan | Triton `ssd_*`, bf16 inputs, f32 accumulation | mma.sync bf16, f32 accumulation, same bf16 rounding points | match |
| | router | f32 (checkpoint: "MoE router remains in FP32") | f32 logits, sigmoid top-22 | match |
| | attention compute | FlashInfer, bf16 query | bf16 | match |
| | drafter (spec decode) | `spark`: LiLiCorr DFlash k=7 (Marlin W4A16 NVFP4); `mtp`: native MTP head k=3 | native MTP head (checkpoint `mtp.layers.*`: BF16 eh_proj/attention, NVFP4 latent MoE, shared FP8 lm_head), k <= 3 chosen per tick; verify span stores the fp16 state round-to-nearest (the peer's speculative path does not round stochastically either) | `mtp` match; `spark` not built |
| | drafter KV cache (`PLOW_FP8_KV` + MTP) | vLLM `mtp` arm: e4m3 at the MTP layer's checkpoint scales (k 0.061, v 0.051), store `sat_e4m3(x / s)`; `--attention-backend TRITON_ATTN` (FlashInfer fp8 prefill fails on sm_121 under MTP), whose fp8-KV path quantizes Q to fp8 too | e4m3 at the same checkpoint scales (`live_kv::KV_SCALE`, `QwenHeadNormRope` saturating RNE store of `x / s`), GQA-16 mma item applies k_scale on the logits and v_scale on the output (XQA's bmm1 / bmm2), bf16 Q | KV match; Q plowrt wider (bf16 vs fp8 Q: TRITON_ATTN serves every attention layer of the peer's MTP arm, backbone included; its no-spec arm keeps FlashInfer bf16 Q): work item, MTP ratios name it |
| | decode rows riding a prefill launch (token batch span riders) | mixed prefill + decode batches: every row W8A8 (FP8 linears, static `input_scale`) and W4A4 (NVFP4 experts) | rider rows take the launch's row-wise arithmetic, W8A8 / W4A4; their Mamba step is the verify span's (fp16 state round-to-nearest, no state write), their attention the decode rung's | match (decode-rung rows: plowrt wider) |

Work items from the wider rows: W4A4 / W8A8 decode arms measured against the W4A16 / W8A16 arms
(decode is weight-bandwidth-bound, so the wider arm may cost nothing). FP8 KV is done (below),
including the MTP drafter layer's cache at its own checkpoint scales (k 0.061, v 0.051).

## Protocol

* One tool for both sides: `scripts/bench/llm_grid.sh plow|vllm <res>` (same `vllm bench serve`
  client, unique prompts per cell and repeat, greedy and sampled cells, 2 repeats, prefill-only and
  decode-only cells, `/metrics`), rendered by `campaign.py report`. The scratch random-token sweep
  is retired: greedy random-token prompts with `--ignore-eos` collapse into repetition that the
  drafter predicts (vLLM DFlash acceptance length 6.7-7.9 of 7), which inflates speculative decode.
* Grid: `CONCS="1 2 4 8"`, one run per ISL in 1000 / 8192 / 32768 / 131072 / 262000, decode-only
  cells `d1x262144 d8x262144` and the 32K/128K equivalents, prefill-only cells for TTFT; natural-text
  `--quality` cells for speculative acceptance.
* Same box for both sides of a comparison (jg01 or jg02, identical GB10 SKUs), same
  `--max-model-len 262144`, max 8 sequences, 8192-token prefill chunk, prefix caching on both sides.
* Speculative decode in three pairs, each with both sides' acceptance length from `/metrics`:
  no-spec vs no-spec (step speed), MTP k=3 vs MTP k=3 (same drafter: the checkpoint's MTP head),
  and best vs best (vLLM `spark` LiLiCorr DFlash k=7 vs plowrt's best).
* TTFT at 128K and longer needs a matched cold start: wait for the GPU to drop to 58 °C or below
  before each arm. Past ~80 °C GB10 enters `sw_thermal_slowdown`, and an arm that always ran second
  read as an FP8-KV TTFT deficit at 261K. Cold-start matched, FP8 KV gives 128.13 s vs 128.78 s for
  bf16.

## Prefill, single request (engine wall, not yet the serving grid)

`step_bench <full packet> 2 <L> 0 --spread` (random-token prompts, 8192-row chunks, serial
prefill, the warm slot), jg01, max-ctx 262144 packet. The vLLM column is the indicative model-card
GB10 run, not a same-box `llm_grid.sh` cell: no ratio is quoted from it.

| prompt | plowrt before (shipped d_flash_prefill) | plowrt after (FA2 hd128 prefill) | vLLM indicative |
|---|---:|---:|---:|
| 1K | 0.55 s | 0.55 s | 0.86 s |
| 8K | 2.67 s | 2.59 s | 2.83 s |
| 32K | 12.71 s | 11.07 s | 10.6 s |
| 128K | 82.2 s | 56.8 s | 54.6 s |
| 256K (262000) | 248.0 s | 141.2 s | 136.5 s |

Per 8192-row chunk after the change (PLOW_PF_SEG_TIME, 40 Mamba / 40 MoE / 8 attention layers):
Mamba 0.91-0.93 s and MoE 1.57 s at every prefix, attention 0.10 s at prefix 0 rising to 3.9 s at
prefix 254K. Attention at long prefix runs at 86-87 TF/s standalone (FlashInfer single_prefill
94-98 TF/s; bf16 mma.sync peak on GB10 124 TF/s); MoE routed experts are at parity with
FlashInfer's CUTLASS fp4 MoE (25.3 ms at 8192 rows). The MoE and Mamba chunk cost is the gap at
1K-8K, attention at 128K+.

### W4A4 MoE grouped GEMM v2 (real-text prompts)

`p2_ttft.sh` with real-text prompts (prompts-262000.json), jg01, max-ctx 262144 packet, mean of
2 slots x 2 rounds. Control: v1 W4A4 kernel (PLOW_NV_W4A4_V2=0). After: v2 routed/shared kernel
(128 x 192/256 CTA, 64-wide warp tiles, bulk-async up epilogue, v1 below 2048 tokens). 1K/8K are
from a88b3d919 explicit trees. 32K/128K/256K are from the 5df023961 trees: the 8192-row chunk path
is identical on both bases, and 32K re-measured on a88b3d919 gave 10.69 -> 9.58 s.

| prompt | plowrt before | plowrt after | vLLM |
|---|---:|---:|---:|
| 1K | 0.567 s | 0.565 s | 0.86 s (indicative) |
| 8K | 2.470 s | 2.199 s | 2.83 s (indicative) |
| 32K | 10.70 s | 9.51 s | 10.6 s |
| 128K | 53.97 s | 49.24 s | 54.6 s |
| 256K (262000) | 139.0 s | 129.3 s | 136.5 s |

L1 MoE block, four-arm A/B on real-text rows (control v2 off): 8192 rows 34.50 -> 27.80 ms
(-19.4%), 4096 rows 19.66 -> 17.99 ms (-8.5%, PASS). Standalone at 8192 rows: routed up
12.3 -> 6.9 ms, down (bf16 parts) 11.8 -> 7.3 ms. These sit at 65% / 82% of their DRAM floors
(weights + fp4 intermediate + parts at 238 GB/s).

### Prefill round 8 (W4A4 MoE kernels) and round 9 (whole-network table, Mamba scan fusion)

Round 8 (landed in the tree): the routed up dropped from 6.08 to 5.27 ms at 4096 rows and from
8.20 to 7.59 ms at 8192 rows, in-block. Full-model TTFT, real text: 8K 2.233 -> 2.191 s,
32K 9.685 -> 9.408 s, 128K 49.65 -> 49.29 s.

Round 9, whole network: one 8192-row chunk of a 32K real-text prompt on jg02, with a fresh
recipe emit (round 8 header). Chunk 1 runs at prior 0, chunk 4 at prior 24K. Each segment is
timed with `PLOW_PF_SEG_TIME=1` events. Chunk sums are 2177.8 / 2274.2 / 2380.9 / 2491.2 ms
(9.32 s). Uninstrumented TTFT is 8K 2.18 s, 24K 6.84 s and 32K 9.33-9.38 s, so the host gaps
between segments are under 1%. Floors are op_roof per instruction (241 GB/s, 91 BF16 / 177 FP8
TF/s). Segments that hold several ops are split by the in-block L1 trace (MoE), or by floor ratio
for the Mamba segment (conv + chunk state / pass / out + gated norm).

| op (instances) | prior 0 ms | floor | % | above floor | prior 24K ms | floor | % | above floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MoE routed down (40) | 348.1 | 238.1 | 68% | +110.0 | 345.1 | 238.1 | 69% | +107.0 |
| Mamba chunk scan, 3 ops (120) | 336.5 | 260.4 | 77% | +76.2 | 339.1 | 260.4 | 77% | +78.7 |
| MoE routed up (40) | 287.5 | 223.7 | 78% | +63.8 | 285.0 | 223.7 | 78% | +61.3 |
| MoE shared down + residual fold (40) | 130.6 | 81.3 | 62% | +49.3 | 131.7 | 81.3 | 62% | +50.4 |
| MoE combine (40) | 106.8 | 64.3 | 60% | +42.5 | 105.9 | 64.3 | 61% | +41.6 |
| FA2 E4M3 hd128 (8) | 55.8 | 48.1 | 86% | +7.6 | 375.8 | 336.8 | 90% | +39.0 |
| norms / residual adds (93) | 83.0 | 52.1 | 63% | +30.9 | 81.9 | 52.1 | 64% | +29.9 |
| MoE router logits f32 + top-k (80) | 45.9 | 18.1 | 39% | +27.9 | 46.3 | 18.1 | 39% | +28.2 |
| MoE shared up (40) | 107.9 | 81.3 | 75% | +26.5 | 108.8 | 81.3 | 75% | +27.4 |
| Mamba gated norm + fp8 (40) | 86.4 | 66.8 | 77% | +19.5 | 87.0 | 66.8 | 77% | +20.2 |
| MoE align + routed quant (80) | 21.7 | 4.2 | 19% | +17.5 | 21.5 | 4.2 | 19% | +17.3 |
| Mamba conv (40) | 72.0 | 55.7 | 77% | +16.3 | 72.6 | 55.7 | 77% | +16.8 |
| MoE fc1 / fc2 latent, bf16 cuBLASLt (80) | 64.3 | 60.1 | 94% | +4.1 | 64.6 | 60.1 | 93% | +4.5 |
| MTP embed / eh_proj / kv, lm_head (7) | 11.1 | 8.8 | 79% | +2.3 | 11.4 | 8.8 | 77% | +2.5 |
| q/k/v/o FP8 cuBLASLt + act quant (48) | 35.4 | 34.1 | 96% | +1.4 | 35.8 | 34.1 | 95% | +1.8 |
| Mamba in / out_proj FP8 cuBLASLt + act quant (120) | 375.4 | 421.5 | 112% | -46.1 | 370.7 | 421.5 | 114% | -50.8 |
| sum of segments | 2177.8 | 1734.7 | 80% | +443 | 2491.2 | 2023.4 | 81% | +468 |

* The FP8 projections run above the 177 TF/s FP8 ceiling (in_proj at about 207 TF/s), so that
  calibration is low.
* The MoE accounts for about 60% of the time above floor.
* Routed down in-block vs standalone. The standalone harness was fed the block's own dumped
  table, meta, row ids, xq and fq. It runs up in 6.20 ms and down in 7.38 ms, against 7.28 and
  8.39 ms in-block. The routing is identical (0 rows differ), and the fq and part outputs are
  bitwise equal to the block's, so routing is ruled out.
* With `PLOW_WEIGHT_VMM=0` the in-block down measured 7.97 ms and up was unchanged (one run each).
* The SASS is not the same. In `interp_sm120_pfseg` (255 registers, 1568 B stack) the routed-down
  k-loop has 3 `LDL.64` per k-step, against none in the standalone cubin. The routed-up k-loop is
  identical, yet up is also 17% slower in-block, so the cause is still open.

Mamba CHUNK_STATE + CHUNK_PASS fused walk (`PLOW_NV_MAMBA_SCAN_FUSE`, object define, default 1):

* CHUNK_STATE runs the next instruction's pass recurrence on its accumulator fragments, one block
  per head (per request and head when packed), and the pass becomes a no-op.
* The f32 per-chunk local states never reach memory.
* When CHUNK_OUT follows, the entering states are stored as CHUNK_OUT stages them, bf16(fp16(E)).
  This halves the scratch traffic.
* The result is bit-identical to the three-op path: `mamba2_oracle.py` serial at 1, 7, 128, 1000
  and 8192 rows from a random slot state, and the 8-request packed table.

Standalone results at 8192 rows:

| op | before | after | after, % of floor |
|---|---:|---:|---:|
| state + pass | 2.35 + 2.49 ms | 1.98 ms | 93% of the state floor |
| chunk out | 2.96 ms | 2.47 ms | 97% |

L6..7 block, four-arm A/B (all points PASS):

| rows | before | after | change |
|---:|---:|---:|---:|
| 1024 | 3.877 ms | 3.444 ms | -0.433 ms |
| 4096 | 17.03 ms | 15.27 ms | -1.757 ms |
| 8192 | 36.21 ms | 32.87 ms | -3.337 ms (-9.2%), 99% of the pre-fusion floor |

Other gates:

* block output bitwise equal at 128 / 1024 / 4096 / 8192 rows;
* block oracle `--kv fp8`: 20/20;
* `certify_kernel.py`: QUALIFIED;
* knob tests: pass.

Full-model TTFT at 32K, real text, ABBA:

| arm | run 1 (slot 0 / slot 1) | run 2 (slot 0 / slot 1) | mean |
|---|---:|---:|---:|
| control | 9.40 / 9.64 s | 9.45 / 9.38 s | 9.47 s |
| treatment | 8.80 / 8.85 s | 8.81 / 8.86 s | 8.83 s |

That is -0.64 s (-6.7%), against vLLM's 8.61 s on the same box. Still open: 8K / 128K TTFT, and
`packed-check` on the block. The treatment fails it on sparse-slot KV parity, and the control arm
has not been run for comparison. (Closed in round 10.)

### Prefill round 10 (scan-fusion gates, W4A4 MoE walks on their own segment entry)

Block-vs-roofline, L1 MoE block (one layer), real-text routing, jg02, four-arm `nv-block-ab`.
Ceilings calibrated on this GB10: 241 GB/s, 91 BF16 / 177 FP8 TFLOP/s. Control = the shared tree.
Treatment = W4A4 up / down on `plow_moe_w4a4_pf`. certify_kernel.py: QUALIFIED.

| block (layer kind, rung, ctx) | block time | roofline floor | % of floor | delta vs current | verdict |
|---|---:|---:|---:|---:|---|
| L1 moe, prefill 1x1024 rows, prior 0 | 9.593 ms | 8.255 ms | 86% | -0.347 ms (-3.5%) | PASS |
| L1 moe, prefill 1x4096 rows, prior 0 | 16.075 ms | 13.230 ms | 82% | -1.704 ms (-9.6%) | UNCONVICTABLE (spread 0.16 ms) |
| L1 moe, prefill 1x8192 rows, prior 0 | 24.743 ms | 21.027 ms | 85% | -2.513 ms (-9.2%) | PASS |

The round-9 whole-network split mis-weighted two rows. It weights each op of a segment by the
median in-block trace time of its op kind, and that median mixes the routed and shared
MoeUp / MoeDown instances. With the per-op split:

* MoE combine runs at 1.67 ms per layer, 96-97% of floor, not 60%.
* The routed up / down rows were under-attributed.

Op timings, L1, 8192 rows:

* Routed up / down, in-block (trace): 7.3-7.9 / 8.0-8.6 ms, with ±0.4 ms run-to-run spread.
* The same walks standalone on the block's dumped routing and activations: 6.3 / 7.2 ms.
* Shared up / down, in-block: 1.5-2.1 / 2.5 ms.
* Router segment, split by skipping one op: logits 0.97 ms (floor 0.38), top-k 0.26 ms (floor
  0.08), QwenRmsNorm about 0.72 ms (floor 0.56).

Why the in-block W4A4 walks ran about 15% slower than standalone:

* Ruled out:
  * Weight placement. In the standalone harness, VMM weights in 2 MiB / 32 MiB / 256 MiB chunks
    and in one chunk time the same as flat (granularity 2 MiB). In-block, `PLOW_WEIGHT_VMM=0` stays
    within the trace spread.
  * Cold caches / TLB. The op timed right after the other one, or after a 3 GiB read, is unchanged.
  * Occupancy. Both run 1 CTA per SM at 255 registers on a 48-block grid.
  * SM sharing or imbalance. The trace shows 48 slices on 48 blocks, gates within 2 us, no
    overlapping instruction, and per-slice bodies within 20 us of each other.
* The cause is codegen under the interpreter's call ABI. Called from the interpreter,
  `nf4v_tiled` allocates to R247 and spills:
  * the routed down has 3 `LDL.64` per k-step;
  * up / down reload 50-65 values per item.
  * The standalone entry has none.
* Tried and rejected:
  * a dmode-specialised routed down: k-loop LDL 3 -> 4, 8192 rows +0.30 ms;
  * `__forceinline__` into the interpreter body: still 3 LDL per k-step, +0.33 ms.

The fix:

* Each W4A4 `MoeUpNvfp4Pf` / `MoeDownNvfp4Pf` prefill instruction gets its own segment. devbuild
  `set_w4a4_moe_segments` turns it on, and only `nemotron_h`, which is sm_12x only, sets it.
* plowrt launches that segment on `plow_moe_w4a4_pf(prog, inst)`, an entry of the segment object.
  * The same bodies run on the same slices (slice = blockIdx.x), on the segment arena.
  * The route mechanism is the one the NormResidual+RmsNorm and HeadNormRope segments use.
  * The waits drop to stream order.
* The entry's walks have 0 LDL / STL in every MMA loop.
* A segment object without the entry runs those segments on the interpreter, with the same output.

Segment times, L1 at 8192 rows (`PLOW_PF_SEG_TIME`):

| op | before (in-block) | after (own segment) |
|---|---:|---:|
| routed up | 7.3-7.9 ms | 6.25 ms |
| routed down | 8.0-8.6 ms | 7.18 ms |
| shared up | 1.5-2.1 ms | 1.51 ms |
| shared down | 2.5 ms | 2.23 ms |

Gates:

* Block output bitwise equal to control at 128 / 1024 / 8192 rows.
* Block oracle L6..8 `--kv fp8`: 20/20.
* packed-check L6..8, 5 cases: PASS.
* devgen golden_blob 25/25, packet lib 167/167, the new plowrt unit test.
* Knob tests: devgen 18/18, plowrt cuda,hsa 28/28.

Round-9 scan fusion, the remaining gates:

* `packed-check` failed in both arms, control included, because of a harness bug. block_run's KV
  snapshot read the unit-scale E4M3 cache (`live_kv::KV_E4M3`, no scale tensor) as bf16 rows.
  * Fixed: block_run detects the writer flag from the packet.
  * One-code E4M3 tolerance: an E4M3 byte may differ by one code of the same sign, with decoded
    rel-L2 <= 6e-3.
  * After the fix, the 5 cases pass on both arms. q4 has one byte one code away in both arms,
    where the request's activations already differ in 796 elements (bucket-dependent GEMM arm).
* Full-model TTFT, fused vs unfused:
  * 8K: 2.195 -> 2.060 s.
  * 128K: 48.92 -> 46.89 s.

Full-model TTFT, real text, ABBA, cold start <= 58 C. Control = the fused shared tree (round 9),
treatment = round 10 (W4A4 segment entry):

| prompt | control | treatment | change |
|---|---:|---:|---:|
| 8K (2 slots x 2 runs) | 2.080 / 2.047, 2.044 / 2.048 s | 1.927 / 1.959, 1.972 / 1.936 s | -0.107 s (-5.2%) |
| 32K (2 slots x 2 runs) | 8.955 / 9.159, 8.772 / 8.846 s | 8.601 / 8.378, 8.595 / 8.360 s | -0.450 s (-5.0%) |
| 32K, first job (3 arms) | 9.029 / 8.893 s | 8.348 / 8.360, 8.303 / 8.357 s | -0.63 s |
| 128K (1 slot x 2 runs) | 46.25, 46.93 s | 45.23, 44.33 s | -1.81 s (-3.9%) |

At 32K the treatment's eight slot-runs average 8.41 s (8.30-8.60), against vLLM's 8.61 s on the
same box.

Open for the next round, ms per 8192-row chunk at 32K:

* Router logits at 39% of floor, about +24 ms.
* Norms / residual, about +30 ms.
* Routed down at 83% / up at 89% of floor (own segment), about +80 ms together.

### Prefill round 11 (whole-network table per instruction, router logits on the segment entry)

Whole network, one 8192-row chunk of a 32K real-text prompt, jg02, fresh emit of the shared tree
after round 10. `PLOW_PF_SEG_TIME=1`: segments are now about one instruction each (713 per chunk),
so every row is that instruction's own segment time, with no trace split. Floors are op_roof per
instruction on the calibrated ceilings (241 GB/s, 91 BF16 / 177 FP8 TF/s). op_roof now prices a
QwenRmsNorm with a fused residual (`radd`) at 8 bytes per element (reads x and radd, writes the sum
and the norm). Round 9 priced it at 4, which is where the "+30 ms of norms" came from.
Uninstrumented TTFT 32K: 8.27 / 8.31 s.

| op (instances) | prior 0 ms | floor | % | above floor | prior 24K ms | floor | % | above floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MoE routed down (40) | 303.4 | 238.1 | 78% | +65.4 | 302.1 | 238.1 | 79% | +64.0 |
| MoE routed up (40) | 276.2 | 223.7 | 81% | +52.5 | 274.4 | 223.7 | 82% | +50.7 |
| FA2 E4M3 hd128 (8) | 56.1 | 48.1 | 86% | +8.0 | 374.6 | 336.8 | 90% | +37.8 |
| MoE router logits + top-k (80) | 45.3 | 18.1 | 40% | +27.3 | 46.4 | 18.1 | 39% | +28.3 |
| MoE shared down (40) | 92.6 | 81.3 | 88% | +11.3 | 94.5 | 81.3 | 86% | +13.1 |
| norm + residual, radd (88) | 106.5 | 98.0 | 92% | +8.5 | 107.2 | 98.0 | 91% | +9.2 |
| MTP embed / eh_proj / k,v (4) | 12.6 | 6.6 | 53% | +6.0 | 12.9 | 6.6 | 52% | +6.2 |
| MoE align + routed quant (80) | 7.5 | 4.2 | 55% | +3.4 | 7.2 | 4.2 | 58% | +3.1 |
| MoE fc1 / fc2 latent, bf16 cuBLASLt (80) | 63.5 | 60.1 | 95% | +3.3 | 63.8 | 60.1 | 94% | +3.7 |
| MoE combine (40) | 66.0 | 64.3 | 97% | +1.8 | 64.9 | 64.3 | 99% | +0.7 |
| MoE shared quant (40) | 15.3 | 14.3 | 93% | +1.0 | 15.2 | 14.3 | 94% | +0.9 |
| Mamba chunk scan, fused (120) | 243.3 | 260.4 | 107% | -17.1 | 246.1 | 260.4 | 106% | -14.3 |
| Mamba conv / gated norm + fp8 (80) | 114.5 | 122.5 | 107% | -8.1 | 115.9 | 122.5 | 106% | -6.8 |
| Mamba in / out_proj FP8 cuBLASLt + act quant (120) | 389.7 | 421.5 | 108% | -31.8 | 389.9 | 421.5 | 108% | -31.6 |
| MoE shared up (40) | 64.4 | 81.3 | 126% | -17.0 | 63.8 | 81.3 | 127% | -17.5 |
| sum of segments | 1903.9 | 1783.7 | 94% | +120.2 | 2224.4 | 2072.4 | 93% | +152.0 |

Rows at or above 100% of floor mean the ceiling under them is low: the FP8 GEMM ceiling, and the
W4A4 tensor rate for the shared up. Per layer at 8192 rows:

* router logits 0.77 ms (floor 0.38);
* top-k 0.27 ms (floor 0.08);
* norm with radd 1.18 ms (floor 1.11);
* norm without radd 0.63 ms (floor 0.56).

What is still above floor per chunk:

* routed down + up: about +118 ms;
* router: +27 ms;
* shared down: +11 ms;
* norms: +8.5 ms;
* align + quant: +3.4 ms.

Norms (task 3) and align + quant are near their floor and were not worked on.

Routed MoE k-loop, clock64 split (warp 0 per block, standalone on the L1 dump at 8192 rows):

| walk | Mclk | wait | k-loop | epilogue + setup |
|---|---:|---:|---:|---:|
| up | 15.8 | 4% | 65% | 30% |
| down | 20.2 | 0% | 82% | 17% |

* The down is MMA-issue bound: its loads are fully hidden.
* The up's epilogue (relu², NVFP4 requant, bulk store) is about a third of the walk. The
  ping-pong arm that hides it is off from 2048 rows; round 8 measured it at +0.5 ms at 4096 and 8192.
* Not changed this round.

Router logits on `plow_moe_w4a4_pf` (`d_moe_router_logits_pf`). Not applied: no in-block gain.
The design:

* Prefill rows run 128 x 128 tiles (2 x 4 warps, warp tile 64 x 32) on a 4-stage cp.async ring
  with BK = 32.
* Every logit gets `nf4_tiled<0>`'s mma sequence: m16n8k16, ascending k from 0, the same fragment
  maps. The f32 logits are bitwise equal to the interpreter's at 128, 1000, 1024, 4096, 8100 and
  8192 rows, with 0 route-set differences.
* devbuild isolates `MoeRouterLogitsF32`. The entry takes it at ABI 2.

Standalone at 8192 rows, L2 flushed:

| kernel | time | % of floor |
|---|---:|---:|
| interpreter path (`nf4_tiled<0>`) | 985 us | 38% |
| 128 x 128 tiles | 712 us | 53% |
| 128 x 256 tiles, fragment double-buffering | 701 us | 54% |

At 1024 rows: 188 -> 92-132 us.

In-block it does not hold:

* L1 segment time: interpreter 0.78-0.83 ms; the same body on the entry 0.79-0.84 ms; the new body
  inside the interpreter 1.03-1.05 ms.
* Four-arm L1 A/B: 4096 rows +0.095 ms (REGRESSION), 8192 rows +0.27 ms (UNCONVICTABLE). The
  interpreter-body arm: 1024 rows +0.06 ms (REGRESSION).
* Full TTFT ABBA: 8K 1.95 -> 1.92 s, 32K 8.29 -> 8.32 s (noise).
* Block output, oracle L6..8 `--kv fp8 --route-check` 20/22 (the two misses are the info-only
  device-routing rows), and packed-check 5/5 all pass. There is no gain to ship.
* Open: the interpreter path runs 21% faster in-block than standalone, and the new body 45%
  slower inside the interpreter than standalone. Both builds use the same nvcc flags and show
  0 LDL / STL in the k-loops. The standalone harness does not reproduce the in-block conditions;
  a hot, no-flush run and a run with hn freshly rewritten keep the standalone ranking.

Open for the next round:

* routed down / up, about +118 ms per chunk;
* the router gap, +27 ms, behind the standalone vs in-block discrepancy above.

### Prefill round 12 (FP4 ceiling, 16384-row prefill rung)

Block-vs-roofline, L1 MoE block, real-text routing, jg02, four-arm `nv-block-ab` (floors still at the
FP8 price for W4A4; FP8 ceilings calibrated on this GB10: 241 GB/s, 91 BF16 / 177 FP8 TF/s). Control =
shared tree, treatment = routed-down partials stored `st.global.cs`. certify_kernel.py: QUALIFIED.

| block (layer kind, rung, ctx) | block time | roofline floor | % of floor | delta vs current | verdict |
|---|---:|---:|---:|---:|---|
| L1 moe, prefill 1x128 rows, prior 0 | 7.509 ms | 6.785 ms | 90% | -0.097 ms | NOISE |
| L1 moe, prefill 1x1024 rows, prior 0 | 9.438 ms | 8.255 ms | 87% | -0.048 ms | UNCONVICTABLE |
| L1 moe, prefill 1x4096 rows, prior 0 | 15.811 ms | 13.230 ms | 84% | -0.094 ms | NOISE |
| L1 moe, prefill 1x8192 rows, prior 0 | 24.499 ms | 21.027 ms | 86% | -0.057 ms | NOISE |
| L1 moe, prefill 1x16384 rows (16384 rung) | 43.859 ms | | | -4.97 ms vs 2 x 8192 | per token -10.2% |

The MMA is already the native NVFP4 form. In SASS (`plow_moe_w4a4_pf`), both walks issue
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` (mma.sync m16n8k64 `kind::mxf4nvf4`, `scale_vec::4X`, ue4m3).
Per k128 step and warp:

* routed down (128 x 256): 64 OMMA, 52 NOP, 16 LDSM, 15 LDS, about 84 integer / address ops, 231
  instructions;
* routed up (128 x 192): 48 OMMA in 197 instructions;
* 0 LDL / STL; issue slots are about 10% used, so neither walk is issue-bound.

FP4 ceiling (`calibrate_ceilings.py` now measures it): cuBLASLt NVFP4 `torch._scaled_mm` (e2m1, e4m3
scale per 16) runs 378.6 TF/s at 8192^3 (344 at 4096^3) against e4m3 184.7 TF/s in the same lease,
2.05x. hwspec `GB10_FP4_TFLOPS`, costmodel `peak_dtype` (W4A4 experts at FP4 only where
`fp4_peak_measured`, else FP8), `op_roof.py` / `block_roofline.py` (`fp4_ceiling`: the ceilings
file's `fp4_tflops`, else the registry's measured value) and `roofline.py` now carry it. Re-priced
per 8192-row chunk (round 11 table):

| op | measured | floor at FP8 | floor at FP4 | % at FP4 | bound |
|---|---:|---:|---:|---:|---|
| MoE routed down (40) | 303.4 ms | 238.1 | 238.1 | 78% | DRAM (weights 793 MB + fq 272 MB + bf16 parts 369 MB per layer) |
| MoE routed up (40) | 276.2 ms | 223.7 | 177.6 | 64% | DRAM after the fix (was matrix) |
| MoE shared down (40) | 92.6 ms | 81.3 | 38.1 | 41% | matrix |
| MoE shared up (40) | 64.4 ms | 81.3 | 38.1 | 59% | matrix |

* Routed down: about 200 GB/s of DRAM traffic (83% of the read ceiling). The clock64 split shows 82%
  of the time in the k-loop with no cp.async wait, but the k-loop also issues the next stage's
  LDGSTS. Its per-k-step time is 1.5x the shared up's, which runs the same loop code at 239 TF/s.
  It is memory-bound, not MMA-bound.
* Routed up: the k-loop runs at about 272 TF/s (72% of FP4); the epilogue is 30% of the walk.
* The shared expert walks are now the largest gap to their (FP4) floor.

Standalone screens (w4ab harness, L1 real text, 1024 / 4096 / 8192 rows):

| candidate | routed up | routed down | verdict |
|---|---|---|---|
| deferred up epilogue (item i's relu^2 + requant in chunks inside item i+1's k-loop, bf16x2 registers) | +0.32 / +0.95 ms at 4096 / 8192 (160-512 B spills, branchy chunks) | = | rejected |
| per-warp m16 trim (skip m16 slices past the item's rows; 8% of MMA rows are padding) | noise | noise; shared up / down +0.05..+0.2 ms | rejected |
| `cvt` e2m1x4 (two cvts instead of four with zeros) | -0.07 ms, within spread | = | not kept |
| routed-down partials `st.global.cs` (`PLOW_NV_W4A4_V2_STCS`, default 1) | = | -0.12 / -0.16 / -0.06 ms | landed |

`st.global.cs` gates:

* L1 block bitwise equal at 128 / 1024 / 4096 / 8192 rows.
* Oracle: L1 20/20; L6..8 `--kv fp8 --route-check` 20/22 (the two misses are info-only
  device-routing rows).
* packed-check L6..8: 5/5 PASS.
* certify: QUALIFIED.
* Full TTFT ABBA: 8K 1.937 -> 1.925 s; 32K 8.447 -> 8.318 s.

16384-row prefill rung. A chunk streams every expert's weights once (all 512 experts are touched
from about 1K rows), so a wider chunk halves the per-token weight traffic of the DRAM-bound walks.

* devgen `nemotron_h` now admits `PLOW_PF_LADDER_APPEND` rungs above the 8192 always rung
  (up to max_ctx, chunk multiples).
* plow-asset `LT_FP8_TENSOR_ROWS` += 16384. cuBLASLt per-tensor FP8 runs at the same rate there:
  the in-proj segment is 1.64-1.75 ms at 16384 rows vs 0.79-0.80 ms at 8192.
* Recipe: `PLOW_PF_LADDER_APPEND = "1152,1536,2048,2176,16384"`.
* Segments at 16384 rows (L1): routed up 11.58 ms, routed down 11.25 ms, against 2 x 6.28 and
  2 x 7.15 ms at 8192.

Gates:

* L6..8 oracle at 7 / 1000 / 8192 / 12000 / 16384 rows: 17/18 (T=16384 rel-L2 1.43e-2; the miss
  is an info-only routing row).
* Recipe block, default rows + 8192 / 12000 / 16384: 22/25, with one decode miss at 2.015e-2 (limit
  2e-2, decode after T=1000, a path the rung does not touch). Re-run below.
* packed-check 5/5 PASS.
* `check_recipe.py` OK on the recipe emit.
* Tests: hwspec, costmodel, plow-asset, devgen knob, plowrt knob, devgen nemotron.

Full-model TTFT, real text, ABBA. Control = the same tree with ladder 1536,2048; 128K cold <= 58 C
each arm, VMM on.

| prompt | control | 16384 rung | change |
|---|---:|---:|---:|
| 8K (2 slots x 2) | 1.911 / 1.921, 1.954 / 1.932 s | 1.947 / 1.939, 1.928 / 1.945 s | noise (small rungs kept) |
| 12K (2 x 2) | 3.087 / 3.126, 3.082 / 3.118 s | 3.109 / 3.122, 3.090 / 3.103 s | noise (planner keeps 8192 + 4096) |
| 24K (2 x 2) | 6.077 / 6.104, 6.178 / 6.153 s | 5.849 / 6.020, 5.838 / 5.957 s | -0.21 s (-3.5%) |
| 32K (2 x 2) | 8.289 / 8.320, 8.356 / 8.468 s | 7.870 / 7.887, 7.872 / 7.849 s | -0.49 s (-5.8%); vLLM 8.61 s |
| 128K (1 x 2) | 44.48, 44.21 s | 42.21, 42.53 s | -1.97 s (-4.4%) |

The planner prices a launch as padded rows plus its measured fixed cost (`pf_chunk_cost_rows`, the
launch fit). A 9K-15K prompt still takes 8192 + tail, not a padded 16384.

Same tree, recipe with and without the 16384 rung (the arms differ only by the rung):

* 256K (262000 tokens, 1 slot, cold <= 58 C, VMM on): 117.63 / 118.23 -> 113.54 / 113.27 s
  (-4.5 s, -3.8%).
* MemAvailable floor during the run: 36.5 -> 34.7-35.5 GiB, so the wider scratch costs about
  1.3 GiB.
* Oracle on the recipe block re-run twice: 20/22 both times (the two info rows only). The 2.015e-2
  decode row above does not reproduce.
* Packed serve sanity (spec3 `plow_quality.sh`, MTP k=3, 1K ISL / 256 OSL, 1 rep, mean TTFT and
  tok/s per request):
  * c1: 0.990 -> 0.985 s, 38.76 -> 38.81 tok/s;
  * c4: 1.206 -> 1.358 s (p50 1.30 -> 1.00, p90 2.26 -> 2.20), 20.65 -> 20.79 tok/s;
  * c8: 2.258 -> 2.006 s, 14.51 -> 15.06 tok/s.
  * 1K prompts never reach the 16384 bucket. The c4 / c8 TTFT moves are single-rep noise in
    opposite directions.

Open for the next round:

* shared down / up at 41% / 59% of the FP4 floor (+54 / +26 ms per chunk);
* routed up epilogue: 30% of the walk, and an overlap without spills;
* routed down: DRAM efficiency (83%);
* router logits (+27 ms).

## Decode step (engine, not yet the serving grid)

`step_bench <full packet> <B> <ctx> 48 --warmup 8 --spread`, GB10, 40K max-ctx packet, mean ms per
step. Before: the a77424e02 packet and binaries. After: fp16 SSM state with the peer's stochastic
rounding, register-resident decode RMSNorm, gated norm / conv loads in one round, shared expert
interleaved with the routed experts, k/v projections off the q dependency, hd128 decode attention
on the row-group body at 4 query heads per item with SM-balanced splits.

| ctx | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---:|---:|---:|---:|---:|
| 1K | 50.68 -> 48.29 | 60.70 -> 57.45 | 70.45 -> 63.58 | 78.46 -> 70.11 | 110.72 -> 101.38 |
| 32K | 53.00 -> 50.02 | 58.60 -> 53.93 | 70.93 -> 61.55 | 89.75 -> 73.34 | 123.87 -> 94.57 |

Four-arm block A/B of the same change (all points PASS): mamba+attention block L6..7 decode
B=1/2/4/8/16 at ctx 1K -4.8/-7.1/-6.3/-21.9/-22.8%, at 32K -12.8/-19.8/-24.7/-34.5/-36.5%
(83-92% of the block floor); attention+MoE block L7..8 -1.1 to -5.5% (B=2/4/16 at 32K not
convictable: treatment spread > 3x drift). That job overlapped CPU builds on the box (GB10's CPU
and GPU share LPDDR5X). The L7..8 rerun (quiet start, a compile during the run): B=1/2/4/8 at 1K
and 32K PASS, B=16 1K noise, B=16 32K still not convictable (0.004 ms drift, 0.045 ms spread).

Attention-layer block (L6..7, `block_run bench`, decode median): B=1 at 260K 2.77 -> 1.91 ms,
B=8 at 128K 9.15 -> 5.71 ms.

## FP8 e4m3 KV (decode round 3)

`PLOW_FP8_KV=1`: unit-scale e4m3 cache written by `QwenHeadNormRope` (i7 bit 8, `live_kv::KV_E4M3`,
object define `PLOW_NV_KV_E4M3`), read by a GQA-16 decode item on mma.sync (one item per KV head,
all 16 query heads as the m16 rows, so each KV byte is read once per step; `PLOW_NV_FA_GF=16`) and
by the FA2 hd128 prefill (e4m3 tile staged by cp.async into a raw smem tile, dequantised into the
bf16 smem ring under the P.V mma stream; "FP8 KV prefill" below). Block oracle
`--kv fp8`: 20/20, B=16 49/49, decode after 32K 5/5, 128K 3/3, 256K 3/3; the device-vs-model error
equals the bf16 packet's (KV rounding modelled exactly), and the e4m3 rounding adds ~1e-3 rel-L2
against a bf16-KV reference.

Mamba+attention block L6..7, four-arm `nv-block-ab` (control bf16 KV, treatment FP8 KV; same lease,
no compiler on the box), decode median, calibrated GB10 ceilings (238 GB/s); certify_kernel.py:
QUALIFIED at every point (>= 80%).

| block (layer kind, rung, ctx) | block time | roofline floor | % of floor | delta vs current |
|---|---:|---:|---:|---:|
| L6 mamba2+full_attn, decode B=1, ctx 1024 | 0.746 ms | 0.635 ms | 85% | +0.003 ms (noise) |
| L6 mamba2+full_attn, decode B=8, ctx 1024 | 0.906 ms | 0.787 ms | 87% | -0.006 ms (noise) |
| L6 mamba2+full_attn, decode B=16, ctx 1024 | 1.151 ms | 0.962 ms | 84% | -0.080 ms (-6.5%) |
| L6 mamba2+full_attn, decode B=1, ctx 32768 | 0.812 ms | 0.703 ms | 87% | -0.072 ms (-8.1%) |
| L6 mamba2+full_attn, decode B=4, ctx 32768 | 1.085 ms | 0.973 ms | 90% | -0.283 ms (-20.7%) |
| L6 mamba2+full_attn, decode B=8, ctx 32768 | 1.448 ms | 1.333 ms | 92% | -0.589 ms (-28.9%) |
| L6 mamba2+full_attn, decode B=16, ctx 32768 | 2.221 ms | 2.055 ms | 93% | -1.257 ms (-36.1%) |
| L6 mamba2+full_attn, decode B=1, ctx 131072 | 1.023 ms | 0.914 ms | 89% | -0.310 ms (-23.3%) |
| L6 mamba2+full_attn, decode B=8, ctx 131072 | 3.137 ms | 3.025 ms | 96% | -2.370 ms (-43.0%) |
| L6 mamba2+full_attn, decode B=1, ctx 262000 | 1.298 ms | 1.196 ms | 92% | -0.650 ms (-33.4%) |
| L6 mamba2+full_attn, decode B=2, ctx 262000 | 1.903 ms | 1.779 ms | 93% | -1.138 ms (-37.4%) |
| L6 mamba2+full_attn, decode B=4, ctx 262000 | 3.046 ms | 2.945 ms | 97% | -2.250 ms (-42.5%) |
| L6 mamba2+full_attn, decode B=8, ctx 262000 | 5.361 ms | 5.278 ms | 98% | -4.577 ms (-46.1%) |

Every 32K+ point PASS except B=2 at 262K (UNCONVICTABLE: 0.6 us control drift against a 28 us
treatment spread, delta -1.14 ms). Repeated on the consolidated commit (control = the same tree
emitted without the flag, quiet box): 32K+ deltas within 0.1 ms of the above; that build had the
decode item out of line (`__noinline__`), which cost +0.05-0.07 ms at 1K B=4/8 and was reverted.

Full-model step_bench on the consolidated commit, mean ms (other agents compiled on the box during
these runs; the B=4 128K control hit a host OOM):

| ctx | B=1 | B=2 | B=4 | B=8 |
|---|---:|---:|---:|---:|
| 128K | 52.35 -> 50.10 | 60.19 -> 55.81 | - -> 67.53 | 102.36 -> 82.16 |
| 256K | 57.16 -> 53.23 | 70.78 -> 61.37 | 95.34 -> 76.93 | 147.13 -> 103.26 |
 Full-model step_bench (`--spread`, 262144
packets) at ctx 32K B=1/2/4/8/16: 49.55 -> 49.49, 54.72 -> 55.34, 65.40 -> 64.13, 73.47 -> 69.62,
94.38 -> 84.45 ms (other agents compiled on the box during that run). At ctx 1K the full model reads
+0.7..+8 ms although the blocks are at parity: the e4m3 rounding changes the greedy token streams of
slots 1.., so the decode MoE touches a different expert union (step_bench has no forced-token mode);
the routing-free block A/B is the measurement of record there.

### FP8 KV prefill (FA2 hd128 E4M3 staging)

The first FP8 KV round timed decode only. Its prefill path loaded tile t+1 into registers and
dequantised it after P.V behind a barrier; inside the prefill object that body hit 255 registers and
spilled the staged tile every 64 keys. Full-model TTFT on that path, against bf16 KV: 32K 9.60 ->
10.19 s, 128K 50.0 -> 67.3 s, 261000 tokens 132.0 -> 201.2 s. Only the attention layers grew
(segment split at 128K: 16.2 -> 32.9 s; Mamba and MoE unchanged).

The new path copies tile t+1's raw e4m3 bytes with cp.async into a 16 KiB tile behind the bf16 ring
(arena 86016 B). Each thread dequantises only the 16-byte chunks it copied itself, so no barrier
sits between copy and conversion. One chunk is converted per P.V k-step, under the mma stream. The
dequantisation is exact, so the output is bitwise equal to the bf16 path on the dequantised K/V
(standalone, prior 0..254K; rel-L2 vs f64 1.0e-3..1.7e-3). Registers drop from 231 to 198
standalone, with 0 spills in the full-model `_pfseg` object. Block oracle `--kv fp8`: 20/20,
B=16 49/49, decode after 32K / 128K / 256K 5/5, 3/3, 3/3, and prefill of 20000 rows 2/2.
certify_kernel.py: QUALIFIED.

L6..7 block, four-arm `nv-block-ab` (jg01). Both arms run FP8 KV; control = the previous staging.
8192 rows, calibrated ceilings 241 GB/s, 91 BF16 TFLOP/s. All points PASS (drift <= 3.3 ms,
treatment spread <= 1.1 ms):

| block (layer kind, rung, ctx) | block time | roofline floor | % of floor | delta vs current |
|---|---:|---:|---:|---:|
| L6 mamba2+full_attn, prefill 8192 rows, prior 0 | 36.157 ms | 32.402 ms | 90% | -1.268 ms (-3.4%) |
| L6 mamba2+full_attn, prefill 8192 rows, prior 32768 | 90.691 ms | 80.521 ms | 89% | -30.485 ms (-25.2%) |
| L6 mamba2+full_attn, prefill 8192 rows, prior 131072 | 258.042 ms | 224.877 ms | 87% | -113.864 ms (-30.6%) |

Standalone FA2, 8192 rows, ms (bf16 / previous E4M3 / new E4M3): prior 0 6.23 / 7.03 / 6.77;
32K 56.7 / 67.8 / 56.8; 128K 210.5 / 256.3 / 215.2-234.8; 254K 404.7 / 491.9 / 411.8.

Full-model TTFT, fresh emits of the same tree, one jg01 lease, `p2_ttft.sh` real-text prompts,
slot 0 / slot 1:

| prompt | bf16 KV | FP8 KV, previous staging | FP8 KV, new staging |
|---|---:|---:|---:|
| 32K | 9.62 / 9.59 s | 10.17 / 10.22 s | 9.58 / 9.60 s |
| 128K | 49.46 / 51.27 s | 67.18 / 67.35 s | 49.50 / 52.61 s |
| 261000 | 129.34 / 130.84 s | 200.96 / 201.50 s | 133.26 / 133.17 s |

At 261000 tokens FP8 KV is still +3 s (+2.4%) slower than bf16. Per-chunk flash segment times
(PLOW_PF_SEG_TIME, one attention layer) follow the same slope on both arms up to ~100K prior:
14.0 ms per 8K of prior, which matches the standalone rate. Past ~100K both arms show irregular
spikes, larger on FP8 KV: 608 ms of excess per layer over the linear trend, against 227 ms on
bf16. The block (contiguous KV, no VMM) and the standalone harness show no spikes. The suspect is
the live-KV VMM path under `PLOW_VMM_LIVE=1`, which is not yet isolated. Running without VMM at
128K or more exhausts GB10 memory, so that check needs a different design.

### FP8 KV with the MTP drafter (decode round 4)

`PLOW_FP8_KV=1` now also covers the MTP drafter: its cache stores `sat_e4m3(x / s)` at the
checkpoint's k/v scales (0.061 / 0.051, `live_kv::KV_SCALE`) and the GQA-16 item applies them as
XQA's bmm1 / bmm2 scales (FlashDecode t7), as the peer's `mtp` arm does. Gates: span-verify
identity 20/20 cases on the RN-state packet; block oracle L6..7 `--kv fp8` 20/20, B=16 49/49,
decode after 32K 5/5, 128K 3/3, 256K 3/3.

Note on packets: since the decode refactor the tensor-core FP8 GEMV needs `PLOW_FP8_DECODE_TC=4`
at emit (the recipe sets it). Block A/Bs emitted without it time the CUDA-core W8A16 GEMV at
B >= 8 (both arms 57-77% of the floor there); they are no-TC arms, not the served path.

Attention change on the backbone (four-arm block A/B, L6..7, 653708754 FP8 block vs + drafter
scales, no-TC arms, quiet box): 12 points -1.9..+0.1%, 7 PASS, no regression.

bf16 KV vs FP8 KV on the served path (both `PLOW_FP8_DECODE_TC=4`), mamba+attention block L6..7,
four-arm `nv-block-ab`, quiet box, calibrated ceilings (238 GB/s). B=32 is the attention width of
a B=8 k=3 verify. The no-TC pair had read +13% at 1K B=8; with the tensor-core GEMV that point is
-2.9% (PASS), so it was a CUDA-core GEMV interaction, not the attention item.

| block (layer kind, rung, ctx) | block time | roofline floor | % of floor | delta vs current |
|---|---:|---:|---:|---:|
| L6 mamba2+full_attn, decode B=1, ctx 1024 | 0.762 ms | 0.635 ms | 83% | -0.001 ms (noise) |
| L6 mamba2+full_attn, decode B=2, ctx 1024 | 0.787 ms | 0.656 ms | 83% | -0.007 ms (noise) |
| L6 mamba2+full_attn, decode B=4, ctx 1024 | 0.812 ms | 0.700 ms | 86% | -0.016 ms (-2.0%) |
| L6 mamba2+full_attn, decode B=8, ctx 1024 | 0.897 ms | 0.787 ms | 88% | -0.026 ms (-2.9%) |
| L6 mamba2+full_attn, decode B=16, ctx 1024 | 1.207 ms | 0.962 ms | 80% | -0.056 ms (-4.4%) |
| L6 mamba2+full_attn, decode B=32, ctx 1024 | 1.661 ms | 1.313 ms | 79% | -0.080 ms (-4.6%) |
| L6 mamba2+full_attn, decode B=1, ctx 32768 | 0.832 ms | 0.703 ms | 84% | -0.078 ms (-8.6%) |
| L6 mamba2+full_attn, decode B=2, ctx 32768 | 0.926 ms | 0.793 ms | 86% | -0.160 ms (-14.7%) |
| L6 mamba2+full_attn, decode B=4, ctx 32768 | 1.120 ms | 0.973 ms | 87% | -0.274 ms (-19.7%) |
| L6 mamba2+full_attn, decode B=8, ctx 32768 | 1.471 ms | 1.333 ms | 91% | -0.583 ms (-28.4%) |
| L6 mamba2+full_attn, decode B=16, ctx 32768 | 2.336 ms | 2.055 ms | 88% | -1.179 ms (-33.5%) |
| L6 mamba2+full_attn, decode B=32, ctx 32768 | 3.895 ms | 3.498 ms | 90% | -2.350 ms (-37.6%) |

All PASS except B=1/2 at 1K (unconvictable: 0.1 us control drift). certify_kernel.py: B=16/32 at
1K 79.8 / 79.1% (below 80%); both qualify with the stream-K rungs below.

Stream-K tensor-core GEMV (`d_gemv_fp8_tc_sk`: the CTA's (16-row tile, 64-wide K chunk) units
split evenly over its warps; the tile walk idles warps on the last round). Standalone (48 CTAs,
L2 evicted) it won only on out_proj (N 4096, K 8192: 192 -> 176 us at M=4, 200 -> 174 at M=16) and
lost on in_proj M=16/32 and kv_proj M=16. In the block (FP8 KV, TC=4, ctx 1K, control = tile walk):

| rung | every shape | in_proj + out_proj only | routed now |
|---|---:|---:|---|
| B=4 | -1.9% (unconvictable) | -0.9% PASS | in_proj + out_proj |
| B=8 | -4.1% PASS | +0.6% noise | every shape |
| B=16 | +0.1% noise | -0.2% noise | tile walk |
| B=32 | -2.0% PASS | -1.5% PASS | every shape |

B=1/2 do not reach the tensor-core walk. Both stream-K arms qualify at every point (certify_kernel
80-90%). Full model, the B=8 k=3 verify (32 rows, the FP8 arm's own token streams in both arms):
207.2 -> 195.3 ms.

Folding the residual add + RMSNorm into the GemvFp8 prologue (every block normalizes all rows,
QwenRmsNorm's bits, residual stream ping-ponged; Mamba in_proj and q/k/v at rows <= 4) did not pay
on GB10: block B=1 noise, B=2 +1.9% / +1.5% (REGRESSION at 1K / 32K), B=4 +2.5% / +1.9%. Not
kept.

Block assets without attention (the MoE-only L1 block, Mamba-only L6) sized their decode objects
from FlashDecode alone, so they built the 1-row GEMV object (`GV_MM_MAX 1`, no GEMV MMA arm,
`PLOW_PACKET_DECODE_BATCH 0`) and ran the widest program at every rung. The manifest now falls
back to the decode GEMV rows; GV_MM_MAX / GEMV MMA / decode batch match the full packet. L1 MoE
block, real rows, ctx 1K, before -> after: B=1 1.030 -> 0.662 ms (73% of floor), B=2 1.319 ->
0.942 (80%), B=4 1.850 -> 1.485 (86%), B=8 2.676 -> 2.326 (94%), B=16 4.053 -> 3.714 (97%), B=32
5.479 -> 5.222 ms. Block numbers on those blocks from before this fix measured the wrong object.
Emitting the decode top-k on its per-token blocks only (so the other blocks start fc1 / the
shared expert under the router tail) measured B=1 +1.0% (REGRESSION), the rest noise: not kept.

Full model before -> after: 653708754 recipe-equivalent packet (bf16 KV, MTP k=3, TC=4) vs FP8 KV
with the drafter at its checkpoint scales and the stream-K rungs above (same `step_bench` binary,
262144 max-ctx packets, 1..32 ladder, quiet box, one rep). Plain: `--force-prompt`, mean ms per
step. Spec: mtp k=3, tick = verify + draft ms (tok/s per slot = emitted / tick).

| ctx | plain B=1 | plain B=2 | plain B=4 | plain B=8 |
|---|---:|---:|---:|---:|
| 1K | 47.65 -> 47.68 | 61.20 -> 60.69 | 84.37 -> 84.13 | 120.06 -> 120.43 |
| 32K | 49.26 -> 48.50 | 65.23 -> 61.95 | 91.07 -> 87.58 | 103.54 -> 97.98 |
| 128K | 53.01 -> 50.18 | 70.53 -> 65.37 | 101.74 -> 89.92 | 126.30 -> 106.34 |
| 256K (261000) | 57.83 -> 52.40 | 80.03 -> 68.97 | 117.72 -> 98.19 | 160.42 -> 121.50 |

| ctx | spec B=1 | spec B=2 | spec B=4 | spec B=8 |
|---|---:|---:|---:|---:|
| 1K | 82.8 -> 81.7 (29.7 -> 31.4) | 106.7 -> 110.3 (24.2 -> 23.0) | 149.3 -> 157.0 (16.6 -> 17.1) | 197.1 -> 212.3 (14.4 -> 13.1) |
| 32K | 92.9 -> 88.7 (41.3 -> 43.3) | 126.3 -> 121.5 (29.8 -> 31.6) | 187.1 -> 175.7 (20.5 -> 21.9) | 226.7 -> 202.7 (16.9 -> 18.9) |
| 128K | 100.0 -> 86.0 (38.4 -> 44.6) | 145.9 -> 119.9 (26.3 -> 32.0) | 215.9 -> 169.0 (17.8 -> 22.7) | 299.7 -> 204.8 (12.8 -> 18.8) |
| 256K (261000) | 117.3 -> 89.3 (32.8 -> 43.0) | 171.8 -> 123.4 (22.4 -> 31.1) | 273.5 -> 179.1 (14.0 -> 21.5) | 414.5 -> 229.4 (9.3 -> 16.7) |

The 32K+ prompts accept the same 3.84 tokens per span in both arms. At 1K the spec rows verify
each arm's own drafts (acceptance 2.46-2.83, different 8-32-row MoE expert unions), so the 1K B>=2
tick deltas are routing, not kernels: the routing-pinned plain rows and the block A/B (B=8 / B=32
-2.9 / -4.6%) show no 1K regression.

The production recipe does not set `PLOW_FP8_KV` yet: the flag ships only once FP8-KV prefill
TTFT is at or below bf16 KV at 32K / 128K / 261K (the e4m3 FA2 prefill staging measured +52% at
256K before its v2 staging, now parity at 32K and +2.4% at 256K).

Full model, 1K, two reps (`step_bench`, natural prompts; plain with `--force-prompt`, so both arms
route the same tokens), ms:

| arm | plain B=4 | plain B=8 | spec B=1 verify + draft (emit) | spec B=8 verify + draft (emit) |
|---|---:|---:|---:|---:|
| bf16 KV, TC | 85.4 | 120.1 | 73.2 + 10.6 (2.46) | 180.5 + 16.6 (2.83) |
| FP8 KV, TC | 84.5 | 120.8 | 72.7 + 10.6 (2.33) | 207.2 + 18.2 (2.78) |
| FP8 KV, no TC | 86.1 | 141.0 | 74.5 + 10.6 (2.46) | 355.6 + 22.3 (2.81) |
| FP8 KV, TC stream-K | 84.3 | 121.4 | 72.0 + 10.6 (2.61) | 195.3 + 17.5 (2.78) |

Speculative rows are not routing-pinned: each arm verifies its own draft tokens, so the 32-row
verify's MoE expert union differs between the bf16 and FP8 arms (the FP8 and stream-K arms share
streams, acceptance 2.783). Long context, spec mtp k=3 B=1, same acceptance (3.84) in both arms:

| ctx | bf16 KV verify + draft | FP8 KV verify + draft |
|---|---:|---:|
| 32K | 82.0 + 10.7 ms | 79.3 + 10.3 ms |
| 128K | 87.2 + 12.6 ms | 75.5 + 11.0 ms |


## Speculative decode: native MTP head vs vLLM `mtp` k=3 (same drafter)

`llm_grid.sh plow|vllm <res> --quality` over the same natural-text corpus (model card + docs/arch
pages), chat template, greedy, 256 output tokens; acceptance from the `/metrics` deltas, decode
tok/s per request = (generation tokens - 1) / `request_decode_time`. c1: 12 prompts at 128 / 1K /
4K; cN > 1: 2N prompts of 1K. GB10 jg01, no compiler running (checked before and during). vLLM 0.34.0
with `--attention-backend TRITON_ATTN` (its FlashInfer fp8-KV ragged prefill JIT module fails to
load on sm_121 under MTP); plowrt packet with a 1..32 decode ladder (32 slots), fp16 SSM state,
SpecPolicy choosing k <= 3 per tick. Aggregate = plowrt ticks with every request decoding.

| | plowrt accept len | vLLM accept len | plowrt tok/s/req | vLLM tok/s/req | plowrt aggregate (tick ms, k) |
|---|---:|---:|---:|---:|---:|
| c1 | 3.382 (0.912 / 0.792 / 0.678) | 3.388 (0.907 / 0.794 / 0.686) | 38.95 | 40.68 | 39.0 (85.7, 3) |
| c2 | 3.359 | 3.380 | 27.34 | 29.52 | 57.5 (116.3, 3) |
| c4 | 3.524 | 3.433 | 18.22 | 21.90 | 85.2 (163.8, 3) |
| c8 | 3.495 (0.933 / 0.838 / 0.724) | 3.396 (0.907 / 0.802 / 0.687) | 11.49 | 14.48 | 117.0 (237.5, 3) |

The decode MoE weight prefetch change (below) moved c1/c2/c4/c8 from 38.07 / 26.34 / 17.30 / 10.97
(aggregate 38.1 / 55.6 / 80.7 / 111.8) to the numbers above. Before this round c8 verified at most k=1 (9.33 tok/s/req, 89 aggregate): decode rungs above 16
rows trapped in the decode MoE and speculative launches never ran the widest program. Now the
decode MoE runs up to 64 rows (router GEMV and grouped GEMV in m16 passes; the emitter refuses a
wider rung) and a verify runs the widest program on the row-slot twin, so 8 x 4 rows verify on the
32-row program. Acceptance is equal at every concurrency. plowrt per-request tok/s trails at c8
by more than its aggregate does: the second wave's prefills stall the running requests' decode
(vLLM chunks prefill into its decode steps), and the tick grows from ~192 ms to ~250 ms within
2-3 s of sustained decode at c4/c8.

The tick growth is routing, not clocks. Plain B=32 on natural prompts runs 204 -> 235 ms over
60 steps with the SM clock flat at 2496-2515 MHz, 57-59 W, 60-71 C and no throttle reason
(nvidia-smi every 200 ms). A 20 s idle inside the process at step 50 does not reset it (230 ms on
the next step), and every process repeats the same per-step curve. The same 32 slots on one prompt
(22 distinct experts per layer) stay flat at 92-100 ms, random-token prompts at 155-163 ms, B=8
at 120 ms. The decode MoE streams each distinct expert once (85-100% of its byte floor), so the
step follows how many distinct experts the batch routes to, and that count rises as the
continuations diverge. vLLM pays the same: its generation rate in the same c8 x 1024-token run falls
from ~128 to ~110 tok/s. Over that run the vLLM step averages ~218 ms (3.36 accepted / 15.4
tok/s/req) against plowrt's 244.5 ms tick, a decode-step gap separate from scheduling.

Serial prefill scheduling (base a88b3d919, packet ladder 128/1024/4096/8192, same packet both arms).
Without packed prefill, a launch serves one prompt and streams the expert weights whatever its rows:
~270 ms fixed + ~0.25 ms/row (128 / 1024 / 4096 rows: 0.30 / 0.58 / 1.29 s). A 1058-row chat
prompt runs [128, 1024] in 0.93 s. The c8 control prefills a burst of 8 one launch per tick, with
a plain (non-speculative) decode step between launches. The runtime change does three things.
First, decode rows still speculate on a serial prefill tick. Second, the serial cover prices a
launch at its measured fixed cost (least-squares fit over each bucket's fastest launch, never below
512 rows). Third, further serial chunks run in a tick while it stays within one widest launch's
measured time, so a burst shares ticks. Two interleaved reps per arm on jg01, no compiler running;
vLLM MTP k=3 from the arms above:

| | control tok/s/req | spec3 tok/s/req | vLLM | control TTFT mean | spec3 TTFT mean | vLLM TTFT |
|---|---:|---:|---:|---:|---:|---:|
| c1 | 39.22 | 38.64 | 40.68 | 1.27 s | 1.27 s | 1.24 s |
| c2 | 27.34 / 27.29 | 27.07 / 27.31 | 29.52 | 1.61 / 1.71 s | 1.73 / 1.61 s | 1.74 s |
| c4 | 18.01 / 17.83 | 17.89 / 18.47 | 21.90 | 2.21 / 2.03 s | 2.06 / 2.14 s | 2.07 s |
| c8 | 11.36 / 11.31 | 11.74 / 11.71 | 14.48 | 3.50 / 3.45 s | 3.30 / 3.10 s | 2.90 s |

c8 +3.4% tok/s/req and -8% TTFT (prefill ticks 31 -> 11-12); c2/c4 inside the rep spread; c1 is
a single request (one rep each, unaffected). The remaining per-request gap is the serial
per-prompt launch (vLLM prefills 8 x 1K prompts in ~2 s in one batched step, and its decode rows ride
along) plus the verify tick (above). Extra prefill rungs are not an option yet: a packet with
256/512/1536/2048/3072/6144 appended failed the block oracle on every prefill case that lands on a
new rung (T = 1500 / 1537 / 2048 / 3000 / 6000: rel-L2 ~0.06 against the 0.02 limit; T <= 1000 and
decode B = 4 / 16 / 32 pass), and its 1536 rung ran 1058 rows in ~1.0 s.

Mamba span snapshots were indexed by verify launch row (`by_row`): one state copy per slot
(2.5 GiB at 32 slots fp16) instead of `[slot][depth]` (7.5 GiB at k=3). Replay rollback (below)
replaces them.

`step_bench --spec mtp` (natural prompts, ctx 1K, greedy; this raw-text continuation accepts less
than the chat capture), verify + draft ms per tick and tokens emitted per span:

| B | plain ms (tok/s) | k=1 | k=2 | k=3 |
|---|---:|---:|---:|---:|
| 1 | 47.9 (20.9) | 58.0 + 3.5, 1.75 (28.5) | 66.6 + 7.2, 2.13 (28.9) | 73.2 + 10.7, 2.46 (29.3) |
| 8 | 120.1 (66.6) | 151.0 + 6.1, 1.81 (92.2) | 168.9 + 11.3, 2.41 (107.0) | 180.7 + 16.6, 2.83 (114.7) |

Plain B=16 178.0 ms, B=32 205-232 ms (median 224.0). A draft step is 3.45 ms at B=1 and
5.77 ms at B=8.

Decode MoE ops vs their byte floor (238 GB/s, random routing, L2 flushed, U = distinct experts):

| op | T=1 | T=4 | T=16 | T=32 | T=64 |
|---|---:|---:|---:|---:|---:|
| routed up, whole-item L2 prefetch (before) | 171 us, 84% | 637 us, 85% | 1957 us, 85% | 2973 us, 85% | 3497 us, 86% |
| routed up, per-row line prefetch (now) | 156 us, 92% | 549 us, 98% | - | 2517 us, 100% | - |
| routed down | 166 us, 86% | 594 us, 91% | 1813 us, 91% | 2778 us, 91% | 3302 us, 91% |
| shared up | 68 us, 76% | 70 us, 74% | 73 us, 71% | 97 us, 54% | 147 us, 35% |
| shared down | 65 us, 80% | 72 us, 72% | 80 us, 65% | 112 us, 46% | 178 us, 29% |
| router logits | 23 us, 76% | 23 us, 76% | 23 us, 76% | 27 us, 65% | 35 us, 51% |

Full-model decode step in situ (TRACE=2 timeline, critical-path ms per step, natural prompts):

| op | B=1 crit / floor | B=4 | B=8 | B=32 |
|---|---:|---:|---:|---:|
| span (ms) | 49.6 | 87.8 | 128.3 | 206.3 |
| MoE up + down | 16.1 / 15.7 | 48.5 / 47.3 | 83.8 / 83.6 | 138.2 |
| Mamba in_proj FP8 | 13.5 / 12.8 | 14.8 / 12.8 | 15.4 / 12.9 | 16.7 / 13.0 |
| Mamba out_proj FP8 | 6.5 / 5.7 | 7.3 / 5.7 | 7.9 / 5.7 | 8.1 / 5.8 |
| Mamba2Step | 0.9 | 2.4 | 5.5 | 23.8 |
| router top-k (latency) | 1.7 | 2.3 | 1.9 | 1.8 |
| QwenRmsNorm x 92 (latency) | 1.0 | 0.9 | 0.9 | 1.0 |
| lm_head FP8 | 2.28 body / 2.26 | | | |

The c1 tick (87.7 ms vs the peer's 83.3) is a 4-row verify (75 ms) plus three 3.45 ms draft
steps. The decode MoE (93-97% of its floor in situ at B=1..8) and the lm_head GEMV (99%) are at
their floors; the remaining gap is single-block latency ops on the critical path (92 RMSNorms,
the router top-k behind the fc1 / shared-expert traffic, Mamba conv / gated norm) and the FP8
projection GEMV at M >= 4 (out_proj 72-78%, in_proj 84-87% of floor), about 3-5 ms per step.

### Verify tick round 4: Mamba replay rollback, single-walk shared expert and router

Mamba span state (`recurrent_state.verify.replay`). A verify no longer writes any Mamba state.
Instead, each span row records its step inputs: the conv input row, and the SSM step's x, B and dt.
Records live at `[2 parities][slots][row]`, about 38 KiB per row per layer.

A commit of `c` rows defers a replay of that span's first `c` records. The next row-slot launch runs
the replay, with the same FMA order and the same per-row fp16 rounding as the span:
- A slot that verifies again replays inside its own span item, with the state still in registers,
  and writes the committed state once.
- A slot outside that launch replays in its own work item.
- Any other reader of the state (plain decode, a state materialize or publish, a bench snapshot)
  first runs the packet's replay program. That program is `recurrent_state.replay`: the widest
  rung's Mamba sites with no span rows.

Effect, per slot and verify:
- SSM state traffic drops from 80 MiB read + 80 write + 3 x 80 snapshot writes (+ 160 MiB per cut
  span) to 80 read + 80 write.
- The per-cut-span snapshot copies go away.
- The snapshot tensors (one state copy) become 2 x 38 KiB x slots of records per layer.
- A `by_row` packet fails closed (unknown manifest field).

`plowrt disasm` names the packed Mamba immediates (`state_bytes`, `depth`, `replay_table`,
`rowslot`, `record`). The span-verify high bits no longer read as a state element size.

Decode MoE, single K walk for two m16 tiles:
- Dense (shared expert) items above 16 rows used to make two sequential walks of the item. They now
  make one walk, and each window's dequantized weights feed both tiles' mma.
- The router logits GEMV gets the same change.
- Each row's mma sequence is unchanged, so outputs are bitwise equal at any width. Op A/B: every op
  output is bitwise equal at T = 1..64.
- The two-tile pass runs out of line (`__noinline__`). Inlined, its register pressure spilled the
  one-tile pass that every routed item runs: B=1 verify +1 ms.

| op (op test, L2 flushed) | T=16 | T=17 | T=24 | T=32 | T=48 | T=64 |
|---|---:|---:|---:|---:|---:|---:|
| shared up before / after (us) | 74.6 / 70.4 | 94.1 / 70.6 | 96.1 / 71.4 | 100.1 / 73.5 | 122.6 / 94.9 | 150.4 / 108.3 |
| shared down before / after | 79.6 / 75.6 | 106.2 / 73.4 | 109.3 / 73.5 | 113.3 / 76.5 | 144.1 / 105.1 | 176.8 / 120.5 |
| router logits before / after | 24.2 / 24.1 | 26.3 / 23.2 | 26.3 / 23.2 | 28.4 / 26.4 | 30.4 / 28.4 | 34.5 / 29.4 |

Gates:
- Identity gates on the round-to-nearest-state packet: every case `SPEC_IDENTITY ok`. Cases are
  cycle, poison, adversarial, oracle, mtp and padding at B = 1/3/4/6/8 on rungs 4/8/16/32.
- New `--spec-plain-every N` cases (b1-cycle-plain, b6-cycle-plain-poison, b8-cycle-plain) put
  plain decode steps between verifies. Those steps run the replay program first.
- Block oracle: L6..7 20/20, B=16 49/49, B=32 97/97.

`step_bench --spec mtp` k=3, ctx 1K, `PLOW_FP8_DECODE_TC=4` packets, quiet box (no compiler during
any run). Each cell is verify + draft ms per tick, one value per rep.

| arm | B=1 | B=8 |
|---|---:|---:|
| control | 73.1 / 73.3 + 10.6 | 181.5 / 180.7 + 16.4 |
| + replay rollback | 73.5 / 73.8 + 10.0 | 179.2 / 178.1 + 15.0 |
| + shared expert single walk (inlined) | 74.3 / 74.5 + 10.3 | 176.8 / 178.3 + 15.1 |
| + router single walk (inlined) | 74.1 / 74.4 + 10.2 | 177.9 / 177.5 + 15.2 |
| + two-tile pass out of line (shipped) | 73.8 / 74.1 + 10.1 | 177.2 / 177.7 + 15.0 |

In situ, t2 verify at B=8 (32 rows, TRACE=2 critical path, 223 ms total):
- MoE down 87.3 ms, MoE up 81.5, FP8 GEMV 28.4 (floor 22.4), Mamba2Step 13.4, Mamba2Conv 2.6.
- Before replay, Mamba2Step was 23.8 ms at 32 rows.

Prefetching the span step's per-row inputs (four rows ahead) left Mamba2Step at 13.4 ms. It also
pushed the interpreter's register pressure into the decode MoE (+5 ms), so it was not kept.

Serving, natural-text corpus, MTP k <= 3, one interleaved rep, control vs round 4. Each cell is tok/s/req
(ms per tick).

| | control | round 4 | vLLM `mtp` k=3 |
|---|---:|---:|---:|
| c1 | 38.93 (85.8) | 39.70 (86.7) | 40.68 |
| c2 | 27.43 (116.6) | 26.55 (116.8) | 29.52 |
| c4 | 18.56 (162.5) | 18.71 (160.6) | 21.90 |
| c8 | 11.59 (235.4) | 11.89 (229.8) | 14.48 |

Token streams are bitwise identical across all arms: step_bench plain and spec FNV hashes match at
B=1 and B=8. The serving acceptance (3.38-3.49) moves with the adaptive draft length, so c1/c2
tok/s/req differences are within that noise. The clean c1 figure is the B=1 step_bench tick above
(parity). The remaining tick gap is the decode MoE (169 of 223 ms at 32 rows) and the FP8
projection GEMV.

### Decode round 6: routed MoE walk after the two-tile pass (not applied)

Plain decode on the round-4 tree is slower than the 653708754 control. A B=1 timeline puts
+0.4-0.6 ms per step in the routed decode MoE walk. With the two-tile call in `nf4_moe_dec`'s loop,
the one-tile walk that every routed item runs lost its schedule.

Candidate: `nf4_moe_dec<PHASE, WIDE>`, with the two-tile entry taken only by dense (shared expert)
items at T > 16. Op A/B is bitwise equal at T = 1..64. Two other changes were dropped on block
regressions: the M <= 8 stream-K GEMV out of line, and a router top-k staging change.

Full model, plain `--force-prompt` 1K, mean ms per step, two interleaved reps, quiet box (no
compiler during any run). Packets:
- ctl: 653708754 tree, FP8 KV, TC=4, no MTP. 653708754 refuses FP8 KV with the drafter.
- base: 653708754 recipe-equivalent (bf16 KV, MTP k=3, TC=4), the control of the round-4 numbers.
- shr: the round-4 tree, the recipe packet.
- moe: shr + the candidate.

Both controls run on the dec4 `step_bench`, since 653708754's has no `--force-prompt`.

| B | ctl | base | shr | moe | moe vs ctl |
|---|---:|---:|---:|---:|---:|
| 1 | 47.38 / 47.47 | 47.55 / 47.60 | 48.38 / 48.37 | 47.85 / 48.09 | +0.54 (+1.1%) |
| 2 | 61.83 / 60.40 | 60.39 / 60.62 | 61.20 / 62.57 | 60.95 / 61.39 | noise (ctl drift 1.4) |
| 4 | 83.84 / 83.60 | 83.95 / 84.82 | 84.62 / 84.25 | 83.75 / 83.59 | -0.05 |
| 8 | 120.22 / 120.10 | 121.54 / 121.88 | 122.72 / 121.25 | 121.73 / 121.89 | +1.65 (+1.4%) |
| 16 | 178.29 / 177.25 | 178.80 / 177.57 | 179.22 / 179.28 | 179.53 / 178.18 | +1.09 (noise: drift 1.04) |
| 32 | 199.69 / 200.09 | 201.63 / 201.99 | 200.12 / 199.94 | 199.17 / 199.38 | -0.62 |

The candidate recovers 0.4-0.8 ms of the round-4 loss at B = 1/4/16/32. At B=1 it stays +0.5 ms
above both controls, beyond noise.

At B=8, ctl's emit had no MTP, while shr's had MTP and span verify. Re-emitted like-for-like (the
same plain FP8-KV TC=4 emit from the dev tree, decode cubin from each tree), B=8 measures:
- ctl 119.87 / 119.58
- shr 121.21 / 120.67
- moe 120.79 / 121.19

So the round-4 tree is +1.2 ms (+1.0%) at B=8 on a matched packet, and the candidate does not move
it. That B=8 gap lies outside the routed MoE walk and is still open.

Spec mtp k=3 B=8 verify (natural prompts, the same streams in both arms): shr 190.5 / 191.6 ms,
moe 191.1 / 191.2 ms (noise).

Neither arm passes `SPEC_IDENTITY` against plain decode on the stochastic-rounding recipe packet.
Both fail with the same spec hash, so the mismatch is not this change.

Block oracle:
- L6..7: 20/20, B=16 49/49, B=32 97/97.
- L0..1 (Mamba + MoE): 14/20 (prefill T >= 1000 at 2.3-2.4e-2 against the 2e-2 limit), B=16
  46/49, B=32 96/97 (decode slot 7 / 10 rows at 4.6-5.3e-2).
- The candidate and a block built wholly from the round-4 tree give identical numbers in every
  case, so the L0..1 failures predate this round: an open item.

`certify_kernel.py` on the four-arm block A/Bs (L0..1, ctx 1K): B=1 / B=2 at 72-76% of the floor
in every arm, below 80% before this round too. B=4..32 at 80-99%.

Decision rule: apply only if moe <= ctl within noise at every B and the B=8 verify <= shr. The B=1
row fails it, so the shared tree keeps the round-4 MoE.

FP8 KV vs bf16 KV MTP drafter, routing-pinned inputs (`--force-prompt` then `--draft-bench`, 16
prompts at four offsets, 64 rows, k=3 drafts against the target's own greedy continuation):

| | FP8 KV packet | bf16 KV packet |
|---|---:|---:|
| mean accepted prefix (of 3) | 1.797 | 1.609 |
| rows with equal prefix | 56 / 64 | |
| k=1 top-1 equal | 61 / 64 | |
| draft k=3, B=8 (ms) | 16.75 / 16.78 | 16.18 / 16.73 |
| draft k=3, B=1 (ms) | 10.55 | 10.13 |

- The drafter cache writer is exact. A cache row read back as e4m3 times the checkpoint scale
  (k 0.0614, v 0.0513) is 2.7% rel-L2 from the drafter's own pre-cache K. That is exactly the
  requantization error of `sat_e4m3(x / s)`. Saturated bytes: k 0, v 3.6e-7.
- The reader is exact. The device attention output matches an fp32 host recompute over the
  dequantized cache (k_scale on the logits, v_scale on the output) to 0.17% rel-L2, the same as
  the bf16 arm.
- The two arms differ upstream of the drafter. The target's final-norm hidden that feeds the
  drafter differs 6-9% between the FP8-KV and the bf16-KV backbone.
- On the same inputs, the FP8 drafter cache does not lower draft quality. A lower serving
  acceptance on the FP8-KV recipe packet is a property of its token streams, not of the drafter.

### Decode round 8: slim Mamba span step, FP8 GEMV stream-K at 9..16 rows

Measured on the second GB10 (jg02). Two packets, both built by `campaign.py build` from the recipe,
both carrying the top-k hoist that round 7 moved into the emitter:
- C: the shared tree before this round.
- N: C plus the two changes below.

Per-op attribution of one verify launch: MTP k=3, ctx 1K, B=4 (16 rows), TRACE=2 critical path.
Each cell is ms (% of floor). A trace adds about 3 ms to the span.

| op | C | slim span | + stream-K 9..16 (trace of that change alone) |
|---|---:|---:|---:|
| verify span | 151.30 | 149.09 | 148.26 |
| Mamba2Step | 6.04 | 3.96 | 6.17 |
| Mamba in_proj GemvFp8 | 15.50 (83%) | 15.52 (83%) | 16.17 (80%) |
| Mamba out_proj GemvFp8 | 7.72 (74%) | 7.49 (76%) | 7.15 (80%) |
| routed MoE up / down | 50.09 / 53.17 | 50.14 / 53.37 | 47.94 / 51.72 |

Single-trace GEMV attribution moves ±1 ms between runs. The tick table below is the evidence.

Mamba span step:
- Round 7's version cost the in/out_proj and MoE +1.6 ms. Register spills caused that, not
  instruction-cache pressure.
- Taking `d_mamba2_step_span` out of line (`__noinline__`) changed register allocation across
  the whole decode interpreter:
  - spill loads went from 302 to 2031;
  - the stack frame went from 624 to 1872 B;
  - every out-of-line helper (the MoE two-tile pass among them) went from 0 to 28-96 spill
    loads.
- Inlining the same batched-register step still spills: 942 spill loads in the kernel body.
  Its tick was +0.7 ms at B=4.
- Shipped version:
  - the step stays inlined;
  - each launch builds a compact per-block list of the units that have work (span starts and
    replay-only slots);
  - every row's inputs (x, B, C, dt, and the replayed records) are requested into L1 with
    `prefetch.global.L1` before the state loads. No registers are held across the row walk.
  - The interpreter's spill loads go from 291 to 372 and its stack from 624 to 640 B.
- Mamba2Step drops from 6.04 to 3.96 ms. The floor is 2.78 ms, so it sits at 70% of the floor.
- The arithmetic is unchanged. The spec streams are bitwise equal to C at B=1/2/4/8.
- Span identity gates on the round-to-nearest-state packet are all `SPEC_IDENTITY ok`:
  b1-cycle-poison, b1-adv, b3-k1-pad-poison, b4-cycle-poison, b8-cycle-poison and
  b6-cycle-plain-poison.

FP8 decode GEMV at 9..16 rows (Mamba in_proj, out_proj and attention projections at B=3/4 verify):
- The tile walk owns one 16-row tile per warp. On GB10 (48 CTAs) in_proj has 25 tiles per CTA and
  out_proj 6, so the last round leaves 22-25% of the warps idle.
- These rows now take the stream-K path that rows 5..8 and 17..32 already use. It is a one-line
  dispatch change.
- Stream-K sums the K partials in a different order: deterministic, but not bitwise equal to C.
- A balanced K split inside the tile walk (ks <= 4) gave the same tick. It regressed L6..7 at
  B=4/8 in the block A/B (+1.6% / +0.8%), so it was dropped.

Four-arm block A/B of the stream-K change (ctx 1K, `nv-block-ab`): no REGRESSION verdict on
either block.

| block | B=1 | B=2 | B=4 | B=8 | B=12 | B=16 | B=32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L6..7 mamba2+attn | +0.6% | +0.7% | -2.7% | -3.2% | -4.3% | -3.4% | -0.6% |
| L0..1 mamba2+moe | -1.7% | -1.4% | -1.4% | -1.0% | -1.7% | -1.4% | -1.8% |

Gates for the stream-K change:
- Block oracle L6..7: 20/20, B=16 49/49, B=32 97/97.
- Block oracle L0..1 B=16: 46/49. `--route-check` shows the three misses are near-tie route flips
  (slots 7 and 10, margin <= 2.4e-4), the same failures as before this round.
- `certify_kernel.py`: QUALIFIED on both block A/Bs, every arm.

Full model, `step_bench --spec mtp` k=3, ctx 1K, natural prompts. Tick ms, two interleaved reps,
quiet box (no compiler during any run). N vs C:

| B | N | C | delta |
|---|---:|---:|---:|
| 1 | 81.67 / 81.39 | 81.15 / 81.56 | noise |
| 2 | 104.42 / 104.67 | 104.78 / 104.75 | -0.2 |
| 4 | 149.67 / 150.27 | 153.94 / 154.35 | -4.2 (-2.7%) |
| 8 | 200.80 / 200.73 | 202.61 / 202.54 | -1.8 (-0.9%) |

Recipe serve:
- Protocol: the packed ship gate's (`plow_quality.sh` natural text, MTP k=3, 1K ISL / 256 OSL;
  c1 uses 12 mixed-length prompts).
- Run on jg02 with the dec8 plowrt build. The ship gate's binary lives on jg01, so the table above
  (jg01) is a reference here, not a control.
- Three reps, arms interleaved. Each cell is the mean of the three reps.

| | N tok/s/req | C tok/s/req | vLLM | N vs vLLM | N tick ms | C tick ms | accept len N / C |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | 38.86 | 39.04 | 40.68 | -4.5% | 84.5 | 84.5 | 3.33 / 3.35 |
| c2 | 28.36 | 28.10 | 29.52 | -3.9% | 114.6 | 114.4 | 3.44 / 3.38 |
| c4 | 19.98 | 19.85 | 21.90 | -8.8% | 160.2 | 161.6 | 3.47 / 3.48 |
| c8 | 14.00 | 13.99 | 14.48 | -3.3% | 225.1 | 227.2 | 3.49 / 3.49 |

- The serve tick moves only 1.4 / 2.1 ms at c4 / c8. Step_bench at B=4 / B=8 moves 4.2 / 1.8 ms.
- Adaptive draft length puts many verify launches at 8 rows or fewer, which already ran stream-K.
  That is the likely reason for the difference; it has not been checked against the served row
  counts.
- tok/s/req differences between N and C are within acceptance noise.

Not applied:
- Routed down vs routed up per byte. Op-level with L2 flushed, down is only about 3% slower than
  up (T=16: 1705 vs 1655 us). The in-situ gap is 6%. Three candidates were all bitwise equal and
  none won:
  - a bulk L2 prefetch of the whole item instead of per-window lines: +4% at T=16/32;
  - the next item's weights in L2: +40-55%;
  - activations two windows ahead: noise.
- lm_head overlapped with the MTP-KV chain (at most 0.26 ms). Global-queue entries are claimed in
  order, so the dependent chain parks every block that claims it. Overlap would need lm_head split
  into uneven slice sets around a narrow chain.

### Decode MoE entry (routed walk items claimed per warp)

Block-vs-roofline: L0..1 (Mamba2 + MoE) decode, ctx 1K, four-arm `nv-block-ab` on jg02.
- Ceilings calibrated on this GB10: 238 GB/s, 94 BF16 / 181 FP8 TFLOP/s.
- Control: the decode interpreter built from 3a507d70b. Treatment: the claimed-item walk below.
- Same block packet in both arms; only `interp_sm120.cubin` differs.

| block (layer kind, rung, ctx) | block time | roofline floor | % of floor | delta vs current | verdict |
|---|---:|---:|---:|---:|---|
| L0 mamba2+moe, decode B=1, ctx 1024 | 1.302 ms | 0.961 ms | 74% | -0.017 ms (-1.3%) | NOISE |
| L0 mamba2+moe, decode B=2, ctx 1024 | 1.616 ms | 1.257 ms | 78% | -0.031 ms (-1.9%) | PASS |
| L0 mamba2+moe, decode B=4, ctx 1024 | 2.173 ms | 1.813 ms | 83% | -0.060 ms (-2.7%) | PASS |
| L0 mamba2+moe, decode B=8, ctx 1024 | 3.047 ms | 2.801 ms | 92% | -0.114 ms (-3.6%) | PASS |
| L0 mamba2+moe, decode B=16, ctx 1024 | 4.405 ms | 4.363 ms | 99% | -0.219 ms (-4.7%) | PASS |
| L0 mamba2+moe, decode B=32, ctx 1024 | 6.128 ms | 6.376 ms | 104% | -0.284 ms (-4.4%) | PASS |

`certify_kernel.py`: B=4..32 QUALIFIED. B=1 and B=2 stay below 80% of the floor in both arms, as
before this change (round 6). The L7..8 (attention + MoE) block A/B could not run: block_run
rejects that block's control arm ("packed requests require complete direct-KV segmented chains").

Step 1: is the decode interpreter's MoE codegen the problem?
- The decode interpreters (`interp_sm120` 255 registers / 608 B stack, `_rs` 255 / 640 B) have 0
  LDL / STL in the routed and dense MoE k-loops.
- A standalone `__launch_bounds__(256, 1)` entry over the same body compiles to the same
  418-instruction k-loop.
- So unlike the prefill W4A4 walks (round 10), there is no spill to remove.

Routed up + down, summed over the 40 layers of one verify (MTP k=3, ctx 1K). In-situ span (TRACE=2,
first gate end to last end) vs the standalone entry on the same weights, at matched distinct-expert
counts U (synthetic Zipf tables, L2 flushed, outputs bitwise equal across entries):

| B (rows) | U mean | in-situ span | standalone, grid 48 | standalone, grid 96 (2 CTA/SM) |
|---|---:|---:|---:|---:|
| 1 (4) | 58 | 37.7 ms (79%) | 33.1-33.3 ms (90%) | 36.4-36.5 ms |
| 2 (8) | 100-112 | 61.1-67.7 ms (84-85%) | 54.4-61.5 ms (93-94%) | 60.4-68.3 ms |
| 4 (16) | 184-187 | 108.9 ms (87-88%) | 99.1-102.7 ms (94-95%) | 111.4-115.0 ms |
| 8 (32) | 282-288 | 162.1-166.0 ms (89%) | 153.5-158.0 ms (94%) | 173.6-178.6 ms |

The in-situ bodies stream at 92-98% of their floor. The span loses to skew, not to the bodies:
- The walk is a fixed warp stride: warp `slice * 8 + w` takes items `gw, gw + nw, ...`.
- Blocks reach the routed up 70-80 us apart, after their shared-expert up item.
- Blocks reach the routed down 35-60 us apart, after the shared down.
- A late block still owns its full share of items, so the op ends when the latest block finishes.

The span comparison also overstates what is left to win. The in-situ span contains shared-expert
bodies that overlap it (40-70 us per op per block). The standalone walk runs no shared expert.

Candidates:
- A (no split). Routed items are claimed per warp, in claim order:
  - The claim word is slot 16 of the instruction's own counter line. It sits in its own 32-B
    sector, away from the gate value in slot 0. Every launch re-zeroes the whole line.
  - The interpreter publishes that address through shared memory before the gate. Its MoE cases
    pass it to the walk.
  - A warp's first item stays `gw`; every later item is a claim issued one item ahead. The next
    item's scales are prefetched as before. A first version that claimed the first item as well
    lost 0.16 ms at plain B=1: every warp waited about 2 us on 384 same-address atomics.
  - Dense (shared-expert) items, prefill objects and entries without a claim word keep the fixed
    stride.
  - Each item's rows and mma order are unchanged. Outputs are bitwise equal: op test at B=1..16,
    spec and plain token streams below.
- B (split the step at the MoE ops into CUDA-graph nodes). A graph node costs 1.83 us on GB10
  (82-node replay 152.6 us vs 2-node 6.1 us). Each split also drains and refills the persistent
  grid. After A, the remaining in-situ loss is no larger than that cost (below). Not built.
- C (a second persistent kernel for the MoE walks). Same bound as B. Not built.

Trace after A: TRACE=2 verify, MoE window = fc1 latent GEMV start to routed-down end, which covers
fc1, shared up / down and routed up / down. Summed over 40 layers against its byte floor:

| B | before | after |
|---|---:|---:|
| 1 | 39.8-40.1 ms (88-89%) | 39.2-39.4 ms (90%) |
| 8 | 167.3-171.2 ms (90%) | 160.4-163.5 ms (94%) |

- The window now runs at the standalone routed walk's efficiency (90% at B=1, 94% at B=8).
- What remains is the shared down at T <= 16. It has 256 items for 384 warps, about 100 us at
  ~50% of its floor, with 16 blocks already in the routed down.
- Moving it in the queue only moves that window. Splitting its K would change the summation order.

Full model, `step_bench`, ctx 1K, natural prompts, quiet box (no compiler during any run). Two
interleaved reps per arm, same streams (spec_fnv / TOK_STREAM fnv equal in every pair):

| B | spec k=3 tick, control | spec k=3 tick, A | delta | plain, control | plain, A | delta |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 80.69 / 81.67 | 79.83 / 80.46 | -1.03 (-1.3%) | 47.19 / 47.34 | 47.31 / 47.37 | +0.08 (noise: control spread 0.15) |
| 2 | 104.56 / 104.80 | 102.68 / 103.00 | -1.85 (-1.8%) | 60.06 / 60.21 | 59.61 / 59.67 | -0.49 (-0.8%) |
| 4 | 149.98 / 150.46 | 145.70 / 145.93 | -4.41 (-2.9%) | 83.29 / 83.81 | 83.39 / 82.16 | -0.77 (-0.9%) |
| 8 | 200.75 / 200.44 | 194.03 / 194.85 | -6.16 (-3.1%) | 119.32 / 121.21 | 117.95 / 117.47 | -2.56 (-2.1%) |

Gates:
- Block oracle L6..7: 20/20, B=16 49/49, B=32 97/97.
- Block oracle L0..1 `--route-check`:
  - B=16 46/49. The misses are slots 7 and 10, near-tie route flips (margin <= 2.4e-4).
  - B=32 96/97. The miss is slot 7, step 0.
  - These are the same misses as before this change (rounds 6 and 8).
- Op test: bitwise equal at B=1..16.
- Tests: devgen knob 18/18, plowrt `cuda,hsa` lib 2224/2224 (knob tests included), packet lib
  167/167.

Recipe serve:
- Protocol: the packed ship gate's (`plow_quality.sh` natural text, MTP k=3, 1K ISL / 256 OSL;
  c1 = 12 mixed-length prompts).
- Same plowrt build and same packet in both arms; only the decode interpreter cubins differ.
- Three reps, arms interleaved, order rotating by rep. Each cell is the mean of the three reps.

| | A tok/s/req | control tok/s/req | change | vLLM | A vs vLLM | A tick ms | control tick ms | accept len A / control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | 39.48 | 39.00 | +1.2% | 40.68 | -2.9% | 82.7 | 83.8 | 3.31 / 3.31 |
| c2 | 29.84 | 28.93 | +3.1% | 29.52 | +1.1% | 110.9 | 113.1 | 3.46 / 3.40 |
| c4 | 21.46 | 21.00 | +2.2% | 21.90 | -2.0% | 153.1 | 157.9 | 3.48 / 3.50 |
| c8 | 15.30 | 14.86 | +3.0% | 14.48 | +5.7% | 212.8 | 218.5 | 3.46 / 3.46 |

- The serve tick drops 1.1 / 2.2 / 4.8 / 5.7 ms at c1 / c2 / c4 / c8, in line with step_bench.
- TTFT, mean over the reps:
  - c1: 1.006 s (A) vs 0.977 s (control).
  - c8: 2.17 s (A) vs 1.90 s (control), with p50 / p90 equal (2.50 / 4.50 s) and 6 vs 7-8
    prefill ticks.
  - The change touches no prefill object, so this is a scheduling interaction. It has not been
    explained yet: an open item.

## Packed prefill (several prompts per launch)

`PLOW_EMIT_PACKED_PREFILL=1` packets (in the recipe, with the 1536 / 2048 rungs of
`PLOW_PF_LADDER_APPEND`) bind the packed request table to the
Mamba-2 prefill chain and the FA2 hd128 prefill. Each request's rows form their own 128-row chunks
at the offsets a launch of its own would use. Conv history, the chunk scan, and the final
state/history writes read and write the request's slot, and causal attention stops at the
segment. A one-request launch dispatches the serial kernel bodies. Decode rows do not ride: riders
are cc 9.0 only, and a riding Mamba row would need the decode step's state rounding.

Precision. The op oracle compares 8 packed requests (lens 1 / 2 / 130 / 7 / 300 / 127 / 129 / 33,
scrambled slots, random history/state) against each request alone: bit-exact, pad rows zero, 75/75.
The block oracle passes 20/20 on both packets. packed-check on L6..8 (Mamba + attention + MoE),
8 cases of 2-8 requests with off-bucket lengths, boundaries inside a chunk and prior KV: PASS, and
the recurrent state differs in 0 bytes. Logits differ only where the request alone ran the
128-row bucket (bucket-dependent GEMM arms); the worst is rel-L2 3.0e-3.

Block L6..8 (block_run packed-check, packed vs one launch per request):

| case | serial ms | packed ms |
|---|---:|---:|
| 8 requests, 1148 rows | 56.5 | 30.3 |
| 4 x 128 rows, 1K prior | 16.9 | 10.8 |
| 3 requests, 4200 rows, 2K prior | 64.4 | 53.5 |
| 4 requests, 1340 rows | 29.1 | 28.2 |

E4M3 KV (the recipe packet: `PLOW_FP8_KV`, 1536 / 2048 rungs, ladder 1..32). The recipe was built
through `campaign.py build` (full 256K packet plus an L6..8 block), and check_recipe on the bundle
passes with the emit knobs verified. Block oracle L6..8: 20/20. packed-check on the E4M3 block:
4/4 PASS, packed KV bytes bit-identical, 0 recurrent bytes differ, every request's activations
bit-identical (rel-L2 0):

| case (E4M3 KV) | serial ms | packed ms |
|---|---:|---:|
| 8 requests, 1148 rows | 57.2 | 17.0 |
| 4 x 128 rows, 1K prior | 16.4 | 13.1 |
| 3 requests, 4200 rows, 2K prior | 40.9 | 54.4 |
| 4 requests, 1340 rows | 28.8 | 18.0 |

At 4200 rows the pack takes the 8192 rung. Serially, the 2000 / 1500 / 700 requests run on the
2048 / 1536 / 1024 rungs, so the pack loses.

Scheduling. ~0.34 s of a 0.57 s 1024-row launch is fixed (the experts stream at any row count),
which sets two rules. With more than one request waiting, each joins the pack with its whole remainder:
the c8 burst of 7 prompts runs as one 7406-row launch (2.35 s) instead of 7 x 128 rows then 7 x 930.
With decoders live, further packs run in the same tick while it stays within one widest launch's
measured time, as the serial chunk loop does: one pack per tick had put a decode step between a lone
prompt's [128, 1024] launches (prefill 1.07-1.10 s vs the serial 0.90 s).

One request on the packed packet. A one-request pack runs the serial packet's own objects and
tensor table with the head tail Nop'ed for the compact terminal. The 1536 / 2048 rungs take the
per-tensor cuBLASLt FP8 rows, and emit fails closed if a rung falls off that route. Four-arm
nv-block-ab, L6..8, control binaries + serial packet vs treatment + packed packet:

| block (L6..8), prefill rows | block time | roofline floor | % of floor | delta vs serial | verdict |
|---|---:|---:|---:|---:|---|
| 1 x 128 | 4.240 ms | 7.862 ms | 185% | -0.6% | NOISE |
| 1 x 1024 | 10.537 ms | 11.735 ms | 111% | +2.5% | NOISE |
| 1 x 4096 | 31.261 ms | 27.919 ms | 89% | +0.4% | UNCONVICTABLE |
| 1 x 8192 | 62.577 ms | 53.041 ms | 85% | +0.9% | NOISE |

certify_kernel.py: QUALIFIED. In the round before the serial pair, 4096 and 8192 read +3.0% and
+2.7% REGRESSION (the packed objects' Mamba segment +0.27 ms and MoE segment +0.4 ms). Block
oracle L6..7 / L6..8: 20/20. packed-check (8 cases): PASS, 0 recurrent bytes differ. Lean
(`--lean-qualify report`): ordering certificates for all 12 programs (10 + the 2 appended rungs).
Bundle qualification is pending only on verifier approval and the MTP tail's unaudited `RowGather`
effects, as in the serial packet.

MTP drafter rows at request boundaries. The prefill tail stores the drafter K/V of row r from
`(h_r, embed(id[r + 1]))`. In a pack, a request's last row took the NEXT request's first token. A
lone launch took a pad row (id 0) or, when the chunk exactly filled its bucket, its own token. The
runtime now stages the shift per launch (`upload_mtp_shift`): next row inside a request (and inside
a rider's verify span), own token at a request's last row and on pad rows, in serial and packed
launches alike. Check: step_bench, full packet, 8 requests, MTP k=3, `kv.mtp.{k,v}` dumps plus a
drafts digest:

* Before the fix, serial vs packed: the drafter K/V differed at exactly the last prompt row of
  slots 0-6.
* After the fix: serial, lone (`--pf-width 1`) and packed are bit-identical over every prompt row.
  Output and accepted drafts match (74 / 867), but the packed drafts digest differs from serial / lone.
* A mid-prompt split (1500 = 1000 + 500 rows): lone and packed are bit-identical in the drafter
  K/V, drafts and output. In both, step_bench's spec-vs-greedy identity check reports a mismatch
  (accepted 37 / 974). No serial arm ran at this split.

Serving: same protocol as the spec3 table. Control = serial packet and binaries; packed = packed
packet (1536 / 2048 rungs) + runtime; both trees from one snapshot. Ladder 1..32, 32K context,
bf16 KV, `fp8_decode_tc: 4`. c2-c8: 3 interleaved reps; c1: 2 reps of 12 requests. vLLM: MTP k=3
from the arms above. Before the drafter-row fix:

| | control tok/s/req | packed tok/s/req | vLLM | control TTFT | packed TTFT | vLLM TTFT | TTFT p90 control / packed | accept len control / packed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | 40.72 | 40.39 | 40.68 | 1.21 s | 1.15 s | 1.24 s | 2.28 / 2.14 s | 3.41 / 3.38 |
| c2 | 27.59 | 28.29 | 29.52 | 1.54 s | 1.33 s | 1.74 s | 2.35 / 2.30 s | 3.33 / 3.35 |
| c4 | 18.87 | 19.55 | 21.90 | 1.84 s | 1.53 s | 2.07 s | 4.00 / 4.00 s | 3.52 / 3.38 |
| c8 | 12.32 | 13.58 | 14.48 | 2.95 s | 2.15 s | 2.90 s | 7.33 / 4.43 s | 3.51 / 3.35 |

* Mean TTFT is lower at every concurrency (c8 -27%, below vLLM's 2.90 s), and c8 p90 drops from
  7.3 s to 4.4 s.
* c1: -0.8% tok/s/req. The decode tick is identical (82.8 ms), so the gap comes from acceptance
  (3.41 vs 3.38).
* c4/c8: tok/s/req +3.6% / +10.2%, despite lower acceptance (3.38 / 3.35 vs 3.52 / 3.51). The drafter-row fix above
  targets that acceptance gap.

After the drafter-row fix: same packets, both binaries rebuilt, c4 / c8, 2 interleaved reps (means):

| | control tok/s/req | packed tok/s/req | vLLM | control TTFT | packed TTFT | vLLM TTFT | TTFT p90 control / packed | accept len control / packed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c4 | 18.57 | 19.87 | 21.90 | 1.93 s | 1.56 s | 2.07 s | 4.00 / 4.00 s | 3.49 / 3.46 |
| c8 | 12.15 | 13.88 | 14.48 | 2.87 s | 2.44 s | 2.90 s | 7.33 / 4.2 s | 3.51 / 3.51 |

* Packed acceptance is back to the control's: c8 3.51 vs 3.51, c4 3.46 vs 3.49 (was 3.35 / 3.38).
* tok/s/req: packed +7.0% at c4 and +14.2% at c8 vs control. Still -9.3% / -4.1% vs vLLM.
* Packed mean TTFT is 2.44 s at c8 (reps 2.55 / 2.34), vs 2.15 s in the 3-rep round above.

The shipped recipe packet: built by `campaign.py build` from the recipe (256K, E4M3 KV, packed
prefill, 1536 / 2048 rungs, ladder 1..32, `fp8_decode_tc: 4`), on the treatment binary with the
drafter-row fix. Same protocol as above, 2 reps, `PLOW_VMM_LIVE=1`. vLLM: MTP k=3, fp8 KV
(spec3 table).

| | tok/s/req | vLLM | vs vLLM | TTFT | vLLM TTFT | TTFT p90 | accept len | vLLM accept len | tick ms (live = c) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | 38.63 | 40.68 | -5.0% | 1.13 s | 1.24 s | 2.17 s | 3.31 | 3.39 | 84.5 |
| c2 | 28.78 | 29.52 | -2.5% | 1.45 s | 1.74 s | 2.30 s | 3.43 | 3.38 | 114.8 |
| c4 | 19.97 | 21.90 | -8.8% | 1.56 s | 2.07 s | 4.00 / 2.30 s | 3.48 | 3.43 | 161.8 |
| c8 | 14.03 | 14.48 | -3.1% | 2.32 s | 2.90 s | 4.33 s | 3.50 | 3.40 | 227.3 |

* TTFT is below vLLM at every concurrency (-9% / -17% / -25% / -20%).
* tok/s/req is still below vLLM at every concurrency: -2.5% to -8.8%.
* c1 loses on acceptance: 3.31 in both reps (deterministic), vs 3.38-3.41 for the bf16-KV arms and
  vLLM's 3.39.
* c2-c8 accept at or above vLLM. The remaining gap is the decode tick (c4 161.8 ms, c8 227.3 ms).


## Decode riders (mixed step)

Decode rows ride the packed Mamba prefill launch as the unified token batch's verify spans
(`exec/gpu/span_riders.rs`). A rider is 1..k+1 rows placed after the prompt rows, so a decoder
stalled by a prefill launch still verifies its MTP drafts in it.

* Row-wise ops (norms, FP8 projections, router, NVFP4 MoE, terminal head) take the rider rows as
  extra rows of the launch.
* Between prefill segments, each layer runs the widest decode rung's own instructions on the
  row-slot interpreter:
  * Mamba layers: span `Mamba2Conv` + `Mamba2Step`, then the bucket's `Mamba2GatedNorm`.
  * Attention layers: `FlashDecode` + `FlashMerge`, carrying the split count and row extent of
    the rung that a verify launch of the same rows would select.
* The rider Mamba state follows verify-span arithmetic: no state write, step records, deferred
  replay.
* Off Hopper the token batch loads only with span riders. Qwen3.8 GDN and sm_120 hd256 packets
  keep the separate decode step.
* The MTP drafter shift is staged per launch, for prefill buckets and decode rungs alike. Inside a
  span, row r reads row r + 1; a span's last row and every plain-step row read their own token.
  Lone vs packed prefill: drafter KV and `draft_fnv` are bit-identical.

Precision: riders use W8A8 / W4A4, the peer's arithmetic for every row of a mixed batch (see the
precision table). Compared with the decode rungs' W8A16 / W4A16 on natural text, rider KL is about
0.5% of the step distribution's entropy:

* 2.6e-5 at H = 0; 2.5e-2 at H = 5.3.
* Rows 1..3 of a span match row 0 at equal entropy.
* Top-1 agreement is 27/28 at k=3, w=7.

Rider logits are bit-identical across prefill salt, launch rows (896 vs 640) and rider-batch
width. Non-rider launches are unchanged: the spec gates give the same `ref_fnv` / `spec_fnv` as the
previous binary in 6/6 cases, block oracle 20/20 on L6..7 and L6..8, packed-check 8/8.

### Cost

Launch cost with 7 riders x 4 rows (k=3), ctx 1K, `step_bench --ride`. Within a bucket riders add
about 70-130 ms (the 1536 row is a spill). The decode tick they replace is about 207 ms.

| prompt rows | alone | + 28 rider rows | decode tick alone |
|---:|---:|---:|---:|
| 128 | 281 ms | 564 ms (spills to 256) | 207 ms |
| 256 | 504 ms | 583 ms | |
| 512 | 527 ms | 594 ms | |
| 1024 | 559 ms | 740 ms (spills to 1536) | |
| 1536 | 705 ms | 833 ms (spills to 2048) | |
| 2048 | 795 ms | 1247 ms (spills to 4096) | |

* The serve planner reserves k + 1 rows per rider and trims the prompt so that the riders never
  spill a bucket.
* Any launch above 128 rows costs at least about 500 ms (the expert-weight stream) and gives
  decoders one tick. Cost per prompt row falls with launch size:

  | launch | ms per prompt row |
  |---:|---:|
  | 1024 | 0.55 |
  | 1536 | 0.46 |
  | 2048 | 0.39 |
  | 4096 | about 0.33 |

* Smaller slices therefore cut decoder progress as well as TTFT. Example: one 1058-row prompt.
  * Two 512-row slices take 1180 ms and give 2 decode ticks.
  * One 1536 launch takes 740 ms; with pure decode ticks after it, the same 1180 ms gives about
    3.1 ticks.
* Serve confirms this. With `PLOW_PF_ITL_BUDGET_MS=650` (the planner slices to fit 650 ms),
  compared with no budget:
  * c8: 12.83 tok/s/req, TTFT 2.65 s.
  * c4: 19.43 tok/s/req, TTFT 1.80 s.
  * c2: 27.02 tok/s/req.
* The remaining prefill stall is the launch floor itself, not scheduling.

### Serve: riders vs packed

Same packed packet and natural-text MTP k=3 protocol. pk is the packed binary without riders; mx
adds riders. 3 interleaved reps with arm order alternated. vLLM is MTP k=3.

| | pk tok/s/req | riders tok/s/req | riders vs pk | vLLM | riders vs vLLM | pk TTFT | riders TTFT | accept len pk / riders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | 39.26 | 39.45 | +0.5% | 40.68 | -3.0% | 1.12 s | 1.15 s | 3.38 / 3.40 |
| c2 | 27.56 | 27.80 | +0.9% | 29.52 | -5.8% | 1.22 s | 1.08 s | 3.35 / 3.40 |
| c4 | 19.18 | 19.34 | +0.8% | 21.90 | -11.7% | 1.50 s | 1.63 s | 3.38 / 3.40 |
| c8 | 13.28 | 13.85 | +4.3% | 14.48 | -4.4% | 2.15 s | 2.10 s | 3.35 / 3.50 |

* Only c8 is outside run-to-run noise: the lowest riders rep, 13.59, beats the highest pk rep,
  13.33.
* At c8, riders remove the k=1 ticks around prefill (accept 3.35 → 3.50), and each riding launch
  yields 23-28 tokens. This is worth about 130 ms per prefill event: a 1536-row launch takes 740 ms
  with riders, against 690 ms plus a 175 ms decode tick without them.
* c2-c4 gains are within noise. c4 TTFT is 0.13 s higher (cause not isolated); c2 TTFT is 0.14 s
  lower.

## Serve tick overhead

Question: why the recipe serve's spec tick (c1/c2/c4/c8 84.5 / 114.2 / 160.7 / 224.1 ms) sits 3-24 ms
above the `step_bench --spec mtp` tick at the same batch (81.5 / 104.5 / 150.0 / 200.8 ms).

Method. With `PLOW_PF_PACKLOG=1`, every speculative tick writes a `PACKLOG SPEC` line:
- device time of the draft launch plus its read-back, and of the verify uploads, launch, accept
  kernel and read-backs (CUDA events);
- host time around each, the commit, and the emit loop;
- the verify launch's rung rows.

`PLOW_HOST_TIMING=1` adds the tick body's host time outside driver waits. Setup: dec8 recipe
packet, `r1_serve.sh` protocol, one rep, jg02, no compiler running. Columns are ms per tick, ticks
with every request decoding.

| | serve tick | draft GPU | verify GPU | host, all (HOSTT) | gap between ticks | verify rows / rung | step_bench raw, 96 steps | raw, 256 steps | serve's chat prompts, 255 steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | 85.15 | 10.50 | 74.56 | 0.10 | 0.04 | 4.0 / 4 | 81.5 | 82.4 | 83.8 |
| c2 | 114.35 | 12.13 | 102.12 | 0.10 | 0.04 | 7.9 / 8 | 104.5 | 109.1 | 111.7 |
| c4 | 160.35 | 13.95 | 146.29 | 0.18 | 0.04 | 15.9 / 16 | 150.0 | 150.5 | 151.8 |
| c8 | 224.32 | 16.73 | 207.45 | 0.14 | 0.04 | 32.0 / 32 | 200.8 | 209.5 | 209.2 |

Host work is 0.1-0.2 ms per tick, and the GPU is idle for no more than that. This covers
scheduling, host accept, commit, the hidden save, emit (detokenize and send), MTP shift staging,
and both synchronizations with their D2H copies. Other serve-only factors are absent:
- Every verify runs the same rung step_bench runs, with no padding (rows = 4 x live).
- Adaptive k stays at 3: drafts 578, draft tokens 1722 at c4.
- Prefill ticks are excluded from the tick column.

So serving adds no host overhead worth removing, and no runtime change was made.

The gap is in the verify launch's device time, and it follows the decoded content:
- step_bench's 96-step raw-text window under-counts it. At 256 steps the tick grows by +0.9 / +4.6
  / +0.5 / +8.7 ms. On the serve's own chat prompts at 255 steps it grows by +2.3 / +7.2 / +1.8 /
  +8.4 ms.
- The rest (+1.4 / +2.7 / +8.6 / +15.1 ms at c1 / c2 / c4 / c8) is also verify device time. Serve's
  token streams are not step_bench's: the second wave's prompts, prefill numerics, staggered
  starts.
- Verify device ms by sixth of each sustained serve run:
  - c4: 129.5 → 152.1 (wave 1); 146.6-151.5 (wave 2).
  - c8: 151.9 → 217.4 (wave 1); 195.6-218.7 (wave 2).

Routed decode MoE against its byte floor, per verify tick. Setup: `-DPLOW_NV_TRACE=2` interp and
`_rs` cubins. The routed MoE packets carry the item's distinct-expert count U (trace builds only).
Routed MoE ms = Σ over the 80 routed up / down instructions of (last signal end - first gate end).
Floor = U x 1.548 MB (2688 x 1024 NVFP4 plus e4m3 scales, per op) / 241 GB/s. U and ms are means
over ticks 4 / 20 / 40 / 60 of one step_bench run each.

| B (rows) | content | U per layer | routed MoE ms | floor ms | % of floor |
|---|---|---:|---:|---:|---:|
| 1 (4) | raw | 53-68 | 34.9-43.6 | 27.4-34.7 | 78-80% |
| 1 (4) | chat reasoning | 61-70 | 39.2-44.2 | 31.4-36.0 | 79-82% |
| 2 (8) | raw | 100-112 | 62.2-67.7 | 51.2-57.3 | 82-85% |
| 2 (8) | chat reasoning | 74-116 | 47.1-70.0 | 38.2-59.7 | 81-85% |
| 4 (16) | raw | 180-190 | 105.5-111.0 | 92.5-97.5 | 88% |
| 4 (16) | chat reasoning | 88 (tick 4), 176-190 | 54.4, 103.5-110.0 | 45.0, 90.4-97.5 | 83%, 87-89% |
| 8 (32) | raw | 276-289 | 159.0-166.2 | 141.9-148.3 | 89% |
| 8 (32) | chat reasoning | 99 (tick 4), 263-289 | 63.1, 152.5-165.8 | 50.8, 135.4-148.3 | 81%, 89% |

- The routed MoE is 50-76% of the verify launch, and its efficiency does not fall as the experts
  spread. It holds at 87-89% of the byte floor at B=4/8 whatever U is. It is lowest (78-83%) when
  U is small: B=1, and the first chat ticks, where every slot is still writing the same reasoning
  preamble.
- A slower tick is more distinct experts, so more bytes: from tick 4 to tick 20, chat B=8 goes
  from U 99 to 263 and from 114.7 to 204.5 ms per launch. vLLM streams the same experts.
- The levers are the per-byte efficiency at small U (B=1/2: 78-85%) and work outside the routed
  MoE, not the serve host path.

Gates: plowrt lib tests on the instrumented tree pass for `token_batch`, `spec`, `decode_rung` and
`sched`. `mux` has 6 failures, all in text-stream tests (stop prefix, UTF-8 tail, reply cap, parked
consumer) that never reach the CUDA spec tick, the only code the instrumentation touches. The
instrumentation is gated on `PLOW_PF_PACKLOG` / `PLOW_NV_TRACE`. It does not change any launch.

### Serve c4 levers

Measured on jg02. `nvidia-smi` sampled every 100 ms, each arm started at 58 °C or below, no
compiler running during any run. Packet: dec8 full-N; plowrt: the `PACKLOG SPEC` build.

Clock and power during the decode window:

| | SM MHz | W | °C | throttle reasons | verify ms by sixth | corr(clock, verify) |
|---|---:|---:|---:|---|---|---:|
| plow c4 | 2521 | 49.5 | 53→69 | none | 137.9 → 152.3 | -0.27 |
| plow c8 | 2512 | 53.7 | 54→72 | none | 180.6 → 219.4 | 0.08 |
| vLLM c4 (MTP k=3) | 2510 | 43.3 | 58→70 | none | | |

- Both engines run at the same ~2.5 GHz (the 3.0 GHz maximum is never reached under load). vLLM's
  SW-power-cap samples all fall in its start-up JIT.
- Plow draws 6-10 W more at the same clock, but nothing is capped. The decode gate wait already
  sleeps (`__nanosleep(64)`), so no backoff change was made.
- The tick's rise within a run (c4 +10%, c8 +21%) comes with a flat clock. It is the
  distinct-expert growth described under "Serve tick overhead".
- On the same box and in the same job, vLLM c4 gave 20.91 tok/s/req and plow 20.11.

Prefill event cost. The chat prompts are 1058 tokens. With 3-7 riders a launch has 1070-1086 rows,
and on the 1024/1536 ladder it pads to 1536.

Two new rungs, `PLOW_PF_LADDER_APPEND = "1152,1536,2048,2176"`:
- 1152 holds one prompt plus its riders.
- 2176 holds two prompts plus their riders. Before, these went to 4096, or split as a lone 128-row
  launch plus a 2048 launch at c8.

Both rungs join `LT_FP8_TENSOR_ROWS`. The per-tensor FP8 cuBLASLt route fails closed: the emitter
refuses any rung outside it. A plowrt without the new rows also refuses the packet ("invalid
packet-declared projection segments"), so the packet and the runtime ship together.

| L6..8 block, packed launch (`block_run prefill-bench`) | 1024 | 1152 | 1536 | 2048 | 2176 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| ms | 9.94 | 10.76 | 13.40 | 15.09 | 16.54 | 28.95 |

- 1058 rows: 10.76 ms in the 1152 rung against 13.40 ms in 1536 (-20%).
- 2116 rows: 16.54 ms in 2176 against 28.95 ms in 4096 (-43%).

At serve the full model's wave-2 prefill launch (one prompt, 3 riders) drops from 680-690 ms to
587-595 ms at c4, and at c8 (6-7 riders) from 707-722 ms to 598-619 ms.

Gates: `check_recipe` OK. Block oracle L6..8: 20/20. packed-check: mix8, big3, mix4, `1058+60`
(1152) and `1058+1058+40` (2176) all `parity=PASS` with packed KV bit-identical. q4 is
`parity=PASS` with request 2's KV off by rel 6e-4, the same as the ship gate (a3-r10). Lib tests:
plow-asset and devgen route tests pass.

Admission alignment. Wave 1 is admitted one prompt per tick because the decode admission rung
starts at 1:
- the cold-start path seats the first arrival;
- the rung controller raises the limit to 4 only after the first launch, once it sees `queued=3`;
- `max_hold_ms` never fires: `plowrt_hold_count_total` is 0 in every serve.

The rest of the burst reaches the mux 2-210 ms after the first prompt. It is still in its handlers
(template and tokenize, 75-240 ms pre-submit; c8's first request took 714 ms). The page-reclaim
jitter on this box (kswapd active, 3 GB swap in use) moves both the arrivals and slot admission
(`BEGIN seq_us` 3-460 ms).

Candidate (not applied, measurement pending):
- In the cold-start branch, the mux waits for requests still in a handler, bounded by one SLO.
- The chat handler now counts its ingress from handler entry instead of after the template.
- `PACKLOG COLD` logs the wait.

Serve, `r1_serve.sh` protocol on jg02:
- K: the shared-tree plowrt with the control packet (the same tree, ladder 1536,2048).
- A: the c4 plowrt with the 1152/2176 packet. In these reps the cold-start wait never fired, so A
  measures the rungs alone.

Means over the reps run:

| | reps | K tok/s/req | A tok/s/req | A vs K | K TTFT | A TTFT | vLLM (jg01 ref) |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | 1 | 38.75 | 38.77 | 0.0% | 1.03 s | 1.03 s | 40.68 |
| c2 | 1 | 28.58 | 28.84 | +0.9% | 1.14 s | 1.04 s | 29.52 |
| c4 | 2 | 19.94 | 20.38 | +2.2% | 1.74 s | 1.41 s | 21.90 |
| c8 | 1-2 | 14.21 | 14.17 | -0.3% | 2.65 s | 1.78 s | 14.48 |

- Run-to-run spread is ±0.3 tok/s/req, so the decode gains are at the noise level.
- TTFT falls 0.1-0.9 s.
- Once per server lifetime, the first launch at a new rung key with decoders live runs unridden.
  `RideCost` needs a pure sample, so riders=0 on the wave-1 1024/2176 launches. This costs about
  1 s of stall at c4 in these short runs.
- The remaining c4 gap to vLLM on the same box (20.4 vs 20.9) is the steady verify tick
  (158-160 ms) and the wave-2 prefill events (4 per run, 3 decoders each, ~590 ms against vLLM's
  mixed chunk).

## Single-SM inner loops (GB10)

Each innermost device op runs on ONE SM (gridDim=1) against a single-SM roofline.

Harness: `runtime/ubench/run_ubench_cu.sh --nv` (`runtime/ubench/README.md`).
- It drives the real op bodies through the interpreter's `plow_exec`, on the real instructions of
  `model.pkt`.
- They are built with the packet's `plow_config.h` and the object's full nvcc define list: the CMake
  axis defines plus the manifest's `PLOW_EXTRA_DEFINES`. The script refuses to build unless that
  list equals the production command in the emit's `.cubin-build`.
- Decode uses `interp_sm120_rs`, prefill `interp_sm120_pf`.

Per case:
- A per-SM slice is slice s of the instruction's `blocks`, with L2 flushed before each launch.
  Time is clock64 cycles scaled by an FFMA-calibrated clock; each slice takes the median of 7 reps.
- The check compares one SM running all slices against the full-grid output, from the same input
  snapshot. The result is bitwise, or nondet when two full-grid runs already differ.

Roofline columns:
- Bytes and FLOPs come from `scripts/bench/op_roof.py --ubench`.
- Gap = n × (max(slice, full-GPU floor) − full-GPU floor), where full-GPU floor =
  max(bytes / 238 GB/s, flops / chip peak) and n = the instances in the program.
- Run: `ub-j4`, 2026-11-25, `compilers_during_run=0`. All cases are bitwise except those marked.

### Single-SM ceilings (2.58 GHz; `hwspec::GB10_SM_CEILINGS`)

| | one SM | ×48 | whole GPU measured |
|---|---:|---:|---:|
| mma.sync bf16 m16n8k16 | 2.640 TF (1024 FLOP/clk) | 126.7 | |
| mma.sync e4m3 m16n8k32 (and kind::f8f6f4, MXFP8) | 5.283 TF | 253.4 | |
| mma.sync NVFP4 mxf4nvf4 m16n8k64 4X ue4m3 (MXFP4 same) | 10.556 TF | 506.7 | |
| FP32 FFMA | 0.645 TF | 30.9 | 30.9 |
| DRAM, ld.global.v8 / v4 / cp.async / cp.async.bulk | 78.1 / 77.8 / 73.4 / 77.1 GB/s | | 234-239 GB/s (4.9/SM) |
| L2 hit | 82.6 GB/s | | 2159 GB/s |
| shared memory | 329.1 GB/s (127.6 B/clk) | 15.8 TB/s | |

The SM's load path (~32 B/clk) bounds a single SM, so one SM can pull a third of the chip's DRAM
bandwidth. Any stream op whose per-SM inner loop runs below ~60 GB/s holds SMs that a faster loop
would hand back for overlap.

### Single-SM inner loops: NVFP4 MMA

`runtime/ubench/fp4_tile_sm120.cu` (`run_ubench_cu.sh --nv --fp4-tile`) runs the W4A4 walk's
k-loop in isolation at gridDim=1: `mma.sync` m16n8k64 `kind::mxf4nvf4` 4X ue4m3 (`OMMA.SF.16864`),
fragments by `ldmatrix` from the production stage layout. Run `j7`, nk = 1024 k128 steps, best of 9.
Modes are listed in `runtime/ubench/README.md`.

Tensor pipe and resident smem:

| loop | TF/s per SM | % of 10.556 |
|---|---:|---:|
| one warp, 2 to 32 accumulators | 2.60-2.64 | 25% (one SMSP) |
| two warps, 32 accumulators each | 5.27 | 50% |
| 128x256 / 128x192 / 128x128 / 64x256, 8 warps, registers only | 10.55-10.56 | 100% |
| same, fragments from one resident stage | 10.54-10.56 | 100% |
| same, plus a block barrier per k128 | 9.87-10.18 | 94-96% |

`OMMA.SF.16864` issues once per 16 clk per SMSP; two accumulators per warp saturate it.

Feeding the ring (3 stages of k128 unless noted). Layouts: `up` = routed up (A reused, weights
K/2 = 512 B per row), `down` = routed down (both streamed, 1344 B), `sdown` = shared down (2688 B),
`tiled` = a contiguous block per stage, `l2` = both operands L2-resident.

| feed | 128x256 | 128x192 | 128x128 |
|---|---|---|---|
| `cp.async`, all threads (production) | up 63%, down 47%, sdown 49%, l2 63% | up 57%, down 41% | |
| `cp.async`, staging only (no MMA) | up 23 B/clk, down 12 B/clk | | |
| `cp.async`, no 8 B scale copies | up 70%, down 67% | up 61%, down 60% | |
| one-thread `cp.async.bulk`, pre-tiled | tiled 79% | tiled 77% | tiled 72% |
| TMA 2-D boxes, 64 B swizzle, row-major | up 73%, down 55%, sdown 59%, l2 73% | up 68%, down 50%, sdown 57% | down 61% |

* The production feed loses to LSU/MIO contention: eight MMA warps' LDGSTS against their LDSM/LDS.
  L2-resident operands are no faster than DRAM, and dropping the scale copies alone lifts it most.
* One thread issuing async-proxy copies removes that contention. TMA 2-D needs no pre-tiling, but
  its 64 B rows strided by 1344-2688 B fetch worse than 512 B rows. Four stages, software
  pipelining and L2 prefetch move nothing. 64 B per-row bulk copies reach 4-13%.

Candidate, not landed: the same TMA feed in the walks (`nf4v_tiled`, packet-encoded
`GEN_TMAP_U8_SW64` maps in `i7`). A and B land as 64-row / BN-row boxes on a per-slot mbarrier;
scales stay on `cp.async` (their 168 B row stride at K = 2688 breaks TMA's 16 B stride rule). The
routed up gathers A and keeps `cp.async`. Dedicated-entry codegen, layer-1 real text, run `g2`:

| walk | cp.async, 1 SM | TMA, 1 SM | full GPU 1024 / 4096 / 8192 rows |
|---|---:|---:|---|
| routed down | 4.11 TF (39%) | 4.39 TF (42%) | -20 / -70 / -201 us |
| shared up | 5.14 TF (49%) | 5.81 TF (55%) | -49 / -57 / -115 us |
| shared down | 5.07 TF (48%) | 5.49 TF (52%) | -28 / -71 / -166 us |

Outputs (`part`, `sout`, `sfq`) stay bitwise. TTFT is 1.5-2.5% lower at 8K / 32K / 128K (`g1`, ABBA).
The four-arm L1 block A/B still fails at small chunks: 128 rows +0.35 ms and 1024 rows +0.11 ms
(REGRESSION), 8192 rows -0.61 ms and 16384 rows -1.60 ms (PASS). With a few rows per expert, the
64-row A box reads rows the `cp.async` feed zero-fills without reading.

### Nemotron prefill (8192-row chunk), ranked by gap

| op | bound | % of 1-SM roofline | slice µs | per SM | n | gap ms / chunk |
|---|---|---:|---:|---:|---:|---:|
| `MoeUpNvfp4Pf` routed (W4A4) | mma fp4 | 31% | 6229 | 3.32 TF | 40 | 69.3 |
| `MoeDownNvfp4Pf` shared | mma fp4 | 31% | 2299 | 3.27 TF | 40 | 53.9 |
| `MoeUpNvfp4Pf` shared | mma fp4 | 38% | 1858 | 4.05 TF | 40 | 36.2 |
| `MoeDownNvfp4Pf` routed | mma fp4 | 29% | 6832 | 3.03 TF | 40 | 32.2 |
| `Mamba2ChunkOut` | mma bf16 | 9% | 3117 | 0.24 TF | 40 | 27.4 |
| `Gemm` bf16 fc1 latent | mma bf16 | 49% | 1110 | 1.29 TF | 40 | 15.1 |
| `MoeRouterLogitsF32` | mma bf16 | 39% | 688 | 1.04 TF | 40 | 12.7 |
| `MoeRouterSigmoidTopk` | latency | 2% | 295 | 1.3 GB/s | 40 | 8.8 |
| `FlashPrefill` hd128 E4M3 | mma | 66% | 6594 | 1.74 TF | 8 | 5.9 |
| `MoeAlignNvfp4` (nondet) | latency | 1% | 79 | 1.0 GB/s | 40 | 2.6 |
| `QwenHeadNormRope` (nondet) | latency | 3% | 68 | 2.6 GB/s | 16 | 0.5 |
| `Mamba2ChunkState` (+ fused pass) | mma bf16 | 8% | 1654 | 0.22 TF | 40 | 0 |
| `Mamba2Conv` | dram | 10% | 896 | 7.8 GB/s | 40 | 0 |
| `Mamba2GatedNorm` → fp8 | dram | 23% | 462 | 18.1 GB/s | 40 | 0 |
| `MoeCombinePf` | dram | 14% | 723 | 11.2 GB/s | 40 | 0 |
| `MoeQuantNvfp4` / `QuantFp8` | dram | 19% / 14% | 30 / 188 | 14.9 / 11.2 GB/s | 40 / 56 | 0 |
| `QwenRmsNorm` / `Embed` | dram | 17% / 15% | 216 / 233 | 13.0 / 12.0 GB/s | 4 / 2 | 0 |

The ranked gaps sum to ~265 ms per 8192-row chunk. The W4A4 MoE walks account for 192 ms of it:
their FP4 MMA loop reaches 3-4 of the SM's 10.56 TF. `Mamba2ChunkOut` is next, at 9% of the BF16
MMA ceiling. The prefill FP8 projections run in cuBLASLt on this packet and are not measured here.

### Nemotron decode (MTP verify, T=4, ctx 1024 unless noted), ranked by gap

| op | bound | % of 1-SM roofline | slice µs | GB/s per SM | n | gap ms / tick |
|---|---|---:|---:|---:|---:|---:|
| `MoeRouterSigmoidTopk` | latency | 0% | 14.6 (mean 2.7) | 0.2 | 40 | 0.56 |
| `Mamba2Conv` span | latency | 2% | 9.2 | 1.7 | 40 | 0.24 |
| `Mamba2GatedNorm` | latency | 1% | 5.1 | 0.8 | 40 | 0.17 |
| `MoeCombinePf` | latency | 2% | 4.2 | 1.8 | 40 | 0.11 |
| `Residual` | latency | 1% | 1.8 | 1.1 | 41 | 0.06 |
| `QwenHeadNormRope` (nondet) | latency | 0% | 3.4 | 0.0 | 16 | 0.05 |
| `FlashMerge` | latency | 2% | 8.2 | 1.6 | 8 | 0.05 |
| `QwenRmsNorm` | latency | 2% | 9.5 | 1.7 | 4 | 0.04 |
| `FlashDecode` GQA-16 E4M3 GF-16, ctx 1k | latency | 5% | 12.8 | 3.5 | 8 | 0.03 |
| `FlashDecode` GQA-16 E4M3 GF-16, ctx 32k | dram | 19% | 96.4 | 14.5 | 8 | 0 |
| `GemvFp8` in_proj / out_proj | dram | 30% / 29% | 67.1 / 30.6 | 23.7 / 22.9 | 40 | 0 |
| `GemvFp8` lm_head | dram | 33% | 430 | 26.1 | 1 | 0 |
| `GemvFp8` attn q / o | dram | 16% / 20% | 27.9 / 22.7 | 12.6 / 15.5 | 8 | 0 |
| `Gemv` bf16 fc1 / fc2 latent | dram | 21% / 22% | 11.0 / 10.4 | 16.0 / 16.9 | 40 | 0 |
| `MoeUpNvfp4` / `MoeDownNvfp4` routed (83 experts) | dram | 15% / 15% | 226 / 236 | 11.9 / 11.4 | 40 | 0 |
| `MoeUpNvfp4` / `MoeDownNvfp4` shared | dram | 9% / 9% | 35.0 / 37.9 | 7.4 / 6.8 | 40 | 0 |
| `Mamba2Step` span / plain | dram | 19% / 9% | 23.2 / 53.1 | 15.2 / 6.6 | 1 | 0 |
| `MoeRouterLogitsF32` | dram | 16% | 7.4 | 12.2 | 40 | 0 |

At T=1 and T=16 (the same op set):
- `GemvFp8` in/out_proj run at 15-17% of the roofline; their T=4 shape is the fast one.
- The routed MoE walks run at 11-16%, the shared walks at 12-21%.
- `Mamba2Step` span runs at 9% at T=1 and 25% at T=16.
- `MoeRouterSigmoidTopk` costs 0.50 ms per tick at T=16.

Every decode stream op runs faster per slice than the full-GPU DRAM share it needs (4.9 GB/s per
SM), so its gap is 0. The decode gaps come from latency ops: about 1.3 ms per tick at ctx 1k.

The phase-2 lever is the stream inner loop itself. A GEMV or MoE walk at ~60-70 GB/s per SM would
finish the same tick on roughly a third of the SMs and free the rest to overlap the latency ops.

### Qwen3.8-27B FP8 (same harness, fresh emit from this tree)

Decode, T=4 rows (`interp_sm120_rs`), ranked by gap:

| op | bound | % of 1-SM roofline | slice µs | GB/s per SM | n | gap ms / step |
|---|---|---:|---:|---:|---:|---:|
| `FlashDecode` hd256 GQA-6, ctx 32k | dram | 4% | 1860 | 3.0 | 16 | 11.7 |
| `GemvFp8` mlp down | dram | 5% | 496 | 3.8 | 64 | 7.7 |
| `GemvFp8` GDN in_proj_qkv | dram | 5% | 285 | 3.8 | 48 | 3.1 |
| `FlashDecode` hd256, ctx 1k | dram | 4% | 57.2 | 3.1 | 16 | 0.35 |
| `QwenGdnConv` span | latency | 1% | 7.6 | 1.1 | 48 | 0.28 |
| `QwenGatedNorm` | latency | 1% | 5.6 | 0.6 | 48 | 0.24 |
| `QwenHeadNormRope` q | latency | 0% | 6.7 | 0.3 | 16 | 0.10 |
| `FlashMerge` hd256 | latency | 3% | 7.2 | 2.6 | 16 | 0.09 |
| `QwenGdnStep` span / plain | dram | 14% / 15% | 47.9 / 44.9 | 11.0 / 11.7 | 48 | 0 |

- `GemvFp8` in_proj_qkv runs at 15% at T=1 (92.5 µs). At T=4 and T=16 the slice is 3× and 4.3×
  slower, at 4-5%: 8.6 ms per step at T=16. The T>1 FP8 GEMV body on this object is the outlier.
  Check that this is the arm the served Qwen decode actually takes before tuning it.
- `QwenGdnStep` runs at 9% at T=1 and 15% at T=16.

Prefill, 8192-row chunk (`interp_sm120_pf`), ranked by gap:

| op | bound | % of 1-SM roofline | slice µs | per SM | n | gap ms / chunk |
|---|---|---:|---:|---:|---:|---:|
| `FlashPrefill` hd256 (bf16 KV) | mma bf16 | 15% | 43665 (mean 30068) | 0.39 TF | 16 | 558 |
| `QwenHeadNormRope` q | latency | 3% | 1592 | 2.6 GB/s | 16 | 11.9 |
| `QwenGdnGatePrep` | latency | 5% | 36 | 3.6 GB/s | 48 | 0.5 |
| `QwenGdnQkvPrep` / `QwenGdnConvPrefill` | dram | 36% / 18% | 252 / 490 | 27.8 / 14.3 GB/s | 48 | 0 |
| `Glu` / `QwenGatedNorm` | dram | 51% / 11% | 449 / 740 | 39.7 / 8.5 GB/s | 64 / 48 | 0 |
| `QwenRmsNorm` / `QuantFp8` | dram | 26% / 15% | 175 / 225 | 20.0 / 11.7 GB/s | 4 / 128 | 0 |

The interpreter's hd256 `FlashPrefill` is the single largest gap measured: 15% of the BF16 MMA
ceiling, with the causal slices unbalanced (max 1.45× mean). The GDN chunk scan runs in its own
cubin (`gdn_prefill_sm120`) and is not covered here.

Caveats:
- Routing is uniform random, so the expert counts are an upper bound.
- Slice timing sees no DRAM contention from other SMs.
- Prefill bodies run in the `interp_sm120_pf` object's `plow_exec`. A served chunk may run them on
  the segment entries (`pfseg`/`pfgemm`), whose register budgets differ.
- The gaps count the inner loop only, not the grid barrier between instructions.
- `QwenHeadNormRope` in the Nemotron packet is nondet: two full-grid runs differ.

## Single-SM inner loops: decode

Changes (outputs bitwise equal to the previous bodies at every T measured):
- **Router top-k** (`op_moe_nvfp4.cuh`): order-preserving u32 keys and warp `redux.sync` rounds
  (max key, then min expert id). Covers the decode E = 512 path and the prefill select. +0 / -0
  share a key; NaN and -inf are never picked, as before.
- **Mamba span** (`op_mamba_span.cuh`): span units are found in one windowed load round instead of
  a dependent row walk. Width-4 conv span body keeps the history in registers.
- **Attention** (`op_attention.cuh`):
  - `flash_merge` lets THREADS / D groups take one (batch, head) item each.
  - `fa_decode_mma16` `PLOW_NV_FA16_PIPE=2`: L2 prefetch of the next 16-key block, and the rescale
    is skipped while c = 1.
  - Mode 1 (register double buffer) is rejected, see below.
- **FP8 TC GEMV** (`op_gemv_fp8_tc.cuh`): `PLOW_NV_FP8TC_PF=1` prefetches the next chunks into L2.

Single SM (`run_ubench_cu.sh --nv`, run `ub3`, max slice µs, before → after). Each case's digest
(a fold of every full-grid output) is equal before and after at every T:

| op | T=1 | T=4 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| router_topk | 14.5 → 9.6 | 14.5 → 9.5 | 14.9 → 10.1 | 14.6 → 9.8 |
| mamba2_conv_span | 6.6 → 9.7 * | 9.4 → 5.3 | 22.6 → 9.9 | 34.7 → 13.7 |
| flash_decode_gqa16 1K | 8.0 → 10.3 * | 20.9 → 10.0 | | |
| flash_decode_gqa16 32K | 28.8 → 21.3 | 144 → 78.9 | | 672 → 574 |
| flash_merge 32K | 5.8 → 4.8 | 22.8 → 6.4 | | 61.5 → 35.7 |
| gemv_fp8 in_proj (TC from T=4) | 131 → 161 * | 67.2 → 65.3 | 200 → 74.2 | |
| gemv_fp8 out_proj | 59.6 → 63.9 * | 52.6 → 29.8 | 55.8 → 43.3 | |
| gemv_fp8 lm_head | 912 → 835 | 440 → 382 | | |

\* Each starred maximum comes from one outlier slice. The means across slices fall: conv span
5.90 → 4.77, flash_decode 1K 7.15 → 6.51, in_proj 123.0 → 121.8. out_proj holds at 57.3 → 57.5.

Full model (`step_bench`, recipe packet). Streams are bitwise equal in every pair: spec `spec_fnv`
and plain `--force-prompt` `TOK_STREAM`. Every run reported `compilers_during_run=0`. The 1K / 32K
rows were measured with attention mode 1, the 128K rows with mode 2.

| point | spec mtp k=3 tick, ms | plain, ms |
|---|---:|---:|
| 1K B=1 / 2 / 4 / 8 | -4.0% / -3.5% / -3.0% / -3.3% | +1.6% / +1.4% / -2.7% / -1.1% (mode 1) |
| 32K B=1 / 2 / 4 / 8 | -2.9% / -3.8% / -2.7% / -4.4% | +1.3% / +1.3% / +0.1% / -2.0% (mode 1) |
| 128K B=1 | 85.98 → 82.86 (-3.6%) | 50.06 → 50.27 |
| 128K B=2 | 118.58 → 114.14 (-3.7%) | 65.64 → 65.24 |
| 128K B=4 | 165.44 → 158.50 (-4.2%) | 91.06 → 87.84 (-3.5%) |
| 128K B=8 | 193.13 → 180.41 (-6.6%) | 106.80 → 103.02 (-3.5%) |

Plain B=1/2 regression and the attention mode:
- Mode 1 made plain B=1/2 slower. The cause was isolated one file at a time on round-to-nearest
  state packets. Plain B=1 1K: base 46.98, all four changes 47.45, all but attention 46.99 ms.
- Attention itself is not slower at T=1. Mode 1's 32 extra registers raise interpreter spills
  elsewhere: `interp_sm120_rs` LDL 528 → 887, stack 640 → 960 B.
- Recipe packet, ms per mode (0 / 1 / 2):

| point | mode 0 | mode 1 | mode 2 |
|---|---:|---:|---:|
| plain B=1 1K (base 47.37) | 47.37 | 48.65 | 47.43 |
| plain B=2 1K (base 60.41) | 60.41 | 60.87 | 60.38 |
| spec B=8 32K tick (base 196.85) | 188.84 | 189.09 | 187.32 |

- Mode 2 keeps the spills at 600 LDL.

Gates on the round-to-nearest-state packet:
- Span-verify identity 20/20 `SPEC_IDENTITY ok`. The recipe packet's stochastic-rounded decode
  state never passes identity, in either arm.
- Block oracle L6..7 `--kv fp8`: 20/20, B=16 49/49, B=32 97/97.
- Four-arm block A/B L6..7 1K: B=1..32 -0.3..-4.8%, PASS or NOISE, 83-92% of the floor.

### Qwen3.8-27B FP8: decode GEMVs on the tensor cores

The recipe (`scripts/campaign/recipes/qwen38-27b.gb10.fp8.toml`) ran every decode `GemvFp8` on the
FFMA W8A16 walk. Single SM (µs, T = 1 / 4 / 16):

| op | FFMA | TC from rung 2 | TC from rung 1 | TC rung 1 + L2 prefetch |
|---|---:|---:|---:|---:|
| in_proj_qkv | 96.9 / 285 / 402 | 92.5 / 61.9 / 91.4 | 48.7 / 58.2 / 90.0 | 41.3 / 43.3 / 57.1 |
| mlp_down | 591 / 496 / 673 | 592 / 93.9 / 152 | 74.8 / 97.0 / 156 | 55.5 / 61.2 / 100 |

Full model (`step_bench` plain `--force-prompt`, ctx 1024, ms/step, mean of 2 reps):

| B | FFMA | TC rung 1 | delta |
|---|---:|---:|---:|
| 1 | 137.0 | 123.2 | -10.1% |
| 2 | 129.4 | 125.7 | -2.9% |
| 4 | 141.9 | 130.3 | -8.2% |
| 8 | 257.2 | 139.3 | -45.8% |
| 16 | 238.6 | 157.4 | -34.0% |

Logit screen: same forced tokens, 64 steps, `step_bench --dump-step-logits`. An FFMA rerun is
bitwise. FFMA vs TC:

| B | positions | top-1 agree | KL mean | KL p95 | KL max |
|---|---:|---:|---:|---:|---:|
| 1 | 64 | 0.969 | 4.4e-4 | 1.8e-3 | 3.4e-3 |
| 8 | 512 | 0.998 | 6.8e-4 | 2.0e-3 | 2.4e-2 |
| 16 | 1024 | 0.995 | 6.6e-4 | 2.1e-3 | 2.4e-2 |

The recipe sets `PLOW_FP8_DECODE_TC = "1"`, pending `[gates.llm_fp32_ref]`: no FP32 reference or
vLLM capture exists for this checkpoint on GB10.

MTP span identity:
- B=1: holds.
- B=4: breaks. `d_gemv_fp8` (`op_gemm.cuh`) sends few-tile shapes at M ≤ 4 to the tile walk
  (K chunks interleaved over `ks` warps) and M = 5..32 to stream-K (contiguous K ranges per warp).
  The f32 accumulation order per row therefore depends on the row count.
- Spec serving is off in this recipe.


## Single-SM inner loops: hd256 attention

Qwen3.8-27B FP8 on GB10: the hd256 full-attention `FlashPrefill` (16 layers, bf16 KV), 24 q-heads /
4 KV heads. The change:
- New body `d_flash_pf256` (`op_flash_pf256.cuh`), FA2 form with split-D warp pairs. Each warp
  keeps Q and O for 16 rows x 128 dims in registers, and the pair swaps partial S through smem.
- K/V pass through a 2-stage cp.async ring of 32-key tiles.
- 64-row items run heaviest first.
- The emitter picks nsplit 1 with the bf16 epilogue fused (no `FlashMerge`) when q tiles x heads ≥
  SMs.
- Manifest tuning `fa2_pf256` selects the body (`PLOW_NV_FA2_PF256`); packed requests and encoder
  spans keep `d_flash_prefill`.

Before the change, the old body had to split each causal chunk into 2 parts. That gave 48 (head,
split) items on 48 SMs, so the slowest slice ran 1.45× the mean.

Block 2..6 (3 GDN, 4 dense FFN, 1 full attention), `campaign.py nv-block-ab --verb prefill-bench`,
8192 rows. Four frozen arms; GB10 calibrated ceilings (238 GB/s, 94 BF16 / 181 FP8 TF/s):

| block (rung, prior) | block time | roofline floor | % of floor | delta vs current | verdict |
|---|---:|---:|---:|---:|---|
| prefill 1x8192, prior 0 | 277.3 ms | 240.9 ms | 87% | -38.1 ms (-12.1%) | PASS |
| prefill 1x8192, prior 8192 | 304.0 ms | 258.5 ms | 85% | -76.1 ms (-20.0%) | PASS |
| prefill 1x8192, prior 24576 | 356.3 ms | 293.6 ms | 82% | -126.5 ms (-26.2%) | PASS |

The kernel alone, oracle `runtime/tests/flashpre_hd256_sm120.cu`:
- Full grid; ms at 126.7 TF bf16 MMA (48 x 2.64).
- Old = `d_flash_prefill<256,64,32>`.
- Rel-L2 is against an f64 reference and is the same for both bodies.

| shape | old | new | % of MMA peak (old → new) | rel-L2 |
|---|---:|---:|---|---:|
| T 8192, prior 0, nsplit 1 | 25.83 | 12.58 | 25% → 52% | 1.3e-3 |
| T 8192, prior 8192, nsplit 1 | 75.02 | 38.14 | 26% → 51% | 1.7e-3 |
| T 8192, prior 24576, nsplit 1 | 171.89 | 88.71 | 27% → 51% | 1.6e-3 |
| T 8192, prior 0, nsplit 2 + FlashMerge (previous packet shape) | 44.01 | 18.84 | 15% → 35% | 1.3e-3 |
| T 1024, prior 7168 | 5.78 | 2.83 | 26% → 54% | 1.7e-3 |
| T 128, prior 32768, nsplit 2 | 3.09 | 1.58 | 26% → 52% | 1.7e-3 |
| Gemma-4 sliding window 1024, 16/8 heads, T 4096, prior 4096 | 2.53 | 1.40 | 21% → 39% | 1.7e-3 |

Single SM (slice 0 of 48 at T 8192, prior 0): old 24.4 ms → new 11.6 ms. The full grid runs about 8%
slower than one SM alone.

Interpreter ubench (`qwen-pf`, `flash_prefill_hd256`, production defines):
- Slice max 43495 → 12130 µs; mean 30023 → 12001 µs.
- max/mean 1.45 → 1.01.
- 15% → 53% of the per-SM MMA ceiling.
- One SM matches the full grid bitwise.

Full model (`step_bench`, 1 slot, synthetic token prompts, 2 alternating reps,
`compilers_during_run=0`):
- TTFT 8K: 5.09 / 5.14 → 4.49 / 4.52 s (-12%).
- TTFT 32K: 26.07 / 26.14 → 20.59 / 20.61 s (-21%).
- Prefill logits vs control: KL 1.4e-3 at 8K, 1.1e-3 at 32K; same top-1; max |Δlogit| 0.30.
  Identical across reps.
- `FlashMerge` is gone from all four prefill buckets: 144 → 80 instructions, which are the 16 layers
  x 5 decode rungs.

Not applied:
- Four query rows per warp pass in `QwenHeadNormRope`: the slice went 1593 → 1758 µs.
- hd256 decode on the row-group mma item with a whole GQA group per item (GF 6). It is
  faster on the full grid at 32K, but:
  - B=4 at 1K regresses.
  - Its sm_12x staging variants trade one-SM rate (10-38 GB/s) against full-grid DRAM
    efficiency (185-232 GB/s).
  - One variant hangs in the interpreter ubench.
  - It stays a separate follow-up.

## Agentic / prefix cache

The VMM prefix cache now serves this packet (recipe `[serve.env]`: `PLOW_PREFIX_CACHE=1
PLOW_VMM_PREFIX=1`). `PLOW_VMM_LIVE=1` makes the auto mode decline it, so both are explicit. A
boundary holds:
- the 8 attention layers' E4M3 KV;
- the MTP drafter's own cache `kv.mtp.{k,v}`, one more pool track, so a hit drafts from the
  publisher's rows;
- the Mamba-2 conv and SSM state, about 82 MiB per boundary.

The packet's serve manifest carries no KV geometry (the `nemotron_h_omni` config nests the text
tower under `llm_config`). `KvGeometry::from_config` now reads `llm_config.layers_block_type`, and
the CUDA load falls back to the checkpoint config.

Recurrent state exists only at a slot's current row. A miss therefore captures its prompt end by
ending a prefill launch there, at `(prompt - 9) / 32 * 32`, then runs the tail as its own launch.
That launch reads every expert's weights, so it costs about 0.2 s (7.8K prompts: TTFT 2.15 s cut
against 1.94 s uncut). A miss cuts only when a later request is expected to attach its prompt end
(`VmmKv::prompt_end_expected`):
- the request attached a boundary,
- or it names a session (`X-Session-Id`) and is not the session's last turn (`X-Session-End`),
- or it continues a recent prompt: at least 256 common rows, ending within 64 rows of that prompt's
  end (a conversation's next turn, a retry).

Any other prompt end is captured only where a launch ends anyway.

### Numerics

gate2 compares 8 one-turn prompts of 7.8K tokens, c1, greedy with logprobs. Each server sends the
prompts in three passes: first-seen, resent (cut), resent again (hit). Spec on is the recipe's MTP
k=3 verify, whose fp16 Mamba state store rounds to nearest. Spec off decode rounds stochastically
with a server-wide step seed, so only token 0 is comparable there.

| comparison | spec on (64 tokens) | spec off (token 0) |
|---|---|---|
| first-seen miss (no cut) vs cache off | 8/8 bitwise | 8/8 bitwise |
| hit vs cut miss (state restore) | 8/8 bitwise | 8/8 bitwise |
| cut miss vs cache off | diverges; token 0 \|Δlogprob\| 0.0002-0.13 | same |
| cache off, same server, repeated | 7/8 bitwise (the server's first request differs) | token 0 8/8 |

- The restore is exact. The whole difference between cache on and cache off is the cut: the tail
  runs as a separate short launch and takes the 128-row bucket's GEMM arms. Packed-check measures
  the same bucket dependence at rel-L2 3.0e-3 ("Packed prefill"), well inside the block oracle's
  2e-2 limit.
- Putting the cut on the 128-row SSD chunk grid did not reduce the drift (0.0004-0.099 for on-grid
  cuts against 0.0002-0.13 off the grid). It raised hit TTFT from 207 to 250 ms (a longer tail),
  so boundaries stay on the 32-row grid.
- vLLM's `--mamba-cache-mode align` makes the same kind of cut: prefill chunks end at block
  boundaries so the fp16 state can be stored there. Its cache-on stream therefore differs from its
  cache-off stream in the same way.

### Serving

Agentic workloads, `llm_grid.sh --agentic` (same client both sides), MTP k=3, 2 reps per cell, each
workload started at 58 °C or below, jg02:
- a: c4 / c8 sessions x 8 turns, history growing to 12K, 128-token replies;
- b: shared 4K system prompt + 256 unique tokens, one turn, 4 sessions per worker.

vLLM: `--enable-mamba-shared-prefix-checkpoint --mamba-cache-mode align`, fp16 SSM state with
stochastic rounding, MTP k=3, TRITON_ATTN. TTFT p50 / p99 in s; hit = prefix-hit token fraction.

| | plow before (cache off) | plow after | vLLM |
|---|---|---|---|
| a c4 TTFT | 3.45 / 6.62 | 1.28 / 3.56 | 1.74 / 4.25 |
| a c4 tok/s/req | 11.6 | 18.1 | 18.8 |
| a c8 TTFT | 5.48 / 9.18 | 2.03 / 7.20 | 2.17 / 7.15 |
| a c8 tok/s/req | 7.2 | 12.1 | 12.2 |
| a hit | 0 | 0.79 | 0.79 |
| b c1 TTFT, tok/s/req | 1.61, 37.7 | 1.83, 37.1 | 1.50, 38.3 |
| b c4 TTFT, tok/s/req | 2.62, 14.3 | 1.35, 18.6 | 2.74, 13.9 |
| b c8 TTFT, tok/s/req | 4.44, 9.5 | 1.58, 13.2 | 2.85, 8.2 |
| b hit | 0 | 0.47 / 0.70 / 0.70 | 0 |

Per-turn TTFT p50 at a c4 (s):

| turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before | 3.49 | 2.47 | 3.87 | 2.69 | 3.11 | 4.26 | 4.79 | 4.31 |
| after | 3.56 | 2.31 | 1.77 | 1.23 | 1.59 | 1.27 | 1.29 | 1.27 |
| vLLM | 4.17 | 2.15 | 2.15 | 1.82 | 1.52 | 2.00 | 1.52 | 1.72 |

- An earlier rep with an unconditional cut gave a c4 1.54 s / 17.7, a c8 1.99 s / 11.8.
- b c1 is slower than cache off. Every one-turn session names a session, so each request pays a cut
  that is never attached, on top of the shared-end cut.
- vLLM does not reuse the shared system prompt (b hit 0).

Non-agentic recipe serve (natural-text MTP k=3, distinct prompts; c1 = 12 prompts of 128 / 1024 /
4096, cN = 2N prompts of 1K), TTFT mean in s:

| | cache off | cut always | cut when expected |
|---|---:|---:|---:|
| c1 | 0.96 | 1.13 | 0.94 |
| c4 | 1.42 | 1.49 | 1.16 |
| c8 | 1.82 | 2.33 | 1.81 |

tok/s/req is unchanged: 38.8 / 20.8 / 14.7 cache off, 39.3 / 21.1 / 15.0 with the conditional cut.
No compiler ran during any serve arm.

Gates: plowrt lib tests prefix 99, vmm 185, mux 135, spec 62, knob 28, kv_blocks 9, slot 98,
recurrent 10, and plow-asset serve_manifest 9, all passed. `check_recipe` OK.

Open:
- capturing the state inside the prompt's last launch, without the extra launch a cut costs on an
  expected-reuse miss;
- TURN_END boundaries under spec decode, which are published only when the position lands exactly
  on a 32-row boundary.

## Serving scoreboard (HEAD 92de9f1d8)

What the phase-2 work bought for concurrency and agentic use: the claimed-item decode MoE walk, the
VMM prefix cache, and the decode inner loops (FA16_PIPE=2, router top-k, conv span, FP8 TC
prefetch). Setup:
- Build: plowrt and the recipe packet built from a clean tree at 92de9f1d8 (packet sha256
  `9b695126d938`), MTP k=3, recipe `[serve.env]` (prefix cache on).
- Box: jg02, one gpuq job, `compilers_during_run=0` in every timed arm.
- Not a strict `campaign.py report` comparison: this packet has no FP32-reference gate.

### Recipe serve, 1K (decode MoE entry protocol)

Protocol:
- Natural text, MTP k=3, 1K ISL / 256 OSL; c1 = 12 prompts of 128 / 1024 / 4096.
- HEAD against the A2 build (the claimed walk only: dmoe2 cubins on the dmoe plowrt, no prefix
  cache), 3 reps, arms interleaved, order rotating by rep.
- tok/s/req = (generation_tokens - nreq) / Σ decode time. TTFT is from the server histogram.
- vLLM: the recorded MTP k=3 arms (tok/s/req, and TTFT mean from "Speculative decode").

| | HEAD tok/s/req | A2 | HEAD vs A2 | vLLM | HEAD vs vLLM | tick ms HEAD / A2 | TTFT mean HEAD / A2 / vLLM (s) | TTFT p50 / p99 HEAD (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | 40.94 | 39.68 | +3.2% | 40.68 | +0.6% | 80.2 / 82.8 | 0.98 / 0.97 / 1.24 | 0.90 / 2.45 |
| c2 | 30.40 | 29.98 | +1.4% | 29.52 | +3.0% | 107.4 / 111.2 | 1.09 / 1.15 / 1.74 | 1.17 / 2.47 |
| c4 | 22.30 | 21.31 | +4.6% | 21.90 | +1.8% | 148.6 / 154.6 | 1.17 / 1.33 / 2.07 | 1.23 / 2.47 |
| c8 | 15.57 | 15.16 | +2.7% | 14.48 | +7.5% | 206.4 / 213.0 | 1.72 / 2.00 / 2.90 | 2.10 / 4.93 |

- tok/s/req spread across reps is at most 2.8%. Accept length is equal in both arms (3.33-3.52).
- The decode inner loops take 2.6-6.6 ms off the serve tick, in line with `step_bench` ("Single-SM
  inner loops: decode").
- c1 was 2.9% behind vLLM with A2 and is now at parity.

### Agentic workloads (prefix cache)

Same client and workloads as "Agentic / prefix cache" (`llm_grid.sh --agentic`, 2 reps, each
workload started at <= 58 °C). The vLLM arm is that section's run. Columns: TTFT p50 / p99 (s),
then tok/s/req (1000 / TPOT p50).

| | HEAD | vLLM | TTFT p50 | tok/s/req | plow before (cache on, previous decode) |
|---|---|---|---:|---:|---|
| a c4 | 1.60 / 4.09, 19.31 | 1.74 / 4.25, 18.80 | -8% | +2.7% | 1.28 / 3.56, 18.1 |
| a c8 | 2.41 / 7.78, 13.05 | 2.17 / 7.15, 12.15 | **+11%** | +7.4% | 2.03 / 7.20, 12.1 |
| b c1 | 1.84 / 2.45, 39.58 | 1.50 / 1.52, 38.26 | **+23%** | +3.4% | 1.83 / 2.71, 37.1 |
| b c4 | 1.34 / 4.98, 19.93 | 2.74 / 5.76, 13.87 | -51% | +44% | 1.35 / 5.72, 18.6 |
| b c8 | 1.63 / 10.63, 13.74 | 2.85 / 11.63, 8.21 | -43% | +67% | 1.58 / 9.10, 13.2 |

- Prefix-token hit: a 0.79 on both stacks; b 0.47 / 0.70 / 0.70 on plow, 0 on vLLM.
- a c4 TTFT p50 is 1.24 s in rep 1 and 1.96 s in rep 2. Its mean is inside that spread; the a c4
  TTFT row is not a result.

Where plow still loses (TTFT; every tok/s/req cell wins):
- **a c8 TTFT p50, +11%.** Turns 2-8 run 2.0-3.2 s, against 1.9-2.7 s before this decode work.
  - The prefill work did not change: 335 vs 346 pack launches, 468 vs 453 state captures, and
    launch medians per bucket within 3%.
  - Decode got faster (TPOT p50 76.6 vs 82.7 ms), so the closed-loop cell is shorter: 103-105 s vs
    108-109 s.
  - Each session's next turn therefore arrives sooner and waits behind serial prefill ticks
    (mean 680 ms each).
  - In a8 rep 2, prefill and mixed ticks take 53% of the wall. Decode rows ride only 12 of its 83
    prefill ticks.
  - The lever is on the prefill side: more riders per prefill launch, and shorter prefill ticks.
    Decode is not the lever.
- **b c1 TTFT, +23%.** Each one-turn session names a session, so it pays a prompt-end cut (a ~0.2 s
  tail launch) that is never attached ("Agentic / prefix cache").

### Long context: 32K, c4

`llm_grid.sh` cell: 12 random 32,768-token prompts, OSL 256, c4. Each rep uses the same seed on
both stacks. MTP k=3, 2 reps, each stack cold-started at <= 58 °C. vLLM flags are those of the
agentic run (bf16 KV; plow serves E4M3 KV).

| | plow HEAD | vLLM | delta |
|---|---|---|---:|
| TTFT mean, r1 / r2 (s) | 17.90 / 16.51 | 28.70 / 20.65 | -30% |
| TTFT p50, r1 / r2 (s) | 15.91 / 12.25 | 24.84 / 13.62 | -27% |
| TTFT p99, mean (s) | 37.6 | 52.2 | -28% |
| TPOT p50, r1 / r2 (ms) | 130.1 / 113.6 | 147.9 / 140.1 | -15% |
| tok/s/req (1000 / TPOT p50) | 8.25 | 6.95 | +19% |
| output tok/s | 20.3 | 16.1 | +26% |
| cell wall (s) | 151 | 191 | -21% |

- TTFT spread across reps is large on both stacks (p50: plow 26%, vLLM 58%). Rep 1 is slower on
  both, so the result is the direction, not the size.
- TPOT here includes the decode stalls behind the other requests' 32K prefills.
