# DeepSeek-V4.1-Flash on 4x H200 (sm_90a) through plowrt

Status and campaign log for `plowrt dsv41-serve`: the host-driven V4.1 engine
(`crates/plowrt/src/dsv41`) over dedicated sm_90a kernels (`runtime/nvidia/dsv41`). Branch:
`dsv41-nvidia`.

## 1. Why a dedicated engine

On NVIDIA, plow's packet/megakernel path could not run V4.1: V4.1 had no NVIDIA emit and no decode
emit on any backend, the sm_90a interpreter lacked ~20 of its opcodes, and CUDA serving was
single-device (no peer memory or collectives). The 476 GiB checkpoint does not fit one 141 GB H200,
so the engine is pipeline-parallel over the four GPUs (10 layers each), with the two Engram tables
(~189 GiB, 24 rows gathered per token) left in host memory.

This bypasses plow's architecture on purpose, to reach a verified end-to-end run first. The
convergence path (TP/EP with in-kernel peer-slot collectives as on AMD, the kernels as packet ops,
the engine behind `SeqEngine`) is section 6.

## 2. Verification

| Check | Result |
|---|---|
| act_quant (fp8, ue8m0), fp4 fake-quant (ue8m0 / e4m3 scales) vs `inference/kernel.py` | bit-exact |
| W8A8 [32,32] GEMM vs reference `fp8_gemm`; bf16 / fp32 GEMMs vs torch | within bf16 output rounding (1.7e-3) / 1e-6 |
| Layer parity vs reference `model.py` Blocks, real weights, prefill + decode, teacher-forced | 5e-3 .. 4e-2 relative per layer |
| Indexer top-512 picks vs reference | 99.8-100% identical per row |
| MoE from identical input | routing 100% identical; output 2.8e-4 (prefill), exact (decode) |
| Engram gate from identical input | exact |
| Full model, greedy | "The capital of France is" -> " Paris. ..."; a correct fibonacci; "17 * 23" -> "391" |

The per-layer differences trace to near-tie flips (one index pick of 512; FP8 activation
quantization steps), not to formula errors: every component matches when given identical input.

Tools: `scripts/dsv41_nv/test_kernels.py`, `test_layers.py` (`DSV41_TEACHER`, `DSV41_SUB`),
`bench_moe.py`, `stream_probe.py`, `run_plowrt.sh smoke|quick|streamtest|sweep`.

## 3. Method: rungs against the roofline

`PLOW_DSV41_PROFILE=2 plowrt dsv41-serve ... --rung-bench "prefill=1024,4096,16384;decode=1x1024,16x1024,64x1024"`
reports, per rung, the unperturbed step time and a per-kernel table: calls, ms, achieved TB/s and
TFLOP/s, the roofline floor `max(bytes / 4.8 TB/s, FLOPs / peak)` (H200 dense peaks: fp8 1979,
bf16 989, fp32 67 TFLOP/s) and efficiency. Bytes are compulsory DRAM traffic (each operand once),
so the floor is what an ideal kernel would reach, not the current kernel's.

## 4. End-to-end: plowrt vs vLLM TP4 (same box, same `vllm bench serve` grid)

vLLM baseline: `/root/dsv41/results/tp4-base4` (Marlin weight-only FP8 dense, Marlin W4A16 MoE,
FlashMLA sparse, DeepGEMM indexer).

Output tokens/s, median TTFT and TPOT. plowrt sweep2 = the first end-to-end run (kernels as of
eedddffd), sweep3 = after the kernel work in section 5 up to 62dd2e0f (before decode pipelining);
every request succeeded in both.

| Workload | c | sweep2 tok/s | sweep3 tok/s | sweep3 TTFT ms | sweep3 TPOT ms | vLLM tok/s | vLLM TTFT ms | vLLM TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k in / 256 out | 1 | 14.4 | 47.2 | 343 | 19.9 | 131.8 | 124 | 7.15 |
| 1k in / 256 out | 4 | 36.1 | 113.0 | 691 | 32.8 | 447.7 | 145 | 8.38 |
| 1k in / 256 out | 16 | 74.8 | 241.6 | 714 | 63.0 | 792.9 | 567 | 13.36 |
| 1k in / 256 out | 64 | 125.4 | 368.6 | 754 | 168.0 | 1765.5 | 1325 | 33.52 |
| 4k in / 512 out | 1 | 14.4 | 46.6 | 824 | 19.9 | 131.6 | 248 | 7.15 |
| 4k in / 512 out | 4 | 35.5 | 108.9 | 1647 | 33.5 | 458.8 | 163 | 8.46 |
| 4k in / 512 out | 16 | 74.6 | 228.8 | 1663 | 66.6 | 936.0 | 1358 | 14.70 |
| 16k in / 128 out | 1 | 8.8 | 24.8 | 2621 | 20.1 | 75.4 | 844 | 7.07 |
| 16k in / 128 out | 4 | 14.1 | 36.2 | 6495 | 60.1 | 424.3 | 107 | 8.42 |

(`/root/dsv41/results/plowrt-sweep2`, `plowrt-sweep3`. vLLM runs with chunked prefill and CUDA
graphs; plowrt admits one unchunked prefill at a time, so TTFT under concurrency includes queueing
behind other prefills.)

The remaining gap is structural as much as per-kernel. vLLM's TP4 splits every layer over the four
GPUs, so one token uses all four HBMs at once; plowrt's PP sends it through four stages in turn
(single-token floor 2.75 ms, spent today in ~1400 small kernels), and with one decode batch in
flight three GPUs wait. Decode lanes (below) keep several groups in flight; the next levers are the
host launch rate (the lanes now make it the limit), fused per-layer blocks, and TP / EP with
in-kernel collectives.

### Decode lanes

The scheduler splits the running sequences into groups (one per decode lane, `--decode-lanes`,
default 4) and keeps each group's step in flight while the others run on other stages; a prefill runs
on lane 0 alongside. Each lane has its own arena, index buffers and Engram staging, so a step's
hand-off buffers survive until the next stage has read them. Greedy outputs are byte-identical run
one at a time or concurrently across lanes (`run_plowrt.sh concur`).

| rung (ctx 1k) | one batch in flight | 4 groups in flight |
|---|---|---|
| 4 sequences | B=4: 135 tok/s, 29.7 ms/step | 4x1: 145 tok/s, 27.7 ms TPOT |
| 16 sequences | B=16: 379 tok/s, 42.2 ms/step | 4x4: 405 tok/s, 39.6 ms TPOT |
| 64 sequences | B=64: 808 tok/s, 79.2 ms/step | 4x16: 1166 tok/s, 54.9 ms TPOT |

Less than the GPUs allow: one step's ~1400 launches cost ~5 ms (B=1) to ~10 ms (B=16) of host
time, and with four groups in flight the single issuing thread is as busy as the GPUs.

## 5. Kernel log

| Change | Rung | Before | After |
|---|---|---|---|
| fp32 GEMM: dot form for small output grids (the 24 x 20480 mHC mix ran as ONE block) | decode | ~240 ms/token | ~95 ms/token (c=1 TPOT) |
| grouped fp4 MoE GEMM: cp.async ring, prmt fp4->fp8 (the __constant__ LUT serialized), staged scales | 1k prefill | 1.87 s | 0.81 s |
| same | decode B=9 | 360 ms | 156 ms |
| fp32 row form for the mHC mix in prefill | 1k prefill | 143 ms (dot) | 82 ms |
| split-K W8A8 / bf16 GEMMs (about two waves on 132 SMs); W8A8 as a 3-stage 128-K cp.async ring | 1k prefill | 593 ms | 496 ms |
| mHC mix fused: one pass over x for the 24 projections and the sum of squares, split-K (DeepGEMM `tf32_hc_prenorm_gemm` idea), then reduce + Sinkhorn | all | 3 launches | 2; matches the unfused path to 6e-7 on live activations |
| grouped fp4 MoE GEMM: tile height 16/32/64 from rows per expert (the 64-row tile was ~75% padding at 1k prefill and decode); warps side by side along N | 1k prefill (MoE w13, bench) | 7.1 ms | 3.4 ms |
| fp4 -> e4m3 by shifts alone (e2m1 s.ee.m -> s.0000.ee.m00 is the e4m3 encoding of w * 2^-6, subnormal included; 2^6 into the scale), activations byte-permuted to the same K order; still bit-exact vs kernel.py `fp4_gemm` | MoE, all T | | 10-25% faster |
| routed experts at decode: 16-row MMA tile instead of the fp32 GEMV (the GEMV was issue-bound at ~1 TB/s) | decode B=64 | 227 ms (MoE 189 ms, 0.97 TB/s) | 106 ms (MoE 51 ms, 3.6 TB/s, 75% of roofline) |
| mHC Sinkhorn on 16 lanes per token (shuffle row/column sums) instead of one thread's serial divides | decode B=1 | 35 us/sublayer | ~3 us |
| sparse attention split-KV (flash-decoding: grid.y splits over the 64-row KV tiles + a merge that adds the sink once) | decode B=1 | 96 us/layer on one SM | 27.6 -> 22.8 ms/token with the above |
| fp32 GEMM dispatch: dot form only for M <= 64 and M*N <= 2^20 (the router at M=64 was 6 tiles; the vocab head is not a dot-form shape) | decode B=64 | 25 ms (tiled router) / 73 ms (dot head) | 6 ms |
| Hopper wgmma prefill GEMMs (`dsv41_wg.cu`): producer warpgroup decodes fp8 / fp4 weights to bf16 in 128B-swizzled smem (bit placement: e4m3 bits in bf16 position are the value * 2^-120, one bf16x2 multiply applies 2^(120+S)); A is the fake-quantized bf16 activation (`act_quant` fq); two consumer warpgroups on m64n256k16; mbarrier ring, cp.async with noinc arrive; scale loads prefetched a lookahead ahead | prefill 16k | 3916 ms (MoE 1574, W8A8 912, wo_a 413) | 2594 ms (MoE 893, dense 671) |
| same, at 4k / 1k | prefill | 1095 / 367 ms | 818 / 338 ms |
| decode W8A8 as a swap-AB tensor-core GEMV (weights on the MMA's M side straight from memory, tokens on N; a K permutation inside each scale block lets every lane load 8 contiguous bytes per row; split-K into the ordered reduce) for M <= 16 | decode B=1 | 22.9 ms (W8A8 6.3 ms, 0.87 TB/s) | 20.2 ms (bench_gemv.py: 2.3-3.2 TB/s on the large shapes) |
| routed experts at decode (T <= 64) as a swap-AB tensor-core GEMV on 8-row expert tiles: lanes load 8 contiguous bytes over a PAIR of fp4 blocks (whole sectors; one shfl.xor hands over the other block's half), scale bytes prefetched per 16-block batch; bit-exact vs fp4_gemm | decode B=16 / 64 | 47.3 / 87.5 ms (16-row MMA tile, ~2.0 TB/s) | 44.6 / 83.9 ms (~2.3 TB/s: the fp4 decode + per-block promotion keep it near issue-bound) |

The wgmma kernels are 1.6-1.8x the mma.sync ones (bench_wg.py: ~350 TFLOP/s dense, ~320 grouped
fp4 at 16k rows) and bit-identical or at bf16 rounding against kernel.py (`test_kernels.py wg`); on
live activations every call matched its mma.sync twin to <= 3.6e-4 (bf16 output rounding).
`PLOW_DSV41_NO_WG=1` pins the mma.sync kernels. Measured ceilings of the same pipeline: ~800
TFLOP/s with no loads and no decode, ~475 with the decode, ~500 with the loads -- the producer and
the bf16 operands' shared-memory traffic bound it; fp8 wgmma with per-32 promotion (DeepGEMM-style,
two consumer warpgroups ping-ponging MMA and promotion) is the next step.

Rungs after these (`/root/dsv41/results/rungs_v8.md`): prefill 1k / 4k / 16k 338 / 819 / 2592 ms
(3027 / 5002 / 6322 tok/s); decode B=1 / 16 / 64 at ctx 1k 20.1 / 44.6 / 83.9 ms (50 / 359 / 763
tok/s).

Open, ranked by that profile: prefill 16k is 35% grouped fp4 wgmma (312 TFLOP/s), 26% dense wgmma
(334), 15% sparse attention (28% of its memory floor); decode B=64 is 57% routed experts at 3.55 TB/s
(74% of roofline); decode B=1 is launch- and latency-bound (~1400 launches, W8A8 at 18% of its
floor): CUDA graphs and fused per-layer blocks.

## 6. Plan

1. Per-rung roofline tables; take the largest gap first, re-measure after each change.
2. GEMMs, DeepGEMM-style (TMA, wgmma, warp-specialized producer/consumer, persistent tiles):
   dense W8A8 needs a per-32-K promotion (DeepGEMM's sm90 kernels assert 128-wide float scales);
   the grouped fp4 MoE GEMM decodes fp4 -> fp8 in the producer path (DeepGEMM has no sm90 FP4).
3. Indexer: adapt DeepGEMM `sm90_fp8_mqa_logits` / `sm90_fp8_paged_mqa_logits` (vendored in
   vllm/third_party/deep_gemm, CUTLASS 4.2.1 headers) and a fused top-k.
4. mHC: fuse the projection with the sum of squares (DeepGEMM `sm90_tf32_hc_prenorm_gemm`).
5. Unique blocks: one fused schedule per layer kind (window-only; ratio-2 source; ratio-2 consumer;
   ratio-1 source/candidate; ratio-1 index consumer; Engram), then CUDA graphs per decode bucket.
6. Parallelism: microbatch pipelining across the four stages; then EP for the MoE with an NVLink
   dispatch/combine (DeepEP is not on this box; FlashInfer's sm90 push-style MoE all-to-all source
   is) and TP for attention with in-kernel collectives, as plow does on AMD.
7. plowc: land the same kernels as NVIDIA packet ops so the emit path gains them too.
