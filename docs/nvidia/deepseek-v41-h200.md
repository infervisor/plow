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

| Workload | c | plowrt tok/s | plowrt TTFT ms | plowrt TPOT ms | vLLM tok/s | vLLM TTFT ms | vLLM TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1k in / 256 out | 1 | 14.4 | 806 | 66.4 | 131.8 | 124 | 7.15 |
| 1k in / 256 out | 4 | 36.1 | 1672 | 104.5 | 447.7 | 145 | 8.38 |
| 1k in / 256 out | 16 | 74.8 | 1751 | 206.4 | 792.9 | 567 | 13.36 |
| 1k in / 256 out | 64 | 125.4 | 1893 | 494.4 | 1765.5 | 1325 | 33.52 |
| 4k in / 512 out | 1 | | | | 131.6 | 248 | 7.15 |
| 4k in / 512 out | 4 | | | | 458.8 | 163 | 8.46 |
| 4k in / 512 out | 16 | | | | 936.0 | 1358 | 14.70 |
| 16k in / 128 out | 1 | | | | 75.4 | 844 | 7.07 |
| 16k in / 128 out | 4 | | | | 424.3 | 107 | 8.42 |

(plowrt sweep `plowrt-sweep2`, kernels as of eedddffd.)

## 5. Kernel log

| Change | Rung | Before | After |
|---|---|---|---|
| fp32 GEMM: dot form for small output grids (the 24 x 20480 mHC mix ran as ONE block) | decode | ~240 ms/token | ~95 ms/token (c=1 TPOT) |
| grouped fp4 MoE GEMM: cp.async ring, prmt fp4->fp8 (the __constant__ LUT serialized), staged scales | 1k prefill | 1.87 s | 0.81 s |
| same | decode B=9 | 360 ms | 156 ms |
| fp32 row form for the mHC mix in prefill | 1k prefill | 143 ms (dot) | 82 ms |

Open, ranked by the last profile: the MoE GEMM is still ~0.5 TB/s against 4.8 (64% of prefill,
48% of a 9-sequence decode step); the W8A8 and bf16 GEMMs are mma.sync without TMA/wgmma.

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
