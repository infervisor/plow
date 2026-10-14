# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl1024`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v17/isl1024`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v17/gates.json` (sha256 7fd6879688b8, packet 6629af02610d, PASS)
- Values are means over repeats; ratios are Infervisor / Baseline. Total throughput = (input + output) tok/s. `*` = repeat spread > 10%.

## g1

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 1024 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 16 prompts, request rate inf / concurrency 1 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 8485e951f711, packet 6629af02610d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,132.3 GiB | 1,494.4 GiB (0.70x) |
| Total throughput | 1,189 tok/s | 727 tok/s (0.61x) |
| Throughput / GPU | 149 tok/s/GPU | 91 tok/s/GPU (0.61x) |
| TTFT P99 | 140.6 ms * | 93.0 ms (0.66x) * |
| TPOT P99 | 6.88 ms | 12.09 ms (1.76x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.1% / 2.2%; Total throughput 1.4% / 0.5%; Throughput / GPU 1.4% / 0.5%; TTFT P99 57.7% / 46.8%; TPOT P99 0.2% / 0.5%.

**FLAGGED: spread > 10%: Baseline TTFT P99 57.7%; Infervisor TTFT P99 46.8%.**

## g8

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 1024 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 24 prompts, request rate inf / concurrency 8 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 8485e951f711, packet 6629af02610d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,137.8 GiB | 1,501.4 GiB (0.70x) |
| Total throughput | 5,256 tok/s * | 3,216 tok/s (0.61x) |
| Throughput / GPU | 657 tok/s/GPU * | 402 tok/s/GPU (0.61x) |
| TTFT P99 | 1,764.9 ms * | 418.5 ms (0.24x) * |
| TPOT P99 | 19.29 ms * | 22.89 ms (1.19x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.6% / 1.2%; Total throughput 54.0% / 5.6%; Throughput / GPU 54.0% / 5.6%; TTFT P99 173.4% / 16.2%; TPOT P99 96.3% / 15.1%.

**FLAGGED: spread > 10%: Baseline Total throughput 54.0%; Baseline Throughput / GPU 54.0%; Baseline TTFT P99 173.4%; Infervisor TTFT P99 16.2%; Baseline TPOT P99 96.3%; Infervisor TPOT P99 15.1%.**

## g32

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 1024 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 96 prompts, request rate inf / concurrency 32 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 8485e951f711, packet 6629af02610d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,503.3 GiB (0.70x) |
| Total throughput | 17,572 tok/s | 9,468 tok/s (0.54x) |
| Throughput / GPU | 2,197 tok/s/GPU | 1,184 tok/s/GPU (0.54x) |
| TTFT P99 | 718.8 ms * | 1,277.7 ms (1.78x) * |
| TPOT P99 | 15.44 ms | 29.50 ms (1.91x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.0%; Total throughput 3.8% / 2.7%; Throughput / GPU 3.8% / 2.7%; TTFT P99 25.5% / 22.2%; TPOT P99 1.2% / 1.6%.

**FLAGGED: spread > 10%: Baseline TTFT P99 25.5%; Infervisor TTFT P99 22.2%.**

## g64

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 1024 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 192 prompts, request rate inf / concurrency 64 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 8485e951f711, packet 6629af02610d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,505.8 GiB (0.70x) |
| Total throughput | 26,683 tok/s | 12,438 tok/s (0.47x) |
| Throughput / GPU | 3,335 tok/s/GPU | 1,555 tok/s/GPU (0.47x) |
| TTFT P99 | 1,157.3 ms | 2,327.9 ms (2.01x) * |
| TPOT P99 | 20.44 ms | 44.54 ms (2.18x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.7%; Total throughput 1.9% / 2.0%; Throughput / GPU 1.9% / 2.0%; TTFT P99 3.4% / 12.6%; TPOT P99 1.3% / 1.7%.

**FLAGGED: spread > 10%: Infervisor TTFT P99 12.6%.**

## g128

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 1024 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 384 prompts, request rate inf / concurrency 128 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 8485e951f711, packet 6629af02610d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,510.8 GiB (0.70x) |
| Total throughput | 34,927 tok/s | 18,323 tok/s (0.52x) |
| Throughput / GPU | 4,366 tok/s/GPU | 2,290 tok/s/GPU (0.52x) |
| TTFT P99 | 2,237.1 ms | 4,606.3 ms (2.06x) * |
| TPOT P99 | 31.28 ms | 60.78 ms (1.94x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.3% / 4.0%; Throughput / GPU 0.3% / 4.0%; TTFT P99 1.7% / 14.6%; TPOT P99 0.6% / 8.6%.

**FLAGGED: spread > 10%: Infervisor TTFT P99 14.6%.**
