# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl4096`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v18/isl4096`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v18-int/gates.json` (sha256 360cd843bdbd, packet 360814fc607b, PASS)
- Values are means over repeats; ratios are Infervisor / Baseline. Total throughput = (input + output) tok/s. `*` = repeat spread > 10%.

## g1

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 16 prompts, request rate inf / concurrency 1 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,135.1 GiB | 1,494.8 GiB (0.70x) |
| Total throughput | 4,153 tok/s | 2,952 tok/s (0.71x) |
| Throughput / GPU | 519 tok/s/GPU | 369 tok/s/GPU (0.71x) |
| TTFT P99 | 511.6 ms * | 119.6 ms (0.23x) |
| TPOT P99 | 6.97 ms | 10.46 ms (1.50x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.0% / 2.2%; Total throughput 5.8% / 0.4%; Throughput / GPU 5.8% / 0.4%; TTFT P99 156.3% / 8.0%; TPOT P99 0.0% / 1.2%.

**FLAGGED: spread > 10%: Baseline TTFT P99 156.3%.**

## g8

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 24 prompts, request rate inf / concurrency 8 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,145.4 GiB | 1,501.8 GiB (0.70x) |
| Total throughput | 14,650 tok/s * | 11,049 tok/s (0.75x) |
| Throughput / GPU | 1,831 tok/s/GPU * | 1,381 tok/s/GPU (0.75x) |
| TTFT P99 | 2,290.6 ms * | 796.2 ms (0.35x) |
| TPOT P99 | 25.48 ms * | 23.03 ms (0.90x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.2%; Total throughput 50.3% / 0.5%; Throughput / GPU 50.3% / 0.5%; TTFT P99 145.1% / 4.8%; TPOT P99 92.4% / 0.7%.

**FLAGGED: spread > 10%: Baseline Total throughput 50.3%; Baseline Throughput / GPU 50.3%; Baseline TTFT P99 145.1%; Baseline TPOT P99 92.4%.**

## g32

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 96 prompts, request rate inf / concurrency 32 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,145.4 GiB | 1,503.7 GiB (0.70x) |
| Total throughput | 35,287 tok/s | 25,022 tok/s (0.71x) |
| Throughput / GPU | 4,411 tok/s/GPU | 3,128 tok/s/GPU (0.71x) |
| TTFT P99 | 2,616.3 ms * | 2,979.4 ms (1.14x) |
| TPOT P99 | 29.09 ms | 40.72 ms (1.40x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.0%; Total throughput 3.7% / 0.9%; Throughput / GPU 3.7% / 0.9%; TTFT P99 19.5% / 2.8%; TPOT P99 6.0% / 0.1%.

**FLAGGED: spread > 10%: Baseline TTFT P99 19.5%.**

## g64

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 192 prompts, request rate inf / concurrency 64 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,145.4 GiB | 1,506.2 GiB (0.70x) |
| Total throughput | 43,108 tok/s | 28,769 tok/s (0.67x) |
| Throughput / GPU | 5,389 tok/s/GPU | 3,596 tok/s/GPU (0.67x) |
| TTFT P99 | 4,662.2 ms | 6,277.8 ms (1.35x) |
| TPOT P99 | 47.99 ms | 70.60 ms (1.47x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.7%; Total throughput 2.4% / 0.6%; Throughput / GPU 2.4% / 0.6%; TTFT P99 0.1% / 4.2%; TPOT P99 4.5% / 0.6%.

## g128

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 384 prompts, request rate inf / concurrency 128 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,145.4 GiB | 1,511.2 GiB (0.70x) |
| Total throughput | 48,305 tok/s | 34,790 tok/s (0.72x) |
| Throughput / GPU | 6,038 tok/s/GPU | 4,349 tok/s/GPU (0.72x) |
| TTFT P99 | 9,295.6 ms | 13,143.7 ms (1.41x) |
| TPOT P99 | 84.73 ms | 120.78 ms (1.43x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.1% / 2.8%; Throughput / GPU 0.1% / 2.8%; TTFT P99 0.6% / 4.9%; TPOT P99 0.5% / 8.0%.
