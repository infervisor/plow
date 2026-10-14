# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl4096`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v13/isl4096`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v13f/gates.json` (sha256 6ca86dd55d71, packet ab03f74fc6e8, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 3e968d2f371c, packet ab03f74fc6e8) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,135.1 GiB | 1,210.9 GiB (0.57x) |
| Total throughput | 4,153 tok/s | 2,469 tok/s (0.59x) |
| Throughput / GPU | 519 tok/s/GPU | 309 tok/s/GPU (0.59x) |
| TTFT P99 | 511.6 ms * | 158.4 ms (0.31x) |
| TPOT P99 | 6.97 ms | 12.35 ms (1.77x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.0% / 2.7%; Total throughput 5.8% / 0.5%; Throughput / GPU 5.8% / 0.5%; TTFT P99 156.3% / 5.3%; TPOT P99 0.0% / 1.4%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 3e968d2f371c, packet ab03f74fc6e8) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,145.4 GiB | 1,217.9 GiB (0.57x) |
| Total throughput | 14,650 tok/s * | 6,911 tok/s (0.47x) |
| Throughput / GPU | 1,831 tok/s/GPU * | 864 tok/s/GPU (0.47x) |
| TTFT P99 | 2,290.6 ms * | 1,148.7 ms (0.50x) |
| TPOT P99 | 25.48 ms * | 36.56 ms (1.43x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.5%; Total throughput 50.3% / 2.3%; Throughput / GPU 50.3% / 2.3%; TTFT P99 145.1% / 8.6%; TPOT P99 92.4% / 4.7%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 3e968d2f371c, packet ab03f74fc6e8) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,145.4 GiB | 1,219.8 GiB (0.57x) |
| Total throughput | 35,287 tok/s | 15,956 tok/s (0.45x) |
| Throughput / GPU | 4,411 tok/s/GPU | 1,995 tok/s/GPU (0.45x) |
| TTFT P99 | 2,616.3 ms * | 4,470.1 ms (1.71x) |
| TPOT P99 | 29.09 ms | 64.14 ms (2.20x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.2%; Total throughput 3.7% / 1.9%; Throughput / GPU 3.7% / 1.9%; TTFT P99 19.5% / 4.0%; TPOT P99 6.0% / 1.1%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 3e968d2f371c, packet ab03f74fc6e8) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,145.4 GiB | 1,222.3 GiB (0.57x) |
| Total throughput | 43,108 tok/s | 18,816 tok/s (0.44x) |
| Throughput / GPU | 5,389 tok/s/GPU | 2,352 tok/s/GPU (0.44x) |
| TTFT P99 | 4,662.2 ms | 9,445.9 ms (2.03x) |
| TPOT P99 | 47.99 ms | 107.37 ms (2.24x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.8%; Total throughput 2.4% / 1.3%; Throughput / GPU 2.4% / 1.3%; TTFT P99 0.1% / 4.3%; TPOT P99 4.5% / 1.4%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 3e968d2f371c, packet ab03f74fc6e8) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,145.4 GiB | 1,227.3 GiB (0.57x) |
| Total throughput | 48,305 tok/s | 23,063 tok/s (0.48x) |
| Throughput / GPU | 6,038 tok/s/GPU | 2,883 tok/s/GPU (0.48x) |
| TTFT P99 | 9,295.6 ms | 19,099.8 ms (2.05x) |
| TPOT P99 | 84.73 ms | 178.05 ms (2.10x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.1% / 3.4%; Throughput / GPU 0.1% / 3.4%; TTFT P99 0.6% / 4.3%; TPOT P99 0.5% / 7.8%.
