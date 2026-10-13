# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-dp2tp4/isl1024`
- Infervisor results: `/home/lava/bench/glm53flash/plow-dp2tp4-v10/isl1024`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-dp-v10/gates.json` (sha256 50da8ab8413a, packet 2beec26e536d, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt bf1d8b8bcc08, packet 2beec26e536d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,136.2 GiB | 1,203.8 GiB (0.56x) |
| Total throughput | 952 tok/s | 686 tok/s (0.72x) |
| Throughput / GPU | 119 tok/s/GPU | 86 tok/s/GPU (0.72x) |
| TTFT P99 | 183.7 ms * | 128.5 ms (0.70x) * |
| TPOT P99 | 8.52 ms | 15.74 ms (1.85x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.7% / 2.5%; Total throughput 0.7% / 8.1%; Throughput / GPU 0.7% / 8.1%; TTFT P99 38.5% / 23.2%; TPOT P99 0.1% / 49.7%.

**FLAGGED: spread > 10%: Baseline TTFT P99 38.5%; Infervisor TTFT P99 23.2%; Infervisor TPOT P99 49.7%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt bf1d8b8bcc08, packet 2beec26e536d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,140.9 GiB | 1,209.9 GiB (0.57x) |
| Total throughput | 4,017 tok/s * | 2,510 tok/s (0.62x) |
| Throughput / GPU | 502 tok/s/GPU * | 314 tok/s/GPU (0.62x) |
| TTFT P99 | 1,654.5 ms * | 483.7 ms (0.29x) |
| TPOT P99 | 22.45 ms * | 27.99 ms (1.25x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.2% / 1.5%; Total throughput 63.0% / 1.9%; Throughput / GPU 63.0% / 1.9%; TTFT P99 149.5% / 9.5%; TPOT P99 81.1% / 2.9%.

**FLAGGED: spread > 10%: Baseline Total throughput 63.0%; Baseline Throughput / GPU 63.0%; Baseline TTFT P99 149.5%; Baseline TPOT P99 81.1%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt bf1d8b8bcc08, packet 2beec26e536d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,153.9 GiB | 1,211.6 GiB (0.56x) |
| Total throughput | 13,516 tok/s | 5,695 tok/s (0.42x) |
| Throughput / GPU | 1,689 tok/s/GPU | 712 tok/s/GPU (0.42x) |
| TTFT P99 | 1,284.0 ms | 1,485.0 ms (1.16x) |
| TPOT P99 | 19.91 ms | 49.97 ms (2.51x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.2%; Total throughput 0.9% / 2.9%; Throughput / GPU 0.9% / 2.9%; TTFT P99 2.7% / 1.9%; TPOT P99 0.6% / 5.4%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt bf1d8b8bcc08, packet 2beec26e536d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,153.9 GiB | 1,214.0 GiB (0.56x) |
| Total throughput | 20,483 tok/s | 9,307 tok/s (0.45x) |
| Throughput / GPU | 2,560 tok/s/GPU | 1,163 tok/s/GPU (0.45x) |
| TTFT P99 | 1,718.4 ms * | 2,793.6 ms (1.63x) |
| TPOT P99 | 26.18 ms | 60.07 ms (2.29x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.8%; Total throughput 5.3% / 0.7%; Throughput / GPU 5.3% / 0.7%; TTFT P99 23.9% / 0.3%; TPOT P99 2.8% / 0.5%.

**FLAGGED: spread > 10%: Baseline TTFT P99 23.9%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt bf1d8b8bcc08, packet 2beec26e536d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,154.0 GiB | 1,218.8 GiB (0.57x) |
| Total throughput | 26,679 tok/s | 12,247 tok/s (0.46x) |
| Throughput / GPU | 3,335 tok/s/GPU | 1,531 tok/s/GPU (0.46x) |
| TTFT P99 | 3,341.2 ms | 5,781.5 ms (1.73x) |
| TPOT P99 | 40.61 ms | 92.63 ms (2.28x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.9% / 3.1%; Throughput / GPU 0.9% / 3.1%; TTFT P99 0.7% / 3.2%; TPOT P99 0.3% / 7.8%.
