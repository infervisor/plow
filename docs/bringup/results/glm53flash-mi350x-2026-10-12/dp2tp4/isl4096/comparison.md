# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-dp2tp4/isl4096`
- Infervisor results: `/home/lava/bench/glm53flash/plow-dp2tp4-v10/isl4096`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-dp-v10/gates.json` (sha256 50da8ab8413a, packet 2beec26e536d, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 912e959a3654, packet 2beec26e536d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,137.1 GiB | 1,203.8 GiB (0.56x) |
| Total throughput | 3,106 tok/s | 2,253 tok/s (0.73x) |
| Throughput / GPU | 388 tok/s/GPU | 282 tok/s/GPU (0.73x) |
| TTFT P99 | 686.4 ms * | 515.7 ms (0.75x) * |
| TPOT P99 | 8.60 ms | 17.81 ms (2.07x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.3% / 2.5%; Total throughput 8.6% / 2.6%; Throughput / GPU 8.6% / 2.6%; TTFT P99 130.0% / 34.3%; TPOT P99 0.1% / 26.8%.

**FLAGGED: spread > 10%: Baseline TTFT P99 130.0%; Infervisor TTFT P99 34.3%; Infervisor TPOT P99 26.8%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 912e959a3654, packet 2beec26e536d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,150.5 GiB | 1,209.9 GiB (0.56x) |
| Total throughput | 11,699 tok/s * | 7,231 tok/s (0.62x) |
| Throughput / GPU | 1,462 tok/s/GPU * | 904 tok/s/GPU (0.62x) |
| TTFT P99 | 2,179.1 ms * | 1,346.8 ms (0.62x) * |
| TPOT P99 | 24.83 ms * | 37.55 ms (1.51x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.5%; Total throughput 38.7% / 3.6%; Throughput / GPU 38.7% / 3.6%; TTFT P99 82.9% / 15.2%; TPOT P99 46.6% / 11.0%.

**FLAGGED: spread > 10%: Baseline Total throughput 38.7%; Baseline Throughput / GPU 38.7%; Baseline TTFT P99 82.9%; Infervisor TTFT P99 15.2%; Baseline TPOT P99 46.6%; Infervisor TPOT P99 11.0%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 912e959a3654, packet 2beec26e536d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,150.5 GiB | 1,211.6 GiB (0.56x) |
| Total throughput | 25,710 tok/s * | 13,304 tok/s (0.52x) |
| Throughput / GPU | 3,214 tok/s/GPU * | 1,663 tok/s/GPU (0.52x) |
| TTFT P99 | 4,290.8 ms * | 5,230.2 ms (1.22x) |
| TPOT P99 | 45.31 ms * | 76.09 ms (1.68x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.2%; Total throughput 15.3% / 0.2%; Throughput / GPU 15.3% / 0.2%; TTFT P99 62.7% / 0.6%; TPOT P99 42.8% / 0.0%.

**FLAGGED: spread > 10%: Baseline Total throughput 15.3%; Baseline Throughput / GPU 15.3%; Baseline TTFT P99 62.7%; Baseline TPOT P99 42.8%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 912e959a3654, packet 2beec26e536d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,150.6 GiB | 1,214.0 GiB (0.56x) |
| Total throughput | 33,405 tok/s | 17,554 tok/s (0.53x) |
| Throughput / GPU | 4,176 tok/s/GPU | 2,194 tok/s/GPU (0.53x) |
| TTFT P99 | 5,967.8 ms | 10,427.4 ms (1.75x) |
| TPOT P99 | 60.81 ms | 115.21 ms (1.89x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.8%; Total throughput 6.5% / 0.6%; Throughput / GPU 6.5% / 0.6%; TTFT P99 8.3% / 0.3%; TPOT P99 7.3% / 0.4%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 912e959a3654, packet 2beec26e536d) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,150.6 GiB | 1,218.8 GiB (0.57x) |
| Total throughput | 39,125 tok/s | 19,827 tok/s (0.51x) |
| Throughput / GPU | 4,891 tok/s/GPU | 2,478 tok/s/GPU (0.51x) |
| TTFT P99 | 11,568.7 ms | 21,477.3 ms (1.86x) |
| TPOT P99 | 102.91 ms | 206.86 ms (2.01x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.9% / 2.6%; Throughput / GPU 0.9% / 2.6%; TTFT P99 4.9% / 0.5%; TPOT P99 2.5% / 6.7%.
