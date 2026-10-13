# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-dp2tp4/isl4096`
- Infervisor results: `/home/lava/bench/glm53flash/plow-dp2tp4-v11/isl4096`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-dp-v11/gates.json` (sha256 b7b144543adb, packet 92ae9bd5840e, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 633f828df80d, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,137.1 GiB | 1,196.7 GiB (0.56x) |
| Total throughput | 3,106 tok/s | 2,231 tok/s (0.72x) |
| Throughput / GPU | 388 tok/s/GPU | 279 tok/s/GPU (0.72x) |
| TTFT P99 | 686.4 ms * | 328.4 ms (0.48x) * |
| TPOT P99 | 8.60 ms | 16.01 ms (1.86x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.3% / 2.5%; Total throughput 8.6% / 8.6%; Throughput / GPU 8.6% / 8.6%; TTFT P99 130.0% / 55.9%; TPOT P99 0.1% / 44.2%.

**FLAGGED: spread > 10%: Baseline TTFT P99 130.0%; Infervisor TTFT P99 55.9%; Infervisor TPOT P99 44.2%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 633f828df80d, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,150.5 GiB | 1,202.8 GiB (0.56x) |
| Total throughput | 11,699 tok/s * | 7,706 tok/s (0.66x) |
| Throughput / GPU | 1,462 tok/s/GPU * | 963 tok/s/GPU (0.66x) |
| TTFT P99 | 2,179.1 ms * | 1,165.8 ms (0.54x) * |
| TPOT P99 | 24.83 ms * | 33.05 ms (1.33x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.5%; Total throughput 38.7% / 6.7%; Throughput / GPU 38.7% / 6.7%; TTFT P99 82.9% / 17.6%; TPOT P99 46.6% / 10.0%.

**FLAGGED: spread > 10%: Baseline Total throughput 38.7%; Baseline Throughput / GPU 38.7%; Baseline TTFT P99 82.9%; Infervisor TTFT P99 17.6%; Baseline TPOT P99 46.6%; Infervisor TPOT P99 10.0%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 633f828df80d, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,150.5 GiB | 1,204.6 GiB (0.56x) |
| Total throughput | 25,710 tok/s * | 13,934 tok/s (0.54x) |
| Throughput / GPU | 3,214 tok/s/GPU * | 1,742 tok/s/GPU (0.54x) |
| TTFT P99 | 4,290.8 ms * | 4,566.0 ms (1.06x) |
| TPOT P99 | 45.31 ms * | 73.41 ms (1.62x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.2%; Total throughput 15.3% / 0.1%; Throughput / GPU 15.3% / 0.1%; TTFT P99 62.7% / 6.1%; TPOT P99 42.8% / 0.5%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 633f828df80d, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,150.6 GiB | 1,206.9 GiB (0.56x) |
| Total throughput | 33,405 tok/s | 18,610 tok/s (0.56x) |
| Throughput / GPU | 4,176 tok/s/GPU | 2,326 tok/s/GPU (0.56x) |
| TTFT P99 | 5,967.8 ms | 9,297.7 ms (1.56x) |
| TPOT P99 | 60.81 ms | 110.41 ms (1.82x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.8%; Total throughput 6.5% / 0.2%; Throughput / GPU 6.5% / 0.2%; TTFT P99 8.3% / 0.8%; TPOT P99 7.3% / 0.4%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 6272 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 633f828df80d, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,150.6 GiB | 1,211.7 GiB (0.56x) |
| Total throughput | 39,125 tok/s | 20,838 tok/s (0.53x) |
| Throughput / GPU | 4,891 tok/s/GPU | 2,605 tok/s/GPU (0.53x) |
| TTFT P99 | 11,568.7 ms | 19,346.9 ms (1.67x) |
| TPOT P99 | 102.91 ms | 205.06 ms (1.99x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.9% / 5.8%; Throughput / GPU 0.9% / 5.8%; TTFT P99 4.9% / 2.2%; TPOT P99 2.5% / 13.7%.

**FLAGGED: spread > 10%: Infervisor TPOT P99 13.7%.**
