# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl16384`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v13/isl16384`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v13f/gates.json` (sha256 6ca86dd55d71, packet ab03f74fc6e8, PASS)
- Values are means over repeats; ratios are Infervisor / Baseline. Total throughput = (input + output) tok/s. `*` = repeat spread > 10%.

## g1

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 16384 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 16 prompts, request rate inf / concurrency 1 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 3e968d2f371c, packet ab03f74fc6e8) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,142.5 GiB | 1,210.9 GiB (0.57x) |
| Total throughput | 13,015 tok/s | 7,951 tok/s (0.61x) |
| Throughput / GPU | 1,627 tok/s/GPU | 994 tok/s/GPU (0.61x) |
| TTFT P99 | 758.3 ms * | 505.1 ms (0.67x) |
| TPOT P99 | 7.03 ms | 12.54 ms (1.78x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.1% / 2.7%; Total throughput 4.2% / 0.1%; Throughput / GPU 4.2% / 0.1%; TTFT P99 103.2% / 0.0%; TPOT P99 1.3% / 0.2%.

**FLAGGED: spread > 10%: Baseline TTFT P99 103.2%.**

## g8

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 16384 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 24 prompts, request rate inf / concurrency 8 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 3e968d2f371c, packet ab03f74fc6e8) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.0 GiB | 1,217.9 GiB (0.57x) |
| Total throughput | 32,753 tok/s | 16,498 tok/s (0.50x) |
| Throughput / GPU | 4,094 tok/s/GPU | 2,062 tok/s/GPU (0.50x) |
| TTFT P99 | 3,178.0 ms * | 4,230.8 ms (1.33x) |
| TPOT P99 | 30.22 ms * | 58.04 ms (1.92x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.5%; Total throughput 9.4% / 0.7%; Throughput / GPU 9.4% / 0.7%; TTFT P99 33.0% / 4.0%; TPOT P99 22.6% / 1.6%.

**FLAGGED: spread > 10%: Baseline TTFT P99 33.0%; Baseline TPOT P99 22.6%.**

## g32

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 16384 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 96 prompts, request rate inf / concurrency 32 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 3e968d2f371c, packet ab03f74fc6e8) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.1 GiB | 1,219.8 GiB (0.57x) |
| Total throughput | 43,718 tok/s | 25,225 tok/s (0.58x) |
| Throughput / GPU | 5,465 tok/s/GPU | 3,153 tok/s/GPU (0.58x) |
| TTFT P99 | 10,716.3 ms | 17,143.1 ms (1.60x) |
| TPOT P99 | 95.07 ms * | 154.74 ms (1.63x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.2%; Total throughput 8.9% / 0.0%; Throughput / GPU 8.9% / 0.0%; TTFT P99 9.6% / 0.4%; TPOT P99 17.3% / 0.0%.

**FLAGGED: spread > 10%: Baseline TPOT P99 17.3%.**

## g64

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 16384 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 192 prompts, request rate inf / concurrency 64 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 3e968d2f371c, packet ab03f74fc6e8) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.1 GiB | 1,222.3 GiB (0.57x) |
| Total throughput | 48,816 tok/s | 25,641 tok/s (0.53x) |
| Throughput / GPU | 6,102 tok/s/GPU | 3,205 tok/s/GPU (0.53x) |
| TTFT P99 | 20,099.2 ms | 37,262.9 ms (1.85x) |
| TPOT P99 | 164.71 ms | 299.12 ms (1.82x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.8%; Total throughput 0.3% / 0.6%; Throughput / GPU 0.3% / 0.6%; TTFT P99 0.1% / 0.1%; TPOT P99 0.2% / 0.1%.

## g128

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 16384 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 384 prompts, request rate inf / concurrency 128 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 3e968d2f371c, packet ab03f74fc6e8) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.1 GiB | 1,227.3 GiB (0.57x) |
| Total throughput | 50,932 tok/s | 26,306 tok/s (0.52x) |
| Throughput / GPU | 6,366 tok/s/GPU | 3,288 tok/s/GPU (0.52x) |
| TTFT P99 | 39,671.5 ms | 74,899.9 ms (1.89x) |
| TPOT P99 | 314.14 ms | 304.86 ms (0.97x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.2% / 0.1%; Throughput / GPU 0.2% / 0.1%; TTFT P99 0.1% / 0.2%; TPOT P99 0.2% / 0.2%.
