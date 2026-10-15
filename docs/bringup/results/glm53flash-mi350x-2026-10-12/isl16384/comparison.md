# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl16384`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v20/isl16384`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v19-r3-g/gates.json` (sha256 d22020e1f1c6, packet 49b696e4e77c, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 49b696e4e77c) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,142.5 GiB | 1,481.9 GiB (0.69x) |
| Total throughput | 13,015 tok/s | 11,101 tok/s (0.85x) |
| Throughput / GPU | 1,627 tok/s/GPU | 1,388 tok/s/GPU (0.85x) |
| TTFT P99 | 758.3 ms * | 689.4 ms (0.91x) * |
| TPOT P99 | 7.03 ms | 8.70 ms (1.24x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.1% / 2.2%; Total throughput 4.2% / 2.6%; Throughput / GPU 4.2% / 2.6%; TTFT P99 103.2% / 87.5%; TPOT P99 1.3% / 0.8%.

**FLAGGED: spread > 10%: Baseline TTFT P99 103.2%; Infervisor TTFT P99 87.5%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 49b696e4e77c) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.0 GiB | 1,489.0 GiB (0.69x) |
| Total throughput | 32,753 tok/s | 27,308 tok/s (0.83x) |
| Throughput / GPU | 4,094 tok/s/GPU | 3,413 tok/s/GPU (0.83x) |
| TTFT P99 | 3,178.0 ms * | 3,036.3 ms (0.96x) |
| TPOT P99 | 30.22 ms * | 34.72 ms (1.15x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.3%; Total throughput 9.4% / 0.8%; Throughput / GPU 9.4% / 0.8%; TTFT P99 33.0% / 3.9%; TPOT P99 22.6% / 1.8%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 49b696e4e77c) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.1 GiB | 1,490.8 GiB (0.70x) |
| Total throughput | 43,718 tok/s | 36,712 tok/s (0.84x) |
| Throughput / GPU | 5,465 tok/s/GPU | 4,589 tok/s/GPU (0.84x) |
| TTFT P99 | 10,716.3 ms | 12,289.8 ms (1.15x) |
| TPOT P99 | 95.07 ms * | 107.09 ms (1.13x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.0%; Total throughput 8.9% / 1.2%; Throughput / GPU 8.9% / 1.2%; TTFT P99 9.6% / 1.5%; TPOT P99 17.3% / 1.1%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 49b696e4e77c) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.1 GiB | 1,493.3 GiB (0.70x) |
| Total throughput | 48,816 tok/s | 38,818 tok/s (0.80x) |
| Throughput / GPU | 6,102 tok/s/GPU | 4,852 tok/s/GPU (0.80x) |
| TTFT P99 | 20,099.2 ms | 25,091.1 ms (1.25x) |
| TPOT P99 | 164.71 ms | 205.97 ms (1.25x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.7%; Total throughput 0.3% / 1.6%; Throughput / GPU 0.3% / 1.6%; TTFT P99 0.1% / 0.7%; TPOT P99 0.2% / 3.6%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 49b696e4e77c) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.1 GiB | 1,498.3 GiB (0.70x) |
| Total throughput | 50,932 tok/s | 37,982 tok/s (0.75x) |
| Throughput / GPU | 6,366 tok/s/GPU | 4,748 tok/s/GPU (0.75x) |
| TTFT P99 | 39,671.5 ms | 52,254.8 ms (1.32x) |
| TPOT P99 | 314.14 ms | 213.22 ms (0.68x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.2% / 0.5%; Throughput / GPU 0.2% / 0.5%; TTFT P99 0.1% / 0.6%; TPOT P99 0.2% / 1.8%.
