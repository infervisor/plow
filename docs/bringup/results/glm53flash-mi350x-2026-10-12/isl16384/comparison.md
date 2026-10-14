# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl16384`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v18/isl16384`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v18-int/gates.json` (sha256 360cd843bdbd, packet 360814fc607b, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,142.5 GiB | 1,494.8 GiB (0.70x) |
| Total throughput | 13,015 tok/s | 9,654 tok/s (0.74x) |
| Throughput / GPU | 1,627 tok/s/GPU | 1,207 tok/s/GPU (0.74x) |
| TTFT P99 | 758.3 ms * | 381.1 ms (0.50x) |
| TPOT P99 | 7.03 ms | 10.65 ms (1.52x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.1% / 2.2%; Total throughput 4.2% / 0.3%; Throughput / GPU 4.2% / 0.3%; TTFT P99 103.2% / 1.9%; TPOT P99 1.3% / 0.1%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.0 GiB | 1,501.8 GiB (0.70x) |
| Total throughput | 32,753 tok/s | 24,299 tok/s (0.74x) |
| Throughput / GPU | 4,094 tok/s/GPU | 3,037 tok/s/GPU (0.74x) |
| TTFT P99 | 3,178.0 ms * | 3,106.0 ms (0.98x) |
| TPOT P99 | 30.22 ms * | 39.33 ms (1.30x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.2%; Total throughput 9.4% / 0.8%; Throughput / GPU 9.4% / 0.8%; TTFT P99 33.0% / 5.2%; TPOT P99 22.6% / 2.0%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.1 GiB | 1,503.7 GiB (0.70x) |
| Total throughput | 43,718 tok/s | 35,987 tok/s (0.82x) |
| Throughput / GPU | 5,465 tok/s/GPU | 4,498 tok/s/GPU (0.82x) |
| TTFT P99 | 10,716.3 ms | 12,408.7 ms (1.16x) |
| TPOT P99 | 95.07 ms * | 108.88 ms (1.15x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.0%; Total throughput 8.9% / 0.2%; Throughput / GPU 8.9% / 0.2%; TTFT P99 9.6% / 1.2%; TPOT P99 17.3% / 0.0%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.1 GiB | 1,506.2 GiB (0.70x) |
| Total throughput | 48,816 tok/s | 36,403 tok/s (0.75x) |
| Throughput / GPU | 6,102 tok/s/GPU | 4,550 tok/s/GPU (0.75x) |
| TTFT P99 | 20,099.2 ms | 26,303.6 ms (1.31x) |
| TPOT P99 | 164.71 ms | 215.66 ms (1.31x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.7%; Total throughput 0.3% / 0.8%; Throughput / GPU 0.3% / 0.8%; TTFT P99 0.1% / 1.3%; TPOT P99 0.2% / 2.6%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.1 GiB | 1,511.2 GiB (0.71x) |
| Total throughput | 50,932 tok/s | 37,367 tok/s (0.73x) |
| Throughput / GPU | 6,366 tok/s/GPU | 4,671 tok/s/GPU (0.73x) |
| TTFT P99 | 39,671.5 ms | 53,364.2 ms (1.35x) |
| TPOT P99 | 314.14 ms | 215.26 ms (0.69x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.2% / 0.5%; Throughput / GPU 0.2% / 0.5%; TTFT P99 0.1% / 1.0%; TPOT P99 0.2% / 1.4%.
