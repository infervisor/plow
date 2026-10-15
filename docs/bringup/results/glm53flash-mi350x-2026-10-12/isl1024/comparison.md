# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl1024`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v19/isl1024`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v19-int-g/gates.json` (sha256 8b94438e8baa, packet 540ba4fa4012, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 540ba4fa4012) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,132.3 GiB | 1,481.8 GiB (0.69x) |
| Total throughput | 1,189 tok/s | 1,013 tok/s (0.85x) |
| Throughput / GPU | 149 tok/s/GPU | 127 tok/s/GPU (0.85x) |
| TTFT P99 | 140.6 ms * | 78.4 ms (0.56x) * |
| TPOT P99 | 6.88 ms | 8.57 ms (1.25x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.1% / 2.2%; Total throughput 1.4% / 0.0%; Throughput / GPU 1.4% / 0.0%; TTFT P99 57.7% / 20.8%; TPOT P99 0.2% / 1.1%.

**FLAGGED: spread > 10%: Baseline TTFT P99 57.7%; Infervisor TTFT P99 20.8%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 540ba4fa4012) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,137.8 GiB | 1,488.8 GiB (0.70x) |
| Total throughput | 5,256 tok/s * | 4,413 tok/s (0.84x) |
| Throughput / GPU | 657 tok/s/GPU * | 552 tok/s/GPU (0.84x) |
| TTFT P99 | 1,764.9 ms * | 405.4 ms (0.23x) |
| TPOT P99 | 19.29 ms * | 16.33 ms (0.85x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.6% / 1.3%; Total throughput 54.0% / 1.6%; Throughput / GPU 54.0% / 1.6%; TTFT P99 173.4% / 2.4%; TPOT P99 96.3% / 3.3%.

**FLAGGED: spread > 10%: Baseline Total throughput 54.0%; Baseline Throughput / GPU 54.0%; Baseline TTFT P99 173.4%; Baseline TPOT P99 96.3%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 540ba4fa4012) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,490.7 GiB (0.70x) |
| Total throughput | 17,572 tok/s | 12,482 tok/s (0.71x) |
| Throughput / GPU | 2,197 tok/s/GPU | 1,560 tok/s/GPU (0.71x) |
| TTFT P99 | 718.8 ms * | 1,049.0 ms (1.46x) * |
| TPOT P99 | 15.44 ms | 22.58 ms (1.46x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.0%; Total throughput 3.8% / 2.5%; Throughput / GPU 3.8% / 2.5%; TTFT P99 25.5% / 22.6%; TPOT P99 1.2% / 7.5%.

**FLAGGED: spread > 10%: Baseline TTFT P99 25.5%; Infervisor TTFT P99 22.6%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 540ba4fa4012) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,493.2 GiB (0.70x) |
| Total throughput | 26,683 tok/s | 16,593 tok/s (0.62x) |
| Throughput / GPU | 3,335 tok/s/GPU | 2,074 tok/s/GPU (0.62x) |
| TTFT P99 | 1,157.3 ms | 1,885.8 ms (1.63x) * |
| TPOT P99 | 20.44 ms | 34.41 ms (1.68x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.7%; Total throughput 1.9% / 0.1%; Throughput / GPU 1.9% / 0.1%; TTFT P99 3.4% / 14.5%; TPOT P99 1.3% / 4.1%.

**FLAGGED: spread > 10%: Infervisor TTFT P99 14.5%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 540ba4fa4012) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,498.2 GiB (0.70x) |
| Total throughput | 34,927 tok/s | 24,476 tok/s (0.70x) |
| Throughput / GPU | 4,366 tok/s/GPU | 3,060 tok/s/GPU (0.70x) |
| TTFT P99 | 2,237.1 ms | 3,604.5 ms (1.61x) * |
| TPOT P99 | 31.28 ms | 47.27 ms (1.51x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.3% / 3.1%; Throughput / GPU 0.3% / 3.1%; TTFT P99 1.7% / 17.2%; TPOT P99 0.6% / 10.4%.

**FLAGGED: spread > 10%: Infervisor TTFT P99 17.2%; Infervisor TPOT P99 10.4%.**
