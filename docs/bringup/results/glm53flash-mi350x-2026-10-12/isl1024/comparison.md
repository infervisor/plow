# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl1024`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v15a/isl1024`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v15a/gates.json` (sha256 03e60307d3ae, packet 0abfd01688cc, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 42c4efc2d63b, packet 0abfd01688cc) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,132.3 GiB | 1,213.3 GiB (0.57x) |
| Total throughput | 1,189 tok/s | 720 tok/s (0.61x) |
| Throughput / GPU | 149 tok/s/GPU | 90 tok/s/GPU (0.61x) |
| TTFT P99 | 140.6 ms * | 171.5 ms (1.22x) * |
| TPOT P99 | 6.88 ms | 12.10 ms (1.76x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.1% / 2.7%; Total throughput 1.4% / 0.8%; Throughput / GPU 1.4% / 0.8%; TTFT P99 57.7% / 89.5%; TPOT P99 0.2% / 1.0%.

**FLAGGED: spread > 10%: Baseline TTFT P99 57.7%; Infervisor TTFT P99 89.5%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 42c4efc2d63b, packet 0abfd01688cc) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,137.8 GiB | 1,220.3 GiB (0.57x) |
| Total throughput | 5,256 tok/s * | 2,431 tok/s (0.46x) |
| Throughput / GPU | 657 tok/s/GPU * | 304 tok/s/GPU (0.46x) |
| TTFT P99 | 1,764.9 ms * | 492.1 ms (0.28x) * |
| TPOT P99 | 19.29 ms * | 29.63 ms (1.54x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.6% / 1.5%; Total throughput 54.0% / 4.3%; Throughput / GPU 54.0% / 4.3%; TTFT P99 173.4% / 16.1%; TPOT P99 96.3% / 11.6%.

**FLAGGED: spread > 10%: Baseline Total throughput 54.0%; Baseline Throughput / GPU 54.0%; Baseline TTFT P99 173.4%; Infervisor TTFT P99 16.1%; Baseline TPOT P99 96.3%; Infervisor TPOT P99 11.6%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 42c4efc2d63b, packet 0abfd01688cc) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,222.2 GiB (0.57x) |
| Total throughput | 17,572 tok/s | 7,368 tok/s (0.42x) |
| Throughput / GPU | 2,197 tok/s/GPU | 921 tok/s/GPU (0.42x) |
| TTFT P99 | 718.8 ms * | 1,509.7 ms (2.10x) * |
| TPOT P99 | 15.44 ms | 37.92 ms (2.46x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.2%; Total throughput 3.8% / 2.5%; Throughput / GPU 3.8% / 2.5%; TTFT P99 25.5% / 14.8%; TPOT P99 1.2% / 1.6%.

**FLAGGED: spread > 10%: Baseline TTFT P99 25.5%; Infervisor TTFT P99 14.8%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 42c4efc2d63b, packet 0abfd01688cc) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,224.7 GiB (0.57x) |
| Total throughput | 26,683 tok/s | 9,599 tok/s (0.36x) |
| Throughput / GPU | 3,335 tok/s/GPU | 1,200 tok/s/GPU (0.36x) |
| TTFT P99 | 1,157.3 ms | 2,864.9 ms (2.48x) * |
| TPOT P99 | 20.44 ms | 57.79 ms (2.83x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.8%; Total throughput 1.9% / 1.1%; Throughput / GPU 1.9% / 1.1%; TTFT P99 3.4% / 12.6%; TPOT P99 1.3% / 1.3%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 42c4efc2d63b, packet 0abfd01688cc) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,229.7 GiB (0.57x) |
| Total throughput | 34,927 tok/s | 14,188 tok/s (0.41x) |
| Throughput / GPU | 4,366 tok/s/GPU | 1,773 tok/s/GPU (0.41x) |
| TTFT P99 | 2,237.1 ms | 5,604.3 ms (2.51x) * |
| TPOT P99 | 31.28 ms | 79.01 ms (2.53x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.3% / 2.8%; Throughput / GPU 0.3% / 2.8%; TTFT P99 1.7% / 12.9%; TPOT P99 0.6% / 4.6%.

**FLAGGED: spread > 10%: Infervisor TTFT P99 12.9%.**
