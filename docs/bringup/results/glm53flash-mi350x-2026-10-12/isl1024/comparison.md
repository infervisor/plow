# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl1024`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v18/isl1024`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v18-int/gates.json` (sha256 360cd843bdbd, packet 360814fc607b, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,132.3 GiB | 1,494.8 GiB (0.70x) |
| Total throughput | 1,189 tok/s | 825 tok/s (0.69x) |
| Throughput / GPU | 149 tok/s/GPU | 103 tok/s/GPU (0.69x) |
| TTFT P99 | 140.6 ms * | 117.2 ms (0.83x) * |
| TPOT P99 | 6.88 ms | 10.79 ms (1.57x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.1% / 2.2%; Total throughput 1.4% / 0.9%; Throughput / GPU 1.4% / 0.9%; TTFT P99 57.7% / 81.4%; TPOT P99 0.2% / 7.7%.

**FLAGGED: spread > 10%: Baseline TTFT P99 57.7%; Infervisor TTFT P99 81.4%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,137.8 GiB | 1,501.8 GiB (0.70x) |
| Total throughput | 5,256 tok/s * | 3,373 tok/s (0.64x) |
| Throughput / GPU | 657 tok/s/GPU * | 422 tok/s/GPU (0.64x) |
| TTFT P99 | 1,764.9 ms * | 394.7 ms (0.22x) * |
| TPOT P99 | 19.29 ms * | 21.34 ms (1.11x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.6% / 1.2%; Total throughput 54.0% / 2.1%; Throughput / GPU 54.0% / 2.1%; TTFT P99 173.4% / 10.8%; TPOT P99 96.3% / 13.9%.

**FLAGGED: spread > 10%: Baseline Total throughput 54.0%; Baseline Throughput / GPU 54.0%; Baseline TTFT P99 173.4%; Infervisor TTFT P99 10.8%; Baseline TPOT P99 96.3%; Infervisor TPOT P99 13.9%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,503.7 GiB (0.70x) |
| Total throughput | 17,572 tok/s | 10,747 tok/s (0.61x) |
| Throughput / GPU | 2,197 tok/s/GPU | 1,343 tok/s/GPU (0.61x) |
| TTFT P99 | 718.8 ms * | 1,083.9 ms (1.51x) * |
| TPOT P99 | 15.44 ms | 25.96 ms (1.68x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.0%; Total throughput 3.8% / 2.3%; Throughput / GPU 3.8% / 2.3%; TTFT P99 25.5% / 20.5%; TPOT P99 1.2% / 1.7%.

**FLAGGED: spread > 10%: Baseline TTFT P99 25.5%; Infervisor TTFT P99 20.5%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,506.2 GiB (0.70x) |
| Total throughput | 26,683 tok/s | 14,373 tok/s (0.54x) |
| Throughput / GPU | 3,335 tok/s/GPU | 1,797 tok/s/GPU (0.54x) |
| TTFT P99 | 1,157.3 ms | 1,985.1 ms (1.72x) * |
| TPOT P99 | 20.44 ms | 38.85 ms (1.90x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.7%; Total throughput 1.9% / 0.6%; Throughput / GPU 1.9% / 0.6%; TTFT P99 3.4% / 15.0%; TPOT P99 1.3% / 1.9%.

**FLAGGED: spread > 10%: Infervisor TTFT P99 15.0%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.8 GiB | 1,511.2 GiB (0.70x) |
| Total throughput | 34,927 tok/s | 21,592 tok/s (0.62x) |
| Throughput / GPU | 4,366 tok/s/GPU | 2,699 tok/s/GPU (0.62x) |
| TTFT P99 | 2,237.1 ms | 3,802.4 ms (1.70x) * |
| TPOT P99 | 31.28 ms | 52.14 ms (1.67x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.3% / 2.9%; Throughput / GPU 0.3% / 2.9%; TTFT P99 1.7% / 16.4%; TPOT P99 0.6% / 9.1%.

**FLAGGED: spread > 10%: Infervisor TTFT P99 16.4%.**
