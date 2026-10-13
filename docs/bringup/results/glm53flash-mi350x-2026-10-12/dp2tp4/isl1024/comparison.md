# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-dp2tp4/isl1024`
- Infervisor results: `/home/lava/bench/glm53flash/plow-dp2tp4-v11/isl1024`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-dp-v11/gates.json` (sha256 b7b144543adb, packet 92ae9bd5840e, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ac22964825b9+dirty, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,136.2 GiB | 1,196.7 GiB (0.56x) |
| Total throughput | 952 tok/s | 682 tok/s (0.72x) |
| Throughput / GPU | 119 tok/s/GPU | 85 tok/s/GPU (0.72x) |
| TTFT P99 | 183.7 ms * | 118.0 ms (0.64x) |
| TPOT P99 | 8.52 ms | 12.49 ms (1.47x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.7% / 2.5%; Total throughput 0.7% / 0.5%; Throughput / GPU 0.7% / 0.5%; TTFT P99 38.5% / 8.3%; TPOT P99 0.1% / 1.3%.

**FLAGGED: spread > 10%: Baseline TTFT P99 38.5%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ac22964825b9+dirty, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,140.9 GiB | 1,202.9 GiB (0.56x) |
| Total throughput | 4,017 tok/s * | 2,561 tok/s (0.64x) |
| Throughput / GPU | 502 tok/s/GPU * | 320 tok/s/GPU (0.64x) |
| TTFT P99 | 1,654.5 ms * | 516.0 ms (0.31x) * |
| TPOT P99 | 22.45 ms * | 28.10 ms (1.25x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 1.2% / 1.5%; Total throughput 63.0% / 1.0%; Throughput / GPU 63.0% / 1.0%; TTFT P99 149.5% / 23.2%; TPOT P99 81.1% / 6.5%.

**FLAGGED: spread > 10%: Baseline Total throughput 63.0%; Baseline Throughput / GPU 63.0%; Baseline TTFT P99 149.5%; Infervisor TTFT P99 23.2%; Baseline TPOT P99 81.1%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ac22964825b9+dirty, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,153.9 GiB | 1,204.6 GiB (0.56x) |
| Total throughput | 13,516 tok/s | 5,901 tok/s (0.44x) |
| Throughput / GPU | 1,689 tok/s/GPU | 738 tok/s/GPU (0.44x) |
| TTFT P99 | 1,284.0 ms | 1,365.6 ms (1.06x) |
| TPOT P99 | 19.91 ms | 47.84 ms (2.40x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.2%; Total throughput 0.9% / 0.8%; Throughput / GPU 0.9% / 0.8%; TTFT P99 2.7% / 9.4%; TPOT P99 0.6% / 1.5%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ac22964825b9+dirty, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,153.9 GiB | 1,206.9 GiB (0.56x) |
| Total throughput | 20,483 tok/s | 9,668 tok/s (0.47x) |
| Throughput / GPU | 2,560 tok/s/GPU | 1,209 tok/s/GPU (0.47x) |
| TTFT P99 | 1,718.4 ms * | 2,588.5 ms (1.51x) |
| TPOT P99 | 26.18 ms | 58.04 ms (2.22x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.8%; Total throughput 5.3% / 0.2%; Throughput / GPU 5.3% / 0.2%; TTFT P99 23.9% / 7.2%; TPOT P99 2.8% / 0.3%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 3200 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ac22964825b9+dirty, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,154.0 GiB | 1,211.7 GiB (0.56x) |
| Total throughput | 26,679 tok/s | 13,287 tok/s (0.50x) |
| Throughput / GPU | 3,335 tok/s/GPU | 1,661 tok/s/GPU (0.50x) |
| TTFT P99 | 3,341.2 ms | 5,083.4 ms (1.52x) |
| TPOT P99 | 40.61 ms | 84.19 ms (2.07x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.9% / 1.5%; Throughput / GPU 0.9% / 1.5%; TTFT P99 0.7% / 5.6%; TPOT P99 0.3% / 1.3%.
