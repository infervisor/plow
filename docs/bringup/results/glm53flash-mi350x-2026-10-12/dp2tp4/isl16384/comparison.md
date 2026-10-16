# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-dp2tp4/isl16384`
- Infervisor results: `/home/lava/bench/glm53flash/plow-dp2tp4-v11/isl16384`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-dp-v11/gates.json` (sha256 b7b144543adb, packet 92ae9bd5840e, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 633f828df80d, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.7 GiB | 1,196.7 GiB (0.56x) |
| Total throughput | 8,609 tok/s | 6,538 tok/s (0.76x) |
| Throughput / GPU | 1,076 tok/s/GPU | 817 tok/s/GPU (0.76x) |
| TTFT P99 | 1,220.2 ms * | 933.4 ms (0.76x) |
| TPOT P99 | 8.59 ms | 12.68 ms (1.48x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 2.5%; Total throughput 5.3% / 0.6%; Throughput / GPU 5.3% / 0.6%; TTFT P99 68.1% / 2.2%; TPOT P99 0.2% / 0.9%.

**FLAGGED: spread > 10%: Baseline TTFT P99 68.1%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 633f828df80d, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.7 GiB | 1,202.8 GiB (0.56x) |
| Total throughput | 25,243 tok/s * | 18,481 tok/s (0.73x) |
| Throughput / GPU | 3,155 tok/s/GPU * | 2,310 tok/s/GPU (0.73x) |
| TTFT P99 | 4,660.4 ms * | 3,788.2 ms (0.81x) |
| TPOT P99 | 40.15 ms * | 48.51 ms (1.21x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.5%; Total throughput 13.6% / 0.0%; Throughput / GPU 13.6% / 0.0%; TTFT P99 44.1% / 1.1%; TPOT P99 36.4% / 0.1%.

**FLAGGED: spread > 10%: Baseline Total throughput 13.6%; Baseline Throughput / GPU 13.6%; Baseline TTFT P99 44.1%; Baseline TPOT P99 36.4%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 633f828df80d, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.9 GiB | 1,204.6 GiB (0.56x) |
| Total throughput | 37,378 tok/s | 25,020 tok/s (0.67x) |
| Throughput / GPU | 4,672 tok/s/GPU | 3,127 tok/s/GPU (0.67x) |
| TTFT P99 | 12,245.2 ms | 15,697.0 ms (1.28x) |
| TPOT P99 | 103.13 ms | 155.18 ms (1.50x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 1.2%; Total throughput 0.4% / 0.1%; Throughput / GPU 0.4% / 0.1%; TTFT P99 0.1% / 0.6%; TPOT P99 0.9% / 0.1%.

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 633f828df80d, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.9 GiB | 1,206.9 GiB (0.56x) |
| Total throughput | 39,646 tok/s | 28,935 tok/s (0.73x) |
| Throughput / GPU | 4,956 tok/s/GPU | 3,617 tok/s/GPU (0.73x) |
| TTFT P99 | 25,276.9 ms * | 31,059.2 ms (1.23x) |
| TPOT P99 | 207.87 ms * | 271.35 ms (1.31x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.8%; Total throughput 5.0% / 0.4%; Throughput / GPU 5.0% / 0.4%; TTFT P99 11.2% / 0.2%; TPOT P99 12.9% / 0.4%.

**FLAGGED: spread > 10%: Baseline TTFT P99 11.2%; Baseline TPOT P99 12.9%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 4 --data-parallel-size 2 --trust-remote-code --max-model-len 18560 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 633f828df80d, packet 92ae9bd5840e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,143.9 GiB | 1,211.7 GiB (0.57x) |
| Total throughput | 42,228 tok/s | 30,416 tok/s (0.72x) |
| Throughput / GPU | 5,278 tok/s/GPU | 3,802 tok/s/GPU (0.72x) |
| TTFT P99 | 46,882.0 ms | 63,821.7 ms (1.36x) |
| TPOT P99 | 376.01 ms | 516.46 ms (1.37x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.0% / 0.1%; Throughput / GPU 0.0% / 0.1%; TTFT P99 0.8% / 0.2%; TPOT P99 0.3% / 0.3%.
