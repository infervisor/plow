# Gemma-4-26B-A4B-it comparison, KV block store (1x H100, vLLM 0.29)

Qualified wins only: `campaign.py report` MATCHED + EQUIVALENT on the open-loop production mix
(`--prod-fidelity`, think lognormal 10 s), where total throughput is pinned by the arrival rate and
the wins are the latency rows. Every matched row is in [comparison.csv](comparison.csv).

- Infervisor: plowrt `f7eba863` (branch `kv-block-store`: the KV block store is the only boundary
  storage, `Prefix` blocks hold every full-layer row, nothing is multi-mapped;
  `docs/arch/26-kv-block-store.md` B.6a), FP8 packet `b86ec26e4262` from
  [`sm90a-h100-tp1-fp8.toml`](../../../../recipes/infervisor/gemma-4-26b-a4b/sm90a-h100-tp1-fp8.toml)
  (Kit 2 assets, max_ctx 16384). Serve defaults from the packet; server env
  `PLOW_VMM_LIVE_RINGS=1 PLOW_VMM_RING_PARK=8 PLOW_PF_ADMIT_OVERLAP=1 PLOW_DECODE_PIPELINE=1
  PLOW_PF_INTERLEAVE=0 PLOW_PF_ITL_FORCE=0 PLOW_VMM_STRICT_PUBLISH=1 PLOW_PF_ITL_BUDGET_MS=60
  PLOW_VMM_HOST_CACHE_MIB=0` (the host tier under the store costs TPOT at 10 s think, below) and the
  Kit 2 cuBLASLt 13.4 on the library path.
- Baseline: vLLM 0.29.0, same client, `fp8` KV, `--max-model-len 16384 --gpu-memory-utilization
  0.93`, prefix caching on. 2 repeats per cell, one server per arm, same seeds per repeat.
- Quality: FP32-reference gate PASS on the served packet (kl_mean 0.2034 vs vLLM 0.202,
  kl_trim_mean 0.1239 vs 0.1242, top1_decisive 0.9622, cont_frac 0.5348, needle 1.0).
- Raw evidence (gpu03 scratch, outside the repo): `/home/lava/scratch/kvstore-strict/{plow3,vllm2,gate,report3}`;
  replays `/home/lava/scratch/kvstore-gates/{r27-7-nohost,long7-nohost,long10-auto-end,r27-10-auto}`;
  engine-thread audit `packlog-{store,oldsnap}/packlog.txt`.

| Cell | Precision | Total throughput | TTFT P99 | TPOT P99 |
|---|---|---:|---:|---:|
| prod mix, 1.8 sessions/s | FP8 | 1.00x | 0.68x | 0.55x |
| prod mix, 2.7 sessions/s | FP8 | 1.01x * | 0.71x * | 0.64x * |
| agentic16k c32 (closed loop) | FP8 | 1.46x | 0.48x | 0.58x |
| agentic16k c64 (closed loop) | FP8 | 1.40x | 0.84x | 0.60x |

`*` = repeat spread > 10% (FLAGGED; direction only; the 2.7 cell's throughput spread is the
arrival process, identical on both arms). Supplementary, Baseline / Infervisor: TTFT P50 69.7 /
36.6 ms and TPOT P50 10.48 / 5.09 ms at 1.8; 119.3 / 53.3 ms and 16.19 / 8.37 ms at 2.7; cached
prompt tokens 80.7 % / 78.6 % (Infervisor); SLO met 100 % on both. Agentic16k (10 turns growing to
15,600 tokens, 128-token replies, closed loop; `llm_grid.sh --agentic`, raw evidence
`/home/lava/scratch/kvstore-agentic/{plow,vllm,report}`): server prefix hit 84.4 % / 84.5 % (Infervisor)
vs 83.2-83.5 % (vLLM); the c32 baseline TTFT P99 spread is flagged (34 %), direction only.

## Prefix cache retention on the production replay (not strict rows)

Ten-minute production-fidelity replays on gpu03, same stream per arm, sampled decoding
(`long_plow_arm.sh`; vLLM numbers from the same harness on 2026-10-22):

| Think median | Arm | Hit rate | TTFT p50/p99 ms | TPOT p50/p99 ms | Goodput |
|---|---|---:|---:|---:|---:|
| 10 s, rate 2.7 | store, no host tier (`f7eba863`) | 78.6 % | 53 / 358 | 10.1 / 37.7 | 100 % |
| 10 s, rate 2.7 | store + host tier (`120a2202`, auto, prefetch restores, admission hold) | 79.7 % | 65 / 707 | 16.5 / 89.2 | 99.5 % |
| 10 s, rate 2.7 | snapshot path (`a803cf92`, store off) | 74.6 % | 66 / 551 | 13.9 / 46.6 | 100 % |
| 10 s, rate 2.7 | vLLM 0.29 | 75.6 % | 119 / 592 | 16.9 / 61.6 | |
| 20 s, rate 1.8 | store, no host tier (`f7eba863`) | 71.7 % | 58 / 317 | 6.0 / 29.5 | 100 % |
| 20 s, rate 1.8 | store + host tier (`9d8fad4b`, auto) | 78.4 % | 53 / 446 | 8.1 / 41.1 | 100 % |
| 20 s, rate 1.8 | snapshot path (`a803cf92`, store off) | 67.1 % | 63 / 408 | 6.8 / 33.8 | 100 % |
| 20 s, rate 1.8 | vLLM 0.29 | 72.6 % | 100 / 489 | 12.2 / 42.1 | 99.6 % |
| 45 s, rate 1.8 | store, no host tier (`f7eba863`) | 58.3 % | 87 / 401 | 6.3 / 34.6 | 100 % |
| 45 s, rate 1.8 | store + host tier (`120a2202`, auto, session-end demotion) | 71.7 % | 70 / 376 | 6.5 / 31.8 | 100 % |
| 45 s, rate 1.8 | snapshot path (`a803cf92`, store off) | 48.0 % | | | |
| 45 s, rate 1.8 | vLLM 0.29 | 60.6 % | 147 / 569 | 13.5 / 53.0 | |

Without a host tier the store is within the ideal's reach at 10 s think (ideal 78.9 %), 7 points
under it at 20 s (ideal 78.5 %, vLLM 0.9 points ahead) and capacity-bound at 45 s think (the LRU drops boundaries at a
median age of 45 s, the think-time median). With the host tier it holds 71.7 % against an ideal
of 73.9 % (the rest is session-end demotion) at 45 s think and 78.4 % at 20 s, with lower TTFT
and TPOT than vLLM at both and than the store without the tier at 45 s; at 10 s think the tier
adds one point (79.7 % vs 78.6 %) and costs TTFT 65/707 ms and TPOT 16.5/89 ms (a per-tick cost
under heavy spilling, amplified by the speculation policy at the larger in-flight batch;
`docs/arch/26-kv-block-store.md` B.6a), so the strict rows above run without it. The recipe
keeps the tier at its default (auto); the production setting depends on the deployment's think
time, and production's is about 45 s. Raw evidence for the tier rows:
`/home/lava/scratch/kvstore-gates/{r27-15-auto,t20-12-auto,long15-auto-end}`.

## Strict tables

## q1800.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | hf-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8 | Same as baseline |
| Input / output length | open-loop mix: 4 system prompts (lognormal median 2000), turns geometric mean 5 max 20, first message lognormal(3600, 0.35), tool output lognormal(100, 0.8), context cap 8192 / output lognormal(46, 0.6) in [16, 1024] tokens, ignore_eos | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, Poisson 1.8 sessions/s, think lognormal(10 s, 0.8) <= 120 s, 300 s (measured 75-275 s), session header on / open loop (achieved concurrency in the supplementary note), seeds 1252723/1260642 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.29.0 (--gpu-memory-utilization 0.93 --max-num-seqs 256 --tokenizer /opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it --kv-cache-dtype fp8 --max-model-len 16384 --tensor-parallel-size 1) | Infervisor (plowrt f7eba863, packet b86ec26e4262) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 74.6 GiB | 77.7 GiB (1.04x) |
| Total throughput | 42,394 tok/s | 42,485 tok/s (1.00x) |
| Throughput / GPU | 42,394 tok/s/GPU | 42,485 tok/s/GPU (1.00x) |
| TTFT P99 | 369.5 ms | 251.0 ms (0.68x) |
| TPOT P99 | 32.60 ms | 17.96 ms (0.55x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.2%; Total throughput 7.8% / 8.2%; Throughput / GPU 7.8% / 8.2%; TTFT P99 2.4% / 6.7%; TPOT P99 6.0% / 4.9%.

Supplementary (outside the strict table; goodput = requests meeting TTFT <= 2000 ms and TPOT <= 100 ms per second, measured window), Baseline / Infervisor: goodput 7.328 req/s / 7.345 req/s; SLO met 100.0% / 100.0%; requests 7.328 req/s / 7.345 req/s; mean in-flight 5.5 / 2.7; mean live sessions 81.5 / 79.0; TTFT P50 69.7 ms / 36.6 ms; TPOT P50 10.48 ms / 5.09 ms; cached prompt tokens - / 80.7%.

## q2700.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | hf-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8 | Same as baseline |
| Input / output length | open-loop mix: 4 system prompts (lognormal median 2000), turns geometric mean 5 max 20, first message lognormal(3600, 0.35), tool output lognormal(100, 0.8), context cap 8192 / output lognormal(46, 0.6) in [16, 1024] tokens, ignore_eos | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, Poisson 2.7 sessions/s, think lognormal(10 s, 0.8) <= 120 s, 300 s (measured 75-275 s), session header on / open loop (achieved concurrency in the supplementary note), seeds 1370623/1378542 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.29.0 (--gpu-memory-utilization 0.93 --max-num-seqs 256 --tokenizer /opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it --kv-cache-dtype fp8 --max-model-len 16384 --tensor-parallel-size 1) | Infervisor (plowrt f7eba863, packet b86ec26e4262) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 74.6 GiB | 77.8 GiB (1.04x) |
| Total throughput | 56,172 tok/s * | 56,494 tok/s (1.01x) * |
| Throughput / GPU | 56,172 tok/s/GPU * | 56,494 tok/s/GPU (1.01x) * |
| TTFT P99 | 485.2 ms | 344.3 ms (0.71x) * |
| TPOT P99 | 57.89 ms * | 37.02 ms (0.64x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 13.4% / 13.4%; Throughput / GPU 13.4% / 13.4%; TTFT P99 3.9% / 11.0%; TPOT P99 18.4% / 30.7%.

**FLAGGED: spread > 10%: Baseline Total throughput 13.4%; Infervisor Total throughput 13.4%; Baseline Throughput / GPU 13.4%; Infervisor Throughput / GPU 13.4%; Infervisor TTFT P99 11.0%; Baseline TPOT P99 18.4%; Infervisor TPOT P99 30.7%.**

Supplementary (outside the strict table; goodput = requests meeting TTFT <= 2000 ms and TPOT <= 100 ms per second, measured window), Baseline / Infervisor: goodput 9.422 req/s / 9.477 req/s; SLO met 100.0% / 100.0%; requests 9.428 req/s / 9.477 req/s; mean in-flight 11.0 / 5.9; mean live sessions 104.1 / 99.3; TTFT P50 119.3 ms / 53.3 ms; TPOT P50 16.19 ms / 8.37 ms; cached prompt tokens - / 78.6%.

## a32.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | hf-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8 | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 32 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.29.0 (--gpu-memory-utilization 0.93 --max-num-seqs 256 --tokenizer /opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it --kv-cache-dtype fp8 --max-model-len 16384 --tensor-parallel-size 1) | Infervisor (plowrt f7eba863, packet b86ec26e4262) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 74.6 GiB | 77.8 GiB (1.04x) |
| Total throughput | 59,696 tok/s | 87,206 tok/s (1.46x) |
| Throughput / GPU | 59,696 tok/s/GPU | 87,206 tok/s/GPU (1.46x) |
| TTFT P99 | 2,147.3 ms * | 1,033.8 ms (0.48x) |
| TPOT P99 | 50.54 ms | 29.29 ms (0.58x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 1.4% / 0.4%; Throughput / GPU 1.4% / 0.4%; TTFT P99 34.2% / 7.5%; TPOT P99 2.0% / 1.1%.

**FLAGGED: spread > 10%: Baseline TTFT P99 34.2%.**

## a64.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | hf-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8 | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 64 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.29.0 (--gpu-memory-utilization 0.93 --max-num-seqs 256 --tokenizer /opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it --kv-cache-dtype fp8 --max-model-len 16384 --tensor-parallel-size 1) | Infervisor (plowrt f7eba863, packet b86ec26e4262) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 74.6 GiB | 77.8 GiB (1.04x) |
| Total throughput | 77,115 tok/s | 108,105 tok/s (1.40x) |
| Throughput / GPU | 77,115 tok/s/GPU | 108,105 tok/s/GPU (1.40x) |
| TTFT P99 | 2,274.9 ms | 1,918.1 ms (0.84x) |
| TPOT P99 | 80.67 ms | 48.02 ms (0.60x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 0.9% / 2.2%; Throughput / GPU 0.9% / 2.2%; TTFT P99 1.2% / 8.0%; TPOT P99 1.2% / 3.1%.
