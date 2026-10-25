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

## Production pass on the branch (2026-10-25; gates and strict reports per recipe, not win rows)

Every BF16/FP8 recipe worked on this branch was re-emitted from the final source, gated on the
branch binary with the store armed, and, where a vLLM 0.29 peer exists, run through
`campaign.py report`. Only the Gemma-4 26B FP8 rows above are qualified wins; the rest is
recorded here for the record, losses included.

| Recipe (packet) | Gate on the branch | Strict report vs vLLM 0.29 |
|---|---|---|
| gemma-4-26b-a4b FP8 (Kit 2 `b86ec26e4262`; re-emit `faaebced1d4a` keeps its `union`/`tuning`/weights) | c16 FP32 PASS on the re-emit: kl_trim 0.1332 vs 0.1242, top1_decisive 0.9582, cont 0.539 | host tier **auto** (the recipe default), prod mix: MATCHED + EQUIVALENT, 1.8/s tok/s 1.00x, TTFT P99 1.04x, TPOT P99 0.82x*; 2.7/s 1.00x / 1.52x* / 1.09x* (the tier pays at 10 s think; the win rows above run with it off) |
| gemma-4-26b-a4b BF16 (`475cb17f9986`, reproduced byte for byte) | c16 FP32 PASS: kl_mean 0.0899 vs 0.1005, kl_trim 0.0462 vs 0.0496, top1_decisive 0.9838 vs 0.9778 | not re-run |
| gemma-4-12b (`06da7686`, reproduced byte for byte from the re-keyed checkpoint) | c16 FP32 PASS (row below) | prod mix, fp8 KV both arms: MATCHED + EQUIVALENT and **loses**: 1.8/s tok/s 0.98x, TTFT P99 1.53x*, TPOT P99 1.65x; 2.7/s 0.95x / 2.62x* / 1.20x*, SLO met 54 % vs 96 % |
| qwen3.8-27b BF16 (`5f934c703d2c`) | prefix cache **on** c1/c8/c32 FP32 PASS: kl_trim 0.000436/0.000470/0.000456 vs 0.000338–0.000356, top1 1/0.9988/1 | agentic16k on vs off (same packet): c8 hit 82 %, out tok/s 180.6 vs 110.7, TTFT p50/p99 483/2333 vs 1721/3081 ms, TPOT 40.0/50.7 vs 58.5/81.8 ms; c32 hit 8–10 % (store capacity beside 48 live slots), within 4 % of off |
| qwen3.8-27b FP8 (`f71e777dea43`, decode ladder 1..48) | c1/c8/c32 FP32 PASS on default thresholds vs vLLM 0.29 hub-FP8 peers: kl_mean 0.0137/0.0148/0.0141 vs 0.0204/0.0213/0.0246, top1 0.9963/0.9951/0.9988 | ISL 1000/OSL 128: NOT MATCHED on model version by construction (plow re-quantizes BF16 per row, vLLM serves the hub's block-scaled FP8); c1 tok/s 0.51x, c8 0.22x, c32 0.23x; TPOT P99 1.9x / 5.2x / 4.8x. Serves where it refused before, not competitive |
| gemma-4-e4b (re-emit `ab2afc639357`, cuBLASLt decode from 1 row + generated prefill roles) | logit parity PASS after the kv-shared tail fix: top1 0.9948, kl_mean 7.5e-4, kl_max 0.0154 (tail route on, prefill-bucket path). Before the fix the same packet failed (top1 0.9754, kl_mean 0.0546, kl_max 7.22): only the first generated token differed, every later step bit-identical. Attribution (`e4b-arms`): prefix cache off, per-rung Lt algorithms off, segment graphs off and a repeat all fail with identical digits; `PLOW_PF_SHARED_TAIL=0` passes; the kit packet `3146fe3e` with the vendor attention off runs the tail on a decode rung and passes; the recipe with cuBLASLt decode from 48 rows (`e4b-3`) runs the tail on a decode rung and passes. Cause: the prefill-bucket tail path uploaded the chain's request table but not the attention's, so the sampled row attended over the first prompt token alone. | — |

Raw evidence (gpu03 NVMe scratch): `/opt/dlami/nvme/lava-tts/kvstore-scratch/{strict-g26-tier23,strict-g12-23,gemma-final-gates,qwenbf16-gate24,qwenbf16-prefix-ab,qwenfp8-ladder3b-run,qwenfp8-vllm-run,e4b-arms}`;
FP8 peer captures `/opt/dlami/nvme/lava-tts/qwen38/fp32gate/vllm029-fp8-c{1,8,32}.json`. `*` = repeat
spread over 10 %. The strict report keys the open-loop plan on a tokenizer digest (it keyed on the
path before, which rendered the 12B run NOT MATCHED), and `llm_grid.sh` records provenance for a
single-flag server argument again.

## Merge-prep checks on the other recipes (2026-10-25, branch `a3bf4162` vs main `9f16f187`; not strict rows)

Every H100 recipe with a packet on this box ran its own `[gates]` on the branch binary. The LLM
packets also ran the recipe `[bench]` on both binaries (random prompts, 32 per cell, 128 output
tokens, input 128/1024/4096 at c1/c8/c32, same packet and cuBLASLt 13.4 on both arms, no
contention; `campaign.py compare`, branch over main):

| Recipe (packet) | Gate on the branch | Bench, branch / main |
|---|---|---|
| gemma-4-26b-a4b FP8 (Kit 2 `b86ec26e4262`, store + host tier auto) | strict rows above; smoke at 10 s think, rate 2.7: hit 79.8 %, TTFT 66/687 ms, TPOT 14.4/82.9 ms, no errors | TTFT 0.70–0.99x, TPOT 0.96–1.00x, tok/s 1.00–1.12x |
| gemma-4-12b (`06da7686`, store + host tier auto) | FP32 c16 PASS: kl_trim 0.049 vs vLLM 0.065, top1_decisive 0.987 vs 0.982, over-0.1 111 vs 123, cont_frac 0.604 vs 0.574 | TTFT 0.36–0.99x, TPOT 0.98–1.00x, tok/s 1.00–1.07x |
| gemma-4-e4b (`da55906d`, store + host tier auto) | logit parity PASS: top1 0.9948, kl_mean 7.6e-4 (limits 0.98, 2e-3) | not benched (chat-kind bench) |
| qwen3.8-27b BF16 (`3fb7cefa`, prefix cache off per recipe) | not re-gated | 1.00x on every cell |
| qwen3.8-27b FP8 TMA (`6c31b3e8`, prefix cache off per recipe) | c1 and c8 captures equal the 2026-10-20 qualifying run digit for digit; the recipe's own thresholds fail that packet on main too (arch/26 B3) | c1 1.00x; c8/c32 incomplete on both arms (single-slot decode) |
| veena, orpheus, chatterbox, chatterbox-mtl (speech kit `6680acc8` bundles) | tts_cer PASS (Veena 0.011, Orpheus 0), s3gen_rel_l2 PASS | — |
| qwen3-asr 1.7B / 0.6B (re-emitted with the branch compiler; the kit's packets carry speech contract 0) | asr_wer PASS 3.913 % / 4.435 % (the kit baseline figures) | — |
| nemotron-3.5-asr | not run: the bundle is an ASR packet, the recipe gate expects a model packet | — |

The non-prefix path costs the same to the digit (Qwen rows); the prefix-on recipes gain TTFT at
c8/c32 because the boundary publish no longer copies on the engine thread. Raw evidence (gpu03
scratch): `/opt/dlami/nvme/lava-tts/kvstore-scratch/{gates,bench,asr}`,
`/home/lava/scratch/kvstore-gates/{e4b,g12,qwen-prod,r27-21-auto}`.

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
