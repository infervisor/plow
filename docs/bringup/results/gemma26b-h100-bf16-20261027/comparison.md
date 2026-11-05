# Gemma-4-26B-A4B-it BF16 vs vLLM 0.29 (1x H100, 2026-10-27)

Production-readiness check of the BF16 packet on the final PR #108 build. **Result: not production
ready against vLLM 0.29. Every measured cell is a loss.** All cells are `campaign.py report`
MATCHED + EQUIVALENT; every rendered row is in [comparison.csv](comparison.csv).

- Infervisor: plowrt `bb9f389fd` (final PR #108 runtime; later commits are recipe and doc only),
  packet `f72d0330f9d8` from
  [`sm90a-h100-tp1-bf16.toml`](../../../../recipes/infervisor/gemma-4-26b-a4b/sm90a-h100-tp1-bf16.toml)
  (max_ctx 16384, decode ladder 1..32), server env = the recipe's `[serve.env]`, Kit 2 cuBLASLt on the
  library path.
- Baseline: vLLM 0.29.0, BF16 weights and KV, `--max-model-len 16384 --gpu-memory-utilization 0.93
  --max-num-seqs 256`, prefix caching on (default), FLASH_ATTN (default; the venv needs
  `PYTHONPATH=<torch site-packages>/nvidia_cutlass_dsl/dsl_packages` or the engine fails to import
  `cutlass`). Same client, 2 repeats per cell, one server per arm, same seeds.
- Quality: FP32-reference gate PASS on the served packet (bin43 gate, c16).
- Raw evidence (campaign scratch, outside the repo): `/opt/dlami/nvme/lava-tts/prod-compare/`
  `g26bf16-{prod,grid}/{plow,vllm,report-plow}`, tunes `g26bf16-prod-t{1,2,4}`.

| Cell | Total throughput | TTFT P99 | TPOT P99 | Peak memory |
|---|---:|---:|---:|---:|
| prod mix, 1.8 sessions/s | 0.96x | 13.27x * | 2.53x | 1.04x |
| prod mix, 2.7 sessions/s | refused (plowrt failed requests) | | | |
| 1000/128 c1 greedy | 0.90x | 1.45x * | 1.44x * | 0.99x |
| 1000/128 c8 greedy | 0.79x | 2.38x * | 1.50x * | 1.02x |
| 1000/128 c32 greedy | 0.62x | 3.52x | 1.80x | 1.04x |

Ratios Infervisor / vLLM; `*` = repeat spread > 10% (FLAGGED). Supplementary at 1.8 sessions/s,
vLLM / Infervisor: goodput (TTFT <= 2 s, TPOT <= 100 ms) 7.28 / 6.19 req/s, SLO met 100 % / 88.2 %,
TTFT P50 110 / 385 ms, TPOT P50 15.1 / 49.5 ms, mean in-flight 7.4 / 23.5. At 2.7 sessions/s
plowrt saturates (repeat 1: goodput 70 %, TTFT P99 31 s; repeat 2: 57 errors, 80 failed requests,
goodput 43 %) while vLLM holds 97-100 % goodput (TTFT P99 435 / 822 ms), so `campaign.py report`
refuses the cell; it is a loss.

Diagnosis (PLOW_PF_PACKLOG, packlog_audit at 1.8 sessions/s): the step itself is slower, not the
scheduling. Decode at 7-16 rows runs 14-15 ms per step (vLLM's engine-step metric: 8.2 ms at c8,
12.6 ms at c32; plowrt 18-21 ms at 32 rows), a 128-row prefill launch takes ~29 ms, and each decode
row riding a prefill launch adds ~1.5 ms. Prefill is 44-46 % and decode 45 % of the wall clock, so
sessions queue (in-flight 23 vs 7). This is the BF16 MoE decode and prefill step gap; it belongs to
kernel work, not the recipe.

Recipe-level tunes, prod mix 1.8 sessions/s, one repeat each (current recipe: TTFT P50/P99
388/5612 and 381/4896 ms, TPOT P50/P99 44/116 and 55/120 ms, goodput 88.5 / 87.9 %):

| Tune (on top of `[serve.env]`) | TTFT P50 / P99 ms | TPOT P50 / P99 ms | Goodput |
|---|---:|---:|---:|
| T1: FP8 recipe's serving set (`PLOW_VMM_PREFIX=1 PLOW_VMM_LIVE_RINGS=1 PLOW_VMM_RING_PARK=8 PLOW_PF_ADMIT_OVERLAP=1 PLOW_DECODE_PIPELINE=1 PLOW_PF_ITL_BUDGET_MS=60 PLOW_PF_ITL_FORCE=0 PLOW_VMM_STRICT_PUBLISH=1 PLOW_KV_POOL_MIB=2048`) | 329 / 3669 | 37.1 / 98.5 | 96.0 % |
| T1 with ITL budget 30 ms | 445 / 7324 | 34.1 / 72.9 | 90.3 % |
| `PLOW_TOKEN_BATCH=0` (no riders) | 13457 / 32981 | 87.3 / 116.6 | 9.8 % |

T1 is the best measured environment (at 2.7 sessions/s, one repeat: goodput 74.2 % vs 70.3 %, TTFT
P99 20.5 vs 31.0 s, no failed requests), but it is still far behind vLLM (TTFT P99 ~9x) and was
measured on a single repeat without a gate re-run, so the recipe is unchanged. Its next steps
(two repeats, closed-loop grid, FP32 gate) were withdrawn to give the GPU to the BF16 decode kernel work.
