# Gemma-4 E4B-it vs vLLM 0.29 (1x H100, 2026-10-27)

Production-readiness check on the final PR #108 build. **Result: production ready against vLLM
0.29 on the voice workload.** At c32 it wins throughput and both P99 latencies. At c1/c8 it is at
parity: throughput 1.01-1.03x, TPOT P99 1.01-1.02x. Two losses remain: c1 TTFT P99 (1.51x, on a
166 % plowrt repeat spread) and TTFT P50 at 200 concurrent calls, which is above the recipe's
128-row decode ladder.

- Infervisor: plowrt `bb9f389fd`, packet `046539c8a8a7` from
  [`sm90a-h100-tp1.toml`](../../../../recipes/infervisor/gemma-4-e4b/sm90a-h100-tp1.toml)
  (max_ctx 8192, decode ladder 1..128), server env = the recipe's `[serve]` env
  (`PLOW_VMM_PREFIX=1 PLOW_LT_RUNG_ALGOS=true`), Kit 2 cuBLASLt on the library path.
- Baseline: vLLM 0.29.0, BF16, `--max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 256
  --limit-mm-per-prompt {"image":0,"audio":0}`, prefix caching on (default), same client.
- Quality: FP32-reference gate (`llm_fp32_ref`, new for this recipe, below) PASS at c1/c8/c32 against
  vLLM 0.29 captured in the same lease. The `llm_logit_parity` gate (HF bf16) also passes on this
  packet: top1 0.9948, KL mean 7.5e-4.
- Raw evidence: `/opt/dlami/nvme/lava-tts/prod-compare/e4b-{grid,voice}/{plow,vllm}`,
  `e4b-grid/report-plow`, `e4bgate/{prompts.json,ref.json,gate}`.

## Strict report (`campaign.py report`, 1000/128 greedy, 2 repeats, MATCHED + EQUIVALENT)

| Cell | Total throughput | TTFT P99 | TPOT P99 | Peak memory |
|---|---:|---:|---:|---:|
| 1000/128 c1 | 1.01x | 1.51x * | 1.01x | **0.64x** * |
| 1000/128 c8 | **1.03x** | **0.70x** * | 1.02x | **0.65x** * |
| 1000/128 c32 | **1.11x** | **0.74x** | **0.93x** | **0.70x** * |

Bold = qualified win. `*` = repeat spread > 10 % (FLAGGED; direction only). The c1 TTFT P99 cell
sits on a 166 % plowrt spread and a 77 % vLLM spread (68.6 vs 103.4 ms P99 over 16 requests). The
peak-memory spread is the live-KV commit timing. All rows are in [comparison.csv](comparison.csv).

## Voice workload (`scripts/llm/gemma_voice_bench.sh`, both sides; not strict rows)

`vllm bench serve`, random 1000/128, ignore_eos, model sampling defaults, one repeat:

| conc | TTFT P50 ms vLLM / plowrt | TTFT P99 ms | TPOT P50 ms | TPOT P99 ms | out tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 36.5 / **16.7** | 65.6 / 155.6 | 5.94 / **5.82** | 5.95 / **5.83** | 161 / 167 |
| 8 | 139 / **46** | 3047 / **113** | 6.13 / 6.51 | 6.71 / 6.79 | 543 / **1156** |
| 32 | 361 / **53** | 523 / **380** | 8.16 / 8.99 | 10.21 / **9.25** | 2939 / **3308** |
| 64 | 484 / **79** | 1088 / **815** | 11.41 / 12.24 | 15.00 / **12.59** | 4078 / **4811** |
| 128 | 659 / **93** | 2124 / **1773** | 20.29 / 20.31 | 24.18 / **21.00** | 4987 / **5732** |
| 200 | 974 / 1359 | 3334 / 3388 | 30.19 / **20.37** | 35.23 / **21.71** | 5348 / **5796** |

Multi-turn sessions (`session_bench.py`, 6 turns, sampled; output lengths differ between the
stacks, so only latency compares), vLLM / plowrt: 64 calls later-turn TTFT P50/P99 45.6/94.4 vs
**16.5/25.3** ms, TPOT P50 8.63 vs 9.05 ms; 200 calls 210.6/450.1 vs **29.6/112.5** ms, TPOT P50
18.9 vs **17.7** ms. plowrt reuses 83 % of later-turn prompts.

Remaining losses:
- 200 concurrent requests: TTFT P50 974 vs 1359 ms. The decode ladder tops at 128 rows (recipe
  `PLOW_DECODE_BATCH_LADDER`, sized for the co-served voice stack's up to 128 concurrent calls), so
  72 of 200 requests queue for a slot. At the same concurrency, TPOT P99 is 0.62x and throughput
  1.08x. Not changed: closing it needs a re-emitted ladder above 128 plus its gate, and 200 is outside
  the recipe's design point.
- TPOT P50 at c8..c64 is 4-10 % behind vLLM. TPOT P99 is equal or better in every cell from c8 up.

## FP32-reference gate added for this recipe

E4B had only the HF bf16 logit-parity gate, and `campaign.py report` requires `llm_fp32_ref`. The
46-case Gemma prompt set (`fp32gate/prompts.json`, same tokenizer) was cut to the 27 cases that fit
max_ctx 8192 (prompt + max_new <= 8192: natural text 128/1024/4096, the six chat cases, the 4096
needles). The FP32 reference was built with `fp32_ref_gate.py reference`: 93 s, ref sha256
`a26cc565fc45`. vLLM 0.29 is the peer, captured in the gate's lease at each concurrency.

| conc | kl_mean plow / vLLM | trimmed | p95 | top1_decisive | continuation |
|---|---:|---:|---:|---:|---:|
| c1 | 0.00049 / 0.00066 | 0.00040 / 0.00059 | 0.0019 / 0.0025 | 1.0 / 1.0 | 0.825 / 0.781 |
| c8 | 0.00053 / 0.00068 | 0.00041 / 0.00058 | 0.0019 / 0.0025 | 1.0 / 1.0 | 0.811 / 0.744 |
| c32 | 0.00055 / 0.00072 | 0.00043 / 0.00057 | 0.0021 / 0.0025 | 1.0 / 1.0 | 0.854 / 0.803 |

`needle_min` is 0 for this gate. With 16 tokens, E4B opens every needle answer with a reasoning
preamble, so the FP32 reference itself answers 0 of 9 (`ref_correct` false on all nine), and so do
vLLM and plowrt. `needle_drop_max` (0) still holds plowrt to vLLM.
