# Gemma-4-26B-A4B-it BF16 vs vLLM 0.29 (1x H100, 2026-10-29)

Standing after the decode router windows moved to light launches (branch `g26-stage-fusion`) on top
of the KV store quiesce fix (prod-next `949e3fdb9`). Both cells are `campaign.py report`
MATCHED + EQUIVALENT; every rendered row is in [comparison.csv](comparison.csv).

- Infervisor: plowrt `65bfae769`, packet `7c33558ca3e7` emitted at `afe7c9eaf` from
  [`sm90a-h100-tp1-bf16.toml`](../../../../recipes/infervisor/gemma-4-26b-a4b/sm90a-h100-tp1-bf16.toml)
  (max_ctx 16384, decode ladder 1..32), server env = the recipe's `[serve.env]`, Kit 2 cuBLASLt on
  the library path. FP32 gate PASS (kl_trim 0.0472 vs vLLM 0.0496, top1_decisive 0.9798).
- Baseline: the existing vLLM 0.29.0 arms of 2026-10-27 (BF16 weights and KV,
  `--max-model-len 16384 --gpu-memory-utilization 0.93 --max-num-seqs 256`), not recaptured.
- Traffic: the production mix (`llm_grid.sh --prod`, prod fidelity), 1.8 and 2.7 sessions/s, 300 s,
  2 repeats per cell.

| Cell | Total throughput | TTFT P99 | TPOT P99 | SLO met (supplementary) |
|---|---|---|---|---|
| 1.8 sessions/s | 42,072 vs 41,394 tok/s (0.98x) | 395.8 vs 1,965.4 ms (4.97x) | 46.67 vs 114.84 ms (2.46x) | 100.0% vs 97.4% |
| 2.7 sessions/s | 55,409 vs 55,133 tok/s (1.00x) | 628.5 vs 2,589.6 ms (4.12x) | 98.42 vs 102.47 ms (1.04x) | 98.4% vs 95.2% |

Change from the 2026-10-28 standing (plowrt `d18e6a707`): 2.7 sessions/s SLO met 84.6% -> 95.2%
and tok/s 0.98x -> 1.00x; 1.8 sessions/s SLO met 99.8% -> 97.4% (repeat 1 had a 73.9% prefix hit
rate against 79.7% before; TTFT P99 spread 27%). The remaining gap is TTFT P99 at both rates:
plowrt holds a mean in-flight of 14.3 / 25.8 sessions against vLLM's 7.4 / 17.8 because its decode
step at the same batch is 1.1-1.2x vLLM's and its prefill launches are serialized with decode.
