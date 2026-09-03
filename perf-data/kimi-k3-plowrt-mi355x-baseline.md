# Kimi-K3 Plow baseline on MI355X

Measured 2026-09-03 on one 8×AMD Instinct MI355X node. The current C1 result
is the predeclared median-of-folds reading from the three exact cells in campaign
`k3-showdown-c1-7a54489-retry2`. The 2026-09-02 single-run values are
superseded.

## Contract

- TP8, native MXFP4 weights, BF16 KV, MTP/speculation and prefix caching off.
- Raw `/v1/completions`, random exact 8192-token inputs and exact 1024-token
  outputs, greedy sampling, `--ignore-eos`, infinite request rate, C1.
- Each fold used one discarded warmup and ten measured requests. All 30
  measured requests completed, with exactly 81,920 input and 10,240 output
  tokens per fold.
- Production B128 packet/interpreter with the B1 low-rung object available.
  Material controls were `--max-hold-ms 8 --slo-ms 250
  --max-queued-requests 0`. Resolved configuration had ragged chunking on,
  MLA PF v2 off, and no global queue.
- Artifact-set digest:
  `8d38f79756a3db6b6e78e5f028a659dadf29f2ab48e132efde97e0b820de51a7`.

## Three-fold result

| metric | Plow | vLLM 0.28 | gap |
|---|---:|---:|---:|
| duration | 683.97 s | 219.12 s | Plow 3.12× longer |
| output throughput | 14.97 tok/s | 46.73 tok/s | vLLM 3.12× higher |
| total token throughput | 134.74 tok/s | 420.60 tok/s | vLLM 3.12× higher |
| median TTFT | 3762.81 ms | 568.35 ms | Plow 6.62× longer |
| P90 / P99 TTFT | 3777.18 / 3787.16 ms | 570.28 / 572.01 ms | Plow 6.62× / 6.62× longer |
| median TPOT | 63.17 ms | 20.86 ms | Plow 3.03× longer |
| median / P90 / P99 ITL | 63.16 / 63.36 / 63.65 ms | 20.86 / 20.96 / 21.06 ms | Plow 3.03× / 3.02× / 3.02× longer |
| median E2E | 68.381 s | 21.910 s | Plow 3.12× longer |

This is endpoint performance, not an isolated-kernel measurement.

## Exact fold provenance

| fold | mean / median / P99 TTFT (ms) | mean / median / P99 TPOT (ms) | output tok/s | mean / median / P99 E2E (ms) |
|---|---:|---:|---:|---:|
| showdown-1 | 3766.66 / 3762.81 / 3787.16 | 63.19 / 63.17 / 63.31 | 14.97 | 68411.20 / 68381.18 / 68554.09 |
| showdown-2 | 3760.99 / 3762.55 / 3769.66 | 63.17 / 63.03 / 63.75 | 14.97 | 68389.02 / 68241.77 / 68966.26 |
| showdown-3 | 3768.48 / 3766.05 / 3801.66 | 63.18 / 63.18 / 63.25 | 14.97 | 68396.69 / 68398.63 / 68470.45 |

The JSON preserves every `cells.tsv` field, source log basename, artifact
digest, and config/tokenizer hashes. Headline values are medians of the three
fold statistics; P90 and duration values come
from the three raw client logs.

## Superseded result

The 2026-09-02 single run (600.45 s, 17.05 output tok/s, 3646.76 ms mean
TTFT, 55.13 ms mean TPOT) remains in the JSON `supersedes` record but is
not the current baseline.

Raw data: `perf-data/kimi-k3-plowrt-mi355x-c1.json`.
Comparator: `perf-data/kimi-k3-vllm-mi355x-c1.json`.
