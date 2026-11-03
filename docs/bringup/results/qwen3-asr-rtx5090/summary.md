# Qwen3-ASR results summary (RTX 5090)

Plow and vLLM served on the same RTX 5090 and measured with the same client. There is no strict
`campaign.py report` for ASR (`serving_comparison.py` takes LLM grids), so the served tables come
straight from `scripts/asr/nvidia/served_bench.py`. Every cell and repeat is in
[served.csv](served.csv).

- Models: Qwen/Qwen3-ASR-0.6B `5eb14417` and Qwen/Qwen3-ASR-1.7B `7278e1e7`, BF16, TP1.
- GPU: RTX 5090 (GB202, sm_120a, 170 SMs, 32 GB GDDR7), driver 595.91.07, 575 W.
- Recipes: [0.6B](../../../../recipes/infervisor/qwen3-asr-0.6b/sm120a-rtx5090-tp1.toml),
  [1.7B](../../../../recipes/infervisor/qwen3-asr/sm120a-rtx5090-tp1.toml) (`e9ca1f0c`).
- Build: `e40bb100`, plowrt `e43bb62e…`, packets 0.6B `2a6a21fd…`, 1.7B `5f3e2641…`, 2026-10-22.
- Baseline: vLLM 0.31.0 (torch 2.13.0+cu129), `--served-model-name` only.
- Gate `asr_wer` PASS (73 clips at c1/c16/c64, 0 errors): 0.6B 4.261% (= H100), 1.7B 3.739%. The
  1.7B reads 3.913% (= H100) when batch timing flips 2 near-tie words.

## Decode rungs

`step_bench` at ctx 1024. Each time is the median over 3 processes of the per-process median of
48 steps; single outlier steps of 5-8 ms make 48-step means unreliable. The floor is the weight
and KV bytes per step at 1672 GB/s, the measured read bandwidth.

| Model | B=1 | B=4 | B=16 | B=32 |
|---|---:|---:|---:|---:|
| 0.6B, ms (floor) | 1.235 (0.78) | 1.343 | 2.109 (1.83) | 3.383 (2.96) |
| 0.6B, % of roofline | 63% | | 87% | 87% |
| 1.7B, ms (floor) | 2.521 (2.13) | 2.652 | 3.598 (3.18) | 4.805 (4.31) |
| 1.7B, % of roofline | 84% | | 88% | 90% |

The 0.6B B=1 rung is latency-bound. A step is a chain of about 290 small dependent ops, and on an
earlier build the gate and dispatch cost about 0.9 µs per op, which is half of the gap. The rest
is the B=1 FlashDecode and norms running well below bandwidth.

## Served

Each value is the mean of 2 interleaved repeats. c1/c16 use the 73-clip, 481 s manifest from
`scripts/asr/nvidia/get_audio.py`; c64/c128 use the 292-clip `manifest_x4` set.

### Qwen3-ASR-0.6B

| Concurrency (clips) | RTFx vLLM | RTFx plow | p50 vLLM / plow (ms) | p90 vLLM / plow (ms) | WER vLLM / plow |
|---|---:|---:|---:|---:|---:|
| c1 (73) | 87 | 173 (1.99x) | 70 / 34 | 106 / 58 | 4.26% / 4.17% |
| c16 (73) | 547 | 975 (1.78x) | 173 / 86 | 280 / 147 | 4.26% / 4.26% |
| c64 (292) | 743 | 1675 (2.25x) | 506 / 192 | 840 / 335 | 4.40% / 4.46% |
| c128 (292) | 787 | 1719 (2.18x) | 916 / 316 | 1282 / 601 | 4.33% / 4.46% |

Peak GPU memory: vLLM 30478 MiB, plow 29104 MiB.

### Qwen3-ASR-1.7B

The plow column is the PF_OPS 63 object (`ebc11371`, run fab8). In run fab9 the canonical PF_OPS
127 object (`648d3c8e`) served within run noise of it: RTFx c1/c16/c64/c128 +0.3/-1.1/0/-1.1%.
The fab9 vLLM repeats were noisy (c128 RTFx 510 and 915), so this table keeps fab8.

| Concurrency (clips) | RTFx vLLM | RTFx plow | p50 vLLM / plow (ms) | p90 vLLM / plow (ms) | WER vLLM / plow |
|---|---:|---:|---:|---:|---:|
| c1 (73) | 58 | 94 (1.61x) | 103 / 62 | 167 / 111 | 3.74% / 3.74% |
| c16 (73) | 410 | 518 (1.26x) | 209 / 169 | 380 / 266 | 3.91% / 3.74% |
| c64 (292) | 678 | 1034 (1.53x) | 554 / 322 | 788 / 562 | 3.85% / 3.87% |
| c128 (292) | 831 | 1059 (1.27x) | 807 / 528 | 1234 / 1050 | 3.83% / 3.83% |

Peak GPU memory: vLLM 30828 MiB, plow 30912 MiB.

The `plow_prev` rows in served.csv come from the previous recipes' packets (0.6B `9f3cc4f8`,
1.7B `7864185e`), measured in the same session. Against them, the 0.6B recipe and the 1.7B
PF_OPS 63 object gain RTFx at c1/c16/c64/c128 of +5.1/+5.2/+3.4/+2.0% (0.6B) and
+3.9/-0.9/+2.6/+0.9% (1.7B). The 1.7B c16 cell is within its repeat spread (505-530).

## Evidence

All on the RunPod network volume, under `/workspace/runs/`:

- `fab8-q06/`, `fab8-q17/` and `fab9-q17/`: served cells, server and client logs, `provenance.txt`.
- `v27a/` and `v27c/` (1.7B PF_OPS 127): step_bench. `ab.txt` holds means and `abm.txt` medians,
  both with token digests.
- `q06-fw-gate/`, `q17-fw-gate/` and `q17-fp-gate/`: WER gates.
