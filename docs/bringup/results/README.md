# Campaign results

One summary per model family: final numbers vs baseline, recipe, plowrt commit, gate status and
the scratch paths that hold the raw evidence. Raw logs, JSON and captures stay in campaign scratch
outside the repo (CLAUDE.md, "Performance campaign evidence and promotion").

| Family | Summary | Qualified win |
|---|---|---|
| Gemma-4 (E4B, 12B, 26B-A4B, 31B) | [gemma4-h100/summary.md](gemma4-h100/summary.md) | 12B FP8 vs vLLM 0.28 FP8 KV: [comparison.md](gemma12b-fp8-20260930/comparison.md), [comparison.csv](gemma12b-fp8-20260930/comparison.csv) |
| Gemma-4 BF16 on Xeon 6975P-C (CPU engine) | [comparison.md](gemma4-xeon6-bf16-20261006/comparison.md) | E2B, E4B, 12B vs vLLM 0.30 CPU: [comparison.md](gemma4-xeon6-bf16-20261006/comparison.md), [comparison.csv](gemma4-xeon6-bf16-20261006/comparison.csv); 26B-A4B gate miss, 31B pending |
| Xeon 6975P-C L2-resident BF16 experiment (in progress) | [p0_baseline.md](xeon6-l2r-bf16-20261008/p0_baseline.md), [p1_residency.md](xeon6-l2r-bf16-20261008/p1_residency.md), [p2_real_bf16_layer.md](xeon6-l2r-bf16-20261008/p2_real_bf16_layer.md), [p3_sync.md](xeon6-l2r-bf16-20261008/p3_sync.md) | none yet (P0 baseline, P1 residency, P2 real layer, P3 sync) |
| Qwen3-ASR | [qwen3-asr-h100/summary.md](qwen3-asr-h100/summary.md) | none (no recorded baseline) |
| TTS (Veena, Chatterbox, Chatterbox MTL) | [tts-h100/summary.md](tts-h100/summary.md) | none (no strict speech report) |

Each campaign keeps one `comparison.csv` for matched serving rows and a neighboring
`comparison.md` for qualified wins only; a final comparison is rendered by `campaign.py report`
([agent-tools.md](../agent-tools.md#final-performance-report-strict)).
