# Campaign results

One summary per model family: final numbers vs baseline, recipe, plowrt commit, gate status and
the scratch paths that hold the raw evidence. Raw logs, JSON and captures stay in campaign scratch
outside the repo (CLAUDE.md, "Performance campaign evidence and promotion").

| Family | Summary | Qualified win |
|---|---|---|
| Gemma-4 (E4B, 12B, 26B-A4B, 31B) | [gemma4-h100/summary.md](gemma4-h100/summary.md) | 12B FP8 vs vLLM 0.28 FP8 KV: [comparison.md](gemma12b-fp8-20260930/comparison.md), [comparison.csv](gemma12b-fp8-20260930/comparison.csv); 31B FP8: [comparison.md](gemma31b-h100-20261010/comparison.md), [comparison.csv](gemma31b-h100-20261010/comparison.csv) |
| Gemma-4 26B-A4B (H100) | [gemma4-h100/summary.md](gemma4-h100/summary.md) | BF16 + FP8 16K vs vLLM 0.28: [comparison.md](gemma26b-h100-20261007/comparison.md), [comparison.csv](gemma26b-h100-20261007/comparison.csv); FP8 at max_ctx 131072: [comparison.md](gemma26b-h100-20261009/comparison.md), [comparison.csv](gemma26b-h100-20261009/comparison.csv); KV block store vs vLLM 0.29: production mix TTFT P99 0.68x / 0.71x, TPOT P99 0.55x / 0.64x; agentic16k c32 / c64 throughput 1.46x / 1.40x: [comparison.md](gemma26b-h100-20261023/comparison.md), [comparison.csv](gemma26b-h100-20261023/comparison.csv) |
| Gemma-4 BF16 on Xeon 6975P-C (CPU engine) | [comparison.md](gemma4-xeon6-bf16-20261006/comparison.md) | E2B, E4B, 12B vs vLLM 0.30 CPU: [comparison.md](gemma4-xeon6-bf16-20261006/comparison.md), [comparison.csv](gemma4-xeon6-bf16-20261006/comparison.csv); 26B-A4B gate miss, 31B pending |
| Xeon 6975P-C L2-resident BF16 experiment | [p0_baseline.md](xeon6-l2r-bf16-20261008/p0_baseline.md), [p1_residency.md](xeon6-l2r-bf16-20261008/p1_residency.md), [p2_real_bf16_layer.md](xeon6-l2r-bf16-20261008/p2_real_bf16_layer.md), [p3_sync.md](xeon6-l2r-bf16-20261008/p3_sync.md), [p4_kv_policy.md](xeon6-l2r-bf16-20261008/p4_kv_policy.md), [p5_serving_report.md](xeon6-l2r-bf16-20261008/p5_serving_report.md), [p6_multimodel_cluster.md](xeon6-l2r-bf16-20261008/p6_multimodel_cluster.md), [TRACKER.md](xeon6-l2r-bf16-20261008/TRACKER.md) | `PLOW_CPU_COMBINE=16` in the E2B / E4B Xeon 6 recipes (P5); go / no-go in the P5 report; P6: E2B-31B socket slices (TP, MoE head / experts) and the projected multi-socket pipeline: GO for a two-server dense pipeline, NO-GO for MoE / long-context full attention |
| Qwen3-ASR | [qwen3-asr-h100/summary.md](qwen3-asr-h100/summary.md) | none (no recorded baseline) |
| Qwen3-ASR 0.6B / 1.7B (RTX 5090) | [qwen3-asr-rtx5090/summary.md](qwen3-asr-rtx5090/summary.md) | none (no strict ASR report); same-box served tables vs vLLM 0.31.0: [summary.md](qwen3-asr-rtx5090/summary.md), [served.csv](qwen3-asr-rtx5090/served.csv) |
| Qwen3.8-27B (H100, BF16) | [comparison.md](qwen38-27b-h100/comparison.md) | TTFT P99 at c1 vs vLLM 0.28 (0.48x / 0.62x) and c8 greedy (0.39x, baseline spread); packed prefill for the hybrid chain lifted c8..c128 throughput 4-7% (0.58-0.88x) and cut c32 TTFT/TPOT P99 24%/7%; throughput and TPOT still lost in every cell; FP32 gate PASS at c1/c8/c32: [comparison.md](qwen38-27b-h100/comparison.md) |
| TTS (Veena, Chatterbox, Chatterbox MTL) | [tts-h100/summary.md](tts-h100/summary.md) | none (no strict speech report) |
| GLM-5.3-Flash (MI350X, TP8) | [glm53flash-mi350x-2026-10-12/summary.md](glm53flash-mi350x-2026-10-12/summary.md) | [comparison.md](glm53flash-mi350x-2026-10-12/comparison.md): TP8 isl4096/c8 1.05x, isl1024/c8 1.06x vLLM throughput, [comparison.csv](glm53flash-mi350x-2026-10-12/comparison.csv); accuracy gate PASS |

Each campaign keeps one `comparison.csv` for matched serving rows and a neighboring
`comparison.md` for qualified wins only; a final comparison is rendered by `campaign.py report`
([agent-tools.md](../agent-tools.md#final-performance-report-strict)).
