# Gemma-4-26B-A4B on plow CPU (bf16 full model, 16 threads, Sapphire Rapids 8c/16t, 2026-09-06)

Blob: devgen ladder rungs 1/2/4/8, prefill buckets 128/512/1024. cpu_bench, batch 1, 4 decode steps.

| prompt | TTFT ms | prefill tok/s | decode ms/tok | commit |
|---|---|---|---|---|
| 64 | 831.9 | 76.9 | 77.6 | 897e117 (AMX grouped-expert prefill, ops 75/76) |
| 128 | 1122.0 | 114.1 | 94.5 | 897e117 |
| 512 | 3052.6 | 167.7 | 86.9 | 897e117 |
| 64 | 1376.5 | 46.5 | 77.9 | b0ecd65 (AVX-512 dots, 8 rows per pass) |
| 128 | 2182.9 | 58.6 | 77.5 | b0ecd65 |
| 512 | 7746.5 | 66.1 | 85.4 | b0ecd65 |

Chat (cpu_chat, Gemma-4 turn template): "What is the capital of France? Answer in one word." -> "Paris" (HF: Paris).
Blocks 0/1/5 match HF hidden states at bf16 precision (single-block oracle, plowc --block N).
