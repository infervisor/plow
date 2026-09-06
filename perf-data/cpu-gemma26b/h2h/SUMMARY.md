# Gemma-4-26B-A4B head-to-head via the OpenAI API (2026-09-06)

Same box, same bench (`bench.py --requests 8 --max-tokens 64 --fresh-prompts`), one server at a time.
plow: `plowrt serve --cpu-threads 16` on the bf16 ladder blob (rungs 1/2/4/8, prefill buckets 128/512/1024),
commit 897e117 (AMX grouped-expert prefill). llama.cpp: see `../llamacpp/SUMMARY.md`. Raw: `g26b-bf16-*.md`.

TTFT / TPOT = mean ms. Lower is better. Bold = plow wins.

| workload | conc | in tok | plow bf16 TTFT | Q8_0 TTFT | Q4_K_M TTFT | plow bf16 TPOT | Q8_0 TPOT | Q4_K_M TPOT |
|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 34 | 800 | 739 | 649 | 83 | 56 | 50 |
| chat_short | 2 | 45 | **1147** | 1351 | 1226 | 130 | 84 | 80 |
| chat_short | 4 | 41 | **1943** | 2852 | 2508 | 210 | 115 | 107 |
| chat_short | 8 | 36 | **4287** | 7024 | 6492 | 347 | 119 | 105 |
| chat_long | 1 | 398 | **2779** | 4532 | 4187 | 85 | 56 | 52 |
| chat_long | 2 | 442 | **4167** | 6252 | 7009 | 154 | 159 | 125 |
| chat_long | 4 | 349 | **5133** | 13887 | 12936 | 268 | 126 | 119 |
| chat_long | 8 | 398–425 | **12496** | 28262 | 29015 | 577 | 130 | 118 |
| code | 1 | 365 | **2484** | 4221 | 3858 | 86 | 60 | 52 |
| code | 2 | 374 | **4018** | 8636 | 7914 | 141 | 112 | 90 |
| code | 4 | 401 | **5871** | 18089 | 16867 | 271 | 131 | 120 |
| code | 8 | 359 | **11427** | 27944 | 26328 | 451 | 132 | 119 |
| summarize | 1 | 1105 | **7667** | 12084 | 11237 | 90 | 63 | 56 |
| summarize | 2 | 1119 | **8972** | 24992 | 23241 | 232 | 102 | 105 |
| summarize | 4 | 1178 | **15335** | 54272 | 50453 | 485 | 159 | 146 |
| summarize | 8 | 1133 | **35315** | 56278 | 48027 | 765 | 227 | 190 |

Reading
* Prefill/TTFT: plow bf16 wins every cell with a real prompt (1.5–3.5×), including c=8.
* Decode/TPOT: plow bf16 loses everywhere — 83–90 ms vs 50–63 at c=1 (bf16 experts read ~7 GB/token;
  llama.cpp's 4/8-bit experts) and 3–5× at c=8 (rung-8 MoE decode reads every selected expert per row: up
  to 64 expert loads per step; llama.cpp batches the 8 rows through each expert once).
* Fixes queued: MXFP4 experts (emitter arm landed in 0890fae; the 14 GB twin needs disk space), and
  expert-deduplicated batched MoE decode (sort the B·k slots by expert, one weight pass per expert).
