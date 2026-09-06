# llama.cpp baseline — Gemma-4-26B-A4B-it (2026-09-06)

Box: Sapphire Rapids 8c/16t, 58 GB. llama.cpp source build (`/home/lava/llamacpp/src`), GGUF converted from the HF
bf16 checkpoint (`convert_hf_to_gguf.py --outtype q8_0`, then `llama-quantize --allow-requantize … Q4_K_M`).
Server: `llama-server -t 16 -c 16384 -np 4 -fa on --reasoning-budget 0 --reasoning-format none` (thinking off —
with the default template the whole reply lands in `reasoning_content` and the bench sees zero content tokens;
`-c 4096 -np 4` gave 1024-token slots and failed the 1100-token summarize prompts). Bench:
`tools/bench-api/bench.py --requests 8 --max-tokens 64 --fresh-prompts` (distinct prompts per cell, so the server's
prefix cache cannot help). Raw tables: `results-q8_0*.md`, `results-q4_k_m-*.md` (chunked runs, same settings).

TTFT/TPOT in ms, mean/p50/p90. `in tok` = mean prompt tokens.

| workload | conc | in tok | Q8_0 TTFT | Q8_0 TPOT | Q4_K_M TTFT | Q4_K_M TPOT |
|---|---|---|---|---|---|---|
| chat_short | 1 | 34 | 739/783/817 | 56/54/60 | 649/668/729 | 50/50/51 |
| chat_short | 2 | 45 | 1351/1546/1559 | 84/88/93 | 1226/1458/1461 | 80/85/86 |
| chat_short | 4 | 41 | 2852/2862/2864 | 115/117/117 | 2508/2567/2567 | 107/108/108 |
| chat_short | 8 | 36 | 7024/11776/11777 | 119/125/125 | 6492/10881/10882 | 105/107/107 |
| chat_long | 1 | 398 | 4532/5090/5649 | 56/56/56 | 4187/4827/5238 | 52/52/54 |
| chat_long | 2 | 442 | 6252/6873/10645 | 159/170/224 | 7009/7357/9875 | 125/109/165 |
| chat_long | 4 | 349 | 13887/14092/14213 | 126/127/127 | 12936/13104/13213 | 119/122/122 |
| chat_long | 8 | 398–425 | 28262/43811/43940 | 130/131/131 | 29015/42753/43020 | 118/122/122 |
| code | 1 | 365 | 4221/4304/5061 | 60/59/61 | 3858/3905/4611 | 52/52/52 |
| code | 2 | 374 | 8636/8994/9525 | 112/120/120 | 7914/8291/8804 | 90/85/106 |
| code | 4 | 401 | 18089/18764/18765 | 131/132/132 | 16867/17597/17598 | 120/121/121 |
| code | 8 | 359 | 27944/40902/41038 | 132/133/133 | 26328/37841/37842 | 119/120/120 |
| summarize | 1 | 1105 | 12084/13646/15377 | 63/63/64 | 11237/12650/14341 | 56/56/56 |
| summarize | 2 | 1119 | 24992/25417/27488 | 102/103/104 | 23241/23613/25631 | 105/113/116 |
| summarize | 4 | 1178 | 54272/58716/59092 | 159/157/159 | 50453/54424/54782 | 146/144/146 |
| summarize | 8 | 1133 | 56278/74128/86784 | 227/193/323 | 48027/71653/77715 | 190/148/236 |

Notes
* c=1 decode: Q8_0 56 ms/tok, Q4_K_M 50–52 — the 4-bit weights buy only ~10 %, so llama.cpp's MoE decode on
  this box is not weight-bandwidth-bound (active bf16-equivalent bytes/token ≈ 7 GB → ~70 ms at 100 GB/s; Q4
  should allow ~20 ms).
* c=1 prefill ≈ 85–95 tok/s at 400–1100 prompt tokens (TTFT includes one decode step).
* Q8_0 chat cells in `results-q8_0.md` were measured with `-c 4096 -np 4` (same slots, same numbers otherwise).

## Correction: the earlier 26B llama.cpp run used 4 server slots (14:2x)

`ls_start.sh` passed `-np 4` while the bench went to concurrency 8, so at c=8 only four requests
decoded concurrently and llama.cpp's per-token latency was measured on half the load it was being
compared against. That is not apples to apples, and it biased llama.cpp favourably on TPOT (and
unfavourably on TTFT). Re-run with `-np 8`, matching plow's 8 slots and vLLM's `--max-num-seqs 8`;
the GPT-OSS and 12B llama.cpp baselines already used 8 (`serve.sh --parallel 8`) and are unaffected.

| workload | conc | Q4_K_M TPOT, 4 slots | Q4_K_M TPOT, 8 slots (fair) | plow MXFP4 TPOT |
|---|---|---|---|---|
| chat_short | 4 | 107 | 111 | **102** |
| chat_short | 8 | 105 | 178 | **151** |
| chat_long | 4 | 119 | 237 | **152** |
| chat_long | 8 | 118 | 217 | 308 |
| code | 4 | 120 | 185 | **155** |
| code | 8 | 119 | 183 | 248 |
| summarize | 4 | 146 | 172 | 323 |
| summarize | 8 | 190 | 257 | 507 |

Raw: `results-q4_k_m-np8*.md`. With the fair slot count plow wins four more decode cells than the
earlier table showed; it still loses the long-prompt c=8 cells and both summarize cells at c>=4.
TTFT under 8 slots gets worse for llama.cpp everywhere (e.g. chat_long c=8 29.0 s -> 34.4 s), so
plow's TTFT wins widen.
