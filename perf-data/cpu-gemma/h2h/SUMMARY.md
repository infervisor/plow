# Gemma-4-12B-it on CPU (16 threads): plow vs vLLM 0.28 vs llama.cpp 6a1a922

Same seeded bench-api workloads (64 output tokens, 8 requests per cell, rag_4k 4), each server alone on the box. p50 TTFT ms / p50 TPOT ms / aggregate out tok/s. plow numbers are the 2026-09-06 00:20 build (before the fp8 dequant, chunked-prefill and AMX staging changes).

**Caveat (c >= 2 cells):** the same 8 prompts were reused across concurrency levels, and llama-server (slot prompt cache) and vLLM (prefix caching, on by default) serve repeated prompts from cache — llama.cpp bf16 chat_long TTFT drops from 13 s at c=1 to 0.66 s at c=2, impossible at its ~35 tok/s prefill. plow has no prefix cache. Only the c=1 rows (first use of each prompt) compare prefill fairly; c >= 2 TTFT/throughput flatter the baselines. Re-run with `bench.py --fresh-prompts` pending.

## chat_short (34 in)

| conc | plow bf16 | plow fp8 | vLLM bf16 | llama.cpp bf16 | llama.cpp Q8_0 | llama.cpp Q4_K_M |
|---|---|---|---|---|---|---|
| 1 | 773 / 217 / 4.4 | 564 / 160 / 6.0 | 649 / 520 / 1.9 | 1496 / 267 / 3.5 | 939 / 133 / 6.9 | 662 / 121 / 7.8 |
| 2 | 1639 / 236 / 8.0 | 1231 / 201 / 9.5 | 1089 / 442 / 4.4 | 664 / 259 / 7.5 | 406 / 154 / 12.7 | 609 / 244 / 8.8 |
| 4 | 1857 / 265 / 13.7 | 1442 / 286 / 13.2 | 1553 / 442 / 8.7 | 1419 / 274 / 13.9 | 769 / 313 / 12.8 | 715 / 289 / 17.0 |
| 8 | 4560 / 438 / 15.6 | 3619 / 469 / 15.0 | 2139 / 447 / 16.8 | 2004 / 359 / 21.1 | 686 / 191 / 40.2 | 1621 / 195 / 36.8 |

## chat_long (398 in)

| conc | plow bf16 | plow fp8 | vLLM bf16 | llama.cpp bf16 | llama.cpp Q8_0 | llama.cpp Q4_K_M |
|---|---|---|---|---|---|---|
| 1 | 2619 / 227 / 3.8 | 2530 / 162 / 5.0 | 1930 / 441 / 2.2 | 13090 / 271 / 2.2 | 6038 / 135 / 4.4 | 4000 / 123 / 5.3 |
| 2 | 2969 / 276 / 5.9 | 2818 / 238 / 6.6 | 1370 / 448 / 4.2 | 1680 / 449 / 4.3 | 921 / 172 / 9.2 | 1050 / 213 / 8.0 |
| 4 | 6407 / 374 / 8.4 | 6113 / 387 / 8.1 | 1589 / 457 / 7.6 | 2605 / 544 / 5.9 | 1577 / 308 / 12.4 | 1956 / 401 / 9.1 |
| 8 | 13785 / 645 / 9.1 | 11945 / 667 / 8.9 | 2306 / 459 / 15.7 | 3038 / 437 / 15.5 | 3983 / 302 / 21.4 | 2167 / 284 / 21.1 |

## code (365 in)

| conc | plow bf16 | plow fp8 | vLLM bf16 | llama.cpp bf16 | llama.cpp Q8_0 | llama.cpp Q4_K_M |
|---|---|---|---|---|---|---|
| 1 | 2477 / 246 / 3.6 | 2160 / 163 / 5.1 | 1626 / 445 / 2.2 | 10360 / 268 / 2.3 | 4886 / 135 / 4.7 | 3199 / 123 / 5.8 |
| 2 | 4480 / 289 / 6.1 | 3807 / 226 / 7.3 | 1385 / 446 / 4.3 | 1367 / 540 / 3.7 | 1148 / 274 / 7.7 | 1349 / 249 / 8.2 |
| 4 | 5390 / 353 / 9.3 | 4726 / 337 / 9.9 | 1959 / 449 / 8.5 | 2299 / 555 / 6.9 | 1757 / 439 / 9.8 | 2203 / 414 / 12.4 |
| 8 | 12749 / 610 / 10.1 | 11335 / 571 / 11.0 | 2703 / 458 / 16.2 | 2907 / 394 / 18.4 | 2150 / 217 / 32.4 | 2003 / 218 / 32.6 |

## summarize (1105 in)

| conc | plow bf16 | plow fp8 | vLLM bf16 | llama.cpp bf16 | llama.cpp Q8_0 | llama.cpp Q4_K_M |
|---|---|---|---|---|---|---|
| 1 | 8601 / 251 / 2.8 | 8030 / 164 / 3.7 | 2634 / 449 / 2.1 | 33275 / 283 / 1.3 | 16236 / 141 / 2.7 | 11349 / 129 / 3.5 |
| 2 | 12418 / 328 / 4.1 | 11560 / 267 / 4.8 | 4154 / 461 / 3.8 | 22793 / 291 / 3.4 | 18558 / 178 / 4.7 | 12632 / 181 / 5.6 |
| 4 | 14871 / 585 / 5.3 | 14046 / 522 / 5.6 | 7110 / 512 / 6.6 | 26536 / 1064 / 3.1 | 19837 / 224 / 7.4 | 9504 / 251 / 9.9 |
| 8 | 32158 / 1005 / 5.5 | 29369 / 1000 / 5.9 | 13637 / 550 / 6.7 | 7392 / 438 / 14.6 | 20842 / 270 / 13.5 | 5396 / 265 / 23.2 |

## rag_4k (~3000 in)

| conc | plow bf16 | plow fp8 | vLLM bf16 | llama.cpp bf16 | llama.cpp Q8_0 | llama.cpp Q4_K_M |
|---|---|---|---|---|---|---|
| 1 | 25946 / 251 / 1.5 | 24725 / 164 / 1.8 | 8771 / 449 / 1.7 | err 5 | 43935 / 141 / 1.0 | 30177 / 127 / 1.3 |
| 2 | 26060 / 748 / 1.7 | 24692 / 615 / 1.9 | 10658 / 597 / 2.7 | err 8 | 1859 / 286 / 6.4 | 1799 / 257 / 4.2 |
| 4 | - | - | - | err 8 | 3105 / 400 / 4.9 | 2179 / 364 / 8.8 |
| 8 | - | - | - | err 8 | 3326 / 984 / 6.2 | 3495 / 1027 / 6.1 |
