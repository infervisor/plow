# plow-fp8 vs vllm-bf16

- **plow-fp8**: `http://localhost:8096` model `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (2026-09-06T00:56:41+00:00, max_tokens=64, seed=1234, prompt tokens via tokenizer)
- **vllm-bf16**: `http://localhost:8094` model `gemma-4-12b-it` (2026-09-06T01:28:41+00:00, max_tokens=64, seed=1234, prompt tokens via tokenizer)

Latency stat: **p50**. Ratio = vllm-bf16 / plow-fp8 (latency: <1 means vllm-bf16 faster; throughput: >1 means vllm-bf16 faster).

| workload | conc | TTFT p50 ms plow-fp8 | vllm-bf16 | ratio | TPOT p50 ms plow-fp8 | vllm-bf16 | ratio | out tok/s plow-fp8 | vllm-bf16 | ratio | req/s plow-fp8 | vllm-bf16 | ratio | err plow-fp8/vllm-bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat_long | 1 | 2530 | 1930 | 0.76x | 162 | 441 | 2.73x | 5.0 | 2.2 | 0.43x | 0.08 | 0.04 | 0.42x | 0/0 |
| chat_long | 2 | 2818 | 1370 | 0.49x | 238 | 448 | 1.88x | 6.6 | 4.2 | 0.63x | 0.11 | 0.07 | 0.60x | 0/0 |
| chat_long | 4 | 6113 | 1589 | 0.26x | 387 | 457 | 1.18x | 8.1 | 7.6 | 0.94x | 0.14 | 0.13 | 0.97x | 0/0 |
| chat_long | 8 | 11945 | 2306 | 0.19x | 667 | 459 | 0.69x | 8.9 | 15.7 | 1.76x | 0.15 | 0.26 | 1.70x | 0/0 |
| chat_short | 1 | 564 | 649 | 1.15x | 160 | 520 | 3.26x | 6.0 | 1.9 | 0.32x | 0.09 | 0.03 | 0.32x | 0/0 |
| chat_short | 2 | 1231 | 1089 | 0.88x | 201 | 442 | 2.20x | 9.5 | 4.4 | 0.47x | 0.15 | 0.07 | 0.47x | 0/0 |
| chat_short | 4 | 1442 | 1553 | 1.08x | 286 | 442 | 1.54x | 13.2 | 8.7 | 0.66x | 0.21 | 0.14 | 0.66x | 0/0 |
| chat_short | 8 | 3619 | 2139 | 0.59x | 469 | 447 | 0.95x | 15.0 | 16.8 | 1.12x | 0.23 | 0.26 | 1.13x | 0/0 |
| code | 1 | 2160 | 1626 | 0.75x | 163 | 445 | 2.73x | 5.1 | 2.2 | 0.42x | 0.08 | 0.03 | 0.42x | 0/0 |
| code | 2 | 3807 | 1385 | 0.36x | 226 | 446 | 1.97x | 7.3 | 4.3 | 0.60x | 0.11 | 0.07 | 0.60x | 0/0 |
| code | 4 | 4726 | 1959 | 0.41x | 337 | 449 | 1.33x | 9.9 | 8.5 | 0.85x | 0.15 | 0.13 | 0.85x | 0/0 |
| code | 8 | 11335 | 2703 | 0.24x | 571 | 458 | 0.80x | 11.0 | 16.2 | 1.47x | 0.17 | 0.25 | 1.47x | 0/0 |
| summarize | 1 | 8030 | 2634 | 0.33x | 164 | 449 | 2.75x | 3.7 | 2.1 | 0.55x | 0.06 | 0.03 | 0.55x | 0/0 |
| summarize | 2 | 11560 | 4154 | 0.36x | 267 | 461 | 1.73x | 4.8 | 3.8 | 0.79x | 0.08 | 0.06 | 0.79x | 0/0 |
| summarize | 4 | 14046 | 7110 | 0.51x | 522 | 512 | 0.98x | 5.6 | 6.6 | 1.17x | 0.09 | 0.10 | 1.17x | 0/0 |
| summarize | 8 | 29369 | 13637 | 0.46x | 1000 | 550 | 0.55x | 5.9 | 6.7 | 1.15x | 0.09 | 0.11 | 1.15x | 0/0 |

Paired requests: 128; prompt mismatches: 0; identical first-80-char outputs: 75 (59%).

## Samples

**chat_long @ 1**

- #0 chat_long
  - plow-fp8: `The passage recommends keeping an emergency fund in "a separate savings account `
  - vllm-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `

**chat_long @ 2**

- #0 chat_long
  - plow-fp8: `The passage recommends keeping an emergency fund in "a separate savings account `
  - vllm-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `

**chat_long @ 4**

- #0 chat_long
  - plow-fp8: `The passage recommends keeping an emergency fund in "a separate savings account `
  - vllm-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `

**chat_long @ 8**

- #0 chat_long
  - plow-fp8: `The passage recommends keeping an emergency fund in "a separate savings account `
  - vllm-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `

**chat_short @ 1**

- #0 chat_short
  - plow-fp8: `Bees kick out drones at the end of summer primarily because they are **non-produ`
  - vllm-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`

**chat_short @ 2**

- #0 chat_short
  - plow-fp8: `Bees kick out drones at the end of summer primarily because they are **non-produ`
  - vllm-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`

**chat_short @ 4**

- #0 chat_short
  - plow-fp8: `Bees kick out drones at the end of summer primarily because they are **non-produ`
  - vllm-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`

**chat_short @ 8**

- #0 chat_short
  - plow-fp8: `Bees kick out drones at the end of summer primarily because they are **non-produ`
  - vllm-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`

**code @ 1**

- #0 code
  - plow-fp8: `### Explanation of the Bug The issue is caused by **closures** and the way the '`
  - vllm-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`

**code @ 2**

- #0 code
  - plow-fp8: `### Explanation of the Bug The issue is caused by **closures** and the way the '`
  - vllm-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`

**code @ 4**

- #0 code
  - plow-fp8: `### Explanation of the Bug The issue is caused by **closures** and the way the '`
  - vllm-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`

**code @ 8**

- #0 code
  - plow-fp8: `### Explanation of the Bug The issue is caused by **closures** and the way the '`
  - vllm-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`

**summarize @ 1**

- #0 summarize
  - plow-fp8: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
  - vllm-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`

**summarize @ 2**

- #0 summarize
  - plow-fp8: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
  - vllm-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`

**summarize @ 4**

- #0 summarize
  - plow-fp8: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
  - vllm-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`

**summarize @ 8**

- #0 summarize
  - plow-fp8: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
  - vllm-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`

