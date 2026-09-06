# plow-bf16 vs vllm-bf16

- **plow-bf16**: `http://localhost:8096` model `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (2026-09-06T00:19:49+00:00, max_tokens=64, seed=1234, prompt tokens via tokenizer)
- **vllm-bf16**: `http://localhost:8094` model `gemma-4-12b-it` (2026-09-06T01:28:41+00:00, max_tokens=64, seed=1234, prompt tokens via tokenizer)

Latency stat: **p50**. Ratio = vllm-bf16 / plow-bf16 (latency: <1 means vllm-bf16 faster; throughput: >1 means vllm-bf16 faster).

| workload | conc | TTFT p50 ms plow-bf16 | vllm-bf16 | ratio | TPOT p50 ms plow-bf16 | vllm-bf16 | ratio | out tok/s plow-bf16 | vllm-bf16 | ratio | req/s plow-bf16 | vllm-bf16 | ratio | err plow-bf16/vllm-bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat_long | 1 | 2619 | 1930 | 0.74x | 227 | 441 | 1.94x | 3.8 | 2.2 | 0.57x | 0.06 | 0.04 | 0.57x | 0/0 |
| chat_long | 2 | 2969 | 1370 | 0.46x | 276 | 448 | 1.62x | 5.9 | 4.2 | 0.71x | 0.10 | 0.07 | 0.70x | 0/0 |
| chat_long | 4 | 6407 | 1589 | 0.25x | 374 | 457 | 1.22x | 8.4 | 7.6 | 0.90x | 0.14 | 0.13 | 0.96x | 0/0 |
| chat_long | 8 | 13785 | 2306 | 0.17x | 645 | 459 | 0.71x | 9.1 | 15.7 | 1.72x | 0.15 | 0.26 | 1.71x | 0/0 |
| chat_short | 1 | 773 | 649 | 0.84x | 217 | 520 | 2.39x | 4.4 | 1.9 | 0.44x | 0.07 | 0.03 | 0.44x | 0/0 |
| chat_short | 2 | 1639 | 1089 | 0.66x | 236 | 442 | 1.87x | 8.0 | 4.4 | 0.55x | 0.13 | 0.07 | 0.55x | 0/0 |
| chat_short | 4 | 1857 | 1553 | 0.84x | 265 | 442 | 1.67x | 13.7 | 8.7 | 0.63x | 0.21 | 0.14 | 0.63x | 0/0 |
| chat_short | 8 | 4560 | 2139 | 0.47x | 438 | 447 | 1.02x | 15.6 | 16.8 | 1.08x | 0.24 | 0.26 | 1.08x | 0/0 |
| code | 1 | 2477 | 1626 | 0.66x | 246 | 445 | 1.81x | 3.6 | 2.2 | 0.61x | 0.06 | 0.03 | 0.61x | 0/0 |
| code | 2 | 4480 | 1385 | 0.31x | 289 | 446 | 1.54x | 6.1 | 4.3 | 0.72x | 0.09 | 0.07 | 0.72x | 0/0 |
| code | 4 | 5390 | 1959 | 0.36x | 353 | 449 | 1.27x | 9.3 | 8.5 | 0.91x | 0.15 | 0.13 | 0.91x | 0/0 |
| code | 8 | 12749 | 2703 | 0.21x | 610 | 458 | 0.75x | 10.1 | 16.2 | 1.60x | 0.16 | 0.25 | 1.60x | 0/0 |
| summarize | 1 | 8601 | 2634 | 0.31x | 251 | 449 | 1.79x | 2.8 | 2.1 | 0.74x | 0.04 | 0.03 | 0.74x | 0/0 |
| summarize | 2 | 12418 | 4154 | 0.33x | 328 | 461 | 1.41x | 4.1 | 3.8 | 0.93x | 0.06 | 0.06 | 0.93x | 0/0 |
| summarize | 4 | 14871 | 7110 | 0.48x | 585 | 512 | 0.88x | 5.3 | 6.6 | 1.25x | 0.08 | 0.10 | 1.25x | 0/0 |
| summarize | 8 | 32158 | 13637 | 0.42x | 1005 | 550 | 0.55x | 5.5 | 6.7 | 1.21x | 0.09 | 0.11 | 1.21x | 0/0 |

Paired requests: 128; prompt mismatches: 0; identical first-80-char outputs: 110 (86%).

## Samples

**chat_long @ 1**

- #0 chat_long
  - plow-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `
  - vllm-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `

**chat_long @ 2**

- #0 chat_long
  - plow-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `
  - vllm-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `

**chat_long @ 4**

- #0 chat_long
  - plow-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `
  - vllm-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `

**chat_long @ 8**

- #0 chat_long
  - plow-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `
  - vllm-bf16: `The passage recommends keeping an emergency fund in "a separate savings account `

**chat_short @ 1**

- #0 chat_short
  - plow-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`
  - vllm-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`

**chat_short @ 2**

- #0 chat_short
  - plow-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`
  - vllm-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`

**chat_short @ 4**

- #0 chat_short
  - plow-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`
  - vllm-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`

**chat_short @ 8**

- #0 chat_short
  - plow-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`
  - vllm-bf16: `Bees kick out drones at the end of summer primarily because they are **non-produ`

**code @ 1**

- #0 code
  - plow-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`
  - vllm-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`

**code @ 2**

- #0 code
  - plow-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`
  - vllm-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`

**code @ 4**

- #0 code
  - plow-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`
  - vllm-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`

**code @ 8**

- #0 code
  - plow-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`
  - vllm-bf16: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`

**summarize @ 1**

- #0 summarize
  - plow-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
  - vllm-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`

**summarize @ 2**

- #0 summarize
  - plow-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
  - vllm-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`

**summarize @ 4**

- #0 summarize
  - plow-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
  - vllm-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`

**summarize @ 8**

- #0 summarize
  - plow-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
  - vllm-bf16: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`

