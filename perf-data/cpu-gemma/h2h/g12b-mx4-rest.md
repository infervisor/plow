# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T13:11:22+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 365 | 64.0 | 2073/2029/2518 | 90/91/91 | 7.78/8.23 | 8.2 | 0.13 |
| code | 2 | 8 | 0 | 374 | 64.0 | 3391/3545/4641 | 142/151/159 | 12.51/12.99 | 10.4 | 0.16 |
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 7096/8157/8970 | 102/102/103 | 14.68/15.46 | 4.7 | 0.07 |
| summarize | 2 | 8 | 0 | 1119 | 64.0 | 8396/8106/8884 | 237/262/266 | 23.00/25.65 | 5.5 | 0.09 |

## Samples

**code @ 1**

- #0 code in=463 out=64 fin=length: `### Explanation The issue is caused by **closure scope**. In JavaScript, the 'va`
- #1 code in=304 out=64 fin=length: `'''sql /* Approach: 1. Use a LEFT JOIN to ensure customers with no orders are in`

**code @ 2**

- #8 code in=497 out=64 fin=length: `'''python import string from collections import Counter  def top_words(text):   `
- #9 code in=283 out=64 fin=length: `### Identified Problems 1.  **Case Sensitivity:** The code does not convert text`

**summarize @ 1**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
- #1 summarize in=1411 out=64 fin=length: `This text explores the geological processes of glacial erosion and deposition, a`

**summarize @ 2**

- #8 summarize in=1223 out=64 fin=length: `The article provides a primer on household budgeting and the production of coffe`
- #9 summarize in=1187 out=64 fin=length: `Here is a summary of the article:  *   **The Role and Responsibility of Keepers:`
