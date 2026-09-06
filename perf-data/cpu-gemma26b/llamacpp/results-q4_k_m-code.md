# bench-api results

- server: `http://localhost:8097`  model: `/home/lava/llamacpp/gemma-4-26b-a4b-it-q4_k_m.gguf`
- time: 2026-09-06T07:26:17+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: /home/lava/llamacpp/gemma-4-26b-a4b-it-q4_k_m.gguf

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 365 | 64.0 | 3858/3905/4611 | 52/52/52 | 7.16/7.87 | 9.0 | 0.14 |
| code | 2 | 8 | 0 | 374 | 64.0 | 7914/8291/8804 | 90/85/106 | 13.62/14.16 | 9.4 | 0.15 |
| code | 4 | 8 | 0 | 401 | 64.0 | 16867/17597/17598 | 120/121/121 | 25.23/25.24 | 10.5 | 0.16 |
| code | 8 | 8 | 0 | 359 | 64.0 | 26328/37841/37842 | 119/120/120 | 45.32/45.32 | 11.3 | 0.18 |

## Samples

**code @ 1**

- #0 code in=463 out=64 fin=length: `<|channel>thought<channel|>### Explanation The issue is caused by the use of 'va`
- #1 code in=304 out=64 fin=length: `<|channel>thought<channel|>'''sql /* APPROACH: 1. Data Cleaning: Use COALESCE an`

**code @ 2**

- #8 code in=497 out=64 fin=length: `<|channel>thought<channel|>'''python import string from collections import Count`
- #9 code in=283 out=64 fin=length: `<|channel>thought<channel|>'''python import string  def top_words(text):     """`

**code @ 4**

- #16 code in=375 out=64 fin=length: `<|channel>thought<channel|>'''javascript /**  * Calculates the sum of the 'price`
- #17 code in=435 out=64 fin=length: `<|channel>thought<channel|>'''python import logging  # Configure logger for oper`

**code @ 8**

- #24 code in=354 out=64 fin=length: `<|channel>thought<channel|>'''python import string from collections import Count`
- #25 code in=408 out=64 fin=length: `<|channel>thought<channel|>'''rust fn lower_bound(xs: &[i32], target: i32) -> Op`
