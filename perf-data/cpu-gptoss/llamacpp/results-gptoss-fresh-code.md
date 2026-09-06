# bench-api results

- server: `http://localhost:8098`  model: `gpt-oss-20b-mxfp4`
- time: 2026-09-06T08:30:38+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gpt-oss-20b-mxfp4

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 344 | 64.0 | 4212/4415/5358 | 50/50/53 | 7.45/8.70 | 8.7 | 0.14 |
| code | 2 | 8 | 0 | 357 | 64.0 | 8780/9510/10014 | 73/75/75 | 14.15/14.73 | 9.5 | 0.15 |
| code | 4 | 8 | 0 | 383 | 64.0 | 18762/20108/20215 | 123/125/125 | 27.97/28.11 | 9.7 | 0.15 |
| code | 8 | 8 | 0 | 337 | 64.0 | 35120/35075/35256 | 207/208/208 | 48.17/48.24 | 10.6 | 0.17 |

## Samples

**code @ 1**

- #0 code in=441 out=64 fin=length: `<|channel|>analysis<|message|>We need to explain why logs 3 three times. Because`
- #1 code in=284 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce a corrected SQL query that list`

**code @ 2**

- #8 code in=471 out=64 fin=length: `<|channel|>analysis<|message|>We need to identify problems: counts logic wrong: `
- #9 code in=257 out=64 fin=length: `<|channel|>analysis<|message|>We need to identify problems in the code:  - count`

**code @ 4**

- #16 code in=355 out=64 fin=length: `<|channel|>analysis<|message|>We need to rewrite totalPrice to be idiomatic, han`
- #17 code in=424 out=64 fin=length: `<|channel|>analysis<|message|>We need to write merge_intervals function. Input l`

**code @ 8**

- #24 code in=330 out=64 fin=length: `<|channel|>analysis<|message|>We need to identify problems: splitting by space o`
- #25 code in=379 out=64 fin=length: `<|channel|>analysis<|message|>We need to fix the lower_bound function. The bug: `
