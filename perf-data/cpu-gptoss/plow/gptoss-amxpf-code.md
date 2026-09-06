# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T08:15:44+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 344 | 64.0 | 1680/1663/1914 | 47/47/48 | 4.71/4.86 | 13.8 | 0.22 |
| code | 2 | 8 | 0 | 357 | 64.0 | 2648/2910/3710 | 81/87/96 | 8.01/8.19 | 16.5 | 0.26 |
| code | 4 | 8 | 0 | 383 | 64.0 | 3913/3734/5670 | 156/160/187 | 13.95/15.65 | 18.5 | 0.29 |
| code | 8 | 8 | 0 | 337 | 64.0 | 7332/7962/11954 | 268/286/330 | 24.29/24.69 | 20.6 | 0.32 |

## Samples

**code @ 1**

- #0 code in=441 out=64 fin=length: `'''javascript /**  * Logs 0, 1, 2 after one, two, and three seconds.  * Uses 'le`
- #1 code in=284 out=64 fin=length: `**Solution**  '''sql /* 1️⃣  Use explicit JOIN syntax and LEFT JOIN to keep cust`

**code @ 2**

- #8 code in=471 out=64 fin=length: `'''python import string from collections import Counter  def top_words(text: str`
- #9 code in=257 out=64 fin=length: `'''python import string  def top_words(text):     """     Return the three most `

**code @ 4**

- #16 code in=355 out=64 fin=length: `**Fixed implementation**  '''javascript /**  * Return the sum of the 'price' fie`
- #17 code in=424 out=64 fin=length: `'''python def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[in`

**code @ 8**

- #24 code in=330 out=64 fin=length: `'''python import string from collections import Counter  def top_words(text: str`
- #25 code in=379 out=64 fin=length: `'''rust /// Returns the index of the first element in 'xs' that is >= 'target'. `
