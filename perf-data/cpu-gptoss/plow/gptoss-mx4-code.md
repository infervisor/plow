# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T09:29:31+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 344 | 64.0 | 1697/1690/1937 | 33/33/33 | 3.74/4.04 | 17.0 | 0.27 |
| code | 2 | 8 | 0 | 357 | 64.0 | 2705/2967/3793 | 68/76/83 | 7.30/7.45 | 18.2 | 0.28 |
| code | 4 | 8 | 0 | 383 | 64.0 | 3971/3757/5746 | 148/152/181 | 13.44/15.21 | 19.2 | 0.30 |
| code | 8 | 8 | 0 | 337 | 64.0 | 7692/8284/12070 | 243/260/301 | 23.07/23.36 | 21.8 | 0.34 |

## Samples

**code @ 1**

- #0 code in=441 out=64 fin=length: `'''javascript /**  * Schedules console.log of the given numbers after the specif`
- #1 code in=284 out=64 fin=length: `**Solution**  '''sql /*     1. Use an explicit LEFT JOIN so that customers with `

**code @ 2**

- #8 code in=471 out=64 fin=length: `'''python import string from collections import Counter from typing import List `
- #9 code in=257 out=64 fin=length: `'''python import string  def top_words(text):     """     Return the three most `

**code @ 4**

- #16 code in=355 out=64 fin=length: `**Fixed implementation**  '''javascript /**  * Returns the sum of the 'price' fi`
- #17 code in=424 out=64 fin=length: `'''python def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[in`

**code @ 8**

- #24 code in=330 out=64 fin=length: `'''python import string from collections import Counter  def top_words(text: str`
- #25 code in=379 out=64 fin=length: `'''rust fn lower_bound(xs: &[i32], target: i32) -> Option<usize> {     // Standa`
