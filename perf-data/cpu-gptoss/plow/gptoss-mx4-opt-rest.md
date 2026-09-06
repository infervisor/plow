# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T11:26:25+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 344 | 64.0 | 1544/1506/1787 | 28/29/29 | 3.33/3.60 | 19.2 | 0.30 |
| code | 2 | 8 | 0 | 357 | 64.0 | 2457/2652/3423 | 58/65/71 | 6.33/6.49 | 21.0 | 0.33 |
| code | 4 | 8 | 0 | 383 | 64.0 | 3613/3440/5304 | 125/129/154 | 11.70/13.26 | 22.1 | 0.35 |
| code | 8 | 8 | 0 | 337 | 64.0 | 6902/7484/10874 | 209/218/265 | 20.11/20.36 | 25.0 | 0.39 |
| summarize | 1 | 8 | 0 | 1111 | 64.0 | 5061/5669/6553 | 29/28/30 | 7.58/8.34 | 9.3 | 0.14 |
| summarize | 2 | 8 | 0 | 1124 | 64.0 | 5828/5600/6342 | 119/133/144 | 13.32/15.40 | 9.6 | 0.15 |
| summarize | 4 | 8 | 0 | 1188 | 64.0 | 10095/6728/17626 | 270/313/348 | 29.00/37.32 | 9.4 | 0.15 |
| summarize | 8 | 8 | 0 | 1136 | 64.0 | 23170/25419/37188 | 431/491/650 | 50.38/50.93 | 10.0 | 0.16 |

## Samples

**code @ 1**

- #0 code in=441 out=64 fin=length: `'''javascript /**  * Log 0, 1, 2 after 1, 2, 3 seconds respectively.  * Uses 'le`
- #1 code in=284 out=64 fin=length: `**Solution**  '''sql /*  Query:  total spend per customer  *  *  1.  Use an expl`

**code @ 2**

- #8 code in=471 out=64 fin=length: `'''python import string from collections import Counter from typing import List `
- #9 code in=257 out=64 fin=length: `'''python import string  def top_words(text):     """     Return the three most `

**code @ 4**

- #16 code in=355 out=64 fin=length: `**Fixed implementation**  '''javascript /**  * Returns the sum of the 'price' fi`
- #17 code in=424 out=64 fin=length: `'''python def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[in`

**code @ 8**

- #24 code in=330 out=64 fin=length: `'''python import string from collections import Counter  def top_words(text: str`
- #25 code in=379 out=64 fin=length: `'''rust fn lower_bound(xs: &[i32], target: i32) -> Option<usize> {     // Standa`

**summarize @ 1**

- #0 summarize in=813 out=64 fin=length: `- **Early time‑keeping**:     * Sundials and water clocks were the first devices`
- #1 summarize in=1434 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and melt‑water action`

**summarize @ 2**

- #8 summarize in=1215 out=64 fin=length: `**Household budgeting**: A budget plans money before it arrives, using envelopes`
- #9 summarize in=1190 out=64 fin=length: `- **Keeper duties and conditions** – A lighthouse keeper’s day revolved around m`

**summarize @ 4**

- #16 summarize in=1203 out=64 fin=length: `**Summary**  The article contrasts the lives of lighthouse keepers with the inne`
- #17 summarize in=1094 out=64 fin=length: `**Summary**   The printing press revolutionized Europe by making books cheaper, `

**summarize @ 8**

- #24 summarize in=1117 out=64 fin=length: `Lighthouses were identified by unique light patterns—fixed, flashing at 5, 10, o`
- #25 summarize in=1363 out=64 fin=length: `- **Ancient & early time‑keeping** – Sundials and water clocks were the first to`
