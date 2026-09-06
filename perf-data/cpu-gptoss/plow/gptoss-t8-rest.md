# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T15:41:28+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 32 | 64.0 | 358/355/376 | 25/25/26 | 1.96/1.98 | 32.7 | 0.51 |
| chat_short | 2 | 8 | 0 | 42 | 63.4 | 513/424/791 | 47/48/49 | 3.48/3.51 | 36.4 | 0.57 |
| chat_short | 4 | 8 | 0 | 38 | 64.0 | 862/822/1179 | 84/84/88 | 6.22/6.47 | 41.4 | 0.65 |
| chat_short | 8 | 8 | 0 | 34 | 63.9 | 1840/1990/2976 | 142/146/155 | 10.89/11.13 | 45.4 | 0.71 |
| code | 1 | 8 | 0 | 344 | 64.0 | 893/882/992 | 25/26/26 | 2.49/2.60 | 25.6 | 0.40 |
| code | 2 | 8 | 0 | 357 | 64.0 | 1405/1619/1929 | 50/55/58 | 4.68/4.73 | 27.9 | 0.44 |
| code | 4 | 8 | 0 | 383 | 64.0 | 2062/1940/2983 | 103/105/119 | 8.59/9.55 | 29.8 | 0.47 |
| code | 8 | 8 | 0 | 337 | 64.0 | 4017/4364/6357 | 176/185/208 | 15.18/15.44 | 32.9 | 0.51 |

## Samples

**chat_short @ 1**

- #0 chat_short in=25 out=64 fin=length: `At the end of the season the colony’s population is already high, and the queen’`
- #1 chat_short in=22 out=64 fin=length: `Below is a concise comparison of the two most common coffee processing methods—*`

**chat_short @ 2**

- #8 chat_short in=46 out=64 fin=length: `**The 50 % rule for fixed costs**  | What it says | How it works | |------------`
- #9 chat_short in=46 out=64 fin=length: `**One‑sentence answer:**   The printing press made books thinner, more uniform, `

**chat_short @ 4**

- #16 chat_short in=32 out=64 fin=length: `**Propolis** is a resin‑like substance that bees collect from trees and plants. `
- #17 chat_short in=41 out=64 fin=length: `Trevithick’s 1804 locomotive was the first to use a steam engine on iron rails, `

**chat_short @ 8**

- #24 chat_short in=33 out=64 fin=length: `**Short answer**  Because the railway timetable had to be the same everywhere.  `
- #25 chat_short in=26 out=63 fin=stop: `At the end of the season the colony has no need for the large, non‑reproductive `

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
