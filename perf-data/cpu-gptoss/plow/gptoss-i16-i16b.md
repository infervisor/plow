# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T19:04:48+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 32 | 64.0 | 350/346/376 | 22/22/22 | 1.72/1.76 | 36.9 | 0.58 |
| chat_short | 2 | 8 | 0 | 42 | 63.5 | 485/401/757 | 41/42/44 | 3.05/3.14 | 41.2 | 0.65 |
| chat_short | 4 | 8 | 0 | 38 | 64.0 | 788/754/1087 | 64/64/68 | 4.87/5.12 | 52.9 | 0.83 |
| chat_short | 8 | 8 | 0 | 34 | 64.0 | 1710/1879/2716 | 110/113/122 | 8.73/8.92 | 56.8 | 0.89 |
| code | 1 | 8 | 0 | 344 | 64.0 | 811/833/888 | 23/23/24 | 2.29/2.36 | 28.1 | 0.44 |
| code | 2 | 8 | 0 | 357 | 64.0 | 1286/1477/1757 | 42/47/49 | 3.99/4.07 | 32.5 | 0.51 |
| code | 4 | 8 | 0 | 383 | 64.0 | 1872/1777/2707 | 81/83/96 | 7.03/7.91 | 36.4 | 0.57 |
| code | 8 | 8 | 0 | 337 | 64.0 | 3667/4015/5868 | 149/157/180 | 13.12/13.33 | 38.1 | 0.60 |

## Samples

**chat_short @ 1**

- #0 chat_short in=25 out=64 fin=length: `At the end of the season the colony’s population drops to a few hundred workers.`
- #1 chat_short in=22 out=64 fin=length: `Below is a concise comparison of the two most common coffee processing methods—*`

**chat_short @ 2**

- #8 chat_short in=46 out=64 fin=length: `**The 50 % rule for fixed costs**  | What it says | How it works | |------------`
- #9 chat_short in=46 out=64 fin=length: `**One‑sentence answer:**   The printing press made books thinner, more uniform, `

**chat_short @ 4**

- #16 chat_short in=32 out=64 fin=length: `**Propolis** is a resin‑like substance that bees collect from trees and plants. `
- #17 chat_short in=41 out=64 fin=length: `Trevithick’s 1804 locomotive was the first to use a steam engine on iron rails, `

**chat_short @ 8**

- #24 chat_short in=33 out=64 fin=length: `**Short answer**  When a railway line was built, the railway company had to deci`
- #25 chat_short in=26 out=64 fin=length: `At the end of the season the colony has no need for the large, non‑reproductive `

**code @ 1**

- #0 code in=441 out=64 fin=length: `'''javascript /**  * Log 0, 1, 2 after 1, 2, 3 seconds respectively.  * Uses 'le`
- #1 code in=284 out=64 fin=length: `**Solution**  '''sql /*  Query:  total spent per customer  *  ------------------`

**code @ 2**

- #8 code in=471 out=64 fin=length: `'''python import string from collections import Counter from typing import List `
- #9 code in=257 out=64 fin=length: `'''python import string  def top_words(text):     """     Return the three most `

**code @ 4**

- #16 code in=355 out=64 fin=length: `**Fixed implementation**  '''javascript /**  * Returns the sum of the 'price' fi`
- #17 code in=424 out=64 fin=length: `'''python def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[in`

**code @ 8**

- #24 code in=330 out=64 fin=length: `'''python import string from collections import Counter  def top_words(text: str`
- #25 code in=379 out=64 fin=length: `'''rust fn lower_bound(xs: &[i32], target: i32) -> Option<usize> {     // Standa`
