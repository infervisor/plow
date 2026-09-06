# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T22:25:36+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 32 | 64.0 | 328/325/340 | 23/23/24 | 1.81/1.83 | 35.5 | 0.55 |
| chat_short | 2 | 8 | 0 | 42 | 63.4 | 461/388/709 | 45/46/48 | 3.33/3.42 | 38.1 | 0.60 |
| chat_short | 4 | 8 | 0 | 38 | 64.0 | 773/727/1053 | 76/76/80 | 5.68/5.86 | 45.5 | 0.71 |
| chat_short | 8 | 8 | 0 | 34 | 63.9 | 1644/1782/2673 | 127/130/138 | 9.72/9.94 | 50.9 | 0.80 |
| chat_long | 1 | 8 | 0 | 392 | 62.8 | 827/858/1046 | 25/25/25 | 2.35/2.62 | 26.6 | 0.42 |
| chat_long | 2 | 8 | 0 | 439 | 61.9 | 1170/1294/1752 | 49/53/61 | 4.32/4.47 | 29.4 | 0.47 |
| chat_long | 4 | 8 | 0 | 344 | 64.0 | 1536/1580/1959 | 88/92/97 | 7.39/7.75 | 35.7 | 0.56 |
| chat_long | 8 | 8 | 0 | 423 | 61.0 | 3562/3692/6153 | 169/182/201 | 14.09/14.31 | 33.6 | 0.55 |
| code | 1 | 8 | 0 | 344 | 64.0 | 732/723/809 | 25/25/25 | 2.30/2.37 | 27.8 | 0.43 |
| code | 2 | 8 | 0 | 357 | 64.0 | 1161/1369/1578 | 47/51/53 | 4.19/4.28 | 31.0 | 0.48 |
| code | 4 | 8 | 0 | 383 | 64.0 | 1750/1668/2498 | 92/94/106 | 7.54/8.41 | 33.6 | 0.53 |
| code | 8 | 8 | 0 | 337 | 64.0 | 3421/3773/5466 | 158/163/187 | 13.42/13.66 | 37.2 | 0.58 |
| summarize | 1 | 8 | 0 | 1111 | 64.0 | 2122/2469/2678 | 25/26/26 | 4.08/4.30 | 17.2 | 0.27 |
| summarize | 2 | 8 | 0 | 1124 | 64.0 | 2466/2427/2659 | 70/78/80 | 6.79/7.68 | 18.6 | 0.29 |
| summarize | 4 | 8 | 0 | 1188 | 64.0 | 4307/3073/7315 | 147/163/177 | 14.19/17.59 | 18.7 | 0.29 |
| summarize | 8 | 8 | 0 | 1136 | 64.0 | 9352/10037/15321 | 249/279/332 | 25.12/25.56 | 19.9 | 0.31 |

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
- #25 chat_short in=26 out=63 fin=stop: `At the end of the season the colony’s brood production slows, so the queen stops`

**chat_long @ 1**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it**   The passage says the emergency fund should be kept “somew`
- #1 chat_long in=383 out=64 fin=length: `**Answer**   An escapement is a mechanical device that transfers the energy of a`

**chat_long @ 2**

- #8 chat_long in=501 out=64 fin=length: `**Answer**  Spring tides are the highest and lowest tides that occur when the Su`
- #9 chat_long in=473 out=64 fin=length: `A halo is a bright ring that appears around the Sun or Moon. It is caused by the`

**chat_long @ 4**

- #16 chat_long in=236 out=64 fin=length: `**Short answer**  The coffee is going stale because the beans are sitting in the`
- #17 chat_long in=357 out=64 fin=length: `**Winter bees vs. summer bees – the big difference is what they’re built for**  `

**chat_long @ 8**

- #24 chat_long in=262 out=64 fin=length: `**Title: “From Oil Lamps to Automation: The Life of a Lighthouse Keeper”**   *(≈`
- #25 chat_long in=581 out=40 fin=stop: `The passage does not mention Fresnel’s lens or how it changed lighthouse illumin`

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
