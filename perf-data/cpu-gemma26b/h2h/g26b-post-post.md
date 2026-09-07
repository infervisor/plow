# bench-api results

- server: `http://localhost:8096`  model: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- time: 2026-09-06T22:34:27+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 4d7ae4984b7db7de8f8457170b3f1a419ee76d52

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 318/321/327 | 36/36/36 | 2.57/2.60 | 25.0 | 0.39 |
| chat_short | 2 | 8 | 0 | 45 | 62.5 | 481/404/754 | 58/60/62 | 4.18/4.28 | 30.4 | 0.49 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 775/739/1088 | 87/87/91 | 6.31/6.57 | 40.6 | 0.63 |
| chat_short | 8 | 8 | 0 | 36 | 62.0 | 1665/1842/2688 | 127/130/140 | 9.71/9.93 | 49.5 | 0.80 |
| chat_long | 1 | 8 | 0 | 398 | 54.6 | 989/1040/1315 | 38/38/38 | 3.07/3.38 | 18.1 | 0.33 |
| chat_long | 2 | 8 | 0 | 442 | 59.2 | 1288/1339/1664 | 68/75/80 | 5.69/5.98 | 22.1 | 0.37 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 1793/1898/2268 | 107/110/117 | 8.86/9.23 | 29.8 | 0.47 |
| chat_long | 8 | 8 | 0 | 425 | 59.0 | 4343/4470/7672 | 190/210/232 | 16.29/16.53 | 28.4 | 0.48 |
| code | 1 | 8 | 0 | 365 | 64.0 | 869/864/988 | 38/38/38 | 3.26/3.42 | 19.6 | 0.31 |
| code | 2 | 8 | 0 | 374 | 64.0 | 1392/1560/1921 | 60/63/68 | 5.29/5.34 | 24.8 | 0.39 |
| code | 4 | 8 | 0 | 401 | 64.0 | 2059/1934/2974 | 108/110/126 | 8.86/9.82 | 28.6 | 0.45 |
| code | 8 | 8 | 0 | 359 | 64.0 | 4037/4409/6520 | 164/173/196 | 14.46/14.68 | 34.6 | 0.54 |
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 2694/3192/3431 | 40/40/40 | 5.71/5.98 | 12.3 | 0.19 |
| summarize | 2 | 8 | 0 | 1119 | 64.0 | 3215/3215/3455 | 95/107/108 | 9.04/10.28 | 13.8 | 0.22 |
| summarize | 4 | 8 | 0 | 1178 | 64.0 | 5450/3756/9425 | 184/205/225 | 18.03/22.31 | 14.9 | 0.23 |
| summarize | 8 | 8 | 0 | 1133 | 64.0 | 11383/11770/19027 | 293/329/394 | 29.94/30.40 | 16.7 | 0.26 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick drones out at the end of summer to ensure the colony's survival during`
- #1 chat_short in=21 out=64 fin=length: `When choosing between washed and natural processed coffee, you are essentially c`

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `To keep your talk engaging and practical, it is best to present this rule as a "`
- #9 chat_short in=50 out=64 fin=length: `The printing press changed books from unique, hand-drawn objects into standardiz`

**chat_short @ 4**

- #16 chat_short in=34 out=64 fin=length: `Since you are a hobbyist, the easiest way to think about propolis is to view it `
- #17 chat_short in=46 out=64 fin=length: `Richard Trevithick’s 1804 locomotive broke the rails because of a mismatch betwe`

**chat_short @ 8**

- #24 chat_short in=34 out=64 fin=length: `The reason for this conflict boils down to a simple physical problem: **before t`
- #25 chat_short in=26 out=48 fin=stop: `Bees expel drones at the end of summer to conserve the hive's limited food suppl`

**chat_long @ 1**

- #0 chat_long in=512 out=55 fin=stop: `The passage recommends keeping an emergency fund in a **"separate savings accoun`
- #1 chat_long in=381 out=64 fin=length: `An escapement is the core mechanism of a mechanical clock that regulates the rel`

**chat_long @ 2**

- #8 chat_long in=502 out=64 fin=length: `The difference between spring and neap tides lies in the range of the tidal heig`
- #9 chat_long in=472 out=57 fin=stop: `Based on the provided text, there is no mention of a halo around the sun or moon`

**chat_long @ 4**

- #16 chat_long in=250 out=64 fin=length: `### The Diagnosis  The most likely cause of the "flat" taste is **oxidation and `
- #17 chat_long in=358 out=64 fin=length: `Think of it this way: a honeybee colony is like a massive, living machine, and t`

**chat_long @ 8**

- #24 chat_long in=283 out=64 fin=length: `To avoid the "romantic" trap, you should frame the lighthouse not as a lonely se`
- #25 chat_long in=587 out=24 fin=stop: `The passage does not mention Fresnel's lens, nor does it state how far a lightho`

**code @ 1**

- #0 code in=463 out=64 fin=length: `### Explanation The issue is caused by the use of 'var', which has function-leve`
- #1 code in=304 out=64 fin=length: `Since the prompt provides a SQL query but asks for a solution that handles "inpu`

**code @ 2**

- #8 code in=497 out=64 fin=length: `### Identified Problems  1.  **Logic Error (Incrementing):** The 'if/else' logic`
- #9 code in=283 out=64 fin=length: `### Identified Problems  1.  **Logic Error (Incrementing):** The 'if/else' logic`

**code @ 4**

- #16 code in=375 out=64 fin=length: `'''javascript function totalPrice(items) {     // Handle cases where items is nu`
- #17 code in=435 out=64 fin=length: `'''python import logging  # Configure logger for operations visibility logger = `

**code @ 8**

- #24 code in=354 out=64 fin=length: `'''python import string from collections import Counter  def top_words(text):   `
- #25 code in=408 out=64 fin=length: `'''rust fn lower_bound(xs: &[i32], target: i32) -> Option<usize> {     // Handle`

**summarize @ 1**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  * **Early Timekeeping:** Ancient methods like`
- #1 summarize in=1411 out=64 fin=length: `This text provides a detailed overview of two distinct subjects: the geological `

**summarize @ 2**

- #8 summarize in=1223 out=64 fin=length: `The first section outlines household budgeting, emphasizing the distinction betw`
- #9 summarize in=1187 out=64 fin=length: `Here is a summary of the article:  * **The Role of the Keeper:** Lighthouse keep`

**summarize @ 4**

- #16 summarize in=1205 out=64 fin=length: `### Summary  The provided text consists of two distinct articles: one detailing `
- #17 summarize in=1083 out=64 fin=length: `### Summary The invention of the printing press revolutionized Europe by making `

**summarize @ 8**

- #24 summarize in=1111 out=64 fin=length: `The first text details the life of lighthouse keepers, whose precise maintenance`
- #25 summarize in=1366 out=64 fin=length: `The provided text consists of two distinct articles. Here is a summary of each: `
