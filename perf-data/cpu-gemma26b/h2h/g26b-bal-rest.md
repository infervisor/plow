# bench-api results

- server: `http://localhost:8096`  model: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- time: 2026-09-06T12:51:40+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 4d7ae4984b7db7de8f8457170b3f1a419ee76d52

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 365 | 64.0 | 1999/1982/2383 | 43/44/44 | 4.77/5.15 | 13.5 | 0.21 |
| code | 2 | 8 | 0 | 374 | 64.0 | 3166/3437/4320 | 75/84/92 | 8.14/8.37 | 16.2 | 0.25 |
| code | 4 | 8 | 0 | 401 | 64.0 | 4681/4470/6765 | 155/161/192 | 14.73/16.72 | 17.6 | 0.28 |
| code | 8 | 8 | 0 | 359 | 64.0 | 8665/9479/13823 | 248/270/323 | 24.35/24.59 | 20.7 | 0.32 |
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 6021/6900/7656 | 45/45/46 | 9.73/10.53 | 7.2 | 0.11 |
| summarize | 2 | 8 | 0 | 1119 | 64.0 | 7003/6814/7487 | 147/168/172 | 16.04/18.39 | 7.9 | 0.12 |
| summarize | 4 | 8 | 0 | 1178 | 64.0 | 11838/7873/20427 | 323/379/402 | 34.36/44.27 | 7.9 | 0.12 |
| summarize | 8 | 8 | 0 | 1133 | 64.0 | 27414/30059/46162 | 507/585/766 | 59.50/60.05 | 8.5 | 0.13 |

## Samples

**code @ 1**

- #0 code in=463 out=64 fin=length: `### Explanation The issue is caused by the use of 'var', which has function-leve`
- #1 code in=304 out=64 fin=length: `Since the input comes from an external system and is prone to malformed data (nu`

**code @ 2**

- #8 code in=497 out=64 fin=length: `### Identified Problems  1.  **Logic Error (Incrementing):** The 'if/else' logic`
- #9 code in=283 out=64 fin=length: `### Identified Problems  1.  **Logic Error (Incrementing):** The 'if/else' logic`

**code @ 4**

- #16 code in=375 out=64 fin=length: `'''javascript /**  * Calculates the sum of the 'price' field across an array of `
- #17 code in=435 out=64 fin=length: `'''python import logging  # Configure logger for operations visibility logger = `

**code @ 8**

- #24 code in=354 out=64 fin=length: `'''python import string from collections import Counter  def top_words(text):   `
- #25 code in=408 out=64 fin=length: `'''rust fn lower_bound(xs: &[i32], target: i32) -> Option<usize> {     // Handle`

**summarize @ 1**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  * **Early Timekeeping:** Ancient methods like`
- #1 summarize in=1411 out=64 fin=length: `This text provides a detailed overview of two distinct subjects: the geological `

**summarize @ 2**

- #8 summarize in=1223 out=64 fin=length: `The article provides two distinct guides. The first covers household budgeting, `
- #9 summarize in=1187 out=64 fin=length: `Here is a summary of the article:  * **The Role of the Keeper:** Lighthouse keep`

**summarize @ 4**

- #16 summarize in=1205 out=64 fin=length: `### Summary  The provided text consists of two distinct articles: one detailing `
- #17 summarize in=1083 out=64 fin=length: `The printing press revolutionized European society by making books affordable an`

**summarize @ 8**

- #24 summarize in=1111 out=64 fin=length: `The first text details the history of lighthouse keeping, where keepers maintain`
- #25 summarize in=1366 out=64 fin=length: `The provided text consists of two distinct articles. Here is a summary of each: `
