# bench-api results

- server: `http://localhost:8096`  model: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- time: 2026-09-06T11:55:15+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 4d7ae4984b7db7de8f8457170b3f1a419ee76d52

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 365 | 64.0 | 2229/2233/2683 | 43/43/44 | 4.93/5.46 | 13.0 | 0.20 |
| code | 2 | 8 | 0 | 374 | 64.0 | 3542/3834/4882 | 76/85/97 | 8.69/8.76 | 15.4 | 0.24 |
| code | 4 | 8 | 0 | 401 | 64.0 | 5275/5046/7730 | 160/167/202 | 15.73/17.96 | 16.6 | 0.26 |
| code | 8 | 8 | 0 | 359 | 64.0 | 10213/11584/16043 | 256/278/345 | 26.41/26.64 | 19.1 | 0.30 |
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 6960/7941/8976 | 45/45/46 | 10.81/11.87 | 6.5 | 0.10 |
| summarize | 2 | 8 | 0 | 1119 | 64.0 | 8046/7732/8660 | 161/184/189 | 18.19/20.59 | 7.0 | 0.11 |
| summarize | 4 | 8 | 0 | 1178 | 64.0 | 13594/9046/23422 | 356/421/447 | 38.56/49.96 | 7.1 | 0.11 |
| summarize | 8 | 8 | 0 | 1133 | 64.0 | 29595/30902/48478 | 593/684/864 | 67.05/67.60 | 7.5 | 0.12 |

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
