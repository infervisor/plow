# bench-api results

- server: `http://localhost:8096`  model: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- time: 2026-09-06T19:54:00+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 4d7ae4984b7db7de8f8457170b3f1a419ee76d52

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 365 | 64.0 | 919/932/1026 | 38/38/39 | 3.33/3.45 | 19.2 | 0.30 |
| code | 2 | 8 | 0 | 374 | 64.0 | 1851/2165/2624 | 69/74/80 | 6.45/6.57 | 20.6 | 0.32 |
| code | 4 | 8 | 0 | 401 | 64.0 | 2892/2717/4057 | 128/131/152 | 10.92/12.12 | 23.3 | 0.36 |
| code | 8 | 8 | 0 | 359 | 64.0 | 4184/4576/6670 | 167/176/201 | 14.76/14.96 | 34.0 | 0.53 |
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 2806/3331/3543 | 40/40/40 | 5.86/6.08 | 12.0 | 0.19 |
| summarize | 2 | 8 | 0 | 1119 | 64.0 | 3396/3401/3652 | 97/108/110 | 9.34/10.54 | 13.5 | 0.21 |
| summarize | 4 | 8 | 0 | 1178 | 64.0 | 5905/4111/9965 | 191/214/235 | 18.75/23.46 | 14.2 | 0.22 |
| summarize | 8 | 8 | 0 | 1133 | 64.0 | 16562/17050/26987 | 389/438/540 | 41.14/41.60 | 12.2 | 0.19 |

## Samples

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
