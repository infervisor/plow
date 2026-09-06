# bench-api results

- server: `http://localhost:8096`  model: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- time: 2026-09-06T07:57:32+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 4d7ae4984b7db7de8f8457170b3f1a419ee76d52

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 365 | 64.0 | 2484/2510/2854 | 86/86/87 | 7.94/8.35 | 8.1 | 0.13 |
| code | 2 | 8 | 0 | 374 | 64.0 | 4018/4505/5537 | 141/156/163 | 13.20/13.32 | 9.9 | 0.15 |
| code | 4 | 8 | 0 | 401 | 64.0 | 5871/5595/8451 | 271/277/318 | 23.04/25.76 | 11.1 | 0.17 |
| code | 8 | 8 | 0 | 359 | 64.0 | 11427/12671/18290 | 451/473/547 | 39.99/40.60 | 12.5 | 0.20 |

## Samples

**code @ 1**

- #0 code in=463 out=64 fin=length: `### Explanation The issue is caused by **variable hoisting** and **closures**. B`
- #1 code in=304 out=64 fin=length: `Since the request asks for a "query" but provides context regarding "malformed i`

**code @ 2**

- #8 code in=497 out=64 fin=length: `### Identified Problems  1.  **Logic Error (Incrementing):** The current code se`
- #9 code in=283 out=64 fin=length: `### Identified Problems  1.  **Logic Error (Incrementing):** The 'if/else' logic`

**code @ 4**

- #16 code in=375 out=64 fin=length: `'''javascript function totalPrice(items) {     // Handle cases where items is nu`
- #17 code in=435 out=64 fin=length: `'''python import logging  # Configure logging for operations to monitor malforme`

**code @ 8**

- #24 code in=354 out=64 fin=length: `'''python import string from collections import Counter  def top_words(text):   `
- #25 code in=408 out=64 fin=length: `'''rust fn lower_bound(xs: &[i32], target: i32) -> Option<usize> {     // Handle`
