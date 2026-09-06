# bench-api results

- server: `http://localhost:8097`  model: `/home/lava/llamacpp/gemma-4-26b-a4b-it-q8_0.gguf`
- time: 2026-09-06T07:04:05+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: /home/lava/llamacpp/gemma-4-26b-a4b-it-q8_0.gguf

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 365 | 64.0 | 4221/4304/5061 | 60/59/61 | 8.16/8.84 | 8.0 | 0.13 |
| code | 2 | 8 | 0 | 374 | 64.0 | 8636/8994/9525 | 112/120/120 | 16.41/16.52 | 8.2 | 0.13 |
| code | 4 | 8 | 0 | 401 | 64.0 | 18089/18764/18765 | 131/132/132 | 26.91/26.92 | 9.7 | 0.15 |
| code | 8 | 8 | 0 | 359 | 64.0 | 27944/40902/41038 | 132/133/133 | 49.31/49.40 | 10.4 | 0.16 |

## Samples

**code @ 1**

- #0 code in=463 out=64 fin=length: `<|channel>thought<channel|>The issue is caused by the use of 'var', which has fu`
- #1 code in=304 out=64 fin=length: `<|channel>thought<channel|>### SQL Solution  '''sql /* APPROACH: 1. Use a LEFT J`

**code @ 2**

- #8 code in=497 out=64 fin=length: `<|channel>thought<channel|>'''python import string from collections import Count`
- #9 code in=283 out=64 fin=length: `<|channel>thought<channel|>### Identified Problems  1.  **Case Sensitivity**: Th`

**code @ 4**

- #16 code in=375 out=64 fin=length: `<|channel>thought<channel|>'''javascript function totalPrice(items) {     // Han`
- #17 code in=435 out=64 fin=length: `<|channel>thought<channel|>'''python import logging  # Configure logging for ope`

**code @ 8**

- #24 code in=354 out=64 fin=length: `<|channel>thought<channel|>'''python import string import collections  def top_w`
- #25 code in=408 out=64 fin=length: `<|channel>thought<channel|>'''rust fn lower_bound(xs: &[i32], target: i32) -> Op`
