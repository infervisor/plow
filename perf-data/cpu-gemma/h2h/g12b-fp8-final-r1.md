# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T17:01:59+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 466/453/503 | 134/134/135 | 8.92/8.94 | 7.2 | 0.11 |
| chat_long | 1 | 8 | 0 | 398 | 59.2 | 2451/2633/3328 | 135/136/136 | 10.32/11.18 | 5.7 | 0.10 |
| code | 1 | 8 | 0 | 365 | 64.0 | 2156/2179/2530 | 136/136/137 | 10.80/11.07 | 6.0 | 0.09 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is harvested, the fruit (cherry) must be removed to get to the seed `

**chat_long @ 1**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**code @ 1**

- #0 code in=463 out=64 fin=length: `### Explanation The issue is caused by **closures** and the way the 'i' variable`
- #1 code in=304 out=64 fin=length: `'''sql /* Approach: 1. Use a LEFT JOIN to ensure customers with zero orders are `
