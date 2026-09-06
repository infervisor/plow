# bench-api results

- server: `http://localhost:8097`  model: `/home/lava/llamacpp/gemma-4-26b-a4b-it-q8_0.gguf`
- time: 2026-09-06T06:54:02+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: /home/lava/llamacpp/gemma-4-26b-a4b-it-q8_0.gguf

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 739/783/817 | 56/54/60 | 4.31/4.51 | 15.0 | 0.23 |
| chat_short | 2 | 8 | 0 | 45 | 62.9 | 1351/1546/1559 | 84/88/93 | 6.84/7.16 | 19.0 | 0.30 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 2852/2862/2864 | 115/117/117 | 10.22/10.22 | 25.4 | 0.40 |
| chat_short | 8 | 8 | 0 | 36 | 64.0 | 7024/11776/11777 | 119/125/125 | 19.65/19.66 | 26.0 | 0.41 |
| chat_long | 1 | 8 | 0 | 398 | 59.1 | 4532/5090/5649 | 56/56/56 | 7.90/8.85 | 7.6 | 0.13 |
| chat_long | 2 | 8 | 0 | 442 | 54.2 | 6252/6873/10645 | 159/170/224 | 17.59/18.08 | 7.1 | 0.13 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 13887/14092/14213 | 126/127/127 | 22.10/22.23 | 11.7 | 0.18 |
| chat_long | 8 | 8 | 0 | 425 | 58.9 | 29260/23304/44214 | 167/141/142 | 46.79/53.15 | 8.9 | 0.15 |
| code | 1 | 8 | 0 | 365 | 64.0 | 4205/4167/4978 | 57/58/58 | 7.83/8.60 | 8.2 | 0.13 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `<|channel>thought<channel|>Bees kick drones out at the end of summer primarily t`
- #1 chat_short in=21 out=64 fin=length: `<|channel>thought<channel|>The choice between washed and natural processed coffe`

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `<|channel>thought<channel|>To keep your talk engaging and practical, it is best `
- #9 chat_short in=50 out=64 fin=length: `<|channel>thought<channel|>The printing press changed books from unique, hand-dr`

**chat_short @ 4**

- #16 chat_short in=34 out=64 fin=length: `<|channel>thought<channel|>Since you're a hobbyist, the easiest way to think abo`
- #17 chat_short in=46 out=64 fin=length: `<|channel>thought<channel|>Richard Trevithick’s 1804 locomotive broke the rails `

**chat_short @ 8**

- #24 chat_short in=34 out=64 fin=length: `<|channel>thought<channel|>To understand why railways forced towns to give up lo`
- #25 chat_short in=26 out=64 fin=length: `<|channel>thought<channel|>Bees kick drones out of the hive at the end of summer`

**chat_long @ 1**

- #0 chat_long in=512 out=57 fin=stop: `<|channel>thought<channel|>The passage recommends keeping an emergency fund in a`
- #1 chat_long in=381 out=64 fin=length: `<|channel>thought<channel|>An escapement is the core mechanism of a mechanical c`

**chat_long @ 2**

- #8 chat_long in=502 out=64 fin=length: `<|channel>thought<channel|>Spring tides and neap tides differ in their tidal ran`
- #9 chat_long in=472 out=41 fin=stop: `<|channel>thought<channel|>Based on the provided text, there is no mention of a `

**chat_long @ 4**

- #16 chat_long in=250 out=64 fin=length: `<|channel>thought<channel|>### The Diagnosis: What is causing the "flat" taste? `
- #17 chat_long in=358 out=64 fin=length: `<|channel>thought<channel|>So, you want to know the difference between the bees `

**chat_long @ 8**

- #24 chat_long in=283 out=64 fin=length: `<|channel>thought<channel|>To write this piece effectively, you need to bridge t`
- #25 chat_long in=587 out=23 fin=stop: `<|channel>thought<channel|>The passage does not mention Fresnel's lens or how fa`

**code @ 1**

- #0 code in=463 out=64 fin=length: `<|channel>thought<channel|>The issue is caused by the use of 'var', which has fu`
- #1 code in=304 out=64 fin=length: `<|channel>thought<channel|>### SQL Solution  '''sql /* APPROACH: 1. Use a LEFT J`
