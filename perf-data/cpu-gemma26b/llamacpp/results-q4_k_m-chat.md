# bench-api results

- server: `http://localhost:8097`  model: `/home/lava/llamacpp/gemma-4-26b-a4b-it-q4_k_m.gguf`
- time: 2026-09-06T07:20:35+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: /home/lava/llamacpp/gemma-4-26b-a4b-it-q4_k_m.gguf

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 649/668/729 | 50/50/51 | 3.84/3.97 | 16.8 | 0.26 |
| chat_short | 2 | 8 | 0 | 45 | 62.0 | 1226/1458/1461 | 80/85/86 | 6.23/6.58 | 20.0 | 0.32 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 2508/2567/2567 | 107/108/108 | 9.30/9.30 | 27.6 | 0.43 |
| chat_short | 8 | 8 | 0 | 36 | 64.0 | 6492/10881/10882 | 105/107/107 | 17.61/17.61 | 29.1 | 0.45 |
| chat_long | 1 | 8 | 0 | 398 | 55.2 | 4187/4827/5238 | 52/52/54 | 7.09/7.51 | 7.9 | 0.14 |
| chat_long | 2 | 8 | 0 | 442 | 58.1 | 7009/7357/9875 | 125/109/165 | 16.68/16.84 | 8.0 | 0.14 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 12936/13104/13213 | 119/122/122 | 20.80/20.89 | 12.5 | 0.20 |
| chat_long | 8 | 8 | 0 | 425 | 58.9 | 29015/42753/43020 | 118/122/122 | 45.84/50.12 | 9.4 | 0.16 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `<|channel>thought<channel|>At the end of summer, worker bees begin to evict dron`
- #1 chat_short in=21 out=64 fin=length: `<|channel>thought<channel|>When deciding between washed and natural processed co`

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `<|channel>thought<channel|>Since you are preparing notes for a talk, I have stru`
- #9 chat_short in=50 out=64 fin=length: `<|channel>thought<channel|>The printing press changed the look of books by repla`

**chat_short @ 4**

- #16 chat_short in=34 out=64 fin=length: `<|channel>thought<channel|>Since you're a hobbyist, the best way to think about `
- #17 chat_short in=46 out=64 fin=length: `<|channel>thought<channel|>Richard Trevithick’s 1804 locomotive broke the rails `

**chat_short @ 8**

- #24 chat_short in=34 out=64 fin=length: `<|channel>thought<channel|>To understand why railways forced towns to give up lo`
- #25 chat_short in=26 out=64 fin=stop: `<|channel>thought<channel|>Bees kick drones out of the hive at the end of summer`

**chat_long @ 1**

- #0 chat_long in=512 out=42 fin=stop: `<|channel>thought<channel|>The passage recommends keeping an emergency fund in "`
- #1 chat_long in=381 out=64 fin=length: `<|channel>thought<channel|>An escapement is the core mechanism of a mechanical c`

**chat_long @ 2**

- #8 chat_long in=502 out=64 fin=length: `<|channel>thought<channel|>The difference between spring and neap tides lies in `
- #9 chat_long in=472 out=64 fin=length: `<|channel>thought<channel|>Based on the provided text, there is no mention of a `

**chat_long @ 4**

- #16 chat_long in=250 out=64 fin=length: `<|channel>thought<channel|>### The Diagnosis: What is causing the "flat" taste? `
- #17 chat_long in=358 out=64 fin=length: `<|channel>thought<channel|>Think of a honeybee colony like a massive, living mac`

**chat_long @ 8**

- #24 chat_long in=283 out=64 fin=length: `<|channel>thought<channel|>This is a compelling subject for a local history piec`
- #25 chat_long in=587 out=23 fin=stop: `<|channel>thought<channel|>The passage does not mention Fresnel's lens or how fa`
