# bench-api results

- server: `http://localhost:8097`  model: `/home/lava/llamacpp/gemma-4-26b-a4b-it-q4_k_m.gguf`
- time: 2026-09-06T14:21:45+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: /home/lava/llamacpp/gemma-4-26b-a4b-it-q4_k_m.gguf

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 4 | 8 | 0 | 34 | 64.0 | 2302/2400/2402 | 111/112/112 | 9.32/9.32 | 27.6 | 0.43 |
| chat_short | 8 | 8 | 0 | 45 | 62.0 | 4948/4949/4951 | 178/179/179 | 16.23/16.23 | 30.5 | 0.49 |
| chat_long | 4 | 8 | 0 | 398 | 53.8 | 11944/16594/16596 | 237/270/273 | 23.19/33.78 | 8.6 | 0.16 |
| chat_long | 8 | 8 | 0 | 442 | 58.2 | 34445/34547/34754 | 217/222/229 | 48.55/48.55 | 9.6 | 0.16 |

## Samples

**chat_short @ 4**

- #0 chat_short in=27 out=64 fin=length: `<|channel>thought<channel|>Bees kick drones out at the end of summer primarily t`
- #1 chat_short in=21 out=64 fin=length: `<|channel>thought<channel|>When deciding between washed and natural processed co`

**chat_short @ 8**

- #8 chat_short in=49 out=64 fin=length: `<|channel>thought<channel|>Since you are preparing notes for a talk, I have stru`
- #9 chat_short in=50 out=64 fin=length: `<|channel>thought<channel|>The printing press changed the look of books by repla`

**chat_long @ 4**

- #0 chat_long in=512 out=42 fin=stop: `<|channel>thought<channel|>The passage recommends keeping an emergency fund in "`
- #1 chat_long in=381 out=64 fin=length: `<|channel>thought<channel|>An escapement is the core mechanism of a mechanical c`

**chat_long @ 8**

- #8 chat_long in=502 out=64 fin=length: `<|channel>thought<channel|>The difference between spring and neap tides lies in `
- #9 chat_long in=472 out=64 fin=length: `<|channel>thought<channel|>Based on the provided text, there is no mention of a `
