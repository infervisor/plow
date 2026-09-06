# bench-api results

- server: `http://localhost:8097`  model: `/home/lava/llamacpp/gemma-4-26b-a4b-it-q8_0.gguf`
- time: 2026-09-06T07:03:06+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: /home/lava/llamacpp/gemma-4-26b-a4b-it-q8_0.gguf

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_long | 8 | 8 | 0 | 398 | 60.4 | 28262/43811/43940 | 130/131/131 | 49.66/51.78 | 9.3 | 0.15 |

## Samples

**chat_long @ 8**

- #0 chat_long in=512 out=57 fin=stop: `<|channel>thought<channel|>The passage recommends keeping an emergency fund in a`
- #1 chat_long in=381 out=64 fin=length: `<|channel>thought<channel|>An escapement is the core mechanism of a mechanical c`
