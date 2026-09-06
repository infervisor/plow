# bench-api results

- server: `http://localhost:8094`  model: `gpt-oss-20b`
- time: 2026-09-06T09:44:51+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gpt-oss-20b

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 32 | 64.0 | 641/644/672 | 69/68/71 | 4.95/5.10 | 12.9 | 0.20 |
| chat_short | 2 | 8 | 0 | 42 | 64.0 | 1022/1066/1094 | 81/82/83 | 6.21/6.30 | 20.9 | 0.33 |

## Samples

**chat_short @ 1**

- #0 chat_short in=25 out=64 fin=length: `We need to answer: Why do bees kick drones out of the hive at the end of summer?`
- #1 chat_short in=22 out=64 fin=length: `The user asks: "Hi! What are the tradeoffs between washed and natural processed `

**chat_short @ 2**

- #8 chat_short in=46 out=64 fin=length: `We need to explain the 50% rule for fixed costs in a household budget. Provide c`
- #9 chat_short in=46 out=64 fin=length: `We need to answer: "How did the printing press change the way books looked, not `
