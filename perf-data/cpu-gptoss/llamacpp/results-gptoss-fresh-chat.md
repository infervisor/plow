# bench-api results

- server: `http://localhost:8098`  model: `gpt-oss-20b-mxfp4`
- time: 2026-09-06T08:24:52+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gpt-oss-20b-mxfp4

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 32 | 64.0 | 748/731/824 | 41/41/42 | 3.36/3.44 | 19.2 | 0.30 |
| chat_short | 2 | 8 | 0 | 42 | 64.0 | 1625/1658/1719 | 65/65/67 | 5.79/5.90 | 22.3 | 0.35 |
| chat_short | 4 | 8 | 0 | 38 | 64.0 | 3059/3083/3178 | 104/105/105 | 9.63/9.71 | 26.6 | 0.42 |
| chat_short | 8 | 8 | 0 | 34 | 64.0 | 5260/5261/5261 | 177/177/177 | 16.40/16.40 | 31.2 | 0.49 |
| chat_long | 1 | 8 | 0 | 392 | 64.0 | 4823/5958/6453 | 51/52/54 | 9.33/9.72 | 8.0 | 0.12 |
| chat_long | 2 | 8 | 0 | 439 | 64.0 | 9520/12030/12442 | 81/75/100 | 16.78/17.14 | 8.7 | 0.14 |
| chat_long | 4 | 8 | 0 | 344 | 64.0 | 14432/15155/15613 | 133/149/150 | 24.79/25.01 | 11.2 | 0.18 |
| chat_long | 8 | 8 | 0 | 423 | 64.0 | 35666/35794/36033 | 215/213/219 | 49.23/49.39 | 10.4 | 0.16 |

## Samples

**chat_short @ 1**

- #0 chat_short in=25 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Why do bees kick drones out of `
- #1 chat_short in=22 out=64 fin=length: `<|channel|>analysis<|message|>The user asks: "Hi! What are the tradeoffs between`

**chat_short @ 2**

- #8 chat_short in=46 out=64 fin=length: `<|channel|>analysis<|message|>We need to explain the 50% rule for fixed costs in`
- #9 chat_short in=46 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "How did the printing press cha`

**chat_short @ 4**

- #16 chat_short in=32 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "What is propolis and what do b`
- #17 chat_short in=41 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Why did Trevithick's first loco`

**chat_short @ 8**

- #24 chat_short in=33 out=64 fin=length: `<|channel|>analysis<|message|>The user asks: "I've read conflicting things onlin`
- #25 chat_short in=26 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Why do bees kick drones out of `

**chat_long @ 1**

- #0 chat_long in=506 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Where does the passage recommen`
- #1 chat_long in=383 out=64 fin=length: `<|channel|>analysis<|message|>The user wants: "What is an escapement and what do`

**chat_long @ 2**

- #8 chat_long in=501 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: difference between spring tides`
- #9 chat_long in=473 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "What does a halo around the su`

**chat_long @ 4**

- #16 chat_long in=236 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: most likely cause of flat taste`
- #17 chat_long in=357 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "What distinguishes winter bees`

**chat_long @ 8**

- #24 chat_long in=262 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce an outline for a 1,200-word pie`
- #25 chat_long in=581 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "According to the passage, what`
