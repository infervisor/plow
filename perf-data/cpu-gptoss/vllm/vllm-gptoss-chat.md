# bench-api results

- server: `http://localhost:8094`  model: `gpt-oss-20b`
- time: 2026-09-06T09:48:45+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gpt-oss-20b

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 32 | 64.0 | 637/633/645 | 71/71/73 | 5.09/5.23 | 12.6 | 0.20 |
| chat_short | 2 | 8 | 0 | 42 | 64.0 | 1220/1252/1272 | 95/99/102 | 7.41/7.71 | 17.7 | 0.28 |
| chat_short | 4 | 8 | 0 | 38 | 64.0 | 1657/1645/1694 | 103/104/104 | 8.19/8.19 | 31.3 | 0.49 |
| chat_short | 8 | 8 | 0 | 34 | 64.0 | 2120/2132/2132 | 136/136/136 | 10.71/10.71 | 47.8 | 0.75 |
| chat_long | 1 | 8 | 0 | 392 | 64.0 | 1346/1462/1483 | 71/69/75 | 5.82/5.95 | 11.0 | 0.17 |
| chat_long | 2 | 8 | 0 | 439 | 64.0 | 1962/2044/2662 | 80/80/80 | 7.11/7.75 | 18.2 | 0.29 |
| chat_long | 4 | 8 | 0 | 344 | 64.0 | 2550/2741/2743 | 102/102/102 | 9.13/9.13 | 28.5 | 0.45 |
| chat_long | 8 | 8 | 0 | 423 | 64.0 | 4591/4592/4593 | 137/137/137 | 13.25/13.25 | 38.5 | 0.60 |

## Samples

**chat_short @ 1**

- #0 chat_short in=25 out=64 fin=length: `We need to answer: Why do bees kick drones out of the hive at the end of summer?`
- #1 chat_short in=22 out=64 fin=length: `The user asks: "Hi! What are the tradeoffs between washed and natural processed `

**chat_short @ 2**

- #8 chat_short in=46 out=64 fin=length: `We need to explain the 50% rule for fixed costs in a household budget, and evalu`
- #9 chat_short in=46 out=64 fin=length: `We need to answer: "How did the printing press change the way books looked, not `

**chat_short @ 4**

- #16 chat_short in=32 out=64 fin=length: `We need to answer: "What is propolis and what do bees use it for? A short exampl`
- #17 chat_short in=41 out=64 fin=length: `We need to answer: Why did Trevithick's first locomotive break the rails it ran `

**chat_short @ 8**

- #24 chat_short in=33 out=64 fin=length: `The user asks: "I've read conflicting things online about this. Why did early ra`
- #25 chat_short in=26 out=64 fin=length: `We need to answer: Why do bees kick drones out of the hive at the end of summer?`

**chat_long @ 1**

- #0 chat_long in=506 out=64 fin=length: `We need to answer: Where does the passage recommend keeping an emergency fund, a`
- #1 chat_long in=383 out=64 fin=length: `We need to answer: "What is an escapement and what does it do? Give a clear answ`

**chat_long @ 2**

- #8 chat_long in=501 out=64 fin=length: `We need to answer: difference between spring tides and neap tides, and what caus`
- #9 chat_long in=473 out=64 fin=length: `We need to answer: "What does a halo around the sun or moon signify, and why?" P`

**chat_long @ 4**

- #16 chat_long in=236 out=64 fin=length: `We need to answer: most likely cause of flat taste, changes to buying and grindi`
- #17 chat_long in=357 out=64 fin=length: `We need to answer: "What distinguishes winter bees from summer bees?" Explanatio`

**chat_long @ 8**

- #24 chat_long in=262 out=64 fin=length: `We need to produce an outline for a 1,200-word piece, with sections, each with a`
- #25 chat_long in=581 out=64 fin=length: `We need to answer: "According to the passage, what did Fresnel's lens change abo`
