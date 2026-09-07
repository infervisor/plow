# bench-api results

- server: `http://localhost:8094`  model: `g12b`
- time: 2026-09-07T01:18:33+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: g12b

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 626/642/691 | 453/450/457 | 28.98/29.46 | 2.2 | 0.03 |
| chat_short | 2 | 8 | 0 | 45 | 63.8 | 1177/1219/1375 | 445/446/447 | 29.26/29.33 | 4.4 | 0.07 |
| chat_long | 1 | 8 | 0 | 398 | 59.4 | 1792/2073/2103 | 448/448/450 | 29.63/30.43 | 2.1 | 0.04 |
| chat_long | 2 | 8 | 0 | 442 | 59.8 | 2523/2724/3068 | 463/474/479 | 31.97/32.42 | 4.0 | 0.07 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is harvested, the fruit (cherry) must be removed to get to the seed `

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `To help you prepare for your talk, here is a breakdown of the 50% rule as it app`
- #9 chat_short in=50 out=64 fin=length: `Here is a way to explain it to a ten-year-old:  **The printing press changed boo`

**chat_long @ 1**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 2**

- #8 chat_long in=502 out=64 fin=length: `The difference between spring and neap tides lies in the range of the water's mo`
- #9 chat_long in=472 out=64 fin=length: `Based on the text provided, there is no mention of a halo around the sun or moon`
