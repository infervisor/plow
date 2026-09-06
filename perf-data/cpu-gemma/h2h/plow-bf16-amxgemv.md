# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T04:24:42+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 4 | 8 | 0 | 34 | 64.0 | 2201/2214/2979 | 323/325/333 | 22.84/23.32 | 11.2 | 0.18 |
| chat_short | 8 | 8 | 0 | 45 | 63.8 | 4892/5413/7840 | 344/352/380 | 26.72/27.28 | 18.5 | 0.29 |
| chat_long | 4 | 8 | 0 | 398 | 57.6 | 7386/7840/10197 | 419/412/470 | 28.63/36.09 | 7.1 | 0.12 |
| chat_long | 8 | 8 | 0 | 442 | 60.5 | 17477/17939/29101 | 535/626/672 | 50.23/51.10 | 9.4 | 0.15 |

## Samples

**chat_short @ 4**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is harvested, the fruit (cherry) must be removed to get to the seed `

**chat_short @ 8**

- #8 chat_short in=49 out=64 fin=length: `To help you prepare for your talk, here is a breakdown of the 50% rule specifica`
- #9 chat_short in=50 out=64 fin=length: `Here is a way to explain it to a ten-year-old:  **The printing press changed boo`

**chat_long @ 4**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 8**

- #8 chat_long in=502 out=64 fin=length: `The difference between spring and neap tides lies in the range of the water's mo`
- #9 chat_long in=472 out=64 fin=length: `Based on the text provided, there is no mention of a halo around the sun or moon`
