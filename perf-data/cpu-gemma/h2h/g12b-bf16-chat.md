# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T13:18:20+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 63.8 | 696/709/719 | 232/232/233 | 15.33/15.37 | 4.2 | 0.07 |
| chat_short | 2 | 8 | 0 | 45 | 64.0 | 1283/1643/1672 | 248/252/255 | 16.99/17.07 | 7.6 | 0.12 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 1800/1651/2592 | 270/272/282 | 18.73/19.75 | 13.5 | 0.21 |
| chat_long | 1 | 8 | 0 | 398 | 57.6 | 2437/2715/3436 | 233/235/237 | 16.32/17.53 | 3.7 | 0.06 |
| chat_long | 2 | 8 | 0 | 442 | 59.8 | 3800/4599/5496 | 274/288/308 | 21.45/21.80 | 5.9 | 0.10 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 4674/4756/6225 | 313/323/338 | 24.99/26.56 | 10.4 | 0.16 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is harvested, the fruit (cherry) must be removed to get to the seed `

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `To help you prepare for your talk, here is a breakdown of the 50% rule specifica`
- #9 chat_short in=50 out=64 fin=length: `The printing press changed the way books looked by standardizing layout, typogra`

**chat_short @ 4**

- #16 chat_short in=34 out=64 fin=length: `Think of **propolis** as **"bee glue."**  ### What is it? Propolis is a resinous`
- #17 chat_short in=46 out=64 fin=length: `Richard Trevithick’s first steam locomotive (1804) broke the rails primarily bec`

**chat_long @ 1**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 2**

- #8 chat_long in=502 out=64 fin=length: `The difference between spring and neap tides lies in the range of the water's mo`
- #9 chat_long in=472 out=64 fin=length: `Based on the text provided, there is no mention of a halo around the sun or moon`

**chat_long @ 4**

- #16 chat_long in=250 out=64 fin=length: `The most likely cause of the "flat" taste is **oxidation and moisture absorption`
- #17 chat_long in=358 out=64 fin=length: `Think of a honeybee colony like a massive, living machine where the "jobs" chang`
