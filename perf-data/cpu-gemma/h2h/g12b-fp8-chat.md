# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T13:29:12+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 456/482/499 | 133/133/134 | 8.84/8.87 | 7.3 | 0.11 |
| chat_short | 2 | 8 | 0 | 45 | 62.9 | 739/690/1065 | 156/158/159 | 10.58/10.72 | 11.9 | 0.19 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 1217/1082/1707 | 219/221/226 | 15.08/15.66 | 16.8 | 0.26 |
| chat_long | 1 | 8 | 0 | 398 | 59.2 | 2424/2627/3278 | 137/137/139 | 10.46/11.29 | 5.7 | 0.10 |
| chat_long | 2 | 8 | 0 | 442 | 59.8 | 3734/4174/5433 | 181/194/218 | 15.53/15.69 | 8.2 | 0.14 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 4465/4737/5603 | 278/288/301 | 22.92/23.75 | 11.6 | 0.18 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is harvested, the fruit (cherry) must be removed to get to the seed `

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `To help you prepare for your talk, here is a breakdown of the 50/30/20 rule spec`
- #9 chat_short in=50 out=64 fin=length: `The printing press standardized the visual layout of books, moving them from uni`

**chat_short @ 4**

- #16 chat_short in=34 out=64 fin=length: `Think of **propolis** as **"bee glue."**  ### What is it? Propolis is a resinous`
- #17 chat_short in=46 out=64 fin=length: `Richard Trevithick’s first steam locomotive (1804) broke the rails primarily due`

**chat_long @ 1**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 2**

- #8 chat_long in=502 out=64 fin=length: `The difference between spring and neap tides lies in the range of the water's mo`
- #9 chat_long in=472 out=64 fin=length: `Based on the text provided, there is no mention of a halo around the sun or moon`

**chat_long @ 4**

- #16 chat_long in=250 out=64 fin=length: `The most likely cause of the "flat" taste is **oxidation and degradation of the `
- #17 chat_long in=358 out=64 fin=length: `Think of a honeybee colony like a massive, living machine where the "jobs" chang`
