# bench-api results

- server: `http://localhost:8094`  model: `gemma-4-12b-it`
- time: 2026-09-06T13:54:24+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gemma-4-12b-it

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 657/672/731 | 460/460/461 | 29.69/29.75 | 2.2 | 0.03 |
| chat_short | 2 | 8 | 0 | 45 | 64.0 | 1291/1329/1551 | 465/465/469 | 30.86/30.96 | 4.2 | 0.07 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 1760/2026/2028 | 462/459/472 | 30.95/30.95 | 8.3 | 0.13 |
| chat_long | 1 | 8 | 0 | 398 | 61.2 | 1959/2184/2295 | 496/489/544 | 31.36/35.73 | 1.9 | 0.03 |
| chat_long | 2 | 8 | 0 | 442 | 59.8 | 2790/3152/3240 | 519/544/572 | 37.74/38.42 | 3.6 | 0.06 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 4051/4649/4837 | 472/463/493 | 33.79/33.92 | 7.6 | 0.12 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is harvested, the fruit (cherry) must be removed to get to the seed `

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `To help you prepare for your talk, here is a breakdown of the 50% rule as it app`
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
