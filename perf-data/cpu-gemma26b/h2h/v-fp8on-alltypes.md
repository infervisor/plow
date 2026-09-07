# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-07T00:57:53+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 437/432/481 | 128/127/129 | 8.48/8.54 | 7.6 | 0.12 |
| chat_short | 2 | 8 | 0 | 45 | 62.8 | 676/656/1003 | 145/148/148 | 9.85/9.97 | 12.8 | 0.20 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 1138/1038/1601 | 205/206/213 | 14.07/14.67 | 18.0 | 0.28 |
| chat_short | 8 | 8 | 0 | 36 | 63.6 | 2386/2673/3874 | 200/203/215 | 15.07/15.40 | 32.7 | 0.51 |
| chat_long | 1 | 8 | 0 | 398 | 58.1 | 2223/2423/2990 | 130/131/132 | 9.81/10.76 | 6.0 | 0.10 |
| chat_long | 2 | 8 | 0 | 442 | 61.2 | 3389/3852/4877 | 168/181/201 | 14.43/14.60 | 8.8 | 0.14 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 4038/4275/5070 | 256/265/276 | 21.06/21.75 | 12.6 | 0.20 |
| chat_long | 8 | 8 | 0 | 425 | 58.8 | 10664/12349/17335 | 388/345/471 | 32.03/32.36 | 14.4 | 0.25 |

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

**chat_short @ 8**

- #24 chat_short in=34 out=64 fin=length: `The reason early railways forced towns to give up local time comes down to one p`
- #25 chat_short in=26 out=61 fin=stop: `Bees kick out drones at the end of summer because they do not contribute to the `

**chat_long @ 1**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 2**

- #8 chat_long in=502 out=64 fin=length: `The difference between spring and neap tides lies in the range of the water's mo`
- #9 chat_long in=472 out=64 fin=length: `Based on the text provided, there is no mention of a halo around the sun or moon`

**chat_long @ 4**

- #16 chat_long in=250 out=64 fin=length: `The most likely cause of the "flat" taste is **oxidation and moisture absorption`
- #17 chat_long in=358 out=64 fin=length: `Think of a honeybee colony like a massive, living machine where the "jobs" chang`

**chat_long @ 8**

- #24 chat_long in=283 out=64 fin=length: `To move away from the "romantic" trope of the lonely lighthouse keeper, your pie`
- #25 chat_long in=587 out=22 fin=stop: `The passage does not provide information regarding Fresnel's lens or how far suc`
