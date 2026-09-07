# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T23:14:16+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 672/673/698 | 93/93/94 | 6.55/6.60 | 9.8 | 0.15 |
| chat_short | 2 | 8 | 0 | 45 | 62.8 | 972/829/1476 | 130/133/137 | 9.18/9.40 | 13.8 | 0.22 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 1623/1545/2244 | 203/205/213 | 14.45/15.15 | 17.6 | 0.27 |
| chat_short | 8 | 8 | 0 | 36 | 64.0 | 3538/3892/5725 | 239/245/265 | 18.71/19.09 | 26.5 | 0.41 |
| chat_long | 1 | 8 | 0 | 398 | 60.1 | 2267/2497/3096 | 95/95/95 | 8.00/8.35 | 7.6 | 0.13 |
| chat_long | 2 | 8 | 0 | 442 | 60.5 | 3418/4152/4971 | 149/161/179 | 13.20/13.41 | 9.7 | 0.16 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 4021/4236/4818 | 245/258/266 | 20.48/21.06 | 13.1 | 0.20 |
| chat_long | 8 | 8 | 0 | 425 | 58.5 | 9761/10281/17404 | 384/433/484 | 33.93/34.37 | 13.5 | 0.23 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily to **conserve resources**.  `
- #1 chat_short in=21 out=64 fin=length: `When choosing between washed (wet) and natural (dry) processed coffees, the prim`

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `To help with your talk, here is a breakdown of the 50% rule for fixed costs, exp`
- #9 chat_short in=50 out=64 fin=length: `The printing press changed the physical appearance of books by standardizing lay`

**chat_short @ 4**

- #16 chat_short in=34 out=64 fin=length: `Think of **propolis** as the **"bee glue"** or the **"sealant"** of the hive.  #`
- #17 chat_short in=46 out=64 fin=length: `Richard Trevithick’s first steam locomotive (1804) broke the rails primarily due`

**chat_short @ 8**

- #24 chat_short in=34 out=64 fin=length: `The reason early railways forced towns to give up local time comes down to one p`
- #25 chat_short in=26 out=64 fin=length: `Bees kick out drones at the end of summer because their primary role—mating with`

**chat_long @ 1**

- #0 chat_long in=512 out=54 fin=stop: `The passage recommends keeping an emergency fund in a **"separate savings accoun`
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a clock. Its `

**chat_long @ 2**

- #8 chat_long in=502 out=64 fin=length: `The difference between spring and neap tides lies in the magnitude of the tidal `
- #9 chat_long in=472 out=64 fin=length: `Based on the text provided, there is no mention of a halo around the sun or moon`

**chat_long @ 4**

- #16 chat_long in=250 out=64 fin=length: `The issue you are facing is a classic case of **stale oxidation.** Even though t`
- #17 chat_long in=358 out=64 fin=length: `Think of a honeybee colony like a giant, living machine where every bee has a sp`

**chat_long @ 8**

- #24 chat_long in=283 out=64 fin=length: `To move away from the "romantic" myth of the lonely lighthouse keeper, your piec`
- #25 chat_long in=587 out=20 fin=stop: `The passage does not mention Fresnel's lens or how far such a light could be see`
