# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-07T01:06:47+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 681/686/704 | 81/81/81 | 5.78/5.84 | 11.1 | 0.17 |
| chat_short | 2 | 8 | 0 | 45 | 63.4 | 974/853/1472 | 119/124/126 | 8.59/8.70 | 14.9 | 0.24 |
| chat_short | 4 | 8 | 0 | 41 | 63.9 | 1730/1564/2436 | 197/198/207 | 14.25/14.44 | 17.9 | 0.28 |
| chat_short | 8 | 8 | 0 | 36 | 63.1 | 3450/3862/5519 | 232/237/257 | 18.12/18.48 | 27.0 | 0.43 |
| chat_long | 1 | 8 | 0 | 398 | 60.5 | 2260/2539/3144 | 83/83/86 | 7.21/7.94 | 8.3 | 0.14 |
| chat_long | 2 | 8 | 0 | 442 | 59.5 | 3421/4039/4899 | 136/151/167 | 12.61/12.71 | 10.2 | 0.17 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 4126/4356/4819 | 237/251/251 | 20.14/20.64 | 13.3 | 0.21 |
| chat_long | 8 | 8 | 0 | 425 | 58.5 | 9854/10319/17405 | 424/459/498 | 33.62/34.19 | 13.6 | 0.23 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily to **conserve resources**.  `
- #1 chat_short in=21 out=64 fin=length: `When choosing between washed (wet) and natural (dry) processed coffees, the prim`

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `To help with your talk, here is a breakdown of the 50% rule, how it works in pra`
- #9 chat_short in=50 out=64 fin=length: `The printing press changed the physical appearance of books by standardizing lay`

**chat_short @ 4**

- #16 chat_short in=34 out=64 fin=length: `Think of **propolis** as the **"bee glue"** or the **"sealant"** of the hive.  #`
- #17 chat_short in=46 out=64 fin=length: `Richard Trevithick’s first steam locomotive (1804) broke the rails primarily due`

**chat_short @ 8**

- #24 chat_short in=34 out=64 fin=length: `The reason early railways forced towns to give up local time comes down to one p`
- #25 chat_short in=26 out=57 fin=stop: `Bees kick out drones at the end of summer because their primary role is to mate `

**chat_long @ 1**

- #0 chat_long in=512 out=54 fin=stop: `The passage recommends keeping an emergency fund in a **"separate savings accoun`
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a clock. Its `

**chat_long @ 2**

- #8 chat_long in=502 out=64 fin=length: `The difference between spring and neap tides lies in the magnitude of the tidal `
- #9 chat_long in=472 out=64 fin=length: `Based on the text provided, there is no mention of a halo around the sun or moon`

**chat_long @ 4**

- #16 chat_long in=250 out=64 fin=length: `The issue you are facing is a classic case of **stale oxidation and moisture abs`
- #17 chat_long in=358 out=64 fin=length: `Think of a honeybee colony like a giant, living machine where every bee has a sp`

**chat_long @ 8**

- #24 chat_long in=283 out=64 fin=length: `To move away from the "romantic" myth of the solitary, brooding keeper and towar`
- #25 chat_long in=587 out=20 fin=stop: `The passage does not mention Fresnel's lens or how far such a light could be see`
