# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T13:04:38+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 686/687/719 | 88/88/89 | 6.23/6.27 | 10.3 | 0.16 |
| chat_short | 2 | 8 | 0 | 45 | 62.8 | 977/834/1500 | 129/132/137 | 9.12/9.38 | 13.9 | 0.22 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 1661/1604/2284 | 216/218/226 | 15.29/16.04 | 16.6 | 0.26 |
| chat_short | 8 | 8 | 0 | 36 | 62.9 | 3573/3974/5735 | 244/250/268 | 19.00/19.38 | 25.6 | 0.41 |
| chat_long | 1 | 8 | 0 | 398 | 60.1 | 2445/2716/3359 | 90/90/92 | 7.98/8.24 | 7.7 | 0.13 |
| chat_long | 2 | 8 | 0 | 442 | 60.2 | 3671/4425/5340 | 152/169/185 | 13.87/13.95 | 9.3 | 0.15 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 4239/4666/5117 | 265/276/285 | 22.12/22.53 | 12.1 | 0.19 |
| chat_long | 8 | 8 | 0 | 425 | 58.5 | 10590/11209/18807 | 405/460/515 | 36.03/36.47 | 12.8 | 0.22 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When choosing between washed (wet) and natural (dry) processed coffees, the prim`

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `To help with your talk, here is a breakdown of the 50% rule for fixed costs, exp`
- #9 chat_short in=50 out=64 fin=length: `The printing press changed the physical appearance of books by standardizing lay`

**chat_short @ 4**

- #16 chat_short in=34 out=64 fin=length: `Think of **propolis** as the **"bee glue"** or the **"sealant"** of the hive.  #`
- #17 chat_short in=46 out=64 fin=length: `Richard Trevithick’s first steam locomotive (1804) broke the rails primarily due`

**chat_short @ 8**

- #24 chat_short in=34 out=64 fin=length: `The reason early railways forced towns to give up local time comes down to one p`
- #25 chat_short in=26 out=55 fin=stop: `Bees kick out drones at the end of summer because their primary role is to mate `

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
