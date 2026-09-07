# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T23:21:50+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 63.8 | 690/686/743 | 233/233/235 | 15.36/15.48 | 4.2 | 0.07 |
| chat_short | 2 | 8 | 0 | 45 | 64.0 | 1267/1629/1645 | 247/251/254 | 16.95/16.98 | 7.6 | 0.12 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 1766/1623/2550 | 268/269/279 | 18.60/19.50 | 13.6 | 0.21 |
| chat_short | 8 | 8 | 0 | 36 | 63.1 | 3857/4228/6213 | 334/340/359 | 25.04/25.64 | 19.5 | 0.31 |
| chat_long | 1 | 8 | 0 | 398 | 57.6 | 2249/2513/3121 | 233/235/235 | 16.15/17.44 | 3.7 | 0.06 |
| chat_long | 2 | 8 | 0 | 442 | 59.8 | 3507/4235/5073 | 269/285/299 | 20.97/21.15 | 6.1 | 0.10 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 3982/4128/4852 | 291/300/310 | 22.98/24.13 | 11.4 | 0.18 |
| chat_long | 8 | 8 | 0 | 425 | 59.0 | 10209/12285/17035 | 460/512/559 | 37.49/38.32 | 12.2 | 0.21 |

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

**chat_short @ 8**

- #24 chat_short in=34 out=64 fin=length: `The reason early railways forced towns to give up local time comes down to one p`
- #25 chat_short in=26 out=57 fin=stop: `Bees kick out drones at the end of summer because they do not contribute to the `

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
- #25 chat_long in=587 out=24 fin=stop: `The passage does not provide information regarding Fresnel's lens or the specifi`
