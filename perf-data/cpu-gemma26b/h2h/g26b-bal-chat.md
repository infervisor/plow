# bench-api results

- server: `http://localhost:8096`  model: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- time: 2026-09-06T12:47:42+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 4d7ae4984b7db7de8f8457170b3f1a419ee76d52

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 464/471/513 | 41/41/41 | 3.04/3.08 | 21.1 | 0.33 |
| chat_short | 2 | 8 | 0 | 45 | 62.5 | 703/619/1081 | 65/67/69 | 4.81/4.83 | 26.3 | 0.42 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 1111/1033/1531 | 102/103/110 | 7.58/8.15 | 33.6 | 0.53 |
| chat_short | 8 | 8 | 0 | 36 | 62.0 | 2370/2617/3782 | 151/155/170 | 11.86/12.09 | 40.6 | 0.65 |
| chat_long | 1 | 8 | 0 | 398 | 54.6 | 2222/2471/2913 | 43/43/44 | 4.21/5.30 | 12.0 | 0.22 |
| chat_long | 2 | 8 | 0 | 442 | 59.5 | 3307/3751/4776 | 86/99/118 | 9.17/9.25 | 14.0 | 0.24 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 3949/4236/4864 | 152/162/176 | 14.38/15.13 | 18.9 | 0.29 |
| chat_long | 8 | 8 | 0 | 425 | 59.0 | 9336/9634/16400 | 308/339/392 | 27.98/28.22 | 16.6 | 0.28 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick drones out at the end of summer to ensure the colony's survival during`
- #1 chat_short in=21 out=64 fin=length: `When choosing between washed and natural processed coffee, you are essentially c`

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `To keep your talk engaging and practical, it is best to present this rule as a "`
- #9 chat_short in=50 out=64 fin=length: `The printing press changed books from unique, hand-drawn objects into standardiz`

**chat_short @ 4**

- #16 chat_short in=34 out=64 fin=length: `Since you are a hobbyist, the easiest way to think about propolis is to view it `
- #17 chat_short in=46 out=64 fin=length: `Richard Trevithick’s 1804 locomotive broke the rails because of a mismatch in we`

**chat_short @ 8**

- #24 chat_short in=34 out=64 fin=length: `The reason for this conflict boils down to a simple physical problem: **before t`
- #25 chat_short in=26 out=48 fin=stop: `Bees expel drones at the end of summer to conserve the hive's limited food suppl`

**chat_long @ 1**

- #0 chat_long in=512 out=55 fin=stop: `The passage recommends keeping an emergency fund in a **"separate savings accoun`
- #1 chat_long in=381 out=64 fin=length: `An escapement is the core mechanism of a mechanical clock that regulates the rel`

**chat_long @ 2**

- #8 chat_long in=502 out=64 fin=length: `The difference between spring and neap tides lies in the range of the tidal heig`
- #9 chat_long in=472 out=64 fin=length: `Based on the provided text, there is no mention of a halo around the sun or moon`

**chat_long @ 4**

- #16 chat_long in=250 out=64 fin=length: `### The Diagnosis  The most likely cause of the "flat" taste is **oxidation and `
- #17 chat_long in=358 out=64 fin=length: `Think of it this way: a honeybee colony is like a massive, living machine, and t`

**chat_long @ 8**

- #24 chat_long in=283 out=64 fin=length: `To avoid the "romantic" trap, you should frame the lighthouse not as a lonely se`
- #25 chat_long in=587 out=24 fin=stop: `The passage does not mention Fresnel's lens, nor does it state how far a lightho`
