# bench-api results

- server: `http://localhost:8096`  model: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- time: 2026-09-06T11:50:59+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 4d7ae4984b7db7de8f8457170b3f1a419ee76d52

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 536/551/585 | 41/41/41 | 3.09/3.14 | 20.6 | 0.32 |
| chat_short | 2 | 8 | 0 | 45 | 62.5 | 812/672/1302 | 65/69/70 | 4.97/5.06 | 25.5 | 0.41 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 1300/1217/1795 | 103/104/112 | 7.84/8.41 | 32.6 | 0.51 |
| chat_short | 8 | 8 | 0 | 36 | 62.0 | 2741/2967/4386 | 154/158/172 | 12.41/12.64 | 38.9 | 0.63 |
| chat_long | 1 | 8 | 0 | 398 | 54.6 | 2537/2853/3279 | 42/42/44 | 4.28/5.68 | 11.3 | 0.21 |
| chat_long | 2 | 8 | 0 | 442 | 59.5 | 3778/4358/5442 | 90/105/129 | 9.90/10.03 | 12.9 | 0.22 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 4552/4901/5639 | 158/171/186 | 15.54/16.39 | 17.5 | 0.27 |
| chat_long | 8 | 8 | 0 | 425 | 59.0 | 11050/12158/18570 | 306/341/423 | 30.32/30.61 | 15.4 | 0.26 |

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

- #24 chat_short in=34 out=64 fin=length: `The reason for this conflict comes down to a simple physical problem: **before t`
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
