# bench-api results

- server: `http://localhost:8096`  model: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- time: 2026-09-06T07:50:27+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 4d7ae4984b7db7de8f8457170b3f1a419ee76d52

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 63.8 | 800/809/834 | 83/83/84 | 6.04/6.07 | 10.6 | 0.17 |
| chat_short | 2 | 8 | 0 | 45 | 62.6 | 1147/979/1788 | 130/135/138 | 9.43/9.57 | 13.4 | 0.21 |
| chat_short | 4 | 8 | 0 | 41 | 64.0 | 1943/1820/2701 | 210/213/223 | 15.17/16.17 | 16.7 | 0.26 |
| chat_short | 8 | 8 | 0 | 36 | 64.0 | 4287/4691/6980 | 347/355/376 | 26.28/26.89 | 18.9 | 0.29 |
| chat_long | 1 | 8 | 0 | 398 | 59.2 | 2779/3071/3604 | 85/85/86 | 8.01/8.43 | 7.6 | 0.13 |
| chat_long | 2 | 8 | 0 | 442 | 59.4 | 4167/4607/6053 | 154/169/197 | 14.21/14.60 | 8.9 | 0.15 |
| chat_long | 4 | 8 | 0 | 349 | 64.0 | 5133/5424/6513 | 268/278/297 | 23.01/24.04 | 11.6 | 0.18 |
| chat_long | 8 | 8 | 0 | 425 | 58.5 | 12496/13073/21284 | 577/609/662 | 45.07/45.65 | 10.2 | 0.17 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick drones out at the end of summer primarily for **survival and resource `
- #1 chat_short in=21 out=64 fin=length: `When choosing between washed and natural processed coffee, you are essentially c`

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `To keep your talk engaging and practical, it is best to present this rule as a "`
- #9 chat_short in=50 out=64 fin=length: `The printing press changed books from unique, hand-drawn works of art into stand`

**chat_short @ 4**

- #16 chat_short in=34 out=64 fin=length: `Since you are a hobbyist, the easiest way to think about propolis is as **"bee g`
- #17 chat_short in=46 out=64 fin=length: `Richard Trevithick’s 1804 locomotive broke the rails because of a mismatch betwe`

**chat_short @ 8**

- #24 chat_short in=34 out=64 fin=length: `The reason for this shift wasn't about a desire for control or a lack of technol`
- #25 chat_short in=26 out=64 fin=length: `Bees expel drones at the end of summer to conserve precious food resources, such`

**chat_long @ 1**

- #0 chat_long in=512 out=53 fin=stop: `The passage recommends keeping an emergency fund in **"a separate savings accoun`
- #1 chat_long in=381 out=64 fin=length: `An escapement is the central mechanism of a mechanical clock that regulates the `

**chat_long @ 2**

- #8 chat_long in=502 out=64 fin=length: `The difference between spring and neap tides lies in the range of the water leve`
- #9 chat_long in=472 out=64 fin=length: `Based on the text provided, there is no mention of a halo around the sun or moon`

**chat_long @ 4**

- #16 chat_long in=250 out=64 fin=length: `### The Diagnosis: Oxidation and Staling  The most likely cause of the "flat" ta`
- #17 chat_long in=358 out=64 fin=length: `Think of it like the difference between a "special forces" team and a "regular c`

**chat_long @ 8**

- #24 chat_long in=283 out=64 fin=length: `To avoid the "romantic" trap, you should frame the history of the lighthouse not`
- #25 chat_long in=587 out=20 fin=stop: `The passage does not mention Fresnel's lens or how far such a light could be see`
