# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T00:19:49+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 751/773/802 | 217/217/218 | 14.48/14.50 | 4.4 | 0.07 |
| chat_short | 2 | 8 | 0 | 34 | 64.0 | 1302/1639/1680 | 233/236/239 | 15.97/16.32 | 8.0 | 0.13 |
| chat_short | 4 | 8 | 0 | 34 | 64.0 | 1953/1857/2834 | 263/265/277 | 18.45/19.43 | 13.7 | 0.21 |
| chat_short | 8 | 8 | 0 | 34 | 64.0 | 4220/4560/6811 | 432/438/456 | 31.62/32.39 | 15.6 | 0.24 |
| chat_long | 1 | 8 | 0 | 398 | 60.8 | 2524/2619/3477 | 224/227/229 | 16.36/16.71 | 3.8 | 0.06 |
| chat_long | 2 | 8 | 0 | 398 | 60.8 | 3188/2969/4273 | 274/276/282 | 20.21/21.67 | 5.9 | 0.10 |
| chat_long | 4 | 8 | 0 | 398 | 60.8 | 5964/6407/8445 | 362/374/419 | 26.07/31.62 | 8.4 | 0.14 |
| chat_long | 8 | 8 | 0 | 398 | 60.8 | 12789/13785/19512 | 627/645/709 | 51.30/52.87 | 9.1 | 0.15 |
| code | 1 | 8 | 0 | 365 | 64.0 | 2435/2477/2718 | 246/246/248 | 17.91/18.36 | 3.6 | 0.06 |
| code | 2 | 8 | 0 | 365 | 64.0 | 3875/4480/5179 | 274/289/293 | 21.12/21.70 | 6.1 | 0.09 |
| code | 4 | 8 | 0 | 365 | 64.0 | 5514/5390/7459 | 346/353/374 | 27.87/29.57 | 9.3 | 0.15 |
| code | 8 | 8 | 0 | 365 | 64.0 | 11892/12749/18973 | 588/610/678 | 49.17/50.06 | 10.1 | 0.16 |
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 7329/8601/9467 | 250/251/254 | 24.19/25.44 | 2.8 | 0.04 |
| summarize | 2 | 8 | 0 | 1105 | 64.0 | 11457/12418/14833 | 316/328/398 | 31.22/35.05 | 4.1 | 0.06 |
| summarize | 4 | 8 | 0 | 1105 | 64.0 | 14595/14871/19168 | 536/585/661 | 49.54/54.73 | 5.3 | 0.08 |
| summarize | 8 | 8 | 0 | 1105 | 64.0 | 32067/32158/51332 | 933/1005/1194 | 91.11/91.99 | 5.5 | 0.09 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is harvested, the fruit (cherry) must be removed to get to the seed `

**chat_short @ 2**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is harvested, the fruit (cherry) must be removed to get to the seed `

**chat_short @ 4**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is harvested, the fruit (cherry) must be removed to get to the seed `

**chat_short @ 8**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is harvested, the fruit (cherry) must be removed to get to the seed `

**chat_long @ 1**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 2**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 4**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 8**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**code @ 1**

- #0 code in=463 out=64 fin=length: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`
- #1 code in=304 out=64 fin=length: `'''sql /* Approach: 1. Use a LEFT JOIN to ensure customers with no orders are in`

**code @ 2**

- #0 code in=463 out=64 fin=length: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`
- #1 code in=304 out=64 fin=length: `'''sql /* Approach: 1. Use a LEFT JOIN to ensure customers with no orders are in`

**code @ 4**

- #0 code in=463 out=64 fin=length: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`
- #1 code in=304 out=64 fin=length: `'''sql /* Approach: 1. Use a LEFT JOIN to ensure customers with no orders are in`

**code @ 8**

- #0 code in=463 out=64 fin=length: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`
- #1 code in=304 out=64 fin=length: `'''sql /* Approach: 1. Use a LEFT JOIN to ensure customers with no orders are in`

**summarize @ 1**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
- #1 summarize in=1411 out=64 fin=length: `This text explores the geological processes of glacial erosion and deposition, a`

**summarize @ 2**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
- #1 summarize in=1411 out=64 fin=length: `This text explores the geological processes of glacial erosion and deposition, a`

**summarize @ 4**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
- #1 summarize in=1411 out=64 fin=length: `This text explores the geological processes of glacial erosion and deposition, a`

**summarize @ 8**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
- #1 summarize in=1411 out=64 fin=length: `This text explores the geological processes of glacial erosion and deposition, a`
