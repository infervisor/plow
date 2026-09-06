# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T00:56:41+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 563/564/596 | 161/160/166 | 10.59/11.07 | 6.0 | 0.09 |
| chat_short | 2 | 8 | 0 | 34 | 64.0 | 987/1231/1314 | 198/201/203 | 13.40/13.79 | 9.5 | 0.15 |
| chat_short | 4 | 8 | 0 | 34 | 64.0 | 1423/1442/1903 | 283/286/291 | 19.24/19.95 | 13.2 | 0.21 |
| chat_short | 8 | 8 | 0 | 34 | 64.0 | 3360/3619/5546 | 464/469/482 | 32.88/33.73 | 15.0 | 0.23 |
| chat_long | 1 | 8 | 0 | 398 | 59.4 | 2472/2530/3427 | 160/162/163 | 11.73/12.81 | 5.0 | 0.08 |
| chat_long | 2 | 8 | 0 | 398 | 58.1 | 3064/2818/4068 | 234/238/244 | 17.84/18.46 | 6.6 | 0.11 |
| chat_long | 4 | 8 | 0 | 398 | 59.2 | 5541/6113/7944 | 382/387/416 | 26.51/31.49 | 8.1 | 0.14 |
| chat_long | 8 | 8 | 0 | 398 | 59.2 | 11630/11945/18187 | 650/667/718 | 51.32/52.70 | 8.9 | 0.15 |

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
