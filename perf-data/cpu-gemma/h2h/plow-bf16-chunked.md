# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T02:42:34+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_long | 4 | 8 | 0 | 398 | 60.8 | 8067/8322/13374 | 370/376/444 | 28.88/38.00 | 7.7 | 0.13 |
| chat_long | 8 | 8 | 0 | 398 | 60.8 | 16884/16727/29305 | 630/685/746 | 54.77/56.35 | 7.9 | 0.13 |
| summarize | 4 | 8 | 0 | 1105 | 64.0 | 18622/15039/27683 | 644/753/765 | 60.15/75.10 | 4.2 | 0.07 |
| summarize | 8 | 8 | 0 | 1105 | 64.0 | 48718/48223/87495 | 1102/1268/1465 | 119.86/131.20 | 3.8 | 0.06 |

## Samples

**chat_long @ 4**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 8**

- #0 chat_long in=512 out=48 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**summarize @ 4**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
- #1 summarize in=1411 out=64 fin=length: `This text explores the geological processes of glacial erosion and deposition, a`

**summarize @ 8**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient civilizati`
- #1 summarize in=1411 out=64 fin=length: `This text explores the geological processes of glacial erosion and deposition, a`
