# bench-api results

- server: `http://localhost:8096`  model: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- time: 2026-09-06T08:01:45+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 4d7ae4984b7db7de8f8457170b3f1a419ee76d52

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 7667/9060/9824 | 90/90/91 | 14.82/15.52 | 4.8 | 0.07 |
| summarize | 2 | 8 | 0 | 1119 | 64.0 | 8972/8828/9831 | 232/261/265 | 23.36/26.53 | 5.4 | 0.08 |
| summarize | 4 | 8 | 0 | 1178 | 64.0 | 15335/10519/26560 | 485/546/603 | 48.62/60.97 | 5.5 | 0.09 |
| summarize | 8 | 8 | 0 | 1133 | 64.0 | 35315/38449/56473 | 765/877/1065 | 83.72/84.95 | 6.0 | 0.09 |

## Samples

**summarize @ 1**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  * **Early Timekeeping:** Ancient civilization`
- #1 summarize in=1411 out=64 fin=length: `This text provides a detailed overview of two distinct subjects: the geological `

**summarize @ 2**

- #8 summarize in=1223 out=64 fin=length: `The article provides guides on household budgeting and coffee production. Effect`
- #9 summarize in=1187 out=64 fin=length: `Here is a summary of the article:  * **The Role of the Lighthouse Keeper:** Keep`

**summarize @ 4**

- #16 summarize in=1205 out=64 fin=length: `### Summary  The provided text consists of two distinct articles.   The first, *`
- #17 summarize in=1083 out=64 fin=length: `The printing press revolutionized European society by transforming books from ra`

**summarize @ 8**

- #24 summarize in=1111 out=64 fin=length: `The article details the history of lighthouse keeping and the chemistry of bread`
- #25 summarize in=1366 out=64 fin=length: `Here is a summary of the provided text:  *   **Evolution of Timekeeping:** Human`
