# bench-api results

- server: `http://localhost:8094`  model: `gpt-oss-20b`
- time: 2026-09-06T09:53:08+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gpt-oss-20b

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 8 | 0 | 344 | 64.0 | 1253/1337/1431 | 76/81/82 | 6.35/6.52 | 10.6 | 0.17 |
| code | 2 | 8 | 0 | 357 | 64.0 | 2151/2317/2436 | 81/80/87 | 7.52/7.65 | 17.6 | 0.27 |
| code | 4 | 8 | 0 | 383 | 64.0 | 2944/2977/3085 | 115/116/116 | 10.15/10.26 | 25.1 | 0.39 |
| code | 8 | 8 | 0 | 337 | 64.0 | 4156/4157/4158 | 152/152/152 | 13.74/13.75 | 37.1 | 0.58 |
| summarize | 1 | 8 | 0 | 1111 | 64.0 | 1829/1821/2403 | 71/71/72 | 6.30/6.89 | 10.1 | 0.16 |
| summarize | 2 | 8 | 0 | 1124 | 64.0 | 3593/4152/4239 | 85/87/88 | 9.20/9.63 | 14.3 | 0.22 |
| summarize | 4 | 8 | 0 | 1188 | 64.0 | 6735/7372/7486 | 116/113/113 | 14.48/14.59 | 18.2 | 0.28 |
| summarize | 8 | 8 | 0 | 1136 | 64.0 | 7219/7281/7282 | 153/153/153 | 16.94/16.94 | 30.2 | 0.47 |

## Samples

**code @ 1**

- #0 code in=441 out=64 fin=length: `We need to produce code that logs 0,1,2 after 1,2,3 seconds. The original code u`
- #1 code in=284 out=64 fin=length: `We need to produce a corrected SQL query that lists every customer with total sp`

**code @ 2**

- #8 code in=471 out=64 fin=length: `We need to identify problems: logic errors: counts update wrong: if word in coun`
- #9 code in=257 out=64 fin=length: `We need to identify problems in the code:  - counts dict logic wrong: if word in`

**code @ 4**

- #16 code in=355 out=64 fin=length: `We need to rewrite function totalPrice(items) to sum price field across array of`
- #17 code in=424 out=64 fin=length: `We need to write merge_intervals function. Input list of tuples (int,int). Need `

**code @ 8**

- #24 code in=330 out=64 fin=length: `We need to identify problems: counts logic wrong: if word in counts: counts[word`
- #25 code in=379 out=64 fin=length: `We need to fix the lower_bound function. The bug: when xs[mid] < target, we set `

**summarize @ 1**

- #0 summarize in=813 out=64 fin=length: `We need to summarize the article in 3-5 bullet points. The article covers sundia`
- #1 summarize in=1434 out=64 fin=length: `We need to produce a two-sentence summary of the entire passage, then a list of `

**summarize @ 2**

- #8 summarize in=1215 out=64 fin=length: `We need to summarize the article in under 120 words, preserving most important s`
- #9 summarize in=1190 out=64 fin=length: `We need to summarize the article in 3-5 bullet points. The article covers lighth`

**summarize @ 4**

- #16 summarize in=1203 out=64 fin=length: `We need to summarize the article above. The article is about lighthouses and kee`
- #17 summarize in=1094 out=64 fin=length: `The user wants: "Give me a two-sentence summary followed by a list of the key fa`

**summarize @ 8**

- #24 summarize in=1117 out=64 fin=length: `We need to summarize the article in under 120 words, preserving most important s`
- #25 summarize in=1363 out=64 fin=length: `We need to summarize the article in 3-5 bullet points. The article is long, cove`
