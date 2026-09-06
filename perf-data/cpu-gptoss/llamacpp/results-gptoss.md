# bench-api results

- server: `http://localhost:8098`  model: `gpt-oss-20b-mxfp4`
- time: 2026-09-05T23:48:31+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gpt-oss-20b-mxfp4

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 32 | 64.0 | 614/616/679 | 35/35/36 | 2.84/2.91 | 22.5 | 0.35 |
| chat_short | 2 | 8 | 0 | 32 | 64.0 | 1063/1057/1468 | 56/56/56 | 4.56/5.02 | 27.9 | 0.44 |
| chat_short | 4 | 8 | 0 | 32 | 64.0 | 1055/1083/1175 | 87/87/88 | 6.61/6.70 | 39.0 | 0.61 |
| chat_short | 8 | 8 | 0 | 32 | 64.0 | 2282/2235/2362 | 145/145/145 | 11.38/11.47 | 44.6 | 0.70 |
| chat_long | 1 | 8 | 0 | 392 | 64.0 | 3833/4715/5128 | 44/45/45 | 7.54/7.95 | 9.7 | 0.15 |
| chat_long | 2 | 8 | 0 | 392 | 64.0 | 872/305/2491 | 62/62/62 | 4.21/6.48 | 26.8 | 0.42 |
| chat_long | 4 | 8 | 0 | 392 | 64.0 | 1754/2986/2989 | 113/125/127 | 10.96/10.97 | 28.8 | 0.45 |
| chat_long | 8 | 8 | 0 | 392 | 64.0 | 1551/1572/1572 | 178/178/178 | 12.76/12.76 | 40.1 | 0.63 |
| code | 1 | 8 | 0 | 344 | 64.0 | 3367/3458/4275 | 43/42/43 | 6.07/7.02 | 10.6 | 0.16 |
| code | 2 | 8 | 0 | 344 | 64.0 | 3917/3998/8448 | 61/62/62 | 7.77/12.34 | 16.5 | 0.26 |
| code | 4 | 8 | 0 | 344 | 64.0 | 486/496/498 | 99/100/100 | 6.79/6.80 | 38.0 | 0.59 |
| code | 8 | 8 | 0 | 344 | 64.0 | 754/812/813 | 172/172/172 | 11.64/11.65 | 44.0 | 0.69 |
| summarize | 1 | 8 | 0 | 1111 | 64.0 | 8566/8221/13450 | 50/52/53 | 11.16/16.71 | 5.5 | 0.09 |
| summarize | 2 | 8 | 0 | 1111 | 64.0 | 6708/7341/13804 | 94/99/106 | 14.05/20.10 | 10.1 | 0.16 |
| summarize | 4 | 8 | 0 | 1111 | 64.0 | 9114/8825/13464 | 206/216/251 | 22.27/22.43 | 11.5 | 0.18 |
| summarize | 8 | 8 | 0 | 1111 | 64.0 | 1200/1117/1338 | 236/236/236 | 15.99/16.15 | 31.7 | 0.50 |
| rag_4k | 1 | 8 | 0 | 2968 | 64.0 | 31663/33195/33961 | 66/66/68 | 37.36/38.28 | 1.8 | 0.03 |
| rag_4k | 2 | 8 | 0 | 2968 | 64.0 | 24616/23726/70371 | 140/133/140 | 37.50/79.19 | 3.8 | 0.06 |
| rag_4k | 4 | 8 | 0 | 2968 | 64.0 | 1014/1097/1335 | 236/242/244 | 16.06/16.08 | 16.1 | 0.25 |
| rag_4k | 8 | 8 | 0 | 2968 | 64.0 | 1057/1011/1011 | 389/390/390 | 25.60/25.62 | 20.0 | 0.31 |

## Samples

**chat_short @ 1**

- #0 chat_short in=25 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Why do bees kick drones out of `
- #1 chat_short in=22 out=64 fin=length: `<|channel|>analysis<|message|>The user asks: "Hi! What are the tradeoffs between`

**chat_short @ 2**

- #0 chat_short in=25 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Why do bees kick drones out of `
- #1 chat_short in=22 out=64 fin=length: `<|channel|>analysis<|message|>The user asks: "Hi! What are the tradeoffs between`

**chat_short @ 4**

- #0 chat_short in=25 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Why do bees kick drones out of `
- #1 chat_short in=22 out=64 fin=length: `<|channel|>analysis<|message|>The user asks: "Hi! What are the tradeoffs between`

**chat_short @ 8**

- #0 chat_short in=25 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Why do bees kick drones out of `
- #1 chat_short in=22 out=64 fin=length: `<|channel|>analysis<|message|>The user asks: "Hi! What are the tradeoffs between`

**chat_long @ 1**

- #0 chat_long in=506 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Where does the passage recommen`
- #1 chat_long in=383 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "What is an escapement and what`

**chat_long @ 2**

- #0 chat_long in=506 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Where does the passage recommen`
- #1 chat_long in=383 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "What is an escapement and what`

**chat_long @ 4**

- #0 chat_long in=506 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Where does the passage recommen`
- #1 chat_long in=383 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "What is an escapement and what`

**chat_long @ 8**

- #0 chat_long in=506 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: Where does the passage recommen`
- #1 chat_long in=383 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "What is an escapement and what`

**code @ 1**

- #0 code in=441 out=64 fin=length: `<|channel|>analysis<|message|>We need to explain why logs 3 three times. Because`
- #1 code in=284 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce a corrected SQL query that list`

**code @ 2**

- #0 code in=441 out=64 fin=length: `<|channel|>analysis<|message|>We need to explain why logs 3 three times. Because`
- #1 code in=284 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce a corrected SQL query that list`

**code @ 4**

- #0 code in=441 out=64 fin=length: `<|channel|>analysis<|message|>We need to explain why logs 3 three times. Because`
- #1 code in=284 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce a corrected SQL query that list`

**code @ 8**

- #0 code in=441 out=64 fin=length: `<|channel|>analysis<|message|>We need to explain why logs 3 three times. Because`
- #1 code in=284 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce a corrected SQL query that list`

**summarize @ 1**

- #0 summarize in=813 out=64 fin=length: `<|channel|>analysis<|message|>We need to summarize the article in 3-5 bullet poi`
- #1 summarize in=1434 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce a two-sentence summary of the e`

**summarize @ 2**

- #0 summarize in=813 out=64 fin=length: `<|channel|>analysis<|message|>We need to summarize the article in 3-5 bullet poi`
- #1 summarize in=1434 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce a two-sentence summary of the e`

**summarize @ 4**

- #0 summarize in=813 out=64 fin=length: `<|channel|>analysis<|message|>We need to summarize the article in 3-5 bullet poi`
- #1 summarize in=1434 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce a two-sentence summary of the e`

**summarize @ 8**

- #0 summarize in=813 out=64 fin=length: `<|channel|>analysis<|message|>We need to summarize the article in 3-5 bullet poi`
- #1 summarize in=1434 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce a two-sentence summary of the e`

**rag_4k @ 1**

- #0 rag_4k in=3317 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: How are tides slowly changing t`
- #1 rag_4k in=3214 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "What happens at the first crac`

**rag_4k @ 2**

- #0 rag_4k in=3317 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: How are tides slowly changing t`
- #1 rag_4k in=3214 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "What happens at the first crac`

**rag_4k @ 4**

- #0 rag_4k in=3317 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: How are tides slowly changing t`
- #1 rag_4k in=3214 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "What happens at the first crac`

**rag_4k @ 8**

- #0 rag_4k in=3317 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: How are tides slowly changing t`
- #1 rag_4k in=3214 out=64 fin=length: `<|channel|>analysis<|message|>We need to answer: "What happens at the first crac`
