# bench-api results

- server: `http://localhost:8097`  model: `gemma-4-12b-it-q8_0`
- time: 2026-09-05T22:39:26+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gemma-4-12b-it-q8_0

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 910/939/1027 | 132/133/133 | 9.30/9.40 | 6.9 | 0.11 |
| chat_short | 2 | 8 | 0 | 34 | 64.0 | 390/406/408 | 154/154/154 | 10.10/10.10 | 12.7 | 0.20 |
| chat_short | 4 | 8 | 0 | 34 | 64.0 | 734/769/773 | 306/313/313 | 20.48/20.49 | 12.8 | 0.20 |
| chat_short | 8 | 8 | 0 | 34 | 64.0 | 686/686/688 | 191/191/191 | 12.72/12.72 | 40.2 | 0.63 |
| chat_long | 1 | 8 | 0 | 398 | 59.4 | 5630/6038/6839 | 135/135/136 | 13.25/14.60 | 4.4 | 0.07 |
| chat_long | 2 | 8 | 0 | 398 | 59.4 | 1697/921/4153 | 183/172/225 | 11.70/15.04 | 9.2 | 0.15 |
| chat_long | 4 | 8 | 0 | 398 | 59.4 | 2553/1577/4769 | 260/308/312 | 16.51/24.43 | 12.4 | 0.21 |
| chat_long | 8 | 8 | 0 | 398 | 59.4 | 3269/3983/3985 | 282/302/309 | 22.02/22.18 | 21.4 | 0.36 |
| code | 1 | 8 | 0 | 365 | 64.0 | 4989/4886/6062 | 135/135/136 | 13.50/14.55 | 4.7 | 0.07 |
| code | 2 | 8 | 0 | 365 | 64.0 | 1126/1148/1380 | 247/274/274 | 18.44/18.62 | 7.7 | 0.12 |
| code | 4 | 8 | 0 | 365 | 64.0 | 1754/1757/1759 | 386/439/439 | 29.40/29.41 | 9.8 | 0.15 |
| code | 8 | 8 | 0 | 365 | 64.0 | 2150/2150/2153 | 217/217/217 | 15.81/15.82 | 32.4 | 0.51 |
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 14572/16236/18507 | 140/141/142 | 25.21/27.43 | 2.7 | 0.04 |
| summarize | 2 | 8 | 0 | 1105 | 64.0 | 13281/18558/19892 | 219/178/281 | 30.98/34.72 | 4.7 | 0.07 |
| summarize | 4 | 8 | 0 | 1105 | 64.0 | 18687/19837/21638 | 251/224/338 | 35.37/35.70 | 7.4 | 0.12 |
| summarize | 8 | 8 | 0 | 1105 | 64.0 | 17700/20842/20845 | 318/270/401 | 37.82/37.83 | 13.5 | 0.21 |
| rag_4k | 1 | 8 | 0 | 2962 | 49.5 | 42632/43935/46010 | 140/141/143 | 50.48/53.70 | 1.0 | 0.02 |
| rag_4k | 2 | 8 | 0 | 2962 | 49.5 | 1876/1859/1944 | 266/286/311 | 14.23/17.67 | 6.4 | 0.13 |
| rag_4k | 4 | 8 | 0 | 2962 | 50.4 | 12763/3105/39750 | 517/400/1060 | 59.54/64.49 | 4.9 | 0.10 |
| rag_4k | 8 | 8 | 0 | 2962 | 50.4 | 15711/3326/31986 | 1000/984/1674 | 63.39/65.34 | 6.2 | 0.12 |

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
- #1 code in=304 out=64 fin=length: `'''sql /* Approach: 1. Use a LEFT JOIN to ensure customers with zero orders are `

**code @ 2**

- #0 code in=463 out=64 fin=length: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`
- #1 code in=304 out=64 fin=length: `'''sql /* Approach: 1. Use a LEFT JOIN to ensure customers with zero orders are `

**code @ 4**

- #0 code in=463 out=64 fin=length: `### Explanation of the Bug The issue is caused by **closure behavior** in JavaSc`
- #1 code in=304 out=64 fin=length: `'''sql /* Approach: 1. Use a LEFT JOIN to ensure customers with zero orders are `

**code @ 8**

- #0 code in=463 out=64 fin=length: `### Explanation of the Bug The issue is caused by **closures** in JavaScript. Th`
- #1 code in=304 out=64 fin=length: `'''sql /* Approach: 1. Use a LEFT JOIN to ensure customers with zero orders are `

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

**rag_4k @ 1**

- #0 rag_4k in=3306 out=64 fin=length: `The tides affect the length of the day and the distance to the Moon through fric`
- #1 rag_4k in=3195 out=55 fin=stop: `During the roasting process, the first crack occurs when the internal pressure o`

**rag_4k @ 2**

- #0 rag_4k in=3306 out=64 fin=length: `The tides affect the length of the day and the distance to the Moon through fric`
- #1 rag_4k in=3195 out=55 fin=stop: `During the roasting process, the first crack occurs when the internal pressure o`

**rag_4k @ 4**

- #0 rag_4k in=3306 out=64 fin=length: `The tides affect the length of the day and the distance to the Moon through fric`
- #1 rag_4k in=3195 out=61 fin=stop: `During the roasting of green coffee beans, the first crack occurs when the inter`

**rag_4k @ 8**

- #0 rag_4k in=3306 out=64 fin=length: `The tides affect the length of the day and the distance to the Moon through fric`
- #1 rag_4k in=3195 out=61 fin=stop: `During the roasting of green coffee beans, the first crack occurs when the inter`
