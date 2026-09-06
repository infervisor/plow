# bench-api results

- server: `http://localhost:8094`  model: `gemma-4-12b-it`
- time: 2026-09-06T01:28:41+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gemma-4-12b-it

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 628/649/690 | 516/520/521 | 33.37/33.53 | 1.9 | 0.03 |
| chat_short | 2 | 8 | 0 | 34 | 64.0 | 1036/1089/1152 | 441/442/443 | 28.90/28.99 | 4.4 | 0.07 |
| chat_short | 4 | 8 | 0 | 34 | 64.0 | 1436/1553/1765 | 445/442/451 | 29.61/29.61 | 8.7 | 0.14 |
| chat_short | 8 | 8 | 0 | 34 | 63.8 | 1941/2139/2139 | 449/447/447 | 30.28/30.28 | 16.8 | 0.26 |
| chat_long | 1 | 8 | 0 | 398 | 61.2 | 1719/1930/2011 | 441/441/442 | 29.11/29.61 | 2.2 | 0.04 |
| chat_long | 2 | 8 | 0 | 398 | 61.2 | 1094/1370/1412 | 446/448/450 | 29.35/29.71 | 4.2 | 0.07 |
| chat_long | 4 | 8 | 0 | 398 | 57.4 | 1487/1589/1751 | 457/457/467 | 29.71/30.55 | 7.6 | 0.13 |
| chat_long | 8 | 8 | 0 | 398 | 61.2 | 2076/2306/2306 | 462/459/459 | 31.20/31.20 | 15.7 | 0.26 |
| code | 1 | 8 | 0 | 365 | 64.0 | 1591/1626/1794 | 445/445/446 | 29.70/29.83 | 2.2 | 0.03 |
| code | 2 | 8 | 0 | 365 | 64.0 | 1326/1385/1493 | 446/446/449 | 29.40/29.87 | 4.3 | 0.07 |
| code | 4 | 8 | 0 | 365 | 64.0 | 1713/1959/1990 | 452/449/460 | 30.29/30.29 | 8.5 | 0.13 |
| code | 8 | 8 | 0 | 365 | 64.0 | 2463/2703/2704 | 461/458/458 | 31.56/31.56 | 16.2 | 0.25 |
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 2732/2634/3741 | 449/449/450 | 30.94/32.10 | 2.1 | 0.03 |
| summarize | 2 | 8 | 0 | 1105 | 64.0 | 4224/4154/6099 | 467/461/498 | 34.29/34.42 | 3.8 | 0.06 |
| summarize | 4 | 8 | 0 | 1105 | 64.0 | 6542/7110/11574 | 509/512/549 | 39.51/41.92 | 6.6 | 0.10 |
| summarize | 8 | 8 | 0 | 1105 | 64.0 | 16141/13637/15920 | 559/550/628 | 48.32/48.76 | 6.7 | 0.11 |

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
