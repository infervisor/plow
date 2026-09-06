# bench-api results

- server: `http://localhost:8097`  model: `gemma-4-12b-it-q4_k_m`
- time: 2026-09-05T23:17:27+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gemma-4-12b-it-q4_k_m

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 657/662/745 | 121/121/121 | 8.29/8.31 | 7.8 | 0.12 |
| chat_short | 2 | 8 | 0 | 34 | 64.0 | 543/609/624 | 222/244/247 | 16.00/16.15 | 8.8 | 0.14 |
| chat_short | 4 | 8 | 0 | 34 | 64.0 | 565/715/719 | 230/289/289 | 18.92/18.92 | 17.0 | 0.27 |
| chat_short | 8 | 8 | 0 | 34 | 64.0 | 1307/1621/1623 | 197/195/201 | 13.89/13.90 | 36.8 | 0.58 |
| chat_long | 1 | 8 | 0 | 398 | 54.0 | 3734/4000/4667 | 122/123/123 | 10.43/11.42 | 5.3 | 0.10 |
| chat_long | 2 | 8 | 0 | 398 | 52.4 | 1471/1050/2801 | 216/213/254 | 13.85/15.67 | 8.0 | 0.15 |
| chat_long | 4 | 8 | 0 | 398 | 52.9 | 1887/1956/1958 | 360/401/407 | 19.13/27.41 | 9.1 | 0.17 |
| chat_long | 8 | 8 | 0 | 398 | 52.9 | 2167/2167/2169 | 260/284/284 | 20.04/20.05 | 21.1 | 0.40 |
| code | 1 | 8 | 0 | 365 | 64.0 | 3271/3199/4023 | 123/123/124 | 10.98/11.82 | 5.8 | 0.09 |
| code | 2 | 8 | 0 | 365 | 64.0 | 1237/1349/1475 | 228/249/249 | 16.76/17.06 | 8.2 | 0.13 |
| code | 4 | 8 | 0 | 365 | 64.0 | 1679/2203/2207 | 301/414/414 | 28.27/28.28 | 12.4 | 0.19 |
| code | 8 | 8 | 0 | 365 | 64.0 | 2002/2003/2003 | 218/218/218 | 15.72/15.72 | 32.6 | 0.51 |
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 10030/11349/12800 | 128/129/129 | 19.52/20.95 | 3.5 | 0.06 |
| summarize | 2 | 8 | 0 | 1105 | 64.0 | 9535/12632/14412 | 209/181/257 | 25.84/28.21 | 5.6 | 0.09 |
| summarize | 4 | 8 | 0 | 1105 | 64.0 | 9541/9504/9505 | 258/251/302 | 28.54/28.55 | 9.9 | 0.15 |
| summarize | 8 | 8 | 0 | 1105 | 64.0 | 4670/5396/5396 | 274/265/288 | 22.08/22.08 | 23.2 | 0.36 |
| rag_4k | 1 | 8 | 0 | 2962 | 44.4 | 29384/30177/32612 | 127/127/128 | 34.90/39.40 | 1.3 | 0.03 |
| rag_4k | 2 | 8 | 0 | 2962 | 44.2 | 5986/1799/2937 | 310/257/276 | 14.19/47.80 | 4.2 | 0.09 |
| rag_4k | 4 | 8 | 0 | 2962 | 44.4 | 2297/2179/3165 | 346/364/384 | 20.12/22.88 | 8.8 | 0.20 |
| rag_4k | 8 | 8 | 0 | 2962 | 44.5 | 13540/3495/30619 | 966/1027/1440 | 56.55/57.51 | 6.1 | 0.14 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is processed, it refers to how the fruit (cherry) is removed from th`

**chat_short @ 2**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is processed, it refers to how the fruit (cherry) is removed from th`

**chat_short @ 4**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is processed, it refers to how the fruit (cherry) is removed from th`

**chat_short @ 8**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones at the end of summer primarily because they are **non-produ`
- #1 chat_short in=21 out=64 fin=length: `When coffee is processed, it refers to how the fruit (cherry) is removed from th`

**chat_long @ 1**

- #0 chat_long in=512 out=47 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 2**

- #0 chat_long in=512 out=39 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 4**

- #0 chat_long in=512 out=39 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**chat_long @ 8**

- #0 chat_long in=512 out=39 fin=stop: `The passage recommends keeping an emergency fund in "a separate savings account `
- #1 chat_long in=381 out=64 fin=length: `An escapement is a mechanical device that serves as the "heart" of a mechanical `

**code @ 1**

- #0 code in=463 out=64 fin=length: `### Explanation of the Bug The issue is caused by **closure scope**. In the orig`
- #1 code in=304 out=64 fin=length: `'''sql /* APPROACH: 1. Use a LEFT JOIN to ensure customers with no orders are in`

**code @ 2**

- #0 code in=463 out=64 fin=length: `### Explanation of the Bug The issue is caused by **closure scope**. In the orig`
- #1 code in=304 out=64 fin=length: `'''sql /* APPROACH: 1. Use a LEFT JOIN to ensure customers with no orders are in`

**code @ 4**

- #0 code in=463 out=64 fin=length: `### Explanation of the Bug The issue is caused by **closure scope**. In the orig`
- #1 code in=304 out=64 fin=length: `'''sql /* APPROACH: 1. Use a LEFT JOIN to ensure customers with no orders are in`

**code @ 8**

- #0 code in=463 out=64 fin=length: `### Explanation of the Bug The issue is caused by **closure scope**. In the orig`
- #1 code in=304 out=64 fin=length: `'''sql /* APPROACH: 1. Use a LEFT JOIN to ensure customers with no orders are in`

**summarize @ 1**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Early Timekeeping:** Ancient methods li`
- #1 summarize in=1411 out=64 fin=length: `The provided text explores the geological processes of glacial erosion and depos`

**summarize @ 2**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Early Timekeeping:** Early methods like`
- #1 summarize in=1411 out=64 fin=length: `The provided text explores the geological processes of glacial erosion and depos`

**summarize @ 4**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Early Timekeeping:** Early methods like`
- #1 summarize in=1411 out=64 fin=length: `The provided text explores the geological processes of glacial erosion and depos`

**summarize @ 8**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Early Timekeeping:** Early methods like`
- #1 summarize in=1411 out=64 fin=length: `The provided text explores the geological processes of glacial erosion and depos`

**rag_4k @ 1**

- #0 rag_4k in=3306 out=64 fin=length: `Tides have a slow, one-directional effect on both the Earth's rotation and the M`
- #1 rag_4k in=3195 out=53 fin=stop: `At the first crack during the roasting of coffee beans, the internal pressure ca`

**rag_4k @ 2**

- #0 rag_4k in=3306 out=64 fin=length: `Tides have a slow, one-directional effect on both the Earth's rotation and the M`
- #1 rag_4k in=3195 out=58 fin=stop: `At the point of the first crack during the roasting process, the internal pressu`

**rag_4k @ 4**

- #0 rag_4k in=3306 out=64 fin=length: `Tides have a slow, one-directional effect on both the Earth's rotation and the M`
- #1 rag_4k in=3195 out=53 fin=stop: `At the first crack during the roasting of coffee beans, the internal pressure ca`

**rag_4k @ 8**

- #0 rag_4k in=3306 out=64 fin=length: `Tides have a slow, one-directional effect on both the Earth's rotation and the M`
- #1 rag_4k in=3195 out=54 fin=stop: `At the first crack during the roasting of green coffee beans, the internal press`
