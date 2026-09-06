# bench-api results

- server: `http://localhost:8097`  model: `gemma-4-12b-it-bf16`
- time: 2026-09-05T21:35:04+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gemma-4-12b-it-bf16

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 1487/1496/1836 | 267/267/268 | 18.26/18.60 | 3.5 | 0.05 |
| chat_short | 2 | 8 | 0 | 34 | 64.0 | 654/664/702 | 259/259/260 | 17.00/17.02 | 7.5 | 0.12 |
| chat_short | 4 | 8 | 0 | 34 | 64.0 | 1225/1419/1644 | 272/274/274 | 18.39/18.88 | 13.9 | 0.22 |
| chat_short | 8 | 8 | 0 | 34 | 64.0 | 1562/2004/2008 | 356/359/359 | 24.24/24.25 | 21.1 | 0.33 |
| chat_long | 1 | 8 | 0 | 398 | 59.4 | 11280/13090/14363 | 269/271/272 | 27.08/30.19 | 2.2 | 0.04 |
| chat_long | 2 | 8 | 0 | 398 | 60.8 | 3054/1680/7910 | 401/449/487 | 31.86/33.13 | 4.3 | 0.07 |
| chat_long | 4 | 8 | 0 | 398 | 59.4 | 4809/2605/8843 | 563/544/563 | 37.45/42.51 | 5.9 | 0.10 |
| chat_long | 8 | 8 | 0 | 398 | 59.4 | 3037/3038/3039 | 421/437/437 | 30.60/30.60 | 15.5 | 0.26 |
| code | 1 | 8 | 0 | 365 | 64.0 | 10348/10360/12726 | 268/268/269 | 27.32/29.66 | 2.3 | 0.04 |
| code | 2 | 8 | 0 | 365 | 64.0 | 1316/1367/1406 | 524/540/853 | 35.31/55.13 | 3.7 | 0.06 |
| code | 4 | 8 | 0 | 365 | 64.0 | 2171/2299/2301 | 554/555/555 | 37.27/37.27 | 6.9 | 0.11 |
| code | 8 | 8 | 0 | 365 | 64.0 | 2907/2907/2909 | 394/394/394 | 27.75/27.75 | 18.4 | 0.29 |
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 29798/33275/37744 | 282/283/285 | 51.11/55.73 | 1.3 | 0.02 |
| summarize | 2 | 8 | 0 | 1105 | 64.0 | 16355/22793/36704 | 344/291/507 | 54.74/57.70 | 3.4 | 0.05 |
| summarize | 4 | 8 | 0 | 1105 | 64.0 | 23671/26536/26537 | 925/1064/1290 | 107.83/107.84 | 3.1 | 0.05 |
| summarize | 8 | 8 | 0 | 1105 | 64.0 | 6735/7392/7394 | 446/438/469 | 35.01/35.01 | 14.6 | 0.23 |
| rag_4k | 1 | 8 | 5 | 3102 | 57.7 | 95105/93676/100187 | 501/367/848 | 111.83/145.52 | 0.4 | 0.01 |
| rag_4k | 2 | 8 | 8 | - | - | -/-/- | -/-/- | -/- | 0.0 | 0.00 |
| rag_4k | 4 | 8 | 8 | - | - | -/-/- | -/-/- | -/- | 0.0 | 0.00 |
| rag_4k | 8 | 8 | 8 | - | - | -/-/- | -/-/- | -/- | 0.0 | 0.00 |

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

**rag_4k @ 1**

- #0 rag_4k in=3306 out=64 fin=length: `The tides affect the length of the day and the distance to the Moon through fric`
- #1 rag_4k in=3195 out=55 fin=stop: `During the roasting process, the first crack occurs when the internal pressure o`

**rag_4k @ 2**

- #0 rag_4k in=3306 out=None fin=None ERR ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 8097): ``
- #1 rag_4k in=3195 out=None fin=None ERR ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 8097): ``

**rag_4k @ 4**

- #0 rag_4k in=3306 out=None fin=None ERR ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 8097): ``
- #1 rag_4k in=3195 out=None fin=None ERR ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 8097): ``

**rag_4k @ 8**

- #0 rag_4k in=3306 out=None fin=None ERR ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 8097): ``
- #1 rag_4k in=3195 out=None fin=None ERR ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 8097): ``
