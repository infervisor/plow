# bench-api results

- server: `http://localhost:8094`  model: `gemma-4-12b-it`
- time: 2026-09-06T02:08:09+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gemma-4-12b-it

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| rag_4k | 1 | 4 | 0 | 3094 | 57.0 | 8569/8771/9124 | 448/449/449 | 33.01/37.33 | 1.7 | 0.03 |
| rag_4k | 2 | 4 | 0 | 3094 | 56.0 | 11533/10658/17947 | 541/597/665 | 42.30/52.57 | 2.7 | 0.05 |

## Samples

**rag_4k @ 1**

- #0 rag_4k in=3306 out=64 fin=length: `The tides affect the length of the day and the distance to the Moon through fric`
- #1 rag_4k in=3195 out=55 fin=stop: `During the roasting process, the first crack occurs when the internal pressure o`

**rag_4k @ 2**

- #0 rag_4k in=3306 out=64 fin=length: `The tides affect the length of the day and the distance to the Moon through fric`
- #1 rag_4k in=3195 out=55 fin=stop: `During the roasting process, the first crack occurs when the internal pressure o`
