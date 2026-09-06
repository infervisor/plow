# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-06T00:48:14+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| rag_4k | 1 | 4 | 0 | 3094 | 57.0 | 24697/25946/26762 | 255/251/269 | 39.71/43.71 | 1.5 | 0.03 |
| rag_4k | 2 | 4 | 0 | 3094 | 57.0 | 31535/26060/52743 | 605/748/758 | 67.24/92.59 | 1.7 | 0.03 |

## Samples

**rag_4k @ 1**

- #0 rag_4k in=3306 out=64 fin=length: `The tides affect the length of the day and the distance to the Moon through fric`
- #1 rag_4k in=3195 out=55 fin=stop: `During the roasting process, the first crack occurs when the internal pressure o`

**rag_4k @ 2**

- #0 rag_4k in=3306 out=64 fin=length: `The tides affect the length of the day and the distance to the Moon through fric`
- #1 rag_4k in=3195 out=55 fin=stop: `During the roasting process, the first crack occurs when the internal pressure o`
