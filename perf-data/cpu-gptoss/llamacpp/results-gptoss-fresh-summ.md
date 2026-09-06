# bench-api results

- server: `http://localhost:8098`  model: `gpt-oss-20b-mxfp4`
- time: 2026-09-06T08:34:52+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: gpt-oss-20b-mxfp4

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| summarize | 1 | 8 | 0 | 1111 | 64.0 | 10693/10352/16831 | 59/62/64 | 13.78/20.76 | 4.4 | 0.07 |
| summarize | 2 | 8 | 0 | 1124 | 64.0 | 23777/28452/30269 | 113/119/122 | 35.96/35.96 | 4.1 | 0.06 |
| summarize | 4 | 8 | 0 | 1188 | 64.0 | 57888/59511/60152 | 195/230/230 | 74.15/74.62 | 3.6 | 0.06 |
| summarize | 8 | 8 | 0 | 1136 | 64.0 | 73138/82130/82524 | 430/399/544 | 100.62/100.74 | 5.1 | 0.08 |

## Samples

**summarize @ 1**

- #0 summarize in=813 out=64 fin=length: `<|channel|>analysis<|message|>We need to summarize the article in 3-5 bullet poi`
- #1 summarize in=1434 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce a two-sentence summary followed`

**summarize @ 2**

- #8 summarize in=1215 out=64 fin=length: `<|channel|>analysis<|message|>We need to summarize the article in under 120 word`
- #9 summarize in=1190 out=64 fin=length: `<|channel|>analysis<|message|>We need to summarize the article in 3-5 bullet poi`

**summarize @ 4**

- #16 summarize in=1203 out=64 fin=length: `<|channel|>analysis<|message|>We need to summarize the article above. The articl`
- #17 summarize in=1094 out=64 fin=length: `<|channel|>analysis<|message|>We need to produce a two-sentence summary of the e`

**summarize @ 8**

- #24 summarize in=1117 out=64 fin=length: `<|channel|>analysis<|message|>We need to summarize the article in under 120 word`
- #25 summarize in=1363 out=64 fin=length: `<|channel|>analysis<|message|>We need to summarize the article in 3-5 bullet poi`
