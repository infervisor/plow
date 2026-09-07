# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-07T06:54:52+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 2263/2603/2860 | 26/26/27 | 4.23/4.55 | 16.5 | 0.26 |

## Samples

**summarize @ 1**

- #0 summarize in=812 out=64 fin=length: `- **Early time‑keeping**:     * Sundials and water clocks were the first devices`
- #1 summarize in=1411 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and melt‑water action`
