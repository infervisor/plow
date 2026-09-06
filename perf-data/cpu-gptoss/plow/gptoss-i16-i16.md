# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T19:02:51+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_long | 4 | 8 | 0 | 392 | 63.1 | 2115/2093/2816 | 84/89/96 | 7.29/8.40 | 34.2 | 0.54 |
| chat_long | 8 | 8 | 0 | 439 | 61.9 | 4329/4450/7178 | 171/194/204 | 14.94/15.14 | 32.5 | 0.53 |
| summarize | 4 | 8 | 0 | 1111 | 64.0 | 4464/4317/6010 | 139/151/172 | 13.86/15.27 | 19.2 | 0.30 |
| summarize | 8 | 8 | 0 | 1124 | 64.0 | 10752/11855/16836 | 246/271/339 | 26.34/26.65 | 19.1 | 0.30 |

## Samples

**chat_long @ 4**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it**   The passage says the emergency fund should be kept “somew`
- #1 chat_long in=383 out=64 fin=length: `**Answer**  An escapement is a mechanical device that transfers the energy of a `

**chat_long @ 8**

- #8 chat_long in=501 out=64 fin=length: `**Answer**  Spring tides are the highest and lowest tides that occur when the Su`
- #9 chat_long in=473 out=64 fin=length: `A halo is a bright ring that appears around the Sun or Moon. It is caused by the`

**summarize @ 4**

- #0 summarize in=813 out=64 fin=length: `- **Early time‑keeping**:     * Sundials and water clocks were the first devices`
- #1 summarize in=1434 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and melt‑water action`

**summarize @ 8**

- #8 summarize in=1215 out=64 fin=length: `**Household budgeting**: A budget plans money before it arrives, using envelopes`
- #9 summarize in=1190 out=64 fin=length: `- **Keeper duties and conditions** – A lighthouse keeper’s day revolved around m`
