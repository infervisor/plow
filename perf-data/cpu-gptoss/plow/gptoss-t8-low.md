# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T15:44:13+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_long | 1 | 8 | 0 | 392 | 62.8 | 978/1065/1242 | 24/24/24 | 2.43/2.80 | 25.4 | 0.40 |
| chat_long | 2 | 8 | 0 | 439 | 61.9 | 1438/1575/2160 | 60/65/75 | 5.34/5.49 | 23.8 | 0.38 |
| summarize | 1 | 8 | 0 | 1111 | 64.0 | 2577/2982/3277 | 25/25/26 | 4.57/4.88 | 15.3 | 0.24 |
| summarize | 2 | 8 | 0 | 1124 | 64.0 | 3014/2940/3239 | 87/97/98 | 8.29/9.41 | 15.1 | 0.24 |

## Samples

**chat_long @ 1**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it**   The passage says the emergency fund should be kept “somew`
- #1 chat_long in=383 out=64 fin=length: `**Answer**   An escapement is a mechanical device that transfers the energy of a`

**chat_long @ 2**

- #8 chat_long in=501 out=64 fin=length: `**Answer**  Spring tides are the highest and lowest tides that occur when the Su`
- #9 chat_long in=473 out=64 fin=length: `A halo is a bright ring that appears around the Sun or Moon. It is caused by the`

**summarize @ 1**

- #0 summarize in=813 out=64 fin=length: `- **Early time‑keeping**:     * Sundials and water clocks were the first devices`
- #1 summarize in=1434 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and melt‑water action`

**summarize @ 2**

- #8 summarize in=1215 out=64 fin=length: `**Household budgeting**: A budget plans money before it arrives, using envelopes`
- #9 summarize in=1190 out=64 fin=length: `- **Keeper duties and conditions** – A lighthouse keeper’s day revolved around m`
