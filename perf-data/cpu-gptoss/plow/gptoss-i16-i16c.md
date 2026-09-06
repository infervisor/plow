# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T19:07:13+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_long | 1 | 8 | 0 | 392 | 63.1 | 960/1072/1235 | 22/22/22 | 2.32/2.65 | 26.9 | 0.43 |
| chat_long | 2 | 8 | 0 | 439 | 61.9 | 1375/1463/2114 | 53/57/67 | 4.87/4.93 | 26.3 | 0.43 |
| summarize | 1 | 8 | 0 | 1111 | 64.0 | 2333/2678/3021 | 23/23/23 | 4.12/4.46 | 17.0 | 0.26 |
| summarize | 2 | 8 | 0 | 1124 | 64.0 | 2721/2647/2944 | 76/84/87 | 7.41/8.42 | 17.1 | 0.27 |

## Samples

**chat_long @ 1**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it**   The passage says the emergency fund should be kept “somew`
- #1 chat_long in=383 out=64 fin=length: `**Answer**  An escapement is a mechanical device that transfers the energy of a `

**chat_long @ 2**

- #8 chat_long in=501 out=64 fin=length: `**Answer**  Spring tides are the highest and lowest tides that occur when the Su`
- #9 chat_long in=473 out=64 fin=length: `A halo is a bright ring that appears around the Sun or Moon. It is caused by the`

**summarize @ 1**

- #0 summarize in=813 out=64 fin=length: `- **Early time‑keeping**:     * Sundials and water clocks were the first devices`
- #1 summarize in=1434 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and melt‑water action`

**summarize @ 2**

- #8 summarize in=1215 out=64 fin=length: `**Household budgeting**: A budget plans money before it arrives, using envelopes`
- #9 summarize in=1190 out=64 fin=length: `- **Keeper duties and conditions** – A lighthouse keeper’s day revolved around m`
