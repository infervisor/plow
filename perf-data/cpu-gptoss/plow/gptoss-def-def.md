# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T16:02:12+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_long | 1 | 8 | 0 | 392 | 62.8 | 1093/1205/1394 | 29/29/30 | 2.93/3.24 | 21.6 | 0.34 |
| chat_long | 4 | 8 | 0 | 439 | 61.9 | 2433/2207/3318 | 118/121/133 | 9.69/10.84 | 25.2 | 0.41 |
| chat_long | 8 | 8 | 0 | 344 | 64.0 | 4331/4798/7021 | 186/194/224 | 16.08/16.33 | 31.0 | 0.48 |
| summarize | 1 | 8 | 0 | 1111 | 64.0 | 2801/3278/3577 | 30/30/30 | 5.19/5.47 | 13.6 | 0.21 |
| summarize | 4 | 8 | 0 | 1124 | 64.0 | 5545/5308/8465 | 173/203/212 | 16.48/21.26 | 15.5 | 0.24 |
| summarize | 8 | 8 | 0 | 1188 | 64.0 | 14242/16279/21528 | 301/316/415 | 33.43/33.80 | 15.0 | 0.23 |

## Samples

**chat_long @ 1**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it**   The passage says the emergency fund should be kept “somew`
- #1 chat_long in=383 out=64 fin=length: `**Answer**   An escapement is a mechanical device that transfers the energy of a`

**chat_long @ 4**

- #8 chat_long in=501 out=64 fin=length: `**Answer**  Spring tides are the highest and lowest tides that occur when the Su`
- #9 chat_long in=473 out=64 fin=length: `A halo is a bright ring that appears around the Sun or Moon. It is caused by the`

**chat_long @ 8**

- #16 chat_long in=236 out=64 fin=length: `**Short answer**  The coffee is going stale because the beans are sitting in the`
- #17 chat_long in=357 out=64 fin=length: `**Winter bees vs. summer bees – the big difference is what they’re built for**  `

**summarize @ 1**

- #0 summarize in=813 out=64 fin=length: `- **Early time‑keeping**:     * Sundials and water clocks were the first devices`
- #1 summarize in=1434 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and melt‑water action`

**summarize @ 4**

- #8 summarize in=1215 out=64 fin=length: `**Household budgeting**: A budget plans money before it arrives, using envelopes`
- #9 summarize in=1190 out=64 fin=length: `- **Keeper duties and conditions** – A lighthouse keeper’s day revolved around m`

**summarize @ 8**

- #16 summarize in=1203 out=64 fin=length: `**Summary**  The article contrasts the lives of lighthouse keepers with the inne`
- #17 summarize in=1094 out=64 fin=length: `**Summary**   The printing press revolutionized Europe by making books cheaper, `
