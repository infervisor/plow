# bench-api results

- server: `http://localhost:8096`  model: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- time: 2026-09-07T06:35:25+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 34 | 64.0 | 872/878/882 | 85/85/87 | 6.16/6.34 | 10.3 | 0.16 |
| chat_short | 2 | 8 | 0 | 45 | 64.0 | 1400/1821/1844 | 129/134/135 | 9.50/9.69 | 13.4 | 0.21 |
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 6634/7994/8047 | 99/99/100 | 14.21/14.27 | 5.0 | 0.08 |
| summarize | 2 | 8 | 0 | 1119 | 64.0 | 8073/8133/8462 | 231/260/264 | 22.21/24.91 | 5.6 | 0.09 |

## Samples

**chat_short @ 1**

- #0 chat_short in=27 out=64 fin=length: `Bees kick out drones (males) at the end of summer primarily to **conserve resour`
- #1 chat_short in=21 out=64 fin=length: `The difference between **washed** (wet) and **natural** (dry) processing refers `

**chat_short @ 2**

- #8 chat_short in=49 out=64 fin=length: `To help you prepare for your talk, here is a breakdown of the 50% rule, how it w`
- #9 chat_short in=50 out=64 fin=length: `Here is a way to explain it to your child:  **The one-sentence answer:** The pri`

**summarize @ 1**

- #0 summarize in=812 out=64 fin=length: `Here is a summary of the article:  *   **Evolution of Timekeeping:** Human metho`
- #1 summarize in=1411 out=64 fin=length: `**Summary** The first text explains the geological processes by which glaciers s`

**summarize @ 2**

- #8 summarize in=1223 out=64 fin=length: `This text provides a primer on household budgeting and a summary of coffee produ`
- #9 summarize in=1187 out=64 fin=length: `Here is a summary of the text:  *   **The Role and Responsibility of Keepers:** `
