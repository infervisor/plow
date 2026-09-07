# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-07T06:50:46+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 2333/2701/2918 | 26/26/26 | 4.33/4.55 | 16.1 | 0.25 |
| summarize | 8 | 8 | 0 | 1119 | 64.0 | 10487/11757/16759 | 266/288/355 | 27.38/27.77 | 18.3 | 0.29 |
| chat_long | 1 | 8 | 0 | 398 | 64.0 | 960/1000/1259 | 26/26/27 | 2.64/2.92 | 24.5 | 0.38 |
| chat_long | 8 | 8 | 0 | 442 | 58.6 | 4246/4377/7037 | 205/215/235 | 16.15/16.52 | 28.2 | 0.48 |

## Samples

**summarize @ 1**

- #0 summarize in=812 out=64 fin=length: `ible summary in 3‑5 bullet points:  - **Early time‑keeping**: Sundials and water`
- #1 summarize in=1411 out=64 fin=length: `Gee, that’s a lot of detail!   **Two‑sentence summary**   Glaciers erode valleys`

**summarize @ 8**

- #8 summarize in=1223 out=64 fin=length: `ina: The article explains household budgeting: track income/expenses, categorize`
- #9 summarize in=1187 out=64 fin=length: ` dole? The user wants a summary of the article in 3-5 bullet points. The article`

**chat_long @ 1**

- #0 chat_long in=512 out=64 fin=length: ` Cahill says the emergency fund should be kept in a **separate savings account t`
- #1 chat_long in=381 out=64 fin=length: `answerAn **escapement** is a mechanical device that controls the release of ener`

**chat_long @ 8**

- #8 chat_long in=502 out=64 fin=length: `tempts to answer the question: The difference between spring and neap tides is t`
- #9 chat_long in=472 out=64 fin=length: `temptation. A halo is a ring of light that appears around the sun or moon. It is`
