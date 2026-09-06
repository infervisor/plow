# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T09:31:46+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| summarize | 1 | 8 | 0 | 1111 | 64.0 | 5437/6135/7019 | 33/33/34 | 8.22/9.12 | 8.5 | 0.13 |
| summarize | 2 | 8 | 0 | 1124 | 64.0 | 6200/6016/6839 | 128/145/153 | 14.19/16.46 | 9.0 | 0.14 |
| summarize | 4 | 8 | 0 | 1188 | 64.0 | 11179/7209/20279 | 283/324/361 | 31.14/39.72 | 8.8 | 0.14 |
| summarize | 8 | 8 | 0 | 1136 | 64.0 | 24319/27158/37870 | 466/504/696 | 53.90/54.32 | 9.4 | 0.15 |

## Samples

**summarize @ 1**

- #0 summarize in=813 out=64 fin=length: `- **Early time‑keeping**:     * Sundials and water clocks were the first devices`
- #1 summarize in=1434 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and melt‑water action`

**summarize @ 2**

- #8 summarize in=1215 out=64 fin=length: `**Household budgeting**: A household budget plans money before it arrives. Start`
- #9 summarize in=1190 out=64 fin=length: `- **Keeper duties and conditions** – A lighthouse keeper’s routine involved main`

**summarize @ 4**

- #16 summarize in=1203 out=64 fin=length: `**Summary**  The text contrasts the lives of lighthouse keepers with the working`
- #17 summarize in=1094 out=64 fin=length: `**Summary**   The printing press revolutionized Europe by making books cheaper, `

**summarize @ 8**

- #24 summarize in=1117 out=64 fin=length: `Lighthouses were identified by unique light patterns—fixed, flashing at 5, 10, o`
- #25 summarize in=1363 out=64 fin=length: `- **Ancient & early time‑keeping** – Sundials and water clocks were the first to`
