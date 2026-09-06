# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T08:18:21+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| summarize | 1 | 8 | 0 | 1111 | 64.0 | 5374/6016/6945 | 47/47/47 | 9.07/9.92 | 7.7 | 0.12 |
| summarize | 2 | 8 | 0 | 1124 | 64.0 | 6185/5986/6782 | 142/159/167 | 15.00/17.28 | 8.4 | 0.13 |
| summarize | 4 | 8 | 0 | 1188 | 64.0 | 10883/7278/18328 | 304/356/378 | 32.04/40.74 | 8.5 | 0.13 |
| summarize | 8 | 8 | 0 | 1136 | 64.0 | 23158/24934/37301 | 507/563/720 | 55.22/55.94 | 9.1 | 0.14 |

## Samples

**summarize @ 1**

- #0 summarize in=813 out=64 fin=length: `- **Early time‑keeping**:     * Sundials used the sun’s shadow but were inaccura`
- #1 summarize in=1434 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and melt‑water action`

**summarize @ 2**

- #8 summarize in=1215 out=64 fin=length: `**Household budgeting**: A plan for future income, not a restriction. Start by t`
- #9 summarize in=1190 out=64 fin=length: `- **Lighthouse keepers**: Their daily routine involved maintaining the lamp, win`

**summarize @ 4**

- #16 summarize in=1203 out=64 fin=length: `**Summary**  The text contrasts two very different “colonies” – lighthouse keepe`
- #17 summarize in=1094 out=64 fin=length: `**Summary**   The invention of the printing press in the mid‑fifteenth century—c`

**summarize @ 8**

- #24 summarize in=1117 out=64 fin=length: `Lighthouses were identified by distinct light patterns—fixed, flashing every 5, `
- #25 summarize in=1363 out=64 fin=length: `- **Early time‑keeping**: Sundials and water clocks were the first devices, but `
