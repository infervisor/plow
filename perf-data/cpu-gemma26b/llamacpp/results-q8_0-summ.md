# bench-api results

- server: `http://localhost:8097`  model: `/home/lava/llamacpp/gemma-4-26b-a4b-it-q8_0.gguf`
- time: 2026-09-06T07:10:58+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: /home/lava/llamacpp/gemma-4-26b-a4b-it-q8_0.gguf

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 12084/13646/15377 | 63/63/64 | 17.74/19.36 | 4.0 | 0.06 |
| summarize | 2 | 8 | 0 | 1119 | 64.0 | 24992/25417/27488 | 102/103/104 | 31.66/34.01 | 4.1 | 0.06 |
| summarize | 4 | 8 | 0 | 1178 | 64.0 | 54272/58716/59092 | 159/157/159 | 68.72/68.91 | 4.0 | 0.06 |
| summarize | 8 | 8 | 0 | 1133 | 64.0 | 56278/74128/86784 | 227/193/323 | 96.15/96.57 | 5.3 | 0.08 |

## Samples

**summarize @ 1**

- #0 summarize in=812 out=64 fin=length: `<|channel>thought<channel|>Here is a summary of the article:  * **Early Timekeep`
- #1 summarize in=1411 out=64 fin=length: `<|channel>thought<channel|>**Summary** The provided text explores the geological`

**summarize @ 2**

- #8 summarize in=1223 out=64 fin=length: `<|channel>thought<channel|>The article provides guidance on household budgeting,`
- #9 summarize in=1187 out=64 fin=length: `<|channel>thought<channel|>Here is a summary of the article:  * **The Role of th`

**summarize @ 4**

- #16 summarize in=1205 out=64 fin=length: `<|channel>thought<channel|>### Summary  The provided text consists of two distin`
- #17 summarize in=1083 out=64 fin=length: `<|channel>thought<channel|>**Summary** The invention of the printing press revol`

**summarize @ 8**

- #24 summarize in=1111 out=64 fin=length: `<|channel>thought<channel|>The article describes the history of lighthouses and `
- #25 summarize in=1366 out=64 fin=length: `<|channel>thought<channel|>Here is a summary of the article:  * **Evolution of T`
