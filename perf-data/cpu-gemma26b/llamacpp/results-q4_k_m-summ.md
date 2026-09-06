# bench-api results

- server: `http://localhost:8097`  model: `/home/lava/llamacpp/gemma-4-26b-a4b-it-q4_k_m.gguf`
- time: 2026-09-06T07:30:13+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: /home/lava/llamacpp/gemma-4-26b-a4b-it-q4_k_m.gguf

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| summarize | 1 | 8 | 0 | 1105 | 64.0 | 11237/12650/14341 | 56/56/56 | 16.24/17.88 | 4.3 | 0.07 |
| summarize | 2 | 8 | 0 | 1119 | 64.0 | 23241/23613/25631 | 105/113/116 | 30.41/32.95 | 4.3 | 0.07 |
| summarize | 4 | 8 | 0 | 1178 | 64.0 | 50453/54424/54782 | 146/144/146 | 63.65/63.76 | 4.3 | 0.07 |
| summarize | 8 | 8 | 0 | 1133 | 64.0 | 48027/71653/77715 | 190/148/236 | 86.54/86.80 | 5.9 | 0.09 |

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

- #24 summarize in=1111 out=64 fin=length: `<|channel>thought<channel|>The article covers two distinct subjects. First, it d`
- #25 summarize in=1366 out=64 fin=length: `<|channel>thought<channel|>Here is a summary of the article:  * **Evolution of T`
