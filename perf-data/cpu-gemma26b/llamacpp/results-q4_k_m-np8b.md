# bench-api results

- server: `http://localhost:8097`  model: `/home/lava/llamacpp/gemma-4-26b-a4b-it-q4_k_m.gguf`
- time: 2026-09-06T14:24:33+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: /home/lava/llamacpp/gemma-4-26b-a4b-it-q4_k_m.gguf

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 4 | 8 | 0 | 365 | 64.0 | 15773/17222/17228 | 185/200/201 | 27.88/27.89 | 9.3 | 0.15 |
| code | 8 | 8 | 0 | 374 | 64.0 | 32080/32166/32169 | 183/184/184 | 43.69/43.69 | 11.7 | 0.18 |
| summarize | 4 | 8 | 0 | 1105 | 64.0 | 47619/48535/48995 | 172/198/201 | 61.47/61.63 | 4.4 | 0.07 |
| summarize | 8 | 8 | 0 | 1119 | 64.0 | 92495/94624/95109 | 257/223/320 | 108.70/108.83 | 4.7 | 0.07 |

## Samples

**code @ 4**

- #0 code in=463 out=64 fin=length: `<|channel>thought<channel|>### Explanation The issue is caused by the use of 'va`
- #1 code in=304 out=64 fin=length: `<|channel>thought<channel|>'''sql /* APPROACH: 1. Data Cleaning: Use COALESCE an`

**code @ 8**

- #8 code in=497 out=64 fin=length: `<|channel>thought<channel|>'''python import string from collections import Count`
- #9 code in=283 out=64 fin=length: `<|channel>thought<channel|>'''python import string  def top_words(text):     """`

**summarize @ 4**

- #0 summarize in=812 out=64 fin=length: `<|channel>thought<channel|>Here is a summary of the article:  * **Early Timekeep`
- #1 summarize in=1411 out=64 fin=length: `<|channel>thought<channel|>**Summary** The provided text explores the geological`

**summarize @ 8**

- #8 summarize in=1223 out=64 fin=length: `<|channel>thought<channel|>The article provides guidance on household budgeting,`
- #9 summarize in=1187 out=64 fin=length: `<|channel>thought<channel|>Here is a summary of the article:  * **The Role of th`
