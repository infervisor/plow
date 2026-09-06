# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T09:38:14+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_long | 4 | 8 | 0 | 392 | 63.1 | 5978/5801/9816 | 188/208/223 | 17.51/22.93 | 14.0 | 0.22 |
| chat_long | 8 | 8 | 0 | 439 | 61.9 | 14310/13850/23309 | 369/396/514 | 36.68/37.97 | 12.8 | 0.21 |
| code | 4 | 8 | 0 | 344 | 64.0 | 4998/5093/6288 | 183/209/212 | 16.95/19.51 | 15.3 | 0.24 |
| code | 8 | 8 | 0 | 357 | 64.0 | 11544/12469/20161 | 309/334/421 | 31.16/32.34 | 15.6 | 0.24 |

## Samples

**chat_long @ 4**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it**   The passage says:   > “The fund should be kept somewhere `
- #1 chat_long in=383 out=64 fin=length: `**Answer**  An escapement is a mechanical device that transfers the energy of a `

**chat_long @ 8**

- #8 chat_long in=501 out=64 fin=length: `**Answer**  Spring tides are the highest and lowest tides that occur when the Su`
- #9 chat_long in=473 out=64 fin=length: `A halo is a bright ring that appears around the Sun or Moon. It is caused by the`

**code @ 4**

- #0 code in=441 out=64 fin=length: `'''javascript /**  * Schedules console.log of the given numbers after the specif`
- #1 code in=284 out=64 fin=length: `**Solution**  '''sql /*     1. Use an explicit LEFT JOIN so that customers with `

**code @ 8**

- #8 code in=471 out=64 fin=length: `'''python import string from collections import Counter  def top_words(text: str`
- #9 code in=257 out=64 fin=length: `'''python import string  def top_words(text):     """     Return the three most `
