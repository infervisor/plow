# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T09:12:17+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 4 | 8 | 0 | 32 | 64.0 | 1305/1265/1777 | 109/110/116 | 8.29/8.63 | 31.1 | 0.49 |
| chat_short | 8 | 8 | 0 | 42 | 63.0 | 3095/3348/4942 | 178/186/200 | 14.38/14.69 | 34.0 | 0.54 |
| chat_long | 4 | 8 | 0 | 392 | 58.6 | 4273/4491/6033 | 177/195/225 | 14.00/16.86 | 16.2 | 0.28 |
| chat_long | 8 | 8 | 0 | 439 | 60.9 | 9539/9755/15353 | 294/307/371 | 27.57/27.72 | 17.3 | 0.28 |

## Samples

**chat_short @ 4**

- #0 chat_short in=25 out=64 fin=length: `At the end of summer, the queen’s egg‑laying rate drops and the colony’s needs s`
- #1 chat_short in=22 out=64 fin=length: `Below is a concise comparison of the two most common coffee processing methods—*`

**chat_short @ 8**

- #8 chat_short in=46 out=64 fin=length: `**The 50 % rule for fixed costs**  > *“In a household budget, no more than about`
- #9 chat_short in=46 out=64 fin=length: `**One‑sentence answer:**   The printing press made books thinner, lighter, and m`

**chat_long @ 4**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it:**   A separate savings account that takes a day to transfer `
- #1 chat_long in=383 out=64 fin=length: `**Answer**   An escapement is a mechanical device that controls the release of e`

**chat_long @ 8**

- #8 chat_long in=501 out=64 fin=length: `**Answer**  Spring tides are the high‑and‑low tides that occur when the Sun, Moo`
- #9 chat_long in=473 out=64 fin=length: `A halo is a bright ring that appears around the sun or moon, usually about 22° i`
