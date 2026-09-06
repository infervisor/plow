# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T08:55:40+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 4 | 8 | 0 | 32 | 64.0 | 1354/1358/1778 | 169/167/176 | 12.25/12.43 | 21.1 | 0.33 |
| chat_short | 8 | 8 | 0 | 42 | 63.0 | 3201/3521/5098 | 232/238/254 | 17.74/18.12 | 27.3 | 0.43 |
| chat_long | 4 | 8 | 0 | 392 | 58.6 | 3718/3341/5748 | 247/266/295 | 17.40/20.65 | 12.7 | 0.22 |
| chat_long | 8 | 8 | 0 | 439 | 60.9 | 9279/9493/15420 | 360/411/433 | 31.40/31.86 | 15.1 | 0.25 |

## Samples

**chat_short @ 4**

- #0 chat_short in=25 out=64 fin=length: `At the end of summer, the queen’s egg‑laying rate drops and the colony’s need fo`
- #1 chat_short in=22 out=64 fin=length: `Below is a concise comparison of the two most common coffee processing methods—*`

**chat_short @ 8**

- #8 chat_short in=46 out=64 fin=length: `**The 50 % rule for fixed costs**  > *“In a household budget, no more than about`
- #9 chat_short in=46 out=64 fin=length: `**One‑sentence answer:**   The printing press made books thinner, lighter, and m`

**chat_long @ 4**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it:**   A separate savings account that takes a day to transfer `
- #1 chat_long in=383 out=64 fin=length: `**Answer**  An escapement is a mechanical device that controls the release of en`

**chat_long @ 8**

- #8 chat_long in=501 out=64 fin=length: `**Answer**  Spring tides are the high‑and‑low tides that occur when the Sun, Moo`
- #9 chat_long in=473 out=64 fin=length: `A halo is a bright ring that appears around the sun or moon, usually about 22° i`
