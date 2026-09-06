# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T11:23:04+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 32 | 64.0 | 553/549/587 | 29/29/29 | 2.36/2.38 | 27.2 | 0.43 |
| chat_short | 2 | 8 | 0 | 42 | 63.4 | 773/659/1198 | 54/57/59 | 4.23/4.30 | 30.0 | 0.47 |
| chat_short | 4 | 8 | 0 | 38 | 64.0 | 1283/1204/1776 | 97/98/105 | 7.43/7.96 | 34.3 | 0.54 |
| chat_short | 8 | 8 | 0 | 34 | 63.9 | 2669/2917/4262 | 155/160/175 | 12.49/12.73 | 39.8 | 0.62 |
| chat_long | 1 | 8 | 0 | 392 | 62.8 | 1928/2196/2437 | 28/28/28 | 3.87/4.26 | 17.1 | 0.27 |
| chat_long | 2 | 8 | 0 | 439 | 61.9 | 2779/3129/4107 | 67/79/95 | 7.23/7.40 | 17.8 | 0.29 |
| chat_long | 4 | 8 | 0 | 344 | 64.0 | 3506/3751/4392 | 128/137/150 | 12.31/13.05 | 22.0 | 0.34 |
| chat_long | 8 | 8 | 0 | 423 | 61.0 | 7994/8198/13994 | 261/292/339 | 24.22/24.47 | 19.8 | 0.32 |

## Samples

**chat_short @ 1**

- #0 chat_short in=25 out=64 fin=length: `At the end of the season the colony’s population is already high, and the queen’`
- #1 chat_short in=22 out=64 fin=length: `Below is a concise comparison of the two most common coffee processing methods—*`

**chat_short @ 2**

- #8 chat_short in=46 out=64 fin=length: `**The 50 % rule for fixed costs**  | What it says | How it works | |------------`
- #9 chat_short in=46 out=64 fin=length: `**One‑sentence answer:**   The printing press made books thinner, more uniform, `

**chat_short @ 4**

- #16 chat_short in=32 out=64 fin=length: `**Propolis** is a resin‑like substance that bees collect from trees and plants. `
- #17 chat_short in=41 out=64 fin=length: `Trevithick’s 1804 locomotive was the first to use a steam engine on iron rails, `

**chat_short @ 8**

- #24 chat_short in=33 out=64 fin=length: `**Short answer**  Because the railway timetable had to be the same everywhere.  `
- #25 chat_short in=26 out=63 fin=stop: `At the end of the season the colony’s brood production slows, so the queen stops`

**chat_long @ 1**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it**   The passage says the emergency fund should be kept “somew`
- #1 chat_long in=383 out=64 fin=length: `**Answer**   An escapement is a mechanical device that transfers the energy of a`

**chat_long @ 2**

- #8 chat_long in=501 out=64 fin=length: `**Answer**  Spring tides are the highest and lowest tides that occur when the Su`
- #9 chat_long in=473 out=64 fin=length: `A halo is a bright ring that appears around the Sun or Moon. It is caused by the`

**chat_long @ 4**

- #16 chat_long in=236 out=64 fin=length: `**Short answer**  The coffee is going stale because the beans are sitting in the`
- #17 chat_long in=357 out=64 fin=length: `**Winter bees vs. summer bees – the big difference is what they’re built for**  `

**chat_long @ 8**

- #24 chat_long in=262 out=64 fin=length: `**Title: “From Oil Lamps to Automation: The Life of a Lighthouse Keeper”**   *(≈`
- #25 chat_long in=581 out=40 fin=stop: `The passage does not mention Fresnel’s lens or how it changed lighthouse illumin`
