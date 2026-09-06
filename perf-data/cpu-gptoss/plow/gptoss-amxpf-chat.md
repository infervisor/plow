# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T08:11:28+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 32 | 64.0 | 576/575/621 | 45/45/45 | 3.40/3.43 | 18.9 | 0.29 |
| chat_short | 2 | 8 | 0 | 42 | 63.0 | 820/698/1269 | 73/75/77 | 5.45/5.51 | 23.4 | 0.37 |
| chat_short | 4 | 8 | 0 | 38 | 64.0 | 1358/1272/1886 | 119/121/129 | 8.91/9.52 | 28.6 | 0.45 |
| chat_short | 8 | 8 | 0 | 34 | 64.0 | 2880/3128/4661 | 200/205/221 | 15.58/15.94 | 31.8 | 0.50 |
| chat_long | 1 | 8 | 0 | 392 | 58.6 | 2035/2276/2612 | 45/46/46 | 4.74/5.50 | 12.5 | 0.21 |
| chat_long | 2 | 8 | 0 | 439 | 60.9 | 2954/3293/4415 | 89/101/117 | 8.71/8.92 | 14.4 | 0.24 |
| chat_long | 4 | 8 | 0 | 344 | 64.0 | 4001/3980/5248 | 156/162/176 | 14.55/15.37 | 18.4 | 0.29 |
| chat_long | 8 | 8 | 0 | 423 | 60.0 | 9320/9647/15578 | 326/350/408 | 29.09/29.63 | 16.1 | 0.27 |

## Samples

**chat_short @ 1**

- #0 chat_short in=25 out=64 fin=length: `At the end of summer, the queen’s egg‑laying rate drops and the colony’s need fo`
- #1 chat_short in=22 out=64 fin=length: `Below is a concise comparison of the two most common coffee processing methods—*`

**chat_short @ 2**

- #8 chat_short in=46 out=64 fin=length: `**The 50 % rule for fixed costs**  > *“In a household budget, no more than about`
- #9 chat_short in=46 out=64 fin=length: `**One‑sentence answer:**   The printing press made books thinner, lighter, and m`

**chat_short @ 4**

- #16 chat_short in=32 out=64 fin=length: `**Propolis** is a resin‑like substance that bees collect from tree buds, sap, an`
- #17 chat_short in=41 out=64 fin=length: `Trevithick’s 1804 locomotive was the first heavy steam engine to run on ordinary`

**chat_short @ 8**

- #24 chat_short in=33 out=64 fin=length: `**Short answer**  Because the railway timetable had to be the same everywhere.  `
- #25 chat_short in=26 out=64 fin=length: `At the end of summer the queen’s egg‑laying rate drops and the colony’s food sto`

**chat_long @ 1**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it:**   A separate savings account that takes a day to transfer `
- #1 chat_long in=383 out=64 fin=length: `**Answer**   An escapement is a mechanical device that controls the release of e`

**chat_long @ 2**

- #8 chat_long in=501 out=64 fin=length: `**Answer**  Spring tides are the high‑and‑low tides that occur when the Sun, Moo`
- #9 chat_long in=473 out=64 fin=length: `A halo is a bright ring that appears around the sun or moon, usually about 22° i`

**chat_long @ 4**

- #16 chat_long in=236 out=64 fin=length: `**Short answer**  The coffee is getting stale because the beans are sitting in t`
- #17 chat_long in=357 out=64 fin=length: `**Winter bees vs. summer bees – the big difference is what they’re built for.** `

**chat_long @ 8**

- #24 chat_long in=262 out=64 fin=length: `**Outline – “From Oil Lamps to Automation: The Life of a Lighthouse Keeper”**   `
- #25 chat_long in=581 out=32 fin=stop: `The passage does not mention Fresnel’s lens or how it changed lighthouse illumin`
