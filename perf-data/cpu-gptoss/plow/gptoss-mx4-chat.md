# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T09:25:42+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 32 | 64.0 | 602/586/633 | 33/31/32 | 2.57/2.59 | 24.0 | 0.37 |
| chat_short | 2 | 8 | 0 | 42 | 62.9 | 829/717/1303 | 61/62/73 | 4.61/5.29 | 27.1 | 0.43 |
| chat_short | 4 | 8 | 0 | 38 | 64.0 | 1369/1303/1899 | 109/110/119 | 8.23/8.87 | 30.9 | 0.48 |
| chat_short | 8 | 8 | 0 | 34 | 63.2 | 2876/3144/4612 | 174/180/195 | 13.90/14.18 | 35.3 | 0.56 |
| chat_long | 1 | 8 | 0 | 392 | 63.1 | 2070/2323/2639 | 32/32/32 | 4.23/4.64 | 15.6 | 0.25 |
| chat_long | 2 | 8 | 0 | 439 | 61.9 | 2969/3315/4445 | 75/88/105 | 7.92/8.12 | 16.1 | 0.26 |
| chat_long | 4 | 8 | 0 | 344 | 64.0 | 3837/3998/4791 | 146/156/171 | 13.71/14.60 | 19.6 | 0.31 |
| chat_long | 8 | 8 | 0 | 423 | 61.0 | 10469/11602/16797 | 285/291/358 | 27.40/27.72 | 17.5 | 0.29 |

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

- #24 chat_short in=33 out=64 fin=length: `**Short answer**  When a railway line was built, the railway company had to deci`
- #25 chat_short in=26 out=58 fin=stop: `At the end of the season the colony’s brood production slows, so the queen stops`

**chat_long @ 1**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it**   The passage says:   > “The fund should be kept somewhere `
- #1 chat_long in=383 out=64 fin=length: `**Answer**  An escapement is a mechanical device that transfers the energy of a `

**chat_long @ 2**

- #8 chat_long in=501 out=64 fin=length: `**Answer**  Spring tides are the highest and lowest tides that occur when the Su`
- #9 chat_long in=473 out=64 fin=length: `A halo is a ring of light that appears around the sun or moon. It is caused by t`

**chat_long @ 4**

- #16 chat_long in=236 out=64 fin=length: `**Short answer**  The coffee is losing flavor because the beans are sitting in t`
- #17 chat_long in=357 out=64 fin=length: `**Winter bees vs. summer bees – the big difference is what they’re built for**  `

**chat_long @ 8**

- #24 chat_long in=262 out=64 fin=length: `**Title: “From Oil Lamps to Automation: The Life of a Lighthouse Keeper”**   *(≈`
- #25 chat_long in=581 out=40 fin=stop: `The passage does not mention Fresnel’s lens or how it changed lighthouse illumin`
