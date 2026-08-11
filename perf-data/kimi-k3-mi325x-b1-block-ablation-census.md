# Kimi-K3 B1 five-layer body-ablation census

Date: 2026-08-11. Hardware: 8x MI325X (`gfx942`, TP8). Toolchain: repository Nix ROCm 7.14.0.

## Method

`scripts/k3_block_sweep.sh` emitted the smallest K3 span containing both mixer types: four KDA
layers plus one MLA layer. `PLOW_K3_ABLATE=<opcode>` replaced only the named device body with
`Nop`; packet order, waits, successors, dispatch widths, and counter storage were unchanged.
Each arm ran 128 B1 decode steps at context 5 through the production interpreter and TP runtime.
Tokens are intentionally invalid. This is a body-cost screen, not a correctness or serving gate.

## Result

The two controls were 4.114 and 4.125 ms/token.

| ablated body | opcode | packets in five layers | ms/token | delta vs 4.120 ms midpoint |
|---|---:|---:|---:|---:|
| KDA fused QKVG | 108 | 4 | 4.009 | -0.111 |
| KDA Conv3 | 111 | 4 | 4.115 | -0.005 |
| KDA state step | 112 | 4 | 4.108 | -0.012 |
| KDA gated norm | 103 | 4 | 4.095 | -0.025 |
| AttnRes | 104 | 11 | 3.999 | -0.121 |
| shared GemvGlu | 19 | 4 | 4.089 | -0.031 |
| MoeCombine | 43 | 4 | 4.102 | -0.018 |
| MLA merge | 57 | 1 | 4.112 | -0.008 |
| all ordinary BF16 Gemv | 10 | 43 | 3.351 | -0.769 |

Router ablation is not attributable: it changes the downstream selected-expert population. Grouped
expert ablation removes the packet that proves expert-table consumption and is correctly refused by
the loader. Removing every collective makes the packet look single-GPU and is also correctly
refused. Removing the only FP8 MLA decode packet changes object-variant selection and is refused
against the remaining FP8 cache-write packet.

## Decision

KDA Conv3/state arithmetic is hidden; another local body optimization there cannot recover the
remaining B1 gap. Even deleting all 43 ordinary GEMV bodies in this five-layer replay saves only
0.769 ms while retaining their packet gates. The next candidate must cross a producer/consumer
boundary or delete protocol depth. Standalone kernel speed alone is insufficient.

Raw logs:

- `/tmp/k3-block-ablate-census.txt`
- `/tmp/k3-block-ablate-census2.txt`

## Reproduction

```bash
nix develop --command env \
  PLOW_K3_CTX=5 PLOW_K3_STEPS=128 PLOW_K3_OUT=/tmp/k3blk-census-108 \
  scripts/k3_block_sweep.sh PLOW_K3_ABLATE=108
```

Repeat with the opcode column above; omit `PLOW_K3_ABLATE` for the matched control.
