# Kimi-K3 MI325X Stage 7 status

Date: 2026-08-10. Branch: `kimi-k3-mi325x` from `6bd17655` plus the harness
fix committed with this record. Hardware: 8 leased MI325X GPUs, gfx942, 304 CUs
each. Toolchain: flake-pinned ROCm 7.14.0.

Scope: real Kimi-K3 checkpoint, native MXFP4 weights, FP8 KV, TP8, batch 1.

## Honesty banner

Ran: OpenAI serve coherence gate and GSM8K accuracy on the adopted wg128 packet
with compact exact TP auditing.

Not run: same-session vLLM/SGLang comparator, concurrency ladder, context
ladder, or peak-throughput campaign. The local Nix vLLM is CPU-only and current
upstream K3 ROCm support does not provide a sound gfx942 MXFP4 arm. No stored or
cross-hardware result is reported as a comparator.

## Accuracy gate

GSM8K test split, first 200 questions, 8-shot chain-of-thought, greedy
temperature 0, concurrency 1, maximum 320 generated tokens. Exact match uses
the final parsed number.

| metric | result |
|---|---:|
| exact match | **197/200 = 0.9850** |
| request errors | 0 |
| median latency/question | 6.29 s |
| mean latency/question | 6.66 s |
| total measured wall time | 1332 s |

The pre-run coherence prompt passed. The repository `gpulease` held all eight
GPUs for the complete server lifetime and accuracy run.

```bash
nix develop --command env \
  PLOW_L2_PLACE_DISPATCH=1 PLOW_TP_AUDIT_COMPACT=1 \
  N=200 SHOTS=8 MAXTOK=320 CONC=1 \
  PLOWRT_BIN=/home/lava/plow/target/release/plowrt \
  perf-data/harness/gpulease -n 8 k3-mi325x-gsm8k \
  scripts/bench_gsm8k.sh /home/lava/models/k3_mi325x 8196 auto 1800
```

## Gate decision

The accuracy requirement passes. Stage 7 remains blocked on a same-box sound
gfx942 comparator and on the unrun capacity/context ladders. Therefore this
record makes no vLLM performance claim.
