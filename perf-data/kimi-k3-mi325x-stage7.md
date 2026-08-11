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
  perf-data/tools/gpulease -n 8 k3-mi325x-gsm8k \
  scripts/bench_gsm8k.sh /home/lava/models/k3_mi325x 8196 auto 1800
```

## Gate decision

The accuracy requirement passes. Stage 7 remains blocked on a same-box sound
gfx942 comparator and on the unrun capacity/context ladders. Therefore this
record makes no vLLM performance claim.

## 2026-08-11 capacity follow-up

The B8 capacity and concurrent 16k/32k context gates now pass. Three identical
vLLM 0.26.0 `bench serve` runs against `plowrt serve` measured
50.604--50.787 aggregate output tok/s with 32/32 requests and 4,096/4,096
generated tokens per run. See `perf-data/kimi-k3-mi325x-b8-serve.md`.

This removes the capacity/context blocker. Stage 7 still requires B8 GSM8K,
state-reuse/ragged correctness, and the same-box vLLM serving-engine comparator.

Audit correction: the original B8 path performed full-vector rank agreement
but skipped the requested compact cross-GPU counter audit. Commit `45851e1e`
fixed the runtime path. Three corrected serving runs measure
50.008--50.116 output tok/s with a 50.082 tok/s median and exact output/error
gates; see `perf-data/kimi-k3-mi325x-b8-serve.md`.

The official K3 recipe now exists at
`https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=mi325x`. It requires a
K3-enabled vLLM 0.27.0+ nightly engine. The Nix benchmark client is now pinned
to vLLM 0.27.0 as well. The published AMD image is
`vllm/vllm-openai-rocm:kimi-k3`, currently
`sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b`
(14.53 GB compressed). The AMD profile enables AITER, SiTUv2 A8W4, and
full-decode graphs. This host has no Docker/Podman binary or runtime socket, so
the engine arm remains blocked until a Nix-managed OCI runtime is added. Do not
substitute the client-only `.#vllm` shell as an engine result.

## Merge disposition

This is a known-open comparison gate, not a merge blocker. The Plow accuracy and
capacity results above remain valid without a cross-engine number. Re-run the
same-session comparator when the bring-up container provides the pinned OCI
engine; until then, report the comparator as unavailable rather than borrowing a
result from another architecture or session.
