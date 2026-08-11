# Kimi K3 MI325X decode ladder: 131.16 aggregate tok/s

Date: 2026-08-11

## Result

Eight MI325X GPUs, TP8, native checkpoint MXFP4, FP8 KV, compact exact TP
counter audit, and the public OpenAI-compatible `plowrt serve` path:

| workload | completed | output tokens | output tok/s | p50 TPOT |
|---|---:|---:|---:|---:|
| C32, 32 prompts, input 32, output 2048 | 32/32 | 65,536 | **131.162** | 238.49 ms |
| pre-soak reuse, output 128 | 32/32 | 4,096 | 88.124 | 275.41 ms |
| post-soak reuse 1, output 128 | 32/32 | 4,096 | 88.039 | 275.64 ms |
| post-soak reuse 2, output 128 | 32/32 | 4,096 | 88.118 | 275.26 ms |

Every request had an empty error, the requested output length, and no in-band
`[error:` marker. All three short-run `generated_texts` arrays have the same
SHA256:

```text
a8f19bfa73d0dfd31cf161e1ac82c9d52146785b0fe99e610a052b0181a000a7
```

Every short output is also an exact prefix of the corresponding long output.
This specifically gates recurrent KDA state, convolution windows, MLA KV, slot
reuse, ladder transitions, and all-rank agreement after a 2,048-step soak.

## Build and asset

```text
binary  /tmp/k3-ladder-slo-plowrt
sha256  1ebb5ee5d7ee8a11cfc352c8cf28d684d457ac0b5a6a7e90530822cd76f45ad2

packet  /home/lava/models/k3_mi325x_ladder_router/model.pkt
sha256  f1f260d69105dffab3a7bd7f256d5fcbc215609f44c033c2cbb025949d14c709
rungs   1,2,4,8,16,32
prefill 128,512,1024,2048,4096,8192
```

All decode rungs use independent-sequence KDA addressing. Runtime selection is
the narrowest rung covering the highest occupied slot. Admission uses a
separate arbitrary-width controller with backlog/SLO widening and hysteretic
narrowing; existing high slots are never moved.

The B32 router TopK launch is 32 workgroups instead of 304 because one
workgroup owns one sequence. The token selection algorithm is unchanged.

## Commands

Every command ran through the Nix ROCm 7.14 environment. The server held an
exclusive eight-GPU lease for the complete sequence.

```bash
nix develop --command env \
  PLOW_L2_PLACE_DISPATCH=1 \
  PLOW_TP_AUDIT_COMPACT=1 \
  PLOW_CTR_DBUF=1 \
  PLOW_DSTEP_LOG=1 \
  PLOW_DSTEP_EVERY=64 \
  perf-data/harness/gpulease -n 8 k3-ladder-slo-serve \
  /tmp/k3-ladder-slo-plowrt serve \
  --assets /home/lava/models/k3_mi325x_ladder_router --port 8018
```

The decisive client invocation was:

```bash
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8018 \
  --endpoint /v1/chat/completions \
  --model k3_farm --served-model-name k3_farm \
  --tokenizer /home/lava/models/k3_tokz --tokenizer-mode hf \
  --dataset-name random \
  --random-input-len 32 --random-output-len 2048 \
  --random-range-ratio 0 \
  --request-rate inf --max-concurrency 32 \
  --num-prompts 32 --num-warmups 1 \
  --ignore-eos --temperature 0 --seed 0 \
  --save-result --save-detailed \
  --result-dir /tmp/k3-ladder-slo-c32-out2048 \
  --result-filename seed0.json
```

Raw results:

```text
/tmp/k3-ladder-slo-c32-out2048/seed0.json
/tmp/k3-ladder-slo-c32-out128/seed0.json
/tmp/k3-ladder-slo-postsoak-c32-out128/seed0.json
/tmp/k3-ladder-slo-postsoak2-c32-out128/seed0.json
```

## Scope

The 131.16 figure is aggregate served throughput across 32 streams. It is not
single-stream speed and is not comparable to Artificial Analysis' headline
single-request output speed. Short bursts still pay serial per-slot recurrent
state initialization and gradual prefill admission; dedicated low-rung objects
and admission-state clear pipelining remain latency work.
