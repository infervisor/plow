# Kimi-K3 MI325X B8 served-throughput gate

Date: 2026-08-11. Branch: `kimi-k3-mi325x`. Runtime source:
`c994c76a`. Hardware: 8 leased MI325X GPUs, gfx942, 304 CUs each.
Toolchain: flake-pinned ROCm 7.14.0. Client: flake-pinned vLLM 0.26.0
`bench serve`; vLLM is the HTTP benchmark client, not the serving engine.

## Artifact and correctness

The B8 TP8 artifact uses the real Kimi-K3 checkpoint, native MXFP4 weights,
FP8 KV, the 128-workgroup GEMV cap, and L2-placed prefill. Batched decode
contains the grouped MXFP4 A4W4 arm; the runtime refuses an object missing that
capability marker.

- OpenAI coherence gate: `The capital of France is **Paris**.`
- 16k and 32k prompt requests ran concurrently and completed successfully.
  Server usage reported exactly 16,000/32,000 prompt tokens and 8/8 completion
  tokens; both ended by the requested length.
- The runtime bound 276 carried recurrent-state tensors per rank and allocated
  independent state/KV geometry for all eight slots.
- Ragged prefill selected the fewest-launch bucket cover and ran the final chunk
  at its real row count.

## Served result

Command shape: random input length 32 before chat templating, 128 forced output
tokens, 32 requests, concurrency 8, 8 warmups, greedy, no prefix cache. vLLM
0.26.0 used the server's usage response: each run contains 1,707 actual input
tokens and exactly 4,096 generated tokens.

| run | duration (s) | output tok/s | median TTFT (ms) | median TPOT (ms) | P99 ITL (ms) | requests | generated |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 80.814 | 50.685 | 826.74 | 148.31 | 480.58 | 32/32 | 4096/4096 |
| 2 | 80.942 | 50.604 | 827.34 | 148.48 | 481.34 | 32/32 | 4096/4096 |
| 3 | 80.650 | 50.787 | 827.02 | 148.28 | 480.38 | 32/32 | 4096/4096 |

Median output throughput is **50.685 tok/s**; range is 50.604--50.787 tok/s.
This clears the 50 aggregate tok/s diagnostic target. It does not claim 50
tok/s for one stream; the B1 latency result remains 18.4 tok/s.

Post-run audit found that the batched TP completion path performed full-vector
rank agreement but bypassed the configured compact cross-GPU counter audit.
These three measurements therefore are not production-adoption numbers. The
runtime now factors drain+audit into the scalar and batched paths; its GPU gate
reports one nonzero compact audit per step (~1.64 ms). The corrected serving
result must replace this table before B8 is adopted.

```bash
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:8018 \
  --endpoint /v1/chat/completions \
  --model k3_farm --served-model-name k3_farm \
  --tokenizer /home/lava/models/k3_tokz --tokenizer-mode hf \
  --dataset-name random --random-input-len 32 --random-output-len 128 \
  --max-concurrency 8 --num-prompts 32 --num-warmups 8 \
  --ignore-eos --temperature 0 --ready-check-timeout-sec 10
```

The server was the only process holding the eight-GPU `gpulease` for all three
runs. The client ran from Nix and used no Docker, system ROCm, or custom metric
implementation.

## Scope and remaining adoption gates

The result is a `plowrt serve` capacity result, not an MI325X vLLM-engine
comparison. The official K3 recipe image remains a separate same-box startup,
correctness, and interleaved showdown gate. B8 also needs the existing GSM8K
and ragged multi-prompt state-reuse battery before replacing the adopted B1
asset.
