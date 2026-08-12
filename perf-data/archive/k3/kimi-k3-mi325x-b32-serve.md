# Kimi-K3 MI325X B32 serving

Date: 2026-08-11. Hardware: 8 leased MI325X GPUs, gfx942, 304 CUs each.
Toolchain: flake-pinned ROCm 7.14.0. Server: `plowrt serve`. Client:
flake-pinned vLLM 0.27.0 `bench serve`, one warmup.

## Build and structural gates

The packet was emitted at TP8 with native MXFP4 weights, FP8 KV, L2 placement,
`PLOW_DECODE_BATCH=32`, `PLOW_GEMV_MM=16`, and `PLOW_GEMV_WALK=1`. It has
programs `[128,512,1024,2048,4096,8192,32]`; decode has 2,942 instructions.
Lean ordering and staged-LDS checks accepted every program.

The B32 decode objects export `plow_gemv_mm_cap_16`, `plow_gemv_walk_1`,
`plow_moe_pf_a4w4_arm`, `plow_k3_arms_1`, and `plow_fp8_kv_1`. Static/GQ
objects remain at 256 VGPR and 64,560/64,568 B LDS with 32 spills. The generic
gfx942 audit and the K3 grouped-A4W4 audit both pass.

The emitter refuses B17..B32 without WALK and all B>32. CMake and
`build_gfx942.sh` enforce the same contract. The TP direct and compact audits
reserve and skip both XArgmax data lines for B32, although this measured packet
keeps the replicated head and therefore uses `ArgmaxFin`.

## Served results

Both cells used random 32-token requests before chat framing, concurrency 32,
greedy generation, `--ignore-eos`, and one warmup.

| prompts | output/request | completed | output tokens | duration | output tok/s | median TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 128 | 64/64 | 8,192 | 95.10 s | 86.15 | 330.78 ms |
| 32 | 512 | 32/32 | 16,384 | 148.79 s | **110.11** | 265.17 ms |

The long cell is the capacity result: it keeps all 32 fixed-width rows occupied
and exceeds the 100 generated-token/s target. Its detailed result has zero
failed requests, 32 output lengths all equal to 512, 32 empty error strings,
and no generated text containing `[error:`. The compact exact TP audit ran once
per decode step. Representative steady-state breakdowns were about 249--255 ms
per step, 98.6% in GPU drain, about 1.50 ms in local counter rearm, and about
1.64 ms in the compact TP audit.

The fixed-width cost is material: the required one-request, 512-token warmup
took 123.45 s because a partially occupied B32 engine still executes all 32
rows. B32 is therefore a throughput asset, not a latency replacement for B1/B8,
until K3 gains a safe decode-width ladder.

Detailed JSON:

- `/tmp/k3-b32-result/seed0.json`
- `/tmp/k3-b32-long-result/seed0.json`

The long client command was:

```bash
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:8032 \
  --endpoint /v1/chat/completions \
  --model k3_farm --served-model-name k3_farm \
  --tokenizer /home/lava/models/k3_tokz --tokenizer-mode hf \
  --dataset-name random --random-input-len 32 --random-output-len 512 \
  --random-range-ratio 0 --request-rate inf --max-concurrency 32 \
  --num-prompts 32 --num-warmups 1 --ignore-eos --temperature 0 --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed \
  --result-dir /tmp/k3-b32-long-result --result-filename seed0.json
```

This is Plow throughput measured with the vLLM client. It is not a comparison
against a vLLM K3 server.
