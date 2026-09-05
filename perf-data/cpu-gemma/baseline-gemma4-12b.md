# vLLM CPU baseline — google/gemma-4-12B-it (bf16)

Date: 2026-09-05. Raw data: `baseline-gemma4-12b.json` (bind 0-15, primary),
`baseline-gemma4-12b-bind0-7.json`, `bench_serve_128in_32out*.json`, `smoke.json`.

## Software

| item | value |
|---|---|
| vLLM | 0.28.0+cpu — official pre-built CPU wheel `vllm-0.28.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl` (GitHub release asset; nixpkgs was rejected because it would build vLLM + torch from source) |
| torch | 2.13.0+cpu (`https://download.pytorch.org/whl/cpu`) |
| transformers | 5.16.1 (has `gemma4_unified`; no upgrade needed) |
| Python | 3.12.14 (nix `python312`) venv at `/home/lava/vllm-cpu/venv` |
| runtime libs | nix `gcc-15.3.0-lib` (libstdc++/libgomp), nix `numactl-2.0.18` (libnuma, without it `vllm._C` fails to load), nix `gcc-wrapper-13.4.0` (g++ for torch.compile/inductor) |
| LD_PRELOAD | `libiomp5.so` (intel-openmp 2024.2.1) + vLLM's bundled `libtcmalloc_minimal.so.4` |

Model: `/home/lava/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`,
architecture `Gemma4UnifiedForConditionalGeneration`, 23 GB safetensors, served as `gemma-4-12b-it`.

## Server flags

```
VLLM_CPU_KVCACHE_SPACE=2 VLLM_CPU_OMP_THREADS_BIND=0-15 \
vllm serve <snapshot> --served-model-name gemma-4-12b-it --dtype bfloat16 --max-model-len 4096 \
  --max-num-seqs 8 --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' --port 8094
```

* Attention backend: `CPU_ATTN` (HND KV layout). torch.compile on (default level), AOT cache under `~/.cache/vllm/torch_compile_cache`.
* AMX: detected by torch (`torch.cpu._is_amx_tile_supported() == True`, amx_fp16 False). vLLM uses it automatically
  (oneDNN GEMMs for the dense projections, SGL AMX kernel for weights <= 1 MiB); no AMX flag exists or is needed.
* KV cache 2 GiB = 8,112 tokens (1.98x concurrency at 4096). Worker RSS ~30 GiB at steady state; peaked 37.7 GiB with
  `VLLM_CPU_KVCACHE_SPACE=8`, which got OOM-killed twice while another job held 20-25 GiB. Single-stream latency does not
  depend on KV size.
* `--limit-mm-per-prompt ... 0` skips multimodal profiling (text-only baseline; the unlimited default raises
  `max_num_batched_tokens` to 2496 for video and adds profiling memory).
* Startup: ~25-30 s weights, ~75 s compile (cache did not reload: "Compiling model again due to a load failure"), ready in ~2.5 min.

## Machine

* INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz (Emerald Rapids class), 1 socket, 8 cores x 2 HT = 16 vCPU, 1 NUMA node
* Flags: amx_bf16, amx_int8, amx_tile, avx512_bf16, avx512_vnni
* RAM: 58 GiB, no swap, no GPU; kernel 7.0.0-1011-gcp
* Box is shared: another job intermittently runs 20-25 GiB processes (`cpu_bench`/`cpu_chat`/`cpu_probe`). Each timed run
  below recorded load and co-tenant processes; none were present during any timed run.

## Results — streaming chat/completions API (`bench_api.py`)

Greedy (temperature 0), `max_tokens=32`, 1 warmup + 3 timed runs, concurrency 1. TTFT = first content chunk;
TPOT = mean gap between streamed chunks (all 32 tokens streamed as single-token chunks). `server prompt_tokens`
includes the chat template (+12 tokens over the raw text).

Binding `0-15` (all 16 vCPUs) — primary:

| prompt tokens (raw / server) | TTFT mean ms | TTFT min ms | TPOT mean ms | TPOT min ms | decode tok/s | e2e s |
|---|---|---|---|---|---|---|
| 46 / 58   | 697.6 | 697.2 | 457.5 | 456.9 | 2.19 | 14.88 |
| 136 / 148 | 742.5 | 736.9 | 455.3 | 454.9 | 2.20 | 14.86 |
| 526 / 538 | 752.4 | 750.4 | 457.1 | 456.8 | 2.19 | 14.92 |

Binding `0-7` (one thread per physical core):

| prompt tokens (raw / server) | TTFT mean ms | TPOT mean ms | TPOT min ms |
|---|---|---|---|
| 46 / 58   | 734.0 | 545.3 | 540.5 |
| 136 / 148 | 765.8 | 520.7 | 485.3 |
| 526 / 538 | 749.7 | 453.9 | 453.3 |

The 0-7 pass drifted 554 -> 453 ms over the first ~2 min after startup with no co-tenant present, then matched the
0-15 pass. Steady-state TPOT is ~455 ms either way; TTFT is within noise. Keep `0-15`.

Observations:
* Decode is ~455 ms/token (2.2 tok/s) regardless of prompt length — memory-bandwidth bound streaming 23.5 GiB of
  bf16 weights per step (~52 GB/s effective).
* TTFT is nearly flat (698 -> 752 ms from 58 to 538 prompt tokens): prefill is compute-bound on AMX, ~0.1 ms/token
  incremental, on top of a ~650 ms fixed cost (one full weight sweep + scheduling/HTTP).
* Run-to-run spread is <1% for TPOT and <2% for TTFT.

## Cross-check — `vllm bench serve`

`--backend openai-chat --dataset-name random --random-input-len 128 --random-output-len 32 --num-warmups 1 --num-prompts 4 --max-concurrency 1 --ignore-eos --temperature 0`

| binding | mean TTFT ms | median TTFT ms | mean TPOT ms | mean ITL ms | mean E2EL ms |
|---|---|---|---|---|---|
| 0-15 | 1008.6 | 1182.0 | 454.6 | 440.4 | 15102 |
| 0-7  | 1003.5 | 1173.9 | 452.6 | 438.5 | 15035 |

TPOT agrees with `bench_api.py` (455 vs 457 ms). vLLM's TTFT is higher because its random dataset uses 128 random
token ids (not natural text) and the tool's first-token accounting includes tokenizer/HTTP overhead differently;
treat the `bench_api.py` numbers as the API-level baseline for the head-to-head.

Offline `vllm bench latency` was not run: it loads a second 23 GB copy of the model, which does not fit alongside the
server on this 58 GiB shared box; `vllm bench serve` against the live server is the cross-check instead.

## Smoke test

"What is the capital of France? Answer in one word." -> `Paris` (prompt 25 tokens, 2 completion tokens, ~1.2 s e2e).
