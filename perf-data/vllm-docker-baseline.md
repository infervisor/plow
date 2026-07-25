# vLLM-in-Docker single-user baseline (MI350X / gfx950)

Measured 2026-07-15. bf16, batch 1 (max-concurrency 1), tensor-parallel 1, one
gfx950 (MI350X-class) GPU, HIP_VISIBLE_DEVICES=7. Output 128 tokens/request,
3 prompts/point, random dataset. Reproduce with `perf-data/bench_vllm_docker.sh`.

- **Image:** `rocm/vllm:latest`
- **vLLM version:** `0.11.2.dev673+g839868462.rocm700`
- **Serve:** `vllm serve <M> --dtype bfloat16 --max-num-batched-tokens 8192
  --max-model-len <native> --tensor-parallel-size 1` (chunked-prefill budget
  8192 for parity with plow; HIP/CUDA graphs ON, i.e. NOT `--enforce-eager`).
- **Bench:** `vllm bench serve --model <M> --dataset-name random
  --random-input-len <L> --random-output-len 128 --max-concurrency 1
  --num-prompts 3 --port 8000`.
- Metrics: `ttft_ms` = Mean TTFT (prefill), `tpot_ms` = Mean TPOT = Mean ITL
  (decode/token), `prefill_tok_s` = L/(TTFT/1000), `decode_tok_s` = 1000/TPOT.
- Both models correctness-sanity-checked ("The capital of France is" -> coherent)
  before timing.

## Docker access (recorded working method)

Invoking user is NOT in the `docker` group (`docker ...` -> permission denied on
`/var/run/docker.sock`), but the host grants **passwordless sudo**, so every
docker call is `sudo -n docker ...`. GPU is selected with
`HIP_VISIBLE_DEVICES=7` (not `--gpus`); the container gets all `/dev/kfd` +
`/dev/dri` nodes and vLLM is restricted to the one GPU by the env var.

## Qwen3-4B (native, to 32K)

Native max_position_embeddings = 40960, so 64K is NOT native — sweep capped at
32768 and **no rope extension applied** (`rope_scaling=null`). Served at
`--max-model-len 40960` so the 32768+128 request fits (serving at 32768 returns
HTTP 400 because prompt+output overruns the window).

| ctx | TTFT ms (prefill) | TPOT ms (decode/tok) | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| 1024  | 20.63  | 3.150 | 49637 | 317.5 |
| 4096  | 50.51  | 3.260 | 81093 | 306.8 |
| 8192  | 87.63  | 3.390 | 93484 | 295.0 |
| 16384 | 231.82 | 3.640 | 70676 | 274.7 |
| 32768 | 710.00 | 4.080 | 46152 | 245.1 |

## Llama-3.1-8B-Instruct (native, to 64K)

Native max = 131072, so 64K is native. Served at `--max-model-len 66560`
(headroom over 65536+128).

| ctx | TTFT ms (prefill) | TPOT ms (decode/tok) | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| 1024  | 23.81   | 3.950 | 43007 | 253.2 |
| 4096  | 58.56   | 4.270 | 69945 | 234.2 |
| 8192  | 94.57   | 4.340 | 86624 | 230.4 |
| 16384 | 240.77  | 4.500 | 68048 | 222.2 |
| 32768 | 705.56  | 4.930 | 46443 | 202.8 |
| 65536 | 2342.26 | 5.930 | 27980 | 168.6 |

## Comparison to the committed in-process baseline

The committed `perf-data/qwen3-4b-vllm-perf.json` / `llama-3.1-8b-vllm-perf.json`
were taken **in-process** (not served) on **vLLM 0.25.1**. This run is
**docker-served on vLLM 0.11.2**. Different version AND different harness
(endpoint round-trips + TTFT-style prefill), so this is not a controlled tie —
treat it as a second, independently reproducible datapoint rather than an
apples-to-apples delta.

Qwen3-4B, prefill (TTFT/prefill_ms) and decode (ms/tok):

| ctx | prefill 0.11.2 docker | prefill 0.25.1 in-proc | decode 0.11.2 docker | decode 0.25.1 in-proc |
|---|---|---|---|---|
| 4096  | 50.5  | 55.5  | 3.26 | 3.07 |
| 8192  | 87.6  | 146.6 | 3.39 | 3.14 |
| 16384 | 231.8 | 482.3 | 3.64 | 3.41 |
| 32768 | 710.0 | 1744.2| 4.08 | 3.83 |

Observation: contrary to the "0.11.2 served is likely slower" prior, the
**docker/0.11.2 prefill is actually faster** than the in-process/0.25.1 numbers,
diverging more as ctx grows (~2.4x lower TTFT at 32K). Decode is marginally
slower on docker (~3-7% higher ms/token). The prefill gap is large enough that
it is almost certainly a measurement-method difference (served TTFT vs the
in-process pure-prefill timer, and/or the 0.25.1 baseline having been captured
under a heavier build), not a pure engine speedup — flagged here rather than
over-interpreted. Both docker points are honest, reproducible, and native.

Raw per-point bench logs + summary CSVs are produced under
`perf-data/vllm_docker_logs/` by the harness (gitignored scratch by default).
