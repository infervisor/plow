# vLLM tensor-parallel decode baseline — Gemma-4-31B on 8× MI350X (gfx950)

**FINDING: vLLM CAN serve Gemma-4-31B multi-GPU on this node** — at TP=2, TP=4, and TP=8 — using the
`vllm/vllm-openai-rocm:latest` image (vLLM **0.25.1+rocm723**, Transformers 5.13.1). All three levels
pass the coherence gate and produce a valid batch-1 bf16 decode baseline. This **contradicts** the
earlier note in `vllm-fp8-baseline.md` ("Gemma doesn't serve on vLLM here"): that finding was specific
to the two *other* images and to single-GPU. See "Why the earlier finding flipped" below.

Measured 2026-07-16 with `perf-data/bench_vllm_tp.sh` (a TP-capable copy of `bench_vllm_docker.sh`).
Batch 1, single-user, bf16, greedy. `vllm bench serve --dataset-name random --random-output-len 128
--max-concurrency 1 --num-prompts 3`, per ctx per TP. Idle GPUs 0–7, HIP_VISIBLE_DEVICES pins the N shards.

## The comparison: plow (bit-exact) vs vLLM 0.25.1 — decode TPOT (ms/tok), lower is better

| ctx  | plow TP2 | vLLM TP2 | plow TP4 (best) | vLLM TP4 | plow TP8 | vLLM TP8 |
|------|---------:|---------:|----------------:|---------:|---------:|---------:|
| 1k   | 15.43    | **9.78** | 13.75           | **7.53** | 15.81    | **6.38** |
| 4k   | —        | 10.75    | —               | 8.50     | —        | 7.45     |
| 8k   | —        | 12.02    | —               | 9.78     | —        | 8.83     |
| 16k  | —        | 12.86    | —               | 10.69    | —        | 9.81     |
| 32k  | —        | 15.64    | —               | 13.56    | —        | 12.67    |
| 64k  | 18.08*   | 17.73    | 14.73           | **15.61**| 16.47    | **14.79**|

plow numbers are the bit-exact TP figures from the design notes §14 (best xr-tuned config: TP2 xr64,
TP4 xr32 = plow's sweet spot 13.75 @1k, TP8 xr16). `*` plow TP2 @64k is the GQ-default value (no xr-tuned
64k point recorded). plow single-GPU reference: **TP1 = 19.1 ms @1k, 22.9 @64k**.

### Verdict
**vLLM is faster than plow at every matched TP level for Gemma-31B batch-1 decode on this hardware.**
At @1k: vLLM TP4 7.53 vs plow TP4 13.75 (1.8×); vLLM even at TP2 (9.78) beats plow's best TP4 (13.75).
The gap narrows at long context (64k: vLLM TP4 15.61 vs plow 14.73 — plow edges ahead), because plow's
fixed per-packet latency F amortizes over the larger KV read while vLLM's advantage is mostly in the
low-context regime. **plow's differentiator is bit-exactness, not raw multi-GPU decode speed** — vLLM
here is torch.compile + TRITON_ATTN + cudagraphs (not bit-exact), whereas plow's TP is token-identical
to single-GPU. The "vLLM cannot serve multi-GPU Gemma here" capability gap does **not** hold.

Two places plow's TP still wins on its own terms:
- **TP=8 scaling.** plow TP8 *regresses* vs TP4 (15.81 > 13.75 @1k — the 8-way all-reduce crosses the
  node0/node1 NUMA boundary). vLLM TP8 keeps improving (6.38 < 7.53 @1k). If plow lands its
  NUMA-hierarchical 2-level all-reduce, TP8 is where it has the most headroom to recover.
- **Long-context TP4 (64k).** plow TP4 14.73 < vLLM TP4 15.61 — the only cell plow leads.

## Reproducibility

- **Image:** `vllm/vllm-openai-rocm:latest` → vLLM `0.25.1+rocm723`, Transformers `5.13.1`, torch 2.11 (ROCm 7.2).
- **Entrypoint quirk:** this image's ENTRYPOINT is already `["vllm","serve"]` (unlike `rocm/vllm:latest`
  which drops to bash). The harness overrides with `--entrypoint vllm` and passes `serve …` uniformly.
- **Exact serve flags (per TP=N):**
  ```
  HIP_VISIBLE_DEVICES=0,…,N-1  --entrypoint vllm  vllm/vllm-openai-rocm:latest
    serve /models/gemma-4-31B-it --dtype bfloat16
      --max-num-batched-tokens 8192 --max-model-len 66560 --tensor-parallel-size N
  ```
  Run flags: `--device=/dev/kfd --device=/dev/dri --group-add video --group-add render
  --security-opt seccomp=unconfined --ipc=host --shm-size=32g`. docker via `sudo -n docker`.
- **Run it:** `TP=4 GPUS=0,1,2,3 IMAGE=vllm/vllm-openai-rocm:latest CNAME=vllm_tp4 PORT=8004
  bash perf-data/bench_vllm_tp.sh`  (add `SERVE_ONLY=1` for just the serve+sanity gate).
- **Data:** `perf-data/gemma4-31b-vllm-tp.json`; raw per-point logs in `perf-data/vllm_tp_logs/`.

## Sanity gate (why raw completions look broken but the serve is fine)

The checkpoint is the **full multimodal** `Gemma4ForConditionalGeneration` (58.25 GiB; vision/audio towers
present). vLLM loads it, shards the text weights across TP (30.16 GiB/GPU at TP=2), forces `TRITON_ATTN`
(heterogeneous head dims 256/512), and logs a non-fatal "Multi-modal warmup failed". Text inference works:

- **Chat (`/v1/chat/completions`, applies the Gemma template + BOS):** coherent at every TP —
  "capital of France" → `Paris`; "three primary colors" → `red, yellow, and blue`.
- **Raw (`/v1/completions`, no template/BOS):** degenerates ("France is France is …"). This is standard
  Gemma-**it** behavior without the chat template, **not** a broken serve. Decode TPOT is identical whether
  output is coherent or repetitive (same per-token forward pass), so the random-dataset timings are valid.

## Why the earlier finding flipped

`vllm-fp8-baseline.md` reported Gemma failing to serve. That was correct **for the images it tried**:
- `rocm/vllm:latest` (Transformers 4.57.3): dies at config-parse — `model type gemma4 not recognized`.
- `vllm-rocm-gemma4:latest` (Transformers 5.14): parses gemma4 but crashed at weight-load, single-GPU.

The third image on the host, `vllm/vllm-openai-rocm:latest` (vLLM 0.25.1, TF 5.13.1), both **resolves**
the multimodal `Gemma4ForConditionalGeneration` arch and **loads** the weights — and TP sharding removes
the single-GPU weight-load pressure. The prior `gemma4-vllm-perf.json` used this same 0.25.1 image but a
*text-only stripped* checkpoint at TP=1; this run serves the *full* checkpoint across TP=2/4/8.

## Notes / caveats

- TTFT at ctx=1k is anomalously high (280–348 ms) vs the 4k point across all TP — a first-request
  torch.compile / cudagraph-specialization artifact; decode TPOT is stable and is the reported metric.
  TTFT/prefill at ≥4k is sane and monotone (TP8 prefill peaks ~60k tok/s @8k).
- vLLM here is **not bit-exact** (inductor + TRITON_ATTN + cudagraphs); plow's TP is bit-exact to TP=1.
  A speed comparison alone understates plow's guarantee.
- All 8 GPUs were idle for this run, so TP=8 was measured directly (not skipped for contention).
