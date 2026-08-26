# w8a8 prefill win extended across context lengths — Gemma-4-12B / RTX 5090 (sm_120a)

Follow-on to `perf-data/prefill-beats-vllm-w8a8-2026-08-25.md`, same asset/cubin, unchanged. That
report validated the w8a8 win at exactly one point (input-len 8192, concurrency 1). This report
answers the obvious next question the mission's success criteria actually asks: does it hold at
other context lengths, and where does the overall "beats vLLM" claim still fall short?

## Setting

Same protocol as every number in this repo's prior reports: `vllm bench serve --backend
openai-chat`, `--random-output-len 8 --ignore-eos` (TTFT-only), concurrency 1, `--num-prompts 5`,
seed 0. Same assets, same cubins as the validated win — `assets/gemma4-12b-prefill-w8a8-mc8192`
(`cubin-w8a8/`, `-DPLOW_NV_W8A8=ON`, no code changes since the 2026-08-25 win). vLLM 0.27.0 bf16,
`perf-data/tools/vllm_gemma4_launch.py`, `--max-model-len 20000 --gpu-memory-utilization 0.90
--no-enable-prefix-caching`. Sequential exclusive runs (`gpulease`), not concurrent — plow served
and measured first, then vLLM, matching this repo's established comparison methodology.

## Result — the win holds at every context length tested, growing with context

| input-len | plow w8a8 median TTFT | vLLM 0.27.0 bf16 median TTFT | plow speedup |
|---|---|---|---|
| 2,048 | 269.5 ms | 289.2 ms (mean 559.1 ms — first-request compile artifact, median is the robust number) | **1.07x** |
| 8,192 | 946.5 ms | 1198.0 ms | **1.27x** |
| 16,000 | 1985.9 ms | 2564.9 ms | **1.29x** |

All three clear the mission's ≥5%-faster gate at concurrency 1; none regress. The 8,192 point
reproduces the prior report's ~959ms number closely (946.5ms here, same asset, different run —
within normal run-to-run variance, no regression). The margin **grows** with context (7% → 27% →
29%), consistent with `docs/flags-reference.md`'s own finding that plow's cp.async pipeline
advantage compounds at longer contexts (`PLOW_NV_FA_PIPE`: "-16%@4k -> -81%@128k").

**Zero failed requests, either engine, every point.** Raw results:
`perf-data/tools/../../plow-work/bench/{plow_w8a8,vllm_bf16}_prefill_L{2048,8192,16000}.json`
(not committed — session-local scratch under `plow-work/`, reproducible via the commands below).

## What this does NOT close

Concurrency-8 **decode/aggregate output throughput** — the mission's other required gate — was
**not** retested this report; it was already measured in `gemma4-12b-sandbox-5090-2026-08-25.md`
and is a known, unrelated loss: **vLLM ~33-37% faster on decode at every concurrency tested**
(c1/4/8/16, 0.63x at c8 and c16). Decode is a GEMV-dominated path, not GEMM — the w8a8 prefill win
this report extends does not touch it, and this session's kernel-level attempt at a decode-side
win (none — this session's kernel work was scoped to prefill GEMM warp specialization, rejected,
see `perf-data/sm120-iter2-ws-gemm-rejected-2026-08-26.md`) does not change this. **This is why
the mission is not "won overall"**: TTFT/prefill now clears its gate at every tested context
length, but aggregate throughput at concurrency 8 does not, and closing that needs separate
decode-side kernel work this session did not attempt.

Also not covered: the mission's 4th test context (127K) — the current asset's `max_ctx=16384`
caps it below that; a longer-context asset would need re-emitting (`--emit-max-chunk` /
`--max-ctx`) and re-validating, not attempted this report given the asset's own single-chunk
design was specifically tuned for the ≤8192 win and its long-context behavior at 127K is unknown.

## Reproduction

```
# plow (port 8080)
/workspace/plow-work/bin/plowrt serve --assets assets/gemma4-12b-prefill-w8a8-mc8192 \
  --rt-checkpoint /workspace/models/gemma-4-12B-it-merged \
  --nv-cubin-pf cubin-w8a8/interp_sm120_pf.cubin --nv-cubin cubin-w8a8/interp_sm120.cubin \
  --nv-cubin-sample cubin-w8a8/sample_sm120.cubin --port 8080

# vLLM (port 8081)
python3 perf-data/tools/vllm_gemma4_launch.py serve /workspace/models/gemma-4-12B-it \
  --served-model-name gemma-4-12b-it --host 127.0.0.1 --port 8081 \
  --max-model-len 20000 --gpu-memory-utilization 0.90 --no-enable-prefix-caching

# sweep (run against each engine's port in turn)
for L in 2048 8192 16000; do
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --model gemma-4-12b-it --tokenizer /workspace/models/gemma-4-12B-it \
    --host 127.0.0.1 --port <8080|8081> \
    --dataset-name random --random-input-len "$L" --random-output-len 8 \
    --num-prompts 5 --max-concurrency 1 --ignore-eos --seed 0
done
```
