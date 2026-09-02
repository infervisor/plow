# Kimi-K3 Plow baseline on MI355X

Measured 2026-09-02 on the same 8×MI355X node and with the same client
contract as `kimi-k3-vllm-mi355x-baseline.md`. This is the current production
`plowrt serve` C1 baseline, not a projected or in-process result.

## Contract

- TP8, native MXFP4 weights, BF16 KV, MTP/speculation and prefix caching off.
- Raw `/v1/completions`, exact random 8192-token inputs and 1024-token outputs,
  greedy sampling, `--ignore-eos`, infinite request rate, concurrency 1.
- vLLM 0.28 `bench serve` drove both engines with the same tokenizer. Plow's
  `/tokenize` endpoint confirmed the exact 8192-token prompt length.
- One discarded warmup and ten measured requests. All 10 completed, none
  failed, and the client counted exactly 81,920 input and 10,240 output tokens.
- Production B128 packet/interpreter with the B1 low-rung object selected for
  single-user decode. `PLOW_MLA_PF_V2=1`, global queue and L2 placement were on.

## Apples-to-apples result

| metric | Plow | vLLM 0.28 | Plow / vLLM |
|---|---:|---:|---:|
| duration | 600.45 s | 218.13 s | 2.75× |
| request throughput | 0.02 req/s | 0.0458 req/s | 0.44× |
| output throughput | 17.05 tok/s | 46.94 tok/s | 0.36× |
| total token throughput | 153.49 tok/s | 422.50 tok/s | 0.36× |
| mean / median TTFT | 3646.76 / 3645.01 ms | 567.52 / 567.03 ms | 6.43× / 6.43× |
| P90 / P99 TTFT | 3657.95 / 3662.82 ms | 569.22 / 570.50 ms | 6.43× / 6.42× |
| mean / median TPOT | 55.13 / 55.15 ms | 20.768 / 20.768 ms | 2.65× / 2.66× |
| median / P90 / P99 ITL | 55.14 / 55.29 / 55.59 ms | 20.755 / 20.878 / 21.008 ms | 2.66× / 2.65× / 2.65× |
| mean / median E2E | 60.044 / 60.062 s | 21.813 / 21.813 s | 2.75× / 2.75× |

Plow does not yet beat vLLM in this cell. The largest relative deficit is
prefill/TTFT. Decode is independently 2.66× slower by mean TPOT, so prefill-only
work cannot close output-throughput or end-to-end latency.

## KDA Conv3 prefill improvement

The production kernel now parallelizes independent suffix rows across workers
while retaining a unique owner for the incoming-window prefix and final state.
The rule is shape-based (`T > 1` and one shared sequence), not model-specific;
decode and strided multi-sequence rows keep the existing path.

At 8192 tokens, a matched one-request production `plowrt bench` gate measured
4232.02→3631.09 ms TTFT (14.2% faster) with unchanged B1 decode. Both arms
generated the exact checksum `fnv1a64:381f131ba12c92c0`. A separate three-request
`vllm bench serve` screen measured 4250.95→3649.18 ms median TTFT. The full
endpoint result above includes the candidate and confirms the gain at N=10.

Raw rounded client output is recorded in `kimi-k3-plowrt-mi355x-c1.json`.
The vLLM comparator is `kimi-k3-vllm-mi355x-c1.json`.
