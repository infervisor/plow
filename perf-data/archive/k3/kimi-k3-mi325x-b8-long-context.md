# Kimi-K3 MI325X B8 long-context serving

Date: 2026-08-11. Server: `plowrt serve` on 8 leased MI325X GPUs.
Client: flake-pinned vLLM 0.27.0 `bench serve`. Native MXFP4 weights, FP8
KV, TP8, compact exact TP audit, one warmup, concurrency 8.

## Results

| requested input | actual input/request | requests | output/request | p50 TTFT | duration | output tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 16,000 | 16,021--16,022 | 8/8 | 128 | 44.50 s | 97.11 s | 10.54 |
| 32,000 | 32,021--32,022 | 8/8 | 128 | 101.70 s | 199.80 s | 5.13 |

Both cells generated exactly 1,024/1,024 requested output tokens, reported zero
failed requests and empty per-request errors, contained no in-band `[error:`
text, and stayed below the 32,768-token asset limit after chat framing and
completion. Detailed JSON:

- `/tmp/k3-b8-longctx-sweep/b8_ctx16000_c8_n8_seed0.json`
- `/tmp/k3-b8-longctx-sweep/b8_ctx32000_c8_n8_seed0.json`

The exact runner was:

```bash
nix develop .#vllm --command env \
  PLOW_K3_BASE_URL=http://127.0.0.1:8033 \
  PLOW_K3_CONTEXTS='16000 32000' PLOW_K3_SEEDS=0 \
  PLOW_K3_OUTLEN=128 PLOW_K3_NWARM=1 \
  PLOW_K3_OUTDIR=/tmp/k3-b8-longctx-sweep PLOW_K3_TAG=b8 \
  scripts/k3_context_sweep.sh
```

## Attribution

This table is prefill/TTFT, not the B32 110.11 tok/s decode-capacity result.
K3 runs one prefill chunk per mux tick and interleaves an already-live B8 decode
after that chunk. At 32K each request needs four causal 8K chunks; the eight
requests therefore serialize 32 large prefill launches. The 32K p50 TTFT is
2.29x the 16K value rather than 2x, consistent with causal MLA flash work growing
with the accumulated KV length.

After the prefill wave, logged B8 steps were about 128--145 ms, with 97%--98% in
GPU drain, about 1.50 ms in local-counter rearm, and about 1.66 ms in compact TP
audit. The low 10.54/5.13 output tok/s values above include the large TTFT and do
not describe steady-state decode capacity.

The next prefill target remains a K3-specific FP8 MLA flash arm at long context,
followed by cross-request packed prefill. The tested KDA state-residency and W2
touch experiments did not improve their measured production paths.
