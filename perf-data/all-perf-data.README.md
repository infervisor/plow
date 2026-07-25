# Consolidated perf-data index

`all-perf-data.json` and `all-perf-data.csv` are a **single flat index of every
measured performance number** in `perf-data/`. They do not replace the
per-campaign source files — those stay as-is; this is a queryable roll-up of them.

Regenerate after adding/changing a source file:

```
python3 perf-data/consolidate_perf.py
```

The generator (`consolidate_perf.py`) parses the structured JSON sources
programmatically and carries the markdown-only tables as transcribed literals.
It prints a row-count + per-model/engine/phase/precision breakdown.

## Schema (one row = one measured value)

| field | meaning |
|---|---|
| `model` | canonical model: `gemma-4-31B`, `gemma-4-12B`, `llama-3.1-8B`, `qwen3-4B`, `qwen3-1.7B` |
| `engine` | `plow` or `vllm` |
| `precision` | `bf16`, `fp8` (weight-only e4m3), `fp8kv` (fp8 weights + fp8 KV cache) |
| `phase` | `decode` or `prefill` |
| `tp` | tensor-parallel degree (int) |
| `ctx` | input context length in tokens (int) |
| `metric` | see metric list below (unit is embedded in the name) |
| `value` | float, transcribed verbatim from the source — never interpolated |
| `source_file` | originating file in `perf-data/` |
| `campaign` | measurement-run label |
| `version` | engine build/version (vLLM version string, or plow branch/source) |
| `git_commit` | commit the source file was committed under, or `(uncommitted)` |
| `date` | measurement date (YYYY-MM-DD) |
| `notes` | per-entry caveats / provenance |

### Metrics

- Decode: `tpot_ms` (time-per-output-token), `itl_ms` (inter-token latency,
  == TPOT in these single-user runs), `decode_tok_s`, `decode_ms_per_token`,
  `decode_step_mean_ms` (profiler mean step), `decode_kernel_<name>_ms`
  (per-kernel decode-step breakdown), `gemv_TBps` (effective GEMV bandwidth).
- Prefill: `ttft_ms` (served: prefill + 1st token), `prefill_ms` (pure prefill,
  in-process/plow), `prefill_tok_s`.

`tpot_ms` / `decode_ms_per_token` are the same quantity under two source
conventions; `ttft_ms` vs `prefill_ms` differ (TTFT includes the first decode
token) — the `notes` field flags which convention each row uses.

## What is and isn't in here

- **Included:** every raw measured value across the 13 JSON files and the 4
  markdown files that carry data found in no JSON (the two plow TP-prefill sweeps,
  `plow-vs-vllm-baseline.md`, and the plow decode figures in `vllm-tp-baseline.md`).
- **Not re-transcribed:** the four summary markdown files whose tables just
  re-render their JSON siblings — `decode-only-sweep.md`,
  `gemma4-31b-longctx-sweep.md`, `vllm-docker-baseline.md`, `vllm-fp8-baseline.md`.
  Read those for prose analysis; their numbers are already indexed via the JSON.
- **Excluded on purpose:** derived analysis present in the sources (plow/vLLM
  ratios, crossover-context estimates, scaling multipliers, speedup percentages).
  Those are computed from the raw rows, not independent measurements.

## Important cross-cutting caveats (also on the individual rows)

- **plow decode has two Gemma-4-31B campaigns.** `decode-only-sweep.json`
  (campaign `decode-only-sweep`) is the DEFINITIVE single-build sweep and
  supersedes the plow decode numbers in `gemma4-31b-longctx-sweep.json`,
  `vllm-tp-baseline.md`, and `vllm_decode_breakdown.json` where contexts overlap.
  The superseded rows are kept (flagged in `notes`) because they are genuinely
  distinct measurements; the 72k longctx point is unique.
- **plow prefill is single-GPU (TP1) in `longctx-sweep`** (no TP prefill in that
  campaign) — the same value fills the TP4/TP8 columns there. TP prefill arrives
  in the `tp-prefill-oneshot` / `tp-prefill-twoshot` campaigns.
- **vLLM is not bit-exact** (torch.compile + TRITON_ATTN + cudagraphs); plow TP is
  token-identical to TP1.
- **`ctx=1024` TTFT is warm-up-contaminated** (first-request HIP-graph capture) in
  the served vLLM runs; decode TPOT is unaffected.
- **vLLM version/harness varies**: `0.25.1+rocm723` served, `0.11.2` docker served,
  and `0.25.1` in-process are three different bases — compare within a base.
- The three `*-vllm-perf.json` in-process files carry a copy-paste `impl:
  "native Gemma4ForCausalLM"` field even for Llama/Qwen models (flagged in notes).
