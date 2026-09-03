# `amd-bench` consumer inventory

`plowrt bench` and `plowrt serve` are the performance authorities because they
exercise the production scheduler and engine lifecycle. `amd-bench` remains a
diagnostic runner until its correctness, dump, and TP-audit
surfaces have production-path replacements.

The active script inventory is machine-readable in
[`scripts/amd-bench-consumers.tsv`](../../scripts/amd-bench-consumers.tsv).
Every entry has one primary class and an explicit binding: `checkpoint` or
`synthetic`. Unbound zero-weight runs use the distinct `plowrt amd-probe`
command; every timing line is labeled `SYNTHETIC DIAGNOSTIC` and is never a
performance result. `amd-bench` now requires a checkpoint. Checkpoint-bound
diagnostic invocations are unchanged.
Performance entries are legacy packet/object A/B probes, not served-model
results. They remain only where `plowrt bench` cannot express the same
compiled-width, separate-object, or per-rank semantics.

Run `scripts/check_amd_bench_consumers.sh` in repository checks. It fails when
an active shell invocation is unclassified, when a class or binding is invalid,
when a synthetic consumer does not use `amd-probe`, when a checkpoint-bound
consumer omits `--checkpoint`, or when a registry entry becomes stale.
Comment-only references are not consumers.

The current `performance` rows are a frozen grandfathered set. CI rejects a
new performance row even when it is classified: new performance work must use
`plowrt bench` or `plowrt serve`. The surviving rows require direct packet,
object, compiled-width, or TP-identity semantics that production bench does
not yet express; none can be migrated faithfully in this tranche.

## Documentation inventory

| document | class | disposition |
|---|---|---|
| `docs/arch/06-runtime.md` | correctness | Lists `amd-bench` as diagnostic-only; production performance uses `bench`/`serve`. |
| `docs/bringup/05-single-block-sweep.md` | correctness | Explains which block assets the diagnostic runner can execute. |
| `docs/bringup/06-runtime-opt.md` | correctness | Keeps direct-engine sanity, prefill-sweep, and TP-audit guidance; all reported performance uses `bench`/`serve`. |
| `docs/bringup/07-perf-campaign.md` | performance | Explicitly rejects `amd-bench` for campaign and headline results. |
| `docs/bringup/agents/05-single-block-sweep.md` | correctness | Routes block numerics to the appropriate diagnostic harness. |
| `docs/bringup/agents/06-runtime-opt.md` | performance | Defines the staged deprecation and production benchmark authority. |
| `docs/bringup/agents/07-perf-campaign.md` | performance | Rejects direct-runner output as a correctness or campaign result. |

Historical references under `perf-data/` are evidence records and are not
rewritten by this migration.

## Production diagnostic coverage

| surface | `plowrt bench` coverage | remaining blocker |
|---|---|---|
| raw packet trace | AMD rank 0's last completed program, written after mux drain with `--trace-raw` | Multi-rank trace comparison is not represented. |
| TP agreement | Decode uses the serving cadence plus every-dispatch counter audit; prefill completion compares all ranks. The JSON records the exact policy. Any observed disagreement fails the request. | A full every-token, every-rank stream dump still belongs to the correctness oracle. |
| prefill selection | `plowrt bench --engine-diagnostics` records ordered AMD TP `slot,row_start,rows,bucket` entries from the dispatched `ChunkStep`. | Single-GPU/decode-only fallback and CUDA selection capture report `complete=false`. |
| decode selection | `plowrt bench --engine-diagnostics` records ordered AMD `occupied_rows,bucket,steps` entries at the actual dispatch site, including multistep quanta. | CUDA selection capture is not wired. |
| exact token stream | `plowrt bench --parity-report` records the measured prompt/output token IDs for one exact B1 request. Non-stream `/v1/completions` returns the corresponding IDs only when `return_token_ids=true`; `check_bench_serve_parity.py` validates tokens, usage, chunk/rung coverage, and TP evidence. | A production-checkpoint GPU parity run is still required per artifact; batched per-slot streams remain diagnostic-only. |
| tensor/logit snapshot | Existing AMD `--amd-ctr-snap` and `--amd-tens-snap` writers run inside `AmdServe`. | They use fixed tensors, ignore file-write failures, and do not cover arbitrary logits/tensors like `amd-bench --dump-logits` / `--amd-dump-act`; no consumer migrates yet. |

Selection capture is opt-in so normal benchmark timing is not perturbed. It is
bounded to 16,384 prefill and 16,384 decode entries. An overflow aborts
`plowrt bench` instead of emitting a partial parity record. Capture covers
warmup and measured requests and says so in the JSON `scope`.

For an exact parity cell, run `plowrt bench` with `--prompt-ids`,
`--concurrency 1 --requests 1 --warmup-requests 0`, `--parity-report`, and
`--engine-diagnostics`; issue the same prompt to non-stream `/v1/completions`
with `return_token_ids=true`, then run
`perf-data/tools/check_bench_serve_parity.py BENCH.json ENDPOINT.json`. TP parity
also requires a positive token-audit cadence, every-dispatch counter audit, and
all-rank prefill completion evidence. Missing or incomplete diagnostics fail.
