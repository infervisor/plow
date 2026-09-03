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

The L1 pair is migrated: `l1_ab.sh` uses `amd-probe` for its explicitly
synthetic packet/kernel timing, while `l1_tokens.sh` uses production
`bench --token-audit` and verifies the exact packet, object directory,
checkpoint, prompt, and complete output row before comparing arms.

The batched correctness gates are migrated. `gate_batched.sh` compares exact
ragged B=4/B=8 rows with their B=1 production streams, while
`k3_batch_gate.sh` preserves its non-vacuous identical-stream and two-distinct-
compiled-width checks under the TP8 lease. Both require full-width production
dispatch diagnostics and verify packet, object-directory, checkpoint, prompt
order, row count, and complete nonzero output streams before comparing tokens.
`l2_place_ab.sh` likewise runs each explicitly paired packet/object placement
arm through `bench --token-audit`, records production TPOT, and refuses missing
artifact identities, incomplete/non-AMD work, scheduler loss, zero output, or
any cross-arm token mismatch.
`glm52_linfp8_stacked_coherence.sh` now forces every-token TP agreement and the
per-dispatch counter audit through production `bench`. Each emit preserves its
own `build.json`; the gate verifies that manifest's canonical path and checksum
alongside prompt, output, packet, object-directory, checkpoint, and TP-width
evidence. Missing or partial diagnostics are rejected.

Run `scripts/check_amd_bench_consumers.sh` in repository checks. It fails when
an active shell invocation is unclassified, when a class or binding is invalid,
when a synthetic consumer does not use `amd-probe`, when a checkpoint-bound
consumer omits `--checkpoint`, or when a registry entry becomes stale.
Comment-only references are not consumers.

The current `performance` rows are a frozen grandfathered set. CI rejects a
new performance row even when it is classified: new performance work must use
`plowrt bench` or `plowrt serve`. The surviving rows require direct packet,
object, compiled-width, or TP-identity semantics that production bench does
not yet express.

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
| TP agreement | Decode uses the serving cadence plus every-dispatch counter audit; prefill completion compares all ranks. The JSON records the exact policy. Any observed disagreement fails the request. `glm52_linfp8_stacked_coherence.sh` pins the cadence to every token and validates the complete TP4 policy. | A full per-rank stream dump remains available only in the direct diagnostic runner. |
| prefill selection | `plowrt bench --engine-diagnostics` records ordered AMD TP `slot,row_start,rows,bucket` entries from the dispatched `ChunkStep`. | Single-GPU/decode-only fallback and CUDA selection capture report `complete=false`. |
| decode selection | `plowrt bench --engine-diagnostics` records ordered AMD `occupied_rows,bucket,steps` entries at the actual dispatch site, including multistep quanta. | CUDA selection capture is not wired. |
| exact token stream | `plowrt bench --parity-report` records one exact B1 request. `--prompt-rows` plus `--token-audit` records ordered ragged production-mux rows; `gate_batched.sh` checks B1↔B4/B8 and `k3_batch_gate.sh` checks two distinct compiled widths under TP8. Non-stream `/v1/completions` provides the corresponding C1 endpoint IDs. | Full-logit and arbitrary tensor comparisons remain focused diagnostics. |
| bounded token audit | `plowrt bench --token-audit` records measured prompt/output rows in request order after timing. `--prompt-rows FILE` accepts one exact comma-separated ID row per warmup and measured request, including ragged row lengths. It refuses more than 64 measured requests or 65,536 measured IDs. `gate_quick.sh`, `gate_batched.sh`, and `k3_batch_gate.sh` validate complete production-mux reports and artifact identity. | Tensor-level and full-logit comparisons remain on their focused diagnostic runners. |
| tensor/logit snapshot | AMD `serve` and `bench` share fail-closed `--amd-ctr-snap DIR` and `--amd-tens-snap DIR` capture after each decode dispatch. `--amd-snap-tensors a,b` (`PLOW_SNAP_TENSORS`) selects up to 16 named tensors totaling at most 64 MiB; omission preserves the legacy `act.qa,act.oat,act.attn,act.xn` list, but packets missing those tensors or exceeding the byte bound fail load. The selected slot and every tensor are validated at load, output directories are created before dispatch, and capture/download/write failures fail the request. Files use exclusive creation, so a reused directory or colliding model tick fails instead of overwriting evidence. Either snapshot mode disables multistep capture so every logical token gets its own file set. | The full-logit TP1↔TP8 consumer remains on `amd-bench` until its exact per-step dump/cadence comparison is migrated and proven. |

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

`--token-audit` is a correctness hook, not a timing mode. It is off by default;
when enabled, token rows are cloned only after the measured interval. The JSON
write remains on stdout, so redirection/parse failures abort the calling gate.
