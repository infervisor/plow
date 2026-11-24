# Model Bringup

From a Hugging Face `config.json` to a verified, served deployment, one gated stage at a time.

Cross-cutting rules, execution environments, and CLI tools live in [`agent-tools.md`](agent-tools.md). Consult it before initiating GPU work.

---

## Bringup Stages

| Stage | Document | Gate Criterion |
|---|---|---|
| 1. NN Graph & Emitter | [01-nn-graph.md](01-nn-graph.md) | Graph builds and infers; manifest matches checkpoint; full and `--block` emissions simulate clean. |
| 2. Egglog Rewrite | [02-egglog-rewrite.md](02-egglog-rewrite.md) | Fusible patterns fire without dropping weights; every rule verified in `soundRules`. |
| 3. Lean Verification | [03-lean-verify.md](03-lean-verify.md) | `proof_audit` clean; emit checkpoints return `ok`; `plowc qualify` has no unapproved gaps. |
| 4. Kernel Tuning | [04-kernel-tuning.md](04-kernel-tuning.md) | Hot kernels verified against oracles and tuned to calibrated roofline floor on `$GPU`. |
| 5. Single-Block Sweep | [05-single-block-sweep.md](05-single-block-sweep.md) | Block matches host oracle; latency certified by `certify_kernel.py` against roofline. |
| 6. Runtime Optimization | [06-runtime-opt.md](06-runtime-opt.md) | Memory fits target concurrency; TTFT/TPOT within budget; recipe TOML records parameters. |
| 7. Performance Campaign | [07-perf-campaign.md](07-perf-campaign.md) | `llm_fp32_ref` passes; strict 12-row report rendered via `campaign.py report`. |
| 8. Rung Campaign | [08-rung-campaign.md](08-rung-campaign.md) | Per-rung lever evaluated through T1–T4 ladder; qualified gains update production recipe. |

* Stages 1–3 are target-independent (CPU only).
* Stages 4–8 evaluate against physical hardware defined in [`target.md`](target.md). Fill target parameters before entering Stage 4.
* Kernel implementations start from the target architecture reference in [`isa/README.md`](isa/README.md).

---

## Stage Execution Protocol

1. **Environment:** Execute all commands within `nix develop --command <cmd>`.
2. **Context:** Provide the agent with the target stage file, model identifier/checkpoint, and the filled `target.md` parameters.
3. **Hard Gates:** Every stage gate must pass before advancing. Gates are never waived without documented, measured justification.
4. **Queueing:** All GPU runs must be dispatched via `scripts/bench/gpuq.py submit <label> <ngpu> <cmd...>`.

---

## Results Directory Structure

* **Campaign Summary:** `docs/bringup/results/<campaign>/comparison.{md,csv}` (indexed in [results/README.md](results/README.md)).
* **Reference CSVs:** `perf-data/campaign/` (used by `campaign.py compare`).
* **Scratch & Artifacts:** All raw logs, traces, captures, and intermediate JSON files must remain in external scratch directories outside the Git repository.
