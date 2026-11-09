# Model bringup

From a HuggingFace `config.json` to a served, measured deployment, one gated stage at a time.
Each stage file holds its steps, gate, pitfalls and stop-and-ask conditions. Cross-cutting rules
live once in [`agent-tools.md`](agent-tools.md); read it before any GPU or performance work.

## Stages

| stage | file | gate (must pass before the next stage) |
|---|---|---|
| 1. nn-graph + emit wiring | [01-nn-graph.md](01-nn-graph.md) | graph builds and infers; manifest matches the checkpoint; full and `--block` devblob emit simulate clean |
| 2. egglog rewrite | [02-egglog-rewrite.md](02-egglog-rewrite.md) | intended fusions fire, no weight dropped; every rule in `soundRules` |
| 3. Lean verify | [03-lean-verify.md](03-lean-verify.md) | `proof_audit` clean; emit checkpoints `ok`; `plowc qualify` has no unnamed gap |
| 4. kernel tuning | [04-kernel-tuning.md](04-kernel-tuning.md) | hot kernels correct and at the calibrated roofline, or the gap is explained |
| 5. single-block sweep | [05-single-block-sweep.md](05-single-block-sweep.md) | block matches the oracle; block latency vs its roofline recorded |
| 6. runtime optimization | [06-runtime-opt.md](06-runtime-opt.md) | memory fits; TTFT/TPOT at target concurrency via `plowrt bench`; recipe TOML records the settings |
| 7. perf campaign | [07-perf-campaign.md](07-perf-campaign.md) | `llm_fp32_ref` gate passes; `campaign.py report` strict report written |
| 8. rung campaign | [08-rung-campaign.md](08-rung-campaign.md) | per-rung lever decided against its floor; recipe updated only for qualified wins |

Stages 4/5 iterate; stage 8 is the steady-state optimization loop after stage 6.

## Target

- Stages 1–3 are target-independent: they compile and prove; their gates hold on any host with
  the dev shell.
- Stages 4+ measure one part. Fill the parameter block in [`target.md`](target.md) first; commands
  are written in its names (`--gpu $GPU --arch $ISA`). A row you cannot fill is a blocker.
- Kernel work starts from the ISA reference for `$ISA`: [`isa/README.md`](isa/README.md).

## Running a stage with an agent

1. Start the agent at the repo root with `nix develop --command` available.
2. Give it the stage file and, for stages 4+, the filled `target.md` block, plus the model
   inputs (HF id or checkpoint dir, reference implementation).
3. The agent passes the stage gate or stops and reports; it never weakens a gate or carries a red
   gate forward. A human reviews the gate evidence before the next stage.

## Cross-cutting rules (agent-tools.md)

- Single-block screening against the roofline before any whole-network run: "HARD RULE".
- Team rules (prefix cache on, one binary, packet contract, static runtime, vLLM-peer judging,
  GPU budget, `prod-next`, kernel reviews): "Working rules".
- Environment variables and the nix shell: "Environment".
- Every GPU process through `scripts/bench/gpuq.py submit`: "Queue".
- Campaign order and recipe start (default-on template, doctor): "Campaign playbook".
- Per-rung choice and the `recipes/infervisor` TOML: "Per-rung gate and recipe selection".
- FP32, multimodal and serve-golden gates, gate relaxation: "Gates".
- Same-session A/B, repeats and noise: "A/B rule".
- Measurement hazards: "Measurement hazards".
- Final plow-vs-baseline comparison: "Final performance report (strict)".

## Results

- Campaign summary: `docs/bringup/results/<campaign>/comparison.{md,csv}`
  ([results/README.md](results/README.md)).
- Raw logs, JSON, captures and temporary CSVs stay in scratch outside the repo.
- `perf-data/campaign/` holds only reference CSVs for `campaign.py compare`.
