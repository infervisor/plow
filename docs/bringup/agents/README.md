# Bringup agent harness

Prompt templates for driving a model bringup with LLM coding agents. Each file
in this directory is a self-contained prompt for one stage of the
pipeline described in [`docs/bringup/00-overview.md`](../00-overview.md):

| Prompt | Stage |
|---|---|
| [`agents/01-nn-graph.md`](01-nn-graph.md) | Operator-IR bringup (HF config → `Nn` builder → weight manifest) |
| [`agents/02-egglog-rewrite.md`](02-egglog-rewrite.md) | Rewrite/fusion rules with the Checkpoint-A soundness obligation |
| [`agents/03-lean-verify.md`](03-lean-verify.md) | Formal checkpoints A–G |
| [`agents/04-kernel-tuning.md`](04-kernel-tuning.md) | Kernel tuning to the measured roofline |
| [`agents/05-single-block-sweep.md`](05-single-block-sweep.md) | Single-block correctness + latency |
| [`agents/06-runtime-opt.md`](06-runtime-opt.md) | Whole-model serving optimization |
| [`agents/07-perf-campaign.md`](07-perf-campaign.md) | End-to-end measured campaign + written results |

Prompts **01–03 are target-independent** and take no target parameters.
Prompts **04–07 each open with the parameter block** from
[`docs/bringup/target.md`](../target.md) — `$VENDOR $ISA $GPU $NCU $NGPU
$PARALLEL $MAXCTX $TOOLCHAIN $BUILD $FEATURES $BW_BOUND $COMPUTE_CEIL
$RESULTS` — which must be filled in before the agent runs anything. Their
commands are written in those names; a literal part name in a command is a
defect, and a row that cannot be filled is a blocker, not a default.

## How to run a stage

1. Open a coding agent (any tool that can read the repo, run shell commands,
   and edit files) in the repo root, inside the nix dev shell or with the
   ability to invoke `nix develop --command`.
2. Paste the stage prompt, filling in the placeholders at the top — the model
   ones (HF model id, parameter budget) and, for stages 4–7, the whole target
   parameter block.
3. The prompt tells the agent what to read first, the edits to make, the
   commands to run, and the **gate** it must pass. The agent reports back in a
   fixed format; a human reviews the gate evidence before the next stage.

## Ground rules for agents (enforced by every prompt)

- **Gates are blocking.** An agent that cannot pass its gate stops and reports;
  it does not proceed or weaken the gate.
- **The target is never hardcoded.** Stages 4–7 write `--gpu $GPU --arch $ISA`;
  a number measured on one part is never carried to another, including between
  two parts at the same `$ISA`.
- **Stop-and-ask conditions are listed per stage** — hardware access, ambiguous
  numerics tolerances, and anything that would change another stage's contract.
- **Measurements follow the campaign conventions**: GPU leasing, same-session
  A/B, one lever at a time, honest write-ups (see
  [`docs/bringup/07-perf-campaign.md`](../07-perf-campaign.md)).
- **Harness selection comes before execution.** Search the existing
  `runtime/bench/`, `runtime/tests/`, `scripts/`, and `perf-data/tools/`
  harnesses and name the selected harness in the report. Extend an existing
  harness when its semantic boundary is incomplete; do not create a
  campaign-specific runner for a case an existing harness can express.
- **Spend the sweep budget on blocks.** Use standalone probes to reject broken
  arms, single-block or truncated-model sweeps to rank the broad grid, and a
  whole-model step only for the 2–3 finalists. Serving is the promotion gate,
  not the tuning loop.
- **Proof obligations are never vacuous.** Stage 2/3 prompts require real
  `rfl`-backed theorems for new rewrite rules; `sorry` fails the gate.

The prompts encode what the historical bringups (Gemma, GLM, Kimi-K3,
DeepSeek, Nemotron) actually required — including the known divergences
between the idealized pipeline and the shipping code paths, which each stage
doc calls out explicitly rather than papering over.
