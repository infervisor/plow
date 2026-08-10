# Bringup agent harness

Prompt templates for driving a model bringup with LLM coding agents. Each file
in [`agents/`](agents/) is a self-contained prompt for one stage of the
pipeline described in [`docs/bringup/00-overview.md`](../docs/bringup/00-overview.md):

| Prompt | Stage |
|---|---|
| [`agents/01-nn-graph.md`](agents/01-nn-graph.md) | Operator-IR bringup (HF config → `Nn` builder → weight manifest) |
| [`agents/02-egglog-rewrite.md`](agents/02-egglog-rewrite.md) | Rewrite/fusion rules with the Checkpoint-A soundness obligation |
| [`agents/03-lean-verify.md`](agents/03-lean-verify.md) | Formal checkpoints A–G |
| [`agents/04-kernel-tuning.md`](agents/04-kernel-tuning.md) | Kernel tuning to the measured roofline |
| [`agents/05-single-block-sweep.md`](agents/05-single-block-sweep.md) | Single-block correctness + latency |
| [`agents/06-runtime-opt.md`](agents/06-runtime-opt.md) | Whole-model serving optimization |
| [`agents/07-perf-campaign.md`](agents/07-perf-campaign.md) | End-to-end measured campaign + written results |

## How to run a stage

1. Open a coding agent (any tool that can read the repo, run shell commands,
   and edit files) in the repo root, inside the nix dev shell or with the
   ability to invoke `nix develop --command`.
2. Paste the stage prompt, filling in the model placeholders at the top
   (HF model id, target arch, parameter budget).
3. The prompt tells the agent what to read first, the edits to make, the
   commands to run, and the **gate** it must pass. The agent reports back in a
   fixed format; a human reviews the gate evidence before the next stage.

## Ground rules for agents (enforced by every prompt)

- **Gates are blocking.** An agent that cannot pass its gate stops and reports;
  it does not proceed or weaken the gate.
- **Stop-and-ask conditions are listed per stage** — hardware access, ambiguous
  numerics tolerances, and anything that would change another stage's contract.
- **Measurements follow the campaign conventions**: GPU leasing, same-session
  A/B, one lever at a time, honest write-ups (see
  [`docs/bringup/07-perf-campaign.md`](../docs/bringup/07-perf-campaign.md)).
- **Proof obligations are never vacuous.** Stage 2/3 prompts require real
  `rfl`-backed theorems for new rewrite rules; `sorry` fails the gate.

The prompts encode what the historical bringups (Gemma, GLM, Kimi-K3,
DeepSeek, Nemotron) actually required — including the known divergences
between the idealized pipeline and the shipping code paths, which each stage
doc calls out explicitly rather than papering over.
