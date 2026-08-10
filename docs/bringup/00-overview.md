# Model bringup — overview

> **Goal:** take a new model architecture from a HuggingFace `config.json` to a
> served, measured deployment, one gated stage at a time.

This playbook is the distilled form of how every model in the tree (Gemma,
Llama, Qwen, DeepSeek, GLM, Kimi-K3, Nemotron) was actually brought up. Each
stage has a **gate** — a concrete, checkable success criterion — and a matching
agent prompt in [`bringup/agents/`](../../bringup/agents/) so the stage can be
executed by an LLM coding agent under supervision (see
[`bringup/README.md`](../../bringup/README.md)).

## The pipeline

| Stage | What | Gate (must pass before next stage) |
|---|---|---|
| [1. nn-graph](01-nn-graph.md) | Add the architecture to the operator IR: HF config struct, `Nn` builder, `weight_manifest`, shape inference | Graph builds, shapes infer, `--no-default-features` IR still compiles |
| [2. egglog rewrite](02-egglog-rewrite.md) | Decide whether the arch needs new rewrite/fusion rules; add them with `; rule:` annotations | Saturation runs, intended fusions fire, every fired rule is in `soundRules` |
| [3. lean verify](03-lean-verify.md) | Discharge new proof obligations across checkpoints A–G | `lake build` clean, all checkpoints certify `ok`, no vacuous proofs |
| [4. kernel tuning](04-kernel-tuning.md) | Tune the model's hot kernels (GEMM/GEMV/attention/MoE) against the measured roofline | Hot kernels at/near the measured ceiling; winners recorded and read back from tunedb |
| [5. single-block sweep](05-single-block-sweep.md) | Extract one transformer block, prove it numerically correct, tune it end-to-end | Block matches the oracle reference; block latency at target |
| [6. runtime optimization](06-runtime-opt.md) | Serve the whole model: KV fit, prefix cache, chunking, admission, TP | TTFT/TPOT at target concurrency; memory fits; no shed-request artifacts |
| [7. perf campaign](07-perf-campaign.md) | The final reproducible measurement: sweeps + correctness battery + written results | Correctness battery passes AND perf targets met, results documented in `perf-data/` |

Stages are ordered by dependency, not by wall-clock share: in practice stages
4–7 dominate the effort, and stages 4/5 iterate (a block-sweep finding often
sends you back to kernel tuning).

## Conventions that apply to every stage

- **Build through nix.** `nix develop --command bash -c '<cargo …>'` — there is
  no cargo on PATH outside the dev shell. GPU kernel builds are separate
  scripts (`scripts/build_*.sh`), not cargo.
- **One lever at a time.** Historical campaigns that changed two knobs in one
  run always paid for it in attribution time.
- **Same-session measurement.** Any A/B comparison must run both arms in the
  same GPU lease (`perf-data/harness/gpulease`); cross-session numbers drift.
- **Honest reporting.** Results are quantified, caveated, and neutral — no
  win/loss framing. A measurement that refutes the hypothesis is a result,
  not a failure; write it down (the tree keeps several such write-ups in
  `perf-data/` deliberately).
- **Gates are real.** Do not carry a red gate forward "to fix later" — every
  historical instance of that turned a one-stage defect into a multi-stage
  bisection.

## Where things live

- `docs/bringup/NN-*.md` — stage playbooks (this directory).
- `bringup/agents/NN-*.md` — per-stage agent prompts.
- `docs/arch/` — the architecture chapters the playbooks cite.
- `perf-data/` — campaign methodology, harness scripts, and written results.
