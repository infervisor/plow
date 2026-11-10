# CLAUDE.md

## Context

* Plans/research live in `plans/` (gitignored). If the user names a plan, read it first.
  Update it only when decisions change.

## Repository layout (strict)

Put code where its kind already lives. Never create a new top-level directory, a new
`scripts/` subdirectory, or a parallel copy of an existing tool. Ask when nothing fits.

* `crates/` — the Rust workspace. `plowc` (compile/emit), `plowrt` (runtime, serve, dist, recipe
  parsing), `plow-asset` (packet and manifest schemas), `devgen`, `hwspec`, `tunedb`, `costmodel`,
  `kernelcaps`, `packet`, `schedule`, `rewrite`, `nn-graph`, `lean_verify`. Runtime, compiler,
  packing and publishing behaviour lives in the crate that owns it, never in a script.
* `include/packet.h` (repo root) — the packet ABI.
* `runtime/` — C/C++/CUDA/HIP kernel sources and CMake: `nvidia/`, `amd/`, `apple/`, `cpu/`,
  `common/`, `tests/`, `ubench/`, `bench/`.
* `scripts/` — operator tooling, one reusable capability per file. New scripts go in a
  subdirectory; top-level scripts are legacy. Subdirectories: `bench/` (plowbench, doctor, gpuq,
  serving comparison, replays), `campaign/` (campaign.py, rung/roofline/evidence tools,
  experimental recipes under `campaign/recipes/`), `asr/`, `tts/`, `voice/`, `llm/`, `mm/`
  (per-modality gates and prep), `gen_kernels/`, `fa4_aot/`, `gdn_aot/`, `serve_test/`,
  `twoengine/`, `vllm_capture_site/`, `tests/`. Recipe tools are `check_recipe.py`,
  `render_recipe.py` and their shared `recipe_common.py`. A `test_<name>.py` sits beside `<name>`.
  No one-off probes, captures, dumps or throwaway drivers: those are scratch outside the repo.
* `recipes/infervisor/<model>/<label>.toml` + `.md` — the one canonical production recipe per
  model/GPU/precision. Experimental recipes stay in `scripts/campaign/recipes/`.
* `perf-data/` — `campaign/` (reference CSVs for `campaign.py compare`), `tools/` (gpulease,
  legacy bringup helpers, quantizers, result validators), `packets/`, `probes/`, `block-configs/`.
  `perf-certs/` — checkpoint P certificates. `tuning/` — kernel-level measurement records keyed
  by hardware fingerprint, written only by a tuning run.
* `docs/` — `BUILD.md`, `DISTRIBUTION.md`, `arch/` (design, organised by technique; never names a
  tool or assistant, model names allowed as examples), `bringup/` (stage playbooks,
  `agent-tools.md`, `isa/`, `results/`), `runtime/`, `schemas/`, `amd/`.
* Campaign summaries: `docs/bringup/results/<campaign>/comparison.{md,csv}`. Raw logs, JSON,
  captures, generated reports and temporary CSVs go to campaign scratch outside the repo, never
  under `docs/`, `scripts/`, `perf-data/` or `recipes/`.
* `lean-plow/` — Lean 4 verification. `tools/` — the `bench` crate and `bench-api`.
  `assets/`, `media/` — images only.

## Nix environment

* Every build, emit, serve, bench and test runs inside `nix develop`
  (`nix develop --command <cmd>`). The flake pins ROCm, CUDA and the Rust toolchain, and provides
  elan (Lean version from `lean-plow/lean-toolchain`). `ROCM_PATH` unset means you are outside it:
  stop and re-enter, do not work around it.
* Never fake `ROCM_PATH`, hardcode a `/nix/store` path, or add a toolchain or Python dependency
  outside `flake.nix`. A hand-built box without nix sets `PLOW_CAMPAIGN_NO_NIX=1`; that is the
  only sanctioned exception.
* The vLLM client and the ROCm lib it links are not from nix: pass them as `PB_VLLM` and
  `PB_VLLM_ROCM_LIB`.
* A published manifest, recipe, doc or script never references a build-host path.

## Tools: reuse, never recreate

Read `docs/bringup/agent-tools.md` before any GPU, benchmark, probe, gate, serve or performance
work; it holds the HARD RULE, working rules and final-report format.

* If a tool does the job, use it. If it almost does, extend it in place and keep the extension.
  A new script is justified only for a capability no existing tool can express; it goes in the
  matching subdirectory, `source`s `scripts/bench/plowbench.sh` for port choice, readiness poll,
  bench invocation and result parsing, and ships with its self-test.
* Never re-implement: GPU queueing (`scripts/bench/gpuq.py submit`, every GPU process, never a
  raw lease), leasing (`perf-data/tools/gpulease`, not on PATH), environment checks
  (`scripts/bench/plowbench-doctor.sh [assets] [objdir] [plowrt] [arch] [serve|block]`, CPU only,
  before any lease), serving bench/parsing (`plowbench.sh`), campaign
  build/bench/gate/compare/report/ledger (`scripts/campaign/campaign.py`), recipe validation
  (`check_recipe.py`), manifest packing and publishing (`plowrt pack|publish --recipe`),
  quantization (`perf-data/tools/quantize_*`).
* Shared Python helpers live beside their callers as a module (`recipe_common.py`,
  `campaign/rung_env.py`); the same helper copied into two scripts is a bug.

## Core Rules

### Think First

Before coding:

* Check assumptions.
* Choose simplest valid approach.
* Note important tradeoffs only.
* Give a brief design/plan before non-trivial changes.
* Skip plan for trivial changes.

### Keep Changes Minimal

* Implement only requested behavior.
* Touch only necessary files/code.
* No speculative features.
* No premature abstractions.
* No unrelated refactors or cleanup.
* Match existing patterns/style.
* Mention unrelated issues; don't fix them.

### Verify

* Define concrete success criteria.
* Reproduce bugs when practical.
* Run relevant tests/checks after changes.
* Fix failures caused by your changes.

## Code

* Prefer small, direct implementations.
* Reuse existing code/patterns when appropriate.
* Prefer Rust types/invariants over runtime checks.
* Check existing dependencies before adding crates.
* Add crates only when justified.
* Register every new `PLOW_*` knob, env read or `#if PLOW_*` define in `crates/devgen/src/knob_spec.rs`
  or `crates/plowrt/src/knob_spec.rs`; run `cargo test -p devgen --lib knob` and
  `cargo test -p plowrt --features cuda,hsa --lib knob`.

### Comments

* Minimize comments.
* Don't explain obvious code.
* Don't narrate implementation.
* Comment only non-obvious constraints, invariants, safety requirements, or reasoning.
* Prefer clear names/types over comments.
* Don't add doc comments unless useful or required by existing style.

## Performance

For `plowrt`:

* Performance > abstraction/convenience.
* Latency matters at microsecond scale.
* Avoid unnecessary allocations, copies, syscalls, locking, and indirection.
* Don't sacrifice performance for cleaner abstractions without reason.
* Measure when performance impact is uncertain.

## Communication

Default output must be compact.

* Short sentences.
* No filler or pleasantries.
* No restating request.
* No long explanations unless asked.
* No play-by-play narration.
* Don't explain obvious commands/code.
* Don't repeatedly summarize completed work.
* Prefer bullets over prose.
* Prefer `→`, `=`, `vs` when clearer.
* Report only decisions, important findings, blockers, changes, and verification.
* Ask questions only when the answer materially affects implementation.

### Final Response

Use this format when applicable:

* Changed: 1–3 bullets.
* Verified: tests/checks run.
* Notes: only blockers, risks, or required follow-up.

Omit empty sections.
