# CLAUDE.md

## Context

* Plans/research live in `plans/` (gitignored).
* Read relevant plan before work. Update only when decisions change.

## Tools & Environment

Read `docs/bringup/agent-tools.md` before writing any benchmark, probe, or serve script.
`scripts/` has 251 scripts; most campaigns still wrote a throwaway probe and re-hit the same
environment failures on leased GPUs.

* Everything runs inside `nix develop`. `ROCM_PATH` unset means you are outside it.
* Not from nix, pass explicitly: vLLM client `/app/plow/build-gemma31/vllm-python` (built from
  source), its `VLLM_ROCM_LIB=/opt/rocm/core-7.14/lib`, and `gpulease`
  (`/app/plow/perf-data/tools/gpulease`, not on PATH).
* Before leasing a GPU: `nix develop --command scripts/bench/plowbench-doctor.sh <assets> <objdir>`.
  CPU only. It checks the shell, hazardous `PLOW_*` leftovers, binaries, packet hash, the object
  set (including the vendor `.co` kernels `build_gfx942.sh` does not emit), the queue, and disk.
* Main driver: `scripts/glm53_mi300x.sh emit|serve|bench|vllm|smoke`. The `serve` and `vllm`
  subcommands are the two halves of a comparison — same client, two base URLs.
* New probes `source scripts/bench/plowbench.sh` instead of re-implementing port choice, the
  readiness poll, the bench invocation, or result parsing.
* Every GPU process goes through the queue, never a raw lease.

## Core Rules

### Think First

Before coding:

* Check assumptions. Ask only when ambiguity blocks progress.
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
* Don't repeatedly summarize completed work.

## Code

* Prefer small, direct implementations.
* Reuse existing code/patterns when appropriate.
* Prefer Rust types/invariants over runtime checks.
* Check existing dependencies before adding crates.
* Add crates only when justified.
* Use **nix** `nix develop` for terminal/build tasks.
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
* Prefer bullets over prose.
* Prefer `→`, `=`, `vs` when clearer.
* Report only decisions, important findings, blockers, changes, and verification.
* Ask questions only when answer materially affects implementation.

### Final Response

Use this format when applicable:

* Changed: 1–3 bullets.
* Verified: tests/checks run.
* Notes: only blockers, risks, or required follow-up.

Omit empty sections.
