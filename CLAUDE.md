# CLAUDE.md

## Context

* Plans/research live in `plans/` (gitignored).
* Read relevant plan before work. Update only when decisions change.

## Tools & Environment

Read `docs/bringup/agent-tools.md` before writing any benchmark, probe, or serve script.
`scripts/` has 400+ scripts; most campaigns still wrote a throwaway probe and re-hit the same
environment failures on leased GPUs.

* Everything runs inside `nix develop`. `ROCM_PATH` unset means you are outside it. A hand-built
  box without nix sets `PLOW_CAMPAIGN_NO_NIX=1` instead of faking `ROCM_PATH`.
* Not from nix, pass explicitly: vLLM client `/app/plow/build-gemma31/vllm-python` (built from
  source; `plowbench.sh` reads `PB_VLLM`), its ROCm lib `/opt/rocm/core-7.14/lib`
  (`PB_VLLM_ROCM_LIB`, exported as `VLLM_ROCM_LIB`), and `gpulease` (`perf-data/tools/gpulease`
  in the checkout, not on PATH). The `/app/...` paths are one lab box's layout.
* Before leasing a GPU: `nix develop --command scripts/bench/plowbench-doctor.sh <assets> <objdir>`.
  CPU only. It checks the shell, hazardous `PLOW_*` leftovers, binaries, packet hash, the object
  set (including the vendor `.co` kernels `build_gfx942.sh` does not emit), the queue, and disk.
* GLM-5.3 driver: `scripts/glm53_mi300x.sh emit|serve|bench|vllm|smoke|stop`. The `serve` and
  `vllm` subcommands are the two halves of a comparison — same client, two base URLs.
* New probes `source scripts/bench/plowbench.sh` instead of re-implementing port choice, the
  readiness poll, the bench invocation, or result parsing.
* Every GPU process goes through the queue (`scripts/bench/gpuq.py submit`), never a raw lease.
* Final plow-vs-baseline performance comparisons: only `campaign.py report` output, in the strict
  12-row format of `docs/bringup/agent-tools.md` "Final performance report (strict)".

### Performance campaign evidence and promotion

* Keep raw benchmark logs, JSON, captures, generated reports and temporary CSVs in campaign
  scratch outside the repo. Do not scatter raw results under `docs/`, `scripts/` or `perf-data/`.
  Keep one serving-comparison CSV per campaign in its designated results directory; record only
  qualified wins and their evidence links in `comparison.md`.
* Before a full-model performance run, measure every affected prefill/decode rung at its actual
  shape, precision, flags and context. Check output accuracy, object resources (registers, spills,
  stack and shared memory), and measured time against a calibrated bandwidth/compute roofline.
  Resolve unexplained gaps; select the fastest correct native Plow kernel or segmented cuBLASLt
  route per rung, then verify block-level behavior. Full-model accuracy and serving comparisons
  validate the selected set; they do not replace the rung gates.
* Maintain one canonical production TOML per model/GPU/precision campaign under
  `recipes/infervisor`. Update that file in place only for qualified per-rung choices, including
  the actual compile flags, object route and runtime settings. Keep experimental recipes in the
  existing `scripts/campaign/recipes` area or scratch, not as extra production TOMLs.
* Reuse validated tuning evidence across commits when source and object hashes, compiler flags,
  kernel geometry, hardware and correctness match. Record the commit for traceability, but do not
  make identical commit hashes a qualification requirement.

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
