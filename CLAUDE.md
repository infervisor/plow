# CLAUDE.md

## Context

* Plans/research live in `plans/` (gitignored).
* Read relevant plan before work. Update only when decisions change.

## Repository layout (strict)

Put code where its kind already lives. Never create a new top-level directory, a new
`scripts/` subdirectory, or a parallel copy of an existing tool. Ask when nothing fits.

* `crates/` — the Rust workspace. `plowc` (compile/emit), `plowrt` (runtime, serve, dist),
  `plow-asset` (packet, manifest and recipe schemas), `devgen`, `hwspec`, `tunedb`, `costmodel`,
  `kernelcaps`, `packet`, `schedule`, `rewrite`, `nn-graph`, `lean_verify`. Runtime, compiler,
  packing and publishing behaviour lives in the crate that owns it, never in a script.
* `runtime/` — C/C++/CUDA/HIP kernel sources and CMake: `nvidia/`, `amd/`, `apple/`, `cpu/`,
  `common/`, `tests/`, `ubench/`, `bench/`. `include/packet.h` is the packet ABI.
* `scripts/` — operator tooling, one reusable capability per file, in its subdirectory:
  `bench/` (plowbench, doctor, gpuq, serving comparison, replays), `campaign/` (campaign.py,
  rung/roofline/evidence tools, experimental recipes under `campaign/recipes/`), `asr/`, `tts/`,
  `voice/`, `llm/`, `mm/` (per-modality gates and prep), `gen_kernels/`, `serve_test/`,
  `twoengine/`, `tests/`. Recipe tools are `check_recipe.py`, `render_recipe.py` and their
  shared `recipe_common.py`. A `test_<name>.py` sits beside `<name>`. No one-off probes,
  captures, dumps or throwaway drivers: those are scratch outside the repo.
* `recipes/infervisor/<model>/<label>.toml` + `.md` — the one canonical production recipe per
  model/GPU/precision. Experimental recipes stay in `scripts/campaign/recipes/`.
* `perf-data/` — serving-level evidence: `campaign/` CSVs, `tools/` (gpulease, bringup gates,
  quantizers, result validators), `packets/`, `probes/`, `block-configs/`.
  `perf-certs/` — checkpoint P certificates. `tuning/` — kernel-level measurement records keyed
  by hardware fingerprint, written only by a tuning run.
* `docs/` — `BUILD.md`, `DISTRIBUTION.md`, `arch/` (design; no tool, agent or model-vendor
  names), `bringup/` (`agent-tools.md` and campaign logs), `runtime/`, `schemas/`, `amd/`.
* `lean-plow/` — Lean 4 verification. `tools/` — the `bench` crate and `bench-api`.
  `assets/`, `media/` — images only.
* Raw logs, JSON, captures, generated reports and temporary CSVs go to campaign scratch outside
  the repo, never under `docs/`, `scripts/`, `perf-data/` or `recipes/`.

## Nix environment

* Every build, emit, serve, bench and test runs inside `nix develop`
  (`nix develop --command <cmd>`). The flake pins ROCm, CUDA, the Rust toolchain and Lean.
  `ROCM_PATH` unset means you are outside it: stop and re-enter, do not work around it.
* Never fake `ROCM_PATH`, hardcode a `/nix/store` path, or add a toolchain or Python dependency
  outside `flake.nix`. A hand-built box without nix sets `PLOW_CAMPAIGN_NO_NIX=1`; that is the
  only sanctioned exception.
* A published manifest, recipe, doc or script never references a build-host path.

## Tools: reuse, never recreate

Read `docs/bringup/agent-tools.md` and `scripts/campaign/README.md` before writing any
benchmark, probe, gate or serve script. `scripts/` holds hundreds of scripts because campaigns
kept writing throwaway probes and re-hitting the same environment failures on leased GPUs.

* If a tool does the job, use it. If it almost does, extend it in place and keep the extension.
  A new script is justified only for a capability no existing tool can express; it goes in the
  matching subdirectory, `source`s `scripts/bench/plowbench.sh` for port choice, readiness poll,
  bench invocation and result parsing, and ships with its self-test.
* Never re-implement: GPU queueing (`scripts/bench/gpuq.py submit`, every GPU process, never a
  raw lease), leasing (`perf-data/tools/gpulease`, not on PATH), environment checks
  (`scripts/bench/plowbench-doctor.sh <assets> <objdir>`, CPU only, before any lease), serving
  bench/parsing (`plowbench.sh`), campaign build/bench/compare/ledger
  (`scripts/campaign/campaign.py`), recipe validation (`check_recipe.py`), manifest packing
  and publishing (`plowrt pack|publish --recipe`), quantization (`perf-data/tools/quantize_*`).
* Shared Python helpers live beside their callers as a module (`recipe_common.py`,
  `campaign/rung_env.py`); the same helper copied into two scripts is a bug.
* Not from nix, pass explicitly: vLLM client `/app/plow/build-gemma31/vllm-python` (built from
  source; `plowbench.sh` reads `PB_VLLM`), its ROCm lib `/opt/rocm/core-7.14/lib`
  (`PB_VLLM_ROCM_LIB`, exported as `VLLM_ROCM_LIB`). The `/app/...` paths are one lab box's
  layout.
* GLM-5.3 driver: `scripts/glm53_mi300x.sh emit|serve|bench|vllm|smoke|stop`. The `serve` and
  `vllm` subcommands are the two halves of a comparison — same client, two base URLs.
* Final plow-vs-baseline performance comparisons: only `campaign.py report` output, in the strict
  12-row format of `docs/bringup/agent-tools.md` "Final performance report (strict)".

### Performance campaign evidence and promotion

* Keep raw benchmark logs, JSON, captures, generated reports and temporary CSVs in campaign
  scratch outside the repo. Do not scatter raw results under `docs/`, `scripts/` or `perf-data/`.
  Keep one serving-comparison CSV per campaign in its designated results directory; record only
  qualified wins and their evidence links in `comparison.md`.
* A new model starts from the full plowrt route set (template
  `scripts/campaign/recipes/templates/llm-sm90a-default-on.toml`); the campaign measures removing
  routes, never discovers them. `plowbench-doctor.sh` fails on a route the packet lacks unless the
  recipe's `[capabilities]` table waives it with the measured reason.
* Qualify numerics at c1 and at the serving concurrencies (`concurrency = [1, 8, 32]`, same-
  concurrency vLLM peers); the gate uses trimmed mean / p95 / over-limit count, p99 is informational.
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
