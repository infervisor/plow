# CLAUDE.md

## Context

* Plans/research live in `plans/` (gitignored).
* Read relevant plan before work. Update only when decisions change.

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
