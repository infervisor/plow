# Contributing

Thanks for contributing to plow. This project is maintained by
[Infervisor](https://infervisor.ai).

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Prerequisites

- [Nix](https://nixos.org/download/) with flakes enabled
- Enter the toolchain shell from the repo root: `nix develop`
- Optional GPU toolchains:
  - NVIDIA: CUDA ≥ 12.9 (`nvcc`) to build interpreter cubins
  - AMD: ROCm ≥ 7.2.4 under `/opt/rocm` to build hsaco objects

CPU-only Rust work does not require a GPU toolkit. Interpreter objects and
`plowrt` GPU paths do.

## Build and test

```bash
nix develop
cargo build --workspace --release
cargo test  --workspace
```

GPU server binary (drivers loaded at runtime via `dlopen`; `cuda` / `hsa`
already pull the tokenizer):

```bash
cargo build --release -p plowrt --features cuda,hsa
```

Optional `hub` additionally enables Hugging Face hub downloads.

Lean:

```bash
(cd lean-plow && lake build)
```

Interpreter objects:

```bash
scripts/build_sm120_cubin.sh <out-dir>
cmake -S runtime -B build-amd -DPLOW_GFX950_HSACO=ON && cmake --build build-amd -j
```

## Pull requests

- Keep changes focused: one problem per PR when practical.
- Match existing style and patterns in the crates you touch.
- Add or update tests for behavioral changes.
- Do not commit model weights, tokenizer dumps, `.cubin` / hsaco objects,
  secrets, or local asset dirs (`plow-out/`, `assets/`, … — gitignored on
  purpose).
- Do not commit the local research scratchpad (`plans/`).
- Prefer short, clear PR descriptions: what changed and why.

## LLM-assisted contributions

Much of this codebase was written with LLM coding agents, and we expect
contributions to be too. The repository ships a bring-up agent harness in
[`docs/bringup/`](docs/bringup/) precisely so that agent-driven work has a
defined shape. Using one is welcome and does not need to be hidden.

What does not change is accountability:

- **A named human is accountable for every merge to `main`.** Not the agent
  that wrote the patch, and not the agent that reviewed it. If it lands and it
  breaks, a person owns that.
- **The submitter is accountable for the contents of their PR** — they must
  understand the change well enough to defend it in review and to fix it when
  it fails. "The agent wrote it" is not an explanation of why a change is
  correct.
- **Gates are evidence, not decoration.** Where a change claims a gate passed,
  paste what the gate actually printed. An agent that cannot pass a gate is
  required to stop and report rather than weaken it; the same rule applies to
  the human sending the PR.
- **Measurements need provenance.** Any performance claim must name the
  hardware, the harness, and enough method to be re-run — see
  [`docs/bringup/07-perf-campaign.md`](docs/bringup/07-perf-campaign.md). A
  number an agent produced but nobody can reproduce is not a result.
- **Proofs must be real.** `sorry` in a Lean obligation fails review, whoever
  or whatever wrote it.

Disclosure of tooling is optional; a `Co-authored-by:` trailer is fine if you
want it. Review is applied to the change, not to how it was produced.

## Security

Do not file public issues for vulnerabilities. See [SECURITY.md](SECURITY.md)
and email **lava@infervisor.ai**.

## License

Contributions are accepted under the [Apache License 2.0](LICENSE).
