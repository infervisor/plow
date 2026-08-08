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

## Security

Do not file public issues for vulnerabilities. See [SECURITY.md](SECURITY.md)
and email **lava@infervisor.ai**.

## License

Contributions are accepted under the [Apache License 2.0](LICENSE).
