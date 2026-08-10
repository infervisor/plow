<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/plow_mark.svg">
    <img src="assets/plow_mark_light.svg" alt="plow" width="96">
  </picture>
</p>

# plow

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-sm__120-76B900.svg?logo=nvidia&logoColor=white)](#fully-supported-today)
[![ROCm](https://img.shields.io/badge/ROCm-gfx950-ED1C24.svg)](#fully-supported-today)
[![Nix](https://img.shields.io/badge/Nix-flakes-5277C3.svg?logo=nixos&logoColor=white)](https://nixos.org/)
[![Lean 4](https://img.shields.io/badge/Lean-4-000000.svg)](lean-plow/)
[![Infervisor](https://img.shields.io/badge/by-Infervisor-111111.svg)](https://infervisor.ai)

**Packet Language for On-device Warps** — an LLM inference stack from
[Infervisor](https://infervisor.ai).

plow compiles a Hugging Face checkpoint into a static **packet stream**, then
runs it with a **persistent on-device interpreter**: one cooperative GPU launch
that stays resident, executes ops at warp granularity, and coordinates with
counters instead of per-op CPU dispatch.

| Component | Role |
|-----------|------|
| `plowc` | AOT compiler (checkpoint → `.pkt` + weight sidecars) |
| `plowrt` | Host runtime + OpenAI-compatible HTTP server |
| `runtime/` | CUDA / HSA persistent interpreters |
| `lean-plow/` | Lean 4 checks for rewrites and the counter protocol |

## Architecture

Architecture chapters: [`docs/arch/`](docs/arch/00-overview.md) — compiler
pipeline, tile graph, scheduler, packet ABI, counter system, runtime, cost
model, formal verification, multi-GPU. Build-system rationale:
[`docs/BUILD.md`](docs/BUILD.md). Every emit/build/runtime flag:
[`docs/flags-reference.md`](docs/flags-reference.md).

Bringing up a new model: a staged playbook — operator IR → rewrite rules →
formal verification → kernel tuning → single-block sweep → runtime
optimization → measured campaign — lives in
[`docs/bringup/`](docs/bringup/00-overview.md), with per-stage LLM-agent
prompt templates in [`docs/bringup/agents/`](docs/bringup/agents/README.md).

## Fully supported today

These paths are the ones exercised end-to-end for serving. Start here.

| GPU | Arch | SMs / CUs | VRAM | `plowc` flags | env |
|-----|------|-----------|------|---------------|-----|
| **NVIDIA RTX 5090** | `sm_120` | 170 SMs | 32 GB GDDR7 | `--gpu rtx5090 --max-ctx 8192` | `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=8` |
| **NVIDIA RTX PRO 6000 Blackwell** | `sm_120` | 188 SMs | 96 GB GDDR7 | `--gpu rtx6000pro --max-ctx 131072` | `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=8` |
| **AMD Instinct MI350X / MI355X** | `gfx950` | 256 CUs | 288 GB HBM3E | `--arch gfx950 --gpu mi350x`\|`mi355x` `--max-ctx 131072` | — |
| **AMD Instinct MI300X** | `gfx942` | 304 CUs | 192 GB HBM3 | `--arch gfx942 --gpu mi300x --max-ctx 131072` | `PLOW_FP8=1 PLOW_W8A8=1 PLOW_FP8_HEAD=1 PLOW_FUSE_HNR=1` |

`--gpu` sets the SM/CU count from the built-in GPU registry (`--n-cu`
overrides it for unknown or partitioned parts). `PLOW_UNISEG=1` is
**NVIDIA-only** — on AMD it collapses wave-class segments and breaks prefill.
An emit is **not** portable across SM counts: emitting for the 188-SM 6000 Pro
and running on a 170-SM 5090 (or the reverse) mis-schedules work.

| Model (first-run) | HF id | Notes |
|-------------------|-------|--------|
| **Gemma-4 12B Instruct** | `google/gemma-4-12B-it` | Dense bf16 primary path below |
| Gemma-4 31B Instruct | `google/gemma-4-31B-it` | Same recipe; needs more VRAM / shorter ctx on 5090 |
| Gemma-4 26B-A4B MoE | `google/gemma-4-26B-A4B-it` | MoE emit + serve on the same GPUs |

Also emit-capable (not the first-run walkthrough): Qwen3, Llama-3.1; bf16 and
weight-only fp8 (e4m3). Descriptors exist for other parts (H100, B200) — do
**not** treat those as drop-in substitutes without matching interpreter
objects.

## Requirements

- **Nix with flakes.** The flake provides everything — Rust, CMake, Lean, and
  the CUDA and ROCm toolchains. Kernel and interpreter builds need no system
  toolchain: no `/opt/rocm`, no `/usr/local/cuda`.
- **A GPU driver at runtime.** NVIDIA: `libcuda.so.1` from the driver. AMD: the
  `amdgpu` kernel driver — user-space ROCr and its libraries come from nix.
- **A local Hugging Face checkpoint directory** for the model you serve. The
  walkthrough uses `google/gemma-4-12B-it` at `$HOME/models/gemma-4-12B-it`.

`plowrt` does not link CUDA/HIP. Features `cuda` / `hsa` `dlopen` the drivers at
runtime.

## Quickstart

All commands run inside `nix develop` (or via `nix develop --command …`).

### 1. Build the host tools

```bash
cargo build --release -p plowc
cargo build --release -p plowrt --features cuda,hsa
```

Binaries: `./target/release/plowc`, `./target/release/plowrt`.
Optional: `cargo test --workspace` and `(cd lean-plow && lake build)`.

### 2. Build the interpreter objects

One command per target — the dev shell provides `nvcc`/`hipcc` from nix:

```bash
ASSETS="$HOME/plow-assets/gemma4-12b"; mkdir -p "$ASSETS"

# NVIDIA sm_120 (5090 / 6000 Pro) — cubins land next to the given path
scripts/build_sm120_cubin.sh "$ASSETS/interp_sm120.cubin" -DPLOW_NV_FA_GF_FULL=4

# AMD gfx950 (MI350X / MI355X)
scripts/build_gfx950.sh build-amd/hsaco
ln -sfn "$(pwd)/build-amd/hsaco" "$ASSETS/hsaco"

# AMD gfx942 (MI300X) — PLOW_OCC4=1 is the batch-1 occupancy profile
PLOW_OCC4=1 PLOW_L2HIER=1 bash scripts/build_gfx942.sh build-amd/hsaco/gfx942
ln -sfn "$(pwd)/build-amd/hsaco/gfx942" "$ASSETS/hsaco"
```

Hermetic alternative: `nix build .#plow-interp-sm120a` / `.#plow-interp-gfx950`
/ `.#plow-interp-gfx942` (objects in `result/cubin/` and
`result/hsaco/<arch>/`).

### 3. Compile the packet

One command; take the flags and env for your GPU from the
[support table](#fully-supported-today):

```bash
CKPT="$HOME/models/gemma-4-12B-it"

# example: MI355X
./target/release/plowc --hf-dir "$CKPT" \
  --arch gfx950 --gpu mi355x --max-ctx 131072 \
  --out "$ASSETS"
```

This writes `model.pkt` + `weights.json` into `$ASSETS` and symlinks
`checkpoint` → `$CKPT` and `tokenizer.json`. The interpreter objects from
step 2 must already be in place.

Sanity check on AMD: a correct Gemma-4 dense emit reports **121 segments per
prefill bucket** in `build.json` (`2·layers + 1`).

On gfx942, the fp8 env selects per-channel fp8 serving (weights *and*
activations); `scripts/quantize_fp8_head.py` builds the quantized lm_head
shard. MX-FP4 stays gfx950-only — CDNA3 has no fp4 hardware and `plowc`
refuses it at emit.

### 4. Serve and chat

```bash
./target/release/plowrt serve --assets "$ASSETS" --port 8080
```

```bash
curl -s http://127.0.0.1:8080/v1/models | jq .

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gemma-4-12b-it",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "max_tokens": 64
  }' | jq .
```

`plowrt serve` after a successful load — Gemma-4 12B (~22 GiB weights) on an
NVIDIA GH200, OpenAI-compatible API on TCP:

![plowrt serving Gemma-4 12B on NVIDIA GH200](media/plowrt-gemma4-12b-gh200.png)

Measured campaigns and their protocols live in
[`perf-data/plow-gfx942/`](perf-data/plow-gfx942/).

## Asset layout (what `serve` expects)

After the quickstart, `$ASSETS` contains at least:

```
model.pkt
weights.json          # "network": "gemma-4-12b-it"
tokenizer.json -> …   # symlink into the HF dir
checkpoint -> …       # symlink to the HF dir (weights mmap’d / uploaded)
# NVIDIA:
interp_sm120.cubin
interp_sm120_pf.cubin
# AMD:
hsaco/ -> …           # interp_decode.elf, interp_prefill.elf, interp_flash.elf, …
```

The API slug is the lowercased checkpoint directory name (e.g.
`gemma-4-12b-it`); `plowc` and `/v1/models` both derive it this way. Confirm it
before curling:

```bash
jq -r .network "$ASSETS/weights.json"
# → gemma-4-12b-it
```

Streaming: `"stream": true`. Also `/healthz`, `/metrics`. Multiple
`--assets DIR` register more models.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Report vulnerabilities privately — see [SECURITY.md](SECURITY.md).

## License

Copyright 2026 Infervisor.

Licensed under the [Apache License, Version 2.0](LICENSE).

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
Report unacceptable behavior to **lava@infervisor.ai**.
