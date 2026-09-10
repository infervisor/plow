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

curl -s http://127.0.0.1:8080/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"gemma-4-12b-it","prompt":"The capital of France is","max_tokens":32}' \
  | jq .
```

`plowrt serve` after a successful load — Gemma-4 12B (~22 GiB weights) on an
NVIDIA GH200, OpenAI-compatible API on TCP:

![plowrt serving Gemma-4 12B on NVIDIA GH200](media/plowrt-gemma4-12b-gh200.png)

Measured campaigns and their protocols live in
[`perf-data/plow-gfx942/`](perf-data/plow-gfx942/).
The full Kimi-K3 TP8/MI325X build and serving recipe is
[`docs/amd/kimi-k3-mi325x.md`](docs/amd/kimi-k3-mi325x.md).

## Getting a model without building one

The quickstart above compiles from source. A released `plowrt` can instead fetch
a build that already exists — it probes the machine, picks the variant that fits,
and fetches only what is missing:

```bash
plowrt load  kimi-k3 --checkpoint "$HOME/models/Kimi-K3"
plowrt serve --model kimi-k3 --port 8080
```

`plowrt show <model>` lists every published variant with the reason each one does
or does not fit this box. Weights are never distributed: they come from your
HuggingFace snapshot or a local directory. `serve` performs no network I/O — the
CLI fetches, in a separate process.

Full contract, including how assets are produced and released:
[`docs/DISTRIBUTION.md`](docs/DISTRIBUTION.md).

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

## CPU-only (no GPU)

Same three steps, minus the interpreter objects — the CPU backend interprets the
packet directly, so **step 2 is skipped entirely** and nothing links CUDA or HSA:

```bash
# 1. Build. --no-default-features drops the GPU backends.
cargo build --release -p plowc
cargo build --release -p plowrt --no-default-features --features cpu

# 2. Compile the packet. plowc always emits for a device target; pick an NVIDIA
#    one — an AMD gfx942/gfx950 packet does NOT load on the CPU backend.
CKPT="$HOME/models/gemma-4-12B-it"
./target/release/plowc --hf-dir "$CKPT" \
  --gpu rtx6000pro --n-cu 96 --max-ctx 2048 \
  --batch 1,4 --seq 128,512 --out "$ASSETS"

# 3. Serve. --rt-checkpoint is required; the bundle carries no weights.
./target/release/plowrt serve --assets "$ASSETS" --rt-checkpoint "$CKPT" \
  --cpu-isa avx512 --cpu-numa auto
```

`--n-cu` sets the packet's *virtual* executor count, not a thread count: the
worker pool maps any number of threads onto it, so one bundle serves any core
count. Fewer threads than executors is fine; more is wasteful, so the automatic
worker width caps itself at `--n-cu`. Compile once at a width that divides the
largest machine you intend to serve (the emitter caps it at 256).

Runtime knobs are all `--cpu-*` (`--cpu-threads`, `--cpu-numa`, `--cpu-isa`,
`--cpu-spin-us`, …), each with a `PLOW_CPU_*` env twin — full table in
[CPU execution](docs/runtime/cpu.md#flags), which also covers AVX-512/AMX
coverage, NUMA policy, and the two A/B knobs that are off because they measured
slower.

## Apple Silicon (Metal + ANE)

macOS on Apple Silicon runs the packet on three units: a persistent Metal
interpreter on the GPU, the NEON CPU tier, and — opt-in — the Neural Engine
through CoreML. There are no interpreter objects to build: the Metal shader is
compiled at load, so **step 2 of the quickstart is skipped**, exactly as on the
CPU backend.

| Chip | `--gpu` | GPU cores | Notes |
|------|---------|-----------|-------|
| **Apple M4 Pro** | `m4pro` | 16 | The part every number below was measured on |
| Apple M4 | `m4` | 10 | |
| Apple M4 Max | `m4max` | 32 / 40 | `--n-cu` overrides the registry for the 40-core SKU |

`--gpu` fixes the packet's executor count, and an emit is **not** portable
across core counts. `--arch` is inferred (`metal3`); do not pass an
NVIDIA/AMD arch.

### What runs

Each row is checked instruction-by-instruction against the CPU golden tier by
`examples/apple_lockstep` (below) — zero mismatching outputs on prefill and
decode:

| Model | Precision | `plowc` flags | Twin |
|-------|-----------|---------------|------|
| Llama-3.2-3B | bf16 | — | — |
| Llama-3.2-3B | w8a16 (fp8 weights) | `--fp8 --w8a16` | fp8 |
| Llama-3.2-3B | w8a8 | `--fp8 --w8a8` | fp8 |
| Llama-3.2-3B | w8a16 + fp8 KV cache | `--fp8 --w8a16 --fp8-kv` | fp8 |
| Llama-3.2-3B | MXFP4 (w4a16) | `--mxfp4` | mxfp4 |
| Gemma-4-E4B | w8a16 + fp8 head | `--fp8 --w8a16 --fp8-head` | fp8 |
| Gemma-4-E4B | MXFP4 | `--mxfp4` | mxfp4 |
| GPT-OSS-20B | MXFP4 MoE | — (the checkpoint is already fp4) | — |

Qwen3 and Gemma-4 12B/26B-A4B emit for `metal3` on the same recipe. The
Metal interpreter implements 50 of the 156 device opcodes — the dense +
GQA-attention + fp8/mxfp4 + GPT-OSS-MoE set. MLA, KDA and the Gemma MoE
families emit but have no Metal kernels yet, so they refuse at load rather
than run wrong.

### 1. Build

```bash
cargo build --release -p plowc
cargo build --release -p plowrt --features metal        # GPU + CPU
cargo build --release -p plowrt --features ane          # + Neural Engine
```

`metal` implies `cpu` (unified memory: the host tensors *are* the device
buffers), and `ane` implies `metal`. Nothing links CUDA or HSA.

### 2. Quantized weight twins (optional)

A twin is a second safetensors file holding quantized copies of the dense
projections; the packet names them and the runtime mmaps them next to the
bf16 checkpoint.

```bash
CKPT="$HOME/models/Llama-3.2-3B-Instruct"

# fp8 (e4m3, per-output-channel) — pure Rust, no torch
cargo run --release --features cpu --example quantize_fp8 -- "$CKPT" "$HOME/plow-assets/llama3b-fp8" "model."

# MXFP4 (e2m1 + E8M0 block scales) — needs the torch shell
nix develop .#quantize -c python3 perf-data/tools/quantize_mxfp4.py "$CKPT" "$HOME/plow-assets/llama3b-mx4" "model."
```

### 3. Compile the packet

```bash
ASSETS="$HOME/plow-assets/llama3b"; mkdir -p "$ASSETS"

PLOW_FA_GF_FULL=1 ./target/release/plowc \
  --hf-dir "$CKPT" --gpu m4pro --max-ctx 4096 --fp8 --w8a16 --out "$ASSETS"
```

`PLOW_FA_GF_FULL=1` is Llama's GQA-3 flash grouping (no CLI twin). Gemma-4-E
adds `--fp8-head`; MXFP4 replaces `--fp8 --w8a16` with `--mxfp4`. `--fp8-kv`
halves the KV cache but is lossy — greedy generation diverges after ~20
tokens, so treat it as a memory lever, not a free one.

> Re-emit after pulling: device opcodes are renumbered when branches collide,
> and a stale `model.pkt` dispatches to the wrong kernel rather than failing
> cleanly.

### 4. Run

```bash
# One-shot chat with per-step timings
PLOW_FP8_DIR="$HOME/plow-assets/llama3b-fp8" \
  ./target/release/examples/apple_chat "$ASSETS/model.pkt" "$CKPT" \
  --tokens 100 --chat "Explain in two sentences why the sky is blue."

# OpenAI-compatible server (Metal is the default on a metal/ane build)
./target/release/plowrt serve --assets "$ASSETS" --port 8080 \
  --fp8-dir "$HOME/plow-assets/llama3b-fp8"
```

`PLOW_BACKEND=cpu` (or `--apple-backend cpu`) runs the same bundle on the NEON
CPU tier instead — useful for A/B and for the golden reference.

### 5. Verify

```bash
# Every instruction on GPU and CPU, operand-for-operand
PLOW_FP8_DIR="$HOME/plow-assets/llama3b-fp8" \
  ./target/release/examples/apple_lockstep "$ASSETS/model.pkt" "$CKPT" \
  --prompt "The history of computation"
# => "0 mismatching outputs" for prefill and decode

# GEMM microbenchmark, golden-checked on the model's real shapes
./target/release/examples/metal_gemm_bench "$ASSETS/model.pkt" "$CKPT" --check
```

### Neural Engine (experimental)

The ANE is **off by default** and never improves a dense decode; it is a
prefill-only lever. Two mechanisms exist:

* **Row split** — `PLOW_ROW_SPLIT=ane=50,cpu=10` (or `plowc --unit-shares
  gpu:5,ane:4`) cuts every prefill bucket into GPU/ANE/CPU row blocks and
  writes a `hetero.json` sidecar; `examples/apple_prefill_calibrate` sweeps the grid and keeps a split
  only when it beats GPU-only by >5% with the same first token.
* **Channel MLP** — `PLOW_ANE_MLP_CHANNELS=<n>` at emit plus `--ane-mlp` at
  run splits each MLP by intermediate channel. Requires the `ane` feature and
  a placement probe; it refuses to combine with row split, w8a8 or serial mode,
  and falls back to unsplit GPU if the graph is rejected.

CoreML compiles one program per (layer, kind, row count) and caches them under
`<assets>/ane/`; the first prefill at a new shape pays that compile.

Runtime knobs are `--ane-*` / `--apple-*` with `PLOW_*` env twins (`plowrt
serve --help`, "Apple runtime" heading). Design notes and measurements:
[`plans/apple-silicon-backend.md`](plans/apple-silicon-backend.md) and
[`plans/apple-heterogeneous-emit.md`](plans/apple-heterogeneous-emit.md).

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
