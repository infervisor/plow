<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/plow_mark.svg">
    <img src="assets/plow_mark_light.svg" alt="plow" width="96">
  </picture>
</p>

# plow

**An open-source inference engine.**

Plow compiles models ahead of time and runs them on your hardware. It includes
an OpenAI-compatible API, streaming responses, and backends for NVIDIA GPUs,
AMD GPUs, Apple Silicon, and CPUs.

Built by [Infervisor](https://infervisor.ai). Licensed under [Apache 2.0](LICENSE).

[Get started](#get-started) · [Build from source](#build-from-source) ·
[Configuration](docs/flags-reference.md) · [Architecture](docs/arch/00-overview.md)

## Get started

You need a compatible model build, enough memory, and the model's checkpoint.
GPU serving also requires the vendor's driver. Check the
[model catalog](https://infervisor.ai/models) for available model and hardware
combinations.

Install the runtime:

```bash
curl -fsSL https://pdist.infervisor.ai/install.sh | sh
```

Choose a model from the catalog and provide its local checkpoint. For example,
on hardware with a compatible Kimi-K3 build:

```bash
plowrt show kimi-k3
plowrt load kimi-k3 --checkpoint "$HOME/models/Kimi-K3"
plowrt serve --model kimi-k3 --bind 127.0.0.1 --port 8080
```

`show` lists builds and their hardware requirements. `load` downloads the
matching compiled assets; model weights come from your checkpoint. `serve`
uses the local assets without downloading anything.

See [model distribution](docs/DISTRIBUTION.md) for downloads, upgrades, and
offline use.

### Send a request

Get the served model ID:

```bash
curl http://127.0.0.1:8080/v1/models
```

Use that ID in a chat request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model-id>",
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "max_tokens": 128
  }'
```

Add `"stream": true` for streaming. OpenAI-compatible clients can use
`http://127.0.0.1:8080/v1` as their base URL.

## How it works

```text
Model checkpoint → plowc → compiled packet + weights → plowrt → API
```

- **`plowc`** compiles a model into a packet: a preplanned sequence of operations
  for a hardware target.
- **`plowrt`** loads the packet and weights, schedules requests, and serves the API.
- **`runtime/`** contains the execution kernels. Persistent GPU interpreters run
  packet operations on-device to reduce CPU dispatch overhead.

Model, precision, and operator coverage vary by backend. Use a matching
[recipe](recipes/infervisor/) when building assets for your hardware.

## Build from source

Install [Nix](https://nixos.org/download/) with flakes enabled, then enter the
development environment:

```bash
git clone https://github.com/infervisor/plow.git
cd plow
nix develop
```

Build the compiler and the runtime for your platform inside that shell:

```bash
cargo build --release -p plowc

# Linux: NVIDIA and AMD
cargo build --release -p plowrt --features cuda,hsa

# macOS: Apple Silicon
cargo build --release -p plowrt --features metal

# CPU only
cargo build --release -p plowrt --no-default-features --features cpu
```

Binaries are in `target/release/`. Follow a matching
[model recipe](recipes/infervisor/) to compile the packet, prepare weights,
and build any required GPU objects. Then serve the resulting asset directory:

```bash
target/release/plowrt serve --assets /path/to/assets --bind 127.0.0.1 --port 8080
```

See the [build guide](docs/BUILD.md) for Nix packages and release builds.

## Documentation

| Guide | Contents |
| --- | --- |
| [Configuration](docs/flags-reference.md) | Common flags and advanced tuning |
| [Model distribution](docs/DISTRIBUTION.md) | Install, load, and manage model builds |
| [Model recipes](recipes/infervisor/) | Model-specific build and runtime settings |
| [Speech recognition](docs/runtime/asr.md) | Transcription and streaming audio |
| [Text to speech](docs/runtime/tts.md) | Speech generation |
| [CPU execution](docs/runtime/cpu.md) | CPU setup and tuning |
| [Architecture](docs/arch/00-overview.md) | Compiler, packets, scheduling, and runtime |
| [Model bring-up](docs/bringup/00-overview.md) | Add and validate a model |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) to get started and the
[Code of Conduct](CODE_OF_CONDUCT.md) for community guidelines.
Report vulnerabilities privately using [SECURITY.md](SECURITY.md).
