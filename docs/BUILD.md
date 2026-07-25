# Building plow on Linux

Everything is driven by the flake. `nix build .#<pkg>` on `x86_64-linux` or
`aarch64-linux`.

| `nix build .#…` | Artifact                | Notes                                 |
| --------------- | ----------------------- | ------------------------------------- |
| `plowrt`        | `bin/plowrt`            | one binary, CPU **and** GPU (default) |
| `plowc`         | `bin/plowc`             | compiler; pure Rust, no system libs   |
| `plow-runtime`  | `lib/libplow_runtime.a` | C core + CPU golden kernels           |

`nix flake check` builds all three (the C core runs its `ctest` suite as part of
its build). `nix develop` gives you the same toolchain for plain `cargo`.

## The nn-graph dependency

`nn-graph` (`github:infervisor/nn-graph`) is a separate repo. The workspace
declares it as a cargo `git` dependency pointing at
`https://github.com/infervisor/nn-graph.git`.

Cargo fetches it directly; no vendor directory or flake input is needed.

## One binary for CPU and GPU

`plowrt` is a single artifact that serves both. Both vendor backends are compiled
in, but **neither is linked**: `device::select` `dlopen`s the driver at startup
and degrades gracefully.

```
dlopen libcuda.so.1     → ok   → CUDA backend (accelerated)
     ↓ fails
dlopen libamdhip64.so   → ok   → ROCm backend (accelerated)
     ↓ fails
CPU reference backend          → interprets the same programs, unaccelerated
```

On a host with no driver at all:

```
WARN plowrt::device: no CUDA backend e=dlopen libcuda: libcuda.so: cannot open shared object file
WARN plowrt::device: no ROCm backend e=dlopen libamdhip64: libamdhip64.so.6: cannot open shared object file
INFO plowrt: backend ready class=Cpu accelerated=false
```

Assets stay servable either way. Every asset is compiled for *some* GPU spec —
`plowc --gpu` defaults to `H100 SXM5`, and `hwspec::Vendor` is only `Nvidia` or
`Amd`, so a "CPU asset" does not exist — and the CPU backend is the reference
interpreter for exactly those programs. A missing driver therefore costs you
acceleration, not availability; `plowrt` logs a warning per asset naming the GPU
it was compiled for.

Because it `dlopen`s, the binary is **dynamically linked**, and this is not a
choice we can undo: a `crt-static` binary cannot `dlopen` at all (musl stubs it
out in static builds; glibc does not support it either), so a fully-static
`plowrt` would skip the GPU on *every* host and the CUDA/ROCm code would be dead
weight. **"Fully static" and "uses a GPU" cannot both be true of one artifact.**

What it links is only libc, libm and libgcc_s:

```
$ readelf -d result/bin/plowrt | grep NEEDED
 (NEEDED)  Shared library: [libgcc_s.so.1]
 (NEEDED)  Shared library: [libm.so.6]
 (NEEDED)  Shared library: [libc.so.6]
```

No `libcuda`, no `libamdhip64` — every Rust crate is statically linked in, so it
is still one self-contained ~2.7 MB file, and one process can drive NVIDIA and
AMD side by side (`cu*` and `hip*` share no symbol names). It just inherits a
glibc version floor.

A fully static CPU-only `plowrt` is still possible if you want one — drop the GPU
features (so nothing needs `dlopen`) and target musl. It was verified to produce a
`static-pie` binary whose entire nix closure is a single store path:

```
cargo build --release -p plowrt --target x86_64-unknown-linux-musl   # note: no --features
```

To get it from the flake, add a package built with `pkgs.pkgsStatic.rustPlatform`
and no `buildFeatures`. It is deliberately not a default output, because it can
never use a GPU.

### Performance

The release profile is tuned for `plowrt` (see the workspace `Cargo.toml`):
`lto = "fat"` and `codegen-units = 1` let the decode → dispatch → counter path
inline end to end, and `panic = "abort"` drops the unwind tables and landing pads
from it. To tune for one specific machine — *not* the default, since the result
will `SIGILL` on an older host:

```
RUSTFLAGS="-C target-cpu=native" cargo build --release -p plowrt
```

## TLS is pure Rust

The workspace pins hf-hub with `default-features = false, features = ["ureq"]`,
which drops its `default-tls` (native-tls → **openssl-sys**) and leaves the
blocking `api::sync` client on ureq's rustls backend. Nothing in the tree links a
system TLS library, so no artifact needs `pkg-config` or openssl at build time.

## The GPU kernel plugins

Separately from the driver `dlopen` above, `runtime/` builds the kernel backends
as **shared plugins** — not static libs, since nothing links them:

```
cmake -S runtime -B build -DPLOW_CUDA=ON      # → libplow_rt_cuda.so, sm_90a/100a/120a
cmake -S runtime -B build -DPLOW_ROCM=ON      # → libplow_rt_rocm.so, gfx942
cmake --build build
```

Each exports one registration entry point per arch —
`plow_register_cuda_hopper`, `plow_register_cuda_blackwell`,
`plow_register_cuda_rtx6000`, `plow_register_rocm_mi300` — which fills a
`dispatch_table` (a flat array of function pointers, so it crosses the `.so`
boundary as plain data). Each plugin absorbs the PIC C core statically, so it is
self-contained apart from its vendor runtime.

They are not part of `nix build`: the CUDA and ROCm toolchains are unfree and are
only needed on a machine that has the hardware.

> Note: nothing in-tree calls `plow_register_*` yet. The plugins build and export
> the ABI, but the host-side loader that `dlopen`s one and selects the per-arch
> table is not written — `plowrt`'s `dlopen` today is of the *driver*, not of
> these plugins.
