# Static binary + CUDA/HIP drivers

Goal: ship `plowrt` as a **static binary**, with the **only** dynamic
dependencies being the minimal CUDA / HIP driver libraries — and have both
vendors work in one process.

## Why the drivers can't be link-time static

`libcuda.so` / `libamdhip64.so` are the vendor kernel-mode driver interfaces;
they are shipped **only** as shared libraries and must match the installed
driver. You cannot statically link them. The clean resolution — used here — is
to **`dlopen` them at runtime** (`libloading`) instead of linking:

- no link-time `-lcuda` / `-lamdhip64`, so the build needs no CUDA/ROCm SDK and
  the binary has zero GPU link dependencies (verified: the default binary links
  only `libSystem`/`libc`);
- the driver is loaded lazily, and a host without a driver simply falls back to
  another backend instead of failing to start;
- see `src/device/cuda.rs` / `src/device/rocm.rs` — `new()` `dlopen`s the driver
  and calls `cuInit` / `hipInit`; the rest of the driver surface resolves the
  same way (signatures from `bindgen`).

## Can CUDA and HIP be compiled together? Yes.

Verified: `cargo build -p plowrt --features cuda,rocm` compiles both backends
into one binary. At runtime a heterogeneous instance `dlopen`s **both**
`libcuda` and `libamdhip64`; their symbol namespaces (`cu*` / `cuda*` vs `hip*`)
don't collide, so one process can drive an NVIDIA GPU and an AMD GPU
concurrently. (Use the **AMD** HIP runtime `libamdhip64`, not HIP-over-CUDA.)

## Build recipes

**Fully static, CPU-only (default features):** pure Rust, no C deps.

```sh
# fully static via musl
cargo build -p plowrt --release --target x86_64-unknown-linux-musl
# → a standalone binary with no dynamic dependencies
```

**GPU build (cuda / rocm):** `dlopen` needs the dynamic loader, so a *fully*
static musl binary can't load the drivers. Build against glibc with everything
else statically linked; the only runtime `.so`s are the loader + libc + the
`dlopen`ed driver(s):

```sh
RUSTFLAGS="-C target-feature=+crt-static" \
  cargo build -p plowrt --release --features cuda,rocm \
  --target x86_64-unknown-linux-gnu
# links libc + ld-linux; libcuda / libamdhip64 are dlopen'd at runtime
```

## Summary

| Build | Static? | Dynamic deps |
|-------|---------|--------------|
| CPU-only (musl) | fully static | none |
| GPU (glibc, crt-static) | static except loader/libc | ld-linux, libc, dlopen'd libcuda/libamdhip64 |

Both NVIDIA and AMD backends live in the same binary; whichever driver is present
at runtime is loaded.
