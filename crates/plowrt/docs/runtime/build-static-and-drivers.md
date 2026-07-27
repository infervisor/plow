# Static binary + GPU runtimes

Goal: ship `plowrt` as a **static binary**, with the **only** dynamic
dependencies being the minimal GPU runtime libraries — and have both vendors
work in one process.

The AMD side talks to **ROCr (HSA) directly, not HIP** (`src/device/hsa.rs`):
HIP is a thin layer over the same calls and plow needs nothing it adds, so the
runtime dependency is `libhsa-runtime64`, not `libamdhip64`.

## Why the runtimes can't be link-time static

`libcuda.so` / `libhsa-runtime64.so` are the vendor runtime interfaces; they are
shipped **only** as shared libraries and must match the installed driver. You
cannot statically link them. The clean resolution — used here — is to **`dlopen`
them at runtime** (`libloading`) instead of linking:

- no link-time `-lcuda` / `-lhsa-runtime64`, so the build needs no CUDA/ROCm SDK
  and the binary has zero GPU link dependencies (verified: the default binary
  links only `libSystem`/`libc`);
- the runtime is loaded lazily, and a host without it simply falls back to
  another backend instead of failing to start;
- see `src/device/cuda.rs` / `src/device/hsa.rs` — `new()` `dlopen`s the runtime
  and calls `cuInit` / `hsa_init`; the rest of the surface resolves the same way.

## Can CUDA and ROCr be compiled together?

The design intends yes: their symbol namespaces (`cu*` / `cuda*` vs `hsa_*`)
don't collide, so at runtime a heterogeneous instance can `dlopen` **both**
`libcuda` and `libhsa-runtime64` and drive an NVIDIA GPU and an AMD GPU
concurrently.

**Caveat (as of this writing): `cargo build -p plowrt --features cuda,rocm` does
not compile** — it fails with `E0603: function 'on' is private`, a pre-existing
visibility bug in an unrelated module that is only reachable when both GPU
features are on. Single-vendor builds (`--features cuda` or `--features rocm`)
are unaffected. The heterogeneous build needs that bug fixed before it works.

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
  cargo build -p plowrt --release --features cuda \   # or rocm; cuda,rocm together is currently broken (see above)
  --target x86_64-unknown-linux-gnu
# links libc + ld-linux; libcuda / libhsa-runtime64 are dlopen'd at runtime
```

## Summary

| Build | Static? | Dynamic deps |
|-------|---------|--------------|
| CPU-only (musl) | fully static | none |
| GPU (glibc, crt-static) | static except loader/libc | ld-linux, libc, dlopen'd libcuda / libhsa-runtime64 |

Both NVIDIA and AMD backends live in the same binary; whichever driver is present
at runtime is loaded.
