# Kimi-K3 DSpark non-causal MLA bring-up

Status: semantic oracle passes; not yet a production interpreter arm.

DSpark evaluates seven parallel query rows against the same committed target-derived context KV.
The rows do not attend one another. `d_flash_mla_decode` now exposes that behavior as a compile-time
specialization; the default causal specialization keeps its original path.

The harness uses the exact TP8 draft geometry (`H=8`, `DK=512`, `DR=64`, `V=128`), seven queries,
four KV splits, the full flash plus `MlaMergeFold` chain, and an independent f64 host reference. Its
causal control intentionally uses the same inputs and progressively truncated prefixes, proving
the fixture distinguishes the two semantics.

Build and run:

```bash
nix develop -c cmake -S runtime -B /tmp/plow-k3-dspark-mla-build \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop -c cmake --build /tmp/plow-k3-dspark-mla-build \
  --target k3_dspark_mla_gfx942_test -j2
nix develop -c perf-data/tools/gpulease -n 1 k3-dspark-mla-oracle \
  /tmp/plow-k3-dspark-mla-build/bench/k3_dspark_mla_gfx942_test \
  /tmp/plow-k3-dspark-mla-build/bench/k3_dspark_mla_gfx942.elf
```

MI325X result:

```text
T=7 H=8 ctx=37 ns=4
noncausal_rms=0.00132644
causal_rms=0.611269
causal_delta=0.611247
PASS
```

The standalone non-causal kernel compiles at 256 VGPR, occupancy two, with 39 spills. That resource
result is a hard warning: this object proves math and packet geometry only. Production work must
replace rather than add the specialization in a DSpark-only object, then pass the normal resource,
interpreter, target-acceptance, and served-performance gates.

Build hashes: code object
`767c433b166da96c1d6ef0844bafb2f8ba9ab498b27c99d4207622218a17b58b`; host oracle
`cd34676e7bafc48bf3bbd6a6e660d53e796b455faa64f7455e0ba6e2d5a75f57`.
