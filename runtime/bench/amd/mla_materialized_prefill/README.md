# Materialized MLA prefill screen

This harness compares Plow's absorbed attention plus fold, its first rectangular
materialized kernel, and the generic gfx950 `D_QK=192`, `D_V=128` Opus schedule.

The measured schedule is vendored as a standalone runtime object under
`runtime/amd/third_party/aiter_opus`. It is based on AITER upstream commit
`10b192f5b5bda90f2af33ceae7a6c2f416bfc674` and retains the MIT license. The
runtime object has no build-time or run-time AITER dependency. A guarded Plow
adaptation decodes the original 3D batch grid from plowrt's flat 1D launch.

The useful schedule is selected by dimensions and architecture, not a model name:

- wave64, eight waves, 32 query rows per wave;
- 64 KV rows per stage;
- full Q fragments retained in registers;
- two K and two V LDS buffers, with V reusing Q's dead LDS lifetime;
- score and value GEMMs split into two super-units;
- a 16-stage K/V/MFMA/softmax pipeline with staggered wave groups.

Run with:

```sh
nix develop -c env SAMPLES=31 \
  runtime/bench/amd/mla_materialized_prefill/run.sh /tmp/plow-mla-opus-gate
```

The script rejects spilling standalone objects through explicit compiler-resource
checks. It compares the flat object byte-for-byte with a 3D-grid oracle for every
head, including a ragged 1025-token launch. Exact 256-token multiples must also
remain within max absolute error `0.02` and RMSE `0.003` against the absorbed form.
The full-path timing includes both sides' query projection GEMMs; the materialized
side additionally includes KV projection and packing.

The full-path oracle derives the absorbed BF16 query and value weights from the same
factor weights used by the materialized side. This is distinct from the kernel-only
oracle above: independently initialized factor and absorbed weights can time the
projection pipelines, but cannot establish their numerical equivalence.

The old in-tree `k_materialized` comparator is retained as a diagnostic but is not a
promotion gate: its output is nondeterministically non-finite on gfx950. Set
`PLOW_MLA_LEGACY_GATE=1` to reproduce its historical hard gate, or
`PLOW_MLA_DIAGNOSTICS=1` to print finite counts after every stage. The production Opus
object and the consistent-weight full-path oracle remain hard gates.
