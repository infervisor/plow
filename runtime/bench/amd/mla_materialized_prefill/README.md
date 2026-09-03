# Materialized MLA prefill screen

This harness compares Plow's absorbed attention plus fold, its first rectangular
materialized kernel, and the generic gfx950 `D_QK=192`, `D_V=128` Opus schedule.

The Opus comparator is compiled from a local AITER checkout. `AITER_ROOT` defaults
to `/tmp/aiter-main`; the measured source is unchanged between local commit
`90e91d5e275216da17f306e35e9b5519c621dbe3` and upstream commit
`10b192f5b5bda90f2af33ceae7a6c2f416bfc674`. AITER is MIT licensed. Its source is
used only to build the isolated comparator and is not a Plow runtime dependency.

The useful schedule is selected by dimensions and architecture, not a model name:

- wave64, eight waves, 32 query rows per wave;
- 64 KV rows per stage;
- full Q fragments retained in registers;
- two K and two V LDS buffers, with V reusing Q's dead LDS lifetime;
- score and value GEMMs split into two super-units;
- a 16-stage K/V/MFMA/softmax pipeline with staggered wave groups.

Run with:

```sh
nix develop -c env AITER_ROOT=/tmp/aiter-main SAMPLES=31 \
  runtime/bench/amd/mla_materialized_prefill/run.sh /tmp/plow-mla-opus-gate
```

The script rejects spilling objects through explicit compiler-resource checks. The
candidate is compiled as wave64 by the gfx950 target and must also remain
within max absolute error `0.02` and RMSE `0.003` against the absorbed form.
