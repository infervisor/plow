# V128 token-blocked MLA merge/fold

This isolates the dense BF16 `MlaMergeFold` shape `T=8192`, `H=12`, `DK=512`,
`V=128`, `nsplit=1` on gfx950. It compares the production one-token work item
with token blocks of 2, 4, and 8. Selection depends only on tensor geometry.

The blocked kernels preserve every token's merge and fold accumulation order.
The harness compares all 12,582,912 BF16 outputs with `memcmp`, flushes 512 MiB
before every timed invocation, and reports the median of 31 samples. Compilation
fails on scratch, VGPR/SGPR spills, occupancy below two waves/SIMD, or LDS above
48 KiB.

```sh
nix develop -c env BUILD_ONLY=1 runtime/bench/amd/mla_merge_fold_v128/run.sh
nix develop -c runtime/bench/amd/mla_merge_fold_v128/run.sh
```

The runtime arm is default-off through `PLOW_MLA_FOLD_TB`. Unsupported widths,
ragged token groups, large split counts, or insufficient work fall through to
the existing scalar work-item mapping.
