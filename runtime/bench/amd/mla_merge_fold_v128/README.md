# V128 token-blocked MLA merge/fold

This isolates the dense BF16 `MlaMergeFold` shape `T=8192`, `H=12`, `DK=512`,
`V=128`, `nsplit=1` on gfx950. It compares the production one-token work item
with token blocks of 2, 4, and 8. Selection depends only on tensor geometry.

The blocked source preserves every token's merge and fold accumulation order.
The harness compares all 12,582,912 BF16 outputs with `memcmp`, flushes 512 MiB
before every timed invocation, and reports the median of 31 samples. Compilation
fails on scratch, VGPR/SGPR spills, occupancy below two waves/SIMD, or LDS above
48 KiB.

The gfx950 experiment is rejected for production. The shipped TB1 output hashes
to `d598c4400e42c6e3`; TB2 differs in 245 outputs and TB4/TB8 differ in 1,463,
with the latter two hashing to `e44226725775468a`. Serializing the merge leaves
the mismatch sets unchanged. Forcing the same packed-FMA count and modifiers per
token also produces `e44226725775468a`, so source-level summation order is not a
sufficient exactness condition when multiple token accumulator chains are live.

```sh
nix develop -c env BUILD_ONLY=1 runtime/bench/amd/mla_merge_fold_v128/run.sh
nix develop -c runtime/bench/amd/mla_merge_fold_v128/run.sh
```

The rejected runtime arm remains default-off through `PLOW_MLA_FOLD_TB` for
isolated investigation only. Unsupported widths,
ragged token groups, large split counts, or insufficient work fall through to
the existing scalar work-item mapping.
