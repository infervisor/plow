# AttnRes f32-mix output-norm gate

This default-off experiment compares the current exact Plow fusion (BF16 mix seam) with the
pinned-vLLM arithmetic order (f32 mix carried through output RMSNorm). It uses the real boundary
geometry, independent score/output epsilons, BF16 prefix/delta rounding, all live ring counts, and
separate BF16 score factors. The run prints hashes for every input component and requires
byte-stable adjacent candidate repeats, a close BF16 result against the CPU transcription, zero
private/spill storage, and at least 10% isolated gain before any TP8 experiment. The latency gate
covers both one-token rows at every live ring count and the exact `T=8192, H=7168, nb=8`
prefill concurrency (`256` workgroups, 32 token rows per workgroup).

The 2026-09-04 MI355X gate rejected the arm: T8192/nb8 improved only 7.25% and T1/nb8
only 8.15%, below the 10% requirement; the candidate also retained SGPR spills. See
`perf-data/mi355x-attnres-f32-mix-norm-20260904.md`. It is intentionally not wired into dispatch.

Run under the shared GPU lease:

```sh
GPU_LEASE_DIR=/tmp/gpulease perf-data/tools/gpulease -n 1 attnres-f32-mix-norm \
  nix develop -c runtime/bench/amd/attnres_f32_mix_norm/run.sh
```

For real captures, first seal the complete residual contract with `scripts/mla_boundary_abi.py`.
The required tensors are `residual.prefix`, `residual.delta`, `residual.ring`, both score factors,
and the output norm gain; the hashed state includes both epsilons and `mixed-f32` ordering. This
microbench deliberately remains separate from production dispatch until it clears its gates.
