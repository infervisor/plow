# Kimi-K3 MI355X parallel MoE align gate

Date: 2026-09-03. GPU: one uncontended MI355X. Shape: E=896, top-k=16, wave64/WG256.
Command: `nix develop -c perf-data/tools/gpulease -n 1 p4-align-clean /tmp/moe_align_pf_bench`.
The harness uses median-of-26 HIP-event samples after five warmups.

| T | router, 304 logical slices | router, candidate schedule | serial align | parallel align | candidate combined |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.621963 ms | 0.283882 ms | 0.077201 ms | 0.052360 ms | 0.336242 ms |
| 8192 | 4.234026 ms | 2.152974 ms | 0.380722 ms | 0.211442 ms | 2.364416 ms |

The T1024 candidate projects to 30.93 ms over 92 MoE layers, below P4's 40 ms isolated target.
The full-network trace remains the promotion gate.

The schedule is shape-based, not model-based. For T >= 1024, `PLOW_MOE_ALIGN_PAR=1` emits one
router logical slice per token up to `4*n_cu` slices, placed round-robin over the available CUs,
followed by four align
packets: 64-way partial histogram, one-workgroup padded prefix, 64-way pad initialization, and
64-way deterministic scatter. Packet completion supplies every grid barrier. The partial histogram
uses an appended `[64,E]` meta slab (224 KiB for E=896 per live activation set).

Correctness gates:

- Router tables are byte-identical between the 304-slice and one-slice-per-token schedules.
- Parallel and serial metadata are byte-identical.
- Every parallel gathered row has the exact token, part index, and f32 gate from its route-table
  slot; expert ranges and padding are exhaustive; row ids are strictly increasing within experts.
- Serial vs parallel gathered arrays are not byte-equal because the old single-workgroup scatter
  explicitly has a nondeterministic LDS-atomic row-order contract. Downstream semantics are fixed by
  `row_partidx`; the candidate improves this to a stable order.

Default remains off pending the matched 8192-to-1 network trace and token-equivalence gate.

The cap is required by the packet ABI: per-domain arrival counts are 9-bit (`<=511`). On the
256-CU TP8 target, 1024 logical slices distribute to 128/domain; emitting all 8192 slices would
produce 1024/domain and is rejected by `DevBuild`.
