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

Matched TP8 8192-to-1 network gate (one uncontended control/candidate fold):

| metric | control | parallel candidate | delta |
|---|---:|---:|---:|
| TTFT | 2825.504 ms | 2817.094 ms | -8.410 ms (-0.298%) |
| router trace total | 52.237 ms | 56.145 ms | +3.908 ms |
| align trace total | 30.060 ms | 18.664 ms | -11.396 ms |
| router + align | 82.297 ms | 74.809 ms | -7.488 ms |

Both runs produced `fnv1a64:7d749e3b002fafa7`. The candidate therefore passes the exact token
gate and the combined trace total improves, but does not meet the <=40 ms full-network target.
The 1024-workgroup router body falls to 13.743 ms, while its successor gate rises to 42.403 ms
(460.897 us/packet). This is a scheduling/convergence penalty from launching beyond the resident
CU count, not router arithmetic. Screen a generic `n_cu`-scale router cap before promotion.
Default remains off.

Matched artifacts:

- control packet SHA256: `dc95d77cd15ae5c1d5556ce96a81c6dfd984a50e4887994f288838b23a8afdf9`
- candidate packet SHA256: `173bfdb5151117af9885959ab43e6ab78a941c489c2389d49c2fd70dcf8bfa39`
- control trace SHA256: `22c0cc98213b228fdbb6c171614e05f25a66dfe171f2721fb929df692de395ee`
- candidate trace SHA256: `ab310e73e114e2dfde62b37dd707bf902677c56942d78d36bdb505b460a373a6`

The cap is required by the packet ABI: per-domain arrival counts are 9-bit (`<=511`). On the
256-CU TP8 target, 1024 logical slices distribute to 128/domain; emitting all 8192 slices would
produce 1024/domain and is rejected by `DevBuild`.

## Resident-CU router follow-up

The generic follow-up caps the router at `min(T,n_cu)` logical slices and leaves parallel align
behind the same default-off knob. An uncontended isolated screen is exact:

| T | router, 304 slices | router, n_cu slices | serial align | parallel align | n_cu combined |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.608283 ms | 0.507203 ms | 0.076600 ms | 0.052560 ms | 0.559763 ms |
| 8192 | 4.230227 ms | 3.985902 ms | 0.373162 ms | 0.210961 ms | 4.196863 ms |

The matched TP8 8192-to-1 fold confirms the convergence fix:

| metric | control | n_cu candidate | delta |
|---|---:|---:|---:|
| TTFT | 2794.259 ms | 2785.620 ms | -8.639 ms (-0.309%) |
| router trace total | 52.330 ms | 52.465 ms | +0.135 ms |
| align trace total | 30.004 ms | 18.492 ms | -11.512 ms |
| router + align | 82.334 ms | 70.956 ms | -11.378 ms |

Both runs produced `fnv1a64:7d749e3b002fafa7`. Router successor gating is 1.655 us/packet,
down from the 1024-slice candidate's 460.897 us/packet. The n_cu cap removes the scheduling
regression and preserves the align gain, but the full-network combined category remains above
40 ms. Keep default-off.

Follow-up artifacts:

- control packet SHA256: `b8a71791114c9fc059253dd8159b32dd9d7169c9cc17fa56f737217f4a5082a5`
- candidate packet SHA256: `6e4cba76e4facf9cf4b5c18f7742ad864ce55bd7088a7291ac8fdaae9b998a91`
- control trace SHA256: `2eff6ec0f444f1cb8fc722cde62adc5edca4818de3c977d6e718440c1685b744`
- candidate trace SHA256: `ee804943cb0c614690f4ecb725db6380a690d76c98c959edd471c78f7ae470d9`
