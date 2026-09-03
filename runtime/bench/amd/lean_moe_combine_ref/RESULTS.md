# Fixed-order MoE combine experiment — gfx950

Date: 2026-09-03. Status: promoted as the generic structural default, with
`PLOW_MOE_COMBINE_LEAN=0` or `false` restoring the interpreter route. Hardware:
MI355X under `gpulease`.

## Current-model attribution

The final P0 BF16-KV 8192-token artifact contains 92 structurally eligible
stage-1 packets and 92 structurally eligible stage-2 packets. Runtime segment
drains provide an independent route check:

| Boundary | Segments | Drain range | Approximate total |
|---|---:|---:|---:|
| lean MXFP4 stage 1 | 92 | 2.100–2.469 ms | 221–224 ms from the qualified folds |
| lean MXFP4 stage 2 | 92 | 0.683–0.803 ms | about 72 ms |

The critical-rank packet trace attributes 146.655 ms to 92
`MoeCombinePf` packets, or 1.594 ms/layer. Those packets still execute in the
primary prefill interpreter, whose object has 256 VGPR, 8 VGPR spills, and
1,348 bytes of private memory per thread. The combine reads
`8192*16*3584*4 = 1,879,048,192` bytes of f32 part data per layer before its
small bf16 output write. At the repository's measured 6.2 TB/s ceiling, the
part read alone is about 0.303 ms/layer; the current body sustains about
1.18 TB/s.

## Candidate

The standalone kernel assigns complete token rows to workgroups and sums f32
parts in the exact interpreter order: residual, shared, then slots 0 through
15. It retains the materialized part tensor. This removes the flat-grid
division/modulo from the hot loop and permits a resident-grid sweep without
changing arithmetic.

The initial build gate reports wave64, workgroup 256, 37 VGPR, 49 SGPR, no LDS,
zero private memory, and zero VGPR/SGPR spills. Disassembly has no divide or
reciprocal sequence. It issues the 16 part loads in two batches and retains 16
sequential f32 additions. Repeated builds have identical executable `.text`
SHA-256 `20852a9f9e4a47779c97668c9877d7df0d0c83666987b3fb5ff744593faa4da8`;
the outer ELF digest varies because the compiler embeds the output path.

Removing the launch boundary later is compatible with fixed-order semantics if
a resident two-phase Down+Combine kernel keeps the f32 part buffer and places a
device-wide phase barrier between scatter and combine. Removing the part buffer
is not compatible with the interpreter association under the current
expert-sorted work assignment: atomic arrival changes addition order, while
token-major computation discards expert weight reuse.

## Isolated GPU gate

The T8192 oracle compared all 29,360,128 bf16 outputs with the
interpreter-order control while both optional residual and shared inputs were
present: zero mismatches.

Median of 31 HIP-event samples after five warmups, with the production-null
residual/shared contract:

| Kernel | Grid | Time |
|---|---:|---:|
| interpreter-order flat control | 256 | 2.874306 ms |
| token-row candidate | 256 | 0.386084 ms |
| token-row candidate | 512 | **0.327283 ms** |
| token-row candidate | 1024 | 0.333483 ms |
| token-row candidate | 2048 | 0.334523 ms |
| token-row candidate | 8192 | 0.330883 ms |

Grid 512, two workgroups per CU, is selected. Its absolute body time projects
to 30.110 ms over 92 layers. Relative to the current in-network 146.655 ms
Combine category, the available gain is about 116.5 ms. The standalone control
is intentionally not used for that projection because it lacks the production
interpreter's queue context; the candidate's absolute time and the current
full-network attribution are the relevant endpoints.

After splitting the candidate and control into separate shipping/test objects,
the oracle was repeated at three shapes. All outputs remained bit exact:
T8192/H3584 = 0 mismatches over 29,360,128 values, T257/H2816 = 0 over 723,712,
and T128/H4096 = 0 over 524,288. The repeated T8192 medians were 2.874664 ms for
the control and 0.393243/0.332603/0.333643/0.333323/0.330522 ms for candidate
grids 256/512/1024/2048/8192. The 0.6% reversal between grids 512 and 8192 is
noise-scale across the two folds, so the bounded `min(T,512)` resident policy
is the production choice.

## TP8 BF16 full-network qualification

Both assets were emitted from one clean `edffb73` snapshot with the measured
TuneDB fingerprint `gfx950-76ef5b9982d04cbd`. Both resolved all 7,650 cases to
measured choices. The only A/B difference was `PLOW_MOE_COMBINE_LEAN=0/1`.
The candidate added exactly 92 singleton Combine segments at every prefill
rung; total instruction count remained 17,619. The common interpreter pairing
was `0x48a4ccb34189de4a`.

Three uncontended order-alternated 8192→1 folds used BF16 KV, one request,
concurrency one, no warmup, and rank agreement every step:

| Fold / order | Interpreter | Lean Combine | Gain |
|---|---:|---:|---:|
| 1, control → candidate | 1621.145048 ms | 1510.046001 ms | 111.099047 ms (6.853%) |
| 2, candidate → control | 1618.959442 ms | 1510.332192 ms | 108.627250 ms (6.710%) |
| 3, control → candidate | 1618.690766 ms | 1508.576991 ms | 110.113775 ms (6.803%) |
| Mean | 1619.598419 ms | 1509.651728 ms | **109.946691 ms (6.789%)** |

Paired-gain sample standard deviation was 1.244340 ms. Every arm completed one
request with zero failures, complete non-overflowed diagnostics, all eight
ranks completing prefill, token 6896, and checksum
`fnv1a64:7d749e3b002fafa7`. The prompt-token array SHA-256 was
`dd51e931308683300f372862a682d56a332e0e3e8d71cd70ef06c34739362dcd`.

The control traces charged 146.458/146.496/146.472 ms to 92 interpreter
Combine packets. Candidate logs contained exactly 92 new raw drains per fold,
totalling 38.879/38.864/38.854 ms. Total segment drains fell by
112.153/109.416/111.042 ms, agreeing with the endpoint gains within 0.93 ms.
Thus the extra boundaries cost about 0.422 ms/layer but retain a stable net
gain; an in-interpreter arm is neither needed nor desirable because the current
interpreter spills.

The independent 8192→256 exact gate also passed. Control/candidate TTFT was
1617.633152/1509.779947 ms and TPOT was 44.409768/44.382407 ms. All 256 output
IDs were identical, with newline-ID SHA-256
`1398465e8212d27e43a6d52e95163ae34912b72255d6d14b82d7eacdcf4d718e` and
checksum `fnv1a64:6bdfaa7b84ee4e7e`. Both arms reported eight ranks, sampled
agreement every token, per-dispatch counter audit, and all-rank prefill
completion.

Raw evidence is under `/tmp/k3-combine-edffb73/results`. Nix printed three
shell-hook lines before each benchmark JSON; the original streams are retained
as `.stdout`, while canonical `.json` files were extracted from the first `{`,
validated with `jq`, and hashed separately. This is a harness-output issue, not
a runtime correctness issue.
