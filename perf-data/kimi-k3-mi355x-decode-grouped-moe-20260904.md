# gfx950 grouped-MoE decode qualification (2026-09-04)

## Scope and inventory

The D10 inventory-pruned B=1 decode packet has one 2,165-instruction global-queue program.
It contains 92 adjacent `MoeGroupGluFp8Blk` + `MoeGroupDownFp8Blk` pairs, each emitted at
256 workgroups. The exact geometry is top-k 16, hidden 3584, intermediate 384, 896 experts,
MXFP4 weights, and SiTU with beta 4 / linear-beta 25. The surrounding order is GEMV, router,
GLU, DOWN, fixed-order combine, then XReduce.

The authoritative D10 trace attributes 2.29318 ms/token to grouped GLU and 2.28741 ms/token
to grouped DOWN: 4.58059 ms/token, or 49.789 us/layer. The inventory interpreter is wave64,
512 threads, 248 VGPR, 106 SGPR, 440 B private memory/thread, occupancy 2, and about 147.5 KiB
LDS/workgroup.

## Transfer and isolated gate

AITER's tuned B=1 H3584/I384/E896/top16 A8W4 entries use four-wave BM32/BK256 kernels
(10.664 us stage 1 and 9.459 us stage 2). Its activation quantization changes Plow's arithmetic,
so only the standalone-family ownership pattern was transferred. AITER's A16W4 stage 1 also
failed its own tolerance gate (34.6% error) and was not used.

The Plow single-pair harness uses the emitted tensor sizes and arithmetic. It gates full `fu` and
f32 partial buffers against the current device bodies before timing. Weight traffic per pair is
35,094,528 B. All tested variants had zero full-buffer difference.

| standalone body | grid | GLU+DOWN (us) | projected 92 layers (ms) |
|---|---:|---:|---:|
| 8 waves, linear | 256 | 25.172 | 2.316 |
| 4 waves, linear | 768 | 19.392 | 1.784 |
| 4 waves, XCD4 swizzle | 768 | 21.736 | 2.000 |
| 8 waves, linear | 768 | **16.776** | **1.543** |

The selected kernels compile without scratch or spills. Shipping-object metadata is:

| kernel | VGPR | SGPR | occupancy | LDS | private |
|---|---:|---:|---:|---:|---:|
| GLU | 79 | 77 | 6 | 0 | 0 |
| DOWN | 94 | 74 | 5 | 0 | 0 |

The isolated chain includes both ordered kernel launches. Against the authoritative body it saves
33.013 us/layer, or 3.037 ms/token before segment transitions. The pure-pair route changes 49 decode
segment steps into 233 and adds 276 AQL dispatches after counting the pair's two raw kernels.

## Route

`PLOW_MOE_DECODE_STANDALONE=1` makes only an exact adjacent grouped GLU+DOWN pair a pure decode
segment. The host validates MXFP4 geometry, tensor continuity, absence of interpreter-counter
obligations, unique segment ownership, object marker, kernarg sizes, and zero private memory. It
then launches the unchanged eight-wave device bodies in order at `3 * packet n_cu`. No model name
or model-specific predicate is used. The route and its CMake object are default-off.

Evidence files: `/tmp/inv-disasm-clean.json`,
`/tmp/k3-inventory-qualified-65fca83/fold1-candidate.trace.report`,
`/tmp/k3-moe-w8-v2.csv`, `/tmp/k3-moe-w4-v2.csv`, and `/tmp/k3-moe-w4-xcd4.csv`.

## Exact TP8 network gate

The first candidate load correctly failed closed because the emitted raw pair retained its internal
GLU-to-DOWN interpreter counter. The packet rewrite now removes the pair's internal counter as well
as its cross-segment waits/signals; the queue barrier between the two raw launches preserves GLU to
DOWN order. Structural tests pin the pure pair, zero counter obligations, external boundaries, and
default-off behavior.

Three clean, order-balanced BF16-KV TP8 8192-to-256 folds used the shared `/tmp/gpulease`. Every
fold completed with zero failures, all-rank agreement, identical 256-token arrays, and checksum
`fnv1a64:6bdfaa7b84ee4e7e`.

| fold order | control TTFT / TPOT / E2E (ms) | candidate TTFT / TPOT / E2E (ms) | candidate - control (ms) |
|---|---|---|---|
| candidate, control | 1414.308 / 30.035771 / 9073.430 | 1411.567 / 29.272813 / 8876.135 | -2.741 / **-0.762958** / **-197.295** |
| control, candidate | 1411.263 / 29.993426 / 9059.587 | 1413.980 / 29.267507 / 8877.195 | +2.717 / **-0.725919** / **-182.392** |
| candidate, control | 1412.052 / 29.922151 / 9042.201 | 1410.959 / 29.262195 / 8872.819 | -1.093 / **-0.659956** / **-169.382** |

Mean TPOT delta is -0.716278 ms (sample SD 0.052173); mean E2E delta is -183.023 ms. TTFT is
neutral within run noise: mean -0.372 ms, sample SD 2.800 ms.

A clean matched trace fold is also exact and measures TPOT -0.707522 ms. Engine diagnostics over
255 decode steps measure 29.971913 ms control versus 29.263226 ms candidate (-0.708687 ms). The
trace's device chain span is 28.1214 versus 27.3892 ms (-0.7322 ms). The control trace attributes
3.21601 ms to grouped GLU+DOWN. Replacing that with the isolated 1.543414 ms raw chain predicts a
1.672596 ms warm-body ceiling, so segmented handoffs consume 0.940-0.964 ms/token, about 10.2-10.5
us per layer. Raw launches do not write interpreter trace records; consequently the candidate
per-op table is not interpretable, while its whole-chain timestamps and engine diagnostics remain
valid and agree within 24 us.

**Decision:** qualify and retain the generic route as default-off. It clears exactness and produces
a stable 0.66-0.76 ms TPOT gain. Default-on is blocked by generic performance coverage: eligibility
accepts every adjacent MXFP4 grouped GLU+DOWN geometry on gfx950, while only k16/H3584/I384/E896
has resource, latency, and network qualification. A model/shape predicate is intentionally not an
option, so unmeasured shapes must not be rerouted by default. A representative cross-shape sweep or
generic runtime profitability rule can clear that gate. Separately, a next variant should amortize
the measured handoff cost by grouping multiple layers or adding a device-side GLU-to-DOWN handoff;
it must keep combine/router/collective order unchanged.

Network evidence is under `/tmp/k3-moe-decode-network`. Packet SHA256 is `f1bf783d...` control and
`a1f7f6f7...` candidate; runtime SHA256 is `b1c4feb4...`; standalone object SHA256 is
`836c3baa...`.

## Cooperative GLU-to-DOWN handoff rejection

A follow-up isolated object put both grouped bodies in one kernel and used the sound hierarchy:
the hardware XCD id is read once, every follower publishes locally with a relaxed arrival, the
last workgroup on each XCD performs one release/writeback, all eight leaders wait for the unchanged
global threshold and acquire, then followers observe an XCD-local flag and invalidate only L1.
The one-workgroup-per-CU grid has 32 arrivals per XCD. Its complete counter census was exact.

| gfx950 B1 hot-shape arm | endpoint (us) | `fu` / partial / sync differences |
|---|---:|---:|
| selected two-launch, grid 768 | **16.807321** | 0 / 0 / - |
| two-launch residency control, grid 256 | 25.182393 | 0 / 0 / - |
| empty sound per-XCD handoff, grid 256 | 7.627928 | - / - / 0 |
| cooperative pair, grid 256 | 54.276973 | 0 / 0 / 0 |

The fused object is wave64, 94 VGPR / 63 SGPR, occupancy 5 waves/SIMD, with zero private memory
and zero VGPR/SGPR spills. It fails the selected endpoint by 3.23x. A 512-workgroup arm sits at the
two-workgroups-per-CU residency edge: one campaign completed at 105.164587 us, then two independent
campaigns stalled at the device barrier with 100% GPU activity until their owning process was
terminated. HSA AQL has no cooperative-launch residency refusal, so resource metadata cannot turn
that scheduling assumption into a safe runtime gate. Grids above one workgroup per CU are rejected.

**Decision:** keep this as an isolated negative benchmark only. Do not add a runtime route. The
sound grid-256 phase interpreter cannot beat either the isolated two-launch endpoint or the measured
network transition budget, while the faster-grid premise is not deadlock-safe on this runtime.
