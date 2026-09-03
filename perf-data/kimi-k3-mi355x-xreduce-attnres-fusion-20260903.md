# XReduceTwoShot → AttnRes phase-2 fold (MI355X, 2026-09-03)

Status: implemented behind `PLOW_FUSE_XR_ATTNRES=1`; rejected for production after the full TP8
fold. The segmented route remains default-off.

## Structural gate

- Selection uses the complete packet dependency graph, tensor identity, dimensions, and WG map.
- Required: adjacent sole coarse consumer, legacy unbanded XReduceTwoShot, exact AttnRes operands,
  `n=T*H`, equal producer/consumer WG sets, `TP>1`, and `T%TP=0`.
- The row divisibility condition is load-bearing: flat reduce-scatter assigns each rank
  `T/TP` complete rows. Phase 2 maps token row `m` to owner `m/(T/TP)`.
- Full graph census: T1 = 0 candidates; BF16 TP8 T8192 = 94 candidates (93 attention seams plus
  the structurally equivalent dense-output seam), all with 256 WGs.
- Direct and materialized-residual contracts are independent; the residual form preserves the
  intermediate BF16 rounding before AttnRes. Reduction rank order and both cross-rank gates are
  unchanged.

## Exactness

`runtime/tests/xreduce_attnres_fused_gfx950.hip` executes the production collective and AttnRes
bodies at T256/H7168/TP8 with 256 WGs, then compares against separate gather/materialization and
AttnRes launches.

| form | reduced output | materialized prefix | final output |
|---|---|---|---|
| direct | bit-identical | bit-identical | bit-identical |
| residual | bit-identical | bit-identical | bit-identical |

Lean proofs also rebuild, and packet regressions pin dependency/counter remapping, fanout refusal,
forced-uniseg isolation, and the exact full-graph census.

## Object resource gate

The first dedicated interpreter build was rejected: 139 VGPR, 106 SGPR, 100 SGPR spills,
zero VGPR spills, and zero private bytes. The shipping candidate is a standalone typed-argument
object, so this heavy arm never enters the mega interpreter.

| object | wave | WG max | occupancy | VGPR | SGPR | VGPR spill | SGPR spill | private | LDS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current prefill control | 64 | 512 | 2 | 256 | 108 | 4 | 76 | 1332 B | 147504 B |
| rejected dedicated interpreter | 64 | 512 | 2 | 139 | 106 | 0 | 100 | 0 B | 147504 B |
| `xreduce_attnres_gfx950.elf` | 64 | 512 | 2 | 160 | 83 | 0 | 0 | 0 B | 147468 B |

The build fails on a wave mismatch, occupancy/register cliff, either spill class, nonzero private
segment, or missing resource marker. The runtime requires both ABI/resource markers, exact packet
hash pairing, an 8-byte pointer kernarg, and zero private bytes. Mixed segments and absent objects
fail closed rather than falling through to the generic interpreter.

The 147468 B LDS allocation means one workgroup per CU is resident despite the compiler's
two-wave/SIMD occupancy record. The cooperative launch deliberately uses the packet's `blocks`
(256 in every qualified full-graph seam) and 512 threads (eight wave64 waves). Each route's typed
argument block is uploaded once and owned by its `AmdProg`; queued launch kernargs contain only
that stable device address, so later segment enqueues cannot overwrite it.

## Full-model gate

The exact BF16 TP8 assets use all 7650/7650 measured TuneDB records for
`gfx950-76ef5b9982d04cbd`. Three independent, order-alternated 8192→1 folds produced token 6896
and checksum `fnv1a64:7d749e3b002fafa7` in every arm.

| fold | control TTFT | segmented candidate TTFT | delta |
|---:|---:|---:|---:|
| 1 | 1616.160 ms | 1707.265 ms | +91.105 ms |
| 2 | 1615.280 ms | 1708.438 ms | +93.158 ms |
| 3 | 1616.523 ms | 1707.473 ms | +90.951 ms |
| mean | 1615.987 ms | 1707.725 ms | +91.738 ms (+5.677%) |

An explicit P0 8192-token prompt followed by 256 decode tokens also passes exactly: both arms
complete 1/1 with no failures, checksum `fnv1a64:6bdfaa7b84ee4e7e`, identical 256-token audit,
and TP agreement checked on every dispatch. Control/candidate TTFT is 1618.254/1708.324 ms;
TPOT is 44.2541/44.2534 ms. The route affects prefill only: decode is equal within noise while
the same approximately 90 ms prefill penalty remains.

The graph rewrite removes 94 AttnRes packets and 24,064 stream entries at T8192
(725,629 → 701,565). Isolating each seam into the dedicated object splits both sides of the
original segment, however, so ordered segments and AQL dispatch/drains per rank rise 325 → 513:
exactly two extra boundaries per seam. The stable endpoint loss is 0.488 ms per added boundary.
Deleting the elementwise work cannot repay 188 extra whole-device drains.

The first campaign also exposed a load-order validator bug before any GPU launch: TP audit stores
its runtime status index in `fj[2]`, while the fused-route validator incorrectly required every
slot after `fj[0]` to remain zero. The fixed contract preserves `fj[0]`, requires reserved
`fj[1]==0`, consumes runtime-owned `fj[2]`, and still fails closed on invalid/mixed segments.

An in-interpreter encoding would avoid the extra segments, but its clean gfx950 resource probe is
also rejected:

| K3 MoE A4W4 GQ object | VGPR spill | SGPR spill | private | wave | occupancy |
|---|---:|---:|---:|---:|---:|
| control | 8 | 74 | 1348 B | 64 | 2 |
| fused call-site arm | 42 | 112 | 1364 B | 64 | 2 |

The arm adds 34 VGPR spills, 38 SGPR spills, and 16 B private memory. It must not enter the mega
interpreter. A future attempt needs grouped segment reuse or a different lean-object schedule; the
raw per-seam segmented route and the spilling interpreter arm are both closed.

Fresh pre-measure rank-0 attribution for the current control reports XReduceTwoShot at 217.52 ms
over 278 packets (782 us each), AttnRes at 88.59 ms over 187 packets (474 us each), and only
6.18 ms total in the collective gate. The 94 eligible seams therefore cover a meaningful fraction
of the 297.5 ms traced residual, but this is sizing evidence only; promotion still requires the
combined endpoint fold.
