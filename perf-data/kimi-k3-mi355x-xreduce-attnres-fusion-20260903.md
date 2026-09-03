# XReduceTwoShot → AttnRes phase-2 fold (MI355X, 2026-09-03)

Status: implemented behind `PLOW_FUSE_XR_ATTNRES=1`; not promoted pending the full TP8 fold.

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

## Pending measurement

No end-to-end speedup is claimed yet. The current model build reports 4290 TuneDB records stale
against the active gfx950 runtime digest. The required three alternating BF16 TP8 8192→1 folds must
run after the whole-graph and KDA changes settle and TuneDB is regenerated.

Fresh pre-measure rank-0 attribution for the current control reports XReduceTwoShot at 217.52 ms
over 278 packets (782 us each), AttnRes at 88.59 ms over 187 packets (474 us each), and only
6.18 ms total in the collective gate. The 94 eligible seams therefore cover a meaningful fraction
of the 297.5 ms traced residual, but this is sizing evidence only; promotion still requires the
combined endpoint fold.
