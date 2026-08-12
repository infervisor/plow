# Kimi-K3 MI325X Stage 4: MXFP4 roofline and tile census

Date: 2026-08-10. Hardware: one leased MI325X (gfx942, 304 CUs). Toolchain:
flake-pinned ROCm 7.14.0, HIP 7.14.60850, clang 23. Every compile and run was entered
through `nix develop`; every timing run held `gpulease`.

## Measured ceilings

The CMake `ceiling_probes` target builds raw gfx942 code objects with the flake's `hipcc`
and `clang-offload-bundler`.

| ceiling | result | campaign denominator |
|---|---:|---:|
| 32x32 BF16 MFMA, production wrapper | 1,063.1 TFLOP/s | 1,063 TFLOP/s |
| 16 GB streaming HBM, three clean runs | 4,143.6 / 4,163.9 / 4,186.9 GB/s | 4,164 GB/s |

The MXFP4 gfx942 path software-dequantizes weights to BF16 and uses the 32x32 BF16 MFMA
wrapper, so the 32x32 result is the relevant compute ceiling. `hwspec::MI325X` records the
measured HBM denominator; MI300X remains unchanged.

## Production-interpreter MXFP4 census

`scripts/rebench_k3_mxfp4_gfx942.sh` measures all 16 unique TP8 projection shapes at
M={128,512,1024,2048,4096,8192}. It dispatches the five production MXFP4 opcodes through
`interp_prefill_fp8kv_k3_moe_a4w4.elf`, not standalone wrappers. Each row has 50 warmups,
12 timing samples of four launches, a full-output NaN sentinel, and 24 FP64 oracle points.

```bash
nix develop --command env \
  PLOW_GEMM_JSONL=/tmp/k3-mi325x-mxfp4-full.jsonl \
  scripts/rebench_k3_mxfp4_gfx942.sh
```

Result: 96/96 shapes correct, 480 raw rows, 672 qualified inventory-expanded records,
and 96 compiler-selectable cases in `tuning/amd/gfx942/mi325x`.

| metric | result |
|---|---:|
| shapes improved by more than 1% vs default 192x256 | 88 / 96 |
| median best-tile speedup | 1.94x |
| maximum best-tile speedup | 2.79x |
| highest measured MXFP4 throughput | 420.2 TFLOP/s |
| highest measured compute-roof utilization | 39.53% |

Best-rung counts were 64x128: 51, 128x128: 19, 128x256: 18, and 192x256: 8.
The largest result was K3 dense-down M8192,N7168,K4224 at 420.2 TFLOP/s. Dense gate-up
M4096,N4224,K7168 improved from 980.2 us to 754.8 us with 128x256 (1.30x).

The generated demand was checked for exact set equality against
`plowrt disasm /home/lava/models/k3_mi325x/model.pkt --program 128`: 16 unique packet
GEMM `(N,K)` pairs, 16 generated pairs, no missing or extra pair. The schedule-model
`tune shapes auto` path still refuses K3, so the checked packet is the source-of-truth guard.

## Fused MXFP4 GLU experiment

The fused harness sends opcode 113 and the counter-gated 93+93+5 sequence through the same
persistent interpreter, with full-output sentinel checks, 128 FP64 points, and 15 alternating
samples. All tested variants were correct.

| compiled tile | M | fused | unfused | change |
|---|---:|---:|---:|---:|
| 128x256x64 | 4096 | 0.4252 ms | 0.8827 ms | -51.83% |
| 192x256x64 | 4096 | 0.5404 ms | 1.1305 ms | -52.20% |
| 64x256x64 | 4096 | 0.5890 ms | 0.6870 ms | -14.26% |
| 128x256x32, double-buffered | 4096 | 0.5647 ms | 1.1092 ms | -49.09% |
| 128x256x64 | 8192 | 0.7965 ms | 0.9508 ms | -16.24% |
| 192x256x64 | 8192 | 0.5692 ms | 1.1700 ms | -51.35% |

BM128 wins through M4096; BM192 wins at M8192. A runtime branch inside one kernel is not a
safe way to select both because it changes register allocation. The next valid design is a
separate fused opcode rung or bucket-specific object.

## Production grouped simulated-A4W4

The grouped harness dispatches production ops 85 and 86 through `plow_interp_gfx942`. Its
small unequal-expert fixture passes an FP64 bridge/DOWN oracle. The timed fixture matches the
emitted TP8 packet geometry at a 4096-token prefill chunk: 896 experts, top-16, H=3584,
I/rank=384, 65,536 routed rows, and 114,688 BM64-padded rows. Each result has 20 warmups and
12 samples of four launches under one MI325X lease.

The existing benchmark's prior H=3584, I=3072, 32-expert geometry was not K3 TP8 demand and
was replaced before taking these measurements.

| object | GLU | DOWN | DOWN change | result |
|---|---:|---:|---:|---|
| pre-change production | 2.880 ms | 6.155 ms | baseline | pass |
| DOWN row-metadata hoist | 2.883 ms | 5.490 ms | -10.81% | pass |
| BK32 falsification arm | 3.640 ms | 6.249 ms | +1.53% | rejected |
| MFMA priority disabled | 2.861 ms | 5.477 ms | -11.02% | neutral/noise vs hoist |
| shipping rebuild | 2.882 ms | 5.497 ms | -10.69% | pass |
| BM64 final-tile wave cull | 2.856 ms | 5.698 ms | -7.43% | rejected: pair +2.09% |

The shipping change loads `row_partidx` and `row_gate` once at the tile head and distributes
them with `ds_bpermute` during the DOWN epilogue. It is enabled only for K3 A4W4 gfx942 rows;
unrelated model objects are unchanged. The shipping object reaches 219.1 TFLOP/s and 42.5%
of its roof on GLU, but only 57.4 TFLOP/s and 11.3% on DOWN.

BM64 padding expands 65,536 useful routes to 114,688 rows (1.75x). This is now the dominant
known structural target: a ragged/sub-quantum grouped path can remove more work than another
small instruction-level epilogue change. BK32 staging is not that design and regressed both
arms. A second falsification kept the 1,792 BM64 expert tiles and culled the lower wave on
final tiles with at most 32 live rows, reducing executed MFMA rows to 86,016. Its FP64 gate
passed, but GLU+DOWN rose from 8.379 to 8.554 ms, so the branch was removed rather than shipped.
This confirms that weight/staging traffic, not padded MFMA issue alone, dominates this shape.

The K3 artifact is a PLOWDEV bundle with its program ladder embedded in `model.pkt` and an
intentionally empty `weights.json.buckets`. Generic `plowrt simulate --all-buckets` therefore
does not apply. `plowrt disasm --counters --range 0..0` parsed all seven embedded programs
T={128,512,1024,2048,4096,8192,1}; the full 41-object gfx942 ISA audit passed 30 applicable
object rules. The only reported dead counter in each program is terminal `ArgmaxFin`.

## Scope

The actual checkpoint's grouped simulated-A4W4/MXFP4 routed-expert hot path is now measured and
has one shipping improvement, but DOWN remains at 11.3% of roof. Stage 4 remains open for a
padding-aware grouped design and end-to-end validation. No vLLM/SGLang result is claimed here;
that comparison requires a same-box, same-session TP8 serving campaign.
