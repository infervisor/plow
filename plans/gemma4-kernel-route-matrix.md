# Gemma-4 on H100: the closed kernel set, and the cuBLAS-vs-native route per kernel

Goal this answers (user, 2026-09-24): *"close the kernels, heavily tuned for prefill and decode,
decide for each either cublas or plow native right cubin config, roofline is the goal, then only
focus on the single block."*

Branch `gemma4-26b-beat-vllm-r2`, cut from main `89fcf6bf` (the campaign branch was merged to
main via PR #44; 0 commits were left unmerged).

## 1. The kernel set, with dims and rungs

Taken from each packet's OWN emitted instruction stream (`build.json` `dispatch_audit`, where
M/N/K are `i[0..3]`), not derived from the HF config. The `insts` multiplicity is what proves the
mapping: on the 26B, 5 = full-attention layers, 25 = sliding, 30 = all, 60 = 2x all.

| | 26B-A4B | 12B |
|---|---|---|
| hidden | 2816 | 3840 |
| layers | 30 = 5 full + 25 sliding | 48 = 8 full + 40 sliding |
| heads / head_dim | 16 / 256 sliding, 512 full | 16 / 256 sliding, 512 full |
| kv_heads (gqa) | 8 sliding, 2 full (gqa 8) | 8 sliding, 1 full (gqa 16) |
| FFN | MoE 128 experts x 704 (n=2112) | dense 15360 |
| vocab | 262144 | 262144 |

Prefill rungs 26B: `128 256 512 1024 1152 2048 4096 4224 8192`; the 12B adds `1088 4160`.
Decode ladder: `1,2,4` on the bf16 lean packet, up to 16 on the FP8 packets.

The 16 dense projection shapes (N, K), which are exactly the set
`runtime/bench/nvidia/bf16_gemm_vs_cublas_bench.cu` sweeps:

    12B   gate_or_up 15360x3840   down 3840x15360   local_q 4096x3840   local_o 3840x4096
          local_k_or_v 2048x3840  global_q 8192x3840  global_o 3840x8192  global_k_or_v 512x3840
    26B   m26_gate_up 2112x2816   m26_down 2816x2112  m26_local_q 4096x2816  m26_local_o 2816x4096
          m26_local_kv 2048x2816  m26_global_q 8192x2816  m26_global_o 2816x8192  m26_global_k 1024x2816

**Not covered by the route matrix**: the routed-expert MoE GEMMs (26B) and `lm_head`
(262144 x hidden). Those remain open.

## 2. The route decision — measured, 448 comparisons

`scripts/bench/gemma4_route_matrix.sh` -> 16 shapes x 14 rungs (1,2,4,8,16,32,64,128,256,512,
1024,2048,4096,8192) x 2 precisions. Protocol: rotated cold weights, alternating arm order, 6
rounds, median of the middle two, bounded-error correctness gate, output guard bytes. The nvcc
flags match the shipped `interp_sm90a_pfgemm.cubin`, so the body measured is the body that serves.

### BF16: cuBLASLt, and that is already what ships. CLOSED.

**cuBLASLt wins 216 of 224 cells.** The 8 plow-native wins are single isolated rungs at 1-8%
margins -- inside the protocol's own spread, so they are not a route. The blanket
`emit.prefill_cublaslt = True` is therefore CORRECT, not a missed per-shape opportunity. This is
the first time that flag has been justified by measurement rather than assumed.

At M=8192 cuBLASLt beats plow-native by 1-35% and reaches 57-83% of the roofline ceiling.

### FP8/W8A8: cuBLASLt wins almost everywhere, and plow routes to the LOSER. OPEN.

The FP8 packets (`p26fp8`, `p12fp8*`) carry **zero** `cublaslt_algos.jsonl` rows: every FP8 GEMM
runs plow-native (`GemmFp8`, `GemmGluFp8`, `QuantFp8`, `MoeGroup*W8a8`). But measured against
cuBLASLt e4m3 with `CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` -- the per-token x per-channel
scaling the checkpoints actually use -- **cuBLASLt wins 212 of 224 cells, by 5-43%**.

(`w8a8_lt_scalar` is faster still but is the WRONG numerics for these checkpoints, so it is never
the decision baseline.)

The per-shape exception, which is where plow-native should stay:

    12B down   n=3840 k=15360   plow wins M = 1024, 2048, 4096, 8192   -- 62-68% of roof

**Run twice, and that is the only exception that survives.** The matrix was re-run on binaries
rebuilt from this branch (2026-09-24 18:35) and scored against the 2026-09-23 run kept as
`route_*.prev.out`:

| | 2026-09-23 | 2026-09-24 re-run |
|---|---|---|
| BF16 | cuBLASLt 216 / plow 8 | cuBLASLt 215 / plow 9 |
| FP8  | cuBLASLt 212 / plow 12 | cuBLASLt 215 / plow 9 |

12B `down` reproduces on the SAME four rungs both times. Every other plow win moves between runs
(26B `m26_down` [1,2,8,16,1024] -> [2,32,1024]; `m26_local_kv` [512,2048] -> none; `m26_gate_up`
none -> [8,64]), which is the proof they are noise and not a route. Same for the 8-9 BF16 plow
wins, whose margins are 1-8%.

Everything else, both precisions: cuBLASLt.

## 3. Roofline: where the headroom actually is

H100 SXM5 constants from `scripts/campaign/roofline.py`: 989 TFLOP/s bf16, 1979 fp8, 3352 GB/s
(ridge 295 FLOP/B bf16).

* BF16 at the big prefill rungs is **82-90% of ceiling** measured in-packet from the tuner's own
  `matmul_us`. There is almost nothing left there.
* **This kills the GEMM path as the big-rung TTFT lever.** At m=8192 on the 26B the entire tuned
  GEMM budget is 1.80 ms and a *perfect* kernel returns only 313 us -- against a measured 30.8 ms
  TTFT deficit at 8192/C1. The deficit is elsewhere (FlashPrefill, the norm/GLU/rope tail).
* FP8 winners sit at only **3-68%** of ceiling, so FP8 has real room -- but the first move is
  routing, not kernel work.
* Worst kernels in both precisions are the **small-N KV projections**:
  12B `global_k_or_v` (n=512) at 2-4% of roof, 26B `m26_global_k` (n=1024) at 3-6%. Memory-bound
  and launch-latency dominated.

Two roofline numbers are NOT interchangeable and are kept separate above: the in-packet tuner
`matmul_us` (the algo actually selected during a serve) and the standalone route-matrix bench
(cold rotated weights). Same shapes, different context.

## 4. The emitter already flags the native path

`dispatch_audit.findings`: 29 on the 26B BF16 packet, 26 on the FP8 one. **28 of 29 are prefill-Gemm
occupancy**, every one at a hardcoded `256x256` tile, occupancy 0.08-0.49 against the 0.50 floor,
with `cus` reaching only 15/33/66/117 of 132 SMs. At M=128, N=2112 that is 9 tiles total. FP8
inherits the identical defect. This is the "right cubin config" half of the question and it is
still open -- but note it only matters for shapes where native is the chosen route.

## 5. Open, in priority order

1. Wire a cuBLASLt route for FP8 (the 212/224 win). Today there is no Lt path for FP8 at all, and
   `campaign.py probe` only selects among cuBLASLt ALGOS -- it never times native at the same
   shape, so no native-vs-Lt decision exists anywhere in the pipeline.
2. Tile/occupancy work is now scoped to almost nothing: after the re-run, the only shape where
   plow-native is the right route is 12B `down` under FP8 at M >= 1024. The `256x256` fixed-tile
   occupancy defect is real but it sits on shapes that should be going to cuBLASLt anyway, so it
   is NOT the lever it first looked like.
3. MoE routed-expert GEMMs (26B) and `lm_head` are not in the matrix at all. On an MoE model the
   expert GEMMs are the bulk of the FFN work, so this is the largest remaining blind spot, and it
   is bigger than it looks: `dispatch_audit.ops` covers ONLY `Gemm`/`Gemv`/`GemvGlu`/`GemvQkv`
   (81/18/3/3 rows) -- **no MoE op is audited at all**, so the routed-expert path has no M/N/K, no
   tile, no occupancy and no tuner row anywhere. It is untuned, unaudited and unmeasured.

   The dense `n=2112` shape in the matrix is the DENSE intermediate (config `intermediate_size`),
   not the experts: routed experts are `moe_intermediate_size` 704, i.e. gate/up 2816->704 and
   down 704->2816 per expert, top-k of 128. Those never appear.

   For MoE the route question is answered at the packet level, not by a microbench: the knobs
   already exist (`PLOW_EMIT_MOE_PF_LT` / `PLOW_MOE_PF_LT`, `PLOW_MOE_DEC_LT`), so it is a
   two-packet A/B with grouped-Lt on vs off, same session.

   **MEASURED 2026-09-24, and it closes the 26B BF16 MoE route.** `p26mlt` (recipe as-is,
   `moe_pf_lt=True moe_dec_lt=True`) vs `p26mnat` (both forced to 0), step_bench prefill wall,
   3 reps, order-reversed within each ctx, one lease:

   | ctx | MoE cuBLASLt | MoE native | delta |
   |---|---|---|---|
   | 1024 | **35.0** +-0.0 ms | 40.0 +-0.0 ms | **+5.0 ms (14.3%)** |
   | 4096 | **98.0** +-0.0 ms | 112.3 +-0.6 ms | **+14.3 ms (14.6%)** |

   Grouped cuBLASLt wins by ~15% of the WHOLE prefill wall -- an order of magnitude more than
   anything available in the dense shapes, where cuBLASLt already runs at 82-90% of roofline.
   The shipped default is right. Both packets emit the identical MoE opcode set, so what differs
   is the dispatch of those ops, not which ops exist; the knob values in `build.json` are what
   distinguishes the arms and they were checked before the bench ran.

   Still open: grouped MoE decode is bf16-gated (task #62, `lib.rs:6083-6100`), so the FP8 arm of
   this A/B cannot be built. FP8 MoE routing is undecided.
4. DONE: route matrix re-run on this branch's binaries (see the table above).

## 6. The dense route at NETWORK level — and why it could not be measured before

Sections 2-3 rank kernels in isolation. `plow-insitu-vs-harness-fat-object` records harness wins
inverting once served, so the dense route had to be settled end-to-end. That required running a
packet built `PLOW_EMIT_PREFILL_CUBLASLT=0` on the 26B, which **hung** — task #88.

### 6.1 Root cause: a wait list that omits the attention-GEMM sites

`gpu.rs` had three arms for `cublaslt_waits`. `ordered_waits_for` marks the instructions a **host
library call** executes; those never signal their device counters, so their consumers must fall
back to stream order. The middle arm, reached only when `projection_segments.is_empty()` (i.e. a
native-dense packet), passed an **all-`None`** list — naming no library launches at all. With the
attention-GEMM route on, the attention sites *are* library launches, so their consumers kept a
counter wait on an instruction that never signals and the prefill **spun forever holding the GPU
lease**. The third arm already did this correctly (`.or(attention_gemm_segments…)`).

It needs BOTH conditions, which is why it survived:
* native dense prefill — to take that arm at all; every shipped packet uses Lt projections, and
* `t >= pf_attn_gemm_min_rows` (**default 1024**, `config.rs:892`) — to have any attention site.

Fix: delete the middle arm and let the general one handle the empty case; its closure already
falls back to the attention sites when `projection_segments` is empty.

Measured, one lease, `p26nat2`/`p26lt2` (pow2 ladders, matched):

| arm | ctx 1024, route default | ctx 1024, route off |
|---|---|---|
| native dense | **HUNG(spin)** | 0.037 s |
| cuBLASLt dense | 0.035 s | 0.035 s |

The threshold coincides with the 1024 sliding window, which is why shape-based explanations looked
plausible. `PLOW_PF_ATTN_GEMM=0` separates them: it moves the route without touching the window.

Refuted on the way, all from the packets alone (no lease): mixed `Gemm`+`FlashPrefill` segments
(**0 in both arms**); the flash segment's neighbourhood (identical — preceded by `HeadNormRope`,
followed by `Gemm`, in both); bucket-dependent packet structure (the segment layout is
**bucket-invariant**, so nothing in the packet changes at 1024); cooperative-launch co-residency
(dies with the neighbourhood result). An earlier "last op = FlashPrefill" reading was an artifact:
`PLOW_NV_TRACE` needs a `-DPLOW_NV_TRACE=1` prefill cubin, which these packets lack, so it emitted
no op trace — the matched string was a startup line present in the passing log too.

### 6.2 The answer: native dense prefill loses the NETWORK by ~5%

Route off on both arms (matched), step_bench prefill wall, 2 reps, spread <= 0.001 s:

| ctx | plow-native | cuBLASLt | native |
|---|---|---|---|
| 1024 | 0.037 s | **0.035 s** | +5.7% |
| 4096 | 0.106 s | **0.101 s** | +5.0% |
| 8192 | 0.227/0.228 s | **0.218 s** | +4.4% |

This is the whole prefill wall, not a kernel. Native carries a structural advantage here and still
loses: it has **85 fewer segment boundaries** per bucket (206 -> 121 `Gemm` segments; every other
segment class is identical). So the Lt glue of task #37 is real but does **not** cover the kernel
deficit — the gap is in the native GEMM kernels themselves. The isolated verdict (BF16 cuBLASLt
216/224) is **confirmed end-to-end, not inverted**.

For "use plow-native wherever it has a chance": on 26B dense prefill it does not, at any measured
rung. The remaining native opportunities are the ones section 5 already names.

### 6.3 Second blind spot: `dispatch_audit` cannot see the route

`dispatch_audit` is **byte-identical** between `p26nat2` and `p26lt2` — same 78 ops, same 29
findings — although one packet runs every dense GEMM through cuBLASLt and the other runs all of
them natively. It records geometry (M/N/K, tile, occupancy), not dispatch. So
alongside "no MoE op is audited at all" (section 5.3), the audit also **cannot distinguish native
from library for the ops it does cover**. Neither blind spot is visible from the file itself.

### 6.4 VERIFIED after the fix — and the route's own value

Re-run on the fixed binary, one lease. Native + attention route ON now runs at every rung where it
spun 1/1 before, and **every arm at a given ctx returns the identical token signature**
(`fnv=08f44507b5900ff3` @1024, `0aadd007b706fc52` @4096, `b6dda40696527f3c` @8192), so the route is
correct and not merely non-hanging. That signature match is also what proves the new code is in the
binary: the rebuild came out byte-identical in size, but the old one could not complete this case.

| ctx | native route ON | native OFF | Lt route ON | Lt OFF |
|---|---|---|---|---|
| 1024 | 0.041 / 0.040 | **0.038** | **0.035** | 0.035 |
| 4096 | **0.105** / 0.105 | 0.106 | **0.098** | 0.101 |
| 8192 | **0.215** / 0.214 | 0.228 | **0.203** | 0.218 |

**`pf_attn_gemm_min_rows = 1024` is too low.** The attention-GEMM route pays only from ~4096 up
(-6.9% Lt / -5.9% native at 8192; -3.0% / -0.9% at 4096) and at 1024 it is neutral on Lt and
**+6.6% worse** on native. The rung it switches on at is the same one that exposed the deadlock.
Raising the default to 4096 is a candidate, but it is a shipped default on the Lt path where the
1024 rung measures neutral, so it needs its own cert run before flipping.

**Route decision, each arm at its own best policy** — the fair end-to-end comparison:

| ctx | plow-native best | cuBLASLt best | native |
|---|---|---|---|
| 1024 | 0.038 | **0.035** | +8.6% |
| 4096 | 0.105 | **0.098** | +7.1% |
| 8192 | 0.214 | **0.203** | +5.4% |

Wider than the matched route-off gap of section 6.2 (+4.4 to +5.7%). The conclusion holds and
strengthens: on the 26B, dense prefill GEMMs belong on cuBLASLt end-to-end, and native's 85-boundary
structural advantage does not close a kernel-level deficit.

## 7. Single block, plow vs vLLM (26B, layer 0 sliding) — RUNS, and plow wins 8/8

`scripts/block_e2e.sh` could not run at all: it invoked `plowc --bin gemma4`, a binary removed when
`--block` moved into the main binary. Five gates had to be cleared, each a real defect or gap:

1. the deleted `gemma4` bin (and `--out` is a DIRECTORY now, not a `.pkt` path);
2. **hash-pinned cubins** — borrowing another packet's objects gives
   `packet/interpreter MISMATCH` (build.json `pairing`), so the block needs its own;
3. `--emit devblob+cubin` builds **no** Gemma-4 role objects and no MoE Lt glue, so its asset dir
   cannot execute its own packet (see #91). Fixed by running
   `build_sm90a_gemma4_segments.sh` with `PLOW_CUBIN_CONFIG=$ASSET/plow_config.h` and pointing
   `PLOW_PF_SEG_DIR` at its output — all 9 role objects then build and pair;
4. environment: `plowc` needs `nix develop` for the toolchain, the harness needs the SYSTEM cuda
   libs (`libcublasLt.so.13` is not on the nix `LD_LIBRARY_PATH`), and the vLLM half needs ninja
   on PATH and must stay OUT of nix;
5. **the default segment-class policy faults the block.** First prefill launch died with
   `CUDA_ERROR_LAUNCH_FAILED` (719) at `bucket_t=128`. Measured: `PURE=1` runs (FA512 either way),
   `PURE=0` and the default both fault. So `block_e2e.sh` now sets `PLOW_PF_SEG_PURE=1`, matching
   every working 26B serve config in this campaign.

Exonerated on the way, with controls: the MoE Lt route (identical fault 4/4 across
`PLOW_MOE_PF_LT` x `PLOW_MOE_DEC_LT`) and the role objects themselves (present and paired, and
unused here: `packet segment roles loaded launches=15 gemm_launches=0 attention_launches=0`).

### Results — `BATCH=1,4 CTX=128,1024`, iters 100 / warmup 20

| B | T | decode plow | decode vLLM | ratio | prefill plow | prefill vLLM | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 128 | 198.73 us | 336.45 us | **0.59x** | 0.57 ms | 3.04 ms | **0.19x** |
| 1 | 1024 | 206.83 us | 341.89 us | **0.60x** | 1.06 ms | 3.11 ms | **0.34x** |
| 4 | 128 | 242.80 us | 455.01 us | **0.53x** | 2.29 ms | 3.04 ms | **0.75x** |
| 4 | 1024 | 315.23 us | 468.38 us | **0.67x** | 4.25 ms | 4.52 ms | **0.94x** |

Median **0.60x decode**, **0.75x prefill**; plow faster in all 8 cells.

**Read this carefully.** The prefill ratio degrades monotonically with work — 0.19x -> 0.94x — so at
B=4/T=1024 the block advantage is nearly gone, which is the same shape as the served ladder where
prefill is the half vLLM competes on. Decode holds 0.53-0.67x across the grid. And the comparison is
TIME ONLY: `block_layer_bench.py` builds vLLM's real `Gemma4DecoderLayer` from the config JSON with
RANDOM weights and no checkpoint, while plow compiles the block from the real checkpoint, so the
outputs are not comparable and no numerics claim can be made from this. One block (layer 0,
`sliding_attention`) is also not the network: `block.json` reports 15 launches for it.

A defect found but not fixed: `devgen/src/lib.rs:9669-9671` hardcodes `arch: "gemma_dense"` and
`kind: ["dense_attn","dense_ffn"]` for EVERY Gemma block, so an MoE layer is misdescribed as dense
in `block.json`. Harmless today only because `block_run` reads `desc.arch` just to print it
(`block_run.rs:185`).
