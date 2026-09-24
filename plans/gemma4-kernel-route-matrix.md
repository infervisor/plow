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
