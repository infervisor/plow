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

### 7.1 The other layer kind: L5 `full_attention` (hd 512, kv_heads 2)

Same harness, fresh emit and role objects, `gemma4-26b-a4b-full.json` layer 5:

| B | T | decode plow | decode vLLM | ratio | prefill plow | prefill vLLM | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 128 | 220.19 us | 352.22 us | **0.63x** | 0.60 ms | 2.99 ms | **0.20x** |
| 1 | 1024 | 227.00 us | 351.97 us | **0.64x** | 1.05 ms | 3.08 ms | **0.34x** |
| 4 | 128 | 306.84 us | 471.58 us | **0.65x** | 2.40 ms | 2.98 ms | **0.81x** |
| 4 | 1024 | 323.01 us | 479.78 us | **0.67x** | 4.18 ms | 5.15 ms | **0.81x** |

Median **0.65x decode**, **0.81x prefill**. **Both layer kinds: 16/16 cells plow-faster.**

Full attention is the harder block for plow on decode — median 0.65x vs the sliding layer's 0.60x,
and the ratio is flat across the grid (0.63-0.67x) where the sliding layer spread 0.53-0.67x. Its
prefill ratio saturates at 0.81x rather than climbing to 0.94x, so at B=4 the sliding layer is
where the block advantage is thinnest, not the full one. The single block does NOT reproduce the
served deficit, which is the point worth keeping: at the block level plow wins every cell of both
kinds, so the served TPOT gap (the 20-cell scoreboard: TPOT 3/20) is not a per-layer kernel
deficit -- it comes from what surrounds the layers.

One reporting caveat: block_compare prints `plow device=?` because sweep.json records no device
string. Both halves ran on this host's single H100 under gpulease in the same script invocation,
so they are the same card; the `?` is a missing field, not an unknown machine.

## 8. The 2176 prefill rung: root cause of the incomplete Gemm segment set (#65)

The 1k/2k/4k/8k/16k ladder needs a rung that covers a 2048-token prompt. The server re-tokenises
and prepends BOS, so that prompt is 2049 rows; with rungs `[..., 2048, 4096, ...]` it pads 2049 ->
4096 and the 2k cell reads as an artificial plow loss. The recipe already appends 256 / 1152 / 4224
(= 128/1024/4096 + 128) for exactly this reason and simply has no 2048+128.

Appending 2176 via `PLOW_PF_LADDER_APPEND` produced a **structurally incomplete rung**:

```
t=2176   segments=333   DIFFERS      (modal 418)
    Gemm: has 121, modal 206
```

### Root cause

`plow_asset::segment_roles::CUBLASLT_PREFILL_WIDE_ROWS` is a **hardcoded M allowlist**:

```rust
pub const CUBLASLT_PREFILL_WIDE_ROWS: [u32; 12] =
    [1024, 1088, 1152, 2048, 4096, 4160, 4224, 8192, 8320, 12288, 12416, 16384];
```

`cublaslt_prefill_bf16(profile, m, n, k)` requires `m` to be in that list (or the narrow
`CUBLASLT_PREFILL_ROWS`), and `dense_cublaslt::prefill_eligible` consults it for every op. 2176 was
absent, so **every projection at that rung failed eligibility and reverted to the native GEMM
object** -- exactly the 206 -> 121 Gemm drop. The array's own doc comment already records this
failure mode for the three rungs added before it: *"1088 / 1152 / 4160 are fine-grained rungs
(`PLOW_PF_LADDER_APPEND`): BOS makes an N-token prompt N+1 rows. Left out, such a rung ran every
projection on the native GEMM object."*

Fix: add 2176 to the allowlist. One row, same provenance as 1152 and 4224.

### The Lt algo table was a symptom, not the cause

The first diagnosis looked at the packet's `cublaslt_algos.jsonl` and found no M=2176 row, which
suggested a GPU re-probe was needed. That reasoning is **circular**:

```rust
// packetize_algo_table
for op in &model.progs[index].insts {
    if prefill_eligible(model, op, rows, profile) { shapes.insert((op.i[0], op.i[1], op.i[2])); }
}
```

The table's shape set is collected *from* `prefill_eligible`. A rung with no Lt segments can never
contribute a row, so the missing row is downstream of the allowlist. `campaign.py probe` writes the
table from what the packet *loads*, so probing before the allowlist fix could not have produced a
2176 row either.

And a missing row does not disable the route. `device::cuda::lt::Lt::plan` pins a stored algorithm
when it has one and otherwise falls through to `cublasLtMatmulAlgoGetHeuristic` plus load-time
timing -- *"which is exactly what produced the table."* So the allowlist fix alone makes the rung
whole; a probe afterwards only upgrades 2176 from a heuristic pick to a measured one. That probe is
worth running because the 2k cell's TTFT is the number under test, and it is only now non-circular.

### Correction to the gate

`lad_verify.py` also refused on `missing=[8192]`. That requirement was wrong: with
`PLOW_MAX_CHUNK=4096` an 8192- or 15000-token prompt is prefilled in chunks of <=4096 rows, so the
8k/16k ladder cells ride the <=4224 rungs -- which is how the shipped recipe has always benched
them. The rung set the ladder actually needs is `{256, 1152, 2176, 4224}`.

## 9. FP8 closure: the kernel set, the route decision, and a CORRECTION to §earlier FP8 accounting

### 9.1 CORRECTION: the 26B MoE experts ARE fp8. Task #92 is void.

An earlier pass in this campaign read `shapes.moe_enc == []` and `precision.expert_enc == "none"`
off the 26B FP8 packet and concluded the MoE experts had stayed bf16, making FP8 worth only 3.7%
of decode bytes at B=32 against 48.3% if they were quantised. **That was wrong.**

`manifest.rs` populates `moe_enc` from only three sources -- `MoeGroupGluPf`/`MoeGroupDownPf`
(field `i[3]`), `MoeAiterFp8Pf`, and the four `*Fp8Blk` ops (field `i[6]`). **None of the Gemma-4
opcode family is among them**, and the Gemma MoE ops carry no encoding field at all: their
encoding is the opcode identity. The proof that the field carries no signal here is that the
**BF16 packet also reports `expert_enc: "none"`** -- it is blind in both directions.

What the packets actually carry (from `union`, and complementary between the two):

| packet | prefill expert GEMMs | decode expert GEMMs |
|---|---|---|
| 26B **fp8** | `MoeGroupGluGemmaPfW8a8`, `MoeGroupDownGemmaPfW8a8` | `MoeExpertGluGemmaFp8`, `MoeExpertDownGemmaFp8` |
| 26B **bf16** | `MoeGroupGluGemmaPf`, `MoeGroupDownGemmaPf` | `MoeExpertGluNormGemma`, `MoeExpertDownGemma` |

The FP8 packet carries **zero** bf16 expert arms and the BF16 packet **zero** quantised ones. The
experts are quantised. #92 ("quantise the 26B MoE experts") is a **non-task**.

What does stay bf16 in both FP8 packets is **lm_head**: the audited op at N=vocab is `Gemm`/`Gemv`
at (262144, 2816) with no `Fp8` sibling. That is #93, and it is real.

### 9.2 Why 26B FP8 decode still loses ~5x: traversal, not bytes

Decode is bandwidth-bound in both precisions, so the natural suspicion is bytes. Corrected
accounting (`scripts/campaign/fp8_decode_bytes.py`, H100 3352 GB/s; 26B active 3.822 G = dense-linear 1.656 + lm_head 0.738
+ experts 1.427 G, 0.178 G per expert):

Expert bytes per step. GROUPED streams each touched expert ONCE (union `n*(1-(1-k/n)^B)`);
PER-SLOT streams one expert per slot-dot, `B*top_k` times:

| B | union | B·k | bf16 grouped | fp8 grouped | fp8 per-slot | fp8 per-slot vs bf16 grouped |
|---|---|---|---|---|---|---|
| 1 | 8.0 | 8 | 2.66 GiB | 1.33 GiB | 1.33 GiB | 0.50x |
| 4 | 29.1 | 32 | 9.68 GiB | 4.84 GiB | 5.32 GiB | 0.55x |
| 16 | 82.4 | 128 | 27.39 GiB | 13.70 GiB | 21.27 GiB | 0.78x |
| 32 | 111.8 | 256 | 37.15 GiB | 18.57 GiB | 42.54 GiB | **1.15x** |

Whole step (lm_head bf16 in both):

| B | bf16 grouped | fp8 as shipped | fp8 + grouped (#62) | + fp8 lm_head (#93) |
|---|---|---|---|---|
| 1 | 7.12 GiB / 2.28 ms | 4.25 / 1.36 (**+40.3%**) | 4.25 / 1.36 (+40.3%) | 3.56 / 1.14 (+50.0%) |
| 4 | 14.14 / 4.53 | 8.23 / 2.64 (+41.8%) | 7.76 / 2.48 (+45.1%) | 7.07 / 2.26 (+50.0%) |
| 16 | 31.85 / 10.20 | 24.19 / 7.75 (+24.1%) | 16.61 / 5.32 (+47.8%) | 15.93 / 5.10 (+50.0%) |
| 32 | 41.61 / 13.33 | 45.46 / 14.56 (**-9.3%**) | 21.49 / 6.88 (+48.3%) | 20.80 / 6.66 (+50.0%) |

The per-slot walk costs `B*top_k` expert streams, growing LINEARLY in B, while grouped saturates as
the union approaches all 128 experts. **They cross at B ~ 28.** Below it fp8 wins on bytes even
per-slot; at B=32 it loses by 9.3%, because 256 slot-streams exceed 2x the 111.8-expert union.

So bytes explain a modest B=32 regression but **not** the measured ~5x. The dominant term is
efficiency: the per-slot dot walk gives up the grouped tensor-core GEMM entirely.

### 9.3 #62 located exactly: the grouped decode route is bf16 all the way down

`crates/devgen/src/lib.rs` decode MoE GLU:

```rust
let c_glu = if fp8 {
    // fp8 path: separate norm + expert GLU (no fused fp8 norm variant)
    ... RmsNorm ... MoeExpertGluGemmaFp8 ...
} else {
    // GROUPED DECODE (PLOW_GEMMA_MOE_DEC_GROUP=min rows) ...
    MoeAlignGemmaPf + MoeExpertGluNormGemma with t[6]=moe_meta, t[7]=moe_rowtok, i[6]=min
};
```

The grouped branch is inside the **bf16-only `else`**. The fp8 arm never emits `MoeAlignGemmaPf`, so
`moe_dec_group` stays `None` and the DOWN op runs ungrouped too. It is not only an emitter gate:
`runtime/nvidia/op_moe.cuh`'s grouped decode body delegates to the **bf16** grouped prefill GEMM,

```c
static __device__ void d_moe_dec_group_glu_gemma(bf16* fug, const bf16* resid, ...) {
    plow_moe_stage_xn(xn, resid, gamma, H, nrow, eps);
    d_moe_group_glu_gemma_pf(fug, xn, ewt, meta, row_token, ...);   /* bf16 body */
}
```

but the **w8a8 twins already exist**: `d_moe_group_glu_gemma_pf_w8a8` (op_moe.cuh:2862) and
`d_moe_group_down_gemma_pf_w8a8` (:2952), both commented "beat26b", mma.sync.m16n8k32, BK8=64,
per-token/-row e4m3 activation. So #62 is well-scoped, not blocked:

1. device: `d_moe_dec_group_glu_gemma_w8a8` + down twin, staging `xn` and quantising it to e4m3
   per row (the w8a8 prefill bodies want a gathered `xq8` by `row_token` + per-token `ascale`);
2. emitter: hoist the grouped branch so the fp8 arm can take it;
3. worth **+48.3% of decode bytes at B=32** against fp8-as-shipped's -9.3% -- a 2.1x byte swing --
   and it is the lever that should close the measured 5x.

### 9.4 The route decision under FP8: the tuner CANNOT pick cuBLASLt, in three layers

"Allow cuBLAS to be used based on the plow tuner's decision" is answerable for BF16 and, today,
structurally unanswerable for FP8. The Lt integration is bf16-only in three independent places:

1. `plow_asset::segment_roles::cublaslt_prefill_bf16` -- the policy predicate, bf16 by name and by
   its `(n, k)` shape tables.
2. `devgen::dense_cublaslt::prefill_eligible` -- admits only `DevOp::Gemm | GemmMed | GemmSmall`.
   A test (`prefill_projection_rejects_bias_fp8_and_nonisolated_packets`) asserts `GemmFp8` is
   rejected, so this is intended, not incidental.
3. `plowrt::device::cuda::lt` -- **every** `cublasLtMatrixLayoutCreate` call passes dtype `14`
   (BF16) hardcoded; there are no FP8 scale-pointer attributes (cuBLASLt FP8 requires
   A/B_SCALE_POINTER and is TN-only); stored algorithms are filtered `if rec.dtype != "bf16" {
   continue }`; and the failure strings read "no supported BF16 cuBLASLt algorithm".

So for FP8 the native kernels are not the tuner's *choice* -- they are the only implemented route,
and every FP8 number in this campaign is plow-native by construction. The measured
cuBLASLt-vs-native route decision is a BF16 result (§6), and that is the honest scope of it.
Extending the tuner to FP8 means implementing an FP8 Lt path (a datatype/scale/layout change in
`lt.rs` plus an fp8 admission predicate), not flipping a knob.

## 10. The 1k/2k/4k/8k/16k x C1/C4/C16/C32 ladder on the 2176 packet, paired same-session

Packet `p26lad2` (sha `ab183a7fa4525492`), rungs `[128, 256, 512, 1024, 1152, 2048, 2176, 4096,
4224]`, all nine structurally complete. vLLM 0.28 measured back to back on the same card in the same
session, same client, same dataset, prefix caching off on both sides. Raw CSVs in
`plans/ladder-2176/`. A cell is 4/4 only when plow wins TTFT, TPOT, p99 ITL and out_tok_s.

**Result: 0/20 cells at 4/4.** Per metric: **TTFT 10/20, p99 ITL 13/20, TPOT 2/20, tok/s 0/20.**

### 10.1 The 2176 rung delivered, and it is measurable

2048/C1 TTFT is **56.8 ms**. Before the fix a 2049-row request had no qualified rung below 4224, and
a 2049-row request on the 4224 rung does the same work as a 4096-row one, which measures **102.5 ms**
in the very next cell. So the rung is worth **~45 ms, a ~1.8x TTFT improvement at the 2k cell**, and
2048 now sits correctly between 1024 (38.2) and 4096 (102.5) instead of collapsing onto the 4096
cost. Padding at the 2k cell is 2049 -> 2176 = 6.2%, against 106% before.

### 10.2 C1 / C4 (realtime profile, nprompt 32), TTFT / TPOT / p99 ITL / tok/s

| in | C | TTFT plow/vLLM | TPOT | p99 ITL | tok/s | score |
|---|---|---|---|---|---|---|
| 1024 | 1 | **38.2 / 43.1** | 5.51 / 5.07 | **5.57 / 5.84** | 173.3 / 186.2 | 2/4 |
| 1024 | 4 | **78.4 / 90.9** | 8.53 / 7.56 | 41.2 / 8.5 | 439.8 / 487.1 | 1/4 |
| 2048 | 1 | 56.8 / 54.4 | 5.53 / 5.08 | **5.62 / 5.82** | 168.6 / 183.0 | 1/4 |
| 2048 | 4 | **115.4 / 134.3** | 8.85 / 7.55 | 61.3 / 8.5 | 412.4 / 468.2 | 1/4 |
| 4096 | 1 | 102.5 / 93.4 | 5.54 / 5.09 | **5.63 / 5.89** | 158.7 / 173.0 | 1/4 |
| 4096 | 4 | **196.7 / 225.4** | 9.59 / 7.94 | 105.1 / 8.5 | 361.4 / 414.5 | 1/4 |
| 8192 | 1 | 209.2 / 179.8 | 5.59 / 5.09 | **5.66 / 5.90** | 139.2 / 154.9 | 1/4 |
| 8192 | 4 | **334.0 / 457.1** | 12.17 / 8.56 | 112.0 / 9.1 | 271.9 / 331.1 | 1/4 |
| 15000 | 1 | 426.4 / 351.5 | 5.66 / 5.09 | **5.72 / 6.04** | 111.7 / 128.3 | 1/4 |
| 15000 | 4 | **628.5 / 811.8** | 17.51 / 10.83 | **124.6 / 147.9** | 178.6 / 233.8 | 2/4 |

TTFT 6/10, p99 ITL 6/10, TPOT 0/10, tok/s 0/10.

**TTFT inverts with concurrency.** plow loses every C1 cell from 2048 up (-4% at 2048 to -21% at
15000) and wins every C4 cell (up to -27% at 8192, -23% at 15000). Adaptive interleave is paying
under load and costing nothing at C1.

**p99 ITL splits cleanly by concurrency.** plow wins every C1 cell -- MULTISTEP=0 removed the wave,
as #49 intended -- and loses every C4 cell but 15000, at 41-125 ms against vLLM's flat 8.5-9.1.
Median ITL is healthy (7.9-8.5), so this is a TAIL: prefill interrupting decode at C4, not the
multistep wave. It is the largest addressable deficit on the C1/C4 half.

### 10.3 C16 / C32 (high_concurrency profile, nprompt 64)

| in | C | TTFT plow/vLLM | TPOT | p99 ITL | tok/s | score |
|---|---|---|---|---|---|---|
| 1024 | 16 | **160.9 / 205.7** | 13.68 / 10.12 | 109.2 / 18.6 | 1070 / 1372 | 1/4 |
| 1024 | 32 | 1593.6 / 356.6 | 13.60 / 13.51 | 108.9 / 68.8 | 1089 / 1965 | 0/4 |
| 2048 | 16 | **244.2 / 346.6** | 16.38 / 10.72 | 113.1 / 95.2 | 873 / 1197 | 1/4 |
| 2048 | 32 | 2044.9 / 546.8 | 16.73 / 16.15 | **112.4 / 131.8** | 872 / 1566 | 1/4 |
| 4096 | 16 | **403.9 / 523.7** | 21.98 / 13.72 | **119.7 / 142.2** | 635 / 901 | 2/4 |
| 4096 | 32 | 2916.8 / 899.7 | 22.39 / 22.03 | **117.8 / 152.8** | 637 / 1097 | 1/4 |
| 8192 | 16 | **811.5 / 928.3** | 35.13 / 20.97 | **133.7 / 159.2** | 384 / 568 | 2/4 |
| 8192 | 32 | 5088.1 / 1680.1 | **35.75 / 36.33** | **135.3 / 160.5** | 385 / 643 | 2/4 |
| 15000 | 16 | 2818.2 / 1573.9 | 55.11 / 34.92 | **162.0 / 190.7** | 204 / 339 | 1/4 |
| 15000 | 32 | 10787.0 / 3114.2 | **55.07 / 62.73** | **161.7 / 189.4** | 204 / 365 | 2/4 |

TTFT 4/10, p99 ITL 7/10, TPOT 2/10, tok/s 0/10.

**The whole C32 column is the 16-slot cap, not a kernel result.** This packet serves 16 slots, so a
C32 client queues 16 requests behind a full generation. The tell is decisive and is exactly the
fingerprint in `plow-vs-vllm-metric-fingerprints`: **C32 tok/s equals C16 tok/s to three digits in
every rung** -- 1088.5/1070.1, 872.2/872.9, 636.5/634.7, 384.8/384.3, 204.1/203.9. C32 TTFT is
correspondingly 3.0-4.5x vLLM's. `gemma4-26b-a4b.h100.bf16-c32-16k.toml` is the 32-slot packet and
the recipe header records that it is NOT a uniform win (better at 128/1024 in, worse from 4096 up),
so the honest reading is: **pick the packet by prompt length, and do not read the C32 column of a
16-slot packet as a throughput result.** The C16 column is the real high-concurrency evidence, and
there plow wins TTFT 4/5 and p99 ITL 3/5.

### 10.4 What the board says the levers are

1. **TPOT is the systematic loss: 2/20, and it fully explains tok/s 0/20.** With 1 prompt to 127
   output tokens, out_tok_s is a TPOT readout, not an independent metric -- these are ONE finding.
   At C1 the gap is a tight 9-11% (5.51-5.66 vs 5.07-5.09). The two TPOT wins are both at C32/8192+,
   where vLLM is itself degrading.
2. **p99 ITL at C4/C16 with short prompts.** 1024/C16 is 109.2 vs 18.6 -- the worst ratio on the
   board (5.9x). Median ITL is fine, so it is prefill interference in the tail. #44 (pipelined
   decode / prefill overlap) is aimed exactly here.
3. **TTFT at C1 for long prompts** (-16% at 8192, -21% at 15000) while C4 wins. Consistent with
   prefill at only 33% of the compute roof at C1 (the run's own roofline line), i.e. a single
   request does not fill the machine.

The roofline diagnosis printed alongside the run agrees: decode at **43.0% of the memory roof** and
prefill at **33.1% of the compute roof** at C1 -- neither phase is near its ceiling at low
concurrency, which is where the TPOT and C1-TTFT losses sit.

## 11. The FP8 kernel set with DIMS and RUNGS, and the roofline comparison per kernel

Source: each emitted packet's own `dispatch_audit`, which carries per-op `m`/`n`/`k`, the rung `t`,
`insts` (layer multiplicity), `bytes` moved, `tile` and `occupancy`. Reproduce with
`scripts/campaign/fp8_kernels.py` and `fp8_kernel_floor.py`.

Roofline, H100 SXM5 (3352 GB/s; bf16 989 / fp8 1979 TFLOP/s):
`FLOPs = 2*m*n*k`, `AI = FLOPs/bytes`, ridge = peak/BW = **bf16 295, fp8 590 FLOP/B**.
`bytes` in the audit is PER INSTRUCTION, so the floor is `max(bytes/BW, FLOPs/peak)` and the
per-layer total is that times `insts`. **ROUTE is derivable, not guessed**: cuBLASLt can only ever
appear on `Gemm`/`GemmMed`/`GemmSmall` (`prefill_eligible`), so every fp8 op is native by
construction (§9.4).

### 11.1 26B-A4B FP8 — prefill, 8 rungs each (128, 256, 512, 1024, 1152, 2048, 4096, 4224)

| op | N | K | insts | tile | occ | AI | bound | floor x insts | route |
|---|---|---|---|---|---|---|---|---|---|
| `Gemm` (lm_head) | 262144 | 2816 | 1 | 256x256 | 0.97 | **1.0** | **mem** | 0.441 | Lt-eligible, **bf16** |
| `GemmFp8` o_proj (full) | 2816 | 8192 | 5 | 256x256 | 0.71 | 1679 | compute | 0.492 | native |
| `GemmFp8` qkv (full) | 8192 | 2816 | 5 | 256x256 | 0.93 | 1679 | compute | 0.492 | native |
| `GemmFp8` o_proj (slide) | 2816 | 4096 | 25 | 256x256 | 0.71 | 1394 | compute | 1.231 | native |
| `GemmFp8` q_proj (slide) | 4096 | 2816 | 25 | 256x256 | 0.82 | 1394 | compute | 1.231 | native |
| `GemmGluFp8` gate+up | 2112 | 2816 | 30 | fused | — | 939 | compute | 0.762 | native |
| `GemmFp8` down | 2816 | 2112 | 30 | 256x256 | 0.71 | 1056 | compute | 0.762 | native |
| `GemmFp8` kv (slide) | 2048 | 2816 | 50 | 256x256 | 0.82 | 1040 | compute | 1.231 | native |
| `GemmFp8` k (full) | 1024 | 2816 | 5 | 256x256 | 0.91 | 690 | compute | 0.062 | native |

Every fp8 linear is **compute bound** (AI 690-1679 vs ridge 590) -- so fp8's 2x peak is the whole
prize, and it lands: see 11.4.

### 11.2 26B-A4B FP8 — decode, rungs 1, 2, 4, 8, 16

| op | N | K | insts | AI | bound | floor x insts | route |
|---|---|---|---|---|---|---|---|
| `Gemv` (lm_head) | 262144 | 2816 | 1 | **15.9** | mem | **0.443** | **bf16** |
| `GemvGluFp8` gate+up | 2112 | 2816 | 30 | 15.8 | mem | 0.108 | native |
| `GemvFp8` o (full) | 2816 | 8192 | 5 | 31.5 | mem | 0.035 | native |
| `GemvFp8` qkv (full) | 8192 | 2816 | 5 | 31.5 | mem | 0.035 | native |
| `GemvFp8` o (slide) | 2816 | 4096 | 25 | 31.4 | mem | 0.088 | native |
| `GemvFp8` q (slide) | 4096 | 2816 | 25 | 31.4 | mem | 0.088 | native |
| `GemvFp8` kv (slide) | 2048 | 2816 | 50 | 31.2 | mem | 0.088 | native |
| `GemvFp8` down | 2816 | 2112 | 30 | 31.2 | mem | 0.055 | native |
| `GemvFp8` k (full) | 1024 | 2816 | 5 | 30.7 | mem | 0.004 | native |

Every decode op is **memory bound** (AI 16-32 vs ridge 590), confirming decode time tracks weight
BYTES in both precisions (§9.2).

### 11.3 12B FP8 — the shapes differ; the fused GLU is the biggest decode item

Prefill adds `GemmGluFp8` 15360x3840 (48 insts, **only at the 4096 rung**) and
`GemmFp8` 15360x3840 at 96 insts; `GemmFp8` 512x3840 is the one **memory-bound** prefill op
(AI 429 < 590). Decode: `GemvArgmax` 262144x3840 = **0.603 ms** (bf16) and
`GemvGluFp8` 15360x3840 x48 = **1.698 ms** -- on the 12B the fused GLU is larger than lm_head.

### 11.4 Roofline verdict: the audited-linear floor, fp8 vs bf16 (same tree, same rungs)

| packet | phase | lm_head | other linears | total |
|---|---|---|---|---|
| 26B fp8 | prefill | 0.441 | **6.262** | 6.703 |
| 26B bf16 | prefill | 0.441 | **14.055** | 14.496 |
| 26B fp8 | decode | 0.443 | **0.501** | 0.944 |
| 26B bf16 | decode | 0.443 | **1.678** | 2.121 |
| 12B fp8 | prefill | 0.601 | 58.275 | 58.875 |
| 12B fp8 | decode | 0.603 | 3.282 | 3.885 |

* **26B prefill: 14.055 -> 6.262 ms = 2.24x better on fp8.** The linears are compute bound, so this
  is fp8's 2x tensor-core peak being collected, slightly better than 2x because the fp8 packet also
  FUSES gate+up (`GemmGluFp8`, 30 insts) where the bf16 packet runs them unfused as 60 separate Lt
  GEMMs -- the deliberate bf16 choice from §1 (Lt gate/up + a separate GeGLU beat the fused role).
* **26B decode: 1.678 -> 0.501 ms = 3.35x better on fp8** on the audited linears.
* These are FLOORS over audited linear ops only. They EXCLUDE the MoE expert GEMMs entirely
  (`dispatch_audit` has no MoE row at all -- the blind spot recorded in §6), and exclude attention,
  norms and rope. The 12B number is not comparable to the 26B's for that reason: the 12B is dense,
  so its FFN is audited, while the 26B's routed experts are not.

### 11.5 The finding that matters most: lm_head is the largest single decode kernel, and it is BF16

`lm_head` is **47% of the 26B FP8 decode linear floor** (0.443 of 0.944 ms) and 21% of the bf16
one -- it does not shrink when everything else does, so its share doubles under fp8. It is memory
bound at AI 15.9, so quantising it is worth **half its floor**:

* 26B: 0.443 -> ~0.221 ms **per decode step**, and 0.441 -> ~0.220 ms per request at prefill.
* 12B: 0.603 -> ~0.302 ms per decode step.

Against the **measured** 26B C1 TPOT gap of **0.42 ms/step** of context-independent work (§12),
fp8 lm_head alone is worth **~53% of that gap from one change**. That makes #93, not #62, the
cheapest TPOT lever on the board -- and unlike #62 it needs no new grouped kernel, since a
`GemvFp8` body at N=vocab is the same shape as the eight that already exist.

Corroboration that this is real and not an artefact of my reading: the FP8 recipe's own header says
so in its first line -- *"FP8 (PTPC) weights: W8A8 prefill, W8A16 decode, BF16 KV + lm_head."*
It is a deliberate, documented exclusion, not an oversight.

## 12. The TPOT gap decomposed: TWO causes in two regimes (corrects "it is all KV traversal")

Fit `TPOT(ctx) ~ fixed + slope*ctx` on the §10 ladder, both stacks, per concurrency
(`tpot_split.py`). A fixed offset and a slope have DIFFERENT fixes, so they must not be pooled.

| C | fixed ms plow/vLLM | slope us per 1k ctx plow/vLLM | slope ratio | gap@1k | gap@15k | fixed share of gap@1k |
|---|---|---|---|---|---|---|
| 1 | 5.50 / 5.08 | ~0 / ~0 | — | 0.44 | 0.57 | **96%** |
| 4 | 7.38 / 7.06 | 0.7 / 0.2 | 2.76x | 0.97 | 6.68 | 33% |
| 16 | 10.30 / 7.10 | 3.0 / 1.8 | **1.65x** | 3.56 | 20.19 | 90% |
| 32 | 10.59 / 8.61 | 3.0 / 3.5 | 0.84x | 0.09 | -7.66 | (slot-capped, ignore) |

**At C1 the gap is 96% a FIXED ~0.42 ms/step, not KV traversal.** Both stacks have essentially zero
context slope at C1, and the reason is structural: 28 of 30 Gemma-4 layers are sliding with a 1024
window, so past 1024 only the two full-attention layers add KV, which at B=1 is negligible. This
**corrects** the earlier campaign note "the TPOT gap is all KV traversal" -- that holds at batch and
is false at C1. 0.42 ms over 30 layers is **~14 us per layer**, and it is 8% of a 5.5 ms step.

**At C16 KV traversal takes over.** plow's effective KV throughput is **1/1.65 of vLLM's**; the
fixed part stays ~3.2 ms but the slope adds ~17 ms by 15k, which is the whole 3.56 -> 20.19 ms
widening. (Converting the slope to absolute GB/s using nominal 240 KiB/token gives an unphysical
695% of peak at C1 -- which is itself proof that the effective bytes/token is far below nominal
because of the sliding window. Trust the 1.65x RATIO, not an absolute achieved-bandwidth number.)

Do not read the C32 row: this packet is slot-capped at 16 (§10.3), so its two apparent TPOT "wins"
at 8192/15000 are vLLM degrading, not plow improving.

### 12.1 What to attack, and what is already closed

**C1, find 0.42 ms/step.** The register/occupancy route is CLOSED by arithmetic (128-reg cap costs
1.3-1.9x in spill against ~463 regs of live state) and decode GRAPH SHAPE is fully closed -- op
count, op width and path depth each refuted, and spine fusion cost +0.29 ms. So this is per-step
entry and per-layer glue, not the graph:
1. **#93, fp8 lm_head: ~0.221 ms/step, = ~53% of the gap, from one change** (§11.5). Cheapest lever
   on the board and needs no new kernel shape.
2. **#71, the megakernel entry (~1.3 ms/step at B=32, cause unidentified)** -- the largest named
   fixed cost.
3. The B=1 MoE glue: 8 of 128 experts are touched yet router + align + combine run every layer.

**C16, close the KV slope** -- worth ~17 of the 20 ms at 15000/C16, far more than anything at C1:
1. **#68** (riding decode rows through the decode attention kernel), plus #12/#59.
2. **An FP8 KV cache, which was NOT on the task list and should be.** `precision.kv_enc` is **bf16
   in every packet here**, fp8 and bf16 alike, and vLLM is also serving bf16 KV
   (`--kv-cache-dtype` unset), so this is not a place plow is behind -- it is an unexploited axis on
   the dominant batched term. Caveat: it changes numerics, and an apples-to-apples claim would have
   to grant vLLM the same, so it is a capacity/quality decision rather than a free win.

Note the interaction with §10: prefill is 85% of the C32 wall, so TPOT work pays at C1-C16 where
decode is the wall. The C1 fixed cost is the cheapest 4/4 flip (those cells already win p99 ITL and
need only TPOT + tok/s, which are ONE metric at 1:127); the C16 KV slope is the largest absolute
prize.

## 13. FP8 SINGLE BLOCK, both layer kinds: FP8 as shipped is a large REGRESSION vs BF16

Same harness as §7 (`block_e2e.sh`), same layers (L0 sliding, L5 full attention), same grid
(B 1,4 x T 128,1024), same vLLM decoder-layer baseline. FP8 configuration is the recipe's own emit /
objects env (`PLOW_FP8` + `PLOW_W8A8`, `PLOW_SEG_PURE_GEMM=fp8`, `PLOW_BUILD_W8A8`, ...).

**Two setup facts worth recording, both of which cost a run to find:**

1. The fp8 WEIGHTS are a separate pre-quantised MIXED checkpoint
   (`/opt/dlami/nvme/tmp/fp8-campaign/gemma26b-w8a8/checkpoint-mixed`): the bf16 shards symlinked
   plus one `fp8-model.safetensors`. Pointing the block at the plain bf16 HF dir fails with
   `MISSING WEIGHT: fp8/model.language_model.layers.0.experts.gate_up_proj`. That error is a THIRD
   independent confirmation of §9.1 -- an fp8 *expert* weight is a tensor the runtime expects by
   name, so the experts are quantised by design. The mixed dir is a symlink farm with no HF index
   and non-standard shard names, so it cannot be the vLLM half's model dir; point only plowrt at it
   with `PLOW_CHECKPOINT` and leave the HF dir for `plowc` and vLLM.
2. It also explains why lm_head stays bf16 (#93): lm_head lives in the bf16 shards.

### 13.1 L5, full attention (median of 3, p95 within 0.3%)

| B | T | decode plow | decode vLLM | ratio | prefill plow | prefill vLLM | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 128 | 1179.60 us | 352.13 us | **3.35x** | 1.00 ms | 3.03 ms | **0.33x** |
| 1 | 1024 | 1208.17 us | 351.49 us | **3.44x** | 1.70 ms | 3.13 ms | **0.54x** |
| 4 | 128 | 1179.65 us | 471.23 us | **2.50x** | 4.02 ms | 3.04 ms | 1.32x |
| 4 | 1024 | 1208.19 us | 480.51 us | **2.51x** | 6.78 ms | 5.13 ms | 1.32x |

L0 sliding is the same shape: decode 2.31-3.13x (median 2.99x), prefill 0.34 / 0.77 / 1.31 / 2.17x.

### 13.2 Against plow's OWN bf16 block (§7.1, same layer, same grid)

The vLLM baseline reproduces across the two sessions to **0.03% at decode** (352.13 vs 352.22 us)
and 1.3% at prefill (3.03 vs 2.99 ms), so the plow-side delta is real and not drift.

| B | T | decode bf16 -> fp8 | | prefill bf16 -> fp8 | |
|---|---|---|---|---|---|
| 1 | 128 | 220.19 -> 1179.60 us | **5.36x worse** | 0.60 -> 1.00 ms | 1.67x worse |
| 1 | 1024 | 227.00 -> 1208.17 us | **5.32x worse** | 1.05 -> 1.70 ms | 1.62x worse |
| 4 | 128 | 306.84 -> 1179.65 us | 3.84x worse | 2.40 -> 4.02 ms | 1.68x worse |
| 4 | 1024 | 323.01 -> 1208.19 us | 3.74x worse | 4.18 -> 6.78 ms | 1.62x worse |

### 13.3 Read against the roofline (§11.4), which predicted the opposite

| phase | roofline floor says | block measures | gap to roofline |
|---|---|---|---|
| prefill | fp8 **2.24x BETTER** than bf16 | fp8 **1.6x WORSE** | ~3.6x off |
| decode | fp8 **3.35x BETTER** than bf16 | fp8 **3.7-5.4x WORSE** | ~12-18x off |

So the fp8 kernels are collecting almost none of the precision win, and at decode they are far
slower in absolute terms than the bf16 ones they replace. Three things this pins down:

* **Decode is the disaster, and #62 is not the whole story.** The per-slot-vs-grouped argument
  (§9.2) predicts trouble at BATCH, yet fp8 decode is already 5.36x worse at **B=1**, where the
  expert union equals top_k and grouping is irrelevant. So a second, batch-independent defect sits
  in the fp8 decode expert path -- consistent with the recipe's "W8A16 decode" (8-bit weights fed to
  16-bit math needs an in-register dequant per use) and with the emitter comment "fp8 path: separate
  norm + expert GLU (no fused fp8 norm variant)". Note the fp8 decode time is also FLAT in B
  (1179.6 at B=1 vs 1179.65 at B=4) and flat in T -- a signature of a fixed per-step cost that
  dominates everything else, not of a bandwidth or batching effect.
* **Prefill still beats vLLM at B=1** (0.33x / 0.54x) and loses at B=4 (1.32x), so the fp8 prefill
  GEMMs are usable but are giving up the 2.24x their compute-bound roofline allows.
* **FP8's case today is capacity, not latency** -- which matches the serving verdict already
  recorded: the 26B bf16 packet peaks at 77-81 GiB of 80 and serves 16 slots, while fp8 weights
  return ~24.5 GiB. That is the reason to keep the arm, and the block numbers say plainly that it is
  not yet a speed win.

One methodology caveat, inherent to the configuration rather than a harness error: the fp8 recipe
ships **no role_files** (packed prefill is unavailable under fp8), so the fp8 block runs a different
object set from the bf16 block, which does build its role objects from the packet. The comparison is
therefore fp8-as-deployed vs bf16-as-deployed, which is the decision-relevant one, not a
single-variable kernel A/B.
