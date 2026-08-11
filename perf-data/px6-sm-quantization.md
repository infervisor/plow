# PX-6 — SM wave quantization on 170 SMs, and the two-arm GEMM split

RTX 5090 (sm_120a, cc 12.0, **170 SMs**, 96 MiB L2, boost 2527 MHz) · CUDA 13.0 / driver 580.159.03 · branch `worktree-px6-sm-quantization` · 2026-07-26

> **Every other Gemma-4-12B number in `perf-data/` is from the 188-SM RTX PRO 6000. Deltas against those rows are invalid.**

## TL;DR

**Premise confirmed. Split killed. Two levers found, plus one incidental finding that is probably worth more than either.**

- Wave quantization is real and large. `down_proj` at chunk 1024 takes **1.07126 ms on 120 SMs and 1.08145 ms on 170** — the extra 50 SMs (29% of the machine) make it 1% *slower*.
- The proposed FFMA-GEMV residue split **cannot** recover it: measured `r_true = 0.063–0.085` against a required 0.50–0.75. Killed by 8–9×.
- It wins at exactly one site, the one the theory predicts: prefill `lm_head` runs a tiled mma GEMM at **M=1** (0.78% row efficiency). Swapping it to the GEMV arm is **1.991 → 1.213 ms, −39%**.
- `PGM_BN=64` is worth **−40…−47%** on q/k/o/down at chunk 128, and **loses at every M for gate/up**.
- **plow's 60.1 ms/chunk prefill constant is this underfill floor**, not launch overhead — computed from measured τ over the real layer mix it brackets **53.9–60.8 ms**, i.e. 90–101% of the constant.
- **The prefill bucket ladder is on the wrong rungs.** `[128,512,1024,2048,4096,8192]` is powers of two; the cost staircase's treads are not. Moving the *same number* of rungs to tread tops cuts covering loss **7.03% → 1.41%** at 170 SMs and **11.41% → 1.41%** at 188. Shipped as `PLOW_PF_LADDER=wave`.

## Thesis

`d_gemm` walks `T = ceil(M/BM)·ceil(N/BN)` output tiles with a grid-stride loop keyed on the packet's `(slice, nblk)` (`op_gemm.cuh:893-900`). Makespan is `ceil(T/P)` tile-times, so the tail wave idles `P·ceil(T/P) − T` SM-slots. `P = 170 = 2·5·17` divides almost nothing.

Proposal: run bulk columns on the mma arm over `P_g` SMs and residue columns on the FFMA GEMV arm over `P_v = P − P_g`, concurrently. plow can express this and cuBLAS cannot — no per-op grid barrier, different blocks run different op bodies simultaneously, and `Builder::split` gives disjoint CU sets on a shared counter.

### The split theorem

With `W = ceil(T/P)`, tail tiles `s = T − (W−1)P`, and `r_true` = residue per-SM rate ÷ mma per-SM rate:

```
split beats baseline  ⟺  r_true > (W−1)/W + s/(W·P_v)
```

That threshold is **strictly decreasing in `P_v`**, and at `P_v = P`:

```
(W−1)/W + s/(W·P)  =  [(W−1)P + T − (W−1)P] / (W·P)  =  T/(W·P)  =  u
```

which is exactly the full-arm-swap condition. **No partial split ever beats simply picking the better arm for the whole op.** The mechanism is dominated by arm selection.

**Corollary — `r_true` is a property of `M`, not of the kernels.** The mma tile computes `BM` rows whether they exist or not; predication suppresses the store and the gmem read, never the mma (`op_gemm.cuh:929-957`). So `r_true ≈ r_raw · BM/min(M,BM)`. Hence the only winning cell in the stack is `lm_head` at M=1.

**Pre-registered prediction** (recorded before running): `r_true ∈ [0.03, 0.35]`; every `W≥2` cell needs `≥0.50`; the split loses everywhere except row-starved M. **Held.**

## Change (surgical)

`runtime/bench/nvidia/px6_wavequant_bench.cu` — new. `runtime/bench/nvidia/gemm_occ_bench.cu` untouched (it is a campaign record). No production kernel or emitter change.

Four things px3 could not see:

1. **L2.** `zg0_bwcal` on this box: **4090 GB/s warm at 32 MB vs 1695.6 GB/s at 2 GB.** px3's `q_proj`/`o_proj` weights are 31.5 MB against a 96 MiB L2 — fully cache-resident across its 30-iteration loop. Here every weight is replicated past 700 MB and cycled per iteration.
2. **M.** px3 measured only M=4096/8192, where `u ≥ 0.94`. The quantization lives at M ≤ 2048.
3. **An oracle grid** `G* | T`, giving the zero-quantization per-tile-per-SM time τ.
4. **A `k_gemv` arm** at identical shape/grid/cold protocol — which is what measures `r`.

## Gates

| gate | result |
|---|---|
| null control (`u=1.000` cell shows zero idle) | **PASS** — 116.5 TFLOPS in exactly one wave |
| cliff | **PASS** — +0.6% work → +65.4% time |
| model agreement | **PASS** — `idle_meas` vs `idle_pred` within 1–2 pts, 20/20 cells |
| τ stability | **PASS** — ~1% per shape across all M; linear in K (3840→138 µs, 8192→289, 15360→534) |
| L2-cold | **ENFORCED** — 700 MB replication + cycling |
| GPU exclusive | **ENFORCED** — all timed runs under `gpulease` |
| numeric oracle / negative control | **NOT RUN** — no production kernel changed; required before any recommendation ships |
| end-to-end | **NOT RUN** — every number here is per-op |

> `/workspace/gpu` did not exist on this box, so `gpulease`'s `flock` target could not be created and the lease was **silently degrading to a no-op for all agents**. Created 0777 as part of this campaign. The E1a bandwidth run predates the fix (GPU was verified idle).

## E1b — the cliff, and a correction to the model

M=128, K=3840, grid=170. Two shapes 0.6% apart in size.

| N | tiles | waves | u | ms | TFLOPS | ratio |
|---|---|---|---|---|---|---|
| 21760 = 170·128 | 170 | 1 | 1.000 | 0.18362 | 116.5 | 1.000 |
| 21888 = 171·128 | 171 | 2 | 0.503 | 0.30368 | 70.9 | **1.654** |

The cliff is real. But the tile-count model predicts `1/u = 1.989×` and measured is `1.654×`: the tail wave's lone block has the whole card's bandwidth to itself, so a tail tile is cheaper than a contended one.

> **The tile-count model over-predicts quantization cost by ~1.5×. Any ceiling computed from `1−u` is an upper bound, not an expectation.**

## E1c — the staircase

`down_proj` M=1024 N=3840 K=15360, T=240.

| grid | waves | u | ms |
|---|---|---|---|
| 40 | 6 | 1.000 | 3.17080 |
| 45 | 6 | 0.889 | 3.16969 |
| 50 | 5 | 0.960 | 2.64290 |
| 55 | 5 | 0.873 | 2.64253 |
| **120** | 2 | 1.000 | **1.07126** |
| 140 | 2 | 0.857 | 1.07281 |
| 160 | 2 | 0.750 | 1.07833 |
| **170** | 2 | 0.706 | **1.08145** |

Perfectly flat between wave boundaries, stepping only when `W` changes — compute-bound quantization, not bandwidth. **Grid 120 → 170 adds 42% more SMs and costs 1% more time.**

## E1c — real shapes, grid=170 vs oracle

| shape | M | tiles | W | u | ms_P | ms_oracle | idle_meas | idle_pred |
|---|---|---|---|---|---|---|---|---|
| o_proj | 128 | 30 | 1 | 0.176 | 0.28753 | 0.28586 | 0.825 | 0.824 |
| down_proj | 128 | 30 | 1 | 0.176 | 0.53238 | 0.52917 | 0.825 | 0.824 |
| synth N=2176 | 128 | 17 | 1 | 0.100 | 0.13943 | 0.13728 | 0.902 | 0.900 |
| q_proj | 1024 | 512 | 4 | 0.753 | 0.57569 | 0.55963 | 0.268 | 0.247 |
| down_proj | 1024 | 240 | 2 | 0.706 | 1.07546 | 1.06708 | 0.300 | 0.294 |
| gate/up | 1024 | 960 | 6 | 0.941 | 1.52650 | 1.55304 | 0.043 | 0.059 |
| gate/up | 2048 | 1920 | 12 | 0.941 | 3.00707 | 3.10493 | 0.028 | 0.059 |

`ms_P ≈ ms_oracle` nearly everywhere. **gate/up — 46% of layer FLOPs — is already well matched to 170** (2.8–4.2% idle), which is what bounds the model-level ceiling.

## E2 — ρ, the decisive number

> **Normalization.** Measured `gemm_ms/gemv_ms` at grid P equals `r_true/u`, **not** `r_true`, because `gemm_ms` already contains the mma arm's own quantization waste. `r_true = (gemm_ms/gemv_ms)·u`. An earlier revision compared the raw ratio against `u`, double-counting `u` and printing SWAP-WINS for a cell 2.7× slower in wall time. Verdicts below are on wall time.

| case | M | u | gemm_ms | gemv_ms | W passes | **r_true** | needs | verdict |
|---|---|---|---|---|---|---|---|---|
| down | 1024 | 0.706 | 1.07627 | 12.07155 | 128 | **0.063** | 0.706 | loses |
| o | 1024 | 0.706 | 0.58890 | 4.99263 | 128 | **0.083** | 0.706 | loses |
| q | 1024 | 0.753 | 0.57492 | 5.09211 | 128 | **0.085** | 0.753 | loses |
| down | 128 | 0.176 | 0.53289 | 1.41931 | 16 | **0.066** | 0.176 | loses |
| **lm_head** | **1** | 0.927 | 1.99149 | **1.21289** | 1 | **1.522** | 0.927 | **WINS** |

The GEMV arm's problem is structural, not tuning: `gemv_walk` makes `ceil(M/GV_MM_MAX) = 128` full passes over the weight matrix at M=1024 (`op_gemm.cuh:118-133`), and `gemv_rows` reads `x` from **global inside the inner loop** with `n` as the outer loop (`op_gemm.cuh:149,175`).

At M=1 the GEMV arm reads 2.01 GB in 1.213 ms = **1657 GB/s = 98% of the measured 1695.6 GB/s ceiling**; the tiled arm manages 60%.

## E3 — BN=64

`PGM_BN=64` (already `-D`-overridable, `op_gemm.cuh:716-718`): regs 94→55, arena 60→45 KiB, occ 1→2.

| shape | M | BN=128 | BN=64 | Δ |
|---|---|---|---|---|
| down_proj | 128 | 0.53238 | 0.28399 | **−46.7%** |
| o_proj | 128 | 0.28753 | 0.15378 | **−46.5%** |
| synth N=2176 | 128 | 0.13943 | 0.07592 | **−45.6%** |
| q_proj | 128 | 0.13966 | 0.08313 | **−40.5%** |
| o_proj | 1024 | 0.58942 | 0.48091 | −18.4% |
| down_proj | 1024 | 1.07546 | 0.92587 | −13.9% |
| gate/up | 128 | 0.25789 | 0.30824 | +19.5% |
| gate/up | 1024 | 1.52650 | 2.03181 | **+33.1%** |
| gate/up | 2048 | 3.00707 | 3.89106 | **+29.4%** |

BN=64 buys finer quantization at a tile-efficiency tax of ~6% (non-GLU) / ~27% (GLU). It wins at M=128 for non-GLU (`tn` doubles while `W` stays 1, so twice as many SMs get work) and loses almost everywhere else.

px3's "small-win-or-wash" verdict was measured at M=4096/8192 where `u ≥ 0.94`. It is **correct for those M and silent about M ≤ 256**.

This is a per-(shape, M) decision, which sm_120 cannot express today: all three GEMM opcodes dispatch to one `d_gemm` body (`interp_sm120.cu:609-614`) and the tile is a compile-time macro.

## The incidental finding — what the 60.1 ms/chunk constant actually is

plow's fitted prefill model is `ttft_ms = 0.112·rows + 60.1·chunks` (`README.md`, `PLOW_PF_CHUNK_COST`).

At chunk t=128, `tm = 1` and every prefill GEMM has `tn ≤ 170`, so **`W = 1` and the op costs exactly one tile-time `τ(K)` independent of N.** Using measured τ and the real layer mix (`config.text_config.layer_types`: 40 sliding hd256/qd4096, 8 full hd512/qd8192):

The emitter puts q/k/v on **disjoint proportional CU sets** (`split3`, `lib.rs:1780-1815`) so they run concurrently — one wave of `tn_q+tn_k`, not two serialized waves. That gives a lower bound; treating them as serialized gives an upper bound:

| model of q/k/v | chunk-128 floor |
|---|---|
| concurrent (what the emitter does) | **53.92 ms** |
| serialized | **60.80 ms** |
| **plow's fitted per-chunk constant** | **60.10 ms** |

**The underfill floor accounts for 90–101% of that constant.**

> **Correction.** The first revision of this campaign reported a single 59.5 ms figure and a "1% match". That was the *serialized* bound — it summed q and k as if each ran alone on all 170 SMs. The honest result is the bracket above. The substantive claim (the constant **is** the floor, not launch overhead) is unchanged and better supported by the bracket; the 1% was the upper bound landing near 60.1 by luck.

τ is per-tile-per-SM and `W=1` holds for any `P ≥ tn`, so this floor is **SM-count-independent** — which is why it fitted as a clean constant on the 188-SM card and reproduces here on 170.

*Caveat: the 60.1 coefficient was fitted on the 188-SM card. Strong evidence, not a controlled comparison; an end-to-end TTFT fit on this GPU is required to confirm.*

## E4 — where the prefill bucket rungs belong

Cost of one launch of `tm` row-tiles is `Σ_op ceil(tm·tn_op/n_cu)·τ_op` — the same staircase. **Rows added inside a tread are free; one row past a tread top costs a whole extra wave of every op that stepped.**

τ needs no measurement: `τ/k` = 0.0355 / 0.0359 / 0.0372 / 0.0373 µs per k-unit at k = 15360 / 8192 / 4096 / 3840 — **linear within 5%**. So the whole ladder is computable at emit time from `(tn, k, glu, n_cu)`.

Scoring rungs by **optimal multi-launch covering** over every prompt length:

| n_cu | cap | shipped | loss | wave ladder (same rung count) | loss |
|---|---|---|---|---|---|
| **170** | 8192 | `[128,512,1024,2048,4096,8192]` | **7.03%** | `[128,512,1408,2176,2688,8192]` | **1.41%** |
| **188** | 8192 | same | **11.41%** | `[128,384,768,2176,3200,8192]` | **1.41%** |
| 170 | 4096 | `[128,512,1024,2048,4096]` | 9.51% | `[128,512,640,2176,4096]` | 3.80% |

Worst shipped cells at n_cu=170:

| L | shipped plan | shipped | best | loss |
|---|---|---|---|---|
| 640 | 128+512 | 140.67 | 99.17 | **+41.9%** |
| 1280 | 512+1024 | 253.39 | 192.61 | **+31.6%** |
| 256 | 512 | 86.75 | 67.47 | +28.6% |
| 1408 | 512+1024 | 253.39 | 199.48 | +27.0% |

**Same rung count — no blob growth, no extra compile time, no runtime change.** Loss cut 5× at 170 SMs and 8× at 188. Not one chosen rung is a power of two (`1408=11·128`, `2176=17·128`, `2688=21·128`, `640=5·128`), and the 170- and 188-SM ladders share only `128` and `2176` — which is exactly why this must be **derived from `n_cu`**, not hardcoded. Note the shipped ladder is *worse* on 188 SMs than on 170, i.e. worse on the card most existing campaigns used.

Shipped as `PLOW_PF_LADDER=wave` (`crates/devgen/src/ladder.rs`). Default off ⇒ byte-identical; all 4 golden-blob tests pass unchanged.

## E5 — block tuner: the microbench predictions on the real runtime

`block_run` (`crates/plowrt/examples/block_run.rs`) drives ONE compiled Gemma-4-12B block on
plow's actual runtime — the megakernel, the global queue, the counter gates. This is the stage
`tuning/README-decode-tuner.md` insists on: *the lab prunes, e2e scores.*

Setup: `plowc --emit devblob --block 0 --max-ctx 4096 --n-cu 170 --gpu rtx5090`, grid=170
confirmed at load, `check` reports finiteness PASS. Two blobs differing **only** in rung
positions — identical packet counts per program.

Two environment fixes were needed and are worth recording:

- `PLOW_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1`. Kernel driver is 580.159.03 but
  `/usr/local/cuda/compat/libcuda.so.1` is 580.167.08 — a compat lib **newer** than the driver,
  which makes `cuInit` return `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`.
- `scripts/block_e2e.sh` is stale: it drives `./target/release/gemma4 --block`, but that binary
  was removed and the emitter now lives in devgen behind `plowc --emit devblob --block`.
- Block mode needs `PLOW_SKIP_COVERAGE=1` — a single-layer blob by construction does not cover
  the checkpoint's other 47 layers, and the coverage gate is unconditional.

### The chunk-128 floor prediction, confirmed on the runtime

Prefill T=128 measures **1.18–1.19 ms per block**. × 48 layers = **56.6–57.1 ms**, inside the
predicted 53.9–60.8 ms underfill bracket and next to plow's fitted 60.1 ms/chunk constant.

### Ladder A/B — off-rung prompt lengths

`--batch 1 --ctx 640,768,896,1152` (none of which is a base rung). Decode is unchanged
throughout (~410 µs), as expected — the ladder does not touch decode.

| T | base `[128,512,1024,2048,4096]` | wave `[128,512,640,2176,4096]` | Δ |
|---|---|---|---|
| 640 | 3.69 ms | **2.17 ms** | **−41.2%** |
| 768 | 3.70 | 3.36 | −9.2% |
| 896 | 3.72 | **3.88** | **+4.3%** |
| 1152 | 4.92 | 4.14 | −15.9% |
| | | **mean** | **−15.5%** |

−41.2% at T=640 against the compile-time model's −41.4% prediction. The +4.3% at 896 is a real
regression and is what a ladder trade looks like: the wave ladder gives up the cells where the
power-of-two rungs happen to land well in exchange for the cells where they land badly.

**Honesty note.** Benchmarking at 128/512/1024/2048 — all *base* rungs — is the worst case for
the wave ladder and would flatter base; the four lengths above are off-rung for base and closer
to real prompt lengths. Neither is the population mean; the DP in E4 is.

### Batch decode — combining requests

Same block, base ladder, T=128 (uncontended section):

| B | decode step | tok/s | per-token | vs B=1 |
|---|---|---|---|---|
| 1 | 407.74 µs | 2,452 | 407.7 µs | — |
| 2 | 417.66 µs | 4,788 | 208.8 µs | **1.95× for +2.4% latency** |
| 4 | 417.85 µs | 9,573 | 104.5 µs | 3.90× for +2.5% |
| 8 | 429.10 µs | 18,644 | 53.6 µs | **7.60× for +5.2%** |

At T=2048: B=1 417.58 µs → B=8 454.30 µs, **7.35× for +8.8%**.

Step time is nearly flat because `gemv_rows<MM>` is weight-stationary — each weight row is
loaded once and dotted against all `MM` activation rows — and decode is already at the HBM
roofline (E2: 98% of ceiling at M=1), so extra rows ride along on a saturated weight stream.

**The cliff.** `gemv_walk` (`op_gemm.cuh:118-133`) walks `M` in blocks of `GV_MM_MAX=8`
(`for (; m0 + GV_MM_MAX <= M; ...)` then a remainder call), and each call is a full sweep over
the block's weight rows. **B=9 therefore costs 2 weight passes ≈ 2× the traffic of B=8.** Decode
batch is the same ceil-quantized staircase as the prefill ladder, in the batch dimension:
efficient rungs are multiples of 8. *(Predicted from code, not measured — `PLOW_DECODE_BATCH`
clamps to 1..32 at `crates/devgen/src/lib.rs:3453`, so B=9 is emittable and the cliff is
testable; the README's "B ∈ 1..8" is stale.)*

## E6 — prefill ∥ decode: where the split theorem does NOT apply

The `r` theorem killed the intra-op two-arm split because both arms did **the same work** and
contended for **the same resource**: `r_true = 0.063–0.085`. Neither premise holds for running
a *decode* request beside a *prefill* request — they bottleneck on different hardware. Measured
at grid 170:

| phase | bandwidth | of 1695.6 GB/s | compute | of 220 TFLOPS |
|---|---|---|---|---|
| decode GEMV (down) | 1260 GB/s | **74%** | 1.3 TFLOPS | **0.6%** |
| prefill gate/up GLU (M=1024) | 177 GB/s | **10%** | 155.4 TFLOPS | **71%** |
| **if overlapped** | | **84%** | | **71%** |

Both sums are under 100%: **they fit on one card at the same time.**

### Sizing — how many SMs decode actually needs

| grid | GEMV down GB/s | % of grid-170 | lm_head GB/s | % |
|---|---|---|---|---|
| 48 | 1010 | 80.4% | 1232 | 74.2% |
| 64 | 1117 | 88.9% | 1463 | 88.1% |
| **80** | 1190 | **94.8%** | 1581 | **95.2%** |
| **96** | 1217 | **96.9%** | 1617 | **97.4%** |
| 128 | 1244 | 99.1% | 1643 | 99.0% |
| 170 | 1260 | 100% | 1659 | 100% |

**Decode reaches 95% of its full-grid throughput on 80 SMs and 97% on 96.** Giving decode 96 of
170 costs it ~3% and frees **74 SMs — 44% of the machine** — for prefill.

Meanwhile prefill's own scaling says the top of the machine is nearly worthless to it at the
shapes that underfill: `down_proj` M=1024 runs **1.07486 ms at grid 128 and 1.08505 ms at grid
170** — 42 more SMs make it 1% *slower* (T=240 tiles ⇒ W=2 either way, so the extra blocks add
only contention). `gate/up` does scale to 170 (T=960 ⇒ W=8 at grid 128 vs 6 at 170), so the
split point is per-op, not global.

### What plow does today, and what it could do

`PLOW_PF_BATCH` packs several requests' prefill chunks into one launch (shared weight reads),
and `PLOW_PF_INTERLEAVE=N` admits N prefill rows per tick *under decode load*. Both are
**time-slicing**: prefill and decode take turns on the whole machine.

The mechanism for **spatial** sharing already exists and is already shipped for a different
purpose — `Builder::split` disjoint CU sets + a shared producer counter, exactly the
`GLM_MOE_CORESIDENT` pattern (`crates/devgen/src/mla.rs:1240-1259`, measured −17.4%). Emit the
decode ops on `cus = [0, P_d)` and the prefill ops on `[P_d, n_cu)` with **no counter edge
between them** — they are different requests over different KV, so they are genuinely
independent — and the global queue runs both concurrently in one launch. E1c's FIFO trace
already confirms two adjacent independent ops become co-resident rather than serializing.

`P_d ≈ 96` is the measured starting point. This is a **design backed by measured sizing, not a
measured result** — it needs the emitter change and an end-to-end run before any number is
claimed. The honest risk is that the two arms contend in L2 and on the memory controller in a
way the isolated per-phase numbers do not capture; the 84%-bandwidth sum has no margin for that.

## E7 — the lm_head arm swap, implemented

Recommendation A, built and gated. Prefill emits lm_head at **M=1** over the last prompt row,
but `pick_tile` hands it to the tiled arm, which computes BM=128 rows to keep one.

**Premise verified, not assumed.** The emitted opcode was `GemmSmall` (14) — and on sm_120
`GemmSmall`/`GemmMed`/`Gemm` all dispatch to the *same* `d_gemm` body with the compile-time
128×128 tile (`interp_sm120.cu:609-614`), so the tile opcode is inert and M=1 really does run a
128-row tile. **0.78% row efficiency.**

| change | where |
|---|---|
| `case PLOW_DOP_GEMV` under `#if PLOW_NV_PREFILL && PLOW_NV_PF_GEMV_HEAD`, calling **`gemv_rows<1>` directly** (not `gemv_walk`, which would drag the `{2,4,8}` batched rungs into a register-critical object). Traps on `M!=1`, `i3!=0`, `K%8!=0`. | `runtime/nvidia/interp_sm120.cu` |
| `PLOW_PF_GEMV_HEAD=1` redirects the prefill lm_head emit to `DevOp::Gemv` (`!decode` guard) | `crates/devgen/src/lib.rs` |
| `option(PLOW_NV_PF_GEMV_HEAD)` — ON compiles **both** arms in, so an A/B is one binary + two blobs | `runtime/CMakeLists.txt` |

### Gates

| gate | result |
|---|---|
| cubin byte-identity, flag off | **PASS** — `cmp` against the pre-change build, not a SASS diff |
| blob byte-identity, flag unset | **PASS** — full 12B blob identical to pre-change emit |
| register envelope | **PASS** — 236 → **238** registers, **0 spill**, occupancy unchanged |
| blob delta is surgical | **PASS** — exactly **5 bytes** differ, each `14 → 10` (`GemmSmall`→`Gemv`), one per prefill bucket; decode untouched |
| numeric parity vs f64 ref | **PASS** — relL2 1.653e-3 / 1.644e-3 / 1.614e-3, **identical for both arms to 4 sig figs**; arm-vs-arm 9.0e-7…8.5e-6 |
| negative control | **PASS** — wrong `a_row0` diverges at relL2 **1.459** |
| numeric oracle | **NOT RUN** — `sm120_interp_op_test.cu` does not compile at HEAD (4 "too few arguments" errors in its MoE router wrappers, lines 632/813/849). **Pre-existing**, reproduced at HEAD before this change. Reported, not fixed. |
| end-to-end TTFT | **BLOCKED** — see below |

### The win, and its ceiling

Op-level: **1.991 → 1.213 ms, −39%.** The GEMV arm reads the same 2.01 GB tied-embedding weight
at **1657 GB/s = 98%** of this card's ceiling; the tiled arm manages 60%. The HBM floor is
1.19 ms, so **1.213 ms is within 2% of optimal and the win cannot exceed ~0.78 ms/launch** by
any method short of a smaller head (`PLOW_FP8_HEAD` halves the bytes but is decode-only today).

TTFT share is **computed, not measured**: the block tuner gives 1.18 ms/block at T=128, ×48 =
56.6 ms, so lm_head is ~3.4% of chunk-128 prefill and the swap saves ~**1.3%**. Less at larger
chunks — the row term grows, lm_head does not.

### Two things worth more than the swap

1. **Every chunk pays for lm_head.** The per-bucket instruction stream is immutable, so an
   8-chunk prefill runs lm_head 8 times (~15.9 ms) to use ~1.99 ms of it. Skipping it on
   non-final chunks is a bigger win than the arm swap and is **not** addressed here.
2. **`gemma4_sm120_chat.cu` cannot load any blob current `plowc` emits.** It accepts only
   `PLOW_BLOB_MAGIC = "PLOWDEV\x07"` with no v6/v7 branch, while plowc emits v7
   (`"PLOWDEV\x09"`) — it fails with `bad blob magic`. That is what blocks the end-to-end TTFT
   A/B. Second stale-harness finding, after `scripts/block_e2e.sh` driving the removed `gemma4`
   binary. `block_run` loads v7 fine but block mode deliberately emits no logits, so it cannot
   score lm_head; a `plowrt serve` TTFT A/B remains open.

## Recommendations

| id | what | measured | blocker |
|---|---|---|---|
| **A** ✅ | prefill `lm_head`: emit `DevOp::Gemv` instead of `pick_tile(1, vocab_l, hidden, n_cu)` (`crates/devgen/src/lib.rs:2853-2856`, `lm_m=1`) | 1.991 → 1.213 ms, **−39%** on that op (~−0.78 ms/launch) | `case PLOW_DOP_GEMV` is inside `#if !PLOW_NV_PREFILL` (`interp_sm120.cu:899`) — the GEMV family is not in the prefill cubin, which is already at 236/256 regs |
| **B** | BN=64 prefill object for non-GLU ops at chunk ≤ 256; BN=128 for gate/up and all larger chunks | −40…−47% on q/k/o/down at M=128; projecting measured τ over the layer mix gives a 38.7 ms vs 59.5 ms per-launch floor, **−35% (~−21 ms)** | needs per-op tile selection, which sm_120 cannot express (one body, three alias opcodes, compile-time tile). This is P4 in the design notes |
| **D** ✅ | **wave-aligned bucket ladder** — `PLOW_PF_LADDER=wave`, `crates/devgen/src/ladder.rs` | covering loss 7.03% → 1.41% (170 SM), 11.41% → 1.41% (188 SM), same rung count | **none — shipped in this PR, default off, byte-identical** |
| **C** | prefer larger prefill chunks where the latency budget allows | idle 82–90% at t=128 → 2.8–5.9% at t=2048 | none — policy only |
| **E** | teach the runtime the real per-rung cost: `pick_prefill_bucket` (`plowrt/src/exec/gpu.rs:2525`) minimizes `padded_rows + R×launches`, but padded rows badly mis-estimates the staircase (640 rows costs only +13.7% over 512, while 768 costs +55% over 640). Ship `launch_cost(tm)` per rung in the manifest and minimize *that* | not yet measured | needs a `plow-asset` manifest field + runtime read; designed, not built |
| **KILLED** | two-arm bulk/residue column split with an FFMA GEMV residue | `r_true` 0.063–0.085 vs 0.50–0.75 required | do not build. The threshold identity also shows no partial split beats a full arm swap |

## Caveats

- **Every number here is a per-op microbenchmark.** `tuning/README-decode-tuner.md:31-38` is explicit: the lab prunes, the end-to-end harness scores — `gemv_lab_h100.cu` won 1.4× on every shape in isolation and **lost** in the real megakernel. A and B are hypotheses until `gemma4_sm120_chat` confirms them.
- The tile-count model over-predicts by ~1.5× (E1b). `1−u` ceilings are upper bounds.
- No 170-SM Gemma-4-12B end-to-end baseline exists yet.
- **`PLOW_NV_OCC` (the per-block `%globaltimer`+`%smid` occupancy tracer) was not built.** It was gated on the premise surviving E1, which it did, but the per-op measurements answered the question without it. It remains the right tool for confirming these findings *inside* the megakernel, where gate/straggler time is visible and a microbench is blind.
- `GV_MM_MAX=32` not swept. It would cut GEMV weight passes 4×; even a full 4× improvement leaves `r_true ≈ 0.25–0.34`, still below every `W≥2` threshold. Verdict unchanged.
- Activations stay L2-warm across iterations (realistic — A is small and genuinely hot in the interpreter), so reported GB/s counts weight traffic honestly and activation traffic optimistically.

## Reproduce

```bash
cd /root/plow
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
unset CPATH LIBRARY_PATH LD_LIBRARY_PATH          # nix CPATH collides with CUDA math headers

# E1a — bandwidth denominator + SM count
nvcc -arch=sm_120a -O3 -o /tmp/zg0bw runtime/tests/zg0_bwcal_sm120.cu
perf-data/tools/gpulease px6-e1a /tmp/zg0bw

# the campaign harness
nvcc -arch=sm_120a -O3 -I runtime/common -I runtime/nvidia \
     runtime/bench/nvidia/px6_wavequant_bench.cu -o /tmp/px6
nvcc -arch=sm_120a -O3 -I runtime/common -I runtime/nvidia -DPGM_BN=64 \
     runtime/bench/nvidia/px6_wavequant_bench.cu -o /tmp/px6_bn64

perf-data/tools/gpulease px6-cliff  /tmp/px6      cliff    # null control + cliff
perf-data/tools/gpulease px6-stair  /tmp/px6      stair    # staircase
perf-data/tools/gpulease px6-shapes /tmp/px6      shapes   # premise test
perf-data/tools/gpulease px6-rho    /tmp/px6      rho      # the decisive number
perf-data/tools/gpulease px6-bn64   /tmp/px6_bn64 shapes   # E3
```

Runtime: ~4 min for `cliff`+`rho`, ~6 min for each `shapes`, ~3 min for `stair`.
