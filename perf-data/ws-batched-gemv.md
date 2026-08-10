# WS-batched-gemv — weight-stationary wide GEMV rungs (Gemma-4-12B, sm_120)

Campaign **WS-batched-gemv**, 2026-07-21, @ `eafea6c` (kernel) — measurement
pass on top of the committed
`op_gemm.cuh` per-MM-unroll wide rungs. Box: 1× RTX PRO 6000 Blackwell 96 GB
(sm_120, 188 SMs), driver 580.82.07, CUDA 13.0.

## Thesis

12B multi-user decode is **HBM-bandwidth-bound on the weight read** (22.2 GiB/step).
`gemv_walk` strides the row-block by `GV_MM_MAX`: a batch of B costs
`ceil(B / GV_MM_MAX)` weight passes. At the shipped `GV_MM_MAX=8`, a **B=16 step
re-reads all weights twice** and a **B=32 step four times** — so batching past 8
buys **no aggregate throughput** (same tokens/s, 2–4× the per-token latency).
That is the whole reason serving plateaued at ~97 tok/s (§serving) and lost the
multi-user contest to vLLM. The committed WS rungs give the wide MM ladder a
**shallower unroll** (UN16=4/GLU16=2, UN32=2/GLU32=1) so a `gemv_*_rows<16>` /
`<32>` fits under the 255-reg occ-1 ceiling; a B=16 step then reads the weights
**once**, a B=32 step once. This file measures whether that converts to scaling.

## 1. ptxas — register/spill per MM rung (decode megakernel)

`nvcc -arch=sm_120a -O3 -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DGV_MM_MAX=<MM>`,
symbol `_Z12interp_sm12011PlowProgram` (the one cooperative megakernel; its
register count is the **max over all instantiated rungs**). Cooperative launch
pins the grid at `n_cu` = **1 block/SM**, so any register count ≤255 with 0
spill is free — occupancy is fixed at occ-1 regardless.

| GV_MM_MAX | registers | stack | spill store / load | note |
|---:|---:|---:|---:|---|
| 8  | 212 | 0  | 0 / 0   | shipped default |
| 16 | 234 | 0  | **0 / 0**   | **max at 0 spill** |
| 32 | 255 | 24 | 20 / 40 | fits under the 255 ceiling, minor spill |

**Max MM at 0 spill = 16** (234 regs). MM=32 hits the 255-register wall and
takes a small spill (20 B store / 40 B load) — vastly better than the pre-WS
full-unroll wide rung (1162 B store / 3364 B load, per the old op_gemm.cuh
note) but not zero. The reg count rises 212→234→255 with the ladder; because
occupancy is pinned at occ-1 this does **not** cost B=1 (measured §3).

## 2. Correctness gates — all pass

- **Oracle** (`runtime/tests/batch_decode_sm120.cu`, `-DGV_MM_MAX=32`, every
  batched GEMV vs an f32 CPU reference): `d_gemv`, `d_gemv_qkv` (q/v),
  `d_gemv_glu` at **M ∈ {1,2,4,8,16,32,5,17}** all **PASS**, worst-row
  relL2 ≤ 1.7e-3 (bf16 noise floor, flat across all M — the wide rungs are as
  accurate as the scalar path). `flash_decode` B=2 per-seq + zero-bleed and
  `headnorm_rope` KV-write B=2 also PASS. `RESULT: PASS`.
- **compute-sanitizer** memcheck, M≤32, on the NEW wide rungs:
  `gemv_*_rows<16>` (B=16 blob) and `<32>` (B=32 blob): **ERROR SUMMARY 0
  errors** for both.
- **Batch isolation** (chat harness, 2 distinct prompts → even/odd slots,
  48 tokens): **B=16 and B=32 both `SLOT PARITY: 0 divergent slot-steps`** —
  no cross-user bleed; every even slot reproduces slot-0's stream, every odd
  slot reproduces slot-1's.
- **Bit-identity vs B=1.** Narrow rungs (B≤8, same UN=8 unroll as B=1) are
  byte-identical to the B=1 run. The wide rungs use a **shallower unroll**, so
  their K-reduction sums in a different order and **can flip a near-tie
  argmax**: B=16 slot streams matched the B=1 streams for both prompts across
  48 tokens; B=32's even stream flipped **one** near-tie at token 11
  (28806 vs 5063 — both valid continuations of a periodic synthetic prompt)
  and then cascaded. This is reduction-order, **not** a bug (oracle + sanitizer
  clean) — the same class of effect the campaign already documents for
  prefill-vs-decode. The serving-correctness gate is **isolation** (no bleed),
  which passes; a wide-rung batched stream is numerically equivalent to but not
  bit-identical to a B=1 decode of the same prompt.

## 3. THE curve — aggregate decode tok/s + per-user TPOT vs B

Raw batched-decode microbench (`gemma4_sm120_chat`, chunk-consumed 3840-token
prompt, 128 gen, 16 warmup discarded, steady-state over the timed window),
Gemma-4-12B bf16, ctx≈4k. `agg_tok_s = B / launch_ms`; TPOT = per-slot
ms/token = the launch time (one launch advances all B slots).

| B | MM | weight passes | **agg tok/s** | per-user TPOT ms | tok/s vs B=1 | TPOT vs B=1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 8  | 1 | 53.7  | 18.63 | 1.00× | 1.00× |
| 2  | 8  | 1 | 102.0 | 19.61 | 1.90× | 1.05× |
| 4  | 8  | 1 | 185.5 | 21.57 | 3.45× | 1.16× |
| 8  | 8  | 1 | 325.5 | 24.57 | 6.06× | 1.32× |
| **16** | **16** | **1** | **475.1** | **33.68** | **8.85×** | **1.81×** |
| **32** | **32** | **1** | **506.6** | **63.16** | **9.43×** | **3.39×** |

Batching **scales all the way to B=16** — 475 tok/s at 33.7 ms/user TPOT, which
is still **under the 50 ms ITL SLO**. B=32 saturates at 507 tok/s (compute-bound
now: the FLOP/byte has doubled again and the MM=32 spill bites) with a 63 ms
TPOT that **breaks** the 50 ms SLO. **B=16/MM16 is the throughput-and-latency
sweet spot.**

### The weight-stationary delta — one pass vs multi-pass at the SAME batch

| B | wide (1 pass) | shipped narrow (multi-pass) | Δ tok/s | Δ TPOT |
|---:|---|---|---:|---:|
| 16 | MM16: 475.1 / 33.68 | MM8 (2×): 353.5 / 45.27 | **+34.4%** | **−25.6%** |
| 32 | MM32: 506.6 / 63.16 | MM8 (4×): 386.0 / 82.89 | **+31.2%** | **−23.8%** |
| 32 | MM32: 506.6 / 63.16 | MM16 (2×): 469.5 / 68.15 | +7.9% | −7.3% |

Reading a B=16 step's weights **once** instead of twice is worth **+34% tokens/s
and −26% latency**, exactly the halving of weight traffic the model predicts.
B=32-in-one-pass (MM32) beats B=32-in-two (MM16) by only 8% because MM32 spills
and is already compute-bound — **MM16 is the rung to ship**; MM32 buys marginal
aggregate at an SLO-breaking TPOT.

### B=1 is not taxed — the reg increase is free at occ-1

| MM build | B=1 tok/s | B=1 TPOT ms |
|---:|---:|---:|
| 8  | 53.7 | 18.63 |
| 16 | 53.7 | 18.61 |
| 32 | 53.3 | 18.75 |

The megakernel grows 212→234→255 regs up the ladder, but with occupancy pinned
at occ-1 the B=1 serving path is **unchanged at MM16** (18.61 vs 18.63 ms, noise)
and pays **0.7%** at MM32 (the spill). The old op_gemm.cuh worry that the wide
rung taxes B=1 is **refuted for MM16** by the per-MM shallow unroll.

## 4. Serving — end-to-end concurrency (plow b16-MM16 vs vLLM)

`plowrt serve` + `huggingface/inference-benchmarker` (rev `bad4f947`,
`perf-data/bench_b2_ib.sh`), 4k prompt / 128 out, ConstantVUs 120 s, default
`--slo-ms 250`, TTFT includes queueing. Blob: **B=16 ctx-8k with the MM16
decode cubin** (`/root/gpu-assets-b4/b16-mm16`, 42 GiB KV + 22 GiB weights).
Correctness gate before perf: "The capital of France is Paris." ✓.

| VU | agg tok/s | TTFT p99 ms | ITL p99 ms | failed | tok/req | SLO ITL≤50 & TTFT≤5s |
|---:|---:|---:|---:|---:|---:|:--|
| 1 | 29.3 | 809  | 28.3 | 0 | 129 | **pass** |
| 2 | 45.8 | 1515 | 43.7 | 0 | 128 | **pass** |
| 4 | 67.8 | 3222 | 71.3 | 0 | 127 | ITL fail |
| 8 | 88.6 | 7030 | 94.9 | 0 | 127 | ITL+TTFT fail |
| 16 | *(shed)* | — | — | — | 12 | invalid — arrival-rate admission shed at default slo-ms; not a capacity point |

Saturated peak (shedder relaxed `--slo-ms 100000`, VU16 = full 16-slot occupancy,
clean 127 tok/req): **101.8 tok/s**, ITL p99 138.8 ms, TTFT p99 15.1 s.
**This barely clears B=8's 97 tok/s** — because at saturation the decode kernel
shares the GPU with 16 concurrent 4000-token **prefills**: serving ITL (138.8 ms)
is **4.1× the pure-decode TPOT (33.7 ms)**. The 475 tok/s the kernel can do
(§3, KV pre-built) does **not** survive prefill interleaving. **Serving peak is
prefill-bound, not weight-pass-bound.**

### The one-pass GEMV beats the two-pass B=16 blob at every VU

Both are B=16 blobs; the only difference is the decode cubin (MM16 = 1 weight
pass vs the shipped MM8 = 2 passes, the committed `plow-b16-bfix`):

| VU | mm16 tok/s | bfix(mm8) tok/s | Δ | mm16 ITL p99 | bfix ITL p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 29.3 | 21.9 | **+34%** | 28.3 | 39.7 |
| 2 | 45.8 | 35.7 | **+28%** | 43.7 | 51.4 |
| 4 | 67.8 | 54.9 | +24% | 71.3 | 68.9 |
| 8 | 88.6 | 75.6 | +17% | 94.9 | 115.1 |

Reading the B=16 weights once lifts serving throughput 17–34% and drops per-token
latency — the microbench win survives end-to-end. It **restores the B=16 blob's
SLO-capacity from 1 user (b16-bfix) back to 2** (`ITL p99 43.7 ms < 50` at VU2).

### But SLO-bounded max-users does NOT beat B=8 — prefill is the wall

| engine / blob | max users (both SLOs) | peak tok/s | single-user TTFT / ITL |
|---|---:|---:|---|
| plow B=8 (mm8) | **2** | 97 (VU8) | 0.80 s / 21.9 ms |
| plow B=16-bfix (mm8, 2-pass) | **1** | 93 (VU16) | 0.82 s / 39.7 ms |
| **plow B=16-mm16 (1-pass)** | **2** | 102 (VU16, prefill-bound) | 0.81 s / 28.3 ms |
| **vLLM** | **8** | **239 (VU32)** | 0.38 s / 20.0 ms |

Max-users stays **2** — the WS decode fix removes B=16's latency penalty (ITL
28→94 ms across VU1–8 vs bfix's 40→115) but the **binding constraint above VU2
is prefill, not the GEMV**: ITL p99 crosses 50 ms at **VU4** because new 4000-token
prefills interleave with decode (a B=16 blob runs the full 16-wide kernel even at
partial occupancy, so per-token work does not shrink with fewer users), and TTFT
p99 blows the 5 s SLO by VU8 — both prefill-driven, untouched by a decode-side
GEMV change. vLLM's faster fused prefill + continuous batching keep both SLOs to
VU8.

## 5. Verdict

- **Does batching finally scale? YES at the decode kernel** — the whole campaign
  question. The one-pass wide GEMV takes 12B decode from the 8-wide 325 tok/s
  ceiling to **475 tok/s at B=16** (and 507 at B=32), a **9× tok/s** climb over
  B=1 with per-user TPOT still under the ITL SLO at B=16. The op_gemm.cuh "~1.3×"
  prediction is confirmed at **+34%** (B=16, one pass vs two).
- **Does it close/win the multi-user serving contest? It CLOSES the decode half,
  does NOT win.** End-to-end, MM16 beats the shipped 2-pass B=16 blob 17–34% and
  restores SLO-capacity to 2 users, but does **not** raise max-users past B=8's 2
  or approach vLLM's 8, and the **saturated serving peak barely moves — 102 tok/s
  (VU16) vs B=8's 97** — because at saturation the decode kernel shares the GPU
  with 16 concurrent 4k prefills (serving ITL = 4.1× the pure-decode TPOT). The
  remaining gap is **prefill** (TTFT ~2× vLLM, ITL inflated by prefill
  interleaving) — a separate, unfixed lever. Weight-stationary
  GEMV was necessary but not sufficient: it is the decode-throughput fix the
  capacity report flagged (§7.1), now measured and shipped-ready as `GV_MM_MAX=16`.
- **Ship MM16, not MM32.** MM16 is the max rung at 0 spill (234 regs), free at
  B=1, and its B=16 TPOT (33.7 ms) stays under the 50 ms ITL SLO. MM32 spills
  (255 regs), costs B=1 0.7%, and its B=32 TPOT (63 ms) breaks the SLO for only
  +7% aggregate over 2×MM16 — not worth it for serving.


