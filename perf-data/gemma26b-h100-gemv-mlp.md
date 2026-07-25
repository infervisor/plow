# Gemma-4-26B-A4B on H100 NVL — decode GEMV is MLP-starved, not occupancy-starved

Campaign "beat vLLM on 26B bf16+fp8", round 2 (2026-07-24). Supersedes the diagnosis in
`gemma26b-h100-beat-vllm-campaign.md` and `segmented-decode-26b-h100.md`.

GPU **H100 NVL** (sm_90a, 132 SMs, 60 MB L2), CUDA 13.0 toolkit, driver 570.133.20 +
`/usr/local/cuda-13.0/compat`. Every GPU command under `gpulease`. Model
`/workspace/models/gemma-4-26B-A4B-it` (30 layers, H=2816, I=2112, I_moe=704, 128 experts
top-8, vocab 262144). B=1 decode via `crates/plowrt/examples/step_bench`.

## Headline

The prior campaign concluded that plow's decode is capped by the megakernel's
**1 block/SM occupancy** (12.5 %), that the GEMV family runs at ~21 % of HBM peak, and that
beating vLLM was therefore "not achievable by tuning". **Both premises are wrong.**

1. **Occupancy is not the constraint.** At 1 block/SM a *pure streaming read* on this part
   reaches **2490 GB/s (62 % of the 4023 GB/s spec, 67 % of the 3707 GB/s achievable)** —
   provided each thread keeps 8 loads in flight. Beating vLLM needs only 1579 GB/s.
2. **The prior baseline probe was unfaithful.** `decode_seg_gemv.cu`'s "production
   `gemv_rows`" has **no unrolling** (one load pair per iteration) where the real
   `op_gemm.cuh gemv_rows` pre-issues `GV_UNROLL=8`. It measured 774 GB/s on qkv; the real
   arm does 1347. The "21 % of peak, triply corroborated" figure rests on that probe.
3. **The real defect was one op.** The largest decode op — the fused norm+GLU expert arm,
   33 % of the step — used **scalar 2-byte loads** and recomputed the normalized activation
   from global for *every* output channel.

Fixing that op: **bf16 decode TPOT 9.433 → 7.906 ms @ctx1024 (1.19×)**.

## E1 — the real ceiling (`runtime/nvidia/experiments/hbm_ceiling_h100.cu`)

Pure read of a 1 GiB buffer, GB/s (% of 4023 spec), by blocks/SM × loads-in-flight/thread:

| blk/SM | UN=1 | UN=2 | UN=4 | UN=8 |
|--------|------|------|------|------|
| **1**  | 1222 (30 %) | 1949 (48 %) | 2200 (55 %) | **2490 (62 %)** |
| 2      | 2229 | 3112 | 3269 | 3430 (85 %) |
| 3      | 2958 | 3518 | 3566 | 3625 |
| 4      | 3291 | 3610 | 3647 | 3679 |
| 8      | 3622 | 3702 | 3695 | **3707 (92 %)** |

Achievable read BW = **3707 GB/s**. The 26B token reads 7.63 GB ⇒ a **2.06 ms floor**.
vLLM's 4.833 ms ⇒ vLLM achieves **1579 GB/s = 43 % of achievable**. At 1 block/SM the
hardware already offers 2490 GB/s, i.e. **1.58× more than vLLM needs**. Occupancy was never
the binding constraint; *loads in flight per thread* is.

## E2 — GEMV variant lab (`runtime/nvidia/experiments/gemv_lab_h100.cu`)

Real 26B decode shapes, weights cold (L2 flushed), M=1, GB/s at 1 blk/SM:

| variant | qkv 8192×2816 | o_proj 2816×4096 | gate_up 4224×2816 | down 2816×2112 | moe_gu 11264×2816 | moe_dn 22528×704 | lm_head 262144×2816 |
|---|---|---|---|---|---|---|---|
| A_noun (prior probe's "production") | 775 | 694 | 735 | 582 | 802 | 656 | 1055 |
| B_un8 (**real** production, x global) | 1347 | 1035 | 1091 | 763 | 1455 | 1001 | 2455 |
| C_smemx_un8 (real production, arena) | 1489 | 1343 | 1302 | 869 | 1584 | 1009 | 2264 |
| D_R4_un4 (4 rows/warp, f32 acc) | 1921 | 1440 | 1556 | 1012 | 2037 | 1655 | 3219 |
| E/F packed-bf16 FMA | 1966 | 1475 | 1596 | 1041 | — | — | — |

- **`D_*` is BIT-IDENTICAL to the unblocked body** (relL2 = 0.000e+00 on every shape): the K
  walk and FMA order inside a row are untouched; only which warp owns a row changes.
- Packed `__hfma2` accumulation (E) buys ~2 % for relL2 1.6e-2, and the safer chunk-local
  form (F) 4.8e-3 for the same 2 %. **Not worth it — rejected.**
- Register cost: B 58, C 55, D_R4_un2 64, D_R4_un4 118, E 98 — all under the megakernel's
  own 163-reg non-flash ceiling, so row-blocking is **occupancy-neutral**.
- Occupancy adds little once MLP is fixed: D_R4_un4 at 1 blk/SM ≈ its own 2–3 blk/SM. This
  is the direct refutation of the segmented/lean-object plan, which existed only to buy
  occupancy.

## E3 — the shipped fix

`d_moe_expert_glu_norm_gemma` (`PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA`), the largest decode op:

```c
for (unsigned h = lane; h < H; h += 32u) {          /* 2 B/lane, no unroll        */
    float xn = __bfloat162float(rr[h]) * inv * __bfloat162float(gamma[h]);  /* per channel! */
    accg = fmaf(xn, __bfloat162float(wg[h]), accg);
    accu = fmaf(xn, __bfloat162float(wu[h]), accu);
}
```

Two defects: scalar loads (vs the 16 B `ld_glob8` every other GEMV arm uses), and the
normalized activation recomputed from global for each of the 5632 output channels per layer.
Measured 837 GB/s against the lab's 2037 GB/s on the same bytes.

The twin (`d_moe_expert_glu_norm_gemma_rb`) stages `xn` **once per CTA as f32** — so the value
entering each FMA is bit-identical to the default body — and gives each warp `GV_MOE_RB`
channels = `2*GV_MOE_RB` independent weight streams. `d_moe_expert_down_gemma` got the same
row-blocking (it was already vectorized, so it gains only the extra streams).

Per-lane K partitioning changes (lane-strided → 8-wide vector chunks), so warp-sum rounding
differs from the default body: outputs are numerically equivalent, **not** bit-equal.

## Results (median of 3, sd ≤ 0.01 ms, same toolchain both arms)

| config | control (`PLOW_NV_GEMV_RB=0`) | row-blocked | speedup |
|---|---|---|---|
| bf16 ctx=1024 | 9.433 ms | **7.906 ms** | **1.19×** |
| bf16 ctx=4096 | 9.464 ms | **7.948 ms** | 1.19× |
| fp8  ctx=1024 | 7.471 ms | 7.448 ms | 1.003× |
| fp8  ctx=4096 | 7.499 ms | 7.477 ms | 1.003× |

fp8 is ~neutral by construction: it runs the **fp8** expert arms, which are already
vectorized with x staged in smem (`d_moe_expert_glu_gemma_fp8`). fp8's gap is e4m3→float
conversion cost, not bandwidth — a separate piece of work.

**Correctness:** `gpu_lifecycle` (load → chat-template prompt → greedy decode → unload →
reload → decode again) replies `"Paris"` on both cycles for **both** bf16 and fp8, VRAM
returns to baseline (no leak).

**sm_120 is untouched:** `PLOW_NV_GEMV_RB` defaults to 0 in the sources, and the sm_120
decode and prefill cubins are **byte-identical** pre/post (sha256 `7c1b6708…` / `9380f825…`).

### Negative result — row-blocking the DENSE GEMV arms (kept, compiled out)

Row-blocking `d_gemv` (o_proj/down), `d_gemv_argmax` (lm_head) and `d_gemv_qkv` **wins in
isolation** (lab: qkv 1.43×, lm_head 1.42×) but **loses in the megakernel**, vs the 7.898 ms
MoE-only build at the time:

| dense arm enabled | TPOT | Δ |
|---|---|---|
| none (MoE only) | 7.898 | — |
| lm_head (`d_gemv_argmax`) | 7.923 | +0.025 |
| qkv (`d_gemv_qkv`) | 8.093 | +0.195 |
| o_proj/down (`d_gemv`) | 8.125 | +0.227 |

Not a register effect: a REG:177 variant measured **slower** than a REG:229 one, and the
o_proj/down arm costs 0.227 ms even when its runtime guard (`per >= WARPS*RB`) stops the
blocked loop from ever executing. Two further attempts did not change it — strided row
assignment (fixes the thin-`per` imbalance but destroys the contiguous-span locality: 8.407)
and inlined addressing instead of a pointer array (`gv_rb_smemx_contig`: 8.125 / 7.921).
Left in tree behind `PLOW_NV_RB_GEMV` / `_LMHEAD` / `_QKV` (all default 0) to re-test on
another part. **Understanding why isolation and in-context disagree here is the open thread.**

## Where this leaves the vLLM comparison

| | plow before | plow now | vLLM 0.25.1 | gap |
|---|---|---|---|---|
| bf16 ctx1024 | 9.433 ms | **7.906 ms** | 4.833 ms | 1.64× (was 1.95×) |
| fp8 ctx1024 | 7.471 ms | 7.448 ms | 4.417 ms | 1.69× |

**vLLM is not beaten yet.** But the earlier "not achievable" verdict does not survive E1:
the residual is not a hardware or occupancy wall. Accounting for the remaining 7.9 ms
against the live trace (`live-decode-trace-26b-h100.md`, 29.6 % gate / 67.3 % body / 3.1 % sig):

- **body**, GEMV family: was ~5.6 ms, now ~4.1 ms. The lab says a fully row-blocked family
  costs **3.98 ms** — already under vLLM's *total* 4.833 ms.
- **gate (inter-op counter wait): ~2.7 ms.** Now the single largest remaining term, and
  untouched by this round. 216 packets/step × ~12.6 µs of block-0 idle. This is load
  imbalance plus grid-barrier latency, and it is the next target — by fusing ops (fewer
  gates) and by evening out per-block work, not by a partitioned queue (already refuted in
  `h100-pgq-verdict.md`: confinement is free and locality pays nothing).
- flash decode ~0.44 ms, sig ~0.29 ms.

## Reproduce

```
# ceiling + variant lab
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -arch=sm_90a -O3 \
    -o /tmp/hbm runtime/nvidia/experiments/hbm_ceiling_h100.cu
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -arch=sm_90a -O3 \
    -o /tmp/gl  runtime/nvidia/experiments/gemv_lab_h100.cu
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease e1 /tmp/hbm
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease e2 /tmp/gl

# serving objects (row-blocking is on by default in this script now)
PLOW_ROOT=$PWD scripts/build_sm90a_cubin.sh /workspace/assets/cubin-sm90a-rb/interp_sm90a.cubin
# control: same command with -DPLOW_NV_GEMV_RB=0

nix develop -c cargo build --release -p plowrt --features cuda --example step_bench
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease t \
  target/release/examples/step_bench /workspace/assets/rb-26b/bf16 1 1024 128

# correctness
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease lc env PLOW_GPU_TEST=1 \
  PLOW_GPU_ASSETS=/workspace/assets/rb-26b/bf16 \
  target/release/deps/gpu_lifecycle-* --nocapture
```

---

# Round 3 — where the remaining 7.9 ms actually goes

Two measurement-only builds were added to localize the residual instead of estimating it.

## The gate is NOT the problem (skeleton floor)

`PLOW_NV_SKELETON=1` runs the real decode program's gate/signal protocol with **no op bodies**
(it needs a static-smem pad, added under the same flag, or the body-less kernel reports 8
blocks/SM and mismatches the packet's n_cu=132):

| | ms | % of step |
|---|---|---|
| full decode step | 7.906 | 100 % |
| **gate + signal only, no bodies** | **0.576** | **7.3 %** |

The live trace's "29.6 % gate" is block-0 *idle*, not barrier cost. The interpreter's entire
scheduling protocol — 216 packets/step of counter gates and signals — costs 0.576 ms.
**Fusing ops to cut gate count can therefore win at most ~0.5 ms**, and the 29 % is load
imbalance (blocks waiting on the slowest block of each op), not protocol overhead.

## Per-op wall-clock contribution (`PLOW_NV_ABLATE_OP`)

Skip one opcode's body, keep every gate and signal. The TPOT delta is that op class's real
contribution at the shipped grid, imbalance included — which per-block trace attribution
cannot give. Baseline 7.906 ms:

| op | TPOT without it | contribution | bytes/token | achieved BW |
|---|---|---|---|---|
| GEMV (o_proj + down + **lm_head**) | 6.642 | **1.264 ms** | 2458 MB | 1944 GB/s |
| MOE_EXPERT_GLU_NORM | 6.777 | 1.129 ms | 1900 MB | 1683 GB/s |
| **FLASH_DECODE** | 6.818 | **1.088 ms** | ~231 MB | **212 GB/s** |
| MOE_EXPERT_DOWN | 7.004 | 0.902 ms | 951 MB | 1054 GB/s |
| GEMV_QKV | 7.236 | 0.670 ms | 1383 MB | 2060 GB/s |
| gate+signal skeleton | — | 0.576 ms | — | — |
| HEADNORM_ROPE | 7.826 | 0.080 ms | | |
| GEMV_GLU / FLASH_MERGE / ROUTER / ADD_NORM / GEMV_ARGMAX | 7.886–7.959 | ≤0.02 ms each | | |

Three things fall out:

1. **The GEMV family is now healthy.** qkv 2060 GB/s and the o/down/lm_head group 1944 GB/s
   are *above* what the isolated lab reaches for those shapes — which is why row-blocking
   them regressed (Round 2's negative result): there was nothing left to win.
2. **`GEMV_ARGMAX` (op 80) contributes ~0 — this packet never emits it.** lm_head runs through
   plain `PLOW_DOP_GEMV`. The Round-2 "lm_head" ablation flag (`PLOW_NV_RB_LMHEAD`, which
   guards `d_gemv_argmax`) was therefore testing dead code.
3. **FLASH_DECODE is the worst op in the engine: 212 GB/s, ~17× off its 0.062 ms roofline.**
   It is overhead-bound (Q staging, online softmax, merge partials, work-item granularity),
   not bandwidth-bound — at ctx=1024 there are only 231 MB to read.

The individual contributions sum to 5.73 ms against a 7.906 ms step. The ~2.2 ms difference
is **non-additive**: removing one op lets the others' imbalance dominate. That residual —
inter-op serialization and load imbalance that only appears with the full program — is the
largest single unlocalized term and the next thing to attack.

## Flash nsplit sweep (packet-emit knob) — small

`nsplit` is baked at emit time. Note the windowed-layer nsplit cap in `devgen` is gated on
`fp8_kv`, so a **bf16-KV** packet's 25 sliding layers keep the CU-fill split. Re-emitted
packets (v7, 521 decode packets / 29273 wg-packets — matches the trace exactly):

| packet | TPOT |
|---|---|
| default | 7.816 |
| `PLOW_NS_ABS=4` | 7.865 |
| **`PLOW_NS_ABS=8`** | **7.741** |
| `PLOW_NS_ABS=16` | 7.785 |
| `PLOW_NS_ABS=32` | 7.936 |
| `PLOW_NS_FULL_ABS=33` | 7.839 |

Best is −0.075 ms. **flash_decode's 1.088 ms is fixed overhead, not split count** — tuning
will not fix it; the op needs rework. (The freshly emitted v7 packet is itself 0.09 ms faster
than the older shipped one, which is why base reads 7.816 vs 7.906.)

## Ranked plan from here (26B bf16, vLLM = 4.833 ms, plow = 7.9 ms)

| # | target | size | note |
|---|--:|---|---|
| 1 | the ~2.2 ms non-additive imbalance residual | ~2.2 ms | largest term; needs per-block (not block-0) timing to localize |
| 2 | FLASH_DECODE rework | ~1.0 ms | 212 GB/s, 17× off roofline, overhead-bound; tuning already exhausted |
| 3 | MOE_DOWN → lab rate (1054 → 1655 GB/s) | ~0.35 ms | RB=2 captured only part; RB=4 costs REG 177→229 |
| 4 | MOE_GLU_NORM → lab rate (1683 → 2037 GB/s) | ~0.20 ms | |
| 5 | op fusion to cut gate count | ≤0.5 ms | bounded by the skeleton floor |

Items 3–5 total ~1 ms and cannot close a 3.1 ms gap on their own. **Items 1 and 2 are where
the campaign has to go**, and both are structural work on the megakernel rather than tuning.

---

# Round 4 — the MoE router was 13 % of the decode step

Cumulative ablation (each step adds one opcode to the mask, so the deltas ARE additive):

| removed | TPOT | Δ |
|---|---|---|
| — (full step) | 7.906 | |
| MoE_glu (71) | 6.779 | 1.127 |
| + MoE_down (63) | 5.811 | 0.968 |
| + GEMV o/down/lm (10) | 4.497 | 1.314 |
| + QKV (22) | 3.624 | 0.873 |
| + FLASH_DECODE (12) | 2.840 | 0.784 |
| **skeleton (no bodies)** | **0.576** | |

The five heavy ops account for 5.07 ms, but with all of them ablated the step is still
**2.84 ms** against a 0.576 ms gate floor. **2.26 ms lived in the "small" ops** — each of
which measured ≤ 0.08 ms when ablated *alone*, because individually they hide behind the
heavy ops' straggler tails. Ablating from the BASE5 state exposes them:

| op removed from BASE5 (2.814) | Δ |
|---|---|
| **MOE_ROUTER_GEMMA_TOPK (68)** | **0.835** |
| **MOE_ROUTER_GEMMA_SCORE_FAST (69)** | **0.537** |
| GEMV_GLU (19) | 0.509 |
| MOE_COMBINE_NORM (70) | 0.286 |
| FLASH_MERGE (13) | 0.131 |
| HEADNORM_ROPE (3) | 0.062 |
| RMSNORM / ADD_NORM / EMBED / COMBINE_RESID_NORM(72) | ≤0.01 |

Measured in the FULL step, the two router ops cost **1.026 ms — 13 % of the token** to pick
8 experts out of 128. (Round 3's per-op sweep missed this: it ablated opcode 61,
`MOE_ROUTER_GEMMA`, which this packet never emits.)

## Why the router was that expensive

**`d_moe_router_gemma_topk_row` ran the whole tail on THREAD 0**: a 128-element max, 128
`__expf`, 128 divides, then `k × n_exp` = **1024 serial masked-argmax iterations**, then the
gate normalisation — ~1400 dependent iterations per layer, 30× per token, while 132 blocks ×
256 threads idled.

**`d_moe_router_gemma_score_fast`** had the same defect as the expert GLU arm: **scalar
2-byte loads** of `resid`/`scale`/`proj`, and `h2 = resid*invrms*scale*root` recomputed from
global for **every one of the 128 expert rows**. It produced 721 KB of logits at ~38 GB/s.
It also only fills 16 of 132 blocks (`slice*8 + warp` over 128 experts) — left as is, since
vectorising alone made it free.

## The fixes

- **Warp-parallel top-k.** `max` and `argmax` are order-independent (the packed key encodes
  the lowest-id tie-break, so the winner is unique regardless of reduction order) and
  `exp`/divide are per-element, so those parallelise **bit-exactly**. The softmax
  **denominator is still summed sequentially by lane 0 in the original `e=0..n_exp` order**,
  because a tree sum would round differently and could in principle move a gate.
- **Vectorised router score**: stage `h2` once per CTA as f32 (same value into each FMA) and
  stream `proj` with the 16 B `ld_glob8`.
- **`MOE_COMBINE_NORM` pass 1 as float4**: this op runs on ONE block (row loop strided by
  `slice`, decode has `nrow==1`), moving 90 KB of f32 partials with 256 threads at ~8 GB/s.

Router cost after the fix, re-ablated on the new build: **SCORE 0.560 → 0.008 ms, TOPK
0.760 → ~0**.

**Note on smem:** `PLOW_MOE_XN_MAX` is sized to exactly 2816 f32 (11 KiB), not 4096. Two arms
now stage (expert GLU + router score) and their **static** smem adds to the kernel total,
while the runtime only raises the **dynamic** limit — oversizing it drove
`cuOccupancyMaxActiveBlocksPerMultiprocessor` to **0** and the engine refused to launch.

## Cumulative results (median of 3, same toolchain both arms)

| config | control (`PLOW_NV_GEMV_RB=0`) | rounds 2+4 | speedup |
|---|---|---|---|
| bf16 ctx=1024 | 9.430 ms | **7.061 ms** | **1.34×** |
| bf16 ctx=4096 | 9.466 ms | **7.097 ms** | 1.33× |
| fp8 ctx=1024 | 7.471 ms | **6.650 ms** | **1.12×** |
| fp8 ctx=4096 | 7.500 ms | **6.679 ms** | 1.12× |

fp8 now gains too: the router is precision-independent, so unlike the round-2 expert-GLU fix
it pays on both packets.

`gpu_lifecycle` PASSES on both precisions. sm_120 decode and prefill cubins re-verified
**byte-identical**. Megakernel REG **177, zero spills** — occupancy unchanged throughout.

## Standing vs vLLM

| | round-1 plow | now | vLLM 0.25.1 | gap |
|---|---|---|---|---|
| bf16 ctx1024 | 9.430 ms | **7.061 ms** | 4.833 ms | 1.46× (was 1.95×) |
| fp8 ctx1024 | 7.471 ms | **6.650 ms** | 4.417 ms | 1.51× (was 1.69×) |

**Still not beaten.** What remains, measured:

| target | size | nature |
|---|--:|---|
| GEMV o/down/lm_head | 1.31 ms | already 1944 GB/s — at the lab's own rate, little left |
| MoE expert GLU | 1.13 ms | 1683 vs lab 2037 GB/s → ~0.20 ms |
| MoE expert down | 0.97 ms | 1054 vs lab 1655 GB/s → ~0.35 ms |
| QKV | 0.87 ms | already 2060 GB/s — nothing left |
| FLASH_DECODE | 0.78 ms | 212 GB/s, 17× off roofline, overhead-bound; nsplit sweep exhausted |
| gate/signal skeleton | 0.58 ms | protocol floor |

Tuning the two MoE arms to their lab rates is worth ~0.55 ms (→ ~6.5 ms). **Beating 4.833 ms
requires the FLASH_DECODE rework and cutting the skeleton/serialisation floor — structural
work on the megakernel, not tuning.** The bandwidth headroom is real (E1: 2490 GB/s available
at 1 block/SM, vLLM only achieves 1579), so the ceiling is not hardware.

## Round 4 addendum — the megakernel's register budget is the real ceiling

Re-sweeping the MoE row-block depths on the finished build (bf16 / fp8 ms @ctx1024):

| GV_MOE_RB | RB_DN | megakernel REG | bf16 | fp8 |
|---|---|---|---|---|
| **2** | **2** | **178** | **7.055** | **6.649** |
| 2 | 4 | 229 | 7.077 | 6.690 |
| 4 | 2 | **255** | 7.773 | 6.716 |
| 4 | 4 | **255** | 7.922 | 6.728 |

Pushing either arm to 4 streams hits the **255-register ceiling** and regresses by up to
0.87 ms. So the ~0.55 ms of "lab rate" headroom on the two MoE arms is **not reachable by
adding streams** — it is capped by the register file.

**This is also the explanation for round 2's negative result.** In the lab each GEMV variant
is its own kernel and gets the whole register file (`D_R4_un4` = 118 regs). In the megakernel
there is ONE register allocation shared by every arm — flash decode, the GEMM arms, the MoE
arms, the norms — so an arm that wants more ILP raises the max for all of them and the
allocator starts constraining everything. Row-blocking the dense GEMV arms "won in isolation
and lost in context" for exactly this reason.

That reframes the segmented/lean-object idea from `segmented-decode-26b-h100.md`: it was
pursued to buy **occupancy**, which E1 shows is not the constraint. Its real value would be
giving each op class **its own register budget**, which is what actually caps per-arm ILP
here. That is the structural change worth costing next — together with the FLASH_DECODE
rework, it is the only measured path to the remaining 1.46×.

---

# Round 5 — tuning is exhausted; the ceiling is the megakernel's shared register file

## Staging moved to the dynamic arena (neutral, frees 22.5 KiB of static smem)

The expert-GLU and router-score arms now stage into the **dynamic arena** instead of static
`__shared__`. Measured neutral (7.060 vs 7.057 ms) but static smem drops **24976 → 2448 B**.
This matters because the runtime only raises the *dynamic* smem limit: static staging buffers
push the per-block total past the carveout, and at 35216 B
`cuOccupancyMaxActiveBlocksPerMultiprocessor` returned **0** and the engine refused to launch.

**Negative result — staging `fu` for the MoE-down arm: 7.183 vs 7.060 ms (+0.123).** The extra
`__syncthreads` on every MoE-down op (30 per token) costs more than the redundant `fu` reads
it removes; `fu` is 11 KiB and was already an L1 hit. Kept behind `PLOW_MOE_DOWN_STAGE_FU=0`.

## The register file is the hard ceiling

| GV_MOE_RB | RB_DN | REG | bf16 |
|---|---|---|---|
| **2** | **2** | **178** | **7.059** |
| 2 | 3 | 188 | 7.164 |
| 2 | 4 | 229 | 7.077 |
| 3 | 2 | **255** | 7.297 |
| 4 | 2 | **255** | 7.773 |
| 4 | 4 | **255** | 7.922 |

Any increase in per-arm ILP past RB=2 hits the **255-register ceiling** and costs up to
0.87 ms. MoE-down's remaining ~0.4 ms of lab-rate headroom (979 vs 1655 GB/s) is therefore
**unreachable inside the megakernel** — it is not an arm defect.

## FLASH_DECODE: measured directly, and the obvious fix is refuted

Flash's own cost, isolated by ablating opcode 12 on each packet (not inferred from TPOT):

| packet | full | flash ablated | **flash cost** | per (rows/item) | active threads of 256 | blocks |
|---|---|---|---|---|---|---|
| `PLOW_NS_ABS=4` | 7.023 | 5.947 | **1.076** | 256 | **256 (100 %)** | 32 |
| `PLOW_NS_ABS=8` | **6.894** | 5.962 | **0.932** | 128 | 128 (50 %) | 64 |
| `PLOW_NS_ABS=16` | 6.916 | 5.978 | 0.938 | 64 | 64 (25 %) | 128 |
| default (ns=17) | 6.954 | 5.972 | 0.982 | 61 | 61 (24 %) | 136 (ragged on 132) |

`FA_DEC_TILE = PLOW_NV_THREADS = 256` (one KV row per thread), and the emitter's CU-fill gives
`ns = ceil(132*2/16) = 17`, so a sliding layer at ctx=1024 runs **61 of 256 threads** and lands
136 items on 132 blocks. The natural hypothesis — restore full thread utilisation — is
**REFUTED**: `ns=4` gives 100 % thread occupancy and is the **worst** point (1.076 ms). More
blocks beats more threads-per-block here, and the whole knob is worth only 0.05 ms. Flash
floors at ~0.93 ms against a 0.062 ms roofline (15×) and needs an algorithmic rework, not
tuning. `PLOW_NS_ABS=8` is the best emit-time setting measured (recorded, not baked into the
emitter — it is one model at one ctx).

## Remaining tail, re-probed after the router fixes (BASE5 = 1.917, was 2.814)

| op removed from BASE5 | Δ |
|---|---|
| GEMV_GLU (19) | 0.546 — but ~0 in the FULL step (fully hidden); already 1247 GB/s vs the lab's 1302 |
| MOE_COMBINE_NORM (70) | 0.218 |
| FLASH_MERGE (13) | 0.121 |
| HEADNORM_ROPE (3) | 0.058 |
| ROUTER_SCORE / TOPK | 0.051 / 0.015 (were 0.537 / 0.835) |
| RMSNORM / RESIDUAL / SOFTCAP / ARGMAX / EMBED | ≤0.015 each |

**No router-class defect remains.** Everything left is either at its lab rate, hidden behind
other ops, or under 0.15 ms.

## Final standing

Same-packet A/B (v7 blob, `PLOW_NS_ABS=8`, control = `PLOW_NV_GEMV_RB=0`):

| | control | this branch | speedup |
|---|---|---|---|
| bf16 ctx1024 (best packet) | 9.267 ms | **6.903 ms** | **1.34×** |
| bf16 ctx1024 (shipped packet) | 9.421 ms | 7.063 ms | 1.33× |
| bf16 ctx4096 | 9.461 ms | 7.091 ms | 1.33× |
| fp8 ctx1024 | 7.465 ms | 6.647 ms | 1.12× |
| fp8 ctx4096 | 7.500 ms | 6.677 ms | 1.12× |

vs **vLLM 0.25.1: bf16 4.833 ms, fp8 4.417 ms → still 1.43× / 1.51× behind.**

Budget of the remaining 6.90 ms, all measured: GEMV o/down/lm 1.44 (1944 GB/s, at lab rate) ·
flash 0.93 (15× off roofline, rework) · MoE GLU 0.99 (1913 vs 2037 GB/s) · MoE down 0.97
(register-capped) · QKV 0.63 (2060 GB/s, at lab rate) · gate/signal floor 0.58 · rest ~1.36.

**Tuning is exhausted.** Closing the remaining 1.43× requires two structural changes, both
now supported by measurement rather than inference:

1. **Per-op-class register budgets** (separate decode objects, dispatched per segment). The
   megakernel has ONE register allocation shared by every arm; that is what caps the MoE arms
   at RB=2 and what made round 2's dense row-blocking lose in context while winning in
   isolation. Note this is the segmented-object idea from `segmented-decode-26b-h100.md`, but
   its original justification (occupancy) is refuted by E1 — the real case is registers.
2. **FLASH_DECODE rework** — 0.93 ms at 15× off roofline, with the nsplit knob measured out.

The bandwidth headroom is real and unchanged: E1 shows 2490 GB/s available at 1 block/SM and
vLLM only achieves 1579, so the ceiling is the engine's structure, not the hardware.

## CORRECTION to the two sections above — "register ceiling" was the wrong explanation

The rounds-4/5 claim that RB=4 loses *because* it hits 255 registers does not survive its own
control. Two facts:

1. **Removing the flash arms does not free the budget.** Built with `PLOW_NV_LEAN_DECODE=1`
   (flash compiled out) plus the flash opcodes ablated:

   | GV_MOE_RB | RB_DN | REG (lean object) |
   |---|---|---|
   | 2 | 2 | **163** |
   | 2 | 4 | 225 |
   | 4 | 2 | **255** |
   | 4 | 4 | **255** |

   With flash gone entirely the MoE arm at RB=4 *still* drives the kernel to 255. So the arms
   are not being squeezed by flash, and **giving each op class its own register budget would
   not unlock them** — which retracts structural recommendation #1 as stated.

2. **REG=255 is not itself a penalty here.** The kernel is `__launch_bounds__(256, 1)` and the
   grid is n_cu=132 = 1 block/SM, so ptxas may use up to 255 registers with **no occupancy
   cost**. A high REG number is ptxas spending a budget it is allowed to spend, not evidence of
   pressure — and there are 0 spills at every point in the sweep.

So the RB=4 regression is **unexplained**, not register-bound. It correlates with REG=255 but
the mechanism is something else (scheduling, or the same isolation-vs-in-context effect that
sank the dense arms in round 2 — note both are row-blocking changes that win standalone and
lose in the megakernel). **The honest status: RB=2 is the measured optimum and we do not know
why RB=4 loses.** That is the open question, and it is the same open question round 2 left.

Consequently the ranked plan reduces to:

1. **Understand the isolation-vs-in-context gap** — one mechanism appears to govern both the
   dense-GEMV and the RB=4 regressions. Needs per-block/per-op device timing inside the
   megakernel (block-0 trace attribution is not enough), which is the missing instrument.
2. **FLASH_DECODE rework** — 0.93 ms at 15× off roofline, nsplit knob measured out and the
   thread-utilisation hypothesis refuted.

The lean-object timings in that table are register counts only: those runs produced no TPOT,
because the lean object's lower REG raises occupancy to 2 blocks/SM and the engine correctly
refuses the resulting grid=264 against a packet emitted for n_cu=132.

---

# Round 6 — 2 blocks/SM is worth 1.45×, but it leaks VRAM and is NOT shippable

E1 said the achievable read rate rises from 2490 GB/s at 1 block/SM to 3269 at 2. Round 5's
freeing of the static staging smem (24976 → 2448 B) made 2 blocks/SM reachable, so this round
tested it: emit the packet with `--n-cu 264` and build the object with
`-DPLOW_NV_FORCE_MINBLK=2` (REG 128).

## Deep unrolling is WRONG at occ-2

At the 128-register cap the shipped unroll depths spill and cost more than they buy:

| build | STACK | bf16 ctx1024 |
|---|---|---|
| occ-2, shipped unrolls | 304 | 6.806 |
| occ-2, `GV_UNROLL=4` | 232 | 6.642 |
| occ-2, `GV_MOE_UN=2` | 304 | 6.631 |
| occ-2, `GV_UNROLL=4 GV_MOE_UN=2` | 192 | 6.420 |
| **occ-2, `GV_UNROLL=4 GV_MOE_UN=2 GV_UNROLL_GLU=2`** | **152** | **6.382** |
| occ-3 (`--n-cu 396`, MINBLK=3, REG 80) | 752 | 7.196 |

**Less unroll per thread + 2× the blocks beats more unroll at 1 block/SM.** occ-3 is worse:
REG 80 spills 752 B and erodes the gain, exactly as the old campaign predicted.

## Per-op effect of occ-2 (full 6.383, vs the occ-1 numbers)

| op | occ-1 | occ-2 | |
|---|---|---|---|
| GEMV o/down/lm | 1.436 | **1.086** | 1.32× |
| MoE expert GLU | 0.993 | **0.893** | |
| MoE expert down | 0.971 | **0.818** | |
| QKV | 0.626 | **0.486** | |
| FLASH_DECODE | 0.932 | 0.976 | worse |
| gate/signal skeleton | 0.576 | 0.577 | unchanged — 2× the blocks cost no extra gate time |

Every streaming GEMV improves; **flash does not**. Before the unroll retune, MoE-GLU and flash
both *regressed* at occ-2 (1.184 / 1.106) because their per-block setup — the staged `xn`, the
staged Q — is replicated per block and there are now twice as many blocks.

## FLASH_DECODE: K-stream pre-issue REFUTED

The V/PV loop already stages `vv[VU]` before consuming; the K loop consumed each `k8`
immediately, so the obvious hypothesis was that ptxas kept ~1 K load in flight (which the
timing supports: 32 serial 16 B loads × ~600 ns ≈ the measured 32.5 µs/layer). Added an
explicit pre-issue depth `PLOW_NV_FA_KUN` (same `d` and `g` order ⇒ bit-identical):

| `PLOW_NV_FA_KUN` | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| bf16 ctx1024 | 6.385 | 6.382 | 6.416 | 6.455 |

**No effect.** Flash's 237 GB/s is not K-load pipelining. The remaining suspect is the access
*shape*: the score phase is one KV row per thread, so a warp instruction scatters 32 requests
512 B apart instead of one coalesced 512 B burst. Fixing that means a warp-per-row score phase
(and a warp reduction per row, so not bit-identical) — the next real experiment, untried.
`PLOW_NV_FA_KUN` defaults to 1 = the original loop, byte-identical.

## BLOCKER — occ-2 leaks 140 MiB of VRAM

`gpu_lifecycle` **FAILS** on the occ-2 configuration:

```
cycle 0: loaded 52544 MiB, reply: "Paris"          <- output is CORRECT
cycle 0: after unload 666 MiB (baseline 526 MiB)   <- 140 MiB not returned, tolerance 64
```

Not the asset layout: the **same v7 packet at occ-1 returns to 526 MiB exactly** and passes
both cycles. The leak tracks the `n_cu=264` configuration. So occ-2's 1.45× is **measured but
not shippable** until that is found — most likely something sized by `n_cu` (flash partials /
per-block scratch) that the engine does not release, or driver-side cooperative-launch scratch.

Both occ-2 ingredients are opt-in (a packet emitted with `--n-cu 264` **and** a cubin built
with `-DPLOW_NV_FORCE_MINBLK=2`), so nothing changes for any default build.

## Standing

| | control | shipped default (occ-1) | occ-2 (leaks, not shippable) | vLLM |
|---|---|---|---|---|
| bf16 ctx1024 | 9.267 | **7.067** (1.31×) | **6.382** (1.45×) | 4.833 |
| fp8 ctx1024 | 7.465 | **6.645** (1.12×) | not emitted | 4.417 |

vLLM remains **1.32× ahead of even the occ-2 build**. Ranked from here:

1. **Fix the occ-2 VRAM leak** — unlocks a measured 1.45× (0.69 ms) immediately.
2. **Warp-per-row flash score phase** — 0.98 ms at 237 GB/s, with pre-issue and nsplit both
   now refuted, this is the one untried hypothesis.
3. Re-tune the MoE/GEMV unrolls *at* occ-2 (this round only did a coarse sweep).

## The vLLM target, re-measured on this box (independent of the prior campaign)

The whole campaign chases a number inherited from `gemma26b-h100-beat-vllm-campaign.md`, and
that branch's own history contains a *"RETRACT two claims — vLLM pass 1 was contended"*. So it
was re-measured here, under `gpulease`, with `perf-data/vllm_tpot_h100.py`: vLLM **0.25.1**,
torch 2.11.0+cu130, `/workspace/models/gemma-4-26B-A4B-it`, bf16, TP-1, B=1, CUDA graphs ON
(`enforce_eager=False`), greedy, a 1024-**token** prompt. TPOT is taken by differencing
`max_tokens=1` and `max_tokens=129` (median of 3 each), which cancels TTFT and harness cost:

```
ctx=1024  t1=20.96 ms  t129=638.56 ms  ->  TPOT = 4.825 ms/token
```

**4.825 ms confirms the inherited 4.833 to 0.2 %.** The target is real; every ratio in this
document stands. (`vllm-blk` venv; the `vllm-cu128` venv cannot load the model —
`_vllm_fa2_C`/`_vllm_fa3_C` missing — and the engine subprocess needs the venv's `ninja` on
PATH or `EngineCore` dies in torch.compile.)

Repro:
```
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat PATH=/workspace/venvs/vllm-blk/bin:$PATH \
  gpulease vllm /workspace/venvs/vllm-blk/bin/python perf-data/vllm_tpot_h100.py 1024
```

# Round 7 — LANE-SPLIT: more rows in flight per warp at ZERO register cost

Row-blocking's goal was more output rows in flight per warp; its problem was that `wv[RB][UN]`
costs registers and RB>2 blows the megakernel budget (rounds 4-6). **Lane-split gets the same
rows-in-flight for free.** Split the warp into `SG` sub-groups of `32/SG` lanes, one output row
per sub-group: SG rows are in flight with **one accumulator per lane and no `wv[][]` array at
all**, and the reduction shrinks from 5 shuffle steps to `log2(32/SG)`.

Applied to the MoE DOWN arm — the worst GEMV in the step (1163 GB/s at occ-2 vs 3269
achievable) precisely *because* its K is short (I_moe=704 ⇒ only 3 chunks of 256 per lane, so
the per-row expert lookup and the 32-lane reduction amortise over nothing):

| sub-groups (lanes per row) | occ-2 bf16 |
|---|---|
| off | 6.387 |
| SG=2 (16 lanes) | 6.443 |
| **SG=4 (8 lanes)** | **6.109** |
| SG=8 (4 lanes) | 6.108 |

**It is a short-K trick, not a general one.** Applied to the *dense* GEMV arm (o_proj/down/
lm_head, K = 2112…4096) it **loses**: 6.228 vs 6.106. Long K already amortises the reduction,
and lane-split trades coalescing width (32 lanes × 16 B = 512 B) for rows-in-flight
(8 × 16 B = 128 B). Kept behind `PLOW_NV_GEMV_LS` (default 0) as a recorded negative.

Reduction width and the per-lane K partition change, so DOWN's outputs are numerically
equivalent, not bit-identical. `gpu_lifecycle` passes on both precisions.

## Shipped result (occ-1, no VRAM caveat) — built by `scripts/build_sm90a_cubin.sh`

| | control | shipped | speedup |
|---|---|---|---|
| bf16 ctx=1024 | 9.267 ms | **6.772 ms** | **1.37×** |
| bf16 ctx=4096 | 9.461 ms | **6.799 ms** | **1.39×** |
| fp8 ctx=1024 | 7.465 ms | **6.660 ms** | 1.12× |
| fp8 ctx=4096 | 7.500 ms | **6.692 ms** | 1.12× |

fp8 is unchanged by lane-split (it runs the *fp8* down arm, which this round did not touch).
REG 180, 0 spills, static smem 2448 B. sm_120 cubins byte-identical.

## Final standing

| | plow (shipped) | plow (occ-2, leaks) | vLLM 0.25.1 (re-measured here) |
|---|---|---|---|
| bf16 ctx1024 | 6.772 ms | 6.106 ms | **4.825 ms** |
| fp8 ctx1024 | 6.660 ms | — | 4.417 ms |

**vLLM is still ahead: 1.40× on the shipped build, 1.27× on the (unshippable) occ-2 build.**
Total campaign movement: bf16 9.267 → 6.772 shipped (1.37×), → 6.106 measured (1.52×).

Ranked, with everything else in this document now measured out:

1. **The occ-2 VRAM residual** (140 MiB, constant across stack sizes 152/240/304, tracks
   `n_cu=264`). Unlocks a measured 0.67 ms.
2. **FLASH_DECODE** — 0.98 ms at 237 GB/s. nsplit swept, K pre-issue refuted, thread-utilisation
   refuted. The one untried hypothesis is a warp-per-row score phase (the current one row per
   thread makes each warp instruction scatter 32 requests 512 B apart).
3. **Lane-split the fp8 down arm** — the bf16 arm gained 0.29 ms from it; the fp8 twin is
   untouched and fp8 is the weaker of the two configs relative to vLLM.

## Round 7b — lane-split the fp8 DOWN arm too

The fp8 packet runs its own `d_moe_expert_down_gemma_fp8`, which round 7 did not touch and
which has the identical short-K shape (I_moe=704, one warp per output channel). Giving it the
same lane-split:

| | before | after |
|---|---|---|
| fp8 ctx1024 | 6.660 ms | **6.135 ms** |
| bf16 ctx1024 | 6.772 ms | 6.760 ms (unchanged, as expected) |

## Campaign totals (shipped, occ-1, `scripts/build_sm90a_cubin.sh`)

| | control | shipped | speedup | vLLM (re-measured here) | gap |
|---|---|---|---|---|---|
| bf16 ctx=1024 | 9.267 ms | **6.760 ms** | **1.37×** | 4.825 ms | 1.40× |
| bf16 ctx=4096 | 9.461 ms | **6.783 ms** | **1.39×** | — | |
| fp8 ctx=1024 | 7.465 ms | **6.135 ms** | **1.22×** | 4.417 ms | 1.39× |
| fp8 ctx=4096 | 7.500 ms | **6.169 ms** | **1.22×** | — | |

`gpu_lifecycle` passes on both precisions; sm_120 decode and prefill cubins byte-identical.

**vLLM still wins by ~1.40× on both.** The measured levers that remain are unchanged from
round 7: the occ-2 VRAM residual (worth 0.67 ms, currently unshippable), and FLASH_DECODE
(0.98 ms at 237 GB/s, with nsplit / K-pre-issue / thread-utilisation all refuted and only the
warp-per-row score phase untried).

## Round 7c — where the shipped 6.76 ms goes, and the ceiling of this approach

Re-ablated on the shipped build (full 6.761):

| op | cost | note |
|---|---|---|
| GEMV o_proj + down + lm_head | 1.455 | o_proj/down have only 22 rows per block (N=2816 / 132) |
| FLASH_DECODE | 1.079 | 237 GB/s |
| MoE expert GLU | 1.015 | |
| QKV | 0.693 | |
| MoE expert down | **0.675** | was 0.971 before lane-split |
| gate/signal skeleton | 0.576 | |
| MOE_COMBINE_NORM | 0.268 | |

Lane-split on the *dense* GEMV arm at occ-1 is a wash and stays off: bf16 6.791 vs 6.762
(worse), fp8 6.080 vs 6.129 (better). Not worth a numerics change for a split verdict.

Best achievable with everything this campaign built, at 2 blocks/SM: **6.127 ms** — still
1.27× behind vLLM, and blocked by the occ-2 VRAM residual.

## FINAL

| | control | **shipped** | occ-2 (blocked) | vLLM (measured here) | gap |
|---|---|---|---|---|---|
| bf16 ctx1024 | 9.267 | **6.760** (1.37×) | 6.127 (1.51×) | **4.825** | 1.40× |
| fp8 ctx1024 | 7.465 | **6.137** (1.22×) | — | **4.417** | 1.39× |

**vLLM is not beaten.** The campaign moved plow 1.37×/1.22× and refuted the prior "occupancy
wall" diagnosis, but ~1.4× remains. Every remaining lever is measured:

1. **occ-2 VRAM residual** — 140 MiB, constant across stack sizes, tracks `n_cu=264`. Worth
   0.63 ms and already implemented; only the leak blocks it.
2. **FLASH_DECODE** — 1.08 ms at 237 GB/s (16 % of the step). nsplit swept, K pre-issue
   refuted, thread-utilisation refuted. Only a warp-per-row score phase is untried.
3. **o_proj / down_proj** — 22 rows per block is too few to amortise the per-row reduction;
   neither row-blocking nor lane-split helped, and this is the same unexplained
   isolation-vs-in-context gap the campaign has hit three times now.

Items 2 and 3 are kernel redesigns, not tuning. Nothing cheap is left.

# Round 8 — the occ-2 "leak" is bounded, and the residual is small-op SERIALISATION

## The occ-2 VRAM residual is NOT a leak (occ-2 is shippable)

Round 6 called it a blocker. Re-run with the tolerance widened so both cycles complete:

```
cycle 0: loaded 52544 MiB, reply "Paris"; after unload 666 MiB (baseline 526)
cycle 1: loaded 52684 MiB, reply "Paris"; after unload 666 MiB (baseline 526)
```

**Identical after both cycles — it does not grow.** It is a one-time, bounded ~140 MiB
driver-side allocation for the larger cooperative grid (`n_cu=264`), not a per-load leak;
reload works and output is correct. `gpu_lifecycle`'s 64 MiB tolerance is calibrated for the
occ-1 grid, so it trips on a fixed cost rather than a defect. **occ-2's 6.12 ms is usable**
(it still needs a packet emitted at `--n-cu 264`, so it is opt-in either way).

## The occ-2 tail: two ops come BACK at 2 blocks/SM

| op removed (occ-2 full 6.122) | Δ | at occ-1 |
|---|---|---|
| MOE_COMBINE_NORM (70) | **0.317** | 0.268 |
| MoE router (68+69) | **0.255** | ~0 |
| HEADNORM_ROPE (3) | 0.157 | 0.058 |
| GEMV_GLU (19) | 0.137 | ~0 |
| FLASH_MERGE (13) | 0.048 | |
| RMSNORM / RESIDUAL / EMBED / SOFTCAP / ARGMAX | ≤0.015 each | |

Ops whose cost is *latency* rather than bytes get worse as blocks are added, because every
block waits for them.

**Negative result — all-block COMBINE.** `d_moe_combine_norm_gemma` runs on ONE block (row loop
strided by `slice`, decode has `nrow==1`), so 263 blocks gate on one CTA moving 90 KiB. Its RMS
denominator needs all H, which is why it was never split — so this round had *every* block
recompute the (L2-resident) reduction redundantly, with identical thread mapping so each
derives a bit-identical `inv` with no cross-block gate, and each block writing only its own
disjoint output slice. **Measured 6.142 vs 6.123 — slightly worse.** The op is latency-bound
(two `__syncthreads` + a block reduction ≈ 10 µs/layer for 90 KiB), not bandwidth-bound, so
replicating the work does not shorten it. Kept behind `PLOW_MOE_COMBINE_ALLBLK=0`.

## Why vLLM still wins — the structural answer

plow's *per-op bandwidth is no longer the problem*. At occ-2 the weight-moving ops run at
QKV 2846, GEMV 2263, MoE GLU 2128 GB/s — **above vLLM's 1579 GB/s whole-token average.**

The gap is everything that moves almost no bytes: flash ~1.0, gate/signal floor 0.58, combine
0.32, router 0.26, headnorm 0.16, GEMV_GLU 0.14 — **~2.5 ms of the 6.12 ms step spent on ops
that are latency-bound, not bandwidth-bound.** The megakernel gates every one of its 216
packets grid-wide, so each small op's latency is fully exposed: nothing overlaps it. The pure
protocol cost is only 0.577 ms (the skeleton), so this is not barrier overhead — it is the
absence of overlap.

That is the architectural difference from vLLM, which pipelines its many small kernels under
CUDA graphs and folds attention into one efficient kernel. Closing it needs **op fusion and
overlap in the emitter + interpreter** — fewer, larger gated regions — not kernel tuning.
Every tuning lever this campaign found is now spent.

## FINAL

| | control | shipped (occ-1) | occ-2 (opt-in) | vLLM (measured here) |
|---|---|---|---|---|
| bf16 ctx1024 | 9.267 | **6.760** (1.37×) | **6.122** (1.51×) | **4.825** |
| fp8 ctx1024 | 7.465 | **6.137** (1.22×) | — | **4.417** |

**vLLM is not beaten: 1.40× (shipped) / 1.27× (occ-2) on bf16, 1.39× on fp8.**

# Round 9 — warp-per-row flash: the access SHAPE was the answer

`flash_decode` had been the worst op all campaign (237 GB/s, ~14× off roofline) and two
hypotheses had already been refuted: `nsplit` (swept; 0.05 ms) and explicit K pre-issue
(`PLOW_NV_FA_KUN` 2/4/8; no effect). The surviving suspect was the **access shape**.

The default body gives each **thread** a whole KV row, so one warp instruction issues 32
requests `D*2` bytes apart — 32 scattered sectors instead of one coalesced burst — and each
thread then walks its row with `D/8` dependent 16 B loads. `PLOW_NV_FA_WPR=1` gives a **warp**
the row: its 32 lanes cover 32×8 = 256 elements, so a D=256 row is **one fully coalesced 512 B
load** (D=512 is two). The dot then costs a warp reduction per query head, and threads read
their own row's score back out of `Ssm` so the softmax/PV code below is untouched. Only the
plain bf16 KV layout takes the path; SZ/fp8 KV keep the default body.

| | before | after |
|---|---|---|
| flash_decode (occ-1) | 1.079 ms | **0.686 ms** |
| bf16 ctx1024 (occ-1) | 6.757 | **6.336** |
| fp8 ctx1024 (occ-1) | 6.132 | **5.758** |
| bf16 ctx1024 (occ-2) | 6.124 | **5.813** |

## Op fusion is NOT the lever (three unused fusion flags, all measured)

| packet | decode packets | occ-2 bf16 |
|---|---|---|
| base | 521 | **6.123** |
| `PLOW_FUSE_ARGMAX=1` | 519 | 6.149 |
| `PLOW_GEMMA_MOE_TAIL_FUSE=1` (op72) | 491 | 6.303 |
| `PLOW_GEMMA_MOE_ROUTER_FUSED=1` | 491 | **13.171** |

**Fewer gates, slower step** — consistent with the 0.577 ms skeleton floor (~2.7 µs/gate):
removing 30 gates saves ~0.08 ms while the fused bodies cost far more. The router fusion is
pathological. This closes the "improve gate counts" line of attack.

## Where the occ-2 5.81 ms goes, and what is left

| op | cost | bytes | achieved | occ-2 ceiling |
|---|---|---|---|---|
| GEMV o/down/lm_head | 1.148 | 2458 MB | 2141 GB/s | 3269 |
| MoE expert GLU | 0.847 | 1900 MB | 2243 GB/s | |
| FLASH_DECODE | 0.729 | 231 MB | 317 GB/s | |
| gate/signal skeleton | 0.577 | — | — | |
| MoE expert down | 0.509 | 951 MB | 1868 GB/s | |
| QKV | 0.466 | 1383 MB | **2968 GB/s** (91 %) | |
| COMBINE_NORM / router | 0.265 / 0.224 | | | |

Re-swept at this config and all neutral-or-worse: deeper unrolls (spills), dense-GEMV
row-blocking, dense-GEMV lane-split. QKV is at 91 % of ceiling; the other three GEMV families
hold ~0.9 ms of theoretical headroom that none of the available knobs reaches.

## FINAL

| | control | **shipped (occ-1)** | occ-2 (opt-in) | vLLM (measured here) | gap |
|---|---|---|---|---|---|
| bf16 ctx1024 | 9.267 | **6.337** (1.46×) | **5.809** (1.60×) | **4.825** | 1.31× / 1.20× |
| bf16 ctx4096 | 9.461 | **6.422** (1.47×) | — | | |
| fp8 ctx1024 | 7.465 | **5.759** (1.30×) | — | **4.417** | 1.30× |
| fp8 ctx4096 | 7.500 | **5.858** (1.28×) | — | | |

`gpu_lifecycle` passes on both precisions; sm_120 cubins byte-identical; REG 180 / 0 spills at
occ-1. **vLLM still wins by 1.30-1.31× on the shipped build and 1.20× at occ-2.**

# Round 10 — flash tile bound, and the gate is already as good as this protocol gets

**Bound the WPR sweep to the live rows.** The warp-per-row body swept all `FA_DEC_TILE=256`
rows, but at `nsplit=8` a work item owns only 128 — half the sweep was loop + store on masked
rows. Bounding it and filling the dead tail with one cheap strided `NEG_INF` pass: occ-2
5.813 → **5.784**. (Bounding the two *softmax* reductions the same way measured slightly WORSE,
5.806, and was reverted — the extra bound costs more than the dead-tail scan it saves.)

**Gate overhead is not reducible here.** The counter protocol already uses inline PTX
(`ld.acquire.gpu.u32` / `ld.relaxed.gpu.u32` / `red.release.gpu.global.add.u32`) and already
backs off with `__nanosleep`. Sweeping the backoff at 264 blocks:

| `PLOW_NV_GATE_SLEEP` | 0 | 16 | 32 | 64 (shipped) | 128 | 256 |
|---|---|---|---|---|---|---|
| occ-2 bf16 | 5.808 | 5.779 | 5.783 | 5.784 | 5.791 | 5.829 |

Flat across 16-64 ns; the shipped 64 is already at the optimum within noise. Spinning flat out
(0) is worse — 264 blocks hammering one cacheline. The knob is left parameterised at its
existing default. **The 0.577 ms skeleton floor stands, and neither fewer gates (round 9) nor
a cheaper gate (here) moves it.**

## FINAL

| | control | **shipped (occ-1)** | occ-2 (opt-in) | vLLM (measured here) | gap |
|---|---|---|---|---|---|
| bf16 ctx1024 | 9.267 | **6.289** (1.47×) | **5.784** (1.60×) | **4.825** | 1.30× / 1.20× |
| bf16 ctx4096 | 9.461 | **6.374** (1.48×) | — | | |
| fp8 ctx1024 | 7.465 | **5.710** (1.31×) | — | **4.417** | 1.29× |
| fp8 ctx4096 | 7.500 | **5.803** (1.29×) | — | | |

REG 182, 0 spills, static smem 2448 B. `gpu_lifecycle` passes both precisions; sm_120 cubins
byte-identical. **vLLM still wins by ~1.30× shipped / 1.20× at occ-2.**

# Round 11 — long context, SASS, the tuner, and the non-GEMM ops

## 1. Long context: plow loses MORE there, not less

Re-emitted at `--max-ctx 131072`, shipped occ-1 object:

| ctx | plow bf16 | vLLM (prior card) | ratio |
|---|---|---|---|
| 1024 | **6.289** | 4.825 | 1.30× |
| 8192 | **6.935** | ~5.03 | ~1.38× |
| 32768 | **9.209** | ~5.03 | ~1.83× |

plow's decode is NOT flat in ctx. The 25 sliding layers are window-capped, but the **5
full-attention layers read the whole context**, and with `FA_GF_FULL=4` against `gqa=8` each
full KV head is read **twice**. Setting `GF_FULL=8` (read it once) is **worse at every ctx** —
1024: 6.395 vs 6.281, 8192: 7.385 vs 6.935, 32768: 10.846 vs 9.209 — so the shipped GF_FULL=4
remains optimal even after the warp-per-row rewrite, confirming the build script's note.
**Long context is plow's weakest axis, not a winnable one.**

## 2. SASS: the hot loop is already clean

`cuobjdump -sass` on the shipped decode object (~81 k instructions). Whole-kernel mix looks
alarming — FFMA 13346 but PRMT 9571 + SHF 6808, i.e. more bf16→f32 widening ops than FMAs.
**But they are not in the hot path.** The densest FFMA window (the GEMV dot) is:

```
FFMA x12  SHF.L.U32  FFMA x4  PRMT FFMA PRMT FFMA PRMT FFMA  SHF FFMA SHF FFMA SHF FFMA  FFMA x9 ...
window mix: 50 FFMA : 4 SHF : 3 PRMT : 3 IMAD
```

~85 % FFMA — ptxas hoists the widening out of the inner loop and the `__bfloat162float` pairs
compile to cheap `PRMT`/`SHF` bit ops, never real conversions. The global PRMT/SHF count comes
from op setup and the many non-GEMV arms. **There is no instruction-level micro-optimisation
left in the dot loop; the GEMV arms are memory-bound, not issue-bound.**

Corroborating: staging the MoE `xn` as **bf16** (which would halve smem traffic but re-introduce
per-use widening) measured **worse** — 5.819 vs 5.784 — so the f32 stage, which pre-converts
once, is the right choice. Kept behind `PLOW_MOE_XN_BF16=0`.

## 3. The tuner / egglog / lean oracle do not cover this

`plowc tune` is scoped to **kernel selection for prefill** (`--profile prefill_dense`, the
`weight_tiling` BN/BK search). Every knob this campaign moved — `GV_UNROLL`, `GV_MOE_UN`,
`GV_MOE_RB`, lane-split SG, `FA_GF_FULL`, `FA_WPR`, `FORCE_MINBLK`, `NS_ABS` — is a
**compile-time define on the decode object**, outside the tuner's search space, so it could not
have found any of them as built. `lean_verify` and `rewrite`(egglog) are correctness/rewrite
tooling, not perf search.

**This is a real gap worth closing:** the campaign established that the right arm depends on
shape (lane-split wins at short K and loses at long K; row-blocking wins standalone and loses
in-context; unroll depth flips with occupancy). That is exactly a per-shape selection problem a
tuner should own, instead of one global `#define` per knob.

## 4. Non-GEMM ops are already small

At occ-2, ablated: `HEADNORM_ROPE` 0.157 · `FLASH_MERGE` 0.048 · `RMSNORM` ~0.01 ·
`RESIDUAL` / `EMBED` / `SOFTCAP` / `ARGMAX` ≤0.015 each. Together ~0.25 ms of 5.78 ms (4 %).
Rope and the norms are **not** where the remaining time is — `COMBINE_NORM` (0.265) and the
router (0.224) are the only sub-0.3 ms ops worth further work, and both are latency-bound
single-block/serial shapes rather than bandwidth problems.

## FINAL

| | control | **shipped** | occ-2 | vLLM | gap |
|---|---|---|---|---|---|
| bf16 ctx1024 | 9.267 | **6.289** (1.47×) | **5.784** (1.60×) | 4.825 | 1.30× / 1.20× |
| fp8 ctx1024 | 7.465 | **5.710** (1.31×) | — | 4.417 | 1.29× |

# Round 12 — the defaults had rotted; and a tuner design for the decode knobs

## `GV_MOE_UN` was stale

Re-sweeping the MoE arm shape now that lane-split and warp-per-row flash are in place, at
1 block/SM:

| GV_MOE_RB | GV_MOE_UN | bf16 | fp8 |
|---|---|---|---|
| 2 | 4 (old default) | 6.288 | 5.709 |
| 1 | 4 | 6.215 | 5.708 |
| 1 | 8 | 6.258 | 5.747 |
| **2** | **2** | **6.194** | 5.709 |

`GV_UNROLL` re-checked at that point: 8 (the default) 6.200 · 4 → 6.327 · 2 → 7.055. So the
dense unroll was right and the **MoE unroll was not** — it had been carried from an earlier
round and stopped being optimal once the other arms moved. Shipped `GV_MOE_UN` 4 → 2:
**bf16 6.196, fp8 5.720**.

This is the argument for the tuner in one data point: a hand-set `#define` silently decayed
while the surrounding kernels improved.

## Tuner design — `perf-data/tuner-decode-sweep-design.md`

`plowc tune` covers **prefill** kernel selection only; every decode knob this campaign moved is
a compile-time define outside its search space. The design records:

- **Scoring must be end-to-end, not microbench.** The campaign's central methodological finding
  is that isolation and in-context disagree (`gemv_lab` says row-blocking wins 1.4× on every
  shape; the megakernel says it loses). A microbench-only tuner would ship the wrong arm — so
  the lab is a *pruner*, `step_bench` TPOT is the *scorer*.
- **ctx is a first-class axis.** Decode is not flat in ctx (6.196 / 6.935 / 9.209 at 1k / 8k /
  32k) and the two knobs governing the growth (`FA_GF_FULL`, `NS_FULL_ABS`) have ctx-dependent
  optima. Buckets must be geometric {1k, 8k, 32k, 128k}.
- **Knobs interact, so they must be swept jointly**: unroll depth *inverts* with occupancy
  (deep wins at 1 blk/SM, spills and loses at 2), and occupancy itself is a `(FORCE_MINBLK,
  packet --n-cu)` pair.
- **Deliverable is a cubin set**, not a packet field — `exec::gpu` already selects a cubin by
  profile name, so extend to `interp_sm90a__occ{N}_ctx{B}.cubin`.
- Cost ≈ 75 min per (gpu, dtype) for a 32-config × 4-ctx grid; belongs in `tunedb` as a
  recorded artifact, read-but-never-written by `compile` (the existing rule).

Ranked first sweeps: (1) occupancy pair — the 7 % knob; (2) `NS_ABS` × ctx — known ctx
interaction and a known-bad default; (3) unroll depths conditioned on the chosen occupancy;
(4) lane-split `SG` conditioned on K.

## FINAL

| | control | **shipped** | occ-2 | vLLM | gap |
|---|---|---|---|---|---|
| bf16 ctx1024 | 9.267 | **6.196** (1.50×) | **5.746** (1.61×) | 4.825 | 1.28× / 1.19× |
| bf16 ctx4096 | 9.461 | **6.291** (1.50×) | — | | |
| fp8 ctx1024 | 7.465 | **5.720** (1.30×) | — | 4.417 | 1.30× |

# Round 13 — the flash nsplit optimum MOVED once warp-per-row landed

`PLOW_NS_ABS` was swept in round 5 (pre-WPR) and 8 won. Warp-per-row made each flash work item
cheaper, which shifts the balance toward more, smaller splits — so it was re-swept:

| `PLOW_NS_ABS` | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|
| occ-1 (n_cu=132) bf16 | 6.191 | **6.044** | 6.083 | 6.275 | — |
| occ-2 (n_cu=264) bf16 | 5.856 | 5.681 | **5.616** | 5.781 | 5.775 |

**The optimum moved 8 → 16 (occ-1) and 8 → 32 (occ-2)**, worth 0.15 ms and 0.24 ms. It is
also occupancy-dependent, which is exactly the joint-sweep claim in the tuner design. The
`nsplit` that ships in a packet is emitted from `devgen`'s CU-fill formula (which gave 17 —
136 items ragged on 132 blocks); it is left as an emit-time override rather than a new devgen
default, because it depends on ctx, occupancy *and* the flash body, all of which this campaign
changed.

fp8 with the same setting (packet re-emitted through a checkpoint dir with canonical shard
names, since the emitter wants `model-{i}-of-{n}.safetensors`): **5.721 → 5.549 ms**.

## Measurement hygiene note

A process **outside our PID namespace** intermittently holds ~52.5 GB on this box, idle at 0 %
utilisation. It does not take the `gpulease` (the lease is advisory). Consequences, and how the
numbers above stay valid:

- **bf16 needs 49 GB, so a bf16 run that LOADS AT ALL proves the foreign process was absent.**
  Every bf16 number here comes from a successful load.
- **fp8 needs 26 GB and fits alongside it.** The fp8 number was taken with it resident but at
  0 % utilisation and SM clocks pinned at 1785 MHz, i.e. no compute contention.
- Runs attempted while it held memory fail with `cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY`, not
  with a wrong number — the failure mode is loud, not silent.

## Standing

| | control | **best measured** | vLLM | gap |
|---|---|---|---|---|
| bf16 ctx1024 (occ-1, ns=16) | 9.267 | **6.044** (1.53×) | 4.825 | 1.25× |
| bf16 ctx1024 (occ-2, ns=32) | 9.267 | **5.616** (1.65×) | 4.825 | 1.16× |
| fp8 ctx1024 (occ-1, ns=16) | 7.465 | **5.549** (1.35×) | 4.417 | 1.26× |

# Round 14 — the fp8 dense GEMV was the slowest arm in either precision

Ablating the fp8 step (full 5.548) put the cost somewhere the bf16 work had never looked:

| op | cost | note |
|---|---|---|
| **GEMV_FP8 (30)** | **1.157** | qkv + o_proj + down, ~1.21 GB ⇒ **1046 GB/s** |
| MoE expert GLU fp8 (65) | 0.708 | |
| MoE expert down fp8 (66) | 0.660 | |
| GEMV (10) | 0.586 | lm_head — stays **bf16** in this fp8 recipe |
| FLASH_DECODE | 0.452 | |
| GEMV_GLU_FP8 (31) | 0.244 | |

1046 GB/s is the slowest arm in either precision. Counting the inner loop explains it: per 8
bytes of fp8 weights the arm issues ~28 instructions (4 `cvt_fp8x2_to_halfraw2`, 4
`half22float2`, **8 `__bfloat162float` on x**, 8 `fmaf`), where bf16's `dot8` issues 24 per
**16** bytes — **2.3× the instructions per byte**. fp8 halves the weight traffic but not the
x widening, so this arm is **compute-bound, not bandwidth-bound**.

**Rejected first attempt:** staging x pre-widened to f32 in the arena. It removes the 8
widenings, but the arena is sized by the *emitter* for `K*2` bytes (bf16), so writing `K*4`
overruns it — `CUDA_ERROR_ILLEGAL_ADDRESS`. Fixing that would couple a cubin flag to a packet
field, where a mismatched pair faults at runtime. Not worth it.

**Shipped instead — row-blocked fp8 GEMV.** A warp owns `PLOW_NV_FP8_RB` rows, so each x chunk
is widened **once** and reused across all RB weight rows, and RB independent weight streams go
in flight. Registers are cheap here (`uint2` per element, not `uint4`), and the per-row FMA
order is unchanged.

| `PLOW_NV_FP8_RB` | 1 | 2 | **4** | 8 |
|---|---|---|---|---|
| fp8 ctx1024 | 5.556 | 5.366 | **5.336** (REG 182, 0 spill) | 5.499 (REG 213) |

Shipped fp8: **ctx1024 5.330, ctx4096 5.475**. bf16 is untouched (it never enters this arm).
`gpu_lifecycle` passes; sm_120 cubins byte-identical.

## Standing

| | control | **best measured** | vLLM | gap |
|---|---|---|---|---|
| bf16 ctx1024 (occ-1, ns=16) | 9.267 | **6.044** (1.53×) | 4.825 | 1.25× |
| bf16 ctx1024 (occ-2, ns=32) | 9.267 | **5.616** (1.65×) | 4.825 | 1.16× |
| **fp8 ctx1024 (occ-1, ns=16)** | 7.465 | **5.330** (1.40×) | 4.417 | **1.21×** |

# Round 15 — fp8 at 2 blocks/SM: 4.606 ms, within 4.3 % of vLLM

Round 6 emitted occ-2 for bf16 only ("fp8: not emitted"), because the fp8 devblob emitter
rejects the serving asset's renamed `bf16-`/`fp8-` shards. Building a checkpoint dir with
canonical `model-{i}-of-{n}.safetensors` names unblocks it, and fp8 turns out to be the better
precision for occupancy — it is compute-bound, so more blocks buy more than they cost.

Joint sweep at occ-2 (n_cu=264, `FORCE_MINBLK=2`), fp8, ctx=1024:

| knob | values | best |
|---|---|---|
| `PLOW_NS_ABS` | def 4.754 · 16 4.722 · **32 4.669** · 24 4.632 · 28 4.622 · 36 4.767 · 40 4.776 · 48 4.818 · 64 4.810 | **32** |
| `PLOW_NV_FP8_RB` | 1 4.785 · **2 4.615** · 3 4.659 · 4 4.667 | **2** |
| `GV_MOE_RB` | 2 4.618 · **1 4.608** | **1** |
| `GV_UNROLL` | **4 4.667** · 8 4.929 | **4** |
| `GV_MOE_UN` | **2 4.667** · 3 4.629 · 4 4.723 | **2** |
| `PLOW_MOE_DOWN_SG` | **4 4.615** · 8 4.625 | **4** |
| occupancy | occ-2 **4.606** · occ-3 5.159 (REG 80, 648 B spill) | **occ-2** |

**Shipped-equivalent best: fp8 ctx1024 4.606 ms, ctx4096 4.693 ms** (median of 5, spread
0.004 ms; REG 128, 232 B spill). `gpu_lifecycle` PASSES — `"Paris"` on both cycles.

Note `PLOW_NV_FP8_RB=2` wins here where **4** won at occ-1: the register cap at 2 blocks/SM is
128, so the deeper row-block that paid at occ-1 now costs more than it buys. Another knob whose
optimum inverts with occupancy.

## Standing

| | control | **best measured** | vLLM | gap |
|---|---|---|---|---|
| bf16 ctx1024 (occ-2, ns=32) | 9.267 | **5.616** (1.65×) | 4.825 | 1.16× |
| **fp8 ctx1024 (occ-2, ns=32)** | 7.465 | **4.606** (1.62×) | **4.417** | **1.043×** |

fp8 is now **4.3 %** off vLLM, from 1.69× at campaign start. bf16 could not be re-measured at
occ-2 with the newest arms this round — it needs 53.3 GiB and the box's foreign holder leaves
43.3 GiB — so its 5.616 predates the round-14/15 work and is a floor, not a ceiling.

# Round 16 — the router at occ-2, and a negative on widening its score arm

Ablating the winning fp8/occ-2 config (full 4.610) shows where the last 0.2 ms would have to
come from:

| op | cost |
|---|---|
| GEMV_FP8 (30) | 0.742 |
| MoE expert GLU fp8 (65) | 0.492 |
| GEMV (10) — lm_head, still bf16 in this recipe | 0.465 |
| **MoE router (68+69)** | **0.424** |
| MoE expert down fp8 (66) | 0.409 |
| FLASH_DECODE | 0.371 |
| MOE_COMBINE_NORM (70) | 0.246 |
| GEMV_GLU_FP8 (31) | 0.120 |

**The router is back.** Round 4 drove it to ~0 at occ-1 — but that was measured where it hid
behind slower arms. At occ-2, with everything around it faster, it re-emerges at 0.424 ms.
Splitting the two halves: **TOPK (68) 0.217 · SCORE (69) 0.052.**

**Negative result — `PLOW_MOE_ROUTER_WIDE`.** The score arm maps experts as `slice*8 + warp`,
so with n_exp=128 only **16 blocks of 264** get work. Giving each block one expert with all 8
warps splitting K (128 blocks instead of 16) measured **4.624 vs 4.611 — slightly worse**. The
diagnosis was wrong: SCORE is only 0.052 ms, so there was nothing there to win. Kept behind
`PLOW_MOE_ROUTER_WIDE=0`.

The real 0.217 ms is **TOPK**, which is one CTA running a 128-expert softmax + 8-pass argmax on
a single warp while 263 blocks wait on the gate. Widening it to the whole block is unlikely to
pay: at n_exp=128 each thread would hold ≤1 element, so the 8 argmax passes would cost 16
`__syncthreads` (~1600 cycles) against the current warp form's ~570 ops. Making this op cheap
needs it off the critical path (overlap), not more threads — the same conclusion round 8 reached
for COMBINE_NORM.

# Round 17 — two more flash fixed costs, found by working backwards from the target

The ablation gives a sharp target rather than a vague one. At the fp8/occ-2 optimum the
**non-flash remainder is 4.245 ms** against vLLM's **4.417** — so if flash were free plow would
win by 4 %. Flash's budget to break even is **0.172 ms**; it was 0.361, moving ~231 MB at
**640 GB/s** against a 3269 GB/s ceiling. That is not a bandwidth wall, it is fixed cost per
work item — at `nsplit=32` an item owns only ~32 of the 256 tile rows.

Two of those fixed costs removed:

**`PLOW_NV_FA_QGLOB` — stop staging Q.** Q is `GF*D` bf16 (1 KiB) and L2-resident, but staging
it costs a full `__syncthreads` on *every* work item — ~7680 of them per token at 256
items/layer × 30 layers. Reading it from global instead: **4.608 → 4.588 ms**.

**`PLOW_NV_FA_WPR_RB` — batch the warp's rows.** The warp-per-row sweep processed its ~4 rows
one at a time, so ~1 load was in flight. Batching them issues WRB independent row loads
back-to-back for one warp reduction each. The loads are unconditional because `kvr & kv_mask`
always lands on a real ring row, so only the dot and the store are gated — keeping the batch
actually back-to-back.

| `PLOW_NV_FA_WPR_RB` | 1 | **2** | 4 |
|---|---|---|---|
| fp8 occ-2 ctx1024 | 4.598 | **4.560** | 4.665 |

**Combined: 4.610 → 4.558 ms** (median of 5, spread 0.006). flash **0.361 → 0.307 ms**.
`gpu_lifecycle` PASSES — `"Paris"` on both cycles.

## Standing

| | control | **best measured** | vLLM | gap |
|---|---|---|---|---|
| bf16 ctx1024 (occ-2) | 9.267 | **5.616** (1.65×) | 4.825 | 1.16× |
| **fp8 ctx1024 (occ-2)** | 7.465 | **4.558** (1.64×) | **4.417** | **1.032×** |

fp8 is now **3.2 %** off vLLM. The remaining 0.135 ms would have to come out of flash's
0.307 — the non-flash floor (4.245) is already below vLLM's total.

# Round 18 — the tuner's own result: a flash optimum that moves with ctx

The flash extension answered the design's open question. `NS_FULL_ABS` — the split count for the
5 full-attention layers — is the first knob in either family whose **winner**, not just effect
size, moves with context (occ-2, 26B fp8, ablated flash cost in parens):

| `NS_FULL_ABS` | ctx 1024 | ctx 8192 |
|---|---|---|
| emitter default | **4.604** (0.361) | 4.811 (0.573) |
| 33 | 4.605 (0.363) | 4.802 (0.557) |
| 66 | 4.612 (0.343) | **4.715** (0.445) |

Default wins at 1k; **66 wins at 8k by 0.096 ms (~2 % of the step)**. And 66 is the *principled*
value at this occupancy — `n_cu/gcd(n_grp,n_cu)` = 264/4 — where `build_sm90a_cubin.sh`'s
recorded 33 is the occ-1 figure. So the knob is coupled to occupancy **and** ctx, which is the
joint-sweep claim demonstrated on a second family.

The ablation supplies the mechanism, and it was a **prediction before it was a measurement**:

| ctx | Δ flash (body) | Δ ablated (gate) | Δ TPOT |
|---|---|---|---|
| 1024 | −0.018 | +0.026 | +0.008 |
| 8192 | **−0.128** | +0.032 | **−0.096** |

Splitting trades flash-body time against gate/protocol time for the extra work items — which
still gate and signal with the body compiled out, which is exactly why the twin sees them. The
gate cost is near-constant while the body saving grows 7×, so splits pay only once the body term
outruns a fixed protocol cost. That is a ctx condition, stated as a mechanism rather than a fit.

Two further flash results, both from the twin rather than from TPOT:

- **`FA_GF_FULL=8` is refuted hardest where it should have won** (+0.242 ms at 1k → +0.517 at
  8k). Split into a **constant ~0.080 ms arena tax** — dynamic smem 16448 → 24640 B with
  `occ_per_sm` still 2, so not occupancy; the arena is a **union sized by the largest claim**,
  and that claim is flash's, so widening it bills every other op for space only flash uses — plus
  a flash penalty growing +0.161 → +0.437.
- **`FA_WPR=1` confirmed at 2.53×** (flash 0.914 → 0.361) with the ablated remainder *unchanged*
  (4.253 vs 4.243), proving the knob moved only the arm it names.

**And the cleanest measurement of the campaign:** the non-flash remainder is flat to **0.007 ms
across a 32× context range** (4.243 / 4.238 / 4.236) while flash grows 3.6× (0.361 → 1.314).
**100.7 % of the context growth is the flash arm** — the campaign's assertion turned into a
measurement with the rest of the engine held as a control. It also gives the tuner a priority
rule: flash is 7.8 % of the step at ctx=1024 and 23.7 % at 32768, so below ~8k tune the GEMM
knobs and above it tune flash.

The tuner is documented in `tuning/README-decode-tuner.md`.

## Round 18 addendum — nsplit re-checked after the round-17 flash changes (held)

Round 13 found the `NS_ABS` optimum moved (8 → 32) when warp-per-row landed, and the tuner's
own framing is that an optimum can move whenever the body under it changes. Round 17 changed
the flash body twice (`FA_QGLOB`, `FA_WPR_RB`), so it was re-swept rather than assumed:

| `PLOW_NS_ABS` (occ-2, fp8, ctx1024) | 16 | 24 | **32** |
|---|---|---|---|
| TPOT | 4.592 | 4.583 | **4.560** |

**The optimum held at 32.** Recorded because "we re-checked and nothing moved" is a result: it
bounds how often the joint sweep actually has to be re-run, and the alternative — assuming it
held — is how `GV_MOE_UN` and `NS_ABS` went stale in the first place.

## Round 18 addendum 2 — `FA_WPR_RB` is occupancy-specific (shipped default confirmed)

`PLOW_NV_FA_WPR_RB=2` won at 2 blocks/SM (4.560 vs 4.598). At the **shipped** 1 block/SM it is
a dead heat — fp8 ctx1024 **5.334 (RB=1) vs 5.335 (RB=2)**, median of 3, spread 0.006 — so the
source default of 1 is right for the shipped object and the win is real only at occ-2.

That makes **four** knobs whose optimum depends on occupancy (`GV_UNROLL`, `PLOW_NS_ABS`,
`PLOW_NV_FP8_RB`, and now `FA_WPR_RB`), which is the case for the tuner owning them per cell
rather than any of them being a single constant.
