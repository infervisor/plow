# GLM-5.2-FP8 TP8 on gfx942 — the CURRENT cost decomposition

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **CDNA3-SPECIFIC** — every row is `busy CU-us / 304 x n_layers`. The CU count, the MFMA rate and the HBM rate are all MI300X; the DECOMPOSITION (which component owns what share) is the transferable part.

**Status: authoritative. Supersedes every trace-based per-layer number in
`glm52-experiments.md (consolidated; superseded)`, `glm52-mla-pf-decomposition.md`,
`glm52-experiments.md (consolidated; superseded by this file's own numbers)` and `glm52-decode-packet-folds.md`** — all of those
were taken before the V2 flash, causal KV-split `ns2`, the SV LDS bank swizzle,
the q-rope and layer-seam packet folds, `PLOW_MOE_PF_EPI` and `PLOW_MOE_DEC_LG`.

## 0. What was measured, and with what

| | |
|---|---|
| blob | `/workspace/assets/gfx942/glm52-tp8-final2/model.pkt` (unchanged) |
| objects | `/root/.claude/jobs/b09a4bcc/tmp/hsaco_glm18` (`PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 PLOW_MOE_PF_EPI=1`; `PLOW_MOE_DEC_LG` default-on) |
| runtime | `plowrt` built at `7ccd601` in worktree `trace-current`, `--features hsa` |
| env | `PLOW_MLA_PF_V2=1` |
| instrument | `PLOW_TRACE_RAW`, `plowrt amd-bench --tp 8 --prompt <ids> --steps 8` |
| reducer | `scripts/glm52_layer_census.py` (new; grouped by `inst`, NOT `pc`) |
| tables | `scripts/glm52_decomp_tables.py`, `glm52_gap_attrib.py`, `glm52_sparse_curve.py`, `glm52_cost_model.py` |
| run | one session, 2026-08-08 12:47–13:00, GPU lock held, `rocm-smi` 0% on acquire |

**Nothing in `runtime/`, `crates/` or the shipped config was modified.** The only
new files are the four analysis scripts above.

Model: 78 layers = **3 dense (L0–L2, 22 packets/layer) + 75 MoE (L3–L77, 26 packets/layer)**
in prefill; 14 / 33 packets/layer in decode. Dense layers reuse the *same*
`MoeGroup*Pf` opcodes with `n_exp=1, I_moe=1536`, so "MoE GLU/DOWN" rows appear in
the dense table too — there they are the dense FFN.

**Two provenance rules this file had to establish, both of which invalidate a naive read:**

1. **`--prompt` decode traces are poisoned.** The trace buffer is indexed per
   (workgroup, packet) and sized for the *widest* program. Prefill writes
   2021×304 = 614k slots; one decode step writes only ~136k. The decode dump
   therefore carries ~454k stale PREFILL records mapped onto decode `inst`
   indices, which read as 360,000 µs packet spans. `glm52_layer_census.py
   --last-dispatch` splits the file at the largest idle gap (222.9 ms here) and
   keeps the 135,834 records of the last decode step. **Every decode number below
   is from the filtered set, at REAL KV** (after a real prefill), not a promptless run.
2. **In prefill, packet spans OVERLAP.** The sum of packet spans exceeds the
   layer span at every context. So `% of layer` computed from spans over-counts;
   the additive quantity is **busy CU-µs**, and that is what the attribution uses.

Two clocks are quoted and they differ: **device prefill wall** (`amd-bench`, this
session: 343.5 / 731.2 / 1390.0 / 3239.4 ms) and **served TTFT** (round-3 gate,
`glm52-experiments.md (consolidated: LANDED table)`: 343.3 / 973.4 / 1677.0 / 3627.4 ms). §1.4
shows the difference is not noise.

---

## 1. DELIVERABLE 1 — single-chunk prefill decomposition

### 1.0 Chunking, first, because it changes what "T" means

`plan_chunks` (`crates/plowrt/src/exec/amd.rs:1347`, `MAX_CHUNK = 8192`) covers a
prompt from the compiled bucket ladder `[128, 512, 1024, 2048, 4096, 8192]`:

```
1024 -> [1024]      4096 -> [4096]      8192 -> [8192]      16384 -> [8192, 8192]
```

The trace buffer holds the LAST dispatch, so the T=16384 row below is **chunk 2**
(8192 queries attending over 16384 KV) — the expensive half. Chunk 1 of a 16k
prompt is priced from the T=8192 row.

### 1.1 One MoE layer (median of L6..L74), busy CU-µs and its share of the layer's CU budget

`busy` is additive; the denominator is `304 CU × layer span`.

| category | T=1024 | | T=4096 | | T=8192 | | T=16384 c2 | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | kCU-µs | %CU | kCU-µs | %CU | kCU-µs | %CU | kCU-µs | %CU |
| flash (`FlashMlaPrefill`) | 48 | 3.6 | 204 | 7.1 | **852** | **15.6** | **2445** | **33.7** |
| MoE DOWN (`MoeGroupDownPf`) | 168 | 12.7 | 437 | 15.2 | 849 | 15.6 | 807 | 11.1 |
| attn+shared GEMM (`Gemm*`, `GemmGlu`) | 207 | 15.6 | 434 | 15.1 | 740 | 13.6 | 742 | 10.2 |
| MoE GLU (`MoeGroupGluPf`) | 150 | 11.3 | 379 | 13.2 | 653 | 12.0 | 637 | 8.8 |
| collectives (`XReduceTwoShot` ×2) | 143 | 10.8 | 353 | 12.3 | 630 | 11.6 | 801 | 11.0 |
| flash merge (`MlaMergeFold`) | 61 | 4.6 | 231 | 8.1 | 467 | 8.6 | 477 | 6.6 |
| MoE combine (`MoeCombinePf`) | 61 | 4.6 | 220 | 7.7 | 460 | 8.4 | 423 | 5.8 |
| norms/residual | 105 | 7.9 | 136 | 4.7 | 205 | 3.8 | 210 | 2.9 |
| MoE router+align | 20 | 1.5 | 53 | 1.8 | 96 | 1.8 | 95 | 1.3 |
| **— dependency wait (gate)** | **224** | **17.0** | **249** | **8.7** | **264** | **4.8** | **287** | **4.0** |
| **— CU in no packet at all** | 138 | 10.4 | 174 | 6.0 | 228 | 4.2 | 332 | 4.6 |
| **layer span (µs)** | **4355** | | **9439** | | **17907** | | **23870** | |
| ×75 layers (ms) | 326.6 | | 707.9 | | 1343.0 | | 1790.3 | |
| packing efficiency | 72.7% | | 85.3% | | 91.0% | | 91.5% | |

### 1.2 One DENSE layer (L0..L2) — it is a different machine

| category | T=1024 %CU | T=4096 %CU | T=8192 %CU | T=16384 c2 %CU |
|---|---:|---:|---:|---:|
| flash | 5.7 | 10.8 | **21.1** | **42.7** |
| collectives ×2 | **37.3** | **24.6** | 19.6 | 12.7 |
| attn+shared GEMM | 18.5 | 18.2 | 15.3 | 10.6 |
| flash merge | 7.0 | 12.6 | 12.0 | 8.0 |
| dense FFN GLU (op 85, n_exp=1) | 7.8 | 10.6 | 10.7 | 7.7 |
| norms/residual | 12.8 | 7.3 | 5.4 | 3.6 |
| dense FFN DOWN (op 86) | 4.5 | 5.9 | 5.2 | 3.6 |
| dense FFN combine | 2.7 | 3.3 | 3.1 | 2.1 |
| dependency wait (gate) | 13.8 | 8.2 | 5.9 | 3.9 |
| **layer span (µs)** | **2855** | **6382** | **12807** | **19061** |
| ×3 layers (ms) | 8.6 | 19.1 | 38.4 | 57.2 |

A dense layer is **0.66–0.80× a MoE layer**, and its cost mix is different: the
routed-expert weight stream is gone, so **collectives are the single largest term
at ≤4k** (37% of the CU budget at 1k). The 3 dense layers are 2.5–2.8% of TTFT —
too small to be a target, but they are *why* the "78 × MoE layer" shortcut
over-predicts TTFT by ~2%.

### 1.3 Dependency wait vs real work — the answer for prefill

The campaign's decode figure is 63.1% of in-packet CU time spent on the gate.
**Prefill is not the same machine:**

| | gate-wait share of in-packet CU time | packing efficiency | wall with no packet running |
|---|---:|---:|---:|
| prefill MoE layer, T=1024 | 18.9% | 72.7% | 154 µs/layer (3.5%) |
| prefill MoE layer, T=4096 | 9.2% | 85.3% | 177 µs/layer (1.9%) |
| prefill MoE layer, T=8192 | **5.1%** | **91.0%** | **286 µs/layer (1.6%)** |
| prefill MoE layer, T=16384 c2 | 4.1% | 91.5% | 396 µs/layer (1.7%) |
| **decode MoE layer, ctx 1024** | **67.5%** | **31.1%** | **33 µs/layer (10.5%)** |

**Prefill is a KERNEL-RATE problem; decode is a SCHEDULING problem.** At 8k, 91%
of every CU-second inside a prefill layer is executing a kernel body. There is no
packing prize in prefill worth naming — the entire wait + idle budget at 8k is
9.0% of the CU-time and shrinks as T grows. Any plan that proposes a claim-path
or protocol rebuild to fix TTFT is aiming at ≤9% of a term that is already at 91%.

Where the wait *is* concentrated at small T: `MoeRouterTopkPf` (62.5% wait @1k),
`GemmWide` (63.3% @1k), `Residual` (28–46% at every T), `HeadNormRope` (~31%).
All of these are the small-fixed-cost packets of §1.5.

### 1.4 Reconciliation, and the 203 ms nobody had priced

Sum the traced layer spans over 3 dense + 75 MoE layers, per chunk:

| T | chunks | model ms | device prefill wall | residual | vs served TTFT | residual |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 335.2 | 343.5 | +8.3 (2.4%) | 343.3 | +8.1 |
| 4096 | 1 | 727.1 | 731.2 | +4.1 (0.6%) | 973.4 | **+246.3** |
| 8192 | 1 | 1381.4 | 1390.0 | +8.6 (0.6%) | 1677.0 | **+295.6** |
| 16384 | 2 | 3228.9 | 3239.4 | +10.5 (0.3%) | 3627.4 | **+398.5** |

**The trace closes on the device wall to 0.3–2.4%.** The decomposition is sound.
The 246 / 296 / 399 ms is a SERVING-PATH term, and it is not noise. Measured
directly:

```
prefill 1024 tokens -> 343.5 ms        prefill 4096 tokens -> 731.2 ms
prefill 1025 tokens -> 546.5 ms        prefill 4097 tokens -> 964.5 ms
        ONE MORE TOKEN  = +203.0 ms            ONE MORE TOKEN  = +233.3 ms
```

**One token past a bucket boundary costs 203–233 ms.** `plan_chunks` covers 1025
as `[1024, 128]`, and a 128-row chunk pays the *entire* T-invariant cost of a
78-layer forward pass. Fitting each traced layer span to `c0 + c1·T` over
1k/4k/8k gives an intercept of **2135 µs/MoE layer + 1144 µs/dense layer = 163.6 ms
per chunk**, and the 128 rows add ~19 ms — 183 ms modelled against 203 measured.

This is not a harness artifact to be waved away, it is the harness *exposing* a
real property: `scripts/bench_speed.sh` builds its 4k prompt as 455×9 = 4095
content tokens plus the chat template, i.e. 4096 < len ≤ 4224, so **every served
TTFT at 4k/8k/16k in the campaign's tables contains one ragged 128-row relaunch.**
The 1k prompt happens to land at ≤1024 and pays nothing, which is exactly why the
1k row reconciles to 8 ms and the others do not.

`plan_chunks` charges `LAUNCH_ROWS = 416` rows per launch. The measured launch
cost is 203 ms ≈ 1400 rows at the 8192-chunk marginal rate. **The DP is optimising
a cost model that understates a GLM prefill launch by ~3.4×.** (It still picks
`[4096,128]` over `[8192]` here — padding to 8192 costs more — but it has no
option that avoids the second launch.)

### 1.5 How each component scales with T

Least-squares `span = c0 + c1·T` per MoE layer over T=1k/4k/8k (R² in parentheses):

| category | c0 (µs/layer) | c1 (µs/token) | ×75 layers: fixed ms | ×75: ms per 1k tokens | scaling |
|---|---:|---:|---:|---:|---|
| attn+shared GEMM | **1588** (1.000) | 0.2546 | **119.1** | 19.1 | linear + huge fixed |
| flash | −309 (0.941) | 0.4010 | — | 30.1 | **quadratic in pairs, see below** |
| MoE GLU | 425 (0.997) | 0.2350 | 31.9 | 17.6 | linear + fixed (weights) |
| MoE DOWN | 229 (0.999) | 0.3141 | 17.1 | 23.6 | linear + fixed (weights) |
| collectives ×2 | 275 (1.000) | 0.2394 | 20.6 | 18.0 | linear |
| flash merge | 45 (1.000) | 0.1883 | 3.4 | 14.1 | **linear** (not quadratic) |
| MoE combine | 18 (0.999) | 0.1849 | 1.4 | 13.9 | linear |
| norms/residual | 456 (0.998) | 0.0512 | 34.2 | 3.8 | linear + fixed |
| MoE router+align | 109 (1.000) | 0.0584 | 8.1 | 4.4 | linear |
| **layer span** | **2135** | **1.9001** | **160.2** | **142.5** | |

Flash is the only term that is not linear in T. Fitting it to the causal pair
count instead is exact:

```
flash span/layer = 287 us + 83.12 us per 1e6 causal (query,key) pairs      R^2 = 1.000
   T=1024        0.52e6 pairs   ->    339 us   (measured 339)
   T=4096        8.39e6 pairs   ->    919 us   (measured 919)
   T=8192       33.56e6 pairs   ->   3154 us   (measured 3154)
   T=8192 @ctx16k 100.7e6 pairs ->   8634 us   (measured 8634)
```

**What this says about 32k.** Per-chunk costs are fixed + linear except flash,
which is quadratic in total context. At T=32768 (4 chunks) the causal pair count
is 5.37e8, so flash alone projects to `78 × (0.287 + 83.12e-6 × 5.37e8/4 …)` —
summed over the four chunks, **3570 ms of flash against roughly 800 ms of
everything else per-chunk × 4**. Flash goes from 16% of the layer at 8k to 35% at
16k(c2) and is the *only* term that keeps growing. **At 32k, attention is the
decomposition.** Everything else in this table is already linear and already at
or near a proven floor.

### 1.6 The MoE grouped pair against its references

| T | plow pair span/layer | plow DRAM | plow effective BW | aiter bytes (fused) | aiter at its measured 1.53 TB/s | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1214 µs | 1364 MB | 1.10 TB/s | 1221 MB | 779 µs | 1.56× |
| 2048 (fitted) | 1779 µs | 1576 MB | 0.90 TB/s | 1233 MB | 788.8 µs **(measured)** | **2.25×** |
| 4096 | 2905 µs | 2000 MB | 0.67 TB/s | 1258 MB | 803 µs | 3.62× |
| 8192 | 5151 µs | 2848 MB | 0.54 TB/s | 1309 MB | 836 µs | 6.16× |

The M=2048 row is the only one where **both** engines were measured on this box
(`glm52-experiments.md (consolidated: MoE k-loop, six arms null)` §4.1: aiter
`fmoe_bf16_blockscaleFp8_g1u1_…32x256.co`, E=256, topk=8, inter=256, M=2048 =
788.8 µs). On the *current* objects that ratio is **2.25×**, down from the audit's
2.76× — `PLOW_MOE_PF_EPI` is the difference. The 3.62×/6.16× rows extrapolate
aiter at a fixed 1.53 TB/s and are **labelled estimates**: aiter's M=8192 point
was never measured and its FLOPs grow 4× from M=2048, so those two ratios are
upper bounds on aiter's advantage.

What is *not* an extrapolation is the direction: **plow's deficit grows with M**,
because plow's DRAM traffic grows with M and aiter's barely does. Per layer per
rank the routed weights are 1.208 GB *regardless of M*; plow adds
`T·k·H·4 = 1.611 GB` of f32 `part` scatter at T=8192 (written by op 86, read back
by op 87) plus 64 MB of `fu`. That is the "≈40% architectural" half of the audit's
split, and at M=8192 it is the majority of the bytes.

plow's own roofline at T=8192: `max(2.848 GB / 3.5 TB/s, 618 GFLOP / 937 TF/s)
= max(814, 660) = 814 µs/layer`. Measured 5151 µs = **6.3× off its own roofline**.

---

## 2. DELIVERABLE 2 — single-step decode decomposition

Real KV, after a real prefill; last of 8 steps; `--last-dispatch` filtered.

### 2.1 ctx 1024 (amd-bench 27.356 ms/token; served 26.72)

| op | pkt/layer | span µs | busy CU-µs/layer | gate-wait CU-µs/layer | **ms/token (real work)** | ms/token (gate wait) |
|---|---:|---:|---:|---:|---:|---:|
| `Gemv` (o_proj, router, shared-down) | 3 | 75.3 | **11063** | 20375 | **2.839** | 5.228 |
| `MoeExpertGluFp8Blk` | 8 | 207.8 | 4937 | 2894 | 1.267 | 0.742 |
| `GemvQkv` | 2 | 37.0 | 4695 | 19677 | 1.205 | 5.049 |
| `FlashMlaDecode` | 1 | 35.8 | 4139 | 4974 | 1.062 | 1.276 |
| `MoeExpertDownFp8Blk` | 8 | 127.0 | 3134 | 9062 | 0.804 | 2.325 |
| `MlaMergeFold` | 1 | 18.2 | 1046 | 3769 | 0.268 | 0.967 |
| `GemvGlu` | 1 | 19.5 | 524 | 233 | 0.135 | 0.060 |
| `XReduce` ×2 | 2 | 30.4 | 336 | 700 | 0.086 | 0.180 |
| `MoeCombine` | 1 | 7.0 | 69 | 451 | 0.018 | 0.116 |
| the five 1-workgroup ops | 6 | 42.4 | 43 | 184 | 0.011 | 0.047 |
| **TOTAL real work** | 33 | | **29987** | | **7.694** | |
| **TOTAL gate wait** | | | | **62317** | | **15.989** |
| **CU in no packet** | | | 4152 | | 1.065 | |
| **layer span ×78** | | 317.3 | | | **24.75 ms** | |
| non-layer + host residual | | | | | 2.61 ms | |

### 2.2 ctx 4096 (amd-bench 29.055; served 29.06)

Identical except `FlashMlaDecode` 4139 → **7948** CU-µs (1.062 → 2.039 ms/token,
+0.98 ms) and gate wait 15.99 → 17.48 ms. **Everything else moves <2%.** Layer
span 317.3 → 346.9 µs; ×78 = 27.06 ms.

The entire 1k→4k TPOT delta (26.72 → 29.06 served, +2.34 ms) is
**+0.98 ms of flash body + ~1.4 ms of extra waiting behind it.**

### 2.3 Confirmation of the current top consumers

- **`Gemv` is now the largest decode consumer, 2.84 ms/token (36.9% of real work).**
  This is exactly what `glm52-packet-protocol-xcd.md` predicted after `LG`
  landed ("largest remaining decode busy row with lgx2 in: `Gemv` at 11285
  CU-µs/layer"). Measured here: **11063 CU-µs/layer.** Of its three packets the
  shared-expert down projection is N=6144 **K=256** — the same narrow-K lane
  defect `PLOW_MOE_DEC_LG` just fixed in the fp8 body.
- **`MoeExpertDownFp8Blk` has collapsed**: 11109 → **3134** CU-µs/layer (−72%),
  matching the reported −81% per-workgroup body. DOWN is now the *fifth* largest
  term, behind GLU.
- **Gate wait ROSE to 67.5%** (from 62.8% post-fold, 63.1% protocol). LG removed
  real work without removing waiting, so the share went up while the token went
  down. Packing efficiency is **31.1%**; perfect packing of the same work is
  **7.69 ms/token against a measured 27.36** — the packing prize is ~19.7 ms/token
  and it is still the largest single number in decode.

### 2.4 The 2.13 µs packet boundary, re-measured — it reproduces, and it is half-exposed

Two instruments, both on the current config:

| instrument | ctx 1024 | ctx 4096 |
|---|---:|---:|
| sum of positive gaps in `inst` order (the method of `glm52-decode-packet-folds.md`) | 66.9 µs/layer = **2.03 µs/boundary** | 64.6 µs = **1.96 µs** |
| ×33 boundaries ×78 layers | **5.22 ms/token = 19.1%** | 5.04 ms = 17.3% |
| **wall time with NO packet executing anywhere in the layer** (interval union) | **33.3 µs/layer = 1.01 µs/boundary** | 30.2 µs = 0.92 µs |
| ×78 layers | **2.60 ms/token = 9.5%** | 2.36 ms = 8.1% |

**The 2.13 µs × 33 × 78 = 5.5 ms (19%) model HELD as stated** — the same
statistic now reads 2.03 µs and 19.1%. But it is **not "fully exposed"**: half of
those gaps have another packet running in them. `GLM_MOE_CORESIDENT=2` puts 8
expert slots on disjoint CU partitions, and the layer-40 timeline shows the
expert packets with gaps of −44 to −24 µs (i.e. overlapping by that much); the
k-rope and kv_a norms likewise finish off the critical path. The honest number for
"wall time the machine is doing nothing" is **2.60 ms/token, 9.5%** at ctx 1024.

Correction of record: **the exposed protocol boundary is 9.5% of the token, not
19%.** The other 9.6% is real work that happens to be on a different packet. The
prize from a protocol rebuild is therefore ~2.6 ms/token, not ~5.5.

---

## 3. DELIVERABLE 3 — the vLLM gap, attributed

### 3.1 A caveat on the vLLM baseline that has to be stated first

vLLM 0.26 AITER on this box measures 69 / 566 / 672 / 1631 ms. Fit
`TTFT = a·T + b·pairs(T)` on the **8k and 16k** rows and evaluate everywhere:

| T | model | measured | residual |
|---:|---:|---:|---:|
| 1024 | 68.3 | 69.0 | **+1%** |
| 4096 | 300.1 | 566.0 | **+89%** |
| 8192 | 672.0 | 672.0 | (fitted) |
| 16384 | 1631.0 | 1631.0 | (fitted) |

`a = 64.5 µs/token, b = 4.28 ns/causal-pair`. Three of the four vLLM points lie
on a two-term curve to within 1%; **the 4k cell is 266 ms above vLLM's own
curve.** The 4k gap column below is therefore an *under*-statement of the work
plow has to do, and the 1.72× ratio at 4k is the softest number in the campaign.
Independent sanity check on `a`: the routed-expert weights are **84.4 GB per
chunk per rank** (75 layers × 1.208 GB), which at aiter's measured 1.53 TB/s is
**59.2 ms** — 90% of vLLM's entire 1k TTFT. vLLM's 1k number is essentially the
MoE weight stream and nothing else, which is a strong sign the baseline is real.

### 3.2 THE GAP ATTRIBUTION TABLE

plow's rows are `busy CU-µs / 304 × n_layers`, i.e. the component's perfect-pack
cost, summed over the chunk plan. `schedule overhead` is the layer span the
components do NOT get back (gate wait + idle CU). `ragged tail chunk` is measured
directly (§1.4).

| component | T=1024 | T=4096 | T=8192 | T=16384 | scaling | at a proven floor? |
|---|---:|---:|---:|---:|---|---|
| **MoE grouped path** (GLU+DOWN+combine+router) | 99.6 | 272.4 | **515.1** | **1006.9** | linear + 49 ms/chunk fixed | **NO — 2.25× off aiter measured same-shape** |
| **flash attention** (`FlashMlaPrefill`) | 12.4 | 52.3 | 218.2 | **845.8** | **quadratic in causal pairs** | kernel yes / **structure NO (dense vs top-2048)** |
| **attn+shared GEMM** | 52.6 | 110.5 | 188.4 | 377.5 | linear + 119 ms/chunk fixed | **NO — 0.51× hipBLASLt** |
| **collectives** (`XReduceTwoShot` ×2) | 38.4 | 91.8 | 162.9 | 367.9 | linear | **YES — see 3.3** |
| **schedule overhead** (gate wait + idle CU) | 89.6 | 105.6 | 124.3 | 281.9 | sublinear | **YES — 91% packed at 8k** |
| **flash merge** (`MlaMergeFold`) | 15.8 | 59.5 | 119.9 | 242.2 | **linear** | unexamined |
| **MoE combine** (`MoeCombinePf`) | 15.2 | 54.9 | 114.7 | 220.2 | linear | folds into the MoE item |
| **norms/residual** | 26.9 | 35.0 | 52.7 | 106.7 | linear + 34 ms/chunk fixed | mostly |
| **ragged tail chunk** (128-row relaunch) | 0.0 | **203.0** | **203.0** | **203.0** | **flat** | **NO — pure waste** |
| unattributed (host, 1st decode step, run-to-run) | 8.1 | 43.3 | 92.6 | 195.5 | | |
| **plow TTFT (served)** | **343.3** | **973.4** | **1677.0** | **3627.4** | | |
| **vLLM TTFT** | 69.0 | 566.0 | 672.0 | 1631.0 | | |
| **GAP to remove** | **274.3** | **407.4** | **1005.0** | **1996.4** | | |

Read as sentences:

- **Of the 1005 ms we must remove at 8k:** 515 ms is the MoE grouped path,
  218+120 = 338 ms is attention (body + merge), 188 ms is GEMM, 203 ms is a
  ragged relaunch that does no useful work, 163 ms is collectives, 124 ms is
  schedule overhead, 53 ms is norms. (These sum past 1005 because plow's total is
  1677, not 1005 — the gap is what is left after vLLM's own 672 ms of the same work.)
- **Of the 1996 ms at 16k:** 846+242 = 1088 ms is attention and it is the
  majority; 1007 ms is MoE; 378 ms GEMM; 368 ms collectives; 203 ms ragged relaunch.
- **Of the 407 ms at 4k:** 272 ms MoE, 203 ms ragged relaunch, 111 ms GEMM,
  92 ms collectives, 52+60 = 112 ms attention. Note two of those alone (MoE +
  ragged) exceed the whole gap, and the gap itself is soft (§3.1).
- **Of the 274 ms at 1k:** 100 ms MoE, 90 ms schedule overhead, 53 ms GEMM,
  38 ms collectives, 28 ms attention. **1k is the fixed-cost context**: 164 ms of
  the 343 is the per-chunk intercept (fitted layer-span intercept ×78), and the
  largest single fixed *category* is narrow-M GEMM at 122 ms of intercept.
  (Category intercepts sum past the layer intercept because prefill packet spans
  overlap, so 122/164 is an indication of dominance, not a strict share.) The 4.97× ratio at 1k is a *small-M* problem, not an
  MoE problem.

### 3.3 Components already at or near a proven floor — excluded from the addressable pool

- **Collectives — 162.9 ms @8k = 11.8% of served TTFT.** This independently
  reproduces `glm52-band-pipeline-cusubset.md`'s 12.2% ceiling, which is the cost
  of deleting them *entirely*. The two-shot is at ~81% of its own microbench
  floor, the fabric law (rate linear in threads to a ~78k-thread peak) is
  measured three ways, banding is closed negative, push/pull/width/depth are all
  measured nulls, and quantized AR's own upper bound is −4.4% of TTFT at a
  numerics cost already ruled unshippable on a gentler tensor. **Addressable ≈ 0.**
- **Prefill schedule overhead — 124 ms @8k.** 91.0% packing efficiency, 5.1%
  gate wait, 1.6% of the layer with nothing running. There is no protocol or
  claim-path prize in prefill. **Addressable ≈ 0.**
- **The flash kernel body.** At 3154 µs/layer @8k it is already *below* the
  5.5–6 ms/layer "honest no-numerics floor" that `glm52-experiments.md (consolidated; superseded by this file's own numbers)`
  published — `ns2` + the SV swizzle landed after that report. Its perfect-pack
  cost is 2802 µs/layer for 584 GFLOP = **208 TF/s = 22% of the 937 TF/s issue
  ceiling**, and the four ablation terms (K-slab bandwidth, softmax VALU,
  LDS-blocked PV, QK MFMA) are each priced and each structural. **The kernel is
  near floor; the addressable attention term is STRUCTURAL and is §4.**

---

## 4. DELIVERABLE 4 — attention specifically: dense vs an ideal top-2048

vLLM runs GLM's trained DSA sparsity (`index_topk = 2048`); plow runs dense
(`PLOW_GLM_DSA=0`, which wins at every context to 32k *today*).

Fit the traced flash span to the causal pair count (R² = 1.000, §1.5), then
re-evaluate at the top-2048 pair count. The counterfactual is **ideal**: flash
busy exactly proportional to selected pairs, i.e. a per-query
membership-skipping kernel.

| chunk | causal pairs | top-2048 pairs | ratio | flash measured | ideal sparse | saved µs/layer |
|---|---:|---:|---:|---:|---:|---:|
| T=1024 | 5.25e5 | 5.25e5 | 1.000 | 339 | 330 | 0 |
| T=4096 | 8.39e6 | 6.29e6 | 0.750 | 919 | 810 | 174 |
| T=8192 | 3.36e7 | 1.47e7 | 0.437 | 3154 | 1507 | **1569** |
| T=8192 @ctx16k | 1.01e8 | 1.68e7 | **0.167** | 8634 | 1681 | **6973** |

`MlaMergeFold` (120–242 ms) does **not** shrink: it is linear in T (the `ns2`
partial merge), not in pairs. It is the price of the 33.7% `ns2` win and it is
untouched by sparsity.

Whole-prompt, ×78 layers. **Basis note:** this section prices flash by its
*packet span* ×78, whereas §3.2 prices it by perfect-pack busy CU-µs; the span
basis is 5–15% larger (240 vs 218 ms @8k), so the saving column below is on the
same 5–15%-generous basis. Against plow's own **measured** indexer price (op 117
`IndexScorePf` materialises a T×ctx f32 score matrix = 3.7 ms/layer at 16k,
`glm52-dsa-sparse-b2.md`, scaled by pair count since it is quadratic in the same
variable):

| T | flash dense ms | flash ideal-sparse ms | **gross saving** | % TTFT | plow indexer ms | **NET** | % TTFT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 77 | 63 | 14 | 1.4% | 18 | **−4** | −0.5% |
| 8192 | 240 | 118 | **122** | 7.3% | 72 | **+50** | +3.0% |
| 16384 | 915 | 249 | **666** | 18.4% | 289 | **+378** | +10.4% |
| 32768 | 3570 | 511 | **3059** | — | 1154 | **+1905** | — |

**The structural attention gap — the work plow does that the model was not
trained to need — is 174 µs/layer at 4k, 1569 at 8k, 6973 at 16k-chunk-2, and it
grows without bound.** In TTFT terms: 14 / 122 / 666 ms at 4k / 8k / 16k, and
~3.1 s at 32k.

**At what T would sparse win by construction?** Both the saving and plow's
indexer are quadratic in the same variable, so their ratio is a constant ≈1.7–2.3
in sparsity's favour above 2048 — but the *net* only clears the serve harness's
0.3–0.5% noise floor at **8k (+3.0%)** and only becomes a lever worth building at
**16k (+10.4%)** and **32k**. Below 4k it is negative. **The construction-level
crossover is T ≈ 6–8k with plow's current indexer, and would move to T ≈ 3–4k
with vLLM's chunked-fp8 indexer design.**

**Cross-check against the campaign's closure.** `glm52-dsa-sparse-b3.md` closed
the **union-of-8** route: adjacent queries share 89% of their top-2048 (measured
1824/2048 vs a 458 random baseline), so a union over an 8-query pack is
3092/10244 = **0.30 of causal** — against the per-query ideal of 2048/10244 =
**0.20**. The union route therefore forfeits 1.5× of the available saving *before*
the kernel does anything, and the shipped B3 kernel then banked only 11% of the
0.30 (it recomputes every union row against all 64 pack rows). **That closure
stands and this table does not contradict it: 0.20 is reachable only by a
per-query membership-skipping flash, which is exactly the route B3 left open.**
Nothing here re-opens the union or the pack.

The 89% figure was measured on the campaign's random-token bench prompt and would
likely be lower on natural text — which makes the union route *worse*, not better,
and leaves the per-query number (a hard 2048) unchanged.

---

## 5. The three largest addressable components, ranked

Addressable = measured cost × (1 − 1/reference ratio), where the reference is a
rate that has actually been observed on this silicon.

| rank | component | @4k | @8k | @16k | what would have to change |
|---:|---|---:|---:|---:|---|
| **1** | **MoE grouped path** — **the 286 ms figure is now MEASURED and it is 66.8 ms; see `glm52-moe-fusion.md`.** The named mechanism (fuse 85→86→87 so the f32 `part` never reaches DRAM) was built on branch `moe-fuse` in the form aiter itself uses (`global_atomic_*` accumulate, no scatter): 2.416 GB/layer removed, plow's MoE pair 2.29× → **1.36× aiter's bytes**, served TTFT **−3.5 / −4.0 / −3.5%** at 4k/8k/16k against a 0.38–1.34% control spread. The bytes went; the time did not follow them, because the pair was never bandwidth-bound. What paid was op 87's SHAPE (k=8 strided streams → 1 contiguous, −86.9% on a probe), not its bytes. The op-graph decomposition is therefore CLOSED and the remaining aiter gap is entirely §4.3's other half — round-trip serialization in the tile structure. (Original aspiration below, kept for the record.) | **151 ms** | **286 ms** | **559 ms** | Fuse ops 85→86→87 so the f32 `part` buffer (1.61 GB written + 1.61 GB read per layer per rank at T=8192) never reaches DRAM — plow moves 2.85 GB where aiter's fused `fmoe` moves 1.31 GB, and the deficit *grows* with M because that term is the only one that does. |
| **2** | **ragged tail chunk** (measured directly: 1025 tokens costs 203 ms more than 1024) | **203 ms** | **203 ms** | **203 ms** | Stop paying a full 78-layer T-invariant pass (163 ms fixed, of which 119 ms is narrow-M GEMM dispatch) for a 128-row remainder: either cover the prompt in ONE chunk (ragged-M dispatch, or bucket widths that do not leave a remainder) or reprice `LAUNCH_ROWS` (416) which understates a GLM launch by ~3.4×. Of that 163 ms fixed pass, the largest category is narrow-M GEMM (122 ms of category intercept), so items 2 and 3 share a mechanism. |
| **3** | **attn+shared GEMM** (188 ms @8k; reference = hipBLASLt at 0.51× on the same shapes, same card, same session) | **54 ms** | **92 ms** | **185 ms** | The occ-1 deep-pipeline GEMM rewrite at `op_gemm.h:1331`. Note this term is **13.6% of the device prefill wall (11.2% of served TTFT) in the trace, against the ~6.6% the standalone pricing in `glm52-experiments.md (consolidated: GEMM rate, OPEN item 4)` implied** — that report priced only the `pick_tile` dense GEMMs at standalone rates, and in situ the whole `Gemm*`+`GemmGlu` family is ~2× more of the wall. |
| (4) | **structural attention** — dense vs top-2048, §4 | −4 ms | +50 ms | **+378 ms** | A per-query membership-skipping sparse flash (weeks, from scratch) **plus** a chunked-fp8 indexer. Below 8k it is negative. At 16k/32k it is the single largest term. |

**Totals.** At 8k the addressable pool is 286 + 203 + 92 + 50 = **631 ms of the
1005 ms gap (63%)**. At 16k it is 559 + 203 + 185 + 378 = **1325 ms of 1996 (66%)**.
At 4k it is 151 + 203 + 54 = **408 ms of a 407 ms gap** — which, taken with §3.1's
finding that the vLLM 4k cell is 266 ms above vLLM's own curve, means **4k is
already within reach and is the wrong context to plan against.**

The residual in every column is collectives (at floor), schedule overhead (91%
packed), `MlaMergeFold` (never examined — 120 ms @8k, linear in T, the price of
`ns2`) and norms.

## 6. Decode, in one paragraph

Real work is **7.69 ms/token** at ctx 1024 and **8.62 ms** at 4096. The token is
26.7–29.1 ms. Everything between those two numbers is packing: 67.5% of in-packet
CU time is gate wait, packing efficiency is 31.1%, and **2.60 ms/token (9.5%) is
wall time with no packet executing anywhere** — half the 19% the boundary model
implied, because `GLM_MOE_CORESIDENT=2` already hides the other half behind
concurrent expert slots. The largest single consumer is now `Gemv` at 2.84
ms/token (36.9% of real work), one of whose three packets is the shared-expert
down projection at N=6144 **K=256** — the same narrow-K lane defect
`PLOW_MOE_DEC_LG` just fixed elsewhere. `MoeExpertDownFp8Blk` has fallen to
3134 CU-µs/layer (−72%) and is no longer a target. The whole 1k→4k TPOT delta is
`FlashMlaDecode` growing +0.98 ms plus the waiting behind it.

---

## Appendix A — raw per-op tables at T=8192 (the reference context)

Reproduce any other context with
`python3 scripts/glm52_layer_census.py /tmp/tc/prog_8192.txt <trace>.prefill --layers 6:74`.
Span percentages sum past 100% because prefill packet spans overlap; the additive
column is busy CU-µs.

### A.1 PREFILL, one MoE layer (median of L6..L74), T=8192

| op | pkt/layer | span µs | % of layer span | busy CU-µs | gate-wait CU-µs | wait % of in-packet | ms of TTFT (×75) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `FlashMlaPrefill` | 1 | 3153.5 | 17.6% | 851574 | 547 | 0.1% | 210.1 |
| `MoeGroupDownPf` | 1 | 2822.6 | 15.8% | 848547 | 48007 | 5.4% | 209.3 |
| `MoeGroupGluPf` | 1 | 2328.0 | 13.0% | 653332 | 269 | 0.0% | 161.2 |
| `XReduceTwoShot` | 2 | 2234.7 | 12.5% | 629794 | 31210 | 4.7% | 155.4 |
| `MlaMergeFold` | 1 | 1589.8 | 8.9% | 467297 | 249 | 0.1% | 115.3 |
| `MoeCombinePf` | 1 | 1543.8 | 8.6% | 459851 | 8827 | 1.9% | 113.5 |
| `Gemm` | 3 | 1393.9 | 7.8% | 353131 | 82614 | 19.0% | 87.1 |
| `GemmWide` | 1 | 728.5 | 4.1% | 186000 | 5156 | 2.7% | 45.9 |
| `RmsNorm` | 4 | 485.7 | 2.7% | 108088 | 9435 | 8.0% | 26.7 |
| `MoeRouterTopkPf` | 1 | 333.5 | 1.9% | 96319 | 9895 | 9.3% | 23.8 |
| `GemmMed` | 2 | 800.4 | 4.5% | 88080 | 552 | 0.6% | 21.7 |
| `GemmSmall` | 2 | 382.9 | 2.1% | 70773 | 4964 | 6.6% | 17.5 |
| `Residual` | 2 | 253.2 | 1.4% | 65730 | 47426 | 41.9% | 16.2 |
| `GemmGlu` | 1 | 370.5 | 2.1% | 41642 | 310 | 0.7% | 10.3 |
| `HeadNormRope` | 2 | 132.7 | 0.7% | 31480 | 14134 | 31.0% | 7.8 |
| `MoeAlignPf` | 1 | 252.6 | 1.4% | 253 | 42 | 14.3% | 0.1 |
| **layer span** | **26** | **17906.8** | 100% | **4951890** | **263637** | **5.1%** | **1343.0** |

### A.2 PREFILL, one DENSE layer (L0..L2), T=8192

| op | pkt/layer | span µs | % of layer span | busy CU-µs | gate-wait CU-µs | wait % of in-packet | ms of TTFT (×3) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `FlashMlaPrefill` | 1 | 3043.4 | 23.8% | 820066 | 638 | 0.1% | 8.1 |
| `XReduceTwoShot` | 2 | 2684.0 | 21.0% | 761181 | 28555 | 3.6% | 7.5 |
| `MlaMergeFold` | 1 | 1600.1 | 12.5% | 468244 | 251 | 0.1% | 4.6 |
| `MoeGroupGluPf` | 1 | 1594.0 | 12.4% | 417601 | 10538 | 2.5% | 4.1 |
| `Gemm` | 2 | 1183.9 | 9.2% | 299267 | 22790 | 7.1% | 3.0 |
| `MoeGroupDownPf` | 1 | 720.2 | 5.6% | 201643 | 66077 | 24.7% | 2.0 |
| `GemmWide` | 1 | 717.3 | 5.6% | 183234 | 6741 | 3.5% | 1.8 |
| `MoeCombinePf` | 1 | 426.3 | 3.3% | 121844 | 16482 | 11.9% | 1.2 |
| `RmsNorm` | 4 | 497.0 | 3.9% | 110919 | 10285 | 8.5% | 1.1 |
| `GemmMed` | 2 | 828.5 | 6.5% | 90272 | 550 | 0.6% | 0.9 |
| `Residual` | 2 | 248.2 | 1.9% | 64934 | 52623 | 44.8% | 0.6 |
| `HeadNormRope` | 2 | 143.8 | 1.1% | 34044 | 15101 | 30.7% | 0.3 |
| `GemmSmall` | 1 | 202.0 | 1.6% | 23190 | 238 | 1.0% | 0.2 |
| `MoeAlignPf` | 1 | 27.5 | 0.2% | 27 | 19 | 40.9% | 0.0 |
| **layer span** | **22** | **12807.4** | 100% | **3596466** | **230888** | **6.0%** | **38.4** |

### A.3 DECODE, one MoE layer, ctx 4096 (`--last-dispatch` filtered, real KV)

| op | pkt/layer | span µs | % of layer span | busy CU-µs | gate-wait CU-µs | wait % of in-packet | ms of TTFT (×78) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Gemv` | 3 | 74.8 | 21.6% | 10902 | 23708 | 68.5% | 2.8 |
| `FlashMlaDecode` | 1 | 67.0 | 19.3% | 7948 | 5241 | 39.7% | 2.0 |
| `MoeExpertGluFp8Blk` | 8 | 205.9 | 59.3% | 4897 | 2888 | 37.1% | 1.3 |
| `GemvQkv` | 2 | 38.3 | 11.0% | 4699 | 20039 | 81.0% | 1.2 |
| `MoeExpertDownFp8Blk` | 8 | 126.0 | 36.3% | 3137 | 9000 | 74.2% | 0.8 |
| `MlaMergeFold` | 1 | 18.1 | 5.2% | 1045 | 5687 | 84.5% | 0.3 |
| `GemvGlu` | 1 | 19.1 | 5.5% | 519 | 252 | 32.6% | 0.1 |
| `XReduce` | 2 | 31.8 | 9.2% | 355 | 689 | 66.0% | 0.1 |
| `MoeCombine` | 1 | 7.0 | 2.0% | 69 | 445 | 86.6% | 0.0 |
| `MoeRouterTopk` | 1 | 15.7 | 4.5% | 16 | 11 | 41.5% | 0.0 |
| `AddNorm` | 2 | 11.8 | 3.4% | 12 | 86 | 87.8% | 0.0 |
| `RmsNorm` | 2 | 10.6 | 3.1% | 11 | 82 | 88.6% | 0.0 |
| `HeadNormRope` | 1 | 4.2 | 1.2% | 4 | 10 | 70.0% | 0.0 |
| **layer span** | **33** | **346.9** | 100% | **33614** | **68138** | **67.0%** | **27.1** |

## Appendix B — how to re-run this

```
# objects/blob/env exactly as in section 0; GPU lock held
export PLOW_MLA_PF_V2=1 PLOW_TRACE_RAW=/tmp/tc/tr_8192.bin
nix develop /app/plow --command target/release/plowrt amd-bench \
  --blob /workspace/assets/gfx942/glm52-tp8-final2/model.pkt \
  --hsaco /root/.claude/jobs/b09a4bcc/tmp/hsaco_glm18 \
  --checkpoint /workspace/assets/gfx942/glm52-tp8-final2/checkpoint \
  --tp 8 --steps 8 --prompt "<COMMA-SEPARATED TOKEN IDS>"   # NOT text: amd-bench parses u32
plowrt disasm <blob> --program 8192 | grep '^#' > prog_8192.txt
plowrt disasm <blob> --program 1    | grep '^#' > prog_1.txt
python3 scripts/glm52_layer_census.py prog_8192.txt tr_8192.bin.prefill --layers 6:74
python3 scripts/glm52_layer_census.py prog_1.txt    tr_8192.bin --layers 6:74 --last-dispatch
python3 scripts/glm52_decomp_tables.py ; python3 scripts/glm52_gap_attrib.py
python3 scripts/glm52_sparse_curve.py ; python3 scripts/glm52_cost_model.py
```

Three traps this run hit, recorded so the next one does not:
1. `amd-bench --prompt` takes **comma-separated u32 token ids**, not text
   (`ParseIntError` otherwise). `--tp 8` is required for the sharded blob.
2. `plowrt` must run inside `nix develop` (else `dlopen libhsa-runtime64:
   libelf.so.1` at startup) and be built `--features hsa`.
3. The decode trace from a `--prompt` run is poisoned by stale prefill records —
   use `--last-dispatch`, and sanity-check that the kept count is ~136k
   (2523 packets at their real workgroup widths), not ~590k.

---

## CORRECTIONS from the ragged-chunk build (2026-08-08, commit 5e96b0e)

Two numbers in this report are wrong and are superseded by direct measurement in
`glm52-ragged-tail-chunk.md`:

1. **The ragged tail is NOT flat in T.** This report modelled it as 203/203/203 ms. Measured
   directly: 1024→336.2 vs 1025→540.7 (+204.5 ms) and 4096→720.2 vs 4097→951.2 (+231.0 ms). The
   tail chunk's flash attends over `c0+128` KV, so it gets **dearer deeper into the prompt**.

2. **It is only 203/40/40 ms ADDRESSABLE**, not 203/203/203. Past `MAX_CHUNK=8192` the extra
   launch is **structural** — a prompt longer than the largest rung needs more than one pass no
   matter how the remainder is handled. Only the sub-`MAX_CHUNK` case can be collapsed into a
   single ragged launch.

   **The addressable pool at 8k therefore drops from 631 ms to 468 ms** of the 1005 ms gap.

   Raising `MAX_CHUNK` above 8192 is the named next lever and is **uncosted**: `act.part` doubles
   to 3.2 GB/rank, which has to be weighed against the launch it saves.

Also corrected: this report's fixed-cost estimate of ~163 ms per pass was fitted. Measured
directly, **a one-row 78-layer prefill pass costs 231 ms**, with a marginal rate of 0.130 ms/row.
So ~85% of a 128-row tail chunk is row-invariant.

**Restated published TTFT** (bench_speed's prompts are 1023 / 4101 / 8196 / 16386 tokens, so 1k
carries no tail at all): 1k 343.3 unchanged; 4k 973.4 → ~735 (**−24%**, vLLM ratio 1.72× → 1.30×);
8k 1677 → ~1637; 16k 3627 → ~3587.
