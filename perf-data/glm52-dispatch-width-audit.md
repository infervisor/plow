# GLM-5.2 decode: a COMPLETE dispatch-width audit of all 18 opcodes (2026-07-28)

Four ops have been caught being handed more workgroups than their own body can use, and each was
paid for: `XReduce` 256→12 (−1.82 ms), `HeadNormRope` 256→2/1 (−1.148), `MoeCombine` 256→12
(−0.506), and the `MlaMergeFold` VT=32 dispatch fix. Nobody had swept the rest. This is the sweep.

**Scope.** The shipping decode program: GLM-5.2 block-fp8, TP4, `--max-ctx 65536`, `--n-cu 256`,
2756 ops, zero `Gemm` (every decode matmul is a `Gemv*`). Dims: `H=6144`, `nh_l=64/4=16`,
`ql=2048`, `dk=512`, `dr=64`, `vd=256`, `E=256`, `imoe_l=2048/4=512`, `dense_inter/tp=3072`,
`PLOW_THREADS=512`, `PLOW_WAVES=8`.

**Two independent instruments, and they agree exactly.**

1. **Static**: the saturation point read out of each kernel body in `runtime/amd/`, against the
   block count `crates/devgen/src/mla.rs` assigns.
2. **Trace**: `perf-data/…/glm52_skew/tr/xrfit` (the post-`xrfit` control), classifying each
   packet's 256 workgroup records into the bimodal body-time clusters. An empty workgroup polls
   the arrival counter, takes the system-scope acquire fence and exits, and that costs ~550 ticks
   against a working `MlaMergeFold` slice's ~2700 — the split is unambiguous.

> **The blob currently on disk at `/home/lava/models/glm52_tp/glm52_tp4_64k.pkt` is STALE**: it
> still carries `HeadNormRope blocks=256` and `MoeCombine blocks=256`, i.e. it predates `da5ae76`.
> Every number below is against a **re-emitted HEAD control**. This is `xrfit`'s caveat again —
> **a sizing change is EMIT-TIME and no existing `.pkt` gets it.**

---

## 1. The ranked table

`sat` is the workgroup count past which the kernel's own work-item loop runs zero times.
Ranked by `(assigned − sat) × occurrences`, per the brief.

| # | op | pkts | asgn | **sat** | work-item rule (kernel) | idle/pkt | idle wg-pkts | idle ms/CU |
|--:|---|--:|--:|--:|---|--:|--:|--:|
| 1 | **`MlaMergeFold` (57)** | 78 | 256 | **128** | `n_batch·n_head·ceil(v/VT)`, VT=32 | **128** | **9,984** | **0.220** |
| 2 | **`GemvQkv` fusion A (22)** | 78 | 256 | **239** | blocked: `per=ceil(2624/256)=11` | **17** | **1,326** | **0.049** |
| 3 | `Embed` (6) | 1 | 256 | **1** | token-parallel `t=slice; t<ntok` | 255 | 255 | <0.005 |
| 4 | `FlashMlaDecode` (50) | 78 | 256 | 256 | `n_batch·n_tok·(nh_l/GF)·nsplit` | 0 | 0 | 0 † |
| 5 | `GemvQkv` fusion G | 78 | 256 | 256 | blocked, `Ntot=9216`, `per=36` | 0 | 0 | 0 |
| 6 | `Gemv` o_proj / shared-down / lm_head | 154 | 256 | 256 | blocked, `N≥6144` | 0 | 0 | 0 |
| 7 | `Gemv` router score | 75 | 256 | 256 | blocked, `N=256`, `per=1` | 0 | 0 | 0 ‡ |
| 8 | `GemvGlu` shared expert | 75 | 256 | 256 | blocked, `N=512`, `per=2` | 0 | 0 | 0 ‡ |
| 9 | `DenseGluFp8Blk` (47) | 3 | 256 | 384 | wave/out `ceil(3072/8)` | 0 (under) | 0 | 0 |
| 10 | `GemvFp8Blk` (44) | 3 | 256 | ≥256 | `N=6144` | 0 | 0 | 0 |
| 11 | `MoeExpertGluFp8Blk` (45) | 600 | 32 | 64 | wave/out `ceil(512/8)` | 0 (under **2×**) | 0 | 0 |
| 12 | `MoeExpertDownFp8Blk` (46) | 600 | 32 | 768 | wave/out `ceil(6144/8)` | 0 (under **24×**) | 0 | 0 |
| 13 | `XReduce` (24) | 156 | 12 | 12 | thread/elt `ceil(6144/512)` | 0 | 0 | 0 |
| 14 | `HeadNormRope` q / k (3) | 78+78 | 2 / 1 | 2 / 1 | wave/head `ceil(nh/8)` | 0 | 0 | 0 |
| 15 | `MoeCombine` (43) | 75 | 12 | 12 | thread/elt | 0 | 0 | 0 |
| 16 | `RmsNorm` (1) | 313 | 1 | **1** | **ROW-parallel** `row=slice; row<rows`, `rows=1` | 0 | 0 | 0 |
| 17 | `Residual` (4) | 156 | 1 | 2 | 8 elt/thread `ceil(6144/8/512)` | 0 (under) | 0 | 0 |
| 18 | `MoeRouterTopk` (56) / `ArgmaxFin` (18) | 76 | 1 | 1 | `if (slice) return` | 0 | 0 | 0 |
| 19 | `Argmax` (17) | 1 | 64 | 303 | thread/elt `ceil(154880/512)` | 0 (under) | 0 | 0 |
| | **TOTAL over-dispatch** | | | | | | **11,565 of 201,503 (5.7%)** | **0.269** |

† Exactly saturated **on a long-ctx blob only** — see §3.
‡ Zero idle *workgroups*, but 7/8 and 6/8 of their *waves* idle — see §2.

**Calibration of the `ms/CU` column.** The same estimator on the same trace format read **0.968**
for `HeadNormRope` (measured **−1.148 ms** on the token), **1.779** for the collective before
`xrfit` (measured **−1.820 ms**), and ~0.45 for `MoeCombine` (measured **−0.506**). It has run
about **1:1 against the token** three times, so 0.269 ms/CU is the honest expectation here — real,
and an order of magnitude smaller than the ones already banked. **The seam is nearly mined out.**

---

## 2. The finding that kills a whole class of proposals: GV_BLOCKED

The obvious reading of `d_gemv`'s inner loop is "one output column per wave, so the packet
saturates at `ceil(N/8)` workgroups" — which would make the router GEMV (`N=256` experts) an 8×
over-dispatch and the shared-expert GLU (`N=512`) a 4× one, together 31,200 wasted wg-packets and
by far the top of this table.

**That reading is wrong, because `GV_BLOCKED` defaults to 1** (`op_gemm.h`, and `PLOW_FINE`'s
dependency map assumes it). Column ownership is CONTIGUOUS, not interleaved:

```c
const unsigned gv_per = (N + nblk - 1) / nblk;      /* ceil(N / nblk) */
const unsigned gv_n0  = slice * gv_per;
for (unsigned n = gv_n0 + wave; n < gv_n1; n += PLOW_WAVES) { ... }
```

So `per` SHRINKS as `nblk` grows, and a workgroup is empty only when `slice*per ≥ N` — i.e. only
on the ceiling tail. At `N=256, nblk=256` every workgroup owns exactly one column; at `N=512` every
workgroup owns two. **No workgroup is idle. The under-fill is at the WAVE level, and narrowing the
packet does not fix it — it moves the same 256 working waves onto 1/8 of the CUs**, cutting the
aggregate read front by 8× to save a gate the workgroups were already contributing to. That is
knob-contract §6b-i's failure mode read forwards instead of backwards, and it is already
corroborated: `GLM_ROUTER_OFF_SHARED`, which narrows exactly this GEMV 256→224, measured **+0.12 ms**.

The audit therefore has to be done **per kernel**, not per opcode family. Three distinct rules live
in this program and they give answers that differ by 24×:

| rule | kernels | saturates at |
|---|---|--:|
| **blocked column runs** | `gemv_rows`, `gemv_qkv_rows`, `gemv_glu_rows` | `N` (only the ceiling tail is empty) |
| **wave-interleaved per output** | `d_moe_expert_*_fp8_blk`, `d_dense_glu_fp8_blk`, `d_headnorm_rope` | `ceil(N / 8)` |
| **thread/element** | `d_xreduce`, `d_moe_combine`, `d_residual` | `ceil(N / 512)` |

Applying the wrong one is not a slow packet, it is **dropped work**.

Two more results that fall straight out of the same discipline:

* **`RmsNorm` at 1 workgroup is NOT under-dispatch.** `d_rms_norm` is ROW-parallel
  (`for (row = slice; row < rows; row += nblk)`) and decode has `rows = 1`. Its saturation point
  *is* 1. All 313 packets are correctly sized; widening needs a different kernel (a feature split),
  which is L3 and which §6b-i already made doubtful.
* **The 1200 MoE expert packets are UNDER-dispatched, 2× and 24×**, and deliberately so —
  `GLM_MOE_CORESIDENT=1` trades saturation for 8 concurrent experts on disjoint 32-CU slices, worth
  −17.4% on the MoE block. Nothing to do.

---

## 3. `FlashMlaDecode` is saturated by a COINCIDENCE that does not hold at short ctx

`n_work = n_batch · n_tok · (n_head/GF) · nsplit`. At `max_ctx 65536`: `glm_nsplit` caps at
`fill = ceil(256 / (nh_l/GLM_MLA_GF)) = 64`, and the interpreter runs GF=4 — so
`(16/4) · 64 = 256`. Exactly the chip. The cap and the kernel cancel because **both are written in
terms of `GLM_MLA_GF = 4`.**

They do **not** cancel at `glm_gf = 2` (`max_ctx ≤ 4096`), where `n_grp` doubles to `nh_l/2` while
`nsplit` stays sized for GF=4. GLM-5.2 TP4 at `--max-ctx 4096`: `(16/2) · 16 = 128` work items on
**256** workgroups — the same latent 2× `MlaMergeFold` has, on every short-ctx blob (the ctx sweeps
emit these).

Verified by emitting both: at `--max-ctx 4096` the fix takes `op50 blocks 256 → 128`; at
`--max-ctx 65536` the blob is **byte-identical md5 `107a907f…` with and without the flash rule**,
so it cannot confound the measurement below.

**A second trap lives here.** `glm_gf(65536)` returns **8** and bakes it into `i[7]`, but
`exec_flash_mla_decode` instantiates `GF ∈ {2,4}` only (`if (gf == 2) … else <4>`) — the same
mismatch `glm_gf_prefill` already documents for the prefill twin. An emitter-side width computed
from `i[7]` **literally** would have halved the packet and **dropped half the attention work**.
`flash_mla_cus` mirrors the dispatch, not the field.

---

## 4. What was fixed

All three are **pure narrowings of a work-item map the kernel already owns**: the surviving
workgroups keep exactly the items they had, only the empty ones go away. Nothing is widened, so
knob-contract §6b-i (a consumer waiting on a max over more stragglers) does not apply in either
direction — the removed workgroups had no body to be waited on.

| helper | site | 64k blob | 4k blob |
|---|---|--:|--:|
| `mla_fold_cus` | `MlaMergeFold`, decode + prefill | 256 → **128** | 256 → **128** |
| `blocked_gemv_cus` | `GemvQkv` fusion A / G | 256 → **239** / 256 | same |
| `flash_mla_cus` | `FlashMlaDecode` / `FlashGatherDecode` | 256 (inert) | 256 → **128** |

`PLOW_GLM_WGFIT=0` restores every one of them, so the control arm comes out of the **same `plowc`
binary**. `runtime/` is **untouched**, so both arms run the **same device object** and the only
difference in the whole experiment is the packet's `blocks` field.

`mla_fold_cus` carries a **refusal** that is load-bearing: `exec_mla_merge_fold` picks VT from
`bh*8 <= nblk`, and VT determines the fold map (`NV`/`LS`/`BL`), so two VTs reassociate the `l`
sum differently. At `v=128` (Kimi-K3) the narrowed width would flip VT 32→128 — a numerics change
wearing a dispatch change's clothes. The helper detects the flip and hands the full list back.

Emitted, TP4, `--max-ctx 65536`: same 2756 ops, same 3589 edges, same 2756 counters, workgroup-
packets **201,503 → 190,193 (−11,310 = 78×128 + 78×17, exactly as derived)**.

---

## 4a. MEASURED: −0.446 ms/token (−1.77%), token-identical, reproduced on two leases

> **Provenance, because the tree moved under this measurement.** Both leases ran against the tree
> at `90ecd05`, i.e. BEFORE `9dc27bb` ("the AMD interpreter had no GF=8 arm"). Both arms used the
> **same** `i_base.elf` from that tree, so the A/B is sound and the −0.446 is attributable to the
> packet's `blocks` field alone. What `9dc27bb` changes going forward is §3: with a real GF=8 arm,
> `glm_gf(65536) = 8` now means `n_grp = 16/8 = 2` and `n_work = 2·64 = 128`, so `flash_mla_cus` is
> **no longer inert on a 64k blob** — it narrows `FlashMlaDecode` 256 → 128 there too. That is a
> THIRD 128-workgroup narrowing on top of the two measured here, and it is unmeasured. The
> `MlaMergeFold` and `GemvQkv` numbers below are untouched by that commit.

`glm52_decode … --tp 4 --sweep 1024 --steps 65 --gen 24` (median of 65 device-side steps AND the
24-id token-identity check out of one weight load). Same device object `i_base.elf` on every arm,
same `plowc` binary, `PLOW_GLM_WGFIT=0` for the controls. Controls interleaved at positions 1/3/5.
Raw: `glm52_wgfit/raw_lease1.txt`, `raw.txt`.

| lease | pos 1 ctl | pos 2 **fit** | pos 3 ctl | pos 4 **fit** | pos 5 ctl | ctl mean (sd) | fit mean | **Δ** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 `glm-wgfit` | 25.450 | **24.962** | 25.239 | **24.718** | 25.213 | 25.301 (0.130) | 24.840 | **−0.445** |
| 2 `glm-wgfit2` | 25.226 | **24.781** | 25.236 | **24.791** | 25.188 | 25.217 (0.025) | 24.786 | **−0.436** |
| **both** | | | | | | **25.259** | **24.813** | **−0.446 (−1.77%)** |

(Δ is against the control interpolated to each fit's own position, so a drift over the lease cannot
be scored as a win. The two leases agree to **0.009 ms**.)

**TOKEN IDENTITY — the gate this whole class of measurement needs.** All ten arms emitted the
**same 24 ids**:

```
264 5777 9125 1948 279 15742 315 458 3766 323 279 1196 13 1084 28995 1246 279 3162 686 387 1483 323 1128 9477
cross-rank disagreements 0        (every arm, both leases)
```

This is what separates a real narrowing from the `PLOW_XR_SHUFFLE` failure mode: on a
data-dependent-MoE model, an arm with wrong numerics collapses the router's top-k and *appears*
faster. Here the trajectory is byte-for-byte the control's, so the 0.446 ms is not routing.

**Both leases were released `rc=76`.** The flagged process is the ctx-sweep agent's
`plowrt serve --assets .../glm52_ctxsweep/tp4`, whose ranks hold **213 GB on each of cards 4–7**
and touch **36,000 B** on cards 0–3 — the TP peer-map/enumeration footprint, not compute. Two
independent leases agreeing to 0.009 ms, with a control sd of 0.025 ms on lease 2, is not what a
contended measurement looks like; but the flag is recorded rather than argued away.

**The estimator UNDER-predicted, and by more than it has before.** §1's idle-workgroup burn is
0.269 ms/CU; the token moved 0.446. On the three earlier narrowings it tracked ~1:1
(0.968 → −1.148, ~0.45 → −0.506, 1.779 → −1.820). The likely reason it is 1.66× low here is that
`MlaMergeFold` sits **on the critical path** — the flash's only consumer — so removing 128
contenders from its counter line shortens the chain as well as the poll. Treat 0.269 as a **lower**
bound for a critical-path packet.

### The §0-BENCH-legal arm CANNOT RESOLVE THIS, and that is the honest result

`vllm bench serve` → `plowrt serve` TP4 (5 programs, prefill armed), ctx 4096, conc 1, 8 prompts,
128 out, coherence before timing on every arm. Same bundles except the packets.

| arm | TPOT mean | TPOT med | ITL med | TTFT mean |
|---|--:|--:|--:|--:|
| ctl (1st) | 28.340 | 28.300 | 27.470 | 4783 |
| **fit** | **27.990** | **27.910** | **27.310** | 4811 |
| ctl (2nd) | 30.110 | 30.010 | 31.140 | 5464 |

The fixed arm is below the first control on every column, but **the two controls differ by
1.77 ms** — **4× the effect being measured** — so at a sample size that fits inside a lease this
instrument is under-powered for a 1.7% decode-internal change. Quoting a delta against the
interpolated control here would be quoting drift. **The device-side harness is the instrument that
resolves this**; the served run's contribution is the correctness gate, not the number.

**Coherence PASSED on all three served arms, with CHARACTER-IDENTICAL output:**

```
'17 * 23 = 391\n\nIn words, the result is three hundred ninety-one.'   (T=128 bucket)
'The capital of Japan is Tokyo.'                                       (T=1024 bucket, 2116-tok prompt)
first SSE chunk: {"delta":{"role":"assistant","content":"The"}}        (63f9957 artefact stays dead)
```

So the narrowing is verified identical on the **served prefill+decode** path too, not only on the
decode-only harness — which matters because `mla_fold_cus` is also wired into the prefill emitter
(where it is inert by construction, and now by observation: `op57 blocks=256` on all four prefill
buckets, `128` on the decode program only).

---

## 5. Placement: the OTHER answer, quantified — and why it is blocked

545 packets/token (19.8%) run on ≤4 of 256 CUs and `mla.rs` puts every 1-workgroup packet on
**CU 0** (12 `vec![0u32]` sites). The brief asks whether spreading them beats shrinking them.
The trace answers it concretely. CU 0, layer 0, ticks relative to the step:

```
  pc inst op                arrive  ready    end     gate    body  blocks
   2    2 GemvQkv             2264   2300    4104      36    1804     256
   3    3 RmsNorm    (q_a)    4288   4356    4744      68     388       1
   4    4 GemvQkv    (fus.G)  4808   4844    6448      36    1604     256
   5    5 HeadNormRope (q)    6588   7468    8632     880    1164     256
   6    6 RmsNorm    (kv_a)   8792   8828    9560      36     732       1   <-- dep ready since 4104
   7    7 HeadNormRope (k)    9612   9648    9972      36     324     256   <-- dep ready since 4104
   8    8 FlashMlaDecode     10032  10072   12840      40    2768     256
```

`inst 6` and `inst 7` depend on **`inst 2`** and nothing else — their gates are 36 ticks, i.e. their
inputs were ready ~4700 ticks earlier. They do not even *arrive* until the packets ahead of them
**in CU 0's stream** have retired, and the flash cannot start until they do. `inst 7` is already
fixed (`rope_cus`' `start` moved the k-rope to CU 2). `inst 6` is not.

Summed over all 78 of them (selected exactly: the `RmsNorm` packets whose `t0` is a `kv.<L>.ckv`
tensor), on the one workgroup each runs on:

```
kv_a RmsNorm packets: 78
  total BODY on its 1 workgroup: 598.7 us = 0.599 ms/token
  total GATE (dep already met):   38.1 us
```

**0.599 ms/token of serial critical path against 38 µs of actual waiting-for-data** — larger than
everything in §1 combined. It is pure stream-order.

**And it cannot be recovered by placement alone.** The static interpreter walks a **per-CU stream**
(`prog.stream_ofs[cu]`, `stream_len[cu]`): a workgroup executes every packet it is a member of, in
order. To run the kv_a norm concurrently with fusion-G it must sit on a CU that is **not in
fusion-G's block list** — and fusion-G is 256 wide with all 256 workgroups doing real work
(`Ntot=9216, per=36`). Reordering the emission does not help either: moving the norm ahead of the
q_a norm just pays the same 732 ticks earlier on the same chain.

So the unlock is **deliberately under-dispatching a hot GEMV to open a lane** (e.g. fusion-G on
`all[..250]`, `per` 36→37, ~+2.8% on its body, buying ~7.3 µs/layer). That is a real lever with a
plausible net of ~−0.5 ms, it is **not** a width fix, and it should be measured on its own arm
rather than folded into this one. **Not built here.** It is the highest-value item this audit
turned up.

---

## 6. Not done, deliberately

* **`Embed` 256 → 1** (255 idle workgroups). One packet, 0.028 ms total; the emit site is shared
  with every model and the payoff is under the noise floor. Recorded, not touched.
* **`Argmax` 64 → 303 and `Residual` 1 → 2** are UNDER-dispatch, the opposite failure. `Residual`
  widening was already measured (`GLM_SPINE_CUS`, −0.178 of the 0.577 ms it costs) and is not a
  default for §6b-i reasons.
* **The router GEMV's 7/8 idle waves and the shared GLU's 6/8.** Real, and NOT a width problem —
  see §2. A fix has to come from the kernel (columns per wave), not the emitter.
* **`GemvQkv` fusion A's load imbalance**: after narrowing, 238 workgroups own 11 columns and one
  owns 6. Sizing cannot fix a ceiling remainder; only a different split can.
* **The placement unlock of §5.** It needs a hot GEMV deliberately under-dispatched, which is not a
  width fix, and folding it into this arm would have confounded the −0.446.

---

## 7. REPRODUCTION

```bash
# objects + harness (ROCm tooling OUTSIDE nix; Rust INSIDE). runtime/ is UNCHANGED by this work,
# so both arms get the same i_base.elf.
/usr/bin/env -i PATH=/opt/rocm/bin:/usr/bin:/bin HOME=$HOME \
  bash /home/lava/models/glm52_skew/build.sh <worktree>

# blobs: control and fixed out of ONE plowc
nix develop -c bash -c 'for a in ctl fit; do
  [ $a = ctl ] && export PLOW_GLM_WGFIT=0 || unset PLOW_GLM_WGFIT
  GLM_FULL=1 PLOW_FP8=1 ./target/release/plowc --hf-dir /home/lava/models/GLM-5.2-plow \
    --emit devblob --max-ctx 65536 --n-cu 256 --num-gpus 4 --no-rope-gen \
    --out /home/lava/models/glm52_wgfit/$a.pkt; done'
#   ctl md5 184b0b4b…   fit md5 107a907f…   201,503 -> 190,193 workgroup-packets

GRAPHSTAT_V=1 nix develop -c ./target/release/examples/graphstat <pkt>   # per-packet `blocks=`

# the A/B (timing AND token identity out of one 4-minute weight load)
perf-data/harness/gpulease -n 4 glm-wgfit sg render -c 'perf-data/glm52_wgfit_ab.sh'

# the §0-BENCH-legal arm (coherence + vllm bench serve -> plowrt)
perf-data/harness/gpulease -n 4 glm-wgfit-srv sg render -c 'perf-data/glm52_wgfit_srv_ab.sh'
```

Raw: `glm52-dispatch-width-raw-lease{1,2}.txt`, `…-raw-serve.txt`. Drivers: `glm52_wgfit_ab.sh`,
`glm52_wgfit_srv_ab.sh`. Serve bundles: `/home/lava/models/glm52_wgfit_srv/{ctl,fit}`.

**`cargo test -p devgen --test tuned_tile_selection`**: the gating
`published_measurements_reach_the_compiler_and_change_its_answer` PASSES.
`the_narrow_shapes_agree_between_model_and_hardware` fails on a Gemma-26B router shape — pre-existing,
not staleness, and no GLM shape is affected. The tunedb is GEMM-only and this decode program has
**zero `Gemm`**, so it cannot move these numbers either way.
