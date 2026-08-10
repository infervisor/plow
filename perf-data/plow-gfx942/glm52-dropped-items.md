# GLM-5.2 gfx942: three DROPPED items, built and priced

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **MIXED -- read per item** — this file collects unrelated items; several are CDNA3 kernel results and several are emit-side. It carries gfx950 references item by item.

2026-08-08, branch `dropped-items` (base `daa894b`, the `worktree-glm52-bringup` head),
8x MI300X (gfx942, 304 CU), ROCm 7.2.4.

Three items that earlier rounds identified, priced and then never built. They are
independent; each is one commit.

| item | what | disposition |
|---|---|---|
| 1 | `q_a_layernorm` -> fusion G `GemvQkv` fold (`PLOW_GLM_FUSE_QNORM`) | see §1 |
| 2 | the two sibling DOWN epilogues (`PLOW_MOE_PF_EPI_SIB`) | see §2 |
| 3 | `gate_ag`'s 304 signallers (`PLOW_XR_AGG`) | see §3 |

Environment for every build and run below:

```
export PATH=/nix/var/nix/profiles/default/bin:/root/.nix-profile/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib
export ROCM_PATH=/opt/rocm-7.2.4 HIP_PATH=/opt/rocm-7.2.4 ROCM_HOME=/opt/rocm-7.2.4
export PLOW_HIPCC=/opt/rocm-7.2.4/bin/hipcc          # objects, OUTSIDE nix
export PLOW_MLA_PF_V2=1                              # serve
```

Canonical object recipe (the control everywhere below):

```
env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 PLOW_MOE_PF_EPI=1 JOBS=12 \
    bash scripts/build_gfx942.sh <dir>
```

Canonical blob recipe, and it is verified rather than assumed — the control blob emitted
from this branch with all three knobs unset is **`cmp`-identical to the shipped
`/workspace/assets/gfx942/glm52-tp8-final2/model.pkt`**:

```
GLM_FULL=1 PLOW_MLA_PREFILL=full GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1 \
PLOW_GLM_DSA=0 PLOW_GLM_FUSE_B1=1 PLOW_GLM_GEMV_WG=152 PLOW_MLA_PF_V2=1 PLOW_GLM_PF_NS=2 \
PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_FUSE_SEAM=1 \
  plowc --emit devblob --hf-dir /workspace/models/GLM-5.2-FP8 --gpu MI300X --arch gfx942 \
        --num-gpus 8 --max-ctx 73728 --out <asset>/model.pkt
```

## 0. Default posture, checked in both directions

Every knob defaults OFF and every default artefact is unchanged. This was **rebuilt in the
same output directory** (hipcc embeds the source path, so a different directory is not a
comparison) and diffed:

| artefact | control vs this branch, knobs unset |
|---|---|
| all 22 prefill/flash objects | **byte-identical** (`diff -rq`) |
| the 6 decode objects | **instruction-for-instruction identical**; the ONLY ELF delta is `d_gemv_qkvg`'s mangled name, which gained two dead default parameters (`...jjjjjjjjS_` -> `...jjjjjjjjS_S1_f`). Disassembly with that symbol masked: **0 differing lines** in all six. Same file size, `VGPR 108 / AGPR 0 / LDS 30776 / spill 0` unchanged. |
| the emitted decode program | **`cmp`-identical blob** to shipped `final2` |

---

## 1. `q_a_layernorm` -> fusion G `GemvQkv` (`PLOW_GLM_FUSE_QNORM`)

### 1.1 What it was

`perf-data/plow-gfx942/glm52-decode-packet-folds.md` §7 named this **the largest bounded
decode item remaining, ~-0.9 ms**, and did not build it. Its §1b traces the GLM decode
critical path and prices the window:

```
  80.0-101.9  GemvQkv  (fusion A)         b=146
 108.0-112.6  RmsNorm  q_a_layernorm      b=1     [12.2 us window for a 4.6 us body]
 114.1-132.2  GemvQkv  (fusion G)         b=149
```

12.2 us is the biggest packet-boundary window left on the chain. The census's own summary is
why a boundary is worth more than work: **63.7% of in-packet CU time is gate-wait**, and the
serial packet-boundary dead time is **78.2 us of a 355 us layer**. A b=1 packet waits for the
slowest of 149 workgroups, then runs alone on one CU while 303 idle, and only then opens the
next gate.

### 1.2 What was built

The norm is computed into the LDS copy of `x` that fusion G already stages, and the
`RmsNorm` packet is not emitted. This is `d_gemv_t`'s existing `norm == 2` mechanism
(`gemv_norm_lds` in `op_gemm.h`) applied to the fused three-stream sweep, as a **runtime
branch inside one body** rather than a second template instantiation — the GF=8 lesson
(`op_attention.h`: a second instantiation grew the decode object 15.6% and cost +32% *with
the registers fitting*, because every packet in a persistent megakernel shares one
instruction stream).

**The fan-out law was checked first, and it is the reason this is not the Gemma null.**
`op_gemm.h`'s `norm == 2` note states it outright: folding a norm into its consumer costs
(N-1) extra reductions, and the same fold LOST on Gemma (22.4 -> 24.4 ms) where one norm fed
five consumers. Here `n.qlat` has exactly **one** consumer — but only because A/G fusion made
it one. So `fuse_g` is a precondition, not an optimisation, and an armed DSA gate (whose
lightning-indexer `q_idx` projection is a second consumer of `n.qlat`) **refuses at emit**.

**Slot budget, and the operand-collision check explicitly.** Op 22 `GemvQkv` spends
`t[0..6]` on its three streams and has **never carried a `t[7]`**:

| need | slot | why it is free |
|---|---|---|
| gamma (`q_a_layernorm.weight`) | `t[7]` | op 108 `GemvQkvg`'s `t7` is `g_out` on a **different opcode**; ops 114/115 spend `i5/i6/i7` on scale rows and leave `t7` alone |
| eps | `f[0]` | op 22 had no `f` slots |
| the raw pre-norm row | `t[1]` | same slot, no cost — `t[1]` becomes `act.qlr` instead of `act.qlat` |

Presence of `t[7]` IS the feature: there is **no bitfield to decode**. This is deliberately
NOT the hazard the brief warns about — dense op 51 `FlashMlaPrefill`'s `i[6]` (low 8 bits =
causal KV-split `ns`, bit 8 = `W_ofold`; the sparse GATHER arm reuses `i[6]` whole as `cap`,
discriminated by `t7`) is a **prefill** opcode. Op 22 is decode-only and the two never meet.
No new bitfield was created anywhere in this branch.

Table row (`crates/packet/src/slots.rs`) and doc-spec code-spans (`crates/packet/src/dev.rs`)
both updated; `packet`'s spec-discipline test `table_matches_doc_comments` passes.

**Arm-check chain**, because a folded blob on a pre-fold object is silently wrong — it
ignores `t[7]`, stages `t[1]` verbatim, and projects an **UNNORMED q_a row**: no trap, no
NaN, a fluent wrong answer. Unlike `PLOW_GLM_FUSE_ROPE`, whose marker is a vintage stamp on
an unconditional runtime branch, this fold is a **build axis**, so the marker is genuinely
load-bearing:

* `-DPLOW_GLM_FUSE_QNORM=1` on the decode rows of `scripts/build_gfx942.sh`
* `interp.hip` exports `plow_glm_fuse_qnorm_arm` **only under that define** (verified: 2
  symbol hits in `hsaco_qnorm/interp_decode_gq.elf`, 0 in `hsaco_ctl`)
* `manifest.rs` derives feature `glm_fuse_qnorm` from any op-22 `t[7]` -> `requires`
  (verified in the emitted `build.json`: the qnorm blob's gfx942 requires list gains
  `PLOW_GLM_FUSE_QNORM=1`, the control's does not)
* `plowrt`'s `DECODE_ARM_MARKERS` refuses the pairing at load
* the kernel **traps** rather than falling back if the shape is not foldable — there is no
  packet to fall back to. `glm_fuse_qnorm()` in `mla.rs` refuses the same shapes at emit.

### 1.3 Emit evidence

`plowrt disasm --program 1` on the two blobs:

| | control | `PLOW_GLM_FUSE_QNORM=1` |
|---|--:|--:|
| decode packets / token | 2523 | **2445** (-78, one per layer) |
| `RmsNorm` packets | 158 | **80** |
| `GemvQkv` packets | 156 | 156 |

and the fold reads as intended (layer 0, fusion G):

```
GemvQkv b=149  q_out<-act.qa x<-act.qlr  W_q<-...q_absorb  k_out<-act.qrr W_k<-...q_rope
               gamma<-model.layers.0.self_attn.q_a_layernorm.weight
               | M=1 Nq=4096 K=2048 Nk=512 Nv=0 eps=0.00001
```

against the control's `x<-act.qlat ... gamma<-—  eps=0`. GLM-5.2's `q_lora_rank` is 2048,
which clears both kernel preconditions (`2048 % 8 == 0`, `2048 <= RN_REG*PLOW_THREADS =
8192`, `2048 + 16 <= GM_LDS_HALVES`).

### 1.4 ISA evidence — the SHIPPED decode object, not a probe TU

`Variant::detect` (`crates/plowrt/src/exec/amd.rs:276`) picks by scanning for
`DevOp::GemvFp8`; GLM's fp8 is BLOCK-scaled (`GemvFp8Blk`) and its MLA is bf16, so **GLM
decode opens `interp_decode_gq.elf`** — that is the object audited here, and it is the one
the run loads.

`_Z11d_gemv_qkvg...`, `llvm-objdump --mcpu=gfx942`:

| | control | qnorm arm |
|---|--:|--:|
| instructions | 2112 | 2585 |
| `v_rsq_f32` | 0 | **1** (the single `rsqrtf(ss/K + eps)`) |
| `global_load` | 6 | 8 (+2 = the gamma row at `RN_VEC = RN_REG/8 = 2` 16-byte loads) |
| `ds_read` | 33 | 37 |
| `ds_write` | 6 | 9 (the normalized write-back) |
| `s_barrier` | 1 | 4 (`block_sum`'s two + the write-back barrier) |

Whole-object cost and the register-headroom claim, **verified rather than assumed**:

| `interp_decode_gq.elf` | control | qnorm arm |
|---|--:|--:|
| size | 528,136 B | 530,744 B (**+0.49%**) |
| VGPR / AGPR | 108 / 0 | **108 / 0 (unchanged)** |
| LDS | 30,776 B | 30,776 B (unchanged) |
| VGPR spill | 0 | 0 |

The 108-of-256 headroom the census cited is real and the fold does not consume any of it.
---

## 2. The two sibling DOWN epilogues (`PLOW_MOE_PF_EPI_SIB`)

### 2.1 What it was

`PLOW_MOE_PF_EPI` (merged, in the shipped object recipe) hoisted the k- and n-invariant
`row_partidx`/`row_gate` pair out of `d_moe_group_pf_t`'s DOWN epilogue: 128 loads and 131
full `vmcnt(0)` drains per output tile collapsed to 0 loads and 3 drains via a tile-head load
plus `ds_bpermute`, worth **-17.9% busy** on that kernel
(`perf-data/plow-gfx942/glm52-experiments.md (consolidated: PLOW_MOE_PF_EPI)`). That report named two sibling sites
carrying the identical pattern and deliberately left them alone. This is those two.

### 2.2 What was built

`MPF_BM == MPF4_BM == PLOW_WAVE == 64`, so one wave holds the whole row block one row per
lane. Lane L loads row `rowbase + L`'s metadata at the tile head — coalesced, in flight
together, latency covered by the k-loop — and the epilogue reads row `rr` back out of lane
`rr` with `ds_bpermute_b32`. 2 loop-carried VGPRs, zero LDS, zero barriers.

**`d_moe_group_gemma_pf_t` takes THREE, not two.** Its `W8A8` arm reads a third k- and
n-invariant per-row quantity, `ascale[rowbase + rr]`. Hoisting only two would leave one
dependent global load per output element and therefore leave the per-element
`s_waitcnt vmcnt(0)` drain exactly where it is — the collapse would not happen. All three
ride the same pre-EXEC window.

A **separate flag** from `PLOW_MOE_PF_EPI` so the GLM canonical recipe (which sets
`PLOW_MOE_PF_EPI=1`) is unperturbed by a change to kernels GLM never dispatches.

### 2.3 THE EXEC TRAP — verified in the ISA, not argued

`ds_bpermute_b32` honours EXEC on the READ side: lane L wants row `rr(L)`, held by lane
`rr(L)`, whose own activity is decided by row `rr(rr(L))` — a *different* row. So neither the
pad-row test nor the `nn < N` tail guard may be live when the bpermutes issue. The tail guard
is lane-varying here (it goes through `mfma_acc_n(lane)`), so the shipped `if (nn >= N)
continue` **does** mask EXEC — it had to become a predicate.

From `llvm-objdump --mcpu=gfx942` on the shipped `interp_prefill_gq.elf`,
`_Z25d_moe_group_down_gemma_pf...`:

```
s_or_b64 exec, exec, s[22:23]        ; EXEC restored to the full wave
s_or_b64 exec, exec, s[20:21]
ds_bpermute_b32 v51, v177, v97       ; row_partidx  <- FULL EXEC
ds_bpermute_b32 v64, v177, v213      ; row_gate     <- FULL EXEC
...
s_waitcnt lgkmcnt(1)                 ; PARTIAL - the second bpermute is still in flight
v_cmp_ne_u32_e64 s[4:5], -1, v51     ; the PLOW_EXPERT_UNUSED test
v_cmp_ge_u32_e64 s[0:1], v54, v12    ; the n-tail test
s_and_b64 s[4:5], s[2:3], s[4:5]     ; ... ANDed as a MASK
s_and_saveexec_b64 s[2:3], s[4:5]    ; only NOW does EXEC narrow
```

Both bpermutes sit **above** the `s_and_saveexec_b64`, at the full wave EXEC the epilogue is
entered with. The intrinsic is `convergent`, so LLVM may not sink them back down.

### 2.4 ISA counts, SHIPPED objects

`interp_prefill_gq.elf` (bf16 ops 75/76) and `interp_prefill_fp8_mla_moe.elf` (w8a8 ops
81/82 — they compile only under `PLOW_FP8`):

| symbol | | control | `_SIB=1` |
|---|---|--:|--:|
| `d_moe_group_down_gemma_pf` | `flat_load` | 69 | **5** |
| | full `vmcnt(0)` drains | 74 | **10** |
| | `ds_bpermute_b32` | 0 | **64** (2 x 32 elements) |
| | instructions | 984 | 879 |
| `d_moe_group_down_gemma_pf_w8a8` | `flat_load` | 134 | **38** |
| | full `vmcnt(0)` drains | 80 | **49** |
| | `ds_bpermute_b32` | 0 | **96** (3 x 32 elements) |
| | instructions | 1572 | 1440 |
| `d_moe_group_glu_gemma_pf` (both) | everything | — | **byte-unchanged** |

The GLU arms being untouched is the check that the hoist is DOWN-only, as it must be
(`row_partidx = row_gate = nullptr` on the GLU calls). The w8a8 residue of 38 `flat_load` is
`wsc[nn]`, the per-output-CHANNEL weight scale — n-varying, so not this transform's to take.

### 2.5 The a4w4 twin: NOT EXERCISABLE, and stated plainly

`d_moe_group_pf_a4w4` is behind `PLOW_MOE_PF_A4W4`, which `op_moe.h` itself `#error`s
without `PLOW_HAS_MX_MMA`: **CDNA3 has no fp4 matrix core**, so the arm is not compiled into
any object on this box and there is no CDNA4 hardware here to run it on.

It is not even compile-checkable here, and that is a **pre-existing** defect rather than
anything this branch did. `--offload-arch=gfx950` on `interp.hip` produces **18 errors, with
and without `PLOW_MOE_PF_A4W4`, and with the base `op_moe.h` as well as this one**:
`mpf_fp8x4_to_bf16_h` / `mpf_fp8v16_to_bf16_h` are defined under `#if PLOW_CDNA4`-conditional
guards but `MPF_FETCH_A`/`MPF_FETCH_B` reference them from the `MPF_W_HALF` path that CDNA4
also takes. Identical error count on both arms, so this change introduces nothing. Flagged,
not fixed — out of scope.

The a4w4 hoist is therefore **written, symmetric with the other two, and reviewed by
inspection only**. It should not be trusted until a gfx950 build of that axis exists.
---

## 3. `gate_ag`'s 304 signallers (`PLOW_XR_AGG`)

### 3.1 What it was, and what it is NOT

`gate_ag` is `i[4]` of `XReduceTwoShot` — the two-shot all-reduce's second rendezvous. Unlike
`gate_rs` it needs `nranks*nblk` arrivals, because PHASE 1 writes the owned slice
COLLABORATIVELY across all `nblk` workgroups and `__syncthreads()` is workgroup-wide (the
asymmetry is a live-race fix, documented in `op_collective.h`, and it stands). As built,
each of `nblk` workgroups issues `nranks` SYSTEM-scope returning RMWs on ONE 128 B line per
peer: at nblk=304, tp=8 that is **2432 remote atomics per rank per collective**.

Measured (`glm52-collective-tuning-mi300x.md` §6.2): a 1-signaller N-way gate round costs
**8.2 us**, this 304-signaller one costs **51.8 us** — 6.3x. The mechanism is confirmed
independently by the atomics probe in `glm52-packet-protocol-xcd.md` §2: a returning atomic
is **392 ns** on a private line and **2770 ns at 304 claimants on one line (7.1x)**.

**It is PREFILL-ONLY, and that is the size of the prize, not a caveat.**
`XReduceTwoShot` appears **156x in every prefill program and 0x in the decode program**
(decode's collective is the one-shot `d_xreduce_mega` with a single `i[3]` gate). So this is
~8 ms of TTFT per launch — ~2.3% at 1k, ~0.5% at 8k — and **exactly 0.0% of TPOT**. It was
never going to move decode and it is not measured against decode below.

### 3.2 What was built

`PLOW_CTR_STRIDE` is 32 words (128 B) per counter and only word 0 is ever used, so word 1 of
this gate's own line is a free device-local aggregation counter that the host already zeroes
every step. Each workgroup keeps its own SYSTEM-scope RELEASE fence — the load-bearing half,
it is what pushes that workgroup's PHASE 1 stores past its XCD L2 — then does an AGENT-scope
RMW on word 1. The workgroup that closes the count acquires and issues the `nranks` remote
signals for the whole device. Remote atomics per rank per collective: **2432 -> 8**.

**The count is deliberately unchanged.** The closer adds `nblk` per peer rather than 1, so
word 0 still lands on exactly `nranks*nblk`. The waiter's threshold, the deadline arm, and
`plowrt`'s host-side `audit_xctr` expectation (`gate_expectations`: `n_gpu * blocks`) are all
untouched — an object with this axis and one without produce the same final counter state.
No blob change, no emitter change, no arm marker: an object built without it is
correct-just-slower, exactly like `PLOW_MLA_PF_SV`.

Ordering, because it is the whole risk: A's stores happen-before A's system release fence,
which happens-before B's agent acquire (release sequence on word 1, same device, and a system
release is at least an agent release), which happens-before B's peer signals, which
happen-before the peer's `xctr_acquire()`. Transitive. BIT-IDENTICAL: no value is touched.

### 3.3 ISA evidence, SHIPPED `interp_prefill_fp8_mla_moe.elf`

Control, per workgroup (thread 0), the peer loop:

```
buffer_wbl2 sc0 sc1
flat_atomic_add v[2:3], v43 sc1        ; v43 = 1, SYSTEM scope, x nranks, x nblk workgroups
```

`PLOW_XR_AGG=1`:

```
buffer_wbl2 sc0 sc1                              ; the same per-workgroup release, kept
flat_atomic_add v2, v[2:3], v43 offset:4 sc0     ; WORD 1 (offset:4), RETURNING, AGENT (no sc1)
s_waitcnt vmcnt(0) lgkmcnt(0)
v_add_u32_e32 v2, 1, v2
v_cmp_eq_u32_e32 vcc, v2, v4                     ; prev+1 == nblk
s_and_saveexec_b64 s[16:17], vcc                 ; only the closer proceeds
s_cbranch_execz 24
buffer_inv sc1                                   ; the agent acquire
  ... peer loop:
  buffer_wbl2 sc0 sc1
  flat_atomic_add v[2:3], v4 sc1                 ; v4 = nblk, SYSTEM scope, x nranks ONLY
```

and the waiter is untouched: `v_mul_lo_u32 v2, s47, v4` is still `nranks*nblk`.

Object cost: `interp_prefill_fp8_mla_moe.elf` 506,632 -> 506,760 B (+0.03%).
---

## 4. The instrument, and what it cost to get a clean one

**Four other agents were benchmarking this box throughout.** At least two of them do not take
the `/tmp/plow_gpu.lock` and one ran `bench_speed.sh` on **port 8196 concurrently with this
battery**. Three concrete failures happened and all three are the classes this directory's
README already names:

1. **A peer's `plowrt` OOM'd mine.** `amd-bench` died with
   `hsa_amd_memory_pool_allocate(1207959552) -> HSA_STATUS_ERROR_OUT_OF_RESOURCES` on ~15
   consecutive attempts. A TP8 GLM rank binds ~93 GiB; two of them do not fit.
2. **A stale `bench_speed.sh` of my own owned :8196 while a later arm measured it.** The
   cause was mine and it is worth recording: a `trap ... TERM` whose handler does NOT call
   `exit` releases the lock and then **keeps benching**. The first battery ran for 90 minutes
   after I thought I had stopped it, spawning servers that a later arm then measured. Every
   number from that window was discarded. The handler now ends with `exit`.
3. **A `kill -9` on a battery orphans its lock**, because SIGKILL cannot run a trap. Reclaimed
   explicitly (mtime-matched to my own `LOCK ACQUIRED` line) rather than left for an hour.

So the battery samples contention **every 3 s for the whole of every arm** and reports it with
the number, and an arm is thrown away and re-run if a peer answered the port at any point:

* `foreign=k/n` — samples in which an exact-named `plowrt` from another worktree existed.
  (`pgrep -x plowrt`, never `pgrep -f "plowrt serve"`, which matches its own launcher.)
* `port_alien=k/n` — samples in which the process LISTENING on the bench port was not mine.
  **This is the hard reject; it is zero on every arm reported below.**

`foreign` was relaxed to <=3 samples after measurement, not for convenience: two back-to-back
control arms that each caught ONE 3-second foreign tick of ~33 read **26.85/29.02 and
26.83/29.03**, i.e. 0.07% apart. A peer merely existing for one tick does not move this
instrument. A peer answering the port does, and that stayed a hard reject.

Bench port :8195, gate port :8196, `NPROMPT=4 CONCS=1`, strict alternation, one arm per serve,
`plowrt serve` killed with **SIGTERM** (a `kill -9` leaves the persistent megakernel resident).

### 4.1 ITEM 1 — served A/B, 3 interleaved rounds

`IN_LENS="1024 4096" OUTLEN=96`. **TPOT ms/token:**

| round | ctl @1k | qnorm @1k | ctl @4k | qnorm @4k | contention (ctl / qnorm) |
|---|--:|--:|--:|--:|---|
| 1 | 26.77 | 26.44 | 28.98 | 28.67 | 1/32, 0/32 · alien 0 |
| 2 | 26.78 | 26.39 | 29.01 | 28.62 | 1/32, 1/32 · alien 0 |
| 3 | 26.74 | 26.44 | 29.00 | 28.66 | 1/32, 1/32 · alien 0 |
| **mean** | **26.763** | **26.423** | **28.997** | **28.650** | |
| own spread | 0.04 | 0.05 | 0.03 | 0.05 | |
| **delta** | **-0.34 ms (-1.27%)** | | **-0.35 ms (-1.20%)** | | |

**The noise floor, and why the win clears it.** The control's own round-to-round spread is
**0.04 ms @1k and 0.03 @4k**; the arm's is 0.05 / 0.05. **The arms do not overlap at either
context**: the WORST qnorm @1k (26.44) beats the BEST ctl (26.74) by 0.30 ms = **6x the arm's
own spread**, and @4k the worst qnorm (28.67) beats the best ctl (28.98) by 0.31 ms = **6x**.
The control also reproduces the campaign's published shipping number for this asset
(26.72 @1k / 29.06 @4k) to within 0.05 ms, so the delta is against the shipped number and not
against a re-derived one.

**TTFT is unmoved** — 340.5 -> 341.8 @1k, 964.7 -> 967.9 @4k, both inside the qnorm arm's own
3.5 / 7.3 ms TTFT spread. Correct: the fold is decode-only.

**Honest reading of the -0.9 ms projection.** The census projected ~-0.9 ms from a 12.2 us
window x 78 layers. Measured is **-0.34**. The projection assumed the whole window is
recoverable; it is not. The rope fold's own record measured the residual gate round trip at
**6.3 us of its 10.6 us window** — the part no fold reaches — and this fold pays, on top, for
149 workgroups each recomputing a 2048-element reduction. ~6 us x 78 layers = 0.47 ms of
window minus that redundancy lands where the measurement is. **The item is real and it is
about a third of what it was priced at.**

### 4.2 ITEM 1 — serve gate, verbatim, both arms

`plowrt serve`, `temperature 0`, `max_tokens 160`. **Character-identical on both arms**:

```
Q1  What is the capital of France? Answer in one short sentence.
    -> The capital of France is Paris.
Q2  What is 17 * 23? Answer with the number and then spell it out.
    -> 391 - three hundred ninety-one
Q3  What is the chemical symbol for gold, and name one common use for it?
    -> The chemical symbol for gold is **Au**.
```

plus `bench_speed.sh`'s own Paris gate: **PASS on all 6 bench serves**.

### 4.3 ITEM 3 — served A/B, 3 interleaved rounds

`IN_LENS="1024 4096 8192" OUTLEN=8` — TTFT is the metric; the objects differ only in the
prefill collective and the blob is the SAME FILE (the xragg asset's `model.pkt` is a symlink
to the control's).

**TTFT ms:**

| | ctl r1/r2/r3 | xragg r1/r2/r3 | ctl mean | xragg mean | delta | worst-arm vs best-ctl |
|---|---|---|--:|--:|--:|---|
| 1k | 344.1 / 340.5 / 340.1 | 332.7 / 335.8 / 335.6 | 341.6 | **334.7** | **-6.9 (-2.01%)** | 335.8 < 340.1 ✓ |
| 4k | 971.0 / 964.5 / 964.0 | 945.4 / 951.6 / 952.2 | 966.5 | **949.7** | **-16.8 (-1.73%)** | 952.2 < 964.0 ✓ |
| 8k | 1674.5 / 1671.3 / 1671.3 | 1654.4 / 1658.5 / 1660.0 | 1672.4 | **1657.6** | **-14.7 (-0.88%)** | 1660.0 < 1671.3 ✓ |

Control's own spread is 4.0 / 7.0 / 3.2 ms; **the arms are disjoint at all three contexts**.
Contention `foreign` 0-1 of ~29 samples, `port_alien` **0/29 on every arm**.

**TPOT is unmoved, and that is the prediction, not a caveat:**

| | ctl | xragg |
|---|--:|--:|
| TPOT @1k | 26.720 (spread 0.16) | 26.733 (spread 0.08) |
| TPOT @8k | 29.290 (spread 0.09) | 29.310 (spread 0.03) |

`XReduceTwoShot` is emitted 0 times in the decode program, so a change to its rendezvous
CANNOT move TPOT, and it does not. The measurement agrees with the blob.

**Against the prediction.** The report priced the whole `gate_ag` cost at ~8 ms/launch =
~2.3% of TTFT @1k and ~0.5% @8k. Measured **-2.01% @1k, -1.73% @4k, -0.88% @8k** — the right
magnitude, the right sign, and the right SHAPE (the share falls with context because the
constant 156-collective cost is amortised over more prefill work). Its own report expected
this to sit inside the box's DVFS noise; interleaved, one arm per serve, it does not.

Serve gate on the xragg arm: **character-identical** to both item-1 arms (same three answers).
### 4.4 ITEM 2 — the CPU-oracle gate, and what it does and does not prove

Neither sibling is on GLM's hot path, so this is gated on the model that uses each path, not
on GLM. For the Gemma-4 twin that is `runtime/tests/moe_gemma_gfx950_test.c`, the CPU-oracle
golden for ops 61-77 and 81/82 — **written from `runtime/nvidia/op_moe.cuh`'s semantics, not
from `runtime/amd/op_moe.h`**, so it validates the kernel against the op's definition rather
than against itself. It runs the real 26B-A4B shapes (H 2816, I_moe 704, 128 experts, top-8).

Built both ways from this branch (`test_kernels.hip`, gfx942, the same `op_moe.h` the shipped
objects compile):

```
[oracle-ctl] 25/25 checks passed
[oracle-sib] 25/25 checks passed
```

including, on the arm:

```
op75 GROUP_GLU pad rows            PASS  (0 of 512 pad rows non-zero)
op76 GROUP_DOWN_GEMMA_PF           PASS  (worst rel 1.882e-04, tol 5e-03)
op82 GROUP_DOWN_GEMMA_PF_W8A8      PASS  (worst rel 1.345e-03, tol 5e-03)
```

The pad-row check is the one that matters most for the EXEC trap: **every** pad row is checked
exhaustively (not sampled), and a bpermute issued under a narrowed EXEC is exactly what would
break them.

**What this is NOT.** It is not a bit-identity proof, and the reason is a property of the
HARNESS, established by running the SAME `test_kernels_ctl.elf` twice:

```
run 1   op76 worst rel 1.897e-04 at 22782      op81 worst rel 3.878e-03 at 13779
run 2   op76 worst rel 1.899e-04 at   587      op81 worst rel 4.406e-03 at 16462
```

Despite a fixed PRNG seed (`0xC0FFEE`), the sampled row set and therefore the reported extremum
move run to run. So the small ctl-vs-sib differences in those columns — which also appear on
`op75`/`op81`, kernels this change does not touch at all — are **not attributable to the
change**, and the honest verdict is **PASS/PASS against an independent oracle**, not `cmp`.

**Not exercisable end to end on this box.** Gemma-4-26B-A4B is the only model that dispatches
these ops, and this directory's README records it as **numerically wrong on gfx942 for an
unresolved reason that is NOT in these kernels** (the same oracle passes 25/25, and the
interpreter-driven twin passes 13/13). A served A/B on a model that is already wrong cannot
gate this change, so none was run and none is claimed. The a4w4 twin is not exercisable at
all — see §2.5.

**No GLM measurement is offered and none should be**: GLM-5.2 never dispatches ops 75/76 or
81/82, so a GLM A/B on this flag would measure nothing and could only manufacture a number.
---

## 5. Disposition

| item | verdict |
|---|---|
| **1** `PLOW_GLM_FUSE_QNORM` | **LANDED.** TPOT **-0.34 ms @1k (-1.27%) and -0.35 ms @4k (-1.20%)** against a 0.03-0.04 ms control spread, arms disjoint at 6x the arm's own spread, gates character-identical, packets/token 2523 -> 2445. **About a third of the -0.9 ms the census projected** — the projection counted the gate round trip, which no fold reaches. |
| **2** `PLOW_MOE_PF_EPI_SIB` | **LANDED for the Gemma-4 twin as a correctness-preserving cleanup with a proven mechanism**: the DOWN epilogue's serialized round trips collapse (flat_load 69->5 and 134->38, full `vmcnt(0)` drains 74->10 and 80->49), the EXEC discipline is verified in the ISA, and an independent CPU oracle passes 25/25 on both arms. **NOT MEASURED end to end and no speedup is claimed** — the only model that dispatches these ops is documented broken on this box. **The a4w4 twin is NOT EXERCISABLE** (CDNA4-only, and its build axis does not compile on this ROCm before or after the change). |
| **3** `PLOW_XR_AGG` | **LANDED.** TTFT **-6.9 ms @1k (-2.01%), -16.8 @4k (-1.73%), -14.7 @8k (-0.88%)**, arms disjoint at all three contexts against control spreads of 4.0/7.0/3.2 ms. **TPOT unmoved (26.720 -> 26.733 @1k, 29.290 -> 29.310 @8k)**, which is the prediction and not a caveat: `XReduceTwoShot` is emitted zero times in the decode program. Objects-only, bit-identical, no blob change. |

Every knob defaults OFF. Item 1 needs BOTH halves (`-DPLOW_GLM_FUSE_QNORM=1` on the objects
and `PLOW_GLM_FUSE_QNORM=1` at emit) and `plowrt` refuses the mismatch. Items 2 and 3 are
objects-only and an object built without them is correct-just-slower.

Proposing any of these default-on is left to consolidation. Item 1 is the strongest candidate
on the merits (bit-exact by construction, +0.49% decode object, registers unchanged), but its
`cmp`-level numerics check is still outstanding — see §6.

## 6. What is still open

* **Item 1's bit-identity run was not obtained.** `PLOW_GLM_FUSE_ROPE` cleared `cmp` on all
  nine dumped logit rows and this fold should too — it is the `gemv_norm_lds` construction,
  which is bit-exact by the same argument (`d_rmsnorm`'s `fits` path replicated
  element-for-element, then the ordinary un-normed hot loop over LDS bytes identical to the
  HBM bytes the deleted packet wrote). It is an ARGUMENT until it is a `cmp`. Two things
  blocked it: `amd-bench --dump-logits` only writes when `--prompt` is given (the dump lives
  inside the prompt branch of `main.rs`; without it the run takes the timing-only path), and
  every attempt after that fix queued behind a peer's TP8 job. Rerun:
  `plowrt amd-bench --blob <asset>/model.pkt --hsaco <objdir> --checkpoint <ckpt> --tp 8
  --steps 8 --prompt "$(cat prompt512.txt)" --dump-logits <dir>` on both arms and `cmp`.
  The serve gate is character-identical on both arms, which is the acceptance test, but it is
  weaker than `cmp` and it is not being represented as more.
* **Item 2's a4w4 twin** is written and unverified — see §2.5. It needs a gfx950 build of that
  axis, which needs the pre-existing `#if PLOW_CDNA4` breakage fixed first.
* **A pre-existing defect found on the way, not fixed:** `runtime/amd/op_moe.h` does not
  compile for `--offload-arch=gfx950` at all on ROCm 7.2.4. `mpf_fp8x4_to_bf16_h` /
  `mpf_fp8v16_to_bf16_h` are defined under a `PLOW_CDNA4`-conditional guard but
  `MPF_FETCH_A`/`MPF_FETCH_B` reference them from the `MPF_W_HALF` path CDNA4 also takes:
  18 errors, identical with and without any change from this branch. Reported, not touched.
* **`cargo test -p devgen --test tuned_tile_selection` fails**, as it does for any edit to
  `runtime/amd/*.h`: the preprocessed build digest moves and the tuned-GEMM records go stale.
  One of its two failures is already present on the base branch. Every other test in `packet`,
  `devgen` and `plowrt` passes, including `packet`'s `table_matches_doc_comments`
  spec-discipline test, which is what the new op-22 `t[7]`/`f[0]` slots had to satisfy.

## 7. Reproduction

```bash
export PATH=/nix/var/nix/profiles/default/bin:/root/.nix-profile/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib
export ROCM_PATH=/opt/rocm-7.2.4 HIP_PATH=/opt/rocm-7.2.4 ROCM_HOME=/opt/rocm-7.2.4
export PLOW_HIPCC=/opt/rocm-7.2.4/bin/hipcc

# objects (OUTSIDE nix). control = the canonical recipe; each arm adds ONE flag.
CANON="PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 PLOW_MOE_PF_EPI=1"
env $CANON JOBS=12                            bash scripts/build_gfx942.sh <dir>/hsaco_ctl
env $CANON PLOW_GLM_FUSE_QNORM=1  JOBS=12     bash scripts/build_gfx942.sh <dir>/hsaco_qnorm
env $CANON PLOW_MOE_PF_EPI_SIB=1  JOBS=12     bash scripts/build_gfx942.sh <dir>/hsaco_sib
env $CANON PLOW_XR_AGG=1          JOBS=12     bash scripts/build_gfx942.sh <dir>/hsaco_xragg

nix develop . --command cargo build --release -p plowrt -p plowc --features hsa

# blobs: the canonical recipe (control), and the same + the emit knob (item 1 only)
BASE="GLM_FULL=1 PLOW_MLA_PREFILL=full GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 \
GLM_SHARD_HEAD=1 PLOW_GLM_DSA=0 PLOW_GLM_FUSE_B1=1 PLOW_GLM_GEMV_WG=152 PLOW_MLA_PF_V2=1 \
PLOW_GLM_PF_NS=2 PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_FUSE_SEAM=1"
env $BASE [PLOW_GLM_FUSE_QNORM=1] nix develop . --command ./target/release/plowc \
  --emit devblob --hf-dir /workspace/models/GLM-5.2-FP8 --gpu MI300X --arch gfx942 \
  --num-gpus 8 --max-ctx 73728 --out <asset>/model.pkt

# served A/B (serve env needs PLOW_MLA_PF_V2=1); one arm per serve, strict alternation
IN_LENS="1024 4096" CONCS=1 NPROMPT=4 OUTLEN=96 bash scripts/bench_speed.sh <asset> 8195 auto

# item 2's oracle
hipcc --offload-arch=gfx942 -O3 -w [-DPLOW_MOE_PF_EPI_SIB=1] --genco runtime/amd/test_kernels.hip \
      -o tk.co -Iruntime/amd -Iruntime/common
clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
      --input=tk.co --output=test_kernels.elf
gcc -O2 -std=gnu11 -o moe_gemma_test runtime/tests/moe_gemma_gfx950_test.c runtime/amd/hsa_backend.c \
      -I/opt/rocm-7.2.4/include -L/opt/rocm-7.2.4/lib -lhsa-runtime64 -lm
HIP_VISIBLE_DEVICES=7 ./moe_gemma_test test_kernels.elf
```

Assets built for this record: `/workspace/assets/gfx942/glm52-di-{ctl,qnorm,sib,xragg}`
(the sib and xragg `model.pkt` are symlinks to the control's — those two items change no blob).
Objects: `/root/.claude/jobs/dropped/hsaco_{ctl,qnorm,sib,xragg}`.
