# GLM-5.2 TP8 decode: the per-packet census, and two chain-hop folds that pay

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **PLOW-ARCHITECTURAL** — chain-hop folds delete PACKETS, so the win is in the gate/boundary term the execution model owns. Arch-independent in direction; -2.7% is gfx942.

2026-08-08, branch `decode-seams` (base `9468fdb`), 8x MI300X (gfx942, 304 CU).
Best-config control reproduces the campaign's published **29.7 @1k / 32.0 @4k** exactly,
so every delta below is against the shipping number and not against a re-derived one.

Two folds landed, both opt-in, measured together and separately:

| | @1k | @4k | numerics |
|---|--:|--:|---|
| control (shipping best config) | 29.70 / 29.76 / 30.46 | 32.00 / 32.06 / 32.05 | ref |
| `PLOW_GLM_FUSE_ROPE=1` (q-rope into the flash) | 29.44 | 31.67 | **logits BYTE-IDENTICAL** |
| `PLOW_GLM_FUSE_SEAM=1` (layer seam AddNorm) | 29.33 | 31.63 | logit-gate (AddNorm class) |
| **both** | **28.94 / 28.89 / 28.88** | **31.23 / 31.19 / 31.25** | gates PASS |
| **delta (both)** | **-0.8 ms (-2.7%)** | **-0.82 ms (-2.6%)** | |

Packets per token **2678 -> 2523 (-5.8%)**; per MoE layer **35 -> 33**.

---

## 1. The census — this is what decided which folds to build

Instrument: `PLOW_TRACE_RAW` on `amd-bench --tp 8 --ctx 1024 --steps 24`, reduced by
`scripts/glm52_decode_census.py` (checked in with this record). **Grouped by `inst`, not
`pc`** — `pc` is the per-workgroup stream slot, so grouping by it splits one packet across
304 rows. Averaged over the 69 steady-state MoE layers (L6..L74); layer span is stable to
+/-2% (median 355.4 us, min 349.9, max 365.0).

### 1a. Per-op, per MoE layer (control)

| op | pkts/layer | wg/pkt | span us | busy CU-us | gate-wait CU-us |
|---|--:|--:|--:|--:|--:|
| MoeExpertDownFp8Blk | 8 | 32 | 378.98 | 11109 | 8496 |
| MoeExpertGluFp8Blk | 8 | 32 | 193.00 | 4451 | 2796 |
| Gemv | 3 | 218.7 | 77.66 | 11674 | 20720 |
| GemvQkv | 2 | 147.5 | 39.95 | 4925 | 22523 |
| FlashMlaDecode | 1 | 128 | 33.57 | 3909 | 5986 |
| XReduce | 2 | 12 | 28.05 | 298 | 859 |
| GemvGlu | 1 | 48 | 20.33 | 529 | 246 |
| MlaMergeFold | 1 | 64 | 18.14 | 1046 | 4038 |
| **RmsNorm** | **3** | **1.0** | 15.31 | 15 | 159 |
| **MoeRouterTopk** | **1** | **1.0** | 14.71 | 15 | 12 |
| **HeadNormRope** | **2** | **1.0** | 8.43 | 8 | 48 |
| MoeCombine | 1 | 12 | 6.84 | 67 | 725 |
| **AddNorm** | **1** | **1.0** | 5.64 | 6 | 23 |
| **Residual** | **1** | **1.0** | 4.20 | 4 | 66 |
| **TOTAL** | **35** | | 844.80 | 38056 | 66697 |

**Gate-wait share = 66697 / (66697 + 38056) = 63.7%** of all CU-time inside packets.
(Sums of `span` exceed the layer span because the 8 expert slot-pairs are concurrent by
construction; the serial chain is §1b.)

**EIGHT of the 35 packets are one workgroup**: 3x RmsNorm, 2x HeadNormRope, AddNorm,
MoeRouterTopk, Residual. They are 47.7 us of span against a 355 us layer, they are 0.1% of
the busy time, and their own gate-wait is ~nil — a 1-workgroup packet waits for nothing and
everything waits for it. This reproduces, on gfx942/TP8/GLM-5.2 and on the current best
config, exactly what the MI355X TP4 attribution
(`perf-data/glm52-gate-stall-attribution.md`) found in July: the 1-CU bucket is the money,
`Dep::Fine` on the collectives is 1% of it, and `d_xreduce`'s whole arithmetic ceiling is
0.026 ms. Nothing in that file needed revisiting; this census extends it to the shipped
config and prices the individual seams.

### 1b. The serial critical path (layer 40, us from layer start)

This is the number that picks the folds — not the op totals, the *windows between them*.

```
  73.3- 78.8  RmsNorm  input_layernorm    b=1     <-- fold S deletes this packet
  80.0-101.9  GemvQkv  (fusion A)         b=146
 108.0-112.6  RmsNorm  q_a_layernorm      b=1     [12.2 us window for a 4.6 us body]
 114.1-132.2  GemvQkv  (fusion G)         b=149
 137.3-141.8  HeadNormRope q              b=1     <-- fold R deletes this packet
 108.0-112.6  RmsNorm  kv_a               b=1  \  concurrent, OFF the critical path,
 108.2-112.1  HeadNormRope k              b=1  /  done ~30 us before the flash gate opens
 142.8-175.7  FlashMlaDecode              b=128
 181.1-199.4  MlaMergeFold                b=64
 201.8-222.6  Gemv     o_proj             b=304
 227.5-242.2  XReduce                     b=12
 242.8-248.5  AddNorm  (fuse_b1)          b=1
 249.7-273.8  Gemv     router             b=304
 ... MoE experts ...
 399.3-406.2  MoeCombine                  b=12
 407.4-423.6  XReduce                     b=12
 424.5-428.6  Residual                    b=1     <-- fold S folds this INTO the next
                                                      layer's input_layernorm
```

Three windows are pure packet-boundary cost:

* `GemvQkv-A` end 101.9 -> `GemvQkv-G` start 114.1 = **12.2 us** to run a 4.6 us
  1-workgroup norm.
* `GemvQkv-G` end 132.2 -> flash start 142.8 = **10.6 us** for a 4.4 us 1-workgroup rope.
* tail `Residual` + next layer's `RmsNorm` = two b=1 packets back to back = **~12 us**.

Serial packet-boundary dead time (sum of positive gaps) is **78.2 us/layer of a 355 us
layer**.

---

## 2. Fold R — the q RoPE into the flash decode (`PLOW_GLM_FUSE_ROPE=1`)

This is task #11's `hnr->flash` item, and the first thing to record is that **the premise
needed correcting**. `perf-data/plow-gfx942/fusion-review-and-crossover-sweep.md` §6
describes this fold as designed-and-unlanded, but §7 of the same file LANDED it (Gemma,
`PLOW_FUSE_HNR`) and measured it a **null** (+0.3%), later +0.04 ms on the 7.2.4 runtime
(§15). And the GLM audit `glm52-fusion-audit.md` seam 5 verdicts it **NO** — correctly, for
PREFILL, where the two ropes are 129 us already overlapped to ~1 us gaps.

None of that transfers to GLM **decode**, and the census says why:

* Gemma's hnr packets carry FINE per-head producer maps, so they start before the slowest
  GEMV workgroup and the level's true cost is a small tail. GLM's decode q rope is a
  **b=1** packet with a COARSE dep on a 149-wide GEMV: it waits for the slowest of 149,
  then runs alone on 1 CU while 303 idle, and only then does the 128-wide flash gate open.
* The prefill verdict is about the v2 flash's pre-formed A-fragments. The decode flash
  stages Q into LDS element-by-element and reads every one of those `nh_l*DR` values
  already.

So the fold was re-derived for this shape rather than ported.

**What landed.** Only the **q** rope. The k rope and the kv_a norm are concurrent with the
q_a norm and finish ~30 us before the flash gate opens (§1b) — folding them buys nothing
and would take on the KV-cache write hazard (the current row must be written for FUTURE
steps, so the writer would have to be whichever split covers `pos`). That hazard is the
entire risk in the Gemma §6 design and this fold does not incur it.

**Slot budget.** `FlashMlaDecode` (op 50) dense arm needed two operands and had exactly
two free:

| need | where | why it fits |
|---|---|---|
| raw `q_rope` | **t[3]**, replacing the roped `act.qr` | same slot, no cost |
| position | **nothing** | `qpos = kv_len[b] - 1`, and every decode entry in plowrt (`decode_step`, `decode_step_batched`, `serve/engine.rs`) calls with `kvlen == pos + 1`, so this is bit-exactly the `pos[0]` the rope packet read. `kv_len` is already t[6]. |
| cos table | **t[7]** | free on dense op 50 |
| sin table | **i[6]** as a demoted tensor handle | the `DevOp::GemvQkvg` rule, which `GemvQkvMxfp4` already reuses for three E8M0 scale rows; legal for a read-only generated table |

**Operand-collision check, explicitly.** `i[6]` here is a HANDLE, not a bitfield, and it is
on **op 50**. The bitfield warned about in the brief is **op 51** `FlashMlaPrefill`
(low 8 bits = causal KV-split `ns`, bit 8 = `W_ofold`; sparse GATHER reuses i[6] whole as
`cap`, discriminated by t7). Different opcode, different phase, and the two never meet.
The discriminator here is `t[7] != NONE`, which is unambiguous on op 50 because the GATHER
twin (54) and the fp8 twin (109) are **separate opcodes** and dense op 50 has never carried
a t7. `i[6]` cannot be the discriminator: 0 is a legal handle. The fold is slot-EXCLUSIVE
with both siblings (they spend t7/i6) and `plowc` asserts the exclusion at emit; the
interpreter traps on a packet with t7 but no resolvable i6.

**Loader trap avoided.** Neither operand is an XReduce `i[2]`, so the
`max(XR i[2]) -> slot_bytes` inference (`asset/devblob.rs:198`) is untouched. Verified:
`slot_bytes` is 100663296 on both arms.

**Kernel.** A RUNTIME BRANCH in `d_flash_mla_decode`'s Q staging, not a template parameter.
That is deliberate and it is the GF=8 lesson applied (`op_attention.h`: a second
INSTANTIATION grew the decode object 15.6% and cost +32% *even with the registers fitting*,
inside a persistent megakernel where every packet body shares one instruction stream).
Measured object cost of the branch: `interp_decode_gq.elf` **523728 -> 525064 bytes
(+0.26%)**, resources **108 VGPR / 0 AGPR / 30776 LDS / 0 spill — unchanged**.

**Bit-identical, and why.** The packet it replaces runs `d_headnorm_rope<64,INTERLEAVE>`
with `gamma == nullptr` and `skip_norm == 1`, so its value is exactly
`f2bf(v*cos -/+ partner*sin)` with `v = bf2f(x)` — no norm, no gamma. The fold is that
expression character for character, with the RoPE partner READ FROM MEMORY instead of
`__shfl_xor(.,1)`. The shuffle is not reusable: it assumes `lane == element index`, true in
the rope kernel (one wave per head) and false in a staging loop that walks a flat `GF*DR`
range with stride `PLOW_THREADS`. A second bf16 load of a line already in L1 is the cheap
way to be exactly right.

**Verified byte-identical on the GPU**, not argued: `amd-bench --dump-logits`, 512-token
seed-7 prompt, TP8 — all 8 decode logit rows AND the prefill row `cmp`-equal to the control.

```
logits_000.bin: IDENTICAL   logits_004.bin: IDENTICAL
logits_001.bin: IDENTICAL   logits_005.bin: IDENTICAL
logits_002.bin: IDENTICAL   logits_006.bin: IDENTICAL
logits_003.bin: IDENTICAL   logits_007.bin: IDENTICAL
logits_prefill.bin: IDENTICAL
```

**Cost, stated.** The 128 flash workgroups each re-derive the rope for their own head group,
and the flash packet grows: span 33.57 -> 35.70 us, busy 3909 -> 4191 CU-us/layer (+7%).
That is real and it is the price; it is far smaller than the b=1 packet's chain cost.

**Measured**: -0.26 @1k, -0.33 @4k. Critical-path window `GemvQkv-G end -> flash start`
**10.6 -> 6.3 us**; the residual 6.3 is the gate round trip itself, which no fold reaches.

---

## 3. Fold S — the layer seam (`PLOW_GLM_FUSE_SEAM=1`)

The census's other back-to-back b=1 pair: the FFN tail's `Residual`
(`x_out = xmid + ffn`) and the NEXT layer's `input_layernorm` `RmsNorm`. That is the same
`AddNorm` pair the ATTENTION seam has fused since `PLOW_GLM_FUSE_B1` — one packet writes
both the residual stream and its norm. The MoE/dense seam simply never got it, because the
gamma belongs to the next layer.

**No plumbing was needed**, which is why this is the cheap one. The next layer's gamma is
`n.lw[slot + 1].gin` — blocks are emitted in `slot` order over the same table — and "my
input is already normed" is exactly `slot > 0`. Both ends of the seam ask ONE predicate
(`glm_fuse_seam(tp)`) so producer and consumer cannot disagree. The last layer keeps the
plain `Residual` (`emit_glm_tail`'s `model.norm` is a different gamma and consumer) and so
does any single-block bring-up program (`n.lw.len() == 1`).

**TP-only, structurally.** At `tp == 1` there is no tail `Residual` to fold: `MoeCombine`
writes `x_out = xmid + ffn` itself. The collective is what forces the partial+Residual
split in the first place. `PLOW_NO_XREDUCE` takes the same branch and is excluded with it.

**Numerics: NOT byte-identical, by construction** — `d_add_norm` reduces over the
UN-ROUNDED `a + b` where the split path norms the bf16-rounded `x_out`. Exactly the class
`PLOW_GLM_FUSE_B1` already ships in. Measured: 8 of 8 decode logit rows differ, prefill row
IDENTICAL (the fold is decode-only, as intended). Serve gate is the acceptance test and it
passes character-identically (§5).

**Measured**: -0.37 @1k, -0.37 @4k. Deletes 77 `RmsNorm` + 77 `Residual`, adds 77
`AddNorm`: **-77 packets/token**.

---

## 4. Interleaved A/B, with the noise floor

`scripts/bench_speed.sh`, `IN_LENS="1024 4096"`, NPROMPT=6, OUTLEN=128, one arm per serve,
strict alternation, three ctl rounds and three both rounds interleaved across the session.
GPU lock held throughout; `rocm-smi` 0% on 8/8 before the batch; no foreign process.

**TPOT ms/token, served:**

| round | ctl @1k | both @1k | ctl @4k | both @4k |
|---|--:|--:|--:|--:|
| 1 | 29.70 | 28.94 | 32.00 | 31.23 |
| 2 | 29.76 | 28.89 | 32.06 | 31.19 |
| 3 | 30.46 | 28.88 | 32.05 | 31.25 |
| mean | 29.97 | **28.90** | 32.04 | **31.22** |

Attribution round (same session, same instrument): rope-only **29.44 / 31.67**, seam-only
**29.33 / 31.63**. The two folds are ADDITIVE to within the noise
(-0.26 + -0.37 = -0.63 vs -0.8 measured at 1k; -0.33 + -0.37 = -0.70 vs -0.82 at 4k), which
is what independent chain-hop deletions should do.

**The noise floor, and why the win clears it.** The box is quoted at +/-20% DVFS spread on
back-to-back walls; this instrument is far tighter. The `both` arm's own round-to-round
spread is **0.06 ms @1k and 0.06 @4k**; ctl is 0.06 @4k and 0.76 @1k (round 3 carries one
slow request — `itl_p99` 29.97 against a 30.46 mean). **The arms do not overlap at either
context**: the worst `both` @1k (28.94) beats the BEST ctl (29.70) by 0.76 ms = 12x the
`both` arm's spread, and @4k the worst `both` (31.25) beats the best ctl (32.00) by 0.75 ms.
Reported conservatively against the tightest ctl pair rather than its mean:
**-0.8 ms @1k, -0.82 ms @4k.**

Independent confirmation on the other instrument (`amd-bench --tp 8 --ctx 1024 --steps 8`,
real 512-token prompt, same session): ctl 29.218, rope 28.844, seam 28.815, both 28.410.
Same ordering, same magnitude.

TTFT is unmoved (344 / 1241 ms on every arm) — correct, both folds are decode-only.

---

## 5. Serve gate — the canonical three, verbatim, all four arms

`plowrt serve` on :8195, `temperature 0`. **Character-identical on all four arms**, so the
seam fold's logit drift does not reach the answers:

```
Q1  What is the capital of France? Answer in one short sentence.
    -> The capital of France is Paris.
Q2  What is 17 * 23? Answer with the number and then spell it out.
    -> 391 - three hundred ninety-one
Q3  What is the chemical symbol for gold, and name one common use for it?
    -> The chemical symbol for gold is **Au**.
       One common use for gold is in **jewelry**, where it is prized for its beauty,
       malleability, and resistance to tarnishing.
```

Plus `bench_speed.sh`'s own Paris gate: PASS on every one of the 8 bench serves, and
`amd-bench` reported "all 8 ranks token-identical" on every step of every arm (the TP
oracle — a rank that skipped a collective still samples fluent ids, so agreement is what
proves the all-reduces ran).

---

## 6. Post-fold census

| | control | both folds |
|---|--:|--:|
| packets / token | 2678 | **2523** (-5.8%) |
| packets / MoE layer | 35 | **33** |
| 1-workgroup packets / MoE layer | 8 | **6** |
| MoE layer span (median) | 355.4 us | **347.2 us** (-2.3%) |
| gate-wait share | 63.7% | 62.8% |
| `GemvQkv-G` end -> flash start | 10.6 us | 6.3 us |

-8.2 us/layer x 78 layers = -0.64 ms, against -0.8 measured; the balance is the dense
layers and the shortened program.

---

## 7. What was NOT landed, and what the census says is next

**The largest remaining single item is `q_a_layernorm -> GemvQkv` (fusion G), worth an
estimated -0.9 ms** and NOT built here. §1b prices its window at **12.2 us/layer** — the
biggest packet-boundary window left on the chain — for a 4.6 us 1-workgroup body. It is the
NRN-fold shape that PAID on Gemma (`fusion-review-and-crossover-sweep.md` §5: NRN2 -> q/k/v
via op 30 `i[3]`, -0.18 ms there) and it applies twice per GLM layer (`input_layernorm ->
GemvQkv-A` is a second 6.7 us window). It needs a norm-in-staging arm on `d_gemv_qkv`,
which `gemv_nrn_lds` in `op_gemm.h` is the template for; the decode object has room
(108 of 256 VGPR, 0 spill) where the flash did not. This is the recommended next build.

**Evaluated and NOT landed, with reasons:**

* **k rope / kv_a norm into the flash** — measured OFF the critical path (§1b): they run
  concurrently with the q_a norm and finish ~30 us before the flash gate opens. Folding
  them buys ~0 and takes on the KV-cache-write hazard. Do not build this.
* **AttnRes (op 104)** — confirmed unused by GLM: it is a Kimi-K3 op (`PLOW_K3` arm) and
  the GLM residual seam is already `AddNorm`/`Residual`. There is no AttnRes seam here to
  fold. The campaign note naming it as a GLM candidate is stale.
* **XReduce merges (`XReduce2`, and `GLM_FUSE_XRN`/op 116 at either seam)** — the census
  says the collective's whole prize is gone. `XReduce` is 28.05 us span and **298 CU-us
  busy** per layer at 12 workgroups; the July attribution already capped the entire
  reduce body at 0.026 ms/token (`-DPLOW_XR_NOREDUCE`) and the rendezvous at 0.347. What
  is left is the system-scope acquire fence x 12 wg x 156 packets, which only FEWER
  COLLECTIVES reach. `GLM_FUSE_XRN` (op 116, already in tree, opt-in) does delete one
  packet per attn seam, but it puts the 6144-element reduction on ONE workgroup where the
  sized collective uses 12 — the census's own §6b-i rule says that trades a narrow packet
  for a straggler tail. Not measured here; it is a separate arm with its own risk and it is
  NOT what the census points at.
* **MoE combine seam** — the `MoeCombine -> XReduce` pair is 12-wide and its `Residual`
  tail is exactly what fold S deleted. Nothing further left at this seam without touching
  the collective.

---

## 8. Reproduction

```bash
export PATH=/nix/var/nix/profiles/default/bin:/root/.nix-profile/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib
export ROCM_PATH=/opt/rocm-7.2.4 HIP_PATH=/opt/rocm-7.2.4 ROCM_HOME=/opt/rocm-7.2.4

# device objects (OUTSIDE nix)
env PLOW_OCC4=1 PLOW_L2HIER=1 JOBS=16 bash scripts/build_gfx942.sh <objdir>

# blobs: best config + the two knobs (unset = byte-identical to the shipping decode program,
# verified by disassembly diff)
nix develop . --command cargo build --release -p plowrt -p plowc --features hsa
env GLM_FULL=1 PLOW_MLA_PREFILL=full GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 \
    GLM_SHARD_HEAD=1 PLOW_GLM_DSA=0 PLOW_GLM_FUSE_B1=1 PLOW_GLM_GEMV_WG=152 \
    PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_FUSE_SEAM=1 \
  nix develop . --command ./target/release/plowc --emit devblob \
    --hf-dir /workspace/models/GLM-5.2-FP8 --gpu MI300X --arch gfx942 \
    --num-gpus 8 --max-ctx 73728 --out <assets>/model.pkt

# census
PLOW_TRACE_RAW=tr.bin nix develop . --command ./target/release/plowrt amd-bench \
  --blob <assets>/model.pkt --hsaco <objdir> --checkpoint <ckpt> --tp 8 --ctx 1024 --steps 24
./target/release/plowrt disasm <assets>/model.pkt --program 1 | grep '^#' > dec.txt
python3 scripts/glm52_decode_census.py dec.txt tr.bin

# A/B
IN_LENS="1024 4096" CONCS=1 NPROMPT=6 OUTLEN=128 \
  bash scripts/bench_speed.sh <assets> 8196 auto 1800
```

Assets built for this record: `/workspace/assets/gfx942/glm52-ds-{ctl,rope,seam,both}`,
objects `hsaco_ds1`.

### 8a. A live CLI bug this needed fixed first

**`amd-bench` could not run at all on this branch, and every script under `scripts/` that
benches a blob was dead with it.** `RuntimeConfig`'s `global = true` args are propagated
into every subcommand, and two of them defaulted their clap **id** to the FIELD name —
`hsaco` and `checkpoint` — colliding with `amd-bench`'s own `--hsaco` (`PathBuf`, required)
and `--checkpoint`. clap then held two definitions under one id and panicked on the type
downcast:

```
Mismatch between definition and access of `hsaco`.
Could not downcast to TypeId(...), need to downcast to TypeId(...)
```

Fixed by giving the globals explicit ids (`rt_hsaco` / `rt_checkpoint`); the global
`--hsaco` long is now `--rt-hsaco`, matching the `rt-` convention every other global there
already used. Nothing in the tree passed the global form. The subcommand flags every script
uses are unchanged.

### 8b. Arm-check chain (a folded blob on a pre-fold object is silently wrong)

The rope fold's arm is a runtime branch, so a pre-fold decode object does not trap on a
folded packet — it stages `t[3]` verbatim, i.e. feeds the flash an **UNROPED query**, and
the model answers fluently and wrongly. The full refusal chain landed with the fold:

* `interp.hip` exports `plow_glm_fuse_rope_arm` (unconditional; presence == "built at or
  after the fold").
* `manifest.rs` detects any `FlashMlaDecode` with `t[7] != NONE` -> feature
  `glm_fuse_rope` -> `requires: ["PLOW_GLM_FUSE_ROPE=1"]` in `build.json` (verified present
  in the emitted manifest, absent on the control).
* `plowrt` gained `DECODE_ARM_MARKERS` + `check_decode_object`, called for `Phase::Decode`.
  It is a SEPARATE table from `PREFILL_ARM_MARKERS` because `requires` is blob-wide: each
  check looks only at the flags its own table names, so a prefill flag cannot refuse the
  decode object.

### 8c. Known stale, pre-existing

`cargo test -p devgen --test tuned_tile_selection` fails: editing `runtime/amd/*.h` moves
the preprocessed build digest and stales the tuned-GEMM records (the documented
§6g-STALE-2 behaviour — the test's whole job is to say so). **One of its two failures is
already present on the base branch `worktree-glm52-bringup` before any change here**; the
kernel edit stales the second. Re-qualifying needs `scripts/rebench_tune_gemm.sh` +
`plowc tune ingest`, a GPU campaign of its own and out of scope for this task. Every other
test in `packet`, `devgen` and `plowrt` passes.

---

## 9. Default posture

Both knobs default **OFF**; unset emits a decode program verified byte-equivalent to the
shipping best config (disassembly diff, 2678/2678 instructions, differences only in the
new slot NAMES the disassembler now prints).

`PLOW_GLM_FUSE_ROPE=1` is the stronger candidate for default-on: it is byte-identical on
the GPU, so it carries no numerics risk at all, and its only cost is +0.26% decode object
and +7% flash busy. `PLOW_GLM_FUSE_SEAM=1` changes logits (AddNorm class) and should follow
`PLOW_GLM_FUSE_B1`'s precedent — opt-in until a broader quality run than the canonical
three has been done. Proposing default-on for either is left to consolidation, with this
record as the evidence.
