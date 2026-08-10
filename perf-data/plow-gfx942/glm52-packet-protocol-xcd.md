# GLM-5.2 decode on MI300X: what the packet protocol actually costs, and the
# instrumentation error that made it look like everything

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **PLOW-ARCHITECTURAL + MI300X XCD geometry** — the protocol analysis is architectural; 8 XCDs and the per-XCD 4 MB L2 are MI300X, and CDNA4 has a different XCD count.

2026-08-08, branch `protocol-xcd` (base `worktree-glm52-bringup` @ `a5b0423`), box
[[plow-devbox-is-gfx942]] (8x MI300X, gfx942, 304 CU, ROCm 7.2.4). Asset
`/workspace/assets/gfx942/glm52-tp8-final`, objects `hsaco_glm17`.

**The question.** `glm52-decode-gemv-aiter.md` §5c reported that deleting one hundred
percent of the work in the largest op of the GLM decode program moved its packet by 0.3%
and the token by 0.0%, and concluded that its 47.4 us is "entirely packet protocol". This
work was commissioned to root-cause that protocol cost on CDNA3's eight-L2 topology and
decide whether the interpreter needs a redesign for this silicon.

**The answer, in one line.** The 47.4 us was **37.6 us of real kernel**. The ablation was
compiled into `interp_decode_fp8_gq.elf`, and a GLM-5.2 packet **never loads that object**.
Rebuilt into the object the run does load, the same ablation moves the token from 28.90 to
25.49 ms (**-11.8%**), and the arm the ablation was built to justify — `PLOW_MOE_DEC_LG` +
`PLOW_MOE_DEC_X2`, previously recorded NULL — measures **-8.2%, token-identical**.

The protocol itself is **~1.5 us per workgroup-packet, 2.3% of the token**, and every one
of the four hypotheses about it is priced below. None of them is the problem.

---

## 0. THE METHOD BUG: which object does a GLM-5.2 run actually load?

`crates/plowrt/src/exec/amd.rs` composes the object filename from a `Variant` that is
**detected from the packet's opcodes**, never from a build flag:

```rust
pub fn detect(progs: &[DevProg]) -> Variant {           // amd.rs:276
    for p in progs { for i in &p.insts {
        if i.op == DevOp::FlashDecodeFp8 as u16
            || i.op == DevOp::FlashMlaDecodeFp8 as u16
            || i.op == DevOp::FlashMlaPrefillFp8 as u16 { return Variant::Fp8Kv; }
        if i.op == DevOp::GemvFp8 as u16 { v = Variant::Fp8; }
    } } v
}
```

`GemvFp8` is the **per-channel w8a8** rung. GLM-5.2's fp8 is **block-scaled**, so its
kernels are `GemvFp8Blk`, `DenseGluFp8Blk`, `MoeExpertGluFp8Blk`, `MoeExpertDownFp8Blk` —
none of which this function matches — and its MLA is bf16. Counted over **every program in
the shipped blob** (`plowrt disasm /workspace/assets/gfx942/glm52-tp8-final/model.pkt`):

```
ops present: AddNorm Argmax DenseGluFp8Blk Embed FlashMlaDecode FlashMlaPrefill Gemm
GemmGlu GemmMed GemmSmall GemmWide Gemv GemvFp8Blk GemvGlu GemvQkv HeadNormRope
MlaMergeFold MoeAlignPf MoeCombine MoeCombinePf MoeExpertDownFp8Blk MoeExpertGluFp8Blk
MoeGroupDownPf MoeGroupGluPf MoeRouterTopk MoeRouterTopkPf Residual RmsNorm XArgmaxFin
XReduce XReduceTwoShot

GemvFp8 / FlashDecodeFp8 / FlashMlaDecodeFp8 / FlashMlaPrefillFp8 occurrences: 0
```

so `Variant::detect` returns `Bf16` and the decode object is **`interp_decode_gq.elf`**
(525 KB), not `interp_decode_fp8_gq.elf` (773 KB). That is CORRECT behaviour, not a bug in
the runtime: the `*Fp8Blk` cases in `interp.hip` sit **outside** `#if PLOW_FP8`
(interp.hip:2246 vs the `PLOW_FP8` block at 2354-2374), deliberately — "the fp8 arms ADD to
the bf16 ones below, they do not replace them" — so the bf16 object carries every kernel
GLM needs, and it is the smaller of the two, which is the better object to be running.

What was wrong was the **instrumentation**. `PLOW_ROWS_ONLY=interp_decode_fp8` builds
`interp_decode_fp8{,kv}{,_gq}.elf` and nothing else. Every arm in
`glm52-decode-gemv-aiter.md` — `PLOW_MOE_DEC_LG`, `PLOW_MOE_DEC_X2`, and both `PLOW_MOE_DEC_ABL`
ceilings — went into four objects that the run opens zero times. The three results the file
reports are exactly what a no-op produces:

| the file's finding | what it really was |
|---|---|
| "lgx2 is NULL, -0.35% inside a 0.9% control spread" | the arm was not in the loaded object |
| "the packet busy time did not move AT ALL (0.1%)" | the kernel was not modified |
| "deleting 100% of the work moves the token by 0.0%" | **nothing was deleted** |

Nothing else in that file is affected — its ISA audit, its bit-identity proofs and its
off-device reduction/coverage checks are all correct and are what make the arm shippable.
Only the four measured tables and the conclusion drawn from them are void.

**Fix landed** (`crates/plowrt/src/exec/amd.rs`): the loader now logs, at INFO, the object
name it is about to open together with the detected `variant`/`prefill_arm`/`sched`. One
line, and this class of error becomes visible in every server log.

```
INFO plowrt::exec::amd: code object object="interp_decode_gq.elf" phase=Decode
     variant=Bf16 prefill_arm=None sched=GlobalQueue
```

**Rule for anyone building decode arms on a GLM/K3-class blob: `PLOW_ROWS_ONLY=interp_decode`,
not `interp_decode_fp8`.** The identity check that would also have caught it is one command:
the control object built by this tree with `PLOW_ROWS_ONLY=interp_decode` is
instruction-mix-identical to the shipped `hsaco_glm17/interp_decode_gq.elf`, and the
`_fp8_gq` control is identical to `hsaco_glm17/interp_decode_fp8_gq.elf` — both were
verified here, which is how the two objects were noticed to be different files at all.

---

## 1. The inter-packet window, from the SHIPPED object

`llvm-objdump -d --mcpu=gfx942` on `hsaco_glm17/interp_decode_gq.elf`, symbol
`plow_interp_dec_gfx942_gq`. The whole object contains **12 `buffer_inv`, 12 `buffer_wbl2`
and 13 `global_atomic_add`**; the ones in the interpreter loop are these, in program order,
between one packet's body and the next:

```
  s_barrier                                     <- __syncthreads(), top of iteration
  global_atomic_add v1, v97, v1, s[4:5] sc0     <- gq cursor fetch_add  (RELAXED, AGENT)
  s_waitcnt vmcnt(0)                            <- DEPENDENT: the claim is a full round trip
  ds_write_b32 ... offset:30768                 <- broadcast gq_claim
  s_barrier
  ds_read_b32
  s_load_dwordx4/x2 s[52:55],s[0:1]             <- the 24 B PlowStreamEnt (scalar cache)
  s_load_dword s[56:57] 0x0 / 0x20 / 0x30       <- PLOW_INST_PF: PlowDevInst above the gate
  global_atomic_add v1, v97, v1, s[6:7] offset:128 sc0   <- GATE_HIER observe election (arr)
  s_waitcnt vmcnt(0)                            <- DEPENDENT
  ds_write_b32 ... offset:30764 ; s_barrier ; ds_read_b32   <- broadcast h_lead
  --- leader only ---
  global_load_dwordx2 v[0:1] ...                <- the PlowWait (id, threshold)
  s_waitcnt vmcnt(0)
  global_load_dword v0, v[2:3], off sc1         <- ctr_poll: DEVICE-SCOPE load, per iteration
  s_waitcnt vmcnt(0) ; s_sleep 8                <- the spin
  --- both ---
  s_barrier                                     <- gate satisfied
  buffer_inv sc1                                <- ctr_acquire(), LEADER, one per XCD
  buffer_wbl2 sc1                               <- the release RMW on `opn`
  global_atomic_add v97, v0, s[6:7] offset:256
  ... follower path: global_load_dword ... offset:256 sc1 ; s_sleep 8 ; buffer_inv (L1 only)
  s_barrier
  <op body>
  s_barrier
  global_atomic_add v1, v97, v1, s[8:9] sc0     <- GATE_HIER publish arrival (ldn)
  s_waitcnt vmcnt(0)                            <- DEPENDENT
  ... last-in-domain only, per successor:
      global_load_dword v96 ...                 <- the successor id
      buffer_wbl2 sc1
      global_atomic_add v[0:1], v2, off         <- RELEASE, adds h_n not 1
  s_barrier
```

Counts per (workgroup, packet), GATE_HIER on, non-collective entry:

| | per workgroup | per packet |
|---|--:|--:|
| `s_barrier` | 4 (5 traced) | — |
| dependent `global_atomic_add` + `s_waitcnt vmcnt(0)` | **3** | — |
| `buffer_inv sc1` | 0 (leader: 1) | **8** (one per XCD) |
| `buffer_inv` (L1 only) | 1 (followers) | 304-8 max |
| `buffer_wbl2 sc1` | 0 (leader: 1, publisher: 1) | **8-16** |
| `global_load ... sc1` (poll) | >=1 per spin iteration | — |
| `s_sleep 8` | 1 per spin iteration | — |

The **cross-GPU** path (`PLOW_SE_XCTR`, XReduce and friends) is the only place SYSTEM scope
appears — `buffer_inv sc0 sc1` on the wait side and `buffer_wbl2 sc0 sc1` + `flat_atomic_add
... sc1` per peer on the signal side. Every LOCAL gate atomic and every counter poll is
AGENT scope. **There is no over-broad scope on the hot path** (hypothesis 3), and section 2
shows there would be little to win even if there were.

---

## 2. Per-unit costs on this silicon (`probes/pktproto.hip`)

Each arm is the same R-iteration loop with `s_memrealtime` around it, `nop` subtracted, max
over blocks, best of 3, 512 threads/block, idle GPU under the lock. **ns per operation:**

| arm | 1 | 8 | 16 | 32 | 64 | 152 | 304 blocks |
|---|--:|--:|--:|--:|--:|--:|--:|
| `nop` loop (absolute) | 100.8 | 101.2 | 101.3 | 101.3 | 101.3 | 101.4 | 101.7 |
| **`buffer_inv sc1`** (agent acquire) | 19.9 | **25.1** | 26.7 | 154.7 | 410.7 | 1115.1 | **2333.9** |
| `buffer_inv` (L1 only, HIER follower) | 7.7 | 7.8 | 7.8 | 7.8 | 7.8 | 8.3 | **8.7** |
| `buffer_inv sc0 sc1` (system acquire) | 24.9 | 25.2 | 26.7 | 154.7 | 410.7 | 1115.0 | 2334.0 |
| **`buffer_wbl2 sc1`** (release writeback) | 26.8 | **27.1** | 27.2 | 154.8 | 410.9 | 1115.0 | **2334.3** |
| **atomic_add ret, PER-BLOCK line** | 332.7 | 334.2 | 343.5 | 340.3 | 364.9 | 383.8 | **391.9** |
| atomic_add ret, ONE shared line | 359.1 | 380.4 | 385.6 | 379.4 | 509.2 | 1337.7 | **2769.5** |
| atomic_add ret, per-block, **SYSTEM** | 366.4 | 390.0 | 396.4 | 377.8 | 396.0 | 417.9 | 426.1 |
| dependent load, agent (`sc1`) | 117.8 | 129.1 | 131.0 | 133.6 | 135.9 | 134.4 | 135.5 |
| dependent load, workgroup (plain) | 53.2 | 54.7 | 55.3 | 55.3 | 55.6 | 55.7 | 55.9 |
| `s_barrier`, 512 threads | 30.3 | 30.6 | 30.6 | 30.6 | 30.6 | 32.5 | 35.5 |

Five facts fall straight out, and three of them retire a hypothesis:

1. **L2 maintenance is FLAT to 16 concurrent issuers (~25 ns) and only then serialises**, at
   a device-wide ~7.7 ns per operation (2334/304 = 7.68; the L1-only form is 7.7 ns and does
   NOT serialise, confirming that 7.7 ns is the issue cost and the `sc1` forms take a
   device-global turn). **`PLOW_GATE_HIER` puts exactly 8 issuers on a packet — squarely
   inside the flat region.** The invalidate that "serialises across concurrent issuers" is
   real, and the hierarchy already moved the design off the part of the curve where it bites.
2. **The `sc1` and `sc0 sc1` invalidate cost the same.** On gfx942 there is no extra charge
   for system-scope cache maintenance over device-scope.
3. **A SYSTEM-scope returning atomic costs 426 ns against AGENT's 392 — +9%.** Even a
   hot-path scope error would be worth <2% of the protocol, which is itself 2.3% of the
   token. Hypothesis 3 has no headroom on this silicon whether or not the code is right
   (it is right).
4. **The expensive thing is CONTENTION ON ONE LINE**: 392 ns -> 2770 ns going from a
   per-block counter to a shared one at 304 claimants. `PLOW_CTR_STRIDE` (128 B, one line
   per counter), the per-domain `gq_cursor` and the per-`(packet, domain)` hier triple keep
   every hot line under ~38 claimants, which is still on the flat part of that curve.
5. **A dependent returning atomic is 0.33-0.39 us and that is the protocol's largest single
   term.** The interpreter takes THREE of them per workgroup-packet (claim, leader
   election, publish arrival), each with an `s_waitcnt vmcnt(0)` that cannot be hidden.

---

## 3. The traced decomposition

`PLOW_TRACE_PHASE=1` (landed with this work) adds two `s_memrealtime` probes inside the
existing `if (prog.trace)` guard and packs their deltas into the trace record's `pc` field,
which every census reducer already discards. One traced run then yields five phases:

```
T0 top of iteration (BEFORE the claim barrier)   -> r.t_arrive
T1 after the gate poll and its __syncthreads()
T2 after the acquire / XCD rendezvous            -> r.t_ready
T3 after the op body and its __syncthreads()
T4 after the successor release                   -> r.t_end
r.pc = min(T1-T0,0xffff) | (min(T3-T2,0xffff) << 16)
```

Under `PLOW_GATE_HIER` a FOLLOWER skips the gate poll entirely and waits on the XCD-local
`opn` inside the acquire section, so `T0->T1` and `T1->T2` are both dependency wait; their
sum is what the census calls gate-wait. Averaged over the 69 steady-state MoE layers,
**CU-us per layer**:

| op | wg-pkt/lay | claim+gate | acquire/rendezvous | body | publish |
|---|--:|--:|--:|--:|--:|
| Gemv | 656 | 3310 | 18613 | 10577 | 692 |
| GemvQkv | 295 | 1861 | 20083 | 4728 | 196 |
| MoeExpertDownFp8Blk | 256 | 3014 | 6188 | 10928 | 191 |
| FlashMlaDecode | 128 | 883 | 4638 | 4120 | 89 |
| MoeExpertGluFp8Blk | 256 | 1645 | 1953 | 4080 | 329 |
| MlaMergeFold | 64 | 845 | 3162 | 1012 | 43 |
| XReduce | 24 | 647 | 249 | 321 | 17 |
| GemvGlu | 48 | 172 | 172 | 496 | 33 |
| MoeCombine | 12 | 506 | 240 | 60 | 8 |
| the five b=1 ops | 6 | 208 | 0 | 39 | 2 |
| **TOTAL** | **1745** | **13090** | **55299** | **36361** | **1600** |
| **share** | | **12.3%** | **52.0%** | **34.2%** | **1.5%** |

`publish` — the post-body barrier, the dependent publish atomic, and the amortised
`buffer_wbl2 sc1` + successor release — is **0.44-1.29 us per workgroup-packet, mean 0.92 us**,
in exact agreement with §2's arithmetic. It is **1.5% of in-packet CU time**.

Caveat, stated because it matters: this arm runs at 40.2 ms/token against the control's
29.2 ms traced. Two extra `s_memrealtime` live across the op body raise the megakernel's
VGPR count 108 -> 110 and perturb the inner loops. The **shape** (which phase dominates)
transfers and the **publish** number is a local cost that does not; the **body** column is
inflated and is not used below — §4 gets the body from an ablation instead.

---

## 4. The 47.4 us DOWN packet, decomposed — and it is not protocol

Three arms, identical in everything but the two `-D` flags, all built into
**`interp_decode_gq.elf`** this time, all on the same `model.pkt`, `amd-bench --tp 8
--ctx 1024 --steps 24`:

* `ctl2` — the shipped bodies (instruction-mix-identical to `hsaco_glm17`)
* `lgx2` — `PLOW_MOE_DEC_LG=1 PLOW_MOE_DEC_X2=1`
* `abl2` — `PLOW_MOE_DEC_ABL=2`, the DOWN op RETIRED (wrong output by construction)

**`busy` = summed per-workgroup `t_end - t_ready`, CU-us per MoE layer:**

| op | ctl2 | abl2 (DOWN deleted) | lgx2 |
|---|--:|--:|--:|
| **MoeExpertDownFp8Blk** | **11095.2** | **1449.9** | **3255.8** |
| MoeExpertGluFp8Blk | 4386.4 | 4704.5 | 4571.2 |
| Gemv | 11243.1 | 11216.1 | 11285.0 |
| GemvQkv | 4895.6 | 4777.9 | 4735.4 |
| FlashMlaDecode | 4186.8 | 4153.5 | 4214.7 |
| DOWN span us/layer | 379.0 | 75.6 | 127.6 |
| **MoE layer span, median** | **348.6** | **301.2** | **316.1** |

Per workgroup-packet (divide the DOWN row by 256): **43.3 us shipped, 5.66 us with the
kernel retired, 12.7 us with the lane-group map.** So the 47.4 us packet is:

| component | us / workgroup | how it was measured |
|---|--:|---|
| dependency wait (gate poll + XCD rendezvous) | **32.9** | ctl2 trace, gate-wait 8412.9 CU-us/lay / 256 |
| **op body (the actual GEMV)** | **37.6** | ctl2 busy 43.3 − abl2 busy 5.66 |
| op header: 3 dependent routing-table loads + tail barrier | 4.9 | abl2 busy 5.66 − publish 0.75 |
| publish + successor release | **0.75** | PLOW_TRACE_PHASE |
| claim + leader election + 4 barriers | **~0.9** | pktproto (outside the traced window) |
| cache maintenance (`buffer_inv sc1` + `buffer_wbl2 sc1`) | **~0.03** | pktproto, 8 issuers |
| **packet SPAN** (min t_ready -> max t_end) | **47.4** | |
| **per-workgroup claim -> end** | **76.9** | |

**Protocol is 1.7 us of a 76.9 us workgroup-packet: 2.2%.** The 47.4 us is a kernel, and
the "protocol floor" was an artifact of ablating an object nothing loads.

**Whole-token protocol budget.** 1745 wg-packets/layer x 78 layers = 136,110 per token;
2523 packets x 8 XCD leaders:

| term | ms/token | % of 28.90 |
|---|--:|--:|
| dependent counter atomics (3 x ~0.37 us) | 0.50 | **1.7%** |
| L2 cache maintenance (2523 x 8 x 2 x 7.7 ns, device-serialised) | 0.31 | **1.1%** (upper bound; it overlaps) |
| `s_barrier` (4 x 30 ns) | 0.05 | 0.2% |
| memory scope (nothing over-broad on the local path) | 0.00 | 0% |
| **TOTAL PACKET PROTOCOL** | **~0.86** | **~3.0%** |

### Verdict on the four hypotheses

1. **L2 writeback / invalidation between packets — REFUTED as a major term, 1.1% ceiling.**
   The mechanism is real and the scaling curve is real (2334 ns at 304 concurrent issuers),
   but `PLOW_GATE_HIER` already elects one leader per XCD and 8 issuers sit on the FLAT part
   of that curve at 25 ns. There is 0.31 ms/token behind this even if it went to zero.
   *XCD-local gating and batched maintenance are therefore not worth building* — the
   placement they would exploit has already paid, through the hierarchy, and the residue is
   a third of a millisecond.
2. **Counter / atomic contention — 1.7%, and it is LATENCY, not contention.** The 128 B
   stride, the per-domain cursor and the per-`(packet, domain)` hier triple keep every hot
   line at <=38 claimants, where a returning atomic is 0.38 us — the same as uncontended.
   What costs is that there are THREE dependent round trips per workgroup-packet and each
   has an unhidable `s_waitcnt vmcnt(0)`. A cheaper protocol would remove round TRIPS, not
   contention; the best case is ~0.5 ms/token.
3. **Memory scope too broad — NOT PRESENT, and it would not pay if it were.** Every local
   gate atomic and poll is AGENT (`sc0`/`sc1`); SYSTEM appears only on the `xctr` peer path,
   which needs it. And on gfx942 SYSTEM costs +9% on an atomic and +0% on cache maintenance.
4. **Claim-path head-of-line blocking / dispatch skew — THIS IS WHERE THE TIME IS, and it is
   not a protocol cost.** 63.1% of in-packet CU time is dependency wait (63.7% published;
   reproduced here at 63.1%). Of the DOWN packet's 32.9 us of wait, 1.7 us is protocol; the
   rest is waiting for producers and for a workgroup to become free. That is a PACKING
   problem — per-CU precomputed schedules, stealing, fewer packets — and the levers are the
   ones `glm52-dsa-sparse-b3.md` already names.

**So: the interpreter does not need a CDNA3-specific redesign.** The three MI300X-specific
worries — eight non-coherent L2s, per-XCD writeback, cross-fabric counters — are together
worth under 3% of the token, and the hierarchical gate that already ships is what took them
there. What is left of the packet is the kernel and the packing.

---

## 5. The win the method bug was hiding

`PLOW_MOE_DEC_LG` (narrow-K lane-group DOWN) and `PLOW_MOE_DEC_X2` (paired gate|up GEMV)
were landed opt-in and recorded NULL. Rebuilt into `interp_decode_gq.elf`, interleaved
`amd-bench --tp 8 --ctx 1024 --steps 24`, all 8 ranks token-identical on every step of
every run:

| round | ctl2 | lgx2 | abl2 (instrument, wrong output) |
|---|--:|--:|--:|
| 1 | 28.890 | 26.548 | 34.171 |
| 2 | 31.577 | 26.501 | 25.494 |
| 3 | 28.905 | 26.534 | 25.488 |
| **median** | **28.905** | **26.534** | **25.494** |

**-8.2%.** `lgx2`'s three rounds span 0.047 ms (0.18%) and every one of them beats every
control round; the control's own spread is 2.7 ms because of one high round, and the
treatment is 9x the control's tight-round spread either way. `abl2` r1 is a matching
outlier. Both arms sample token **347** exactly as the control does, on all 8 ranks, on
every step — the bit-identity `glm52-decode-gemv-aiter.md` §3 proves by construction, and
its off-device reduction and row-coverage checks, all still stand.

The ablation frames the ceiling: deleting DOWN entirely is 25.49 ms, so DOWN's kernel is
worth **3.41 ms/token** and `lgx2` recovers **2.37 ms of it (70%)**. Per workgroup-packet
the DOWN body goes 37.6 us -> 7.0 us (**-81%**), which is what a 4.3x leaner inner loop with
24 rows in flight should do. GLU's `busy` moves the wrong way slightly (4386 -> 4571 CU-us/
layer): the `X2` half is not obviously carrying its weight and an `LG`-only arm is built
(`/tmp/pxcd/o_lg`) for the follow-up A/B.


---

## 6. The number that DOES matter: 2.13 us of exposed latency per packet boundary

The 3.0% figure in §4 is protocol as a share of **aggregate CU-time**, and on its own it
undersells the protocol, because the protocol is **latency on the dependency chain** and
nothing overlaps it. The census's serial packet-boundary dead time is the right instrument,
and it is the one quantity in the whole trace that does NOT move when the kernels get faster:

| arm | MoE layer span | total busy CU-us/lay | perfect-pack floor | packing efficiency | **serial boundary dead time** |
|---|--:|--:|--:|--:|--:|
| ctl2 (shipped) | 348.6 us | 37813 | 124.4 us | 35.7% | **70.25 us/layer** |
| lgx2 | 316.1 | 30103 | 99.0 | 31.3% | **69.67** |
| abl2 (DOWN retired) | 301.2 | 28290 | 93.1 | 30.9% | **67.29** |

**70 us of every 348 us layer is packet-boundary dead time, and deleting the single largest
kernel in the program leaves it at 67.** Per boundary (33 packets/layer) that is
**2.13 us**, and it is essentially all protocol — here is the chain, priced from §2:

| exposed step | ISA | ns |
|---|---|--:|
| producer's last workgroup: publish arrival | `global_atomic_add sc0` + `s_waitcnt vmcnt(0)` | 350 |
| producer: publish the writeback | `buffer_wbl2 sc1` (8 issuers) | 27 |
| producer: release the successor counter | `global_atomic_add` (RELEASE) | 350 |
| consumer leader: observe it | `global_load ... sc1` + `s_sleep 8` granularity | 135 + ~250 |
| consumer leader: invalidate this XCD's L2 | `buffer_inv sc1` (8 issuers) | 25 |
| consumer leader: release its followers | `buffer_wbl2 sc1` + `global_atomic_add` (RELEASE) | 377 |
| consumer followers: observe it | `global_load ... sc1` + `s_sleep 8` granularity | 135 + ~250 |
| consumer followers: invalidate own L1 | `buffer_inv` | 8 |
| barriers on the way | 3 x `s_barrier` | 90 |
| **TOTAL** | | **~2000 ns** |

against a measured 2130 ns. **The model closes.** At 33 boundaries x 78 layers that is
**5.5 ms of the 28.9 ms token (19%)** — an order of magnitude more than the aggregate-CU
share, and it is the honest cost of the protocol.

Note what it is made of: **five dependent global round trips and two poll-granularity
delays. NOT cache maintenance** — the two `buffer_inv sc1` and two `buffer_wbl2 sc1` on the
chain are 52 ns of the 2000, i.e. **2.6% of the boundary and 0.5% of the token**. Hypothesis
1 is refuted twice over: once in aggregate and once on the critical path.

`PLOW_GATE_HIER` is the reason the maintenance is cheap (8 issuers, flat region) and it is
ALSO what adds the second observe hop (leader -> `opn` -> follower), worth ~770 ns of the
2130. It measured -16% on Gemma and -12% on GLM, so it is a large net win; but a protocol
rebuild's first target should be **collapsing the two-hop observe into one** while keeping
one L2 invalidate per XCD, not removing maintenance that is already almost free.

### What a claim-path rebuild has to beat, and what it must not bother with

* **Worth attacking (~5.5 ms/token):** the five serialized dependent round trips per
  boundary. A sentinel-in-the-data publish (the `[gate-sentinel]` note in interp.hip) removes
  the producer's counter release AND the consumer's counter observe, i.e. two of the five,
  plus one poll granularity — call it 0.8-1.0 us of the 2.13. Software-pipelining the CLAIM
  (issue the next `fetch_add` before the current body, the cursor is monotonic so an early
  reservation is sound) removes a sixth round trip that sits just off this chain.
* **Worth attacking (~14 ms/token):** packing. Perfect packing of the SAME work is 124 us
  against a 348 us layer; the arms above move the busy time and barely move the efficiency
  (35.7% -> 30.9%). 63.1% of in-packet CU time is dependency wait and the 8 co-resident
  expert slots start up to 46 us apart because their workgroups are not free yet, not
  because they are waiting on a counter.
* **NOT worth attacking:** XCD-local gating that skips cache maintenance (0.31 ms/token
  ceiling), scope narrowing (0 ms; SYSTEM is +9% on an atomic and +0% on maintenance here),
  per-XCD counter fan-out beyond what exists (every hot line is already under ~38 claimants,
  on the flat part of the contention curve), and batching maintenance across packets.
  **The interpreter does not need an MI300X-specific coherence redesign.**

### And the largest single lever in decode is still a kernel

With `lgx2` in, the biggest `busy` row is no longer the routed DOWN (3256 CU-us/layer) but
**`Gemv` at 11285 CU-us/layer** — three packets, of which the shared-expert down projection
is N=6144 K=256, i.e. **the same narrow-K lane defect in the bf16 `gemv_rows` body** that
`PLOW_MOE_DEC_LG` just fixed in the fp8 one. §8.3 of `glm52-decode-gemv-aiter.md` predicted
this and deferred it because "the floor stands". The floor does not stand.

---

## 7. A/B — `scripts/bench_speed.sh`, interleaved, on the serving path

Objects: `hsaco_glm17` with ONLY the six `interp_decode*` rows replaced, built by the SAME
`scripts/build_gfx942.sh` invocation differing only in the two `-D` flags. Same
`model.pkt` (kernel-only axis). Rounds A/B/A/B/A/B so the control's own round-to-round
spread is measured in the same session. `IN_LENS="1024 4096" CONCS=1 NPROMPT=8 OUTLEN=128`,
port 8195, lock held throughout, no foreign `plowrt` resident, fresh bind + temp-0 'Paris'
coherence gate on every one of the six server starts (**all six PASS**).

**TPOT, ms/token:**

| in_tok | arm | r1 | r2 | r3 | mean | ctl spread | vs ctl |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1024 | ctl | 28.98 | 28.94 | 28.95 | 28.957 | **0.04 (0.14%)** | — |
| 1024 | **lgx2** | 26.64 | 26.61 | 27.03 | **26.760** | | **−7.6%** |
| 4096 | ctl | 31.28 | 31.27 | 31.32 | 31.290 | **0.05 (0.16%)** | — |
| 4096 | **lgx2** | 28.96 | 28.98 | 29.10 | **29.013** | | **−7.3%** |

**TTFT, ms (median of 8):** 1024 — ctl 349.1 / 352.6 / 352.4 vs lgx2 352.5 / 351.2 / 352.2;
4096 — ctl 1001.0 / 1007.4 / 1005.8 vs lgx2 1007.5 / 1006.4 / 1005.9. **Unmoved, by
construction**: this is a decode-only axis and prefill runs `interp_prefill_gq.elf` /
`interp_flash_gq.elf`, neither of which the flags touch. That split is itself a consistency
check on the attribution.

The effect is **50x the control's own round-to-round spread** at both contexts, and every
individual `lgx2` round beats every individual `ctl` round with no distribution overlap.
`amd-bench`, a different instrument on the same objects, reads −8.2% (§5). The box's ±20%
DVFS drift shows up as the occasional whole-run outlier (§5 r2/r1) and not as within-round
noise; the served harness saw none in six starts.

### Safety

This arm changes NO coherence operation, no fence, no scope and no counter. It is a lane
map and a loop shape inside two op bodies, and it is **bit-identical by construction** —
the `(lane -> k)` fragment set per output row is unchanged, each row's partials are summed
by the same xor-butterfly (the 64-lane `wave_sum`'s leading offsets 32 and 16 were adding
lanes holding exact `+0.0`), and each accumulator in the paired GLU sees the same terms in
the same order. `glm52-decode-gemv-aiter.md` §3 proves both halves off-device (200,000
random reductions, 0 differing bit patterns; the `(slice, wave, f, u, grp)` row walk
simulated at six geometries, every output row written exactly once). The only representable
difference is the sign of a zero, which the f32 `MoeCombine` cannot distinguish.

**Evidence, not just argument:** `amd-bench` reports all 8 ranks token-identical on every
step of every run, and the arm samples token **347** — the control's token — on all 8 ranks
of all three rounds.

### Serve gate

`plowrt serve` on each arm's asset (same `model.pkt`, only the six `interp_decode*` objects
differ), real prefill, temp 0, the three canonical prompts **plus a ~14.7k-token
long-context prompt** — because the one failure mode a bit-identity claim could hide is a
data-dependent stale read that a short prompt never reaches. **All four answers are
CHARACTER-IDENTICAL between the arms:**

```
--------- GATE arm=ctl | lgx2      (identical output, both arms)
model: glm-5.2-fp8
Q: What is the capital of France?
A: The capital of France is Paris.
Q: Compute 17*23
A: 17 * 23 = 391
Q: Name the chemical symbol for gold and one common use
A: * **Chemical Symbol:** Au
   * **Common Use:** Jewelry making (though it is also heavily used in electronics due to
     its excellent conductivity and resistance to corrosion).
Q: [58935 chars ~14733 tok long-context]
A: The filler text was about distributed GPU inference, memory coherence, and scheduling.
```

Plus `bench_speed.sh`'s own 'Paris' gate on all six server starts, and all 8 ranks
token-identical on every step of all 15 `amd-bench` runs.

### Which half of the arm carries it — `LG` alone is almost all of it

Three arms, three interleaved rounds, `amd-bench --tp 8 --ctx 1024 --steps 24`:

| round | ctl2 | `LG` only | `LG`+`X2` |
|---|--:|--:|--:|
| 1 | 28.887 | 26.788 | 28.705* |
| 2 | 29.139 | 26.770 | 26.534 |
| 3 | 28.928 | 26.747 | 26.519 |
| median | 28.928 | **26.770 (-7.5%)** | **26.527 (-8.3%)** |

(*a whole-run DVFS outlier; `LG`'s three rounds span 0.041 ms.) **`PLOW_MOE_DEC_LG` is the
arm.** `PLOW_MOE_DEC_X2` adds ~0.24 ms (0.9%) on top and its own census row moves the wrong
way (GLU `busy` 4386 -> 4571 CU-us/layer), so it is at the edge of what this box can
resolve. Ship `LG`; `X2` needs its own gate.

**Shipped: `PLOW_MOE_DEC_LG` is now DEFAULT ON in `scripts/build_gfx942.sh`** (opt out with
`PLOW_MOE_DEC_LG=0`); `PLOW_MOE_DEC_X2` stays opt-in. No blob change — this is a kernel-only
axis on the best-config emit, and the canonical recipe is unchanged:

```bash
env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 bash scripts/build_gfx942.sh <outdir>
```

The default flip is verified in BOTH directions on the instruction mix of
`interp_decode_gq.elf`: default-on is identical to an explicit `PLOW_MOE_DEC_LG=1` build, and
`PLOW_MOE_DEC_LG=0` is identical to the shipped `hsaco_glm17/interp_decode_gq.elf`. The flag
lands on `AX_DECODE`, which reaches all six `interp_decode*` rows, so unlike the campaign that
first measured it this needs no `PLOW_ROWS_ONLY` at all.

---

## 8. Answers to the three questions the coordinator raised

**(a) `gate_ag`'s 304 signallers — VERIFIED, PREFILL-ONLY, AND NOT MINE TO CLAIM.**
The ambiguity in the sibling's note resolves decisively from the blob itself, at zero GPU
cost. `gate_ag` is `i[4]` of `PLOW_DOP_XREDUCE2`, the two-shot's all-gather rendezvous, and
`XReduceTwoShot` is emitted **156 times in every prefill program and ZERO times in the
decode program**:

```
program T=1     XReduceTwoShot 0    XReduce (one-shot) 156
program T=1024  XReduceTwoShot 156  XReduce (one-shot)   0
```

Decode's collective is the coarse one-shot `d_xreduce_mega`, which takes a single `i[3]`
xctr gate, not the `gate_rs`/`gate_ag` pair. So `gate_ag` fires **156x per prefill launch
and 0x per decode token**: it is worth the sibling's ~8 ms/launch (~2.3% of TTFT @1k,
~0.5% @8k) and **exactly 0.0% of TPOT**. It is therefore NOT recurring per token, the
condition for me to build it is not met, and it appears nowhere in the decode decomposition
above. **I am not claiming it** — it belongs with prefill/collective work, and the sibling's
own ranking (ride it with `PLOW_GLM_XR_BAND` + `PLOW_XR_CUS`, which the band-pipeline agent
is mid-flight on) is the right home. I did not touch `mla.rs:4936`.

The MECHANISM is independently confirmed by §2 on the same silicon: a returning
`global_atomic_add` costs 392 ns when each claimant owns its own 128 B line and **2770 ns
at 304 claimants on one line — 7.1x**, the same order as the sibling's 51.8/8.2 = 6.3x on
the remote system-scope form. Their aggregation-into-word-1 fix moves the signaller count
from 304 to 1 and lands the gate back on the flat part of that curve, which is exactly what
`PLOW_GATE_HIER` did for the local gate. It should work.

**(b) The fabric law (rate linear in workgroup count, 19->304).** Nothing in this
decomposition attributes any cost to per-thread memory depth on the peer path. The only
peer-path terms here are the `xctr` system-scope acquire/release on `PLOW_SE_XCTR` entries
(§1), and they are priced as LATENCY per gate crossing, not as bandwidth. The two numbers
are consistent: §2 measures a SYSTEM-scope returning atomic at 426 ns against AGENT's
392 ns, i.e. the peer path's per-operation cost is nearly the same and what differs is how
many operations are outstanding — which is the sibling's law stated from the other side.

**(c) How much of the 47 us is already hidden by slot concurrency?** This is the right
question and the trace answers it three ways.

* **The KERNEL is fully exposed, once per layer, not eight times.** The 8 routed slots run
  concurrently on disjoint 32-CU sets, so their kernel time contributes ONE slot's worth to
  the layer. Deleting the DOWN kernel entirely shortens the MoE layer 348.6 -> 301.2 us
  (-47.4) and the token by 11.8%, i.e. essentially the whole per-slot packet cost is on the
  layer's critical path. That is why `GLM_GROUP=1` loses 15%: collapsing 16 packets into 2
  does not remove work, it **serialises 8 concurrent slots onto one CU set**, so it trades
  33 packet boundaries (2.13 us each, 70 us/layer) for 8x the exposed kernel time
  (8 x 43 us). The lever was never packet COUNT — §5 and the coordinator's GLM_GROUP result
  are the same finding from two directions, and both are now explained by the same fact:
  **the 47 us was a kernel, so of course deleting the packets that carry it cannot pay.**
* **The PROTOCOL is fully exposed and NOT overlapped.** The serial packet-boundary dead
  time is 70.25 us/layer in the control and **67.29 us/layer with the largest kernel in the
  program deleted** (§6). It does not shrink when the work does, which is the definition of
  exposed. 2.13 us x 33 boundaries x 78 layers = **5.5 ms/token of genuinely critical-path
  protocol**, against ~0.86 ms of protocol measured as a share of aggregate CU-time. The
  aggregate figure is the one that would have been misleading; the boundary figure is real.
* **The 49 us of dispatch skew is structural, not kernel-driven.** Measured across the 8
  co-resident expert-GLU start times, median over the 69 steady MoE layers:

  | arm | GLU start skew across the 8 slots | DOWN end skew |
  |---|--:|--:|
  | ctl2 | **47.0 us** (45.4-49.2) | 35.4 us |
  | lgx2 | 43.6 (41.4-46.8) | 34.0 |
  | abl2 (DOWN kernel retired) | 37.5 (35.2-40.7) | 28.3 |

  Deleting the single largest kernel in the program removes only 9.5 us of 47.0. The skew
  is not the experts waiting on each other's work; it is their workgroups not being FREE —
  a claim-order property of one monotonic per-domain cursor draining an op-major stream.
  **That is the claim-path target**, and it is worth ~47 us of a 348 us layer (13%),
  independent of and additive to the 70 us of boundary protocol.

**(d) On fine-grained deps.** Noted and it does not change anything here: this decomposition
reads `wait_len` off the shipped stream entries and prices the coarse gate that actually
runs. Nothing proposed above needs `Dep::Fine`.

---

## 9. Provenance

Branch `protocol-xcd`, forked from `worktree-glm52-bringup` @ **`a5b0423`** (before the
decode-gemv / moe-design-review / moe-innerloop-audit / coll-tune / gemm-shape merges).
Not rebased. Asset `/workspace/assets/gfx942/glm52-tp8-final`, objects `hsaco_glm17` with
the six `interp_decode*` rows rebuilt per arm; the control rows are instruction-mix-identical
to the shipped ones. Recipe:

```bash
export PATH=/nix/var/nix/profiles/default/bin:/root/.nix-profile/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib ROCM_PATH=/opt/rocm-7.2.4 \
       HIP_PATH=/opt/rocm-7.2.4 ROCM_HOME=/opt/rocm-7.2.4 PLOW_HIPCC=/opt/rocm-7.2.4/bin/hipcc

# objects (OUTSIDE nix) — NOTE the row filter, this is the whole point of section 0
env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 PLOW_ROWS_ONLY=interp_decode \
    [PLOW_TRACE_PHASE=1] [PLOW_MOE_DEC_LG=1] [PLOW_MOE_DEC_X2=1] [PLOW_MOE_DEC_ABL=2] \
    bash scripts/build_gfx942.sh <outdir>

# unit costs
/opt/rocm-7.2.4/bin/hipcc -O3 --offload-arch=gfx942 -o /tmp/pktproto \
    perf-data/plow-gfx942/probes/pktproto.hip && /tmp/pktproto 4000

# census / phases
./target/release/plowrt disasm <asset>/model.pkt --program 1 | grep '^#' > dec.txt
PLOW_MLA_PF_V2=1 PLOW_TRACE_RAW=/tmp/t.bin nix develop . --command ./target/release/plowrt \
  amd-bench --blob <asset>/model.pkt --hsaco <objdir> \
    --checkpoint /workspace/models/GLM-5.2-plow-lite --tp 8 --ctx 1024 --steps 24
python3 scripts/glm52_decode_census.py dec.txt /tmp/t.bin

# A/B + gate
PLOW_MLA_PF_V2=1 IN_LENS="1024 4096" CONCS=1 NPROMPT=8 OUTLEN=128 \
  PLOWRT_BIN=$PWD/target/release/plowrt bash scripts/bench_speed.sh <asset> 8195 auto
```
