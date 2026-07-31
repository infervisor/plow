# HIER2 on K3: the ceiling is −13.8%, and the thing that blocked it is already built

**Measured 2026-07-30**, 8× gfx950 (MI355X), K3 93 layers TP8, `/home/lava/models/k3_base/model.pkt`,
ctx 32000, `--steps 200`, UNBOUND weights, every run under `perf-data/harness/gpulease`,
`foreign-during=0` on every run quoted.

**§0-BENCH.** `plowrt amd-bench` with no `--checkpoint` — the schedule, the counters and the
memory traffic are real, the tokens are not. That is the instrument
`perf-data/amd-bench-ab-order-bias.md` prescribes for sub-1-ms work, because a bound run re-reads
195 GiB/rank and the second of any pair is penalised 5–10 ms.

---

## 0. TL;DR

| arm | ms/token | vs control |
|---|--:|--:|
| control (`ba_ctl`) | 33.621 / 37.008 / 34.876 / 33.603 | — |
| **`PLOW_GATE_HIER_CEIL`** | **28.956 / 28.959 / 28.950 / 28.958** (sd **0.003**) | **−4.65 ms, −13.8%** |

Against the two clean controls (33.621, 33.603). The HIER2 arm's four runs span **9 µs**, so the
effect is three orders of magnitude above its own noise. This is the **unsafe ceiling** — the
saving with the ordering omitted — so the sound version lands under it.

---

## 1. Where the per-packet cost actually is

`runtime/bench/ctr_convergence.hip`, re-measured under lease, control run twice (13.120 / 13.200).
One empty b=256 packet, first-arrive to last-end:

| arm | packet µs | signal µs |
|---|--:|--:|
| BASELINE — 1 counter, release signal + agent acquire | **13.16** | 0.070 |
| FAN=32 — 32 counters | 10.20 | 0.080 |
| **FAN=256 — one counter per workgroup, zero contention** | **8.95** | 0.080 |
| NOACQ — drop `buffer_inv` (unsafe) | 9.16 | 0.080 |
| RELAXSIG — drop `buffer_wbl2` (unsafe) | 5.49 | 0.070 |
| **HIER2 — per-XCD leader does the maintenance** | **3.46** | 0.140 |

**The atomic is never the cost: 0.07–0.14 µs in every arm.** What costs is that all 256 workgroups
issue `buffer_wbl2` and `buffer_inv`, which are per-L2, so each XCD's L2 performs the same
writeback and the same invalidate 32 times and they serialise.

### The counter-per-workgroup design is FAN=256, and it is measured

A "completion flag per packet per workgroup, consumer scans the list" scheme is exactly the
`FAN=256` arm: each producer signals its OWN counter at threshold 1, and each consumer scans all
256 with one thread per counter (`for (w = tid; w < nfan; w += THREADS)`) — the same shape
`interp.hip`'s gate already uses for its wait list. It is **worth 13.16 → 8.95 µs (−32%)**, it
removes the 4.4 µs of counter-line contention, and the scan is genuinely cheap because it is
parallel across threads.

But it does **not** touch the fences, which are the remaining 8.95 µs, and **HIER2 is 2.6× better
than it** (3.46 vs 8.95). The two are not alternatives at the same level: FAN removes contention,
HIER2 removes duplicated cache maintenance. Semantically a threshold-counter and a per-workgroup
flag array are the same statement ("all `b` slices of `p` are done"); they differ only in whether
the signal side contends, and the contention is a third of the problem.

---

## 2. Why NOT `PLOW_GATE_SC1`, now that both are measured

Both attack the same term and they do not compose:

* `PLOW_GATE_SC1` makes the writeback **unnecessary** — `sc0 sc1` stores leave no dirty line. It is
  **unsound** (the defect is ordering, not coverage — see `runtime/amd/amd_common.h`) and the
  numerically-correct variant measured **+0.67%**.
* HIER2 makes the writeback **amortised** — one per XCD instead of one per workgroup — and keeps
  the release, so the ordering is untouched.

Under `sc1` stores there is nothing dirty for HIER2's leader writeback to find, so combining them
pays SC1's store cost for a saving HIER2 has already taken. **Drop SC1; keep HIER2.**

---

## 3. The blocker dissolved: `PLOW_L2_PLACE_DISPATCH` already exists

`plans/k3-decode-perf.md` blocks HIER2 on *"the global queue having no per-packet leader"*: a
workgroup claims whatever entry is next, so which slices of a packet land on which XCD — and
therefore the per-(packet, XCD) arrival count the rendezvous needs — is decided at run time.

**`interp.hip:1968` already implements per-XCD queues.** Under `PLOW_L2_PLACE_DISPATCH` all 8
domains drain concurrently inside one launch, each workgroup serving the domain it is *physically*
on, read from `HW_REG_XCC_ID` rather than inferred from `blockIdx.x` — so window `d` is executed
only by XCD `d`'s workgroups **by construction**. Each domain has its own cursor
(`PLOW_CTR(prog.gq_cursor, my_seg)`), so stealing survives *within* a domain. The compiler
stable-sorts `gq_stream` by domain and emits per-domain windows in `gq_seg_ofs`, and the
deadlock-freedom argument for 8 concurrent queues is already written out at `interp.hip:2040`.

**With that dispatch mode, `nper[p][d]` is a static emit-time constant** — the count of entries
with `inst == p` in window `d`. The blocker is a property of the *global* queue, not of the design.

`PLOW_L2_PLACE` on its own measured **ZERO** (39.579 vs 39.628), which is why it is off. That
result is not an argument against it here: locality alone bought nothing because there was no
per-XCD maintenance to exploit. HIER2 is what makes the domain structure load-bearing.

---

## 4. What the sound version needs

Two XCD-local rendezvous, both using `nper[p][d]`, and **neither needs a fence** — all
participants share one L2 partition (Fleet, arXiv 2604.15379, implements exactly this on MI350
and says so explicitly).

```
publish   follower: __syncthreads() retires its stores into THIS XCD's L2, then
                    atomicAdd(&ldn[p][d], 1) RELAXED — no writeback
          leader:   the workgroup whose add returns nper[p][d]-1 issues ONE release RMW on the
                    global counter, bumping it by nper[p][d]. Its buffer_wbl2 writes back the
                    whole L2 — which is exactly the followers' stores — and publishes.
observe   leader:   the workgroup whose atomicAdd(&arr[c][d],1) returns 0 polls the global
                    counter, does ONE buffer_inv for the XCD, then bumps open[c][d]
          follower: waits open[c][d] and does NO invalidate, and never polls the global
                    counter at all — less fabric traffic, not more
```

Global-counter semantics are preserved: each domain's leader bumps by `nper[p][d]`, so the total
still reaches `blocks[p]` and no consumer's threshold changes.

**Cost:** three XCD-local counter arrays (`ldn`, `arr`, `open`) at `n_ops × 8`. For K3 that is
59k slots ≈ 7.6 MB at the current 128 B stride, and it triples the per-token counter re-arm
(315 KB → 1.26 MB); a device-side reset is the obvious answer if the memcpy shows up.

**Risk:** this is the class `plans/fine-counter-deadlock-fix.md` warns about. The leader election
is by return value of an atomic, so exactly one workgroup wins each role, but the publish leader
must not be able to run before a follower has retired — which is what the follower's
`__syncthreads()` before its bump guarantees.

---

## 5. Next step, and the one thing in the way

> **§5 IS SUPERSEDED BY §7 — the block below was cleared the same day.** The refusal was a STALE
> BINARY, not a missing capability: `/home/lava/plow/target/release/plowc` was built 07-29 and
> predates the `K3_FULL` path, so it fell through to the capability report. A `plowc` rebuilt from
> this tree emits K3 fine, with and without `PLOW_L2_PLACE`. Both blobs are measured in §7. Read
> §5 only for the order of work, which still holds.

**Blocked:** `PLOW_L2_PLACE_DISPATCH` refuses a blob that was not emitted with `PLOW_L2_PLACE`
(`devbuild.rs:1650`), and `/home/lava/models/k3_base/model.pkt` is `l2_domains 0,
l2_placed=false`. Re-emitting it on **main fails** — `crates/devgen/src/mla.rs:6287` refuses with
*"kimi_k3: 2 unimplemented capabilities"*, so main cannot currently produce any K3 blob. The
on-disk one came from a branch. **That re-emit is the gating step**, and it is worth doing before
any of §4: an L2-placed blob run on today's interpreter costs one lease and says what domain
dispatch costs on its own at K3's shape, which is the denominator every HIER2 number is against.

Order of work:

1. Re-emit K3 with `PLOW_L2_PLACE=1` (needs a tree whose K3 emitter is not refusing).
2. Price `PLOW_L2_PLACE_DISPATCH` alone against the global queue on that blob — expect a small
   loss (the global queue beat static 39.571 vs 41.657) and confirm it is smaller than 13.8%.
3. Build the §4 rendezvous and gate it against `PLOW_GATE_HIER_CEIL`'s 28.957 ms.
4. Only then consider XCD-affine tensor partitioning, where a producer slice on XCD `d` writes
   what the consumer slice on XCD `d` reads and the chain never leaves that L2. Reductions
   (RMS over the full hidden, the all-reduces) break affinity by construction, so it is a hybrid,
   not a mode.

---

## 6. Reproduce

```bash
# microbench (one GPU)
hipcc --offload-arch=gfx950 -O3 -std=c++17 -w [-DFAN=256|-DHIER2|-DRELAXSIG|-DNOACQ] \
      runtime/bench/ctr_convergence.hip -o ctrconv
perf-data/harness/gpulease -n 1 ctrconv sg render -c './ctrconv 400'

# the arm (8 GPUs, unbound, interleaved with order reversal)
cmake -S runtime -B ba_hier <flags> -DPLOW_HSACO_EXTRA_DEFINES="-DPLOW_GATE_HIER_CEIL=1"
cmake --build ba_hier --target gfx950_hsaco -j 32
perf-data/harness/gpulease -n 8 hier sg render -c "nix develop <repo> --command \
  plowrt amd-bench --blob /home/lava/models/k3_base/model.pkt --hsaco ba_hier/hsaco \
  --steps 200 --ctx 32000 --tp 8"
```

**Trap that cost this campaign its first day of numbers:** a bare `flock /tmp/plow_gpu.lock`
serialises against other runs but does **not** take the gpulease lease, so it neither waits for
nor warns about a concurrent agent's campaign. Timed that way, one arm measured **46.418 ms at
position 2 and 33.802 at position 1**. Use `perf-data/harness/gpulease`; it audits and it logs.

---

# 7. LOCALITY RE-EMITTED AND MEASURED (2026-07-30, later)

Both blobs re-emitted from the SAME `plowc` (the on-disk `k3_base` predates the transitive
reduction, so it is not a valid control): `k3_ctl` = global queue, `k3_l2` = `PLOW_L2_PLACE=1`.
Both report `counter-graph reduction: 69 of 3038 … 207 wait entries removed` and 2459 decode
instructions. The placed blob emits `8 domains × 32 SM, map round-robin (wg n -> dom n%domains),
skew 0.1%`, and the runtime **refuses** it without `PLOW_L2_PLACE_DISPATCH` — the interlock works.

3 reps, interleaved, order-reversed, leased, `foreign-during=0` on all 12 runs:

| arm | blob | interp | ms/token | sd | vs base |
|---|---|---|--:|--:|--:|
| base | `k3_ctl` | global queue | 33.115 | 0.122 | — |
| l2 | `k3_l2` | `L2_PLACE_DISPATCH` | 32.892 | 0.033 | **−0.223 (−0.67%)** |
| hier | `k3_ctl` | `HIER_CEIL` | 28.304 | **0.002** | **−4.811 (−14.53%)** |
| **l2hier** | `k3_l2` | both | **27.751** | 0.095 | **−5.364 (−16.20%)** |

**They compose** (−0.223 + −4.811 = −5.03 against a measured −5.36) and **per-XCD dispatch is not
a cost** — it is slightly positive, which is the result that matters for the sound HIER2, because
that design needs domain dispatch to make `nper[p][d]` static. The old "`PLOW_L2_PLACE` measured
ZERO" stands as a statement about locality alone; it is not an argument against paying for it here.

## 7.1 A smarter CU assignment cannot help. Measured, on K3.

`PLOW_PLACE_REPORT=1` runs the emitter's own locality census (`devbuild.rs:641`). K3 decode:

```
locality census (2459 ops, 321,996 slices, 8 domains, map round-robin):
  slice-level producer->consumer pairs: 41,871,806
      (100.0% on ALL-TO-ALL edges, where 1/8 = 12.5% is the ceiling for ANY placement)
  same-domain pairs: current 12.50% | greedy pred-affinity 12.50% | per-slice argmax ceiling 12.90%
  greedy moves 280,077/321,996 slices in 1652/2459 ops
```

**100.0% of the pairs are on all-to-all edges.** A consumer slice of `out = W @ x` reads every
producer slice, so under any assignment that keeps the producer balanced exactly `1/domains` of
pairs are same-domain. Round-robin already sits exactly on 12.50%; a greedy predecessor-affinity
pass relocates **87% of the program** and changes locality by **zero**; even the balance-free
per-slice argmax reaches 12.90%. `devbuild.rs`'s own unit tests pin this
(`an_all_to_all_edge_pins_same_domain_locality_at_one_over_domains`,
`the_greedy_pass_moves_most_slices_and_buys_nothing`).

So routing placement through the Rust scheduler — or any smarter assignment — has a **0.40
percentage-point** ceiling it cannot reach. The constraint is the DATAFLOW, not the assignment.
Beating it requires making the edges not all-to-all (partitioning the tensor so a consumer needs
only its own domain's slice, with explicit cross-domain reductions where the math demands them),
which is tensor parallelism inside the GPU and a different project.

## 7.2 Caveat on the ceiling number, stated rather than hidden

`ctr_maint_leader()` elects `blockIdx.x < 8`. For a **wide** packet all 8 XCDs are involved and
blocks 0–7 are among its executors, so the arm pays exactly one maintenance op per XCD — the true
HIER2 cost. For a **narrow** packet under the global queue the executing workgroups may not
include any block < 8, in which case the arm pays ZERO maintenance where real HIER2 would still
pay one. K3 decode has ~1041 narrow packets (468 at b=1, 203 at b=2, 186 at b=14, 184 at b=7)
against 1068 at b=256. At the grid sweep's 1 WG/XCD cost of 1.05 µs, that is up to ~1.1 ms of the
4.81 ms that the sound version will not recover — call the honest expectation **~3.7–4.8 ms**.
Under `L2_PLACE_DISPATCH` the election should read `HW_REG_XCC_ID` (as the dispatch path already
does) rather than `blockIdx.x`, which also removes the dependence on the round-robin map.

---

# 8. THREE BUGS IN THE CEILING INSTRUMENT, FOUND BY AUDIT AND FIXED (2026-07-30, later)

An interpreter-wide dead-code/consistency audit found three defects in `PLOW_GATE_HIER_CEIL`
itself. All three are fixed; the headline moved by less than a point, which is the useful part —
the lever is not resting on any of them.

**8.1 The leader test also gated `xctr_acquire()` — a measurement bug on a TP8 run.**
`xctr_acquire` is the SYSTEM-scope acquire for data arriving over XGMI. HIER2 is about per-L2
maintenance inside one device and `ctr_convergence.hip`'s HIER2 arm has no cross-GPU component at
all, so gating it was never part of the design. It mattered: leaders were `blockIdx.x < 8` while
K3's collectives are b=14 after `xrfit`, so the expected leader count on a collective is
14·8/256 = **0.44** — nearly every cross-GPU acquire was being deleted. Now the leader test
applies to the local acquire only. Worth **0.2 ms**: hier −14.53% → −14.24%.

**8.2 `PLOW_GATE_HIER_CEIL` and `PLOW_GATE_SC1` silently mis-composed.** The HIER_CEIL follower
hand-rolls its relaxed RMW instead of calling `ctr_signal`, so under SC1 it would also skip that
arm's `s_waitcnt vmcnt(0)` — 248 of 256 workgroups bumping their counter without retiring their
stores. Now a `#error` (verified: the pair fails to compile, each alone still builds).

**8.3 The leader was derived from `blockIdx.x`, in the file that argues against exactly that.**
`PLOW_L2_PLACE_DISPATCH`'s own comment: *"Deriving it from blockIdx.x instead would make the whole
feature silently depend on the round-robin map continuing to hold."* Now `blockIdx.x == xcc` with
`xcc` read from `HW_REG_XCC_ID`, so the hardware confirms the map rather than the map being
assumed, and a changed map degrades (an XCD gets no leader) instead of electing two on one XCD.

**And the register read must be HOISTED.** Read per call it lands on the gate and on every
successor signal — twice per packet, 324,940 workgroup-packets per token — and it MEASURED
**0.43 ms** (l2hier 27.849 → 28.271, ~1.3 ns/workgroup-packet, ~8 sd). Bound once at kernel entry
(`s_getreg` count in the object: 2 → 1) it comes back.

## 8.4 The number, with all three fixed

3 reps, interleaved, order-reversed, leased, `foreign-during=0`:

| arm | ms/token | sd | vs base |
|---|--:|--:|--:|
| base | 33.325 | (one 38.7 outlier; clean 33.220 / 33.325) | — |
| **hier** | **28.437** | 0.037 | **−4.888 (−14.67%)** |
| **l2hier** | **28.150** | 0.129 | **−5.175 (−15.53%)** |

Across every variant of the election tried — `blockIdx.x < 8`, `== xcc` per-call, `== xcc` hoisted
— HIER2 lands between **−14.0% and −14.7%** and HIER2+locality between **−14.9% and −16.2%**. The
lever is robust to how the leader is picked, which is what §4's sound version needs to be true.

The §7.2 caveat still stands and is the honest discount: under the global queue a narrow packet
may be executed by no elected leader at all, so the arm pays zero maintenance where real HIER2
pays one. K3 decode has ~1041 narrow packets against 1068 at b=256.

---

# 9. THE SOUND VERSION IS BUILT, AND IT SHIPS

`PLOW_GATE_HIER` (sound) vs `PLOW_GATE_HIER_CEIL` (the unsafe ceiling this document priced),
K3 TP8 ctx 32000, leased, interleaved, `foreign-during=0`:

| arm | ms/token | sd | vs base |
|---|--:|--:|--:|
| base | 33.134 | 0.019 | — |
| **`PLOW_GATE_HIER`** | **29.110** | 0.096 | **−4.024 (−12.14%)** |
| `PLOW_GATE_HIER_CEIL` | 28.226 | 0.075 | −4.908 (−14.81%) — UNSAFE |

**The sound version captures 82% of the ceiling.** The missing 18% is exactly the ordering the
ceiling omits — two XCD-local rendezvous — which is the right shape for that gap to have.
Bound TP8 on real weights: prefill → 17374, all 8 ranks agree, 32-token stream identical to the
control.

## 9.1 Combined with the sharded `lm_head`, which is what actually ships

| arm | ms/token | sd | tok/s |
|---|--:|--:|--:|
| base | 33.088 | 0.042 | 30.2 |
| **`PLOW_GATE_HIER` + `PLOW_L2_PLACE` + `PLOW_K3_SHARD_HEAD`** | **28.876** | 0.046 | **34.6** |

**−4.212 ms, −12.73%, 30.2 → 34.6 tok/s**, token-identical, on a blob and an object set that are
both emitted from this tree.

## 9.2 Two failures worth keeping

* **An `#error` on the preconditions broke 33 of 49 objects.** `PLOW_HSACO_EXTRA_DEFINES` applies
  to EVERY object, including the static-path and flash-class ones built deliberately without the
  global queue. The guards now degrade: an object that cannot support the hierarchy compiles
  without it and never reads `hier_base`.
* **A new blob under a `plowrt` built BEFORE `hier_base` existed faulted the GPU.** The field
  lives in the kernarg's alignment pad, which the older host left uninitialised, so a garbage base
  indexed off the end of the counter region. That is precisely the trap `dev_isa.h` records having
  already cost one debugging session — caught here by the correctness gate rather than shipped.
