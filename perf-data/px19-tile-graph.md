# PX-19 — the tile graph already ships. Co-location and paged SRAM cannot pay on this cell.

RTX 5090 (sm_120a, 170 SM, 96 MiB **unified** L2, 101,376 B smem optin) · Gemma-4-12B-it,
w8a8 fp8 weights · single-block assets, layer **4 (sliding)** and layer **5 (full)** ·
`crates/plowrt/examples/tilegraph_stat.rs` (added by this note) + `block_run bench` under
`perf-data/tools/gpulease`.

Companion to `px18-egglog-wholemodel.md` (the compiler/graph axis) and `px9-gemm-body.md`
(the kernel axis). **This is a NEGATIVE result delivered before implementation, at the
scoping gate.** No serving-path code was changed; see §7.

---

## Question

Take the graph to **tile** granularity, fuse tiles by **co-location** so intermediates stay
in SRAM, add a **paged SRAM** arena, and pick the right **counter granularity** — the
MPK/`ttGraph` programme (arXiv 2512.22219). Size it before building it.

**The answer is that item 1 already exists, and items 2–4 are each independently dead on
this hardware.** The arithmetic is in §2–§5 and it is measured, not assumed.

---

## 1. The tile graph is not missing. It is what `--emit devblob` already produces.

`StreamEnt` (`crates/packet/src/dev.rs:853`) is `{inst, slice, wait_ofs, succ_ofs, wait_len,
succ_len, flags, seg}` — one record per **`(op, slice)` work item**, with per-task dependency
edges. The matmul bodies then stride the output tile space by slice
(`runtime/nvidia/op_gemm.cuh:902`, `:989`, `:1132`, `:1213`):

```c
for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
    const int tm = (tile / tiles_n) * PGM_BM;
    const int tn = (tile % tiles_n) * PGM_BN;
```

Measured on the emitted blobs (`tilegraph_stat`, T=1024, one layer, `n_cu=170`):

| block | insts (ops) | **tasks (StreamEnts)** | tasks/CU | GEMM output tiles |
|---|---|---|---|---|
| L4 sliding | 19 | **2,890** | 17 (min = max) | 1,712 |
| L5 full | 18 | **2,890** | 17 (min = max) | 1,984 |

At 48 layers that is **138,720 tasks per prefill chunk**. This is an SM-level task graph with
per-task edges — MPK's `ttGraph` in all but name. There is nothing to build here, and building
a second tile representation would rediscover the shipped one.

---

## 2. Co-location has nowhere to land: `stream_ofs[cu]` is **dead in the shipped build**

`PLOW_NV_SCHED` defaults to **1** (`runtime/nvidia/interp_sm120.cu:1467-1469`) — the
**global-queue** scheduler. Every block claims work off ONE atomic cursor over the op-major
`gq_stream` (`:1535-1541`):

```c
__syncthreads();
if (threadIdx.x == 0) gq_claim = atomicAdd(cursor, 1u);
```

`blockIdx` selects nothing. The per-CU streams the emitter builds (`stream_ofs`/`stream_len`,
`devbuild.rs:578-585`) are only read by the `PLOW_NV_SCHED=0` arm, which is **not built**. So
today the emitter's `cus[]` vector — the thing a co-location pass would rewrite — has **no path
to hardware at all**.

Switching to the static scheduler is affordable, and this note measures it rather than assuming:

| L4 sliding, B=1, T=1024 | prefill ms | decode µs |
|---|---|---|
| `PLOW_NV_SCHED=1` (shipped, global queue) | **2.36** | 208.72 |
| `PLOW_NV_SCHED=0` (static per-CU streams) | **2.38** | 210.59 |

0.8% / 0.9% — inside the run-to-run spread (prefill reproduced 2.35 / 2.36 / 2.37 / 2.38 across
four leases, 1.3% total). So the mechanism could be turned on. **It still buys nothing**, for a
reason that is architectural:

* `blockIdx → physical SM` is not controllable. The source says so where it tries:
  *"vs blockIdx, which the HW scheduler maps arbitrarily to SMs"* (`interp_sm120.cu:1500-1504`).
* The only in-tree fix, `PLOW_NV_PLACE_DISPATCH`, needs `-DPLOW_NV_L2_SMS=<SMs per L2
  partition>` and is documented as *"MUST be built and measured on a partitioned GPU
  (H100/B200/MI300/MI350)"*. **The 5090 has one unified 96 MiB L2** — the repo's own hardware
  spec says so: `RTX_5090 { l2: Bytes::mib(96), l2_partitioning: None }`
  (`crates/hwspec/src/nvidia/blackwell.rs:118-137`), against `partition_count: 8` for both H100
  and GB100. There is no L2 domain on this cell to co-locate into.

That leaves exactly one locality tier co-location could target: the SM's own shared memory.
§3 measures whether anything fits in it.

---

## 3. **0 of 23 producer→consumer handoffs fit in SRAM.** The smallest misses by 1.3×.

For a co-located pair, the value must stay in SMEM instead of round-tripping. So the question
is: how many bytes of the producer's output does ONE consumer work item read? Measured over
every edge of the real emitted graph (`tilegraph_stat`, T=1024):

**L4 (sliding)** — 21 edges, 12 analysable:

| edge (tensor) | producer tiles fanned in | handoff | vs 101,376 B cap | max busy CU |
|---|---|---|---|---|
| QuantFp8 → q_proj (`act.xqh`) | 170 | 480 KiB | **4.8×** | 8 |
| QuantFp8 → k_proj (`act.xqh`) | 170 | 480 KiB | 4.8× | 8 |
| QuantFp8 → v_proj (`act.xqh`) | 170 | 480 KiB | 4.8× | 8 |
| q_proj → HeadNormRope (`act.qg`) | 32 | 1,024 KiB | 10.3× | 8 |
| k_proj → HeadNormRope (`act.kg`) | 16 | 512 KiB | 5.2× | 8 |
| v_proj → HeadNormRope (`act.vg`) | 16 | 512 KiB | 5.2× | 8 |
| QuantFp8 → o_proj (`act.xqo`) | 170 | 512 KiB | 5.2× | 8 |
| o_proj → NormResidual (`act.og`) | 30 | 960 KiB | 9.7× | 8 |
| QuantFp8 → gate\|up (`act.xqh`) | 170 | 480 KiB | 4.8× | 8 |
| gate\|up → QuantFp8 (`act.fu`) | 120 | **3,840 KiB** | **38.8×** | 8 |
| QuantFp8 → down (`act.xqi`) | 170 | 1,920 KiB | 19.4× | 8 |
| down → NormResidual (`act.dg`) | 30 | 960 KiB | 9.7× | 8 |

**L5 (full)** — 20 edges, 11 analysable, same shape; best case 128 KiB (`k_proj → HeadNormRope`,
N=512) which is still **1.3×** over the cap, worst 3,840 KiB.

**Handoffs under the cap: 0 of 12 (sliding), 0 of 11 (full).** Not "few". Zero.

The `max busy CU` column is the second, independent kill. A consumer output tile contracts over
the producer's **entire feature axis**, so pinning its producers to one CU means one CU owns a
whole row block. At T=1024 with BM=128 there are only **8 row blocks**, so the machine drops to
**8 of 170 SMs = 4.7%**. Buying an SRAM handoff costs 21× the machine.

### The escape hatch is worse than the disease

The handoff is `BM × K × elem`, so shrinking `BM` makes it fit. It also makes the matmul re-read
its whole weight once per M-tile. `BM_free` is the largest row block that fits **after** the
body's own `cp.async` staging buffers (`PGM_ARENA_BF16` = 61,440 B plain, doubled for GLU) come
out of the same arena:

| matmul (L4) | K | BM under the cap | **BM with staging** | **weight traffic ×** |
|---|---|---|---|---|
| q_proj | 3840 | 26 | 18 | 7.1× |
| k_proj / v_proj | 3840 | 26 | 18 | 7.1× |
| o_proj | 4096 | 24 | 17 | 7.6× |
| gate\|up (GLU) | 3840 | 26 | 10 | **12.9×** |
| **down_proj** | 15360 | **6** | **4** | **32.0×** |

Weights are ~78% of prefill bytes at T=1024 (216 MB/layer fp8 vs ~60 MB of activations). Paying
7–32× on the dominant term to delete part of the minority term is not a close call. And px9
already closed the door from the other side: `PGM_BN=256` raised arithmetic intensity 1.33× and
measured **0.92–0.96×**, i.e. *slower* — bytes through L2 are not the wall on this cell.

---

## 4. Paged SRAM cannot buy occupancy, because **registers bind first**

The brief's premise was that the prefill object's arena is a co-limiter on occupancy 1. Measured
on this worktree (`-Xptxas -v`, `sm_120a`):

| object | registers | spill | static smem | dynamic arena |
|---|---|---|---|---|
| `interp_sm120` (decode) | **241** | 0 | 2,192 B | 34,432 B |
| `interp_sm120_pf` (prefill) | **238** | 0 | 2,320 B | **81,664 B** |

238 regs × 256 threads = **60,928 of the SM's 65,536**. Two blocks would need 121,856. Occupancy
2 requires **≤128 registers** — a 1.87× gap that **no arena change closes**. At a 0-byte arena
the object is still occupancy 1.

Two corrections fall out:

1. **81,664 B, not 89,104.** Measured by compile-time probe on `PLOW_NV_ARENA_FLOATS`. 89,104 is
   the **fp8-KV** prefill object (px8's table), a different build.
2. **Flash sets the arena, not the GEMM.** The union's members are flash-prefill (hd256 generic)
   = 81,664 B vs the GEMM staging `PGM_ARENA_BF16` = 61,440 B. Paging every GEMM tile buffer to
   **zero** leaves the arena at 81,664 B. The one thing a per-tile paged arena would attack is
   not the term that sets the maximum.

---

## 5. Counter granularity: measured, and the whole gate protocol is 2.0% of prefill

The `SE_FINE` machinery is real, but on the emitted blocks it is **not used**:

* `SE_FINE tasks: 0 of 2,890` in both blocks — `select_granularity` downgrades every fine edge
  (the `collapse` theorem, `Plow/CounterGranularity.lean`).
* The sm_120 loader **refuses** fine programs outright: `check_coarse_single_segment`
  (`crates/plowrt/src/asset/devblob.rs:391-402`).
* `sefine-decode.md` already measured forced-fine on decode: **0% recovered, 0.25–0.67% slower**.

So the counter-granularity question is not "would co-location move the coarse/fine decision" but
"how big is the whole gate bill". Measured directly with the body-less build
(`PLOW_NV_SKELETON=1`: same dispatch, same gates, same counters, **no op bodies**):

| block, B=1 T=1024 | real | skeleton (dispatch + gate floor) | floor share |
|---|---|---|---|
| L4 sliding, **prefill** | 2.36 ms | **0.047 ms** | **2.0%** |
| L5 full, **prefill** | 2.78 ms | **0.047 ms** | **1.7%** |
| L4 sliding, **decode** | 208.72 µs | **22.50 µs** | **10.8%** |

The decode figure independently reproduces the sibling's ~12%. The prefill figure is the one
that matters: **the entire interpreter — 2,890 atomic task claims, 2,890 gate evaluations,
2,890 release signals, 19 counters — costs 2.0% of a prefill layer.**

---

## 6. The estimate, with the arithmetic

81% of plow's wall is prefill, 19% decode (px12).

**Scheduler-side items (co-location assignment, cross-task pipelining, counter granularity).**
Their entire addressable cost is the dispatch/gate floor of §5. Ceiling if it became **free**:

```
0.81 × 2.0%  +  0.19 × 10.8%  =  1.6%  +  2.1%  =  3.7% of end-to-end wall
```

That is the ceiling on making dispatch *vanish*. Cross-task software pipelining hides the *next*
task's staging behind the current task's compute; it cannot remove the atomic claim, the
`__syncthreads` pair, or the release. A generous half of the floor is **~1.9%**.

**Data-locality item (SRAM residency of intermediates).** **0%**, and not by estimate: 0 of 23
handoffs fit in SMEM (§3), and the 5090 has no L2 partition to fall back to (§2). The BM-shrink
route costs 7.1–32.0× on the term that is 78% of the bytes.

**Both are below the cell's 3% reproducibility band** (px12 §3), and the second is not "small",
it is structurally zero. **Scoping gate: STOP.** Nothing was implemented.

Additionally, the change would have had to cross `plowc` + `devgen` + `packet` + the interpreter,
and would have had to **flip the default scheduler** (`PLOW_NV_SCHED=1 → 0`) to have any effect
at all — a change that cannot be gated off byte-identically at the cubin level, because it is a
different compiled object.

---

## 7. Results

1. **The tile graph already ships.** `StreamEnt` is one record per `(op, slice)`; a single
   Gemma-4-12B layer at T=1024 emits **2,890 tasks** with per-task edges (**138,720** per prefill
   chunk). The brief's premise that plow emits op-level packets over whole tensors is wrong at
   the wire level.
2. **Co-location has no mechanism.** `PLOW_NV_SCHED` defaults to the **global queue**; the
   emitter's per-CU streams are dead code in the shipped cubin. Static scheduling costs only 0.8%
   (2.38 vs 2.36 ms) so it *could* be enabled — but `blockIdx→SM` is uncontrollable and the 5090
   has a **single unified L2**, so there is no locality tier between SMEM and L2.
3. **0 of 23 producer→consumer handoffs fit in shared memory** — 128 KiB … 3,840 KiB against a
   101,376 B cap, i.e. **1.3× … 38.8× over**. Co-location would also idle **162 of 170 SMs**
   (only 8 row blocks exist at T=1024, BM=128). Shrinking BM to fit costs **7.1×–32.0×** weight
   traffic.
4. **Paged SRAM cannot buy occupancy.** Prefill is **238 registers** — occupancy 2 needs ≤128, a
   1.87× gap independent of the arena. And the arena (**81,664 B**, measured) is set by
   flash-prefill (81,664) not the GEMM staging (61,440), so paging GEMM tiles moves it by **0 B**.
5. **The whole dispatch + gate protocol is 2.0% of prefill** (skeleton 0.047 ms vs 2.36 ms) and
   10.8% of decode. Ceiling on every scheduler-side idea combined, if dispatch became free:
   **3.7% of the wall**. Below the 3% band once halved for realism.
6. **`SE_FINE` is emitted 0 times** on both blocks and the sm_120 loader refuses fine programs
   anyway, so co-location could not have changed the coarse/fine decision even in principle.

### Bugs found, recorded

1. **The campaign tip does not compile.** `crates/plowrt/src/exec/gpu.rs:2506` carries committed
   `<<<<<<< HEAD` / `=======` / `>>>>>>>` markers from merge `a31d568`
   (merged from another worktree into the completion-tokens branch).
   `cargo build -p plowrt` fails with *"encountered diff marker"*. The conflict is **comment-only
   — no code differs**. Fixed here by keeping both facts in one sentence. Any agent that has not
   built `plowrt` since that merge has not noticed.
2. **`PLOW_NV_SKEL_PAD` is dead.** The skeleton build pads static smem to pin occupancy at 1
   (`interp_sm120.cu:1481-1489`), but nvcc eliminates the array — `warning #550-D: variable
   "skel_occ_pad" was set but never used`, and ptxas reports **4 bytes smem at both `PAD=1` and
   `PAD=48`**. The consequence is not silent: the skeleton decode object launches at occupancy 2
   and dies on the grid gate (*"interpreter grid 340 (2/SM × 170 SMs) != packet n_cu 170"*).
   Work around it with `PLOW_NV_SMEM=65536`, which is what §5's decode row used. Not fixed here
   (it is a measurement-only build path).
3. **The 89,104 B prefill-arena figure is being quoted for the wrong object.** It is the fp8-KV
   build; the shipped bf16-KV prefill object is **81,664 B**. Both are far from being the
   occupancy limiter (§4), so the distinction only matters to anyone sizing an arena change.

---

## 8. Gates

| gate | result |
|---|---|
| tile-graph representation measured on the real emitted blob | **PASSED** — §1, `crates/plowrt/examples/tilegraph_stat.rs`, both blocks |
| SRAM-handoff feasibility measured, not assumed | **PASSED** — §3, all 41 dataflow edges of both blocks walked, 23 tile-analysable, against the device's own 101,376 B optin |
| occupancy limiter identified from `-Xptxas -v`, not argument | **PASSED** — §4, 238/241 registers, 0 spill, both objects rebuilt in this worktree |
| arena value measured, not quoted | **PASSED** — §4, compile-time probe on `PLOW_NV_ARENA_FLOATS` → 81,664 B |
| gate/dispatch cost measured end to end | **PASSED** — §5, `PLOW_NV_SKELETON=1` A/B on both blocks, prefill and decode |
| **single-block measurement before any whole-model claim** | **PASSED** — §2/§5, one sliding (L4) and one full (L5) block, `block_run bench`, all four arms |
| scheduler A/B (global queue vs static per-CU streams) | **PASSED** — §2, 2.36 vs 2.38 ms |
| reproducibility | **PASSED** — L4 prefill 2.35 / 2.36 / 2.37 / 2.38 ms across four separate leases = **1.3%** spread, inside the 3% band |
| GPU exclusive | **ENFORCED** — every run under `perf-data/tools/gpulease`; another run was active on the card and the lease serialised it |
| **byte-identical blob when the feature is off** | **PASSED** — nothing was implemented, so there is no feature. The working tree touches exactly two files, both in `plowrt` (a new example + a comment-only conflict fix); `devgen`/`packet`/`plowc`/`runtime` are untouched. Re-emit hash: `b3c03f8c…011e9` for both the original and a fresh emit of `blk-slide` |
| **byte-identical cubins** | **PASSED, structurally** — no `.cu`/`.cuh` was modified (`git status` shows zero files under `runtime/`) |
| `cargo test --workspace` | **PASSED** — **95 suites, 666 tests, 0 failures** |
| greedy-token parity at fixed chunk | **NOT RUN — vacuous.** No serving-path change exists to perturb the token stream; the blob is byte-identical |
| end-to-end 127k cell | **NOT RUN, deliberately** — no serving-path change. Spending a shared card to re-measure a byte-identical blob would burn contended GPU time to reproduce noise |
| co-location implementation | **NOT BUILT, deliberately** — §6. Estimated at 0% for the data half and ≤1.9% for the scheduler half, both under the 3% band |
| paged-SRAM allocator + lifetime rules | **NOT DESIGNED PAST THE GATE** — §3 shows the allocator was never the hard part; nothing fits in the arena at any lifetime policy |
| cross-task software pipelining | **NOT BUILT** — bounded by §5's 2.0% prefill dispatch floor, which it can only partly recover |
| new opcodes + CPU reference arms | **NOT ADDED** — none is warranted (§7.3–7.5); consistent with px18's Q3 |
| L2-domain placement (`PLOW_NV_PLACE_DISPATCH`) | **NOT APPLICABLE ON THIS CELL** — needs a partitioned L2; `hwspec`'s own `RTX_5090` declares `l2_partitioning: None` |

---

## 9. Recommendation

1. **Correct the README.** `README.md:125-132` says plow emits *"op-level packets … there is
   **no per-weight-tile packet**"*. The second half is true of *weights*; the first half misleads
   about *work*, and it cost this note (and its brief) a full investigation. `StreamEnt.slice`
   is the tile. One-paragraph fix, highest value-per-byte item here.
2. **Fix the campaign tip's build** (§7 bug 1) before another agent branches off it.
3. **The leverage is still where px9 left it.** 81% of the wall is prefill, 98% of prefill FLOPs
   are `linear`, and plow's w8a8 GEMM is at 61–66% of a ceiling cuBLASLt reaches at 95–99%. The
   gap is the `cp.async` staging path inside the op body. That is 30% on the dominant term. The
   scheduler above it is worth 2.0%. **Do not spend the next agent on the scheduler.**
4. If MPK's paged-shared-memory idea is ever revisited, it needs a model whose *intermediates*
   are small relative to SMEM. Gemma-4-12B's are 5–39× too large at the tile shape the tensor
   cores need. That is a property of the model and the cap, not of plow's compiler.
