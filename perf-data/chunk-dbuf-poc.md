# CHUNK-1 — prefill M-chunk cross-op pipeline POC (double-buffer / cross-SM stagger)

Standalone POC measuring the user's thesis: split prefill M into chunks and let the
CONSUMER op of chunk c overlap the PRODUCER op of chunk c+1, so coarser packets enable
real cross-op pipeline overlap that the per-op-drain interpreter forbids today. Distinct
from the refuted T10 (occ-2 per-segment relaunch, +22..45%) and T11 (packet-merge on the
drain-per-op kernel, ~0). DECODE out of scope.

Hardware: RTX PRO 6000 Blackwell (sm_120a, 188 SMs, L2 128 MB, ~1.6 GHz).
Code: `runtime/nvidia/experiments/chunk_dbuf_poc.cu` (build: plain env, NOT nix).

---

## Stage 0 — GROUND IT (measured)

### 0.1 DevInst = 104 bytes — CONFIRMED
`crates/packet/src/dev.rs` `DevInst`: *"One instruction. Fixed 104-byte stride — no
variable-length records on device."* (the `#[repr(C)]` struct: `op u16, blocks u16,
wait_len u16, succ_len u16, wait_ofs u32, succ_ofs u32, t[8] u32, i[8] u32, f[4] f32, j…`).

### 0.2 Prefill packet + byte breakdown (measured, parsing the shipped .pkt)
Parsed `gemma-4-12b-fp8/{prefill,decode}_b1_s512.pkt` (ver5, host wire format; the device
DevInst stream is `n_packets × 104 B`). One 512-token prefill chunk program (12B, 48 layers):

| program | packets | counters | succ-edges | host .pkt | device DevInst stream |
|---|--:|--:|--:|--:|--:|
| **prefill** b1 s512 (one 512-tok chunk) | **1817** | 1592 | 3692 | 95.6 KB | **1817×104 = 189 KB** |
| decode b1 s512 (one step) | 561 | 381 | 699 | 26.5 KB | 58 KB |

Per-family packet histogram (prefill 512-chunk):

| family | packets | per layer | note |
|---|--:|--:|---|
| DMA (TMA load/store) | 1210 | ~25 | weight/activation tile staging (604 load + 606 store) |
| **GEMM** | **544** | ~11.3 | the tiled prefill matmuls (qkv/o/gate-up/down, M-tiled) |
| FLASH | 48 | 1.0 | one flash-prefill packet per layer |
| ROW (norms) | 14 | ~0.3 | RMSNorm / residual-norm |
| TOKEN | 1 | — | sample |

So a 4k prompt at chunk-512 runs 8 such chunks; the T9c "97 segments/chunk" figure is the
*wave-class segment grouping*, not the raw packet count — the raw device stream is ~1817
DevInsts / 512-chunk. **GEMM family = 544 packets is the bulk of the compute-op boundaries**,
each separated by a coarse counter edge.

### 0.3 The per-op cost T11 named — the arena-drain + acquire-fence gate
`interp_sm120.cu` main loop (verified): every block runs one packet to completion,
`__syncthreads()`, claims the next entry from ONE atomic cursor (`gq_claim = atomicAdd(cursor,1)`,
`PLOW_NV_SCHED=1` GQ), then **gates on a per-consumer COUNTER**:
```
for (w = tid; w < wait_len; w += blockDim.x)
    while (ctr_poll(counters[pw.id]) < pw.threshold) __nanosleep(64);
__syncthreads();
if (tid==0 && wait_len) __threadfence();   // ACQUIRE — load-bearing
```
The dynamic smem **arena is a UNION across op bodies** (`PLOW_NV_PRE_A`): *"each fully
consumes its arena before the next instruction's gate"* — GEMM arena 60 KiB
(PGM_STAGES=3, 128×128×32). Per GEMM packet the block re-stages its A+B operand tiles from
HBM/L2 into the 60 KiB arena; the acquire `__threadfence()` after the gate forbids reading
across the boundary, so there is **no cross-op prefetch**. T11 dispatch floor: ~1 µs/packet
(0.659 ms / 676 pkts on the 31B decode skeleton), *largely overlapped by bodies at real ctx*.

### 0.4 THE decisive mechanism question (coordinator refinement) — ANSWERED by reading the code
**Is the per-op gate a GLOBAL grid-sync barrier, or a per-consumer-counter wait?**

**It is a per-consumer-COUNTER wait — there is NO `grid.sync()` / cooperative barrier in the
interpreter loop.** A block that finishes a packet claims the next and spins ONLY on that
packet's specific `wait` counters (`counters[id] >= threshold`). Different blocks are at
different PCs; nothing forces the whole grid to rendezvous per op.

**BUT** the *threshold* is the producer's full block count, and for UNISEG prefill
`select_granularity` downgrades EVERY edge to **coarse** (T11: "0 fine edges kept, 270
downgraded to coarse"; the interp's fine path `if (flags & PLOW_SE_FINE) __trap()` is stubbed
out). A coarse edge sets `threshold = producer.blocks = n_cu = 188`, so the consumer op cannot
start until the producer op has finished on **every SM**. → The coarse all-SM threshold makes
each op-to-op edge a **de-facto grid-wide op barrier**, even though the *mechanism* is a cheap
per-counter poll.

**Consequence for the thesis:** the chunk-pipeline win is **NOT** "remove a grid.sync" (there
is none) — it is **replace the coarse all-188-block counter edge with per-CHUNK 1:1 counter
edges** so consumer-chunk-c waits on only its chunk's producers (e.g. 47 blocks), letting an
SM-set advance to chunk c's consumer while another SM-set still runs chunk c+1's producer.
The kernel *mechanism* (per-counter poll) ALREADY supports this; what's missing is (a) the
emitter emitting chunk-partitioned packets + 1:1 chunk counters, and (b) un-stubbing the
fine/per-chunk gate path in the interp. **This is a SCHEDULING + EMITTER change, not a GEMM
kernel-body rewrite** — the pivotal cost finding.

---

## Stage 1 — POC (2-op GEMM→GEMM chain, 1:1 M-chunk producer/consumer)

Op-pair: `Y = X·W1ᵀ` (GEMM1) → `Z = Y·W2ᵀ` (GEMM2), 12B prefill shapes
H=3840, M=2048 (representative chunk). GEMM2-chunk-c consumes GEMM1-chunk-c (same M rows) —
the ideal 1:1 chunk dependency. Identical 128×128×32 cp.async mma body as production `d_gemm`.

Three arms, same data, same cooperative launch (grid 188, occ-1):
- **mode 0 SERIAL** — full GEMM1 → `grid.sync()` (models the coarse all-SM op barrier) → full GEMM2.
- **mode 1 CHUNK_GQ** — k M-chunks, per-chunk counters, ONE atomic work cursor (GQ work-stealing).
  GEMM2-chunk-c waits ONLY on counter[c]. Cross-SM overlap, no full-op barrier, no intra-block dbuf.
- **mode 2 CHUNK_STATIC** — per-chunk counters, block→chunk STATICALLY pinned (producer+consumer of
  chunk c on the SAME SM set → GEMM2 reads Y-chunk-c from hot L2). L2-locality vs GQ work-stealing.

### ptxas (all three arms) — occ-1 target MET
| arm | regs | spill | smem arena |
|---|--:|--:|--:|
| mode 0 SERIAL | 124 | 0 | 60 KiB |
| mode 1 CHUNK_GQ | 131 | 0 | 60 KiB (+16 B) |
| mode 2 CHUNK_STATIC | 125 | 0 | 60 KiB |

0 spill, 60 KiB ≤ 100 KiB → **occ-1, grid 188** (matches production `_pf`). Note the cross-SM
arms need **NO** double-buffered arena (each block runs one op-slice at a time; the "double
buffer" is across SMs, per §0.4), so they fit the SAME 60 KiB the serial kernel uses.

### k-sweep (a)-vs-(b), cost-model shapes (H=3840, N1=N2=3840)
Cost-model chunk count (CHUNK-2, "largest chunk whose output fits ~half L2"): M=2048→k2,
M=8192→k8, M=16384→k16. Swept k∈{2,4,8,16} at each M; the cost-model k is **bold**.
`ovlp(µs)` = grid-wide `max(GEMM1-end) − min(GEMM2-start)` via `%globaltimer` (>0 ⇒ real
cross-SM overlap). Δ% vs the SERIAL (coarse grid.sync barrier) baseline.

**M=2048** (cost-model k=**2**) — SERIAL 0.6877 ms/pair, **op-boundary drain 0.7 µs**
| k | GQ Δ% | GQ ovlp µs | STATIC Δ% | STATIC ovlp µs | parity |
|--:|--:|--:|--:|--:|:--:|
| **2** | −0.2 | 0.0 | +0.1 | 3.5 | bit-identical |
| 4 | −0.5 | 3.1 | −0.4 | 8.0 | bit-identical |
| 8 | **+12.9** | 100 | −0.5 | 5.6 | bit-identical |
| 16 | **+12.3** | 105 | −0.7 | 7.4 | bit-identical |

**M=8192** (cost-model k=**8**) — SERIAL 2.4271 ms/pair, **op-boundary drain 0.6 µs**
| k | GQ Δ% | GQ ovlp µs | STATIC Δ% | STATIC ovlp µs | parity |
|--:|--:|--:|--:|--:|:--:|
| 2 | −0.0 | 2.4 | +0.1 | 27 | bit-identical |
| 4 | −0.1 | 1.6 | −0.0 | 27 | bit-identical |
| **8** | **+33.4** | 1033 | +0.0 | 134 | bit-identical |
| 16 | **+33.3** | 1038 | −0.1 | 135 | bit-identical |

**M=16384** (cost-model k=**16**) — SERIAL 4.5777 ms/pair, **op-boundary drain 0.6 µs**
| k | GQ Δ% | GQ ovlp µs | STATIC Δ% | STATIC ovlp µs | parity |
|--:|--:|--:|--:|--:|:--:|
| 2 | −0.4 | 0.0 | −0.4 | 24 | bit-identical |
| 4 | +1.3 | 2.2 | −0.4 | 36 | bit-identical |
| 8 | **+40.2** | 2080 | −0.5 | 145 | bit-identical |
| **16** | **+40.0** | 2084 | +4.0 | 244 | bit-identical |

Reading the table:
1. **The op-boundary bubble is ~0.6 µs** — SERIAL's coarse all-SM `grid.sync` barrier costs
   0.6–0.7 µs against a 0.69–4.6 ms op-pair (<0.03%). **There is essentially no bubble for a
   cross-op pipeline to fill.** The occ-1 GEMM is already cp.async-latency-hidden AND
   tile-balanced (2.5–40 tiles/block), so the op boundary is nearly free.
2. **STATIC (colocated per-chunk counters) produces REAL cross-SM overlap** — `ovlp` grows to
   130–145 µs at M≥8192 (consumer chunk c genuinely starts before producer chunk c+1 ends) —
   **yet wall time moves ~0%** (−0.7…+0.1% at the cost-model k, within run noise). The overlap
   is real but *redundant*: the GEMM already saturates all 188 SMs, so overlapping consumer-c
   with producer-(c+1) only reshuffles work that was already filling the machine — there is no
   idle SM capacity to reclaim.
3. **GQ (work-stealing) REGRESSES** +12.9% (M=2048) → +40% (M=16384) at k≥8, and the regression
   GROWS with M. The single atomic cursor + finer slices + op-major worklist serialize the
   dispatch faster than the tiny overlap helps. GQ is the wrong scheduler for chunked prefill.
4. STATIC is the only non-regressing arm — flat (~0%, ±noise) at every k except k=16/M=16384
   (+4%, over-chunking past the L2-fit point). It never *wins* either.

**Deadlock note (fixed in POC):** the contiguous static placement requires the block→chunk
map `g=⌊b·k/n_cu⌋` and the cu_range `[⌈g·n_cu/k⌉, ⌈(g+1)·n_cu/k⌉)` to use CEIL boundaries;
floor boundaries (the naive `c·n_cu/k`) leak a boundary block into the wrong chunk when
`n_cu%k≠0` (k=8→23.5, k=16→11.75), starving a per-chunk counter → cooperative-grid hang.
The production emitter must derive the block→chunk inverse consistently with the placement.

---

## §A The mechanism, end-to-end (POC ⋈ CHUNK-2)

CHUNK-2's `build_counters(Fine)` emits one threshold-1 counter per consumer chunk on coupled
boundaries (confirmed NOT a global barrier at the scheduler level). This POC confirms the
matching runtime fact by reading `interp_sm120.cu`: **the runtime per-op gate is a
per-consumer-COUNTER poll, NOT a `grid.sync()`** (§0.4). `dev.rs` independently documents the
same and quantifies the cost of the coarse form: *"every op is a full N-way barrier, and a
consumer waits for the SLOWEST producer workgroup… 2.63 ms of a 16.9 ms token waiting for one
straggler after half the machine is done… the straggler is diffuse… so it is recoverable"*
and `SE_FINE` already exists to *"block only on the producer slices that actually feed it."*

⟹ **Both halves are in place: the emitter can emit per-chunk 1:1 edges, and the runtime gate
already supports per-counter waits. The coarse all-188-block threshold is the ONLY thing making
it a de-facto barrier.** The chunk-pipeline win is a scheduling+emitter change (per-chunk edges
+ CU-placement), NOT a GEMM kernel rewrite — the POC uses the *unmodified* production
128×128×32 mma body in every arm.

## §V Verdict — NO-GO for the chunked cross-op double-buffer prefill kernel

**The POC does NOT beat the serialized model on the op-pair — it ties at best (STATIC, ±noise)
and regresses badly at worst (GQ, +13…+40%). NO-GO.**

Honest negative, root-caused (and it matches the p9 doc's prior skepticism and T11):
- **No bubble to fill.** The coarse all-SM op boundary costs **~0.6 µs** (<0.03% of the pair).
  The occ-1 prefill GEMM is already cp.async-latency-hidden and tile-balanced, so the drain the
  thesis targets is already near-zero. The T10/T11 story repeats one level down: the double
  buffer, like occ-2, has nothing to reclaim.
- **Overlap ≠ speedup on a saturated machine.** STATIC *does* pipeline (measured `ovlp`
  130–145 µs of consumer-c running before producer-(c+1) ends), proving the mechanism works
  end-to-end — but the GEMM already fills all 188 SMs, so the overlap reshuffles already-packed
  work and nets 0%. Cross-op overlap only pays when the producer op leaves SMs idle (it doesn't
  here); this is the *opposite* regime from decode (BW-bound, idle SMs).
- **GQ work-stealing is actively harmful** for chunked prefill (+40% at M=16384): the single
  atomic cursor and finer slices serialize dispatch faster than the tiny overlap helps.
- The thesis' "40 KiB × 2 smem double buffer" premise is also moot: the cross-SM arms need **no**
  double-buffered arena at all (they fit the same 60 KiB, occ-1), and a true intra-block
  producer→consumer fusion is infeasible anyway (the GEMM1→GEMM2 intermediate is 128×3840 f32 ≈
  1.9 MB per m-tile — orders of magnitude over smem), so the "double-buffer the sub-arena across
  M-chunks" idea reduces to the existing intra-op cp.async ring (T11's point) for same-op chunks,
  and to cross-SM counter edges (this POC) for cross-op — neither of which needs 2× smem.

**What is NOT refuted (out of scope here):** the *diffuse-straggler* lever — `dev.rs` measures a
coarse gate burning 2.63 ms / 16.9 ms token waiting for the slowest of 256 producers on
**decode**; `SE_FINE` narrows the *wait set* (fewer producers), a different lever from
chunk-pipelining and one this POC neither tests nor refutes. On prefill GEMM the straggler spread
is small (balanced, compute-heavy tiles), consistent with the ~0.6 µs drain measured here.

**Recommendation:** do NOT build the full chunked-double-buffer prefill kernel. The prefill TTFT
gap vs vLLM (§gemma4-12b-plow-prefill-sm120.md) is FLASH + fixed launch overhead + the GEMM tile,
not a recoverable cross-op bubble — pursue the p9 §2c/§2e levers (w8a8 shipped, MXFP8 absent,
per-opcode GEMM tilings, flash) instead.

## §B Full-path design (reference only — NOT to build given §V)

Recorded so a future revisit doesn't re-derive it. Per the user's production design, encode the scheduling discipline INTO THE PACKET (no global
`PLOW_NV_SCHED` build flag), derived from CHUNK-2's `PerChunkPlan.placement`:

- **Descriptor in DevInst (no growth).** `DevInst` (104 B, `dev.rs`) has a spare `j:[u32;2]`
  (j[0]=KV-cache stride, read ONLY by attention ops; **free on GEMM/prefill packets**). Encode
  `STATIC{cu_base, cu_span}` there — e.g. `j[0]=cu_base | cu_span<<16` (both ≤ 188 ⇒ 16 bits
  each) + one flag bit (reuse a `StreamEnt.flags` bit alongside `SE_FINE`) to select GQ vs
  STATIC. The static schedule is **reconstructable from the single packet** (`cu_base, cu_span,
  blocks, work` fully determine the slice→CU map) → no separate schedule table. **Answer: YES,
  the CU-id-in-packet descriptor fits 104 B without growing it.**
- **Per-phase policy.** PREFILL emits STATIC packets: chunk c's whole producer→consumer chain
  carries `STATIC{⌈c·n_cu/k⌉, span}` (colocated SM-set → consumer reads producer output from hot
  L2; cross-SM overlap via the per-chunk 1:1 edges). DECODE keeps GQ (work-steal — the shipped
  decode win). The policy is a property of the emitted packet, chosen at emit by phase.
- **Interp change (small).** Add a STATIC branch to the existing GQ loop: a workgroup checks its
  CU id against the packet's `cu_base/cu_span` and runs the slice it owns, instead of pulling the
  atomic cursor. This is exactly the "un-stub the fine gate + per-chunk edge" change §0.4
  diagnosed; the CU-id-in-packet is how the emitter hands the interp the static placement.
- **Which op-pairs chunk (prefill bucket):** the coupled 1:1-M chains — qkv-GEMM→(headnorm)→
  FLASH→o-GEMM, and gate|up-GEMM_GLU→down-GEMM. Norms/residuals ride the same chunk. FLASH is
  per-token independent on M so it chunks trivially. lm_head (M=1 tail) stays single-chunk.
- **Arena:** unchanged 60 KiB, occ-1 — the cross-SM arms need NO double-buffered arena (verified
  ptxas above). The "double buffer" is across SMs (per-chunk counters), not within a block.

---
