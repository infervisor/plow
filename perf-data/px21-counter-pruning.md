# PX-21 — the emitted dependency graph is already minimal. 0 of 717 edges are redundant.

RTX 5090 (sm_120a, 170 SM) · Gemma-4-12B-it · **all-fp8** (fp8 weights + all-layer fp8 KV,
`PLOW_FP8=1 PLOW_FP8_KV=1 PLOW_FP8_KV_FULL=1 --weight-dtype fp8`) · whole 48-layer model
**and** single sliding/full blocks · `crates/plowrt/examples/graphstat.rs` (added by this note).

Companion to `px19-tile-graph.md` (which bounded the whole gate protocol at 2.0% of prefill /
10.8% of decode). **This is a NEGATIVE result delivered at the scoping gate. Nothing was
implemented; no serving-path file was touched; no GPU time was spent.**

---

## Question

Prune the dependency graph the emitter produces: (1) **transitive reduction** — drop A→C when
A→B→C exists; (2) **counter pruning** — fewer distinct counters; (3) **scheduler-aware
emission** — stop optimising structures the global-queue runtime ignores.

**The answer is that there is nothing to prune.** The emitted graph is a series-parallel chain
whose transitive reduction is itself, the counter count already equals the op count exactly, and
the one prunable item in the whole 48-layer model is a single wasted atomic. Counts in §1–§3,
arithmetic in §5.

---

## 1. What the 12B actually emits today, measured

`graphstat` parses the blob and recovers the op-level DAG from the stream entries' wait lists
(gates live on `StreamEnt`, never on the 64-byte `DevInst64`). All-fp8 whole model:

| program | ops | **counters** | SE_FINE | **edges** | **edges after TR** | dead ctr | work items | wait-polls | counter bumps |
|---|---|---|---|---|---|---|---|---|---|
| T=128 prefill | 814 | **814** | **0** | **957** | **957** | 1 | 106,731 | 131,083 | 106,731 |
| T=512 prefill | 718 | **718** | **0** | **813** | **813** | 1 | 106,719 | 122,869 | 106,719 |
| T=1024 prefill | 718 | **718** | **0** | **813** | **813** | 1 | 106,719 | 122,869 | 106,719 |
| **T=1 decode** | 622 | **622** | **0** | **717** | **717** | 1 | 42,253 | 58,572 | 42,253 |

Two facts fall straight out of the table and neither is an estimate:

* **`counters == ops`, exactly, in every program.** One counter per op is the floor of a
  counter-per-op protocol. There is no fine-counter overhang to reclaim.
* **`edges == edges after transitive reduction`, exactly, in every program.** Zero redundant
  edges. Not "few" — zero.

Reproduced on the single-block assets in the same all-fp8 config (`px15-slidekv-b1`,
`px15-fullkv-b1`), on the bf16 whole model, and on prefill and decode alike. The result is
invariant to configuration.

| asset | program | ops | counters | edges | edges after TR |
|---|---|---|---|---|---|
| `px15-slidekv-b1` (sliding, all-fp8) | T=1024 | 15 | 15 | 16 | **16** |
| `px15-fullkv-b1` (full-attn, all-fp8) | T=1024 | 14 | 14 | 15 | **15** |
| `plow-out/gemma-4-12B-it` (bf16) | T=1 | 542 | 542 | 637 | **637** |

---

## 2. Why it is zero: the graph is a chain with 3-wide diamonds and no cross edges

The per-layer decode subgraph, dumped verbatim (`GRAPHSTAT_V=1`, all-fp8, layer 0):

```
 0 Embed             blocks=1     deps=[]
 1 RmsNorm           blocks=1     deps=[0]
 2 GemvFp8   (q)     blocks=85    deps=[1]
 3 GemvFp8   (k)     blocks=42    deps=[1]
 4 GemvFp8   (v)     blocks=43    deps=[1]
 5 HeadNormRope      blocks=2     deps=[2]
 6 HeadNormRope      blocks=2     deps=[3]
 7 HeadNormRope      blocks=2     deps=[4]
 8 FlashDecode       blocks=170   deps=[5, 6, 7]
 9 FlashMerge        blocks=16    deps=[8]
10 GemvFp8   (o)     blocks=170   deps=[9]
11 NormResidualNorm  blocks=1     deps=[10]
12 GemvGluFp8        blocks=170   deps=[11]
13 GemvFp8   (down)  blocks=170   deps=[12]
14 NormResidualNorm  blocks=1     deps=[13]   -> next layer
```

**Every op has exactly one dependency except the two Flash ops, which have three.** Those three
are the q/k/v branches — mutually unreachable by construction, so not one of them is transitively
implied by another. A transitive reduction has no work to do on this shape, at any layer count.

The residual connections do **not** add the long-range `x → residual_add(x, …)` edge a reader
would expect: `NormResidualNorm` fuses the add and reads the residual tensor that the chain
already orders. That is precisely the edge transitive reduction would have deleted, and the
emitter never creates it.

### The gate is a parallel max, not a sum — so the edge count would not have mattered anyway

`interp_sm120.cu:1577`:

```c
for (unsigned w = threadIdx.x; w < wait_len; w += blockDim.x) {
    const PlowWait pw = prog.waits[wait_ofs + w];
    while (ctr_poll(PLOW_CTR(prog.counters, pw.id)) < pw.threshold) { ... }
}
```

`blockDim.x` is 256 and the **measured maximum `wait_len` is 3**. So every entry executes exactly
**one** loop trip; the 1–3 counters are polled concurrently in 1–3 threads of the same warp. Going
from `wait_len=3` to `wait_len=2` changes the executed instruction count by **zero**. `wait_len`
only matters if it crosses 256 (a second trip) or crosses 0 (skips the acquire fence) — and the
realizable range is {0, 1, 3}.

And a transitively redundant edge is, by definition, never the critical one: if A→B→C, then A's
counter is already at threshold when C evaluates its gate, so the redundant poll is a single
relaxed load that succeeds first try, in a thread that would otherwise be idle. **Even a graph
that had redundant edges would not pay for removing them on this gate implementation.**

---

## 3. Counter pruning: 1 wasted atomic in the whole 48-layer model

`succs` unconditionally carries `op.counter` (`devbuild.rs:519`), so every slice of every op bumps
its counter whether or not anyone waits on it. Counters nobody waits on, measured:

| program | dead counters | op | blocks | wasted atomics | of total bumps |
|---|---|---|---|---|---|
| T=1 decode | **1** | `ArgmaxFin` | 1 | **1** | 1 / 42,253 = **0.0024%** |
| T=1024 prefill | **1** | terminal | 1 | **1** | 1 / 106,719 = **0.0009%** |

That is the entire counter-pruning opportunity in the shipped graph. On a **single-layer block**
asset the terminal `NormResidual` has 170 blocks and no consumer (170 / 2,210 bumps = 7.7%), but
that is an artifact of the block being cut out of the model — in the 48-layer program that op's
consumer is the next layer's `RmsNorm`.

Nothing below one-counter-per-op is reachable without breaking the protocol: consecutive ops need
distinct counters precisely because op N+1's gate is what orders it after op N. Collapsing the
chain onto a single monotonic sequence counter keeps the atomic count identical (one bump per
slice) while funnelling all 42,253 bumps to **one** address — strictly worse contention.

---

## 4. What the global-queue path actually reads (item 3)

`PLOW_NV_SCHED` defaults to **1** (`interp_sm120.cu:1467-1469`). The live arm reads:

| structure | read by shipped cubin? |
|---|---|
| `gq_stream`, `gq_seg_ofs` | **yes** (`interp_sm120.cu:1529-1543`) |
| `insts`, `waits`, `succs`, `counters` | **yes** |
| `stream`, `stream_ofs`, `stream_len` | **no** — behind the `#else` on `PLOW_NV_SCHED == 1`, i.e. the `SCHED=0` arm, which is not built |

This reproduces px19 §2 independently. Consequence for this brief: **the per-CU partition is not
a thing to prune, because it is not a thing that runs.** Any emitter change scoped to `cus[]`
placement has no path to hardware.

**New, and not in px19:** the dead `stream` array is still serialised into the blob and uploaded.
362,422 stream entries × 24 B = **8.70 MB of a 17.73 MB blob — 49% of the devblob is a
permutation of `gq_stream` that the shipped cubin never reads.** This is load-time H2D and device
memory, **not** per-token time, so it is not a latency win and is not claimed as one. Recorded
because it is free to drop behind the same `PLOW_NV_SCHED` flag that already selects the arm.

---

## 5. The estimate, with the arithmetic

Whole-graph addressable cost, from px19 §5: the entire dispatch + gate protocol is **2.0% of
prefill** and **10.8% of decode**; at the 81/19 split, **3.7% of the wall if it became free**.
Every item in this brief is a strict subset of that.

**Item 1 — transitive reduction.** 0 of 717 edges (decode), 0 of 813 (prefill).

```
0 edges removed × (anything) = 0.0%
```

Structurally zero, per §2, and doubly so because the gate cost is invariant to `wait_len` over
{0,1,3}.

**Item 2 — counter pruning.** 1 wasted atomic of 42,253 decode bumps:

```
decode:   (1 / 42,253) × 10.8%  =  0.00026% of decode
prefill:  (1 / 106,719) × 2.0%  =  0.00002% of prefill
wall:     0.81 × 0.00002%  +  0.19 × 0.00026%  =  5e-7 % of end-to-end wall
```

Six orders of magnitude below the cell's **~3% reproducibility band**.

**Item 3 — scheduler-aware emission.** 0% on latency; the structure in question is not executed
(§4). The 8.70 MB blob saving is size, not time.

**Scoping gate: STOP.** Nothing implemented.

---

## 6. Against MPK's 37–118× event-fusion claim — plow is already at 68–149×

The brief asks whether plow's graph is tight or fat next to MPK (arXiv 2512.22219). It is
measurable directly. An unfused per-work-item event graph has one event per work item; plow emits
one counter per **op**, with every slice of that op sharing it:

| program | work items | counters | **fusion ratio** |
|---|---|---|---|
| T=1 decode | 42,253 | 622 | **67.9×** |
| T=1024 prefill | 106,719 | 718 | **148.6×** |

MPK reports **37–118×**. plow's decode graph sits inside that band and its prefill graph sits
above it — and it gets there **by construction** (`Builder::emit` allocates one counter per op),
not via a fusion pass. **The 37–118× that MPK recovers is a reduction plow never has to make,
because it never emits the unfused form.** That closes the comparison: there is no order of
magnitude waiting in plow's event graph.

The corollary for `SE_FINE`: it is the one mechanism that would *raise* the event count (one
counter per producer slice, up to 256×256 atomics on an all-to-all edge, per the `Dep` doc at
`devbuild.rs:58`). `select_granularity` downgrades **every** fine edge — the emit log for the
all-fp8 model reads `0 fine edges kept, 336 downgraded to coarse` on decode and `48 downgraded`
per prefill program — and the sm_120 loader refuses fine programs outright
(`check_coarse_single_segment`, `devblob.rs:391-402`). Measured `SE_FINE` count: **0 of 42,253**.
The five-minute check in the brief holds; it is what ended this task.

---

## 7. Results

1. **Zero redundant edges.** Transitive reduction removes **0 of 717** decode edges and **0 of
   813** prefill edges on the all-fp8 48-layer 12B; likewise 0 on both single-block assets. The
   graph is a chain of single-dependency ops punctuated by 3-wide q/k/v diamonds with no cross
   edges (§2).
2. **Counters already equal ops**, exactly, in all four programs (622, 718, 718, 814). One
   counter per op is the protocol's floor.
3. **The entire counter-pruning opportunity is 1 atomic** — the terminal `ArgmaxFin`, 1 of 42,253
   decode bumps = 0.0024%, worth **5e-7 % of the wall** (§5).
4. **The gate cost is invariant to the edge count** in the realizable range: `wait_len ∈ {0,1,3}`
   against `blockDim.x = 256`, so 1–3 waits are one loop trip polled in parallel threads. Edge
   pruning could not have paid even had there been edges to prune (§2).
5. **`SE_FINE` = 0 of 42,253** entries; `select_granularity` downgrades all 336 declared fine
   decode edges and the loader refuses fine programs. Confirms px19 §5 in the all-fp8 config.
6. **The global-queue path does not read `stream`/`stream_ofs`/`stream_len`** (§4) — so per-CU
   emission cannot be optimised into the shipped cubin at all. Confirms px19 §2 independently.
7. **plow's event graph is at 68× (decode) / 149× (prefill) fusion** vs a per-work-item event
   graph, i.e. at or above the top of MPK's reported 37–118× band, reached by construction (§6).

### Bugs found

None. Every gate that ran, passed. `crates/plowrt/src/exec/gpu.rs`'s conflict markers (px19 bug 1)
are already fixed on the campaign tip this branched from; the `cuda,hf-tokenizer` release build is
clean here.

### Dead code noticed, not deleted (per CLAUDE.md §3)

`Program::stream` / `stream_ofs` / `stream_len` (`devbuild.rs:578-585`) and their blob sections —
**8.70 MB of a 17.73 MB all-fp8 blob**, unread by the shipped `PLOW_NV_SCHED=1` cubin. Kept
because the `SCHED=0` arm still compiles against them and px19 used that arm for its scheduler
A/B. Mentioned, not removed.

---

## 8. Gates

| gate | result |
|---|---|
| **(a) counters + edges emitted today, per program, MEASURED** | **PASSED** — §1, all four all-fp8 programs plus two single-block assets and the bf16 whole model, via `graphstat` on the real blob |
| **(b) how many survive transitive reduction** | **PASSED** — §1/§2, **all of them**: 717/717 decode, 813/813 prefill, 16/16 and 15/15 on the blocks |
| **(c) which structures the global-queue path reads** | **PASSED** — §4, read off `interp_sm120.cu:1492-1543`; `stream*` is on the unbuilt `SCHED=0` arm |
| **(d) estimate with arithmetic** | **PASSED** — §5. 0.0% for transitive reduction, **5e-7 %** of wall for counter pruning, 0% latency for item 3 |
| **estimate below the ~3% band ⇒ STOP** | **HONOURED** — nothing implemented, no serving-path file touched |
| measured in the **all-fp8** configuration | **PASSED** — §1; the 48-layer blob emitted here with `PLOW_FP8=1 PLOW_FP8_KV=1 PLOW_FP8_KV_FULL=1 --weight-dtype fp8`, plus `px15-slidekv-b1` / `px15-fullkv-b1` |
| single block measured before any whole-model claim | **PASSED** — §1, one sliding and one full all-fp8 block, both at T=1024 |
| `SE_FINE` five-minute check | **PASSED** — §6, **0 of 42,253**; emit log confirms all 336 fine decode edges downgraded |
| **byte-identical blob when the feature is off** | **PASSED** — no feature exists; the emitter (`packet`/`plowc`/`devgen`) and `runtime/` are untouched. Proved by re-emit: `b4b01a73…3d998` for two independent emits of the all-fp8 48-layer blob |
| **byte-identical cubins** | **PASSED, structurally** — zero files modified under `runtime/`; no `.cu`/`.cuh` touched, so no cubin was rebuilt |
| **`cargo test --workspace`** | **PASSED** — **95 suites, 669 tests, 0 failures** |
| **`cargo build --release -p plowrt --features cuda,hf-tokenizer`** | **PASSED** — clean, warnings only |
| greedy-token parity at fixed chunk | **NOT RUN — vacuous.** No serving-path change exists to perturb the token stream; the blob hashes identical |
| block-level `block_run bench` A/B | **NOT RUN, deliberately** — the estimate is 0%, so there is no arm to A/B. Spending contended GPU time to time a byte-identical blob would measure noise. Same call px19 made |
| end-to-end 127k cell | **NOT RUN, deliberately** — same reason |
| GPU exclusivity (`gpulease`) | **NOT APPLICABLE** — no GPU work was done; every number here is read off the blob on CPU |
| fp8-KV block prefill `CUDA_ERROR_LAUNCH_FAILED` (PX-15) | **NOT ENCOUNTERED** — this note never launches a block asset; it only parses one |

---

## 9. Reproduce

```bash
nix develop -c cargo build --release -p plowc -p plowrt --example graphstat
scripts/px21_emit_allfp8.sh /root/px21/allfp8     # CPU only, ~5 s after weights load
./target/release/examples/graphstat /root/px21/allfp8/model.pkt
GRAPHSTAT_V=1 ./target/release/examples/graphstat /root/px21/allfp8/model.pkt 1
./target/release/examples/graphstat /root/plow-out/px15-slidekv-b1/model.pkt
```

`graphstat` reads the blob and touches no GPU. `GRAPHSTAT_V=1` dumps the redundant-edge list, the
dead-counter list, and the full per-op dependency table that §2 quotes.

---

## 10. Recommendation

1. **Consider the graph-pruning question closed.** The counts are in §1 and they are exact:
   `counters == ops`, `edges == transitive reduction of edges`, `SE_FINE == 0`. This is the
   measured "already near-minimal" the brief asked for, and §6 states it in MPK's own units so
   the comparison does not have to be re-litigated.
2. **The leverage is unchanged from px9 / px19 §9.3.** 81% of the wall is prefill, ~98% of prefill
   FLOPs are `linear`, and plow's w8a8 GEMM runs at 61–66% of a ceiling cuBLASLt reaches at
   95–99%. That is ~30% on the dominant term. The scheduler above it is 2.0%, and the graph
   feeding the scheduler is already minimal.
3. If the 8.70 MB dead `stream` section is ever worth reclaiming, gate it on the same
   `PLOW_NV_SCHED` value that already selects the arm — but bank it as blob size and load time,
   never as tokens/s.
