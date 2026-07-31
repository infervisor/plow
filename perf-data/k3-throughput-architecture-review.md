# plowrt/plowc throughput architecture review — what supports batching today, and what does not

**Method.** Source review of `crates/plowrt` (serve, exec, memory, sched) and `crates/devgen` on
branch `k3-batched-decode`, 2026-07-31. Every claim below carries a `file:line`. No GPU runs — the
numbers quoted are from existing measurement docs and this campaign's runs, and are labelled.

Scope is the question asked: chunked prefill, prefill+decode interleave, batched decode, address/VMM
behaviour as requests fold in, the launch/mux path, address reclaim on completion, and preservation
of KV for a prefix cache.

## 0. The one-paragraph answer

Batched decode is real and works (`PLOW_DECODE_BATCH`, sequence-row carriers, per-slot KDA state).
**Everything else on the list is absent on the shipped AMD TP path**, and the code says so
deliberately: `serve/engine.rs:13-15` — *"What is deliberately NOT here: VMM/prefix sharing, the S1
`ModelManager`, and multi-model residency. Those are CUDA-only today and stay CUDA-only — an AMD
serve is one model, B fixed sequence slots, no paging."* The machinery to fix most of it already
exists in-tree; it is the AMD/TP wiring that is missing, not the algorithms. There are also **two
correctness defects** (§2) that outrank every throughput item here.

## 1. What the AMD serving path actually is

| axis | CUDA engine | AMD engine (shipped, TP8) |
|---|---|---|
| slots | dynamic, continuous batching | **B fixed at emit**, `PLOW_DECODE_BATCH` |
| prefill | chunked, interleaved | **exclusive, whole-device, one tick** |
| tick | fused prefill+decode | **prefill XOR decode** (`mux.rs:1333` returns) |
| KV allocation | paged / growable | **flat, preallocated `[B][kv_head][ring][hd]`** |
| address reclaim | `begin_slot` drops blocks | **none on TP** |
| prefix cache | `PrefixCache` in `VmmKv` | **none** |
| VMM | always | `PLOW_VMM_KV=1`, **single-GPU only** |

`AmdServe::release` (`serve/engine.rs:344-353`) states the reclaim position outright: *"There is no
cache to reclaim — the block is fixed and preallocated."*

## 2. TWO CORRECTNESS DEFECTS, which outrank the throughput work — **BOTH FIXED**

Both are fixed as of this commit; the sections below are kept as the record of what was wrong and
why the fix is shaped the way it is. `scripts/k3_batch_gate.sh` **PASSES at B=4** with the fix in
(Check A: 4 copies of one prompt give 4 identical streams; Check B: 4 ragged prompts give the same
per-slot streams at a second batch width), which is what proves the per-slot striding did not
break batched decode.

### F1. KDA recurrent state is NEVER cleared between requests on the TP path

`begin_slot` is what clears carried state, and it is called only for the single-GPU variant:

```rust
// serve/engine.rs:257
if let Ranks::One(e) = &mut self.ranks { e.begin_slot(slot)?; }
```

`crates/plowrt/src/exec/amd_tp.rs` contains **no `begin_slot`, no `is_carried_state`, no memset**
(grepped). K3 is served at TP8, i.e. `Ranks::Tp`. So when a slot is handed to a new request, the
incoming sequence inherits the outgoing one's `kv.{l}.state` and `kv.{l}.conv_state.{q,k,v}`.

Why that is not benign: `exec/amd.rs:4249-4253` — *"the KDA recurrence READS `state` on its very
first token, and the conv arms read a window that is supposed to hold the `W - 1` tokens before the
sequence began. With no clear, 'the tokens before this sequence began' were the previous REQUEST's
— so a second prompt started from the first one's accumulated state."* That is a recorded,
understood failure mode; the fix landed for `Ranks::One` and never reached `Ranks::Tp`.

**69 of K3's 93 layers are KDA.** Every request after the first on a given slot is affected.

This is present in BOTH GSM8K arms (B=1 and B=4 are both TP8), so it does not explain the
grouped-vs-per-slot MoE gap in `k3-gsm8k.md` §2 — but it is a plausible depressor of the ABSOLUTE
number in both. Untested claim, flagged as such: the experiment is a GSM8K run with a
state clear between requests, which does not exist yet.

### F2. `begin_slot`'s clear is WHOLE-TENSOR, so wiring F1 naively would corrupt live slots

```rust
// exec/amd.rs:4268-4275
if is_carried_state(name) {
    let m = &self.devp[i];
    EngineDevice::memset_d8(&*self.be, m.base, 0, m.len as usize)?;   // m.len = ALL B slots
}
```

`declare_kda_state(b, c, prefix, slots)` sizes these tensors `× slots` (`devgen/src/kda.rs:280-287`)
and K3 passes `slots = t` for a sequence-rows program (`k3.rs:2332-2335`). So at B>1 the tensor holds
B independent states and this memset zeroes **all of them**. Admitting a request into slot 2 would
wipe slots 0/1/3's recurrence mid-stream.

Today this is latent, not live, precisely because F1 means it is never called on TP. **The two must
be fixed together**: make the clear `base + slot*stride .. + stride`, then call it from the TP path.

### F3 (documentation, but load-bearing)

Two comments now assert the opposite of what the code does, and both would mislead someone
implementing the above:

* `exec/amd.rs:4255-4262` — *"NOT PER-SLOT ... there is exactly one recurrent state in the model, so
  two KDA sequences cannot be in flight at once no matter what this function does."* False since
  `slots` was threaded through; `k3.rs:2332` allocates B.
* `exec/amd.rs:73-79` — *"Batch > 1 is not merely unimplemented here, it is not expressible with that
  blob."* Superseded; batch > 1 ships and passes `scripts/k3_batch_gate.sh` at B=4.

## 3. THROUGHPUT: prefill blocks every decode stream

`mux.rs:1277-1334`. One prefill per tick, lowest slot first, and the arm **returns** (`:1333`)
before reaching the decode launch. The code names the consequence itself (`:1283-1289`):

> *"The AMD tick is EITHER a prefill OR a decode ... so `mixed_decode_ns` is ZERO BY CONSTRUCTION.
> What disaggregation can recover here is therefore the WHOLE prefill tick, during which every live
> decode stream is stalled."*

MEASURED (this campaign): **49.3 tok/s** through `plowrt serve` at concurrency 16 on the B=16 packet
against **91.3 tok/s** from the same packet under `amd-bench`, TTFT **3.3 s**. The gap is this.

`AmdServe::prefill` (`serve/engine.rs:236-241`) is the other half: *"One prefill occupies the WHOLE
device — the prefill program is single-sequence and its dispatch is exclusive."*

**The good news, and it is genuinely good: the chunk loop already exists.** `AmdEngine::prefill`
(`exec/amd.rs:4082-4087`) is already

```rust
for step in steps { self.prefill_prepare(prompt, step)?; self.run_segmented(step.prog)?; }
```

over `plan_for` → `plan_chunks` (`:1081`, `:4042`) — a prompt is already covered by compiled bucket
widths and executed chunk by chunk. What is missing is only that the loop is **inside one blocking
call with no yield point**. Chunked prefill in the interleaving sense = hoist that loop into a
resumable per-slot cursor the mux can advance one chunk per tick.

The slot machinery it needs also already exists: `kv_rebase` (`exec/amd.rs:4120`) is a
pointer-table edit — *"exact, one 8-byte edit per KV buffer, and it needs no second prefill program
and no kernel change"* — and `AmdTpGroup::prefill_slot` (`amd_tp.rs:730-736`) already rebases every
rank and restores on the failure path.

### 3.0 BUILT AND MEASURED — whole-prompt interleave, and what it actually bought

The `return` at `mux.rs:1333` is gone: a tick now runs the pending prefill and THEN the decode,
instead of being one or the other. `PLOW_PF_NO_INTERLEAVE=1` restores the old behaviour, which is
what the A/B below uses.

MEASURED, `bench_speed.sh`, B=16 packet, TP8, `IN_LENS=1024 CONCS=16 NPROMPT=32 OUTLEN=128`:

| | TTFT mean ms | TTFT med ms | TPOT ms | ITL p99 ms | out tok/s |
|---|--:|--:|--:|--:|--:|
| prefill-only ticks | 10025.8 | 9897.8 | 265.9 | 388.8 | **42.9** |
| **interleaved** | 6396.0 | **3305.1** | 317.2 | 1803.7 | **41.1** |

**TTFT median improves 3.0x** (9.90 s -> 3.31 s), mean 1.57x. **Throughput does not improve**
(-4.2%), and **ITL p99 is 4.6x worse** (0.39 s -> 1.80 s).

That is the honest shape of it, and the reason is arithmetic rather than scheduling: interleaving
REORDERS work, it does not remove any. This benchmark sends 32 x 1024 = 32768 prompt tokens against
32 x 128 = 4096 completion tokens, so it is **8:1 prefill-dominated** and no tick policy can move
its aggregate. What interleaving fixes is the queue: a new arrival no longer waits behind N whole
prefills. What it costs is that a live decode stream can now sit behind a prefill inside its own
tick, which is the p99.

**So the 49.3-vs-91.3 served/kernel gap is NOT a scheduling gap.** On a prefill-dominated load it is
a prefill-VOLUME gap, and the levers are prefix caching (§5 — the 8-shot exemplars are ~900 of ~1000
tokens and identical across every request) and faster prefill, not tick policy. Interleave is worth
keeping for TTFT; it is not the throughput fix, and it should not be sold as one.

### 3.0b A REAL OUT-OF-BOUNDS GPU WRITE, found by driving all 16 slots

The first interleaved run died: `Memory access fault by GPU node-7`. Cause, and it is a shipped-path
bug rather than anything interleaving introduced:

`kv_slot_stride` (`exec/amd.rs`) took EVERY tensor named `kv.*` with stride `bytes / batch`. That
swept in **`kv.blkres`**, which is not a per-sequence cache: it is K3's snapshot ring,
`[t][nb_cap][hidden]`, sized at the LARGEST `t` in the blob — the widest PREFILL bucket, not
`batch`. So the per-slot rebase invents a stride unrelated to its layout and walks off the end:

```
T_max 8192, batch 16 -> stride 512 rows.  Slot 15 starts at row 7680,
and a 1024-row prefill chunk writes through row 8704 of an 8192-row tensor.
```

First overrunning slot is **15**, which is why B=4 never hit it (stride 2048, slot 3 ends at 7168).
This was flagged as a hazard in §4 item 5 of this file before it was ever observed — "masked today
by lowest-slot-first exclusivity; nothing bounds it" — and interleaving removed the mask by keeping
all 16 slots in flight.

Fixed by excluding `blkres` from the rebase. Nothing is lost: prefill and decode alternate rather
than overlap, and layer 0 resets the ring at the head of every pass, so both can use rows `[0, t)`
as scratch.

Also worth correcting from an earlier draft of this file: a ~1000-token 8-shot prompt does **not**
plan as one chunk. MEASURED from the server log, `TP prefill plan tokens=1038 chunks=2`.

### 3.0c CHUNKED PREFILL — BUILT (`PLOW_PF_NO_CHUNK=1` to disable)

The mux now advances a pending prefill by ONE CHUNK per tick instead of running the whole prompt,
and every other slot decodes in between. What made it safe is §3.0d's parked mask plus one more
thing: a mid-prefill slot is fed `pos = frontier`, the row its NEXT chunk overwrites anyway, so a
decode dispatch's KV write cannot clobber the prefix already built. That is the "live slot not in
`advance`" case `dispatch_all` already documented as sound; the recurrence is handled separately by
the mask.

MEASURED, `bench_speed.sh` B=16 TP8, `IN_LENS=1024 CONCS=16 NPROMPT=64 OUTLEN=128`:

| | TTFT med ms | TPOT ms | ITL p99 ms | out tok/s |
|---|--:|--:|--:|--:|
| whole-prompt prefill | 2435.0 | 333.2 | 1530.3 | **41.1** |
| **chunked prefill** | **1818.4** | 342.2 | **1132.2** | 39.1 |

**ITL p99 26% better, TTFT median 1.34x better, throughput 4.9% WORSE.** That is the textbook
chunked-prefill trade and it should be quoted as one: chunking is a TAIL-LATENCY feature, not a
throughput feature. It buys a decode stream the right to wait for one chunk instead of a whole
prompt, and it pays for that in extra dispatches and less efficient tiles.

Correctness: GSM8K 8-shot greedy N=200 at B=4/CONC=4 scores **196/200 = 0.9800 with zero request
errors**, against 193/200 without chunking. `k3_batch_gate.sh` passes at B=4.

Falls back to the whole-prompt path where there is no ladder to walk: single-GPU, `decode_only`,
a 1-token prompt, or when the prefix cache is on (its split points are its own, not the bucket plan).

### 3.0e THE COMBINED RESULT, on a real workload — 26% less wall

Chunking and the prefix cache now COMPOSE (the cache decides which span still needs prefilling,
chunking decides how that span is split across ticks). The first cut had them mutually exclusive,
which was an artefact of the implementation and not a design constraint.

MEASURED, GSM8K 8-shot greedy N=200, K3 TP8, B=4 packet, CONC=4 — a REAL workload whose 8 exemplars
are ~900 of ~1000 prompt tokens and identical across every request:

| serving config | exact_match | median s/q | **total s** |
|---|--:|--:|--:|
| interleave only | 193/200 = 0.9650 | 14.06 | 758 |
| **chunked prefill + prefix cache** | **195/200 = 0.9750** | **10.63** | **561** |

**26% less wall, 24% lower median latency, accuracy equal or better, zero request errors.** This is
the clause "improve prefill numbers significantly" being answered on a workload rather than on a
synthetic one.

### 3.0f A NOTE ON `bench_speed`'s `out_tok_s`, which DISAGREES

The same composed configuration on `bench_speed.sh` (B=16, `IN_LENS=1024 CONCS=16 NPROMPT=64
OUTLEN=128`) reports:

| | TTFT med ms | ITL p99 ms | req_s | out_tok_s |
|---|--:|--:|--:|--:|
| interleave only | 2435.0 | 1530.3 | 0.39 | **41.1** |
| chunked + prefix cache | **524.3** | **1066.5** | **0.50** | **38.1** |

TTFT median **4.6x** better, ITL p99 30% better, requests/s **+28%** — and `out_tok_s` DOWN. Those
cannot both be right about the same work, and the resolution is that they are not measuring the
same work: mean completion length is 105 tokens in one arm and 76 in the other. `bench_speed`'s
prompt is `"the quick brown fox jumps over the lazy dog"` repeated ~113 times, so the continuation
is degenerate and a stop token fires at a position that moves with any change to the token stream —
and the prefix cache legitimately changes the stream (different accumulation order).

**On a fixed-request benchmark with variable completion length, `req_s` is the honest throughput
metric and `out_tok_s` is confounded.** The GSM8K table above is the one to trust; it has a fixed
question set and a real answer.

### 3.0d PER-ROW PARKED MASK — the prerequisite, and a correctness fix on its own

`in.parked`, `[t]` u32, non-zero = skip that row's KDA recurrence and conv window. Carried in the
one free slot on each op (`i[7]` on `KdaStateStepG`, `j[1]` on `KdaConv3`) and read only under the
seq-rows condition, so a non-batched blob stays byte-identical (verified: B=1 md5
`7db2fbb34230050f0508a4e706523a98`, unchanged).

The sense is PARKED rather than ACTIVE for a safety reason, and the gate caught the alternative: an
all-zero or never-written mask must mean "every row participates". `amd-bench` drives the engine
directly and never publishes one — with active-semantics that parks every row and emits fluent
garbage.

Independently of chunking this fixes a live defect: `serve` now publishes `parked[s] = !live[s]`,
so an IDLE slot no longer has its recurrence advanced by a throwaway token on every single step.

### 3.1 Sub-prompt interleave needed a per-row mask — now built, see §3.0d.

An earlier draft of this file ranked "yieldable prefill cursor + interleave" as an M-cost wiring
job depending on nothing. **That was wrong**, and the thing that refutes it is one line of
`serve/engine.rs:387-434`: a decode dispatch advances **all B rows unconditionally**, because `t`
is compiled rather than passed.

For KV that is provably harmless and the code argues it correctly (an idle row rewrites row 0,
which an admitted request rewrites before reading). **For the KDA recurrence it is not harmless.**
The state step reads `state[slot]` and writes `state[slot]` for every row, every dispatch. So a
slot that is MID-PREFILL has its recurrence advanced by a garbage token the moment any other slot
decodes — which corrupts exactly the 69-of-93 layers the prefill just built.

`runtime/amd/op_kda.h` has **no active/enable/mask concept** (grepped: `active` appears only in a
comment about workgroup occupancy). There is no way today to make a row a no-op.

So true interleave — a prefill chunk and a decode step in the same tick — is blocked on one of:

1. **A per-row active mask in the KDA kernels** (`d_kda_state_step_t`, `d_kda_conv3`): skip the
   recurrence when the row is inactive. Small kernel change, but it is a KERNEL + EMIT change with
   a rebuild and a revalidation, not host wiring. This is the right fix and it composes with the
   batch-axis parallelisation the kernel review independently recommends.
2. **Host save/restore of the prefilling slot's carried state around each decode step.** ESTIMATE:
   K3 per-slot KDA state is ~`H*D*D*4` per layer over 69 layers — order tens of MB per slot per
   rank — so a save+restore pair is ESTIMATE ~30-40 µs against a 175 ms B=16 step (~0.02%). Cheap
   in time, ugly in design, and it needs measuring rather than assuming.

**What is NOT blocked**, and is still worth doing on its own: making a long prompt yield between
chunks so it does not hold the device for one enormous dispatch. That needs only the cursor, and
`AmdTpGroup` already exposes exactly the two calls for it — `plan_for` returning `Vec<ChunkStep>`
and `prefill_chunk`, whose own doc says *"a server chunks prefill itself, so it needs the plan and
`prefill_chunk` rather than the whole loop"*.

But note the payoff is workload-dependent and it is NOT the GSM8K shape: the K3 blob's ladder is
`[128, 512, 1024, 2048, 4096, 8192]`, so a ~1000-token 8-shot prompt plans as **one 1024 chunk**
and there is nothing to yield between. Chunk-level yielding pays only for prompts that span
several buckets, or if `plan_chunks` is given a cap that deliberately prefers narrower buckets
(a `PLOW_PF_CHUNK_MAX`-style knob, which does not exist).

### 3.2 The remaining constraints, which are real but not blockers

1. The decode program must run with the KV base at slot 0 (`exec/amd.rs:4115-4119`), so interleaving
   costs a rebase-to-slot / rebase-to-0 pair per chunk. That is a few-KiB table upload, today
   documented as *"off the per-token path — it happens once per prefill"* (`:4136`); per-chunk it is
   more often, and wants measuring before it is assumed free.
2. The prefill program is single-sequence by construction, so chunks of two different prompts cannot
   share a dispatch. Interleave gets you prefill-between-decodes, **not** batched prefill.

## 4. ADDRESS / VMM: nothing scales, because nothing is dynamic

* **No VMM on TP at all.** `vmm_ensure` (`exec/amd.rs:4227`) and `begin_slot` are reached only via
  `Ranks::One`. `PLOW_VMM_KV` defaults off anyway (`:2428`).
* **Without VMM the cache is `B × max_ctx` preallocated** and there is nothing to reclaim; `release`
  only clears `live/pos/next_id` (`serve/engine.rs:347-353`).
* **`VmmKv` is the right thing and it is already built** — `memory/vmm.rs` (1368 lines) holds a
  `PrefixCache` (`:272`, `:364`), `ensure_rows` grows a sequence on demand, `begin_seq` drops the
  outgoing sequence's physical blocks. It is reachable from AMD single-GPU today. The work is
  extending it rank-wise, not writing it.
* The mux's `KvArena`/`release_kv` (`mux.rs:251`, `:778`) comes from *"the first decode bucket that
  declares paging"* (`:287`), which a K3 AMD blob does not declare — so on AMD those calls are no-ops
  and slot `kv` handles are `None`. The reclaim path the user asked about is, on this backend,
  literally empty.

## 5.0 PREFIX CACHE — BUILT AND MEASURED (`PLOW_PREFIX_CACHE=1`, TP, default off)

Per slot, plow now keeps the last prompt it prefilled plus a snapshot of that slot's CARRIED
recurrent state at a token offset. A later prompt agreeing over that span restores the recurrence
and prefills only the suffix; the KV rows are already the slot's own (same tokens, same positions).

MEASURED, correctness first — GSM8K 8-shot greedy, N=200, K3 TP8, B=1 packet, CONC=1 (so every
request after the first hits):

| | exact_match | median s/q | total s |
|---|--:|--:|--:|
| no cache | 196/200 = **0.9800** | 4.88 | 1027 |
| **prefix cache** | 196/200 = **0.9800** | **3.70** | **825** |

**Identical accuracy, 24% lower per-question latency, 20% less wall.** A stale-prefix or
stale-state bug would show here as an accuracy collapse, not a crash, which is why this is the gate
that matters.

MEASURED, throughput — `bench_speed.sh`, B=16 packet, `IN_LENS=1024 CONCS=16 NPROMPT=64 OUTLEN=128`
(48 of 64 requests hit, 75%):

| | TTFT med ms | TPOT ms | ITL p99 ms | out tok/s |
|---|--:|--:|--:|--:|
| no cache | 2435.0 | 333.2 | 1530.3 | **41.1** |
| **prefix cache** | **1267.8** | 301.9 | 1663.7 | **44.0** |

**TTFT median 1.92x better, throughput +7.1%.** At the earlier NPROMPT=32 it was a small LOSS
(39.5 vs 41.1) — with only 2 waves over 16 slots, half the requests pay a snapshot and only half
can hit, which is the worst case for measuring it. Report the hit rate with any number from this.

### Why the win is smaller than the prefill it removes, and what to do next

75% of prefill tokens were eliminated for +7% throughput. The gap is the restore itself: the carried
state is **56 MiB per slot** (MEASURED, 276 tensors — 69 KDA layers x state + 3 conv) and a restore
touches all 8 ranks, so ~448 MiB moves per hit. That is the next thing to attack, and it is not the
byte count alone:

* `memcpy_dtod` was create-signal / copy / **BLOCKED wait** / destroy PER CALL, so a snapshot was
  276 blocked host waits per rank. Fixed here — `memcpy_dtod_batch` issues N copies against ONE
  completion signal (`hsa_amd_memory_async_copy` decrements it, so a signal armed at N and waited
  once is a correct barrier). That took the n=32 arm from a clear loss toward break-even.
* NOT yet done: the restore has never been timed directly. Before optimising further, time it —
  the same discipline §4 of `k3-hier2-ceiling.md` applies, and everything above about where the
  remaining cost sits is INFERRED from the throughput delta, not measured.
* A pointer-swap instead of a copy does not work: the recurrence writes in place, so the snapshot
  would be destroyed on first use.

## 5.0b The design, and the three uncertainties it had to resolve first

An earlier draft of this file ranked prefix caching below "VMM on TP" as a dependency. **That was
wrong.** VMM is what a *paged, cross-slot, evicting* cache needs. A **same-slot** prefix cache needs
none of it, and same-slot is enough for the traffic that matters (a client pool round-robins over
slots, so each slot sees the same shared prefix again and again).

Three uncertainties blocked the design; all three are now resolved:

1. **Can a prefill resume at an arbitrary position?** YES. `chunk_steps` already emits `ChunkStep`
   with an arbitrary `c0`, `prefill_prepare` reads `prompt[c0 + i]` and writes ABSOLUTE positions,
   and `patch_prefill`/`rebase_chunk` set `FlashPrefill`'s `i[1] = c0 + clen` so attention covers
   `[0, c0+clen)`. A chunk starting at `c0 = P` with the KV for `[0, P)` already resident is
   *exactly* the situation an ordinary second chunk is already in.
2. **Does the split point have to be bucket-aligned?** NO, and this was the one I expected to be a
   problem. `rebase_chunk` sets **every KDA op's row count to `clen`**, not to the padded bucket
   width `T` (`exec/amd.rs`, and `EVERY KDA OP'S ROW COUNT BECOMES clen` is an existing regression
   test). So a chunk with `clen = P` advances the recurrence exactly `P` rows and leaves the state
   at exactly `P`. Pad rows write KV nothing reads and do NOT enter the recurrence.
3. **Is there a device-to-device copy for the state snapshot?** YES —
   `EngineDevice::memcpy_dtod`, implemented on the HSA backend.

### The design

Per slot, keep `cached_prompt: Vec<u32>` and a snapshot of the CARRIED state (KDA `state` +
`conv_state`, per-slot stride, `blkres` excluded) taken at position `P`.

```
P = lcp(new_prompt, cached_prompt)
hit  : restore_state(slot, snap)          ; prefill [P, len)     <- skips P tokens of work
miss : prefill [0, P) ; snapshot(slot)    ; prefill [P, len)     <- pays once, arms the next hit
```

Converges after two requests per slot, then saves `P` tokens of prefill on every subsequent one.

**Cost.** ESTIMATE: carried state per slot per rank is ~`H*D*D*4` over 69 KDA layers, order **54 MB**
— ~870 MB for 16 slots against 288 GB of HBM, and a snapshot/restore pair is ~18 us each at HBM
bandwidth against a 175 ms step. Both negligible.

**Payoff (as predicted before building; see the measured table above for what it actually did).** §3.0 measured that interleaving
cannot move a prefill-dominated load because it reorders rather than removes work. Prefix caching
REMOVES work. On the `bench_speed` shape (32 x 1024 prompt tokens vs 32 x 128 completion) an
8-shot-style shared prefix of ~900 of ~1000 tokens cuts prefill volume ~9x, which on an 8:1
prefill-dominated load is most of the total. ESTIMATE: 41 tok/s served toward the decode-bound
ceiling. This is the item to build next, and it is now unblocked.

**Correctness gates it must pass.** Token-identity against a cold run for the same prompt (a hit and
a miss must produce the same stream); `k3_batch_gate.sh`; and GSM8K, where a stale-prefix bug would
show as an accuracy drop rather than a crash.

## 5. PREFIX CACHE: absent today, and the largest single throughput item

There is no prefix reuse on AMD. `vmm_ensure` calls only `ensure_rows` — it never consults
`PrefixCache::lookup`. So every request re-prefills its whole prompt.

Why this matters more than it looks: the accuracy harness itself is the worst case. GSM8K 8-shot
sends **the same 8 exemplars on all 200 requests** — a shared prefix of roughly 900 of ~1000 prompt
tokens. MEASURED TTFT is 3.3 s under load. An exact-prefix hit would skip ~90% of prefill work.
Real chat traffic (shared system prompt, multi-turn history) has the same shape.

Preserving KV across a completed request — the user's "reclaim the address but keep the KV for
prefix cache" — is exactly `VmmKv`'s existing design: `begin_seq` releases the sequence's *physical
blocks* while the `PrefixCache` retains the *hashed block identities*, and `evict_lru` reclaims only
under pressure (`memory/vmm.rs:41`). Nothing new needs inventing; it needs to reach TP.

## 6. WASTED WORK: idle slots do a full row

`dispatch_all` (`serve/engine.rs:387-434`) runs all B rows every dispatch — *"the program's `t` is
compiled, not passed"* — feeding idle slots `pos=0, kvlen=1, id=0`. Sound (an admitted request
rewrites row 0 before reading it, `:377-383`), but it means a B=16 blob pays B=16 cost at
concurrency 1. MEASURED: B=16 is 175.3 ms/step vs B=1's 29.0. This is why admission should keep
slots full, and why a large fixed B is a bad default for mixed load.

## 7. Ranked plan

| # | item | why | cost | expected |
|---|---|---|--:|---|
| 1 | ~~**Per-slot state clear + call it from TP**~~ **DONE** (F1+F2 together) | correctness; 69/93 layers inherited the previous request's recurrence | S | landed: `AmdEngine::begin_slot` strides by `len/batch`, `AmdTpGroup::begin_slot` calls every rank, `serve` calls it on both arms |
| 2 | ~~**Fix the stale comments**~~ **DONE** (F3) | they actively misdirect the work below | XS | — |
| 3a | **Per-row active mask in the KDA kernels** | **PREREQUISITE for interleave** — see §3.1; without it a decode dispatch advances a mid-prefill slot's recurrence | M (kernel + emit) | unblocks 3b; composes with the batch-axis work |
| 3b | **Yieldable prefill cursor + interleave in the mux tick** | closes 49.3 → 91.3 tok/s; TTFT 3.3 s | M | `plan_for`/`prefill_chunk`/`kv_rebase` all exist; **blocked on 3a** |
| 4 | **Prefix cache on AMD** (`VmmKv` → TP, consult `lookup` in `vmm_ensure`) | ~90% of prefill on shared-prefix traffic | L | needs VMM on TP first |
| 5 | **VMM on TP / address reclaim** | prerequisite for 4; also unbounds `B × max_ctx` | L | `memory/vmm.rs` is written, not wired |
| 6 | Admission that keeps slots full | idle rows cost full work at fixed B | S | scheduler-only |

Order matters: 1-2 are correctness and cheap. 3 is the biggest throughput win per unit work and
depends on nothing else. 4 depends on 5.

## 8. What this review did NOT do

No GPU runs (the lease was held by an accuracy run). No measurement of the per-chunk `kv_rebase`
cost that item 3 hinges on. F1's accuracy impact is reasoned from the code and the recorded failure
mode, **not measured** — the A/B does not exist yet. `plowc`/devgen was reviewed only where it bears
on these questions (`slots` threading, `RowKind`, bucket ladder); the kernel-side questions (skinny
GEMM shape at B=4-16, flash split counts) are a separate review.
