# Prefix caching for Kimi-K3: the KV cache problem, why linear attention breaks the standard answer, and what plow does instead

**Status:** built, measured, shipped on branch `k3-batched-decode`
(`PLOW_PREFIX_CACHE=1`). Every number below is MEASURED on 8× MI355X (gfx950) at TP8 unless
labelled otherwise.

---

## Part 1 — The KV cache problem, from first principles

### 1.1 Why the cache exists at all

A transformer decoder generates one token at a time. To produce token *n* it attends over tokens
`0..n`. Written naively, generating a sequence of length *N* re-computes attention over the whole
prefix at every step, so the total work is O(N²) *and* every step re-reads the entire prompt through
the model.

The fix is old and universal: for each layer, **cache the key and value projections of every token
you have already processed**. Then step *n* computes K/V for exactly one new token, appends it, and
attends over the cache. Generation goes from O(N²) recomputation to O(N) with an O(N) memory cost.

That memory cost is the entire subject. For a model with *L* layers, *H* KV heads and head dim *D*,
in bytes-per-token:

```
kv_bytes_per_token = L × H × D × 2 (K and V) × sizeof(elem)
```

Multiply by context length and by the number of concurrent sequences and it dominates everything
that is not weights. Two structural consequences follow, and they are what every serving system is
really designed around:

1. **The cache is per-sequence.** Two requests cannot share one cache, because their token streams
   differ. So capacity scales with *concurrency × context*, not with the model.
2. **The cache is append-only and positionally addressed.** Token *i*'s K/V lives at row *i* and is
   never rewritten. This is what makes reuse possible at all.

### 1.2 The prefix-sharing observation

Real traffic is not made of independent prompts. It is overwhelmingly made of prompts that **share
long prefixes**:

* a system prompt repeated on every request,
* few-shot exemplars repeated on every request,
* a document that many questions are asked about,
* a chat history where turn *k+1* extends turn *k*,
* a code file re-sent with each edit request.

If two prompts agree on their first *P* tokens, then — because K/V for token *i* depends only on
tokens `0..i` — **their caches agree on the first *P* rows, exactly and bit-for-bit.** Recomputing
those rows is pure waste.

This is the prefix-cache observation, and its payoff is not marginal. In the accuracy harness used
throughout this campaign (GSM8K 8-shot), the eight worked exemplars are **~900 of ~1000 prompt
tokens, identical across all 200 requests**. A perfect prefix cache eliminates ~90% of all prefill
work on that workload.

### 1.3 How the field solves it, and what those solutions assume

The standard answer has two layers:

* **Paged attention** (vLLM). Stop allocating the cache as one contiguous per-sequence block. Cut
  it into fixed-size *pages* and give each sequence a **block table** — an indirection from logical
  position to physical page. Fragmentation disappears, and two sequences can point at the *same*
  physical page.
* **A prefix index over the pages** (RadixAttention / automatic prefix caching). Hash each block of
  tokens, keep a radix tree or hash map from block-hash to physical page, and on admission walk the
  new prompt against the tree to find the longest cached prefix. Bump refcounts, share the pages,
  and prefill only the suffix. Evict by LRU under pressure.

Both layers rest on **one assumption**, and it is so universally true for softmax transformers that
it usually goes unstated:

> **The entire per-sequence state of a layer is its KV cache, and that state is positional:
> the bytes for token *i* depend only on tokens `0..i` and can be shared, copied, or re-pointed
> without touching anything else.**

Under that assumption, resuming at position *P* is trivial: point at the shared pages, set the
position counter to *P*, and continue. There is nothing else to restore.

**Kimi-K3 violates this assumption.** That is the whole reason this document exists.

---

## Part 2 — Why K3 breaks it: linear attention has state that is not positional

### 2.1 K3's layer mix

K3 is 93 layers, and they are not homogeneous:

| layers | mixer | per-sequence state |
|---:|---|---|
| 24 | **MLA** (multi-head latent attention) | an ordinary KV cache — `kv.{l}.ckv` (fp8 latent), `kv.{l}.krot` (RoPE half), `kv.{l}.scale` |
| **69** | **KDA** (Kimi delta attention — a *linear / recurrent* attention) | a **recurrent state** `kv.{l}.state`, shape `[H, D, D]` f32, plus short-conv windows `kv.{l}.conv_state.{q,k,v}` |

The 24 MLA layers behave exactly as Part 1 describes. The 69 KDA layers do not.

### 2.2 The difference that matters

A softmax attention layer keeps a **list**. Token *i* appends a row; nothing already written is
modified. The state after *n* tokens is literally the concatenation of *n* independent
contributions, so a prefix of the state is the state of the prefix.

A linear-attention layer keeps an **accumulator**. Conceptually the recurrence is

```
S_i = decay_i ⊙ S_{i-1} + β_i · (k_i v_iᵀ)        # S is [H, D, D]
out_i = q_iᵀ S_i
```

Every token *folds into* a fixed-size matrix. That buys the thing linear attention exists for —
**state size is O(1) in context, not O(N)** — and it costs the property prefix caching depends on:

> **`S_P` is not recoverable from `S_N` for `P < N`.** The recurrence is a lossy fold. There is no
> inverse, no "prefix" of the state to point at, and no way to rewind it.

Concretely, in this codebase (`runtime/amd/op_kda.h`, `d_kda_state_step_t`):

```c
for (unsigned t = 0; t < T; t++) {
    ...
    float* col = st_h + (size_t)t * bstride + (size_t)j * D;   /* READ-MODIFY-WRITE */
}
```

Each row read-modify-writes `state[row]` in place. After processing tokens `0..N`, the bytes that
held `S_P` are simply gone.

### 2.3 What that does to every standard technique

This single property has consequences well beyond prefix caching, and this campaign hit all of them:

| technique | why it assumes positional state | what breaks on K3 |
|---|---|---|
| **prefix caching** | share pages, set position, continue | there is no page holding `S_P` |
| **paged attention** | logical→physical indirection per block | the recurrence has no blocks to page |
| **speculative decoding** | reject tokens by rewinding the position | rejected tokens are already folded into `S`; see `k3-speculative-decoding.md` §3 |
| **slot reuse between requests** | a new sequence starts from an empty cache | a new sequence inherits the previous one's `S` unless explicitly cleared |

That last row was a **live bug** in this tree, and it is worth stating because it shows the failure
mode is silent rather than loud. `begin_slot` — the function that clears carried state when a slot
is handed to a new request — was only called on the single-GPU path. K3 serves at TP8. So every
request after the first on a slot began from its predecessor's accumulated recurrence across 69 of
93 layers. Nothing crashed; the model simply got quietly worse.

**Fixing it took K3's GSM8K from 81.0% to 98.0%** (+17.0pp, z = 5.77, p < 1e-8). It also explains a
77.5–90.3% spread across nominally identical greedy runs that had looked like nondeterminism: the
score depended on what the *previous* request had left behind, so it moved with request ordering.

The lesson generalises: **on a recurrent-state model, per-sequence state hygiene is a correctness
property, not a performance one.**

---

## Part 3 — plow's architecture, and the four properties that made the fix small

plow is a **compiler** (`plowc` / `devgen`) plus a **persistent-megakernel runtime** (`plowrt` +
`runtime/amd/interp.hip`). The compiler emits a *device program* — a graph of packets with
counter-gated dependencies — and the runtime keeps one resident kernel that pulls packets off a
queue. There is no launch per layer.

Four properties of that design did the heavy lifting. None was built for prefix caching; all four
turned out to be exactly what it needed.

### 3.1 The KV cache is flat, and a slot is a pointer edit

There is no block table and no paging on the AMD path. The cache is one flat allocation shaped
`[B][kv_head][ring][head_dim]`, sized at compile time from `PLOW_DECODE_BATCH` and `max_ctx`.

Addressing sequence slot *s* is therefore not an indirection but an **arithmetic offset**, and
retargeting a whole program at a slot is a rewrite of the tensor pointer table:

```rust
// crates/plowrt/src/exec/amd.rs — kv_rebase
for &(i, stride) in &self.kv_slot_stride {
    let base = self.devp[i].base + stride * slot as u64;
    self.tens_table[i * 8..i * 8 + 8].copy_from_slice(&base.to_le_bytes());
}
```

One 8-byte edit per KV buffer, then one upload of the table. This is why the *single-sequence*
prefill program can fill any slot without a second program and without a kernel change.

*(This is also where a real out-of-bounds GPU write lived: `kv_slot_stride` swept in `kv.blkres`,
a snapshot ring sized by the widest **prefill** bucket rather than by batch, so at B=16 slot 15
wrote past the end. Found by driving all 16 slots at once; fixed by excluding it.)*

### 3.2 A prefill chunk can start anywhere

Prefill is already chunked over a compiled bucket ladder `[128, 512, 1024, 2048, 4096, 8192]`, and
a chunk carries an explicit origin:

* `ChunkStep { prog, c0, clen }` — `c0` is arbitrary,
* `prefill_prepare` writes **absolute** positions, so RoPE and the KV row agree with what later
  decode steps assume,
* `rebase_chunk` sets `FlashPrefill`'s `i[1] = c0 + clen`, so attention covers `[0, c0+clen)` —
  including rows this chunk did not write.

A chunk starting at `c0 = P` over resident KV is therefore **not a new mode**. It is exactly the
situation an ordinary second chunk is already in: attending over KV that an earlier chunk wrote.

### 3.3 The recurrence runs `clen` rows, not the padded bucket width

This is the property that decides where a prefix may be split, and it was the one I expected to be
a blocker. Buckets are quantised, so the last chunk of a prompt is zero-padded — a 1038-token
prompt plans as a 1152-row cover. If those pad rows entered the recurrence, the state after prefill
would include ~114 phantom tokens.

They do not. `rebase_chunk` rewrites **every KDA op's row count to `clen`**:

```rust
} else if KDA_ROW_COUNT_OPS.iter().any(|&k| op == k as u16) {
    d.i[0] = clen;
```

and there is a regression test pinning it (*"EVERY KDA OP'S ROW COUNT BECOMES `clen`"*). Pad rows
write KV that nothing reads; they never touch the recurrence.

**Consequence: a chunk with `clen = P` leaves the recurrence at exactly `P`.** The prefix split
point needs no bucket alignment — it can be any token offset at all.

### 3.4 There is a device-to-device copy

`EngineDevice::memcpy_dtod`, implemented on the HSA backend over
`hsa_amd_memory_async_copy`. Unglamorous, and the thing that makes state checkpointing possible.

---

## Part 4 — The design: a same-slot prefix cache with recurrent-state checkpoints

### 4.1 The core idea

Split the per-sequence state into its two kinds and treat each according to its nature:

| state | nature | how the prefix is reused |
|---|---|---|
| MLA KV (24 layers) | positional, append-only | **nothing to do.** Rows `[0, P)` are already in the slot from the previous request — identical tokens at identical positions produce identical K/V. Just don't overwrite them. |
| KDA recurrence (69 layers) | folded, not positional | **checkpoint it.** Copy `S_P` out when you first reach *P*; copy it back when you want to resume there. |

So the KV half of prefix caching is *free* on this architecture — a consequence of §3.1's flat,
positionally-addressed cache — and the recurrent half is bought with one device-to-device copy.

### 4.2 The protocol

Per slot, keep the last prompt it prefilled and a snapshot of its carried state taken at offset
`P`. The invariant is: **the snapshot describes `cached_prompt[..P]`.**

```
P = longest_common_prefix(new_prompt, cached_prompt[slot])

HIT   (a snapshot exists at P' ≤ P):
        restore_carried(slot)          # S ← S_P'
        prefill [P', n)                # KV rows [0,P') already resident
                                       #   → P' tokens of prefill skipped entirely

MISS  (P is worth caching but nothing is armed):
        prefill [0, P)                 # same tokens as before, just split
        snapshot_carried(slot)         # S_P captured — arms the NEXT request
        prefill [P, n)

NEITHER (P below MIN_PREFIX):
        prefill [0, n)                 # ordinary path, no cost added
```

Three things worth drawing out:

* **A miss costs only the copy.** It prefills exactly the same tokens either way; it just splits
  the run and snapshots at the seam. The steady state is reached after two requests per slot.
* **The split point is the LCP, not a bucket boundary** — legal because of §3.3.
* **`begin_slot` still clears first, then the restore overwrites.** Clear-then-restore, never
  restore-then-clear.

### 4.3 Why *same-slot* is the right scope

The obvious generalisation — a global prefix pool shared across slots, as RadixAttention does — is
the wrong first move here, for a reason specific to §3.1: the KV half is free **only because rows
`[0,P)` are already in *this slot's* block**. Sharing across slots would mean copying KV between
slots, which is far more bytes than the recurrent state and would give back the saving.

Same-slot is also enough for the traffic that matters. A client pool round-robins over slots, so
each slot sees the same shared prefix again and again. On the benchmark below, 48 of 64 requests
(75%) hit with same-slot scope alone.

### 4.4 What the implementation actually is

Modest, and almost all of it is bookkeeping:

* `AmdEngine::snapshot_carried` / `restore_carried` — copy one slot's carried tensors to/from a
  lazily-allocated device buffer. The tensor set is `is_carried_state` (`kv.*state*`) **minus
  `blkres`**, which is per-pass scratch reset by layer 0 every forward pass and sized by the widest
  prefill bucket rather than by batch.
* `AmdTpGroup::{snapshot,restore}_carried` — the same on **every rank**. The recurrence is sharded
  by head, so a rank that skipped a snapshot would resume one prefix behind its peers and the group
  would disagree from the sequence's first token.
* `AmdServe::plan_prefix` — the LCP → `(resume, arm)` decision.
* `AmdTpGroup::plan_span(from, to)` — the chunk plan for an arbitrary span.

### 4.5 One performance bug found underneath it

The first implementation was a net **loss**. The cause was not the bytes:

`memcpy_dtod` was *create-signal → async-copy → **BLOCKED wait** → destroy*, **per call**. A
snapshot is **276 separate tensors** (69 KDA layers × [state + 3 conv windows]) × 8 ranks, so a
single snapshot meant thousands of kernel-level host waits.

Fixed with `memcpy_dtod_batch`: `hsa_amd_memory_async_copy` decrements its completion signal, so a
signal armed at *N* and waited on **once** is a correct barrier for *N* copies.

**Then it was measured rather than assumed** — which mattered, because the assumption was wrong:

| | calls | mean | total |
|---|--:|--:|--:|
| snapshot | 128 | 3728 µs | 477 ms |
| restore | 256 | 3667 µs | 939 ms |

**1.42 s of copies across a 139 s run — about 1%.** I had recorded the 56 MiB restore as the likely
reason the gain was smaller than the prefill it removed. It is not, and that claim was withdrawn.
(`PLOW_PFX_LOG=1`, `crates/plowrt/src/obs/pfx.rs`.)

---

## Part 5 — Results

Carried state is **56 MiB per slot per rank** (276 tensors) — O(1) in context, which is exactly the
property linear attention was chosen for, now working in our favour: a *softmax* model's prefix
checkpoint would be the whole KV block and would scale with *P*.

### 5.1 Correctness first

GSM8K 8-shot greedy, N=200, K3 TP8, B=1, CONC=1 (every request after the first hits):

| | exact_match | median s/q | total |
|---|--:|--:|--:|
| no cache | 196/200 = **0.9800** | 4.88 s | 1027 s |
| **prefix cache** | 196/200 = **0.9800** | **3.70 s** | **825 s** |

**Identical accuracy, 24% lower latency per question, 20% less wall.** A stale prefix or a stale
snapshot shows up here as an accuracy *collapse*, not a crash — which is why this is the gate that
matters rather than a token-identity spot check.

### 5.2 Throughput and latency

`bench_speed.sh`, B=16 packet, `IN_LENS=1024 CONCS=16 NPROMPT=64 OUTLEN=128`, 75% hit rate:

| | TTFT med | ITL p99 | out tok/s |
|---|--:|--:|--:|
| no cache | 2435.0 ms | 1530.3 | 41.1 |
| prefix cache | **1267.8 ms** | 1663.7 | **44.0** |

### 5.3 Composed with chunked prefill

The cache decides *which* span still needs prefilling; chunking decides how that span is split
across scheduler ticks. They compose. GSM8K at B=4 / CONC=4:

| | exact_match | median s/q | total |
|---|--:|--:|--:|
| interleave only | 193/200 = 0.9650 | 14.06 s | 758 s |
| **chunked + prefix cache** | **195/200 = 0.9750** | **10.63 s** | **561 s** |

**26% less wall, accuracy equal or better, zero request errors.**

---

## Part 6 — Limits, and what is deliberately not built

* **Same-slot only.** No cross-slot or cross-request sharing (§4.3). A global radix index would
  need KV movement between slots, which is the expensive half.
* **One snapshot per slot.** No tree of prefixes, no LRU. Two alternating prefixes on one slot
  thrash.
* **`PLOW_PREFIX_CACHE=1`, default off**, pending a decision on the memory (56 MiB × B, allocated
  lazily on first arm — a workload with no shared prefixes never allocates).
* **TP path only.** The single-GPU arm uses the whole-prompt path.
* **The restore has been timed but not optimised.** At ~1% of wall it is not currently worth it.

### What would extend it

1. **Cross-slot sharing** — needs KV movement or genuine paging on the AMD path; `memory/vmm.rs`
   (a `VmmKv` with a `PrefixCache`) already exists in-tree for the CUDA engine and is the natural
   thing to extend rather than rewrite.
2. **A prefix *tree* per slot** — several snapshots, LRU-evicted. Cheap; the copy is already
   batched.
3. **Compressing the checkpoint.** 56 MiB of f32 recurrent state is a natural target for bf16 or
   fp8 storage, at some accuracy risk that would have to be gated on GSM8K.

---

## Part 7 — The transferable conclusion

If you are building serving infrastructure for a linear, recurrent, or hybrid-attention model —
Mamba, RWKV, GLA, KDA, or any of the hybrids now shipping — the single most useful thing in this
document is this:

> **Prefix caching, paged attention, and speculative decoding are all built on the assumption that
> a sequence's state is positional and append-only. Linear attention breaks that assumption, and
> every one of those techniques needs a state checkpoint to be re-established.**

The good news is that the checkpoint is **cheap and constant-sized**, precisely because the state is
O(1) in context. On K3 it is 56 MiB per sequence and ~3.7 ms to move, against ~90% of prefill
eliminated on prefix-heavy traffic.

The bad news is that the same property makes **rejection** expensive, which is why speculative
decoding does not pay on this model while prefix caching does: prefix caching pays the checkpoint
once per *prefix* and amortises it over many requests, whereas speculation would pay it on every
*rejected step*. Same mechanism, opposite economics — and that asymmetry, not the raw cost, is what
decides which technique is worth building.
