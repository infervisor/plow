# Packet geometry: taking the static-geometry cliff out of plow serving (H100, Gemma-4)

Branch `agent/packet-geometry` (from `worktree-gemma4-26b-beat-vllm` @ `16501159`), 2026-09-21.
Code facts are from this tree; measured facts from `plans/gemma4-dense-realtime-tracker.md`, the
campaign ledgers and the JIT emit-timing review (`/home/lava/.claude/jobs/ef9d0e7f/tmp/jit_review/`).

## 0. Ranked recommendation

| # | Item | Verdict | Cost | What it buys |
|---|---|---|---|---|
| 1 | **Ring sized by the per-request chunk, not the launch** (`PLOW_MAX_REQUEST_CHUNK` below `PLOW_MAX_CHUNK`) | **implemented, emit-side only** (`crates/devgen/src/lib.rs`), runtime already enforces it | 1 assert + recipe | 32 slots at chunk 4096 with the attention roles on; ring 2.5 GiB -> 640 MiB per slot; C32 long-prompt TTFT no longer queues on 16 slots |
| 2 | Dynamic slot count at serve time | **already exists**: `PLOW_VMM_LIVE=1 PLOW_VMM_LIVE_RINGS=1` maps each slot's sliding rings on admission (`memory/vmm.rs::VmmRings::ensure_slot`), full layers are VMM-live already | 0 code; a served A/B | emit once at 32-64 slots, memory follows load. Unmeasured on the 12B: the auto rule (`live_rings_for_capacity`) only fires at B >= 64 or ctx >= 128k |
| 3 | In-program sub-chunk pipeline (HNR_i -> FP_i per 1024 rows inside a 4096-row launch) | designed, not built | kernel row-offset field on `HeadNormRope` + q-row window on 4 flash objects, devgen emit loop | single request keeps 4096-row launches on a 2048-row ring (C1 8192/15000 TTFT at the chunk-4096 number with 32 slots) |
| 4 | plowc as a JIT (per-rung compile on demand) | **not worth it as a JIT**; worth it as a 9-second re-emit | — | a devblob re-emit against existing cubins is 3.5-9 s CPU; the 20 min "packet build" is the nvcc object build, which geometry does not change |
| 5 | JIT-lite on single-block packets composed at serve time | **no** | — | loses the one-launch megakernel (48 launch boundaries per decode step on a 10.5 ms step), the cross-layer counter graph and the shared tensor table; nothing it gains is not already given by 1 + 4 |

The static geometry was three couplings: slots x ring (memory), ring x chunk (the dev_isa invariant),
chunk x roles (4096/8192 rungs). Item 1 breaks ring x chunk at the request level; item 2 breaks
slots x memory; item 3 would break ring x chunk for a single request. Nothing needs a new object.

## 1. The facts

Gemma-4-12B (`config.json`): 48 layers, 40 sliding (hd 256, 8 KV heads, window 1024) + 8 full
(hd 512, 1 KV head). Sliding KV = 40 x 8 x 256 x 2 x 2 B = **320 KiB per row per slot**; full
= 16 KiB per row, VMM-mapped on demand (`kv_budget`, block granularity).

| geometry | ring rows | ring/slot | slots | rings | measured peak |
|---|---:|---:|---:|---:|---|
| ladder16k / p12mq: chunk 4096 | 8192 | 2.5 GiB | 16 | 40 GiB | 65.7-67.7 GiB |
| c32-16k / p12c32b: chunk 1024, roles off | 2048 | 640 MiB | 32 | 20 GiB | 45.5-48.5 GiB |
| p12c32c: chunk 2048, roles off | 4096 | 1.25 GiB | 32 | 40 GiB | 66-71 GiB |
| **c32-req1k-16k (this work): chunk 4096, request 1024, roles on** | 2048 | 640 MiB | 32 | 20 GiB | expected ~52-56 GiB |

Ring rule (`kv_ring_rows`): `next_pow2(window + chunk - 1)`; the kernels index `row & (ring-1)`.
The invariant (dev_isa.h "SLIDING-WINDOW KV RING"): a launch writes ALL its K/V rows before any
flash reads, so within one launch a request's rows must not wrap onto the rows its own queries read:
`ring >= window + rows_written_per_request_per_launch - 1`.

Served numbers that frame the problem (12B, vLLM 0.28 / plow, TTFT ms):

| in / C | 16-slot p12mq | 32-slot p12c32b (chunk 1024) | vLLM |
|---|---:|---:|---:|
| 128 / C32 | 1283 | **113.7** | 160.8 |
| 8192 / C16 | **1366** | 2167 | 2024 |
| 8192 / C32 | 8062 | 10314 | 3645 |
| 15000 / C32 | ~12000 | 27062 | 6443 |

The 16-slot packet's C32 TTFT is queueing (a request waits a whole generation for a slot); the
32-slot chunk-1024 packet's long-prompt TTFT is prefill throughput (1024-row launches, roles off).
The geometry that wins both is 32 slots AND 4096-row launches — item 1.

## 2. Q1: ring depth decoupled from the prefill chunk

### What assumed `ring >= chunk`

| site | assumption | status |
|---|---|---|
| `devgen/lib.rs` bucket ladder (was `ring >= window + widest_rung - 1`) | the widest rung is one request | **changed**: with `PLOW_MAX_REQUEST_CHUNK` set, `ring >= window + request_chunk - 1` and the request chunk must be a rung |
| `devgen/lib.rs::kv_ring` | `r >= window + chunk - 1` | already keyed on `request_chunk()` — unchanged |
| `devgen/lib.rs::appended_rungs` | a rung past MAX_CHUNK needs `window + rung - 1 <= ring` | unchanged; 4160/4224 drop out at ring 2048 |
| `runtime/common/dev_isa.h` static asserts | `PLOW_KV_RING >= 1024 + PLOW_MAX_CHUNK - 1` on header constants | the kernels read stride/mask per op from the packet; the constants are not the served ring |
| `plow-asset/packed_prefill.rs::Manifest::validate` (load time) | `stride >= window + write_rows - 1` with `write_rows = max_request_rows` | **already the per-request rule** — the runtime was ahead of the emitter |
| `plow-asset/packed_prefill.rs::plan_with_limit` | rejects a slice longer than `max_request_rows` | enforced on every packed launch |
| `plowrt` `pf_request_max_rows()` (`mux.rs:1880,3991`, `gpu.rs::prefill_chunk`) | per-request rows per launch | every chunker caps on it |
| masked padding (`PLOW_BUILD_MASKED_PADDING`, pad rows slot -1) | pad rows write nothing | no interaction |
| hd256 TMA K/V maps (`tmap_kv(.., kvr, ..)`), sm90 flash `kv & kv_mask` | ring extent = stride, coordinates masked | exercised by p12c32b at 16k positions on a 2048 ring |

So the only thing standing between the tree and "ring from the request chunk" was one emit-side
assert that measured the widest LAUNCH instead of the widest REQUEST SLICE. The runtime contract
(`max_request_rows` in the packed manifest, checked at load against every sliding cache's stride)
is what makes the relaxation safe: a packet emitted this way refuses to load on a runtime that does
not enforce the cap (the manifest section is mandatory when `max_request_chunk` is set,
`lib.rs:7809`).

### Cost model

* Per slot: ring 2048 = 640 MiB vs 2.5 GiB (chunk 4096) — 32 slots = 20 GiB vs 80 GiB.
* Aggregate prefill at C16/C32: unchanged from the 16-slot packet — a 4096-row launch holds four
  1024-row slices from four requests (`admit` FairSplit/Greedy, `PLOW_PF_INTERLEAVE=0` = uncapped
  tick budget), the GEMMs run at M=4096, the GLU/attention roles apply.
* Single request (C1): 1024 rows per launch. 4096 in = 4 launches vs 1: measured C1 TTFT 1024 in
  47 ms x4 vs 171 ms -> ~+10-15 %; 8192 in ~8 x 50 = ~400 vs 348 ms. The realtime profile keeps
  the 16-slot chunk-4096 packet for C1/C4 unless item 3 lands.
* Extra launches at C32 for a 15000-token request: 15 slices instead of 4, but each launch carries
  four requests, so launches per token are equal; the cost is host-side planning only.

### The chunk-2048 "doubled ring slows decode" null (p12c32c)

The ledger attributes 128/C16 TPOT 13.05 -> 17.97 ms to the ring. The decode kernel's sliding
window read is `[len - window, len)` (`d_flash_decode`, op_attention.cuh:777) — independent of the
ring; nothing in the step scales with ring rows. What differed: p12c32b's decode object was built
at 04:45 from `3851e88a` (1.49 MB), p12c32c's at 10:57 from a DIRTY `4af4c85f` (2.16 MB; p12mq's
from the same day is 1.83 MB). `plow_config.h` differs only by `PLOW_NV_GEMV_MMA_PAIR 1`. The A/B
below re-emits both chunks from the same tree against the same objects so that only the ring
differs. RESULT: see §6.

### Item 3 (designed): sub-chunk pipeline inside one launch

Within a launch of C rows and a ring of R rows, a request's rows [c0+iS, c0+(i+1)S) may be written
only after the flash for rows below c0+(i+1)S-R+window-1 has finished. With R=2048, window 1024:
S=1024, and the per-layer chain becomes HNRk/v_0 -> FP_0 -> HNRk/v_1 -> FP_1 -> ... (4 stages for
C=4096; HNR_{i+1} is WAR-dependent on FP_i because at R=2S it overwrites rows FP_i reads).
Cost: 3 extra fan-in/fan-out points x 40 sliding layers = 120 x ~5-8 us = ~0.6-1 ms on a ~100 ms
4096-row chunk; FP_i still has 16 q-tiles x 16 heads = 256 items > 132 SMs.
Kernel touch points: `d_headnorm_rope` needs a row offset (`i[7]`, free; add to `t` after the
`w / nhead` split so x, pos, pfslot and the out row all shift); `d_flash_prefill_mux`
(interp_sm120.cu:1520) and the four role objects need a q-row window (start tile, count) and the
span table clipped to it; `FlashMerge` the same window. All non-GEMM prefill ops share the fat
`pfpackedseg` object at the 255-register cap — validate on a block (see the fat-object memory
note) before a packet. Emit side: `emit_prefill`'s sliding-layer HNR/FP block becomes a loop over
`i in 0..C/S` with `Dep::Coarse` chains. Not started: item 1 covers the serving cells first.

## 3. Q2: dynamic slot count at serve time

Already in the tree, opt-in:

* `PLOW_VMM_LIVE=1` — full-attention KV lives in a VMM pool mapped per block (`vmm_live_bringup`).
* `PLOW_VMM_LIVE_RINGS=1` — every sliding ring tensor is VA-reserved at the packet's `dbatch`
  and physically mapped per slot on `ensure_slot` (admission) / unmapped on `release_slot`
  (`memory/vmm.rs:372-465`; map unit = `PLOW_VMM_BLOCK_MIB`). `live_rings_for_capacity`
  auto-enables it only at `batch >= 64 || max_ctx >= 131072`.
* Requires prefix caching off (`vmm_bringup` rejects otherwise) — the vLLM-matched recipes have
  it off already.

What still assumes the static slot table: the decode rung programs index KV batch-major
`[dbatch][kvh][ring][hd]` with a per-rung `decode_slot[b]` physical map, so a packet must be
emitted at the MAX slot count (its ladder must include the widest rung, e.g. 64) — that is emit
time, not serve time, but it is one packet. `kv_budget` admission already charges only what will
be mapped ("with live rings every cache maps lazily and the average stays the honest bound",
`gpu.rs:5382`). Nothing in the batch ladder or the mux depends on the rings being resident.

Unmeasured: the per-admission `cuMemMap` cost (640 MiB / 2 MiB = 320 map calls per slot at ring
2048) and whether a mapped-on-demand ring is slower to read than a cudaMalloc'd one. Test: serve
the item-1 packet with both knobs, compare `peak_mem_mib` and TTFT/TPOT at C1 and C32 against the
same packet without them. This is the cheapest next step after item 1 and would let the ladder go
to 64 slots (40 GiB of rings only when 64 are live).

## 4. Q3: plowc as a JIT

Measured (jit_review, CPU only, one core): full 12B devblob emit 9.09 s with verify + hipcc probe,
3.47 s without; **~15 ms per prefill bucket, 17-21 ms per decode rung** for the instruction
program itself, ~0.2 s per program of manifest/audit/digest post-processing, ~0.45 s per new
prefill program of Lean verify (cached for identical payloads). The cubin build (nvcc) is 207 s and
is what the ~20 min packet build is made of (objects script + role emit).

* Geometry (slots, chunk, request chunk, ring, ladder) does NOT change any cubin: the kernels take
  stride/mask/rows per op from the packet. So "a packet per geometry" is a 4-9 s devblob re-emit
  against existing objects (`reemit.py` pattern, `--emit devblob`), not a JIT.
* What a true per-rung JIT would need: (a) the tensor table is shared and sized at the widest rung
  (`declare(dbatch)`, `chunk_rows`) — a JIT rung must fit the pre-sized envelope or every program's
  handles change; (b) the packet hash pairs the decode object (`decode_object::check_image`,
  `spec.matches_image`) and the LIVE-KV/packed manifests digest every prefill program
  (`program_digest`), so appending a program means re-emitting the manifests; (c) verify off or
  cached. Feasible at ~0.3 s per rung, but the serving loop already selects among 10 prefill + 5
  decode programs, and the traffic-shaped gain is a few rungs no one has asked for.
* Verdict: expose the re-emit, not a JIT. A `plowc --reemit-geometry` (devblob only, reuse
  `objects/`) makes a geometry change a 10 s operation; the serve side would still restart.

## 5. Q4: JIT-lite on single-block packets

`gemma4 --block L` emits one layer in <1 min (objects ~6 min); `block_run` drives it with its own
tensor table and KV. Composing the full model from block packets at serve time would mean per layer:
its own program, its own counter block, its own activation/KV tensors, and one launch per layer.
Lost: the single cooperative launch whose counter graph overlaps layer L's tail with L+1's head
(542 DevInsts in one launch at B=1; 48 launch boundaries would add ~48 x 5-10 us = 0.25-0.5 ms to
a 10.5 ms step and worse under multistep), the shared tensor table (each block's residual would be
a copy or an alias table), segment-major prefill across layers, and the packet-level manifests
(packed prefill, LIVE-KV, decode objects) that the runtime validates as one unit. Gained: seconds
per geometry — which item 4's re-emit already gives without losing anything. No.

## 6. Implementation and validation

Commits on `agent/packet-geometry`:

* devgen: `PLOW_MAX_REQUEST_CHUNK` below the widest rung sizes the ring from the request chunk
  (`crates/devgen/src/lib.rs`, the bucket-ladder assert; `docs/flags-reference.md`). Default
  behaviour byte-identical (no packet today sets the knob below `PLOW_MAX_CHUNK`; it panicked).
* recipe `scripts/campaign/recipes/gemma4-12b.h100.bf16-c32-req1k-16k.toml`: ladder16k + ladder
  to 32 + `PLOW_MAX_REQUEST_CHUNK=1024`, both profiles.

Validation plan: (1) ring A/B on the c32c null (step_bench B=16, same tree/objects, chunk 2048 vs
1024); (2) packet build after 16:45 UTC; (3) served `high_concurrency` (C16/C32) then `realtime`
(C1/C4) ladders vs the vLLM 0.28 uniform baseline and the p12mq / p12c32b rows.

### Results

(filled in below as they land)
