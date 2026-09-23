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
| **c32-req1k-16k (this work): chunk 4096, request 1024, roles on** | 2048 | 640 MiB | 32 | 20 GiB | **45.1-53.0 GiB measured** |

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
differs. RESULT (§6): **the ring is not the cause** — 17.8 ms at both ring sizes.

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
* recipe `scripts/campaign/recipes/gemma4-26b-a4b.h100.bf16-c32-req1k-16k.toml`: the same lever on
  the 26B (ctx16k + ladder to 32 + request 1024; 1.76 -> 0.59 GiB per slot). Unmeasured.

Validation plan: (1) ring A/B on the c32c null (step_bench B=16, same tree/objects, chunk 2048 vs
1024); (2) packet build after 16:45 UTC; (3) served `high_concurrency` (C16/C32) then `realtime`
(C1/C4) ladders vs the vLLM 0.28 uniform baseline and the p12mq / p12c32b rows.

### Results

* **Emit check (CPU, 15:09 UTC)**: the recipe's base packet emits with my plowc (`--emit devblob`):
  prefill rungs `128 256 512 1024 1088 1152 2048 4096`, decode `1 2 4 8 16 32`; the packet's
  LIVE-KV manifest carries 40 sliding caches at `stride 2048, window 1024` and 8 full caches at
  `stride 16384`, packed-prefill `max_request_rows 1024`; the same `Manifest::validate` the runtime
  runs at load passed at emit. Before the change this emit panicked at the ladder assert.
* **26B emit check (CPU)**: the 26B recipe emits too — 25 sliding caches at `stride 2048`, 5 full at
  `stride 16384`, `max_request_rows 1024`, rungs `128 256 512 1024 1152 2048 4096` / decode `1..32`.
* **Ring A/B (GPU, 15:30 UTC, `step_bench` B=16 ctx 128, 64 steps, arms interleaved r4096 r2048
  r4096 r2048)**: the c32-16k recipe re-emitted from the campaign tree (`16501159`) against
  p12c32c's objects, `PLOW_MAX_CHUNK=2048` (ring 4096) vs `1024` (ring 2048), decode object built
  by the same emit: **17.824 / 17.778 ms vs 17.803 / 17.793 ms** (sd 0.04-0.05). Ring depth is
  worth 0.0 ms per decode step. Both arms sit at p12c32c's served 17.97, not p12c32b's 13.05: the
  regression the ledger charged to "the doubled ring" is in the decode OBJECT/tree of the 09-21
  ~11:00 build (1.49 -> 2.16 MB `interp_sm90a.cubin`, `PLOW_FA_MMAQK` unset). p12mq (same tree,
  `PLOW_FA_MMAQK=3`) serves 11.83 at 128/C16, so the tensor-core attention path is unaffected; the
  legacy FA path in the fat decode object is what got slower. Not root-caused here (out of scope);
  it means a chunk-2048 / ring-4096 32-slot packet was never actually refuted, and any packet
  emitted today should carry `PLOW_FA_MMAQK=3` (the req1k recipes do).
  Caveat: the A/B ran without the exclusive CPU-quiet lock (a shared-lock `cargo build` from
  another agent was idling the leased GPU behind `quietx.sh`); the arms are GPU-bound kernel
  steps, interleaved, sd 0.05 ms.

#### Served `high_concurrency` (p12rq, 10 cells; TTFT ms / TPOT ms / tok/s; peak GiB in the last column)

| in / C | **p12rq req1k (32 slots, chunk 4096)** | p12r (16 slots, chunk 4096) | p12c32b (32 slots, chunk 1024) | vLLM 0.28 e2e | p12rq peak |
|---|---|---|---|---|---:|
| 128 / 16 | **108 / 11.83 / 1271** | 78 / 11.85 / 1286 | 68 / 13.05 / 1179 | 101 / 10.96 / 1370 | 46.0 |
| 128 / 32 | **199 / 14.56 / 1996** | 1292 / 11.59 / 1297 | 114 / 16.63 / 1808 | 154 / 11.71 / 2487 | 47.0 |
| 1024 / 16 | **336 / 16.79 / 826** | 323 / 16.94 / 824 | 240 / 18.46 / 783 | 408 / 13.79 / 946 | 47.0 |
| 1024 / 32 | **591 / 23.99 / 1110** | 2081 / 16.79 / 860 | 568 / 27.01 / 986 | 693 / 18.31 / 1351 | 47.0 |
| 4096 / 16 | **1126 / 30.22 / 410** | 678 / 30.84 / 443 | 806 / 35.61 / 376 | 1164 / 22.79 / 503 | 47.5 |
| 4096 / 32 | **2142 / 48.51 / 484** | 4311 / 31.82 / 444 | 2420 / 62.34 / 372 | 1991 / 37.88 / 598 | 49.0 |
| 8192 / 16 | **2168 / 45.84 / 254** | 1265 / 50.39 / 265 | 1790 / 67.87 / 192 | 1955 / 38.04 / 301 | 48.5 |
| 8192 / 32 | **4291 / 79.00 / 280** | 7552 / 51.44 / 265 | 10314 / 71.71 / 190 | 3648 / 67.18 / 334 | 51.0 |
| 15000 / 16 | **4063 / 83.01 / 139** | 2449 / 86.47 / 151 | 10443 / 86.28 / 93 | 3086 / 67.79 / 174 | 49.5 |
| 15000 / 32 | **8271 / 143.88 / 151** | 13646 / 87.34 / 152 | 27062 / 88.62 / 90 | 6440 / 123.45 / 184 | 53.0 |

#### Served `realtime` (p12rq, 10 cells; TTFT ms / TPOT ms / tok/s; peak GiB in the last column)

| in / C | **p12rq req1k (32 slots, chunk 4096)** | p12r (16 slots, chunk 4096) | p12c32b (32 slots, chunk 1024) | vLLM 0.28 e2e | p12rq peak |
|---|---|---|---|---|---:|
| 128 / 1 | **19 / 10.79 / 92** | 18 / 10.53 / 94 | - | 31 / 10.54 / 94 | 45.1 |
| 128 / 4 | **38 / 11.08 / 354** | 39 / 10.80 / 362 | - | 56 / 10.56 / 366 | 45.3 |
| 1024 / 1 | **47 / 10.86 / 90** | 47 / 10.61 / 92 | - | 47 / 10.61 / 92 | 45.3 |
| 1024 / 4 | **103 / 12.11 / 312** | 103 / 11.83 / 318 | - | 131 / 11.14 / 331 | 45.3 |
| 4096 / 1 | **188 / 10.90 / 81** | 170 / 10.64 / 84 | - | 171 / 10.62 / 84 | 45.3 |
| 4096 / 4 | **418 / 14.53 / 225** | 332 / 14.10 / 241 | - | 468 / 12.14 / 255 | 45.5 |
| 8192 / 1 | **387 / 10.93 / 72** | 360 / 10.67 / 75 | - | 348 / 10.62 / 75 | 45.5 |
| 8192 / 4 | **706 / 18.85 / 164** | 718 / 17.50 / 174 | - | 995 / 13.60 / 188 | 45.8 |
| 15000 / 1 | **774 / 10.99 / 59** | 750 / 10.74 / 60 | - | 673 / 10.63 / 63 | 45.8 |
| 15000 / 4 | **1329 / 26.26 / 109** | 1478 / 22.72 / 117 | - | 1667 / 18.13 / 129 | 46.0 |

#### Reading (both profiles, 17:01-17:38 UTC, quiet host, 0 faults)

* Memory: 45-53 GiB at 32 slots with the attention roles on (p12r 65-69 GiB at 16 slots).
* C32: the best plow packet in every cell but 128/1024 TTFT (p12c32b 114/568 vs 199/591): 8192/C32
  7552 -> 4291 ms, 15000/C32 13646 -> 8271 (p12c32b 10314/27062); decode 1996 tok/s at 128/C32 vs
  p12c32b's 1808. Still behind vLLM at 4096+/C32 (1991/3648/6441) — the adoption gate (8192/C32 <
  3648) is NOT met. Beats vLLM at 1024/C16 (336 vs 408) and 1024/C32 (591 vs 693).
* C16: +66-72% TTFT vs p12r at 4096/8192/15000 (1126/2168/4063 vs 678/1265/2449); TPOT level or
  better. A 4096-row launch carries four requests' 1024-row slices (Greedy; the turn is held until the
  pack's last request finishes, `mux.rs last_finished`), so four requests finish together after four
  launches instead of one per launch. The 128/C16 +30 ms (108 vs 78) with identical launch shapes is
  NOT explained by the cap — unattributed (PACKLOG diag prepared, `diag.sh`).
* C1/C4: 128/1024 level; C1 4096/8192/15000 +11/+7/+3% (188/387/774 vs 170/360/750: 1024-row
  launches per request, as the cost model predicted); C4 4096 +26% (418 vs 332), 8192/15000 -2/-10%
  (706/1329 vs 718/1478). TPOT +0.2-0.4 ms everywhere (the 32-slot decode object).
* Verdict: req1k REPLACES p12c32b as the 32-slot / C32 serving packet; it does NOT replace p12r for
  C1/C4/C16. One packet winning both needs item 3 (in-program sub-chunk pipeline: a single request
  keeps 4096-row launches on a 2048-row ring). The live-rings A/B (Q2) was gated on adoption and not
  run. Ledger: cell `gemma4-12b.h100.bf16-c32-16k` (both sessions).

#### C16 attribution from the client per-request data (no PACKLOG session: queue had 5-6 waiters)

* 4096/8192/15000 C16: steady-state per-request TTFT p50 888 ms (req1k) vs 379 (p12r); first wave
  1779 vs 1696. Each request's prefill is four 1024-row slices in four consecutive shared launches, so
  a request finishes after ~4 launches instead of 1 — the request cap's packing, TPOT unchanged.
* 1024/C16: steady p50 331 vs 320, first ITL 162 vs 160 on both — level.
* 128/C16: a CONSTANT +42-44 ms on every staggered arrival (steady p50 102.3 sd ~1 vs 58.0), all of
  it before the first token (first ITL 11.7 vs 11.4, TPOT 11.83 vs 11.85). Not the admission rung
  (both sat at rung 16, no transitions), not launch shape (129-row prompt, bucket 256, token-batch
  route firing on both). Candidates: the unified launch's decode-row staging sized by the packet
  batch (`TokenBatchStaging::with_capacity(pf_max_rows, batch)` = 32 vs 16), the per-row quota
  `pf_max_rows / active`, or the seg-graph warm state (`buckets=8` vs `10`). `diag.sh` (two
  `PLOW_PF_PACKLOG=1` sessions) decides it; not run.
* Admission-rung flapping on the 32-rung packet at C16 for prompts >= 1024: with 16 live and 1-3
  queued the controller widens 16 -> 32 (Backlog) and narrows back (LowLoad) every 2-15 s. Not the
  128/C16 gap, but serve C16 on a 32-rung packet with `PLOW_DECODE_MAX_RUNG` capped at the profile's C.

## §7 l8192 verdict (2026-09-22, agent/packet-geometry-l8192, integrated in `75ce0d26`..`27d35315`)

* Block, one packet, `--pf-cap` 4224 vs 8192: 12B sliding -6%, full -4%; 26B sliding -10%, full -7%.
  Dense GEMM per-row cost does not improve at M=8192; the 12B gain is attention-role coverage + launch
  glue, the 26B gain is the MoE grouped GEMM (2.48 -> 2.16 ms, 512 vs 264 tokens/expert).
* Served: 12B l8192 not adopted (P99 -3..-7% but mean TTFT +19..44% at C16). 26B 4224-slice arm OOMs at
  8192/C16 (free at load 4.78 -> 3.63 GiB, MoE Lt prefill scratch 404 -> 660 MiB; admission blind to it).
  26B r3072 (4096 ring): peak 64.5-68.2 GiB, 15000/C16 TTFT 3015 -> 2205, but C1 TTFT +5..17%.
* Adopted: GQA2 role on the 4160/4224 rungs (block 3.45 -> 3.32 ms per 4224-row launch).
* Mux: unified prefill + decode rows stay in the prefill's bucket; trim only when the spill exceeds
  PLOW_PF_CHUNK_COST (4224+3 on an 8192 rung), never 1024+1.

### Item 3 revision (2026-09-23): no kernel changes, and the emit ordering is not a blocker

Two corrections to the design above, both verified by reading the kernels and the emit path.

* **No kernel touch points.** The memo asked for a row offset on `d_headnorm_rope` and a q-row
  window on `d_flash_prefill_mux` plus the four role objects. Neither is needed:
  * Flash already self-derives its q origin — `q_pos0` (`i[4]`) is OVERWRITTEN in the packed path
    with `qp0 = kvlen - qlen` (`op_attention_sm90.cuh:356,717`). Handing it a span table clipped
    to the stage (`plan_stage`: `rq0+taken`, `len`, `slot`, `start+taken+len`) therefore moves the
    q window with no new operand. This is also why `q_pos0` cannot carry a stage offset itself.
  * HNR already skips rows outside the stage — `d_headnorm_rope` drops any row with `pfslot[t] < 0`
    (`op_norm.cuh:781`, fp8 arm `:985`), which is exactly what `stage_slots` writes.
  This keeps the change out of the fat `pfpackedseg` object at the 255-register cap, so the
  in-situ-vs-isolated hazard does not apply.

* **Per-stage tensor declaration does not need hoisting.** `pf.request.slot` / `pf.request.table`
  are declared AFTER the programs are emitted (`devgen/src/lib.rs:9717-9727`) and reach the
  instructions by load-time operand patching (`packed_prefill::bind_request`), not by being in
  scope at emit time. The per-stage tensors ride the same route: declare
  `pf.request.slot.{i}` / `pf.request.table.{i}` in that same block, then have devgen rewrite the
  stage-i sites to those handles and record `stages[i]`. `bind_request` already skips anything
  `is_staged_site` matches (`247dae64`), so load time leaves them alone. The emit loop only has to
  hand back which instruction indices belong to which stage — a side table, not a reordering.

Remaining for item 3: the `HNR_i -> FP_i` emit loop with `Dep::Coarse`, the runtime per-stage
table fills, the packet build, and the C32 ladder.
