# Scaling audit — Kimi-K3 MI355X promotions of 2026-09-03/04

Question: do the promoted changes scale to long context (>8192 prompt, 16K/32K) and to
batched/concurrent serving, or only to the gated cell (8192→1024, C1, decode batch b=1, TP8)?

Method: CPU-only source audit of `codex/amd-agent-harness` (no GPU). Line refs are to that
tree. Numbers from `perf-data/kimi-k3-mi355x-campaign-summary-20260904.md` (summary).

## Verdict

* Every gate ran one cell: prompt 8192 (one full bucket, no ragged tail, no continuation
  chunk), 1024 output, C1 on a `PLOW_DECODE_BATCH=1` blob (one decode rung, t=1).
* No promoted change is keyed to KV length. Long context is mostly "more 8192 chunks" —
  except two silent-wrong paths in the KDA carry route (below) that a 8192-multiple prompt
  cannot hit.
* Four of the decode wins are **t==1-only by construction** (tagged XReduce, grouped-MoE
  route, GLU UN=7, MLA fold DVT). They neither apply nor break at b>1; the throughput regime
  gets none of their ms/token.
* The throughput regime today is not a tuned regime at all: the only C128 8192-prompt cell is
  1304 total tok/s at TPOT 1483 ms / TTFT 383 s vs vLLM 10205 tok/s / 107 ms / 5.4 s
  (7.8× / 13.8× / ~70×). Its bottleneck is serialized prefill chunks stalling all decode
  streams, not any kernel these promotions touched.

## Shared facts

* Prefill bucket ladder `[128,512,1024,2048,4096,8192] ∩ ≤ctx` — `crates/devgen/src/mla.rs:4773-4778`;
  `MAX_CHUNK_MAX = 8192` — `crates/devgen/src/lib.rs:2365`. A 32K blob has the same six programs.
* Ragged chunk plan (default on, `crates/plowrt/src/config.rs:538`): widest bucket while
  `rem > 8192`, then the SMALLEST bucket ≥ remainder — `crates/plowrt/src/exec/amd.rs:4836-4859`.
  16384 → [8192,8192]; 9000 → [8192, 1024 @ clen 808]; 8400 → [8192, 512 @ clen 208].
* Per chunk `patch_prefill_rows` rewrites `i[0]=clen` on interpreter KDA ops and rebases only
  the three KDA carry-family lean routes' `t` — `amd.rs:11984`, `amd.rs:1515-1523`. All other
  lean routes run at bucket width (pad rows unread; perf only).
* Lean routes are built once at load from slot-0 tensor bases — `amd.rs:9795-9812`;
  `kv_rebase` rewrites only the device tensor table the interpreter reads — `amd.rs:12322-12340`.
* K3 emits one-shot `XReduce` and the grouped-MoE / GemvGlu decode ops only when `t == 1` —
  `crates/devgen/src/k3.rs:1352, 1575, 1111, 349`. Batched rungs use `XReduceTwoShot` and the
  prefill-style MoE ops.
* max_ctx is a blob property (`plowc --max-ctx`, `in.pos` bytes/4 — `amd.rs:7322, 9783`;
  `k3.rs:2851`); `plowrt serve` has no `--max-ctx/--batch` flags (`crates/plowrt/src/main.rs:40-65`).
  Prompt ≥ max_ctx → `RuntimeError::Device` (`serve/engine.rs:733, 921`); padded cover > max_ctx →
  429 (`amd.rs:12397-12424`). MLA KV ring is `ctx` rows per slot, flat `B × max_ctx`
  (`amd.rs:4820-4830`, `memory/vmm.rs:17-22`).

## Per-change findings

### 1. Register-resident KDA carry — `PLOW_KDA_CARRY_REGSTATE` (default on)
* (a) Compile-time D=V=128, BT=64 (`runtime/amd/op_kda_carry_regstate.h:27`); `n_chunks=(t+63)/64`
  runtime, rows masked for T%64≠0 (`op_kda_carry_regstate.h:195-233`); grid heads×8, T-independent
  (`amd.rs:1575-1577`). Selected on buckets with `i[0] >= 512` (`crates/packet/src/devbuild.rs:1902,
  2506`) and runtime `i[0]>=512, dim=V=128, qpre` (`amd.rs:1559-1573`). Prefill-only
  (`amd.rs:7664-7666`). State read/written through the tensor → continuation chunks carry naturally.
* (b) Outside the cell:
  * **Ragged tail 129..511 tokens on the 512 program**: route `t` is rebased to clen
    (`amd.rs:1515-1523`, `11984`); kernel early-returns on `t < 512`
    (`runtime/amd/kda_chunk_carry_regstate.hip:39`); host launches unconditionally with no
    interpreter fallback (`amd.rs:11132-11150`). `out` stale, `state` not advanced → wrong KDA
    output for the chunk and a wrong recurrent state for the whole decode. Reachable for any prompt
    with `n mod 8192 ∈ [129,511]` and any standalone 129..511-token prompt. The pre-regstate
    `kda_chunk_key_factor_carry.hip` has no such guard, so this is new with the promotion.
    (`kda_chunk_intra_wave_items.hip:35` / `intra_cached.hip:21` share the guard but their `t` is
    not rebased, so they run at 512 and stay correct.)
  * **Batch>1 blob, prefill into slot ≠ 0**: `prefill_slot` rebases `kv.{l}.state` per slot in the
    device table (`amd.rs:10262-10272`, `12332-12335`) but the route's `state` pointer is the
    load-time slot-0 address (`amd.rs:1612-1625`). Concurrent prefill into slot s reads/writes
    slot 0's recurrence while the interpreter ops and decode use slot s. Pre-exists in
    `KdaChunkKeyFactorCarry` (`amd.rs:1494`); invisible at C1.
  * 16384 / 32K as exact multiples: correct, just more chunks.
* (c) **HIGH** — two silent-wrong paths the cell could not observe. Long-context claims with
  arbitrary prompt lengths and any C>1 serving both hit them.

### 2. f32-mix AttnRes prefill object — `PLOW_ATTNRES_F32MIX` (default on)
* (a) Compile-time `HID=7168`, `AR_MAXB=8` (`runtime/amd/attn_res_f32mix_gfx950.hip:48,52`;
  out-of-contract geometry poisons `out` with NaN, `:150-156`). Grid `min(768,T)` persistent
  (`amd.rs:647-654`); ring `kv.blkres [T][NBCAP][HID]` at widest bucket (`amd.rs:10240-10245`).
  Selection `i[0]>=256`, `i[1]==7168`, `i[2]<=8` (`devbuild.rs:489-509`), prefill programs only
  (`amd.rs:7739-7741`).
* (b) `args.t` is never rebased: a ragged tail on the 8192 program runs 8192 rows (2× work, pad
  rows unread — same as the interpreter). 128 bucket and all decode (T=b≤128<256) use the
  interpreter BF16-seam arm (`runtime/amd/interp.hip:2814-2817`) → numerics contract differs by
  prompt length: ≤128-token prompts and decode are BF16-seam, ≥256 prefill is f32-mix. Slot-safe
  (operands are activations + blkres, excluded from slot rebase). No KV dependence.
* (c) **LOW** correctness; note the GSM8K 122 vs 124 signal was measured only on the mixed
  regime and the ragged-tail row waste (up to one bucket per tail).

### 3. Tagged one-shot decode XReduce — `PLOW_XR_TAGGED` (CMake default on)
* (a) Slot 20480 B fixed (`runtime/common/dev_isa.h:1343`, `crates/plowrt/src/exec/tp.rs:170`);
  max width 7680 bf16 (`tp.rs:172`); hidden 7168 → 19,120 B/slot (1.3 KiB headroom). `PLOW_XRT_MAXR
  8` peers hard-coded (`runtime/amd/op_collective.h:579, 643-646`). Slot = gate&1, tag = gate+1 per
  collective site (`op_collective.h:629-632`); uniqueness across tokens relies on host per-token
  zeroing (`tp.rs:259-264, 573-584`; `XctrReset::Host` hard-coded `amd_tp.rs:633`).
* (b) b>1: not reached — K3 emits one-shot only at `t==1` (`k3.rs:1352,1575`), batched rungs use
  two-shot (`lib.rs:2750-2790`). Loader checks every program: `n<=hidden`, `hidden<=7680`, gather
  `row_w==n`, `gate<0xffff`, alternating parity (`amd_tp.rs:1668-1710`, called `amd.rs:7902-7910`)
  → a t>1 one-shot (only reachable with `PLOW_XR2_GATHER=0`) is refused at load, never
  overflows. KV length: no dependence. TP<8 fine; **TP>8 silently drops ranks ≥8** (no host guard).
  Region id comes from the compact audit: `PLOW_TP_AUDIT_COMPACT=0` or a fine-xctr blob leaves
  `fj[2]==0` → device `__builtin_trap` (`interp.hip:4539-4542`), not a load error.
* (c) **LOW** for scaling (inert at b>1, fail-closed). MED for the audit-compact precondition
  (GPU trap) and unguarded TP>8. Gain is C1/B1-only.

### 4. Split-tile MLA merge-fold — `PLOW_MLA_FOLD_DVT=8` (default on)
* (a) Arm gate `bh*8 <= nblk(256)`, V<256, V%32==0 → DVT=8 else DVT=16 (`interp.hip:1191-1207`),
  `bh = n_batch*nh_l` (12 per rank at TP8). `nsplit` is runtime `i[4]`; LDS ≈700 floats
  (`op_attention.h:5279-5307`); partials `tt*nh*dk*nsplit` (`k3.rs:1821-1827`), batch-scaled.
* (b) KV growth: K3 `nsplit` is a constant 64 (`kimi_k3.rs:1090-1100`, `let _ = ctx`) unless a
  tunedb attention record matches the current digest; flash splits the live span
  (`op_attention.h:2033-2037`) → 128/256/512 KV rows per split at 8K/16K/32K, fold always merges
  64 partials. Batch: B1 → DVT=8, B2 → DVT=16, **B≥3 (`bh*8 > 256`) skips the arm** and runs the
  pre-existing `d_mla_merge_fold<512,128>` (`interp.hip:1216`). Bit-identical by construction.
* (c) **LOW** correctness; gain is B1/B2-only; constant ns=64 is a known long-KV perf gap
  (`mla.rs:517-523`).

### 5. ASAP global-queue window order (default on, `PLOW_GQ_ORDER=emit` rollback)
* (a) Shape-independent: unit-cost list schedule over the op DAG, stable-sorted within each
  (segment, XCD) window (`devbuild.rs:630-644, 2538-2551`); every `Builder::finish` (all buckets,
  all rungs) gets it.
* (b) Applies everywhere; topological order preserved (`devbuild.rs:618-624`). Only the b=1 decode
  graph was measured; other rungs/buckets get a different, unmeasured reordering.
* (c) **LOW**.

### 6. Grouped-MoE decode route via TuneDB rule (`crates/tunedb/src/moe_decode.rs`)
* (a) Cell key `hardware|n_cu|decode_rung|topk|hidden|inter_local|experts|weight_enc`
  (`moe_decode.rs:40-64`). Store: exactly 2 records, both `decode_rung:1, topk:16, hidden:3584,
  inter_local:384, experts:896, mxfp4, n_cu:256`, digest `gfx950-aa66f160ef65781f`
  (`tuning/amd/gfx950/mi350x/moe_decode_measurement.jsonl`). Evaluated per rung
  (`crates/devgen/src/mla/kimi_k3.rs:1219, 1239-1261`; `lib.rs:1103-1150`).
* (b) b>1 rung: no record → `FixedFallback` → interpreter, stderr note only (`moe_decode.rs:128-137`,
  `lib.rs:1152-1178`). More fundamentally the standalone pair exists only at `t==1`
  (`k3.rs:1111-1152`); t>1 emits `MoeGroupGluPf/DownPf`, never classified `GroupedMoeMxfp4`
  (`amd.rs:387-395`). Grid 768 = blocks×3 with a one-row `x` (`amd.rs:2484`,
  `runtime/amd/moe_decode_grouped_mxfp4.hip:8-16`). Other TP → `inter_local` changes → cell miss.
  Digest mismatch → silent interpreter. Rule says standalone but ELF absent (built only with
  `PLOW_MOE_DECODE_STANDALONE=1`, `scripts/build_gfx950.sh:287-295`) → loud load failure
  (`amd.rs:8526-8541`).
* (c) **LOW** for scaling (silent fallback). **MED operational**: pinned to one build digest.

### 7. GLU GEMV K=7168 UN=7 (`runtime/amd/op_gemm.h:4410-4414`, no flag)
* (a) Keyed on K only; MM is the object's compile-time `PLOW_GEMV_MM`. Reached only through
  `GemvGlu`, which K3 emits only when `fuse_shared_glu(t,·)` → `t==1` (`k3.rs:347-353, 1453-1470`).
* (b) b>1 rungs: shared expert is the unfused Gemv/Gemv/Glu triple → rung never hit. In a ladder
  blob the b=1 rung hits it on the wide object; "resources unchanged" was measured at MM=1
  (`op_gemm.h:2310`); MM=8 is already at 256 VGPR / 19 spills (`build_gfx950.sh:349-351`),
  UN=7 there unmeasured.
* (c) **LOW** correctness; low-med perf on wide objects.

### 8. GemmWide c8 tile `8192x1536x7168` (default OFF, `--emit-gemm-wide-c8-shape`)
* (a) Exact `MxNxK` string + `blocks == n_cu` + bf16 + measured c8 winner (`lib.rs:395-411`;
  `emit_config.rs:633-652`); interp requires `(M/128)*(N/384) == nblk` (`interp.hip:1611-1612`).
  Shape = prefill `q_a_proj` at T=8192 (`k3.rs:1886`).
* (b) T=4096: 128≠256 → untagged c2 128×256×64 via `pick_tile` (`interp.hip:1618-1621`); only the
  8192 bucket carries the tag; T=16384 chunk does not exist. Silent fallback.
* (c) **LOW** (opt-in; hard-coded to the benchmark shape by design).

### 9. Align-parallel MoE — `PLOW_MOE_ALIGN_PAR` (default OFF)
* (a) `align_par = flag && t >= 1024` (`k3.rs:1215`); `ALIGN_BLOCKS=64` fixed (`k3.rs:1214`);
  router blocks `t.min(n_cu)` regardless (`k3.rs:1249`). Prefill only.
* (b) T<1024 buckets: single-block align (`k3.rs:1283`), silent. Kernel grid-strides `T*k`
  (`runtime/amd/op_moe.h:2094-2100`), nothing sized to 8192.
* (c) **LOW**.

### 10. MoE stage-1 A4 reuse — `PLOW_MOE_STAGE1_A4_REUSE` (default on, runtime)
* (a) Geometry-only gate: `inter>=256`, `inter%32==0`, `hidden%128==0`, mxfp4, ≥2 N-tiles
  (`amd.rs:1241-1258`). Scratch = max over programs of `row_token.bytes/4 × (hidden/2+hidden/32)`
  (`amd.rs:1672-1711`); `pad_rows = t*top_k + n_exp*(BM-1)` per bucket (`k3.rs:1212-1219`).
  `devbuild.rs:3427-3442` (`8192*16*…`) is `#[cfg(test)]`.
* (b) Small/ragged T launches the widest-bucket grid; quant clamps rows to `row_capacity`, GEMM
  exits on `mt >= total_tiles` (`runtime/bench/amd/lean_moe_stage1_ref/reuse_kernel.hip:47-52,
  156-161`) → perf only. T=16384 unreachable (chunk cap). `quant_grid: 1024` hard-coded (`amd.rs:2344`).
* (c) **LOW** correctness; low-med perf at small T.

### 11. HSA queue 4096 + exact AQL chain reservation
* (a) `QUEUE_SIZE = 4096` constant (`crates/plowrt/src/device/hsa.rs:230`; `runtime/amd/hsa_backend.c:13`
  tests only). Chain packets = exact per-segment sum (`amd.rs:11394-11417`), one reservation per
  rank per chunk (`amd_tp.rs:1391-1396`).
* (b) `packets > 4096` → `Err("AQL chain has N packets but queue capacity is 4096")`
  (`hsa.rs:3081-3085`); over/under-emission refused (`hsa.rs:1971-1977, 3111-3119`). Chunks are
  sequential (each `prefill_chunk` drains, `amd_tp.rs:1409-1412`) so ≤1 chain (~1157 packets) is
  ever in the ring at 16K/32K; decode is not chained; per-rank queues.
* (c) **LOW** (fail-closed).

### 12. Per-XCD segmented dispatch (83120a7)
* (a) `l2_domains`/CU map from hwspec (`crates/plowc/src/main.rs:1190-1210`), blob `n_cu` must equal
  device (`amd.rs:7469-7495`); `ProgramDispatch::classify` (`amd.rs:267-278`).
* (b) Prefill always segmented (`amd.rs:11673`), all chunks; decode single launch unless raw routes
  (`amd.rs:2400, 11726`); validation loops all rungs (`amd.rs:497-530`). Geometry-only.
* (c) **LOW**.

### 13. HIER gate — `PLOW_GATE_HIER` (gfx950 decode default on)
* (a) Build: decode + GQ + `PLOW_L2_PLACE_DISPATCH` (`interp.hip:597-607`,
  `runtime/cmake/hipcc_hsaco.sh:173-186`); runtime `nper>1 && !FINE && !xctr` (`interp.hip:4362-4380`),
  `nper` per (packet, XCD) at emit (`devbuild.rs:2554-2598`, ≤511).
* (b) Per-rung scratch and `nper` for every decode rung; xctr excluded; no KV/TP/batch sizing.
  Sizing gate (−5.9 ms) was measured at B1 only.
* (c) **LOW**.

### 14. Decode batching / serve engine (context for the throughput verdict)
* Capacity = widest decode program `t` (`amd.rs:7454-7457, 12304`); rungs `dec_lo..=decode`
  (`amd.rs:12952-12957`); emit `PLOW_DECODE_BATCH_LADDER`, hsaco `PLOW_GEMV_MM` ≤16 with WALK above
  (`amd.rs:10083-10099`; `runtime/CMakeLists.txt:749-800`). Mux `capacity=batch`,
  `ingress_capacity = max_queued_requests || 4×capacity` (`serve/mux.rs:372-410`); beyond → 429
  (`mux.rs:263-270`, `serve/completion.rs:107-115`); admission shed 429 when predicted wait >
  `max(slo_ms, 8×service)` (`mux.rs:770-800`).
* The gated blob is B=1: `dispatch_all` scalar path (`serve/engine.rs:1488-1547`); mux comment
  "one tick is one token … the only shape PLOW_DECODE_BATCH=1 admits" (`mux.rs:1944-1946`).
* Prefill and decode are not fused: per tick one prefill chunk (≤`pf_interleave` 2048 rows,
  `config.rs:188-193`) then one decode step for all live slots (`mux.rs:1849-1857, 1927-1990`);
  every decode stream stalls for the chunk.
* Batched KDA decode accepts rows ∈ {1, 8} only (`crates/devgen/src/kda.rs:136-138`,
  `amd.rs:2495-2513`); other rungs take the unfused path (perf cliff, not a crash).

## Cross-cutting: TuneDB digests disagree
`want` = current `BuildId::label()` (`crates/kernelcaps/src/build.rs:63-77`, `lib.rs:1139-1144`),
exact-string staleness (`crates/tunedb/src/record.rs:33-48`). Records carry three different
digests: MoE-decode route `aa66f160…`, attention nsplit `af87731e…`, GEMM c8 `870078e9…/58fbf5c8…`.
Under any one build at most one of {MoE reroute, attention nsplit, c8} fires; the rest silently
fall back. Any `op_gemm.h` edit (e.g. UN=7) re-keys all of them. The promoted "rule" is
effectively pinned to one build hash.

## Latency-path-only vs throughput-relevant

| Change | C1/B1 only? | Throughput-relevant? |
|---|---|---|
| Tagged XReduce (−2.12 ms/tok) | yes (`t==1` emit) | no — b>1 rungs run two-shot |
| Grouped-MoE route (−0.67) | yes (`t==1` ops, rung-1 records) | no |
| GLU UN=7 (−0.11) | yes (`GemvGlu` only at t==1) | no |
| MLA fold DVT (−0.31) | B1/B2 | no at B≥3 |
| HIER gate (−5.9, sized at B1) | measured B1 | applies to all rungs, unmeasured |
| ASAP order (−0.21 TPOT, −6 ms TTFT) | measured B1 | applies everywhere, unmeasured |
| KDA carry regstate (−112 ms TTFT) | prefill, any C | yes, but broken at slot≠0 and 512-tails |
| AttnRes f32mix (−48 ms TTFT) | prefill ≥256 | yes (prefill time is the C128 bottleneck) |
| A4 reuse (−88 ms TTFT) | prefill | yes; grid waste at small/ragged chunks |
| HSA 4096 / chain, per-XCD dispatch | neutral | yes (infrastructure) |
| c8 tile, align-par | off | n/a |

Decode body at B1 is ~28 → 25 ms/token; C128 wide decode is 1.4–1.7 s/step
(`kimi-k3-plowrt-mi355x-wide-decode-20260902.md`, folded), so the ~3.4 ms of B1-only wins are
<0.3% of a B128 step even if they applied.

## Throughput regime today

* vLLM 0.28 C128 (`perf-data/kimi-k3-vllm-mi355x-c128.json`): 1133.9 output tok/s, 10205 total
  tok/s, TTFT p50 1437 / p99 70943 ms, TPOT p50 109.6 ms, 1280 req in 1156 s.
* Plow C128 8192→32 (throughput-gates, folded into the summary): 1304 total tok/s, TPOT 1482.9 ms,
  TTFT mean 383 s; blob `[1,32,64,128]`, `PLOW_GEMV_WALK=1`, MM8, max_ctx 16384, 66.5 GiB/rank
  resident (48 GiB of it MLA KV at B128×16K, ~1 KiB/token/layer/rank BF16). ≈ 1,048,576 prompt
  tokens / 807 s: the cell is serialized prefill. Wide decode MM8 2→16 C128: 57.1 tok/s.
* Memory envelope: 258.6/288 GiB at B128×16K; B128×32K BF16 KV does not fit (~+48 GiB). Long
  context at C>1 needs a lower B or FP8 KV (off-contract).
* Plow has never run a served K3 cell at prompt >8192 on MI355X under the BF16 contract; the
  16K–128K TPOT ladder in `kimi-k3-consolidated.md:33-40` is MI325X B1, and the 200K/250K runs in
  `kimi-k3-README.md:437-470` were FP8-KV.

## What is hard-coded to the benchmark shape

* c8 tile: literal `8192x1536x7168` (opt-in).
* Every gate cell: 8192 = exactly one widest bucket → no ragged tail, no continuation chunk,
  no slot≠0, no 512-rung.
* TuneDB MoE-decode records: `decode_rung:1` only, single build digest.
* Tagged XReduce: `hidden ≤ 7680`, one row, ≤8 ranks, compact audit required.
* AttnRes f32mix: `HID=7168`, `AR_MAXB=8` (emit allows 16).
* KDA fused decode rows ∈ {1,8}; KDA regstate `t ≥ 512`.
* HIER/dispatch: 8 XCD × 32 CU (refused otherwise, not silently wrong).

## Minimal changes before claiming generalization

1. KDA carry regstate tail: either route `clen < 512` chunks to the interpreter carry (host-side
   fallback in `amd.rs:11132-11150` when `args.t < 512`) or drop the kernel's `t < 512` guard and
   validate at 129..511. Add a load-time refusal if neither.
2. KDA carry-family slot aliasing: rebase `args.state` (and the key-factor carry's) in
   `prefill_slot`/`kv_rebase`, or rebuild the three routes per slot. Add a unit test that the
   route state pointer moves with `kv_slot`.
3. Tagged XReduce: turn the `fj[2]==0` device trap into a load-time refusal when
   `!tp_audit_compact || has_fine_xctr`; add a `tp <= 8` host check.
4. TuneDB: re-key the MoE-decode, attention and c8 stores to one current digest (or accept a
   digest set) so the promoted rule actually fires on the shipped build.
5. Throughput blob: emit `PLOW_DECODE_BATCH_LADDER=1,8,32,128` + `PLOW_GEMV_WALK=1`, max_ctx
   16384; confirm `check_xr_tagged_blob` passes (t>1 rungs are two-shot) and that decode rungs
   2..128 other than 8 do not silently lose the fused KDA decode.

## Gates required (all via `perf-data/tools/gpulease -n 8 showdown perf-data/tools/bringup_showdown.sh`)

Ragged / continuation correctness first (no perf claim without it):
* Exactness probe, C1, prompts {300, 8400, 9000, 12000, 16384} → 32, checksum vs
  `PLOW_KDA_CARRY_REGSTATE=0` control. 8400 and 300 hit the 512-rung tail.
* Slot probe on the ladder blob: two concurrent prompts, second lands in slot 1, checksum vs
  sequential C1.

Long context (asset emitted `--max-ctx 32768`; the 16384 asset refuses):
```
ROUNDS=3 INPUT_MAP="16384" CONCURRENCY_MAP=default=1 PROMPT_MAP=default=10 WARMUP_MAP=default=1 \
OUTLEN_MAP=default=1024 MAX_MODEL_LEN=17408 PLOW_ENGINE_ARGV="--amd-tp-no-audit" …
ROUNDS=3 INPUT_MAP="31744" CONCURRENCY_MAP=default=1 PROMPT_MAP=default=3 OUTLEN_MAP=default=32 \
MAX_MODEL_LEN=32800 …    # 32K probe on the 32768 asset
```
Throughput (ladder blob, max_ctx 16384; raise the SLO shed and ingress):
```
ROUNDS=3 INPUT_MAP="8192" CONCURRENCY_MAP=default=8   PROMPT_MAP=default=80   WARMUP_MAP=default=8 \
OUTLEN_MAP=default=1024 PLOW_ENGINE_ARGV="--amd-tp-no-audit --slo-ms 600000 --max-queued-requests 1024" …
… CONCURRENCY_MAP=default=32  PROMPT_MAP=default=320  WARMUP_MAP=default=32 …
… CONCURRENCY_MAP=default=128 PROMPT_MAP=default=1280 WARMUP_MAP=default=128 …
```
`bringup_bench.sh` fails a cell on any 429/refusal, so the SLO/ingress flags are mandatory at
C≥8. One `INPUT_MAP` length can carry one concurrency; run one campaign per C.

Expected outcome at C8–C128: none of the B1-only decode wins appear; TTFT-side wins
(regstate, f32mix, A4) scale with prompt count only if items 1–2 are fixed; the dominant term
remains serialized prefill chunks stalling decode (`mux.rs:1849-1857`), which none of the
promotions address.
