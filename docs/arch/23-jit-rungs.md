# 23 — JIT Rungs (future plan)

> **Status: not implemented.** This chapter is a plan, written from a read-only code review and
> a CPU-only timing study on 2026-09-21 (Gemma-4-12B and 26B-A4B, `sm_90a`, H100). Nothing here
> changes behaviour today. It builds on **[19 — Packet Extensions](19-packet-extensions.md)**,
> whose phases 1–3 have landed and whose phase 4 (`plowc extend`) is the first step below.
>
> **The question.** Device objects (cubins) can be prebuilt for many combinations. Can the
> runtime then build a *new rung* — a decode program for batch 6 or 12, a prefill bucket for an
> exact row count, a context-band decode program — while serving, and add it to the model as a
> packet extension, without the offline compiler and without touching the hot path?
>
> **Short answer.** Yes, in stages. Emitting a rung is already fast and needs no weights. What
> stands between today and a JIT is plumbing (nothing loads an extension yet), one explicit
> contract case for reusing prebuilt objects, and the emitter's process-global state.

---

## 1. What a rung costs today (measured)

`plowc --emit devblob` on the shipped recipes, single-threaded, `/usr/bin/time -v` around plowc
only (`nix develop` entry adds ~1.0 s outside these numbers). `model.pkt` was byte-identical to
the served packets in every verify-on run.

| run | programs (prefill + decode) | Lean/knob verify | `hipcc` probe | wall s |
|---|---|---|---|---|
| 12B, recipe as-is | 10 + 5 | on | on | 9.09 |
| 12B | 10 + 5 | off | on | 5.88 |
| 12B | 10 + 5 | on | off | 6.55 |
| 12B | 10 + 5 | off | off | 3.47 |
| 12B | 5 + 1 | off | off | 1.56 |
| 26B MoE, recipe as-is | 8 + 5 | on | on | 6.77 |
| 26B MoE | 8 + 5 | off | off | 1.77 |

Where the 9.09 s goes (12B, by difference between runs plus one `strace` run):

| phase | s | share |
|---|---|---|
| weights read or packed | ~0 | 0% |
| `hipcc -E` kernel-inventory probe (`amd_gemm_inventory`, `crates/devgen/src/lib.rs`) | 2.5 | 28% |
| Lean verify + knob certificate (40 `plow_verify` processes) | 3.2 | 35% |
| `build.json` manifest + dispatch audit (`manifest::build_for_packet`) | ~1.7 | 19% |
| post-processing before the write (cuBLASLt algos, roles, packed-prefill metadata, digests) | ~1.1 | 12% |
| **the 15 instruction programs** (`declare` + `Builder` per program) | **0.25** | **3%** |
| startup, config, rewrite-site extraction | 0.29 | 3% |
| `model.pkt` write (49.8 MB, one `write(2)`) | 0.011 | 0.1% |

Per rung:

* **The instruction program itself: 15 ms per prefill bucket, 17–21 ms per decode rung** on the
  12B; 11.5 ms and 14 ms on the 26B MoE.
* The whole pipeline charges 0.21 s per program even with verify and the probe off. The extra
  ~0.19 s is the dispatch audit (~0.11 s) and section/digest/serialize passes (~0.07 s) — not
  needed to *run* a rung.
* Verifying one **new** program: ~0.27 s (decode) or ~0.45 s (prefill; the ordering certificate
  walks ~82k queue entries). A program whose D/F payload matches one already certified costs
  ~0.12 s: the verdict is cached by payload hash (`crates/lean_verify/src/lib.rs`, `call`).
* Two fixed costs a JIT would not pay per rung: process start (~0.3 s) and the `hipcc` probe
  (2.5 s). The probe is the AMD gfx950 inventory; on `sm_90a` it changes no output byte (same
  `model.pkt` sha256 with `hipcc` off `PATH`). Skipping it on NVIDIA targets is a free 2.5 s on
  every emit today, independent of this plan.

**Emit does not depend on weight values.** The tensor table is names and byte counts
(`init: None`); the runtime binds weights through the `checkpoint` symlink. The only values read
are the per-layer `layer_scalar` immediates (`crates/devgen/src/checkpoint.rs`), plus safetensors
header names for the coverage gate. The asset directory is ~60 MB; the "22 GiB" an emit appears to
write is `du -L` following the `checkpoint` symlink.

The device side is the expensive half: one full object build is ~207 s of `nvcc`. A JIT is only
interesting for rungs the *prebuilt* objects already cover.

## 2. How a rung is produced, and what rungs share

`run_devblob` (`crates/plowc/src/main.rs`) → `devgen::run_verified` → `emit_config::install` →
per-architecture emitter (`emit_dense_gqa` for Gemma). Inside it:

* **Shared by every rung:** the config, the layer scalars, the bucket list, the KV-ring assert
  `ring >= window + widest - 1`, worst-case activation rows, and **one tensor table** declared at
  the widest decode batch (`declare(dbatch)`) and the widest prefill rung (`chunk_rows`).
* **Per rung:** a fresh `Builder::new(n_cu)` adopts a clone of that table, then
  `emit_prefill(b, t)` or `emit_decode(bd, rb)`, then `progs.push(b.finish())`. Iterations are
  independent of each other.
* **Whole-model post-passes keyed by program index:** cuBLASLt algo rows per exact shape, segment
  roles, the `packed_prefill` / `live_kv` manifests with per-program digests, attention role
  objects (row-keyed rules such as `4096 | 8192`), the Lean gate, and the manifest.

The consequence that matters: **a JIT rung must fit the envelope the parent declared** — tensor
table, activations, KV ring, smem arena, decode slot count. Anything wider is a re-emit and a
reload, not an extension. Role objects constrain the ladder too: a 12B ladder capped at 128 rows
fails to emit, because the GQA2 attention role requires the 4096 rung.

## 3. What is packet-only and what needs a new cubin

| new rung | needs |
|---|---|
| decode rung with B ≤ the parent's `decode_batch` (e.g. 6, 12) | packet only — `gemv_walk` handles arbitrary M |
| prefill bucket no wider than the parent's widest | packet only (new exact-shape cuBLASLt algos and role objects are performance, not correctness) |
| context-band decode program | packet only, but see §4: bands are digest-pinned in the base packet today |
| B > `decode_batch` | new cubin (`gv_mm_max`), and the KV slot count forces a reload |
| any new kernel arm, or a change of a paired value (`PLOW_NV_FA_GF_FULL` and the packet's nsplit) | new cubin |

The pairing hash (`manifest::pairing_hash`) covers `union`, `objects`, `tuning` and
`backends.requires`. It does **not** cover the rung ladder, and no kernel reads
`PLOW_PACKET_DECODE_LADDER`. Untested rung widths can still reach latent object bugs (rung 4 once
faulted on the 1 KiB device stack), so a new width needs its own qualification run.

## 4. What the runtime already has

* **The extension container** (chapter 19, phases 1–3): `Model::to_ext_blob`
  (`crates/packet/src/devbuild.rs`), `DevBlob::parse_extension`, the six-rule
  `plow_asset::extension::merge`, discovery through `<assets>.ext/*` or `PLOW_EXTENSIONS`
  (`crates/plowrt/src/asset/extension.rs`). **Gaps:** no engine calls `extension::load`, no emitter
  calls `to_ext_blob` (tests only), `plowc extend` does not exist, and `build.json` lacks
  `kv_window` / `kv_ring_rows`, so a bucket wider than the parent's widest cannot even be judged.
* **Decode rung upload is self-contained:** `DecodeRung::upload`
  (`crates/plowrt/src/exec/gpu/decode_rung.rs`) — its tables, its own counter slab, a kernarg
  copied from the base. Without a per-rung object it runs on the base decode function. Extension
  rungs must stay outside the `decode_objects` table, which binds exact programs.
* **Context bands already split CPU work from device work:** `prepare` / `materialize` in
  `crates/plowrt/src/exec/gpu/decode_context.rs`. That split is the template for a hot-add. Today
  an `aux.pkt` must carry the base's exact tensor table, may differ only in FlashDecode `i[5]` and
  FlashMerge `i[2]`, and its band table is digest-pinned inside the base packet.
* **Programs are already synthesised in-process:** `packed_terminal::chain`, the token-batch
  bodies built from prefill bodies, and `patch_nv_nsplit` rewriting live immediates.
* **Prefill buckets are the harder case:** per-bucket state is built inline in the load loop with
  index-keyed caches (`ensure_seg_graph`, `ensure_batch_patch`), not in a reusable function.

**Modular block packets are metadata only today.** `emit.block_packets` / `pf_modular` add a JSON
section whose entries all name the same whole-model `program_idx`; no emitter sets
`ProgramRole::ModularBlock`, `DevBlob::modular_block_progs` has no caller, and `rt.block_packets`,
`rt.pf_modular` and `rt.block_stage` are declared but never read in `plowrt`. Real single-block
programs exist only as the `--block l..r` harness artifact. "Emit one block, reuse it for every
layer" runs into per-layer tensor handles and immediates, cross-layer seam fusions (AddNorm uses
the next layer's gamma; `fuse_nrn` is off in block mode), and Gemma-4's two layer kinds plus
KV-shared layers. A per-layer tensor-pointer table (as `d_tens_slots` already does per KV slot)
could rebind one block program without a cubin change, at one launch per layer and the lost seam
fusions. **Unmeasured** — treat it as a separate track (§6, stage 6).

## 5. Two guardrails a JIT must respect

**The pairing guard.** As built, an extension's objects must be stamped with the *extension's*
hash; an object stamped for the parent is refused (`plow_asset::extension::check_object_stamp`).
A JIT rung that reuses prebuilt cubins therefore needs a new, explicit contract case — not a
bypass: *the extension declares zero objects, its per-phase arms are a subset of the parent's
`objects.*.arms`, and its `tuning` equals the parent's.* That gives what the hash gives, and the
runtime can check it as a set compare over the two `build.json` files with no `devgen` dependency.

**The verifier.** `plow_verify` is a subprocess and the hook already iterates per program, so
certifying one new program is natural (§1: 0.27–0.45 s). Verify where the rung is emitted, record
the certificate in the extension's `build.json`, and have the runtime refuse an uncertified rung
unless explicitly opted in. Verification can run off the hot path: a rung goes live only after it
passes.

**Why not link `devgen` into `plowrt` first.** There is no dependency cycle and nothing heavy, but:
the emit config is process-global and last-install-wins (`emit_config::INSTALLED`), as are
`packet::devbuild::KNOBS` and the first-wins `TUNED_GEMV_CASES`, so one emit per process at a
time; `UNRECORDED_ENV` lists emit-affecting variables read straight from the environment and
replay goes through `std::env::set_var`; `lib.rs` has on the order of 150 panic/assert/unwrap
sites and two `process::exit` calls that a `catch_unwind` in the engine thread cannot stop;
`tune_demand` atomics accumulate across emits and feed the hashed `tuning.tile_lookups`; and
`kernelcaps` looks for a source checkout.

## 6. Stages

0. **AOT extra rungs, to learn whether they pay.** `PLOW_DECODE_BATCH_LADDER` and
   `PLOW_PF_LADDER_APPEND` today. A full re-emit, a new hash, an object rebuild — only to measure
   step time at B = 6 / 12 against 8 / 16 before any of the below is built.
1. **`plowc extend` for dense GQA** (chapter 19, phase 4). Split `emit_dense_gqa` into a shared
   context plus `emit_one(role, rows)`; assert the re-derived tensor digest equals the parent's;
   verify the single program; add `kv_window` / `kv_ring_rows` to `build.json`.
2. **Load extensions in `GpuEngine::load`**, decode rungs first, with the arm-subset pairing case
   of §5. Still a restart — but no re-emit and no cubin.
3. **Hot-add between ticks.** Reuse the `prepare` / `materialize` split; submit the materialise
   step as an engine-thread closure between ticks so the hot path never waits on it. Prefill
   buckets after decode rungs.
4. **Lazy sidecar.** `plowrt` spawns `plowc extend` in the background when it sees demand (for
   example, sustained occupancy between two rungs); `<assets>.ext/` doubles as a persistent cache,
   so a rung is built once per model.
5. **In-process JIT**, only if sidecar latency proves to matter. Needs the emit config passed by
   value, no environment reads, and `Result` in place of panics and `process::exit`.
6. **True block programs**, as a separate track. Measure the per-layer launch cost first.

Expected latency per rung, from §1: an in-process JIT reusing the declared tensor table should be
~20–50 ms unverified and ~0.3–0.5 s verified (**estimate**); a cold sidecar with today's pipeline
is 1.5–3.5 s (**measured** bounds: 1.56 s for a 6-program packet, 3.47 s for 15, verify and probe
off).

## 7. What to measure before building

* Step time at B = 6 / 12 against 8 / 16, and TTFT at exact-row buckets. CUDA already patches MoE
  ragged rows (`patch_moe_rows`), and bench-style prompts already land exactly on the base rungs,
  so the prefill win may be small; the decode win is occupancy sitting between power-of-two rungs.
* The engine-thread stall of `DecodeRung::upload` (its allocations may synchronise; consider
  pre-pooling).
* Single-rung emit split into emit / verify / serialise once `emit_one` exists.

## 8. Open questions

* Whether the 26B serves decode through the token-batch route; if so, prefill-body widths matter
  more than decode rungs for that model.
* Whether the tensor table stays byte-stable across `plowc` versions (stage 1's digest assert will
  answer it).
* The AMD engine was not reviewed.
