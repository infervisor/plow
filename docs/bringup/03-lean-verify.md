# Stage 3 — Lean verification and bundle qualification

Target-independent: a checkpoint that holds on one part only is not a proof.

Goal: every checkpoint the model's emit path calls returns `ok`, and `plowc qualify` finds no gap
you cannot name. A rejection is a compiler bug, not a warning. Design:
[`docs/arch/08-formal-verification.md`](../arch/08-formal-verification.md).

New models rarely need new theorems: the universal lemmas are proved once and each checker
re-checks per-instance data. The exception is Checkpoint A: a new Stage-2 rule adds a `rule_*`
obligation ([02-egglog-rewrite.md](02-egglog-rewrite.md)).

## Endpoints

All endpoints dispatch through `Plow.CLI.Dispatch.endpoints` (`lean-plow/Plow/CLI/Dispatch.lean`);
handlers live in `lean-plow/Plow/CLI/Checkpoints.lean`. `lean-plow/proof-manifest.json` lists the
acceptance theorems per endpoint (proof roots below).

| ID | Proves | Proof roots (`Plow.` omitted) |
|---|---|---|
| A | rewrite rules and bodies expand to their unfused composition | `RewriteBody.check_sound`, `Rewrite.rule_*` |
| B | tiling covers the GEMM with no gap/overlap, per-tile work ≤ cost bound | `TilePartition.check_sound` |
| C | SRAM hand-offs never overcommit the page budget | `Sram.occupancy_le_of_temporal_fit` |
| D | counter schedule enforces every dependency; logical tensor effects conflict-free | `Verify.{verifyDependencies,checkPaths,verifyAddressMapVia,treeBefore}_sound`, `Verify.verifyAddressMap_sound_strict`, `Effects.{checkConflicts,checkAccesses,checkReuse}_sound`, `Effects.cancellation_retired` |
| E | wire encoding round-trips | `Wire.decodeProgram_encodeProgram` |
| F | allocation safety (same checker and bundle as D) | `Verify.verifyAddressMap_sound_strict`, `Verify.verifyAddressMapVia_sound` |
| G | every always-staged GEMV fits the LDS arena | `LdsFit.fits_of_check_ok` |
| K | knob consistency | `Knobs.checkK_sound` |
| S | knob scope: a knob change touches only its declared rungs/routes | `Knobs.Scope.{checkS_sound,diff_complete,untouched_rungs,route_untouched}` |
| P | perf floor ledger | `Knobs.Ledger.checkP_sound` |
| R | measured / selected GEMM policy | `MeasuredPolicy.check_sound` |
| L | padded-MLA layout mapping | `MlaLayout.{address_in_bounds,ignored_padding,same_consumer_output}` |
| `media_geometry.v1` | speech/multimodal capacity and shapes hold for every input up to the declared maxima | `MediaGeometry.check_sound` + per-family `every_*_fits`, `CodecLm.*`, `Multimodal.soft_ids_disjoint` |
| `kv_ring.v1` | no packed-prefill span writes a sliding-ring row its own queries read | `KvRing.{checkLaunch_sound,launch_safe,launch_safe_masked,slots_distinct,mask_eq_mod}` |
| `vmm_trace.v1` | VMM traces keep mappings on live handles, disjoint, inside reservations | `KvRing.{traceOk_sound,step_preserves,run_preserves}` |
| `speech_fusion.v1` | fused LayerNorm-prologue / Conv1dF32 `row_scale` sites meet `Plow.Speech`'s preconditions | `SpeechFusion.{check_sound,ln_site_stats,ln_site_gemm_eq,conv_site_in_bounds,conv_site_eq}` |
| `page_envelope.v1` | the paged library attention loader reads only its request's padded page-table row | `PageEnvelope.{checkRoute_sound,route_safe,read_in_envelope,padded_entry_own,envelope_tight}` |

`lake exe proof_audit` checks that manifest ids equal the dispatched ids, every listed theorem
exists, its statement mentions the named checker, and its axiom closure stays inside
{propext, Classical.choice, Quot.sound}. D/F acceptance always runs the proven reference checker;
the stack-safe `FastCheckD` can only reject early.

## Call paths

| CP | caller | runs when | persisted |
|---|---|---|---|
| A | `devblob_verify_hook` (`crates/plowc/src/main.rs`), whole rule catalog + bodies | every devblob emit (`--lean-verify-devblob`, default on) | `RewriteBodyExpansion` receipt |
| D coarse | devblob hook, programs with a reduction witness | default | `CoarseDependencyPreservation` |
| D effects | devblob hook, programs where `logical_effects::obligation` is `Ok` | default; the first unaudited op is a logged gap | `LogicalTensorEffects` |
| D ordering + G | devblob hook per program (empty address map, topological GQ order, staged-LDS fit) | default | `build.json` `verified`, `lds_fit_verified` |
| D sidecar | devgen `write_sidecar_packet` (codec, s3gen, encoder, mm towers) | sidecar verifier installed | `<stem>.lean-checks.json` |
| K | devgen `knob_spec::gate` | every devblob emit unless `--no-knob-verify`; an unusable binary is fatal | `build.json` `knobs.K` |
| R | devblob hook, measured / GEMM policies | policies recorded | `MeasuredPolicy` / `SelectedGemmPolicy` |
| L | devblob hook, padded-MLA `FlashMerge` producers | MLA-class packets | `LayoutMapping` |
| A,B,D,E,F | `run_lean_verify` (`crates/plowc/src/lib.rs`), schedule path | `--lean-verify` (opt-in) | `build.json`; E sends a fixed sample, not the program |
| S | `plowc knob-scope` (compares via `plowrt knob-scope`), run by `scripts/knob_scope_ci.sh` | CI on a knob change | report file |
| P | `scripts/perf_cert.py` via `scripts/perf_gate_ci.sh` | CI on a default flip | `perf-certs/*.json` |
| C | none (tests only) | — | — |

The devblob path therefore runs A/D/G/R/L (+K); the schedule path A/B/D/E/F. No path runs C.
The D/F verdict cache key includes the checkpoint letter.

## Qualification

`plow_asset::certificates` derives each packet's required obligations, matches receipts and records
a `Qualification` (required, satisfied, policy receipts, verifiers, gaps); `plowc::qualify` replays
them on the current verifier. plowrt does not read or replay receipts; packet integrity at load is
the dist signatures and digests.

- Required set:
  - compiler packet (`model.pkt`, or any packet whose `lean-checks.json` binds its bytes): A once;
    coarse D and logical-effects D per program; L per padded-MLA producer;
  - any other packet is a sidecar: logical-effects D per program via `program_checks`;
  - versioned endpoints where they apply: `media_geometry.v1` on the packet that owns a
    speech/multimodal pipeline, `kv_ring.v1` on a packet with packed prefill over a sliding ring,
    `speech_fusion.v1` on a packet with a fused speech site;
  - policy receipts (R) bind when present, never required.
- Gaps: absent receipts or a `packet_sha256` mismatch; a receipt that does not rebind from the
  loaded packet; a wrong/absent program, duplicate or unused receipt, unsupported scope; a `None`,
  out-of-range or short `program_checks`; an unapproved receipt verifier; an unavailable or
  unapproved current verifier, a replay rejection, or a changed envelope.
- Verifier identity: `lean-plow/approved-verifiers.json` pairs each approved `plow_verify` sha256
  with the Lean-sources digest (`lean_verify::lean_sources_sha256`). A rebuilt verifier needs an
  entry; `lean_verify` tests fail without one.
- Replay cache key: (receipts digest, verifier, required-scope-set digest).
- Entry points:
  - `plowc qualify <dir|packet>...` replays every `*.pkt` and exits non-zero on any gap
    (`--no-replay` cannot qualify; `--format json`). A sidecar reads `<stem>.lean-checks.json`.
  - Every devblob emit qualifies its output: `--lean-qualify report` (default) records
    `build.json` `lean.bundle_qualification`; `strict` fails the emit. `campaign.py build` passes
    `strict` for a `recipes/infervisor` recipe.
  - `scripts/campaign/campaign.py qualify --assets <dir> --strict`.

### Versioned endpoints — inputs

- `media_geometry.v1`: `plow_asset::media_geometry::request` derives one contract per family
  (audio_lm, rnnt, codec_lm, guided_lm, vad, multimodal) from the packet pipeline sections,
  `asr_vocabulary.json` and the multimodal contract. A missing parameter, absent sidecar or
  overflowing shape product rejects; Lean rejects unknown fields and kinds.
- `kv_ring.v1`: `plow_asset::kv_ring` builds launches `{ring_log, window, capacity, spans}` from
  the packet's `live_kv` sliding rings and `packed_prefill` limits, using the runtime planner
  (`plan_with_limit`, `stage_slots`) on boundary mixes (full request at context, staggered wrapping
  requests, one-row request). Contract: `0 < window`, `ring_log < 32`, every span
  `window + len ≤ ring + 1` and `start + len ≤ capacity`, distinct non-idle slots. Out of scope:
  device completion before reuse (covered by D effects), growth beyond the packet `max_ctx`.
- `vmm_trace.v1`: driver event traces (`reserve`, `address_free`, `create`, `release`, `map`,
  `unmap`, `access`, `quiesce`) recorded by the mock `VmmOps` in `memory::vmm` ring tests (CPU
  lane). With `quiesce` everything must be returned.
- `speech_fusion.v1`: `plow_asset::speech_fusion` derives one site per fused instruction —
  LayerNorm prologue (`DenseGemmF32` flag bit 5: the last earlier writer of `stats` is
  `RowStatsF32` over the GEMM's A, nothing rewrites A or stats in between) and Conv1dF32
  `row_scale` (sizes recomputed from raw fields, no aliasing of output/residual). Assumes programs
  run in index order.
- `page_envelope.v1`: runtime constants, not packet data. The ignored test
  `exec::gpu::lib_attention::tests::paged_table_pad_satisfies_the_lean_envelope` sends page size,
  split cap and table pad; contract `0 < page`, `kv_tile = page`, `0 < max_splits ≤ pad + 1`.
  Only the TMA paged route is in scope; a const assert in `lib_attention.rs` repeats the
  hypothesis.

## Build and run

Toolchain: `lean-plow/lean-toolchain` via `nix develop`.

```bash
nix develop --command bash -c 'cd lean-plow && lake build plow_verify proof_audit && lake exe proof_audit'
nix develop --command scripts/lean_correctness_ci.sh      # required CPU lane (also CI)
nix develop --command plowc --hf-dir <checkpoint> --emit devblob --out <scratch>/emit   # A/D/G/R/L/K on
nix develop --command plowc qualify <scratch>/emit
```

- `lake build` is a proof check: `warningAsError := true`, so `sorry` fails the build;
  `proof_audit` is the axiom gate.
- `lean-verify` is a default `plowc` feature. `--lean-verify` additionally forces the schedule path.
- Binary lookup: `PLOW_VERIFY_BIN`, then `lean-plow/.lake/build/bin/plow_verify` relative to the
  crate, then `plow_verify` on `PATH`.
- Only `VerifyError::is_binary_unusable` downgrades to a recorded skip; every rejection and
  spawn/marshal failure is fatal (`PlowcError::LeanVerify`, `LeanVerifySpawn`).
- Debug: `PLOW_VERIFY_DUMP=<dir>` writes every request; `plow_verify < payload.json` replays one.

## Gate (all before Stage 4)

- [ ] `lake build plow_verify proof_audit && lake exe proof_audit` clean;
      `scripts/lean_correctness_ci.sh` passes.
- [ ] Every devblob emit of the model returns `ok` from A/D/G/R/L (and K); no rejection.
- [ ] `plowc qualify` lists no gap you cannot name; a production recipe emits with
      `--lean-qualify strict`.
- [ ] No `sorry`, no vacuous theorem (`True`, restating a term as itself); every `rule_*` is a
      real equality closed by `rfl`; `soundRules` ↔ `; rule:` annotations ↔ `rule_*` one-to-one.
- [ ] No silent skip: a skip is acceptable only for a genuinely unusable binary, and is reported.

## Pitfalls

- Outside `nix develop` the verifier can die before `main` (nix ELF interpreter) with empty stdout:
  that is a skip, not a pass. Re-run inside the shell or set `PLOW_VERIFY_BIN`.
- Known silent gaps: non-Linux skips every batch; logical-effect gaps are log-only; some
  example-built speech packets bypass the emit gates.
- D and F share a checker by construction; never fix a D/F disagreement in one handler.
- E proves the framing invariant on an abstract model; byte layout regressions belong to the
  `crates/packet` round-trip suite.
- Keep the stack-safe `FastCheckD` path for real schedules; the reference recursion overflows.

## Stop and ask

- A checkpoint rejects and the compiler output looks right: report the payload
  (`PLOW_VERIFY_DUMP`) before touching a proof.
- A new rule's `rfl` theorem does not close.
- A change needs a new verifier approval entry.

## Pointers

- Rust client: `crates/lean_verify/src/lib.rs` (`call`, `require`, `query`, `VerifyError`);
  per-checkpoint wrappers in `crates/lean_verify/src/checkpoints/`.
- Schedule-path bridge: `crates/schedule/src/lean_verify.rs`.
- Qualification: `plow_asset::certificates` (`qualify_packet`, `qualify_sidecar`),
  `crates/plowc/src/qualify.rs` (`qualify_dir`).
- Lean: `lean-plow/Plow/CLI/{Dispatch,Checkpoints,FastCheckD,Payload,Schema}.lean`,
  `lean-plow/ProofAudit.lean`, `lean-plow/proof-manifest.json`.
