# Iteration 8 — object-level tuning database: typed schema, no live sweep yet

Iteration:      8 (`/root/.claude/plans/glimmering-soaring-stream.md`)
Commit before change: `a75b675` (Iteration 2's tool-limitation close-out)
Hypothesis:     N/A — this iteration is infrastructure, not a performance experiment. Iterations
                3/5/6 (mbarrier-based GEMM/attention pipelining) are parked pending Iteration 2's
                unresolved interpreter hang; Iteration 4 (cuBLASDx) is blocked (no NVIDIA
                Developer Program access in this environment, confirmed: not installed, not in
                nixpkgs). Iteration 8 does not depend on either blocker.
Expected mechanism: N/A.
Expected maximum end-to-end benefit: N/A — no kernel change, no benchmark claim.

## What existed already (surveyed before writing anything)

`crates/tunedb` already has two record kinds: `KernelMeasurement` (isolated op-case → kernel_id,
`crates/tunedb/src/record.rs`) and `DecodeMeasurement` (whole-decode-object define-set, ranked by
end-to-end `step_bench` TPOT, `crates/tunedb/src/decode.rs`). Neither can express a **prefill
GEMM tile** as a build-identity axis: `tuning/README.md` already documented this gap explicitly —
`nvidia/sm_120a/rtx-5090/prefill_tile_measurement.jsonl` is a hand-rolled Python/JSON schema
(`perf-data/px13_emit_tuning.py`) that the README states outright is "not loadable by
`TuneStore::load_kernels`... until that entity grows a build-identity column." That file already
does almost everything the mission's Iteration 8 spec asks for (`object.pf_cubin_md5`,
`object.registers`, `object.stack_bytes`, `score.metric = conc1_prefill_wall_s`, a `microbench`
block explicitly marked "recorded, NOT the ranking key") — it just isn't a typed Rust record
integrated with `TuneStore`.

## What this iteration adds

`crates/tunedb/src/object.rs` (new, ~330 lines with tests): `ObjectCell` (key: hardware, sm_count,
toolchain, model, dtype, kv_dtype, batch, m_bucket, n, k, head_dim, window class),
`ObjectConfig` (tile, warp_split, pipeline_depth, raster_order, split_k, bq/bkv, buffer_depth,
plus an `extra_defines` open-extension map mirroring `DecodeKnobs`'), `ObjectMeasurement`
(cell + config + `object_hash` + registers/stack_bytes/spill_bytes/shared_mem_bytes +
`isolated`/`complete_object`/`end_to_end` `Stats` + correctness/state/campaign), `ObjectRanking`,
and `rank_by_cell` — ranking strictly on `end_to_end`, never `isolated` or `complete_object`.

`crates/tunedb/src/store.rs`: `object_path`, `load_objects`, `publish_object` (gated through the
same `blockers_for` correctness/sample-count check every other record kind uses —
**the ad hoc `prefill_tile_measurement.jsonl` rows would fail this gate**, since they carry
`"correctness":"unchecked"` next to `"state":"qualified"`; this is a real, intentional behavior
change for any future migration of that file, not a bug), `record_object_unqualified`,
`best_object_for`.

`crates/tunedb/src/lib.rs`: re-exports (`rank_by_cell` aliased to `rank_objects_by_cell` at the
crate root to avoid colliding with `decode::rank_by_cell`, which already owns that name).

`tuning/README.md`: documented the new schema, explicitly noted `prefill_tile_measurement.jsonl`
is **not yet migrated** (13 existing rows, still in the old ad hoc format) — the honest state,
not a claimed completion.

## Files changed

- `crates/tunedb/src/object.rs` (new)
- `crates/tunedb/src/store.rs` (+`object_path`/`load_objects`/`publish_object`/
  `record_object_unqualified`/`best_object_for`, +9 tests)
- `crates/tunedb/src/lib.rs` (+re-exports)
- `tuning/README.md` (+schema doc, +layout entry, honest migration-status note)

## Exact build/test commands

```
nix shell nixpkgs#cargo nixpkgs#rustc nixpkgs#gcc --command cargo build -p tunedb -p devgen -p plowc
nix shell nixpkgs#cargo nixpkgs#rustc nixpkgs#gcc --command cargo test -p tunedb
nix shell nixpkgs#cargo nixpkgs#rustc nixpkgs#rustfmt --command cargo fmt -p tunedb
```

(Used the lighter `nix shell nixpkgs#cargo nixpkgs#rustc nixpkgs#gcc` rather than `nix develop`
— the latter unconditionally pulls the full CUDA+ROCm devShell, unneeded for a pure-Rust crate
change and documented elsewhere in this repo's own sandbox notes as the thing to avoid.)

## Correctness result

`cargo test -p tunedb`: **66 passed, 0 failed** (57 pre-existing + 9 new: 6 in `object.rs`
covering the PX-13-style isolated-vs-end-to-end disagreement, cell separation, correctness
gating, indecisive-margin handling, config-label stability, and JSON round-trip; 3 in `store.rs`
covering file separation from the other two record kinds, the unchecked-correctness publish
refusal, and per-cell best-by-end-to-end ranking). `cargo build -p tunedb -p devgen -p plowc`:
clean (only pre-existing, unrelated warnings in `devgen/src/mla.rs` and `packet/src/devbuild.rs`).
`cargo fmt -p tunedb -- --check`: applied two pre-existing style diffs, re-tested after, still 66/66.

## Isolated / complete-object / end-to-end result

Not applicable — no kernel was built or measured this iteration. This is schema/infrastructure
work: the new record kind can now *express* isolated/complete-object/end-to-end numbers for a
future sweep; it does not itself produce any.

## Register count / Stack / Spills / Dynamic shared memory

Not applicable — no kernel changed.

## Decision: ACCEPT

## Reason

Formalizes an already-documented, already-acknowledged gap (`tuning/README.md`'s own words: "not
loadable... until that entity grows a build-identity column") without touching any production
kernel or runtime code — pure library/schema addition, fully tested, builds clean, does not
change `plowrt`/`plowc` behavior (`crates/devgen`/`crates/plowc` still build and their existing
tests are untouched). Does not migrate the existing 13-row `prefill_tile_measurement.jsonl` file
(a separate, deliberately-deferred task — migrating data is a different risk profile than adding
a schema, and the file's `"correctness":"unchecked"` rows would need real correctness
verification, not just a format conversion, before they could pass the new gate honestly).

## Commit

(this iteration's commit follows this report)

## Next experiment

This schema exists to be filled by a future GEMM-tile or attention-pipeline sweep once Iteration
2's interpreter hang is resolved or Iteration 3/5/6 find a synchronization approach that doesn't
hit it (`perf-data/sm120-iter2-ws-gemm-rejected-2026-08-26.md`'s own next-steps section). Until
then, this module has no live data — `TuneStore::load_objects` on any real hardware cell returns
an empty `Vec` today, and that is the correct, honest state to report.
