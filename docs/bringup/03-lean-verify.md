# Stage 3 — Formal Verification with Lean (the checkpoint system)

> Part of the model bringup playbook. Stage 2 produced fusion rules and a
> compiled schedule; Stage 3 proves that schedule correct with the Lean 4
> verifier before Stage 4 (kernel tuning) touches performance.

## Goal

Run every compiled bucket of the new model through the `plow_verify` Lean CLI
and get an `ok` certificate from all seven checkpoints (A–G). A rejection is a
real correctness finding, not a warning: it means the compiler produced a
schedule that the formal model can prove is wrong (a fusion that changes
semantics, a tiling with a gap, an SRAM overcommit, a counter race, a lossy
wire encoding, an unsafe allocation, or a staged-GEMV that overruns the LDS
arena). The bar is a clean `lake build` with no `sorry` and no vacuous proofs,
and every checkpoint emitting `ok`.

The proofs are foundational and model-independent. Bringing up a new model does
**not** normally require writing new theorems — the universal lemmas are proved
once and applied per-instance by checking preconditions. The one exception is
Checkpoint A: a new model that adds a **new fusion rule** in Stage 2 adds a new
proof obligation here (see [Checkpoint A](#checkpoint-a--rewrite-rule-soundness)).

Authoritative reference: [`docs/arch/08-formal-verification.md`](../arch/08-formal-verification.md).

## What each checkpoint proves

Seven checkpoints are dispatched by `runCheckpoint` in
[`lean-plow/Main.lean`](../../lean-plow/Main.lean); each handler (`checkA` …
`checkG`) lives in
[`Plow.CLI.Checkpoints`](../../lean-plow/Plow/CLI/Checkpoints.lean). The IDs
track the compile-pipeline stages: A Rewrite, B Assemble, C Collapse/Relax, D
Schedule, E Emit, F Memory. G (staged-LDS fit) is a later addition that
re-checks an emit-time obligation and has no stage of its own.

| ID | Name | Proves | Payload | Backing theorem / symbol |
|----|------|--------|---------|--------------------------|
| A | Rewrite rule soundness | Every fired fusion rule preserves semantics | `{"rules": [...]}` | `Plow.Rewrite.soundRules`, `Plow.Rewrite.rule_*` |
| B | Tile partition validity | Tiling covers the GEMM shape with no gaps/overlaps; per-tile work ≤ cost bound | `{"candidates": [...]}` | `Plow.TilePartition.tile_partition_covers`, `check_sound` |
| C | SRAM temporal fit | Producer/consumer never co-hold pages across a hand-off; page budget never overcommitted | `{"budget", "handoffs": [...]}` | `Plow.Sram.occupancy_le_of_temporal_fit` |
| D | Counter protocol correctness | Counter-gated schedule enforces every dependency; deadlock-free | `(TaskGraph, CounterProtocol, AddressMap)` bundle | `Plow.Protocol.protocol_covers_deps`, `happensBefore_acyclic`; `Plow.Memory.AddressMapSound` |
| E | Wire format round-trip | `decode(encode(f)) = some f` and `encode(f) = raw` — encoding is lossless | `{"frames", "raw"}` | `Plow.Wire.decodeProgram_encodeProgram` |
| F | Allocation safety | No byte-overlapping address-map entries lack a bridging ordering (reclamation-safe) | same bundle as D | `Plow.Memory.AddressMapSound` (via `Plow.Verify`) |
| G | Staged-LDS fit | Every always-staged GEMV instance fits the decode object's LDS arena | `{"arena", "ops": [...]}` | `Plow.LdsFit.fits_of_check_ok` |

Detail per checkpoint:

### Checkpoint A — Rewrite rule soundness
Each Stage-2 egglog fusion rule (annotated `; rule: <name>` in
[`crates/rewrite/src/egl/rules.egg`](../../crates/rewrite/src/egl/rules.egg))
fuses a composition of base ops into one fused op. In Lean each fused op
unfolds via `Plow.Rewrite.expand` to exactly its unfused composition, and each
rule's soundness is a definitional-equality (`rfl`) theorem `Plow.Rewrite.rule_*`
in [`Plow/Rewrite.lean`](../../lean-plow/Plow/Rewrite.lean). `checkA` submits the
list of fired rule names and rejects any name not in the closed enumeration
`Plow.Rewrite.soundRules`. Today that table holds 19 entries, one per `rule_*`
theorem, mirroring the 19 `; rule:` annotations in `rules.egg`.

### Checkpoint B — Tile partition validity
`checkB` runs `checkTileCandidate` over each candidate: positive dims, each tile
dim ≤ its GEMM dim, and the cost bound `tileCount · bm · bn · bk ≤ cost_bound`.
Coverage completeness is the universal theorem `tile_partition_covers`; the
executable check is backed by `check_sound`
([`Plow/TilePartition.lean`](../../lean-plow/Plow/TilePartition.lean)).

### Checkpoint C — SRAM temporal fit
`checkC` re-checks each `Handoff` record (`producer_pages`, `consumer_pages`,
`producer_release`, `consumer_acquire`, `consumer_release`) against a budget;
rejection names the first violating hand-off. The Rust `sram_fit` pass already
filters candidates against this rule — the Lean check closes the promotion story
with the universal theorem `occupancy_le_of_temporal_fit`
([`Plow/Sram.lean`](../../lean-plow/Plow/Sram.lean)).

### Checkpoint D — Counter protocol correctness
`checkD` runs the executable address-map verifier via the stack-safe twin
`Plow.CLI.FastCheckD` (the reference recursion overflows on real ~590k-task
schedules) plus a reader/writer disjointness check, together establishing the
**strict** `AddressMapSound` guarantee. The universal happens-before theory
(`protocol_covers_deps`, `happensBefore_acyclic`) is proved once in
[`Plow/Protocol.lean`](../../lean-plow/Plow/Protocol.lean).

### Checkpoint E — Wire format round-trip
`checkE` requires both `encode(frames) = raw` and `decode(raw) = some frames`,
catching schema drift on either side; rejection names the first diverging byte
([`Plow/Wire.lean`](../../lean-plow/Plow/Wire.lean)). The Lean model is abstract,
not byte-identical: it proves the framing invariant every scheme must satisfy;
the concrete `packet::Program::to_bytes` layout has its own round-trip suite in
`crates/packet`.

### Checkpoint F — Allocation safety
`checkF` runs the **same** verifier and disjointness check as `checkD` on the
same bundle — F is post-emit re-verification of identical math. The Rust `call`
helper caches the D/F verdict by payload hash so the second spawn is free
([`Plow/Memory.lean`](../../lean-plow/Plow/Memory.lean),
[`Plow/Verify.lean`](../../lean-plow/Plow/Verify.lean)).

### Checkpoint G — Staged-LDS fit
`checkG` accepts one `StagedOp` record per always-staged GEMV instance (`op`,
`idx`, `rows`, `k`, `scratch`) plus the arena size in halves (from `hwspec`);
`fits_of_check_ok` guarantees an accepted list contains no instance whose staged
demand exceeds the arena ([`Plow/LdsFit.lean`](../../lean-plow/Plow/LdsFit.lean)).
The always-staged family reads `x` only through LDS, so an over-fit path would
read garbage and answer fluently-but-wrong; this checkpoint re-checks the
emitter's fusion gate.

## When a new model triggers new proof obligations

- **New fusion rule (the common case).** If Stage 2 added a `; rule: <name>` to
  `rules.egg`, Checkpoint A will reject that name until the Lean side covers it.
  Discharging it soundly requires all four steps in the `Plow.Rewrite` module
  contract:
  1. add a fused variant to the `Op` inductive,
  2. extend `Plow.Rewrite.expand` so the fused op unfolds to its unfused
     composition,
  3. write a `rule_<name>` theorem whose statement is
     `expand (fused …) = <unfused composition>` and prove it (should be `rfl`
     if `expand` is defined correctly),
  4. add the exact rule-name string to `Plow.Rewrite.soundRules`.

  The theorem must be a genuine equality between the fused op and its unfused
  meaning — not `True`, not a tautology, not `sorry`. If it does not close by
  `rfl`, the `expand` definition and the intended semantics disagree; fix the
  definition rather than weakening the theorem.

- **New tile shapes, buckets, schedules, allocations, wire frames, staged
  GEMVs.** No new theorems. The universal lemmas already quantify over all
  inputs; the per-instance checkers (`checkB`…`checkG`) just re-check the new
  data against them. If one rejects, the compiler output is wrong — investigate
  the compiler, not the proof.

- **New op with no fusion.** No Checkpoint-A obligation until a rule fuses it.

## Build Lean and run verification

Requires the Lean toolchain (`elan` + `lake`); the pinned version is in
[`lean-plow/lean-toolchain`](../../lean-plow/lean-toolchain)
(`leanprover/lean4:v4.15.0`). Use `nix develop` for a shell that has them.

Build the proofs and the CLI binary:

```bash
cd lean-plow
lake build            # builds the Plow library (all proofs) + plow_verify + bench
```

A clean `lake build` is itself a proof check: the `Plow` library is the
`@[default_target]`, so if any theorem fails or contains `sorry` the build
fails. `lakefile.lean` sets `autoImplicit := false` and
`relaxedAutoImplicit := false`, so the build is strict by default. The CLI lands
at `lean-plow/.lake/build/bin/plow_verify`.

Smoke-test the binary by hand (the JSON-IPC protocol reads one request from
stdin, writes one `Certificate` to stdout, exits 0 iff `ok`):

```bash
echo '{"checkpoint":"B","payload":{"candidates":[
  {"gemm":{"m":2048,"n":4096,"k":512},
   "tile":{"bm":128,"bn":128,"bk":64},
   "cost_bound":4294967296}]}}' \
  | ./.lake/build/bin/plow_verify
# => {"ok":true,"checkpoint":"B","notes":"tile-partition + cost bound verified: 1 candidates"}
```

How `plowc` invokes it. The Rust client lives in
[`crates/lean_verify/`](../../crates/lean_verify/) (entry points `call`,
`require`, `query` in [`lib.rs`](../../crates/lean_verify/src/lib.rs); typed
per-checkpoint wrappers under `checkpoints/`), and the payload bridge that turns
a compiled bucket into a `ScheduleRequest` is
[`crates/schedule/src/lean_verify.rs`](../../crates/schedule/src/lean_verify.rs).
The whole bridge is gated behind the `lean-verify` cargo feature.

Compile a model with the schedule-path gate on (opt-in, hard-fail on rejection):

```bash
cargo build --features lean-verify        # or the workspace-wide equivalent
plowc <model> --lean-verify               # runs the verifier once per bucket
```

The binary is located via `PLOW_VERIFY_BIN` if set, else `plow_verify` on
`PATH`, else `lean-plow/.lake/build/bin/plow_verify` relative to the crate root:

```bash
export PLOW_VERIFY_BIN="$PWD/lean-plow/.lake/build/bin/plow_verify"
```

On the JSON-IPC round trip, `plow_verify` returns a `Certificate {ok, notes}`
(or `ok:false` + `reason`); a rejection surfaces in `plowc` as
`PlowcError::LeanVerify { bucket, reason }`. A missing/unrunnable binary is a
distinct failure: only `VerifyError::is_binary_unusable` (no runnable
`plow_verify`) is downgraded to a warning and recorded as a skip; every
rejection and every spawn/marshal failure is fatal
(`PlowcError::LeanVerifySpawn`). Building `plowc` without the feature and asking
for verification returns `PlowcError::LeanVerifyDisabled`.

Debug aids: `PLOW_VERIFY_DUMP=<dir>` writes every request to a file for replay;
`cat payload.json | plow_verify` reproduces a single call.

## Success criteria

- `cd lean-plow && lake build` completes clean (no errors, no `sorry`).
- No `sorry` and no vacuous/tautological proofs anywhere in `lean-plow/`
  (`grep -rn 'sorry' lean-plow --include='*.lean'` returns nothing outside
  comments; every `rule_*` theorem is a real equality that closes by `rfl`).
- `soundRules` has exactly one entry per `; rule:` annotation in `rules.egg`,
  and per `rule_*` theorem — no orphans in either direction.
- Every bucket of the new model verifies: all seven checkpoints emit `ok` under
  `plowc --lean-verify` (schedule path: D/F; the devblob path also runs G).
- No checkpoint is silently skipped: a skip is only acceptable when the binary
  is genuinely unusable, and it must be logged and recorded, never read as a
  pass.

## Pitfalls

- **The exit-127 trap.** `lean-plow`'s binary links a `/nix/store` ELF
  interpreter, so outside `nix develop` it can die before `main` with empty
  stdout. That is an unusable binary (downgraded to a skip), not a rejection.
  Run inside `nix develop`, or point `PLOW_VERIFY_BIN` at a runnable build.
- **A skip is not a pass.** If you see "SKIPPED, no runnable plow_verify" the
  gate did not run. Fix the binary and re-run before claiming Stage 3 passed.
- **Vacuous Checkpoint-A proofs.** Making `rule_*` prove `True` or restating the
  fused op as itself certifies nothing. The statement must equate the fused op
  to its unfused composition via `expand`. If `rfl` does not close it, the
  semantics are wrong.
- **Table drift.** Adding a `; rule:` in `rules.egg` without a matching
  `rule_*` + `soundRules` entry makes `checkA` reject the new model; adding a
  `soundRules` entry with no theorem is a hole in coverage. Keep all three in
  lockstep.
- **D/F cache.** D and F run identical math on the identical bundle and share a
  cached verdict by payload hash; do not "fix" a D/F discrepancy by editing one
  handler — they must agree by construction.
- **Abstract E/wire model.** Checkpoint E proves the framing invariant, not the
  concrete `packet` byte layout. Byte-level regressions are the job of the
  `crates/packet` round-trip suite, not this checkpoint.
- **Stack depth on real schedules.** Checkpoint D uses the stack-safe
  `FastCheckD` twin because the reference recursion overflows on ~590k-task
  schedules. If you add a new reference verifier, keep the stack-safe path.

## Code and proof pointers

- CLI dispatch: `runCheckpoint`, `runQuery` in
  [`lean-plow/Main.lean`](../../lean-plow/Main.lean).
- Handlers `checkA`…`checkG`:
  [`lean-plow/Plow/CLI/Checkpoints.lean`](../../lean-plow/Plow/CLI/Checkpoints.lean);
  stack-safe D/F twin
  [`FastCheckD.lean`](../../lean-plow/Plow/CLI/FastCheckD.lean);
  payload/schema types
  [`Payload.lean`](../../lean-plow/Plow/CLI/Payload.lean),
  [`Schema.lean`](../../lean-plow/Plow/CLI/Schema.lean).
- Checkpoint proofs: `Plow.Rewrite.soundRules` / `rule_*`
  ([`Rewrite.lean`](../../lean-plow/Plow/Rewrite.lean)),
  `tile_partition_covers` / `check_sound`
  ([`TilePartition.lean`](../../lean-plow/Plow/TilePartition.lean)),
  `occupancy_le_of_temporal_fit`
  ([`Sram.lean`](../../lean-plow/Plow/Sram.lean)),
  `protocol_covers_deps` / `happensBefore_acyclic`
  ([`Protocol.lean`](../../lean-plow/Plow/Protocol.lean)),
  `AddressMapSound`
  ([`Memory.lean`](../../lean-plow/Plow/Memory.lean),
  [`Verify.lean`](../../lean-plow/Plow/Verify.lean)),
  `decodeProgram_encodeProgram`
  ([`Wire.lean`](../../lean-plow/Plow/Wire.lean)),
  `fits_of_check_ok`
  ([`LdsFit.lean`](../../lean-plow/Plow/LdsFit.lean)).
- Rust client: `call` / `require` / `query`, `Certificate`, `VerifyError`,
  `is_binary_unusable`, `binary_available`, `locate_binary`
  ([`crates/lean_verify/src/lib.rs`](../../crates/lean_verify/src/lib.rs)).
- Payload bridge: `build_schedule_request`, `request_for_bucket`
  ([`crates/schedule/src/lean_verify.rs`](../../crates/schedule/src/lean_verify.rs)).
- plowc integration + errors: `PlowcError::LeanVerify` / `LeanVerifySpawn` /
  `LeanVerifyDisabled`
  ([`crates/plowc/src/lib.rs`](../../crates/plowc/src/lib.rs)); `--lean-verify`
  flag ([`crates/plowc/src/main.rs`](../../crates/plowc/src/main.rs)).
- Fusion rules with `; rule:` annotations:
  [`crates/rewrite/src/egl/rules.egg`](../../crates/rewrite/src/egl/rules.egg).
- Full design: [`docs/arch/08-formal-verification.md`](../arch/08-formal-verification.md).
