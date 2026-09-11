# 19. Packet extensions: adding buckets and rungs to a model that is already compiled

## The problem this solves

A packet is emitted once from a checkpoint and then frozen: `model.pkt` carries the tensor table,
every program (the prefill bucket ladder and the decode rung ladder), their instruction streams and
segment metadata, and `build.json` records the knobs that produced it. Optimisations, however, are
found continuously and almost always arrive as **one more rung** or **one better program for a rung
that already exists**: a 4096-row prefill bucket between 2048 and 8192, a decode rung at 12, a
token-batch body for a bucket that had none, a packed sibling, a split-object variant for a narrow
shape. Today each of those costs a full re-emit of a 745 GB checkpoint, which re-reads 141 shards,
re-derives every program, and produces a packet whose other 12 programs must then be re-qualified
because nothing proves they are byte-identical.

The asymmetry is the point: the expensive, risky part of emit is the part that did not change.

## The shape of the answer

An **extension** is a second, small artifact that adds programs to a parent packet without
rewriting it. The runtime loads the parent, then merges any extensions, and serves the union. The
parent's bytes never change, so every program that was qualified stays qualified by construction.

```
assets/                        assets.ext/bucket-4096/
  model.pkt      <- parent       extension.pkt     <- programs only
  build.json                     build.json        <- knobs + parent identity
  plow_config.h                  plow_config.h     <- must agree on shared axes
  weights.json                   requires.json     <- object stems + hashes
  checkpoint/                  (no checkpoint, no tensors, no weights)
```

An extension may only **add**. It cannot edit an existing program, cannot add a tensor that the
parent does not already declare (except host-filled pointer tables, which are derived, not data),
and cannot change any emit knob that the parent's shared state depends on.

## What has to change, in order

### Phase 1 — programs carry a role, and the ladders derive from roles

Today the split is positional: `dec_lo` is an index boundary, programs `[0, dec_lo)` are the
prefill ladder and `[dec_lo, decode]` the decode rungs (`exec/amd.rs`), with two booleans
(`packed_prefill_only`, `token_batch_body`) carving exceptions out of the prefill range. A merged
program table arrives in whatever order the extensions were found, so a positional boundary cannot
survive.

Replace it with an explicit role on `plow_asset::program::Program`, written by the emitter and
validated at load:

```rust
enum ProgramRole {
    PrefillBucket { rows: u32 },
    DecodeRung    { rows: u32 },
    PackedSibling { of_rows: u32 },
    TokenBatchBody{ band: u32, rows: u32 },
}
```

`prefill_rungs()`, `decode_rungs()`, `decode_rung_lo()`, `plan_chunks` and the object-phase
selection then filter by role and sort by width instead of slicing by index. This is a pure
refactor with no behaviour change, and it is pinned by re-deriving both ladders from an existing
packet and asserting the same sequences the positional code produced.

Do this first and land it alone. Every later phase depends on it, and it is the only phase that
touches code paths a shipped packet already runs.

### Phase 2 — the extension container and its load-time contract

`extension.pkt` is the same container as `model.pkt` with the tensor table replaced by a
**reference** to the parent's: a parent packet hash plus the tensor count and a digest of the
(name, dtype, shape, tp-axis) tuples. At load the runtime refuses unless, in this order:

1. **Parent identity** matches the loaded `model.pkt` hash exactly.
2. **Tensor-table digest** matches, so every tensor index the extension's instructions carry means
   what it meant at parent-emit time. This is the invariant that makes an extension safe at all.
3. **Config compatibility**: the extension's `plow_config.h` agrees with the parent on every axis
   that shapes shared device state — hidden size, head geometry, KV dtype and ring layout, TP
   degree, CU count, decode band width, counter and segment-class limits. Axes that are local to a
   program (tile choice, split count, object family) may differ; that is the whole point.
4. **KV ring invariant**: a new prefill bucket wider than the parent's widest requires
   `ring >= window + chunk - 1`. Full-causal MLA returns `(ctx, MASK_NONE)` and is unaffected;
   a windowed model must be refused by name rather than half-applied.
5. **Ladder well-formedness**: after the merge the prefill widths and the decode rungs are each
   strictly ascending with no duplicates, the decode rungs stay `<= batch`, and every
   `PackedSibling`/`TokenBatchBody` names a bucket that exists in the union.
6. **Budget**: instruction-stream bytes, counters, segments and workspace all fit the arenas the
   parent reserved, or the runtime grows them through the existing VMM pools and says so.

Every refusal names the extension and the rule, the way object-load refusals already do.

### Phase 3 — objects follow the same rule as programs

An extension declares the object stems its programs need in `requires.json` with the same
hash-pinning the object contract (doc 16) already uses, and the loader resolves them from the same
`PLOW_HSACO` directory. The packet-pairing stamp (`plow_packet_hash_{lo,hi}`) must name the
**extension's** hash, not the parent's, so an object built for a bucket cannot silently be loaded
beside a different bucket. `scripts/build_gfx942.sh` already takes `PLOW_HSACO_CONFIG` and builds
for one packet; it gains an extension mode that builds only the rows the extension declares.

### Phase 4 — `plowc extend`

```
plowc extend --parent assets/ --add-prefill-bucket 4096 \
             --replay-knobs assets/build.json --out assets.ext/bucket-4096/
```

It loads the parent's `build.json`, reconstructs the emit context from the checkpoint **metadata
only** (shapes and tensor identities, not the weights), emits the requested programs, and writes the
extension. It refuses any knob that phase 2 rule 3 would reject at load, so an unusable extension
cannot be produced in the first place. The variants worth having on day one are
`--add-prefill-bucket`, `--add-decode-rung`, `--add-packed-siblings`, `--add-token-batch-bodies`.

### Phase 5 — tooling and the audit trail

`plowrt disasm`, `op-audit` and the manifest take `--ext <dir>` (repeatable) and report the union
with each program's origin. `build.json` in the serving directory gains an `extensions` list so a
served packet's provenance is one file. `scripts/asm_audit.py --contract` covers extension objects
under the same baseline.

## What this does not do

It does not let a later discovery change an existing program. If a better kernel for the 8192
bucket is found, the extension mechanism can add a *new* program for it, and the runtime will pick
whichever the ladder rules select — but replacing the old one in place is still a re-emit. That is
deliberate: in-place edits are exactly what would silently invalidate a qualification.

It also does not make extensions free at load: each one costs its instruction-stream upload and its
objects. A serving directory accumulating dozens of extensions should be folded back into a fresh
parent packet periodically, and `plowc extend --fold` is the natural place for that.

## Why this ordering

Phase 1 is a refactor of code that ships today, so it lands alone and is pinned by re-derivation
tests. Phases 2 and 3 are pure additions: until an extension is present, no code path changes.
Phase 4 is the only phase that touches the emitter, and by then the contract it must satisfy is
already enforced by the loader, so the emitter cannot invent its own rules. Phase 5 is reporting.

The first extension worth building, once phase 4 exists, is the 4096-row prefill bucket: the
attribution shows a ragged tail landing in the dense 512 bucket costs 1.9 s at 65k prior context,
and the ladder's gap between 2048 and 8192 is where that tail falls.
