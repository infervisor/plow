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

---

## What phases 2 and 3 actually landed

Branch `packet-extensions`. Container, contract and the object rule; no emitter, no reporting.
There is no extension to load yet, so the unit tests **are** the artifact: they construct
synthetic parent/extension pairs and assert each rule refuses exactly what it should.

### The container as built

`extension.pkt` is the v6/v7 container with two differences and nothing else, so one parser
walks both (`packet::ext`, `packet::devbuild::Model::to_ext_blob`,
`plowrt::asset::devblob::DevBlob::parse_extension`):

```
magic        PLOWEXT\x01   (EXT_MAGIC)        — its own 8 bytes, see below
n_tensor     0
init_bytes   0
sections     SECT_PARENT_REF (kind 7), FIRST, carrying BlobParentRef:
               magic "PEXT" | version 1 | parent_hash[32] | tensor_digest[32] | n_tensor | pad
                                                                              (80 bytes)
kvrow, programs, GQ01, section data, section directory   — the parent format, unchanged
```

**Its own magic, not `n_tensor == 0` on `PLOWDEV`.** A reader that mistook an extension for a
model would declare zero tensors and run programs whose every handle is out of range, and
`DevBlob::find_in_dir` — whose only test is the `PLOWDEV` magic — would report it as a second
model in the serving directory. `is_blob_magic` stays false for `PLOWEXT`; `parse` and
`parse_extension` each refuse the other container by name.

**The digest is over what the container carries.** The design names the tuple
`(name, dtype, shape, tp-axis)`; the tensor table has none of those as fields. `bytes` *is*
dtype × shape, and a tp-sharded tensor already declares its 1/tp share. So
`plow_asset::extension::tensor_table_digest` is SHA-256 over the ordered table —
`(index, name, bytes, initialized)` per entry — with the recovered TP degree folded in, because
that is the one part of the tp axis a per-rank byte count cannot show. Order is digested because
an index is a position. `initialized` is digested because a handle that was a compiler-filled
RoPE table and becomes a runtime-filled buffer means something different to the same
instruction.

**One program-role bit was needed early.** `BlobProgHeader::t` gains
`DECODE_RUNG_PROG = 1 << 29`, set **only in an extension**. A parent's decode ladder is a
trailing ascending run and `decode_rung_lo` finds it positionally — unchanged, byte-identical.
An extension has no position to read a role out of: a lone program is `prog_t.len() - 1`, so the
positional rule calls every extension's single program a decode rung, and a 4096-row prefill
bucket is then refused as a rung outside the decode band. An extension therefore *states* its
roles and an unmarked program is a prefill bucket. This is a down payment on phase 1, in phase
1's own bit space.

### The six rules, and what each refuses

`plow_asset::extension::merge(&ParentFacts, &[ExtensionFacts])`. Pure — no filesystem, no
device — so phase 4's `plowc extend` applies the same six at emit. The rules run in order, per
extension, and the first failure refuses the whole merge; a merge is never half-applied.

Every refusal is `extension `<name>` refused by rule <n> (<rule>): <detail>`.

| # | rule | refuses |
|---|------|---------|
| 1 | parent identity | ``declares parent <sha256> but the loaded model.pkt is <sha256>`` |
| 2 | tensor-table digest | ``was emitted against a N-tensor table, the parent declares M`` — then ``tensor-table digest <a> does not match the parent's <b> — every tensor index in this extension's instructions would mean something else`` |
| 3 | config compatibility | ``<axis> is X here and Y in the parent — that axis shapes shared device state, so it cannot differ (tile choice, split count and object family may)`` |
| 4 | KV ring invariant | ``adds a W-row prefill bucket (the parent's widest is P) at chunk C, which needs a KV ring of R rows; the parent sized its sliding ring at S for window N. The ring is device state the parent allocated and an extension cannot enlarge it — a chunk's rows would wrap onto their own history`` |
| 5 | ladder well-formedness | ``duplicate <prefill bucket\|decode rung> at W rows — <origin> already has one, and an extension may only ADD (replacing a program in place is a re-emit)``; ``decode rung W from <origin> is outside the decode band 1..=B``; ``<packed sibling\|token-batch body> from <origin> names a W-row prefill bucket, which the union does not have`` |
| 6 | budget | ``needs N <arena>, the parent reserved M, and this backend maps its arenas once at load — it cannot grow them``; ``declares N ordered segments, over the fixed ceiling of 2048 — the segment class table is not growable`` |

Notes on three of them.

**Rule 3 is a closed allowlist, not a prefix rule** (`SHARED_CONFIG_MACROS` plus the `blob.*`
axes). From `plow_config.h`: `PLOW_PACKET_GQA`, `PLOW_PACKET_DECODE_BATCH`,
`PLOW_PACKET_LINEAR_BIAS`, `PLOW_PACKET_ROPE_HALF_HD64`, `PLOW_PACKET_ATTENTION_SINKS`,
`PLOW_PACKET_HAS_FLASH_DECODE_FP8`, `PLOW_PACKET_HAS_FLASH_MLA_PREFILL_FP8`,
`PLOW_HAS_FLASH_HD{64,128,256,512}`. From the container: `blob.n_cu`, `blob.target`,
`blob.tp_degree`, `blob.hidden`, `blob.tp_slot_bytes`, `blob.l2_domains`, `blob.l2_sms`.
Everything else — `GM_*`, split count, object family, every `#ifndef`-defaulted knob — is
program-local and MAY differ. An axis present on one side and absent on the other is a
disagreement, reported as `absent`: two artifacts emitted by different compilers cannot be shown
to agree.

**Rule 4 has three states, not two.** `KvRing::FullCausal` (full-causal MLA — the model this
mechanism exists for, and rule 4 never arms for it), `KvRing::Windowed { window, ring_rows }`,
and `KvRing::Unstated`. Unstated is the current state of every shipped packet: `build.json`
records `shapes.max_chunk` and the bucket list but neither the window nor the ring, and
recovering them from instructions means reading a mask operand that lives in `j[1]` on one op
family and `i[7]` on another — the slot-blindness doc 16 records as a live defect class. So
Unstated **refuses a bucket wider than the parent's widest, by name**, and admits everything
else: a decode rung, a packed sibling, a token-batch body, or a narrower bucket never arms the
rule. The motivating 4096 bucket sits between 2048 and 8192 and is unaffected.
`plow_asset::extension::kv_ring_rows` mirrors `devgen::kv_ring_rows`, pinned by
`devgen`'s `extension_kv_ring_rows_mirrors_devgen`.

**Rule 5's "strictly ascending" is really "no duplicates".** Once both ladders derive by
filtering and sorting (phase 1), ascending is by construction; a merged table arrives in
whatever order the extensions were found, so ascending as a property of the *table* is exactly
what the merge gives up. What is left, and what bites, is that no two programs of the same role
sit at the same width — the sorted ladder could not then say which one it means. The parent's
own ladder is checked before anything is merged onto it, and each extension is checked
immediately after it is merged, so the refusal names the extension that broke the ladder rather
than the last one in the directory.

### Loading

`plowrt::asset::extension`. Discovery is `<assets>.ext/*/extension.pkt`, sorted, a **sibling**
of the serving directory (a subdirectory would collide with `find_in_dir`'s ambiguity check).
`PLOW_EXTENSIONS` (colon-separated) overrides it; empty means serve the parent alone.
`load(assets, image, blob, l2_dispatch_ok)` returns the merged ladder plus the loaded
extensions, and logs each arena it grew.

With no extensions present, `discover` returns empty, `merge` is the identity on the parent's
ladder, nothing grows, and no code path changes. That is asserted, not assumed
(`a_serving_directory_with_no_extensions_loads_exactly_as_before`,
`a_v6_parent_blob_is_byte_identical_after_the_container_refactor`).

### Phase 3 — objects

`requires.json` beside `extension.pkt`, hash-pinned the way `decode_objects.json` already pins
objects:

```json
{ "version": 1, "arch": "gfx942", "pairing_hash": "0x…",
  "objects": [ { "stem": "interp_prefill", "sha256": "…", "rows": 4096 } ] }
```

`pairing_hash` is the **extension's** `PLOW_PACKET_HASH`, and `load_one` refuses a
`requires.json` that pins anything else — including, by name, the parent's. `check_object_stamp`
does the same for the object's `plow_packet_hash_{lo,hi}`: a stamp equal to the parent's hash is
reported as *"it was built against the parent's plow_config.h and has none of this extension's
arms"*, and an **unstamped** object — accepted with a warning beside a parent, since that is the
shipped general-object state — is refused beside an extension, because nothing then shows it has
the arms this bucket needs.

`scripts/build_gfx942.sh` gains `PLOW_HSACO_EXTENSION=<extension dir>`, which is exactly two
facts: `PLOW_HSACO_CONFIG` points at the *extension's* `plow_config.h` so every row stamps the
extension's hash, and `PLOW_ROWS_ONLY` is derived from `requires.json`'s stems so only the
declared rows are built — one more rung costs one object, not twenty-eight. It refuses an arch
mismatch, a `requires.json` whose pin disagrees with the extension's own header, and a
conflicting `PLOW_ROWS_ONLY`. `PLOW_ROWS_ONLY` itself now takes a comma-separated list (each
entry `=exact-stem` or a substring), since one substring cannot name a set; a single entry
behaves exactly as before.

### What phase 4 must satisfy

1. **Write the roles.** Every program in an `extension.pkt` states its role; an unmarked program
   is a prefill bucket. Decode rungs carry `devbuild::decode_rung_program_t(rows)`.
2. **Declare no tensors.** `Model::to_ext_blob` asserts an empty tensor table and no generated
   tensors, rather than letting the container carry an index the parent's table does not have.
3. **Record the KV geometry.** `plowc` must add `shapes.kv_window` and `shapes.kv_ring_rows` to
   `build.json` (`kv_window == 0` ⇒ full-causal). Until it does, no extension can add a prefill
   bucket wider than the parent's widest — rule 4 refuses it for lack of the fact, which is the
   conservative direction but blocks exactly the case a wider bucket is wanted for.
4. **Emit a `plow_config.h` whose shared axes match the parent's** on every macro in
   `SHARED_CONFIG_MACROS`, and a `requires.json` pinning its own `PLOW_PACKET_HASH`.
   `plowc extend` should call `plow_asset::extension::merge` against the parent it was given and
   refuse at emit whatever the loader would refuse at load — that is why the contract is a pure
   function in a crate below both.
5. **`--fold` must re-derive, not concatenate.** Folding extensions back into a parent produces
   a packet whose tensor table is the parent's, so every folded program's indices still resolve;
   a fold that renumbered tensors would invalidate every extension still on disk.

### Not done here

Phase 1 (`ProgramRole` on `plow_asset::program::Program`) had not landed, so roles are derived
behind an adapter: `roles_from_positional` reproduces today's `decode_rung_lo` + two-boolean
split for a parent, `roles_from_flags` reads an extension's stated bits. When phase 1 lands,
both collapse into reading `p.role`; the rules only ever see `&[ProgramRole]` and do not change.
`ProgramRole` is defined in `plow_asset::extension` with the design's exact spelling, and moves
to `program.rs` with phase 1.

Rule 6's `workspace_bytes` is the parent's `weights.json` arena high-water mark; an extension's
own workspace demand is `0` until `plowc extend` writes a `BucketStat` for the programs it
emits. The other three arenas (instruction-stream bytes, counters, segments) are derived from
the container itself and are live now.

Phase 4 (`plowc extend`) and phase 5 (disasm / op-audit / manifest `--ext`) are untouched.
