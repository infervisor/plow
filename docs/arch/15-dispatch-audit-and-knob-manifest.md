# The dispatch audit and the knob manifest

Two `build.json` sections that exist to make one bug class visible at emit instead of by
disassembly, and one threshold knob that decides how loudly.

The object-side complement is [the object contract](16-object-contract.md): the same doctrine
applied to the built code object rather than the emitted packet, asserting the `-D` set that
actually compiled, spill counted from the ISA, and the dispatch arms that exist. It shares
`PLOW_AUDIT_STRICT` and the committed-baseline-plus-`--bless` shape with this one. Where a
finding needs both halves — the GEMV ceiling is the clear case, `gv_mm_max` at emit against
`plow_gemv_mm_cap_<n>` in the object — the two are checked independently and by different
means, on purpose.

## The bug class

A **compiled ceiling**: a compile-time constant that does not match the shape the runtime
presents. Two shipped on Gemma-4-31B/MI300X, each worth 20–40%, each invisible in every log,
manifest and test, and each found only by disassembling the built object.

| | what was wrong | worth |
|---|---|---|
| `PLOW_GEMV_MM` | the decode object compiles ONE `gemv_rows` body serving every `M <= MM` by predicating each activation row and computing its dot product anyway. A `MM=4` object against a blob carrying T=1/2/4 decode rungs ran four dot chains per batch-1 token and discarded three. | +22–37% throughput, −28% TPOT at concurrency 1 |
| `pick_tile`'s CU budget | ranked candidates by `rounds × per-tile cost` with `rounds = ceil(tiles / n_units)` and was handed the **global** 304 CUs, while `split3`/`split2` give q/k/v and gate/up **disjoint** sets of 76 or 152. | −5.2% / −4.6% TTFT at 128/512 |

Neither needed new information. `cus.len()` was in scope at the emit site; the GEMV ceiling is
`tuning.gv_mm_max`, which the manifest has always written. What was missing was that nobody
wrote the two numbers down next to each other, where a diff would show one moving.

## `dispatch_audit`

`crates/devgen/src/dispatch_audit.rs`. Derived from the **emitted instruction stream**, like
the rest of `manifest.rs`: `M`/`N`/`K` are `i[0..3]` (uniform across the matmul family), the
workgroup count is `DevInst::blocks`, and `cus` is the distinct CU count carrying a stream
entry for the instruction — recovered from `Program::stream`/`stream_ofs`, i.e. the dispatch
the blob on disk actually carries, not the emitter's intent.

Rows are deduplicated by everything except the instruction count, so a 60-layer model produces
a table a person can read (36 rows for Gemma-4 31B, 48 for GLM-5.3 TP4).

```
ops[]              op, kind, t, m, n, k, cus, workgroups, tile, tiles, rounds,
                   occupancy, bytes, waste_bytes, insts
gemv_ceiling       compiled_m (= tuning.gv_mm_max) and, per row, live_rows,
                   computed_per_useful, wasted_row_fraction
worst_by_cost[]    the top 16 by (1 - occupancy) x bytes x insts
findings[]         what tripped a threshold
thresholds         the floors in force and the env vars that move them
```

**`occupancy = tiles / (rounds * cus)`**, which equals `tiles / cus` in the single-round case.
Defined over `rounds * cus` rather than clamping `tiles / cus` at 1 because clamping destroys
the multi-round signal: gate/up at T=128 is 168 tiles on 152 CUs, which clamps to a meaningless
`1.0` and is really 55.3% of two rounds.

Ordering is by **what the shortfall costs**, not by how bad the ratio looks — a 5% occupancy on
a 2 MiB op is noise and a 55% occupancy on a 300 MiB op is the whole regression. The cost is
exact integer arithmetic (weights counted in half-bytes so mxfp4 is an integer), because a float
sort key would make two emits of one blob differ in the last digit and defeat the point.

### What is deliberately not scored

Occupancy needs a tile, and the tile is only knowable where **this compiler chose it** — the
`pick_tile` rungs, via `gemm_tile_of` off `GFX950_RUNGS`. The GLU-fused GEMMs take theirs from
`GM_BM`/`GM_BN`, which are `-D` defines of the object, and the grouped-MoE ops carry no token
count in the packet. Those get a dispatch row with `tiles: null` and no occupancy. A guessed
tile would put a fabricated percentage in the manifest, which is worse than an absent one.

### What it reports on the two real blobs

Gemma-4 31B, gfx942/MI300X, `PLOW_DECODE_BATCH_LADDER=1,2,4` — both known bugs, at emit:

```
occupancy: GemmSmall 128x5376x21504 @t128 fills 27.6% of its dispatch
           — 84 tile(s) over 304 CU(s) in 1 round(s), ~9859 MiB moved at that fill.   (down_proj)
occupancy: GemmSmall 128x5376x8192  @t128 fills 27.6% ...                             (o_proj, sliding)
occupancy: GemmSmall 128x5376x16384 @t128 fills 27.6% ...                             (o_proj, full)
GEMV ceiling: Gemv @t1 runs 1 live row(s) under a compiled M=4
           — 4.0x computed per useful row, 75.0% of the row work discarded.
```

and the `pick_tile` budget defect is visible as the two rows differing in `cus`: gate/up is
`GemmWide 128x21504x5376` on **152** CUs at 55.3%, down_proj is `GemmSmall` on **304** at 27.6%.
The GEMV ceiling is a per-rung table — T=1 → 4.0x, T=2 → 2.0x, T=4 → 1.0x — which is the
`PLOW_GEMV_MM` bug stated as a number, per program.

GLM-5.3 TP4 at 131072: 17 occupancy findings (worst are MLA's narrow projections — `kv_a_proj`
`8192x64x6144` at 42.1%, `q_lora` `128x6144x4096` at 31.6%), and a **clean** GEMV ceiling,
because that blob's decode ladder is T=1 only and `gv_mm_max` is 1. The matched configuration
is silent, which is what keeps the check worth reading.

## The threshold check

| env | default | effect |
|---|---|---|
| `PLOW_AUDIT_OCC_FLOOR` | `50` | occupancy percent below which a matmul is a finding |
| `PLOW_AUDIT_GEMV_WASTE_MAX` | `25` | percent of GEMV row work that may be spent on dead rows |
| `PLOW_AUDIT_STRICT` | `0` | `1` promotes findings from a `WARN` to a refusal (exit 1) |

A warning by default, by the rule `warn_uniseg_on_amd` states — the test is "what does the
caller get if I ignore this?", and here the caller gets a **correct** blob that is slower than it
needed to be. `PLOW_AUDIT_STRICT=1` is for the build that has already been tuned: a campaign
re-emitting a configuration it measured, or CI holding a blob to a floor it has met before.

The 50% floor is where a dispatch stops being explicable as tile quantization and starts being a
half-idle machine: it names the 27.6% case and not the 55%/84% ones, which is the intended
sensitivity — the tile-budget defect is a *ratio between two rows*, and `worst_by_cost` is what
surfaces that.

The 25% GEMV ceiling admits a T=3 program on an `MM=4` object (the honest cost of a
power-of-two ceiling) and refuses T=1-on-`MM=4`.

## `emit_config`

The deliberate exception to `manifest.rs`'s "derive from the instruction stream, never from
intent" rule, and it has to be one: the point is precisely to record the intent, because a
rebuild cannot reconstruct it from the packet.

Before this, only GEMM tiles were measured and persisted, keyed by a digest that goes stale on
any `runtime/amd` edit; the ~146 emit knobs were env reads recorded nowhere, and the only durable
record of a winning configuration was prose in a markdown file.

```
knob_count   146
knobs[]      id (the field name), env, value, source
replay{}     the env assignments that reproduce this configuration
```

`source` is **clap's own `ValueSource`** — `cli`, `env`, `default` — not inferred. Inferring it
("the env var is set, so that must be the source") is wrong the moment a flag overrides an env
var. `plowc` therefore parses with `Cli::command().get_matches()` and hands the matches to
`emit_config::record_knobs`; the `from_env` path records `env`-or-`default` by probing, which is
exact there because that path has no command line to lose to. A fourth value,
`production_default`, marks the two places the emitter overrides a parsed value after the fact
(`apply_production_defaults`, `effective_uniseg`).

`replay` carries only knobs whose value came from a flag or an env var. **Defaults are omitted
deliberately**: writing them down would pin them, so a default that is later promoted would
reach a fresh build and not a replayed one.

### Replaying

```
plowc --replay-knobs path/to/build.json --hf-dir … --emit devblob …
```

Applied to the environment before clap parses, so an explicit flag in the replaying invocation
still wins, and an already-exported variable is not overwritten.

### The coverage boundary

The section covers the knobs **`EmitConfig` declares**. An emit-affecting env var read outside
`EmitConfig` is invisible here exactly as it is invisible everywhere else. Measured: a GLM-5.3
TP4 replay from this section reproduces the blob **byte-for-byte** once `PLOW_MLA_PF_V2` is also
supplied — that one variable is read in `crates/packet/src/devbuild.rs` and is not a field, so
nothing in devgen can see it. Promoting such a read to an `EmitConfig` field is what makes it
recordable; that is the follow-up, not a gap in the recorder.

## Where the sections sit

Both are **new top-level keys**. `pairing_hash` covers `union`/`objects`/`tuning` because those
are what `plow_config.h` compiles; an occupancy number or a knob that changed no arm must never
invalidate an otherwise-good packet/object pair. Emitted blob bytes are unchanged — the golden
blob hashes in `crates/devgen/tests/golden_blob.rs` are untouched, and `build.json` is written
after the blob in any case.
