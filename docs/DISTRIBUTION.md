# Distributing plow

```
plowrt load kimi-k3          # probe this box → pick the matching build → pull → serve
```

A released `plowrt` binary, no repo checkout, no compiler, no nix on the serving
host. This document is the contract between the producers (which use nix) and
the consumers (which use HTTPS).

## What is distributed, and what is not

| class | contents | distributed | keyed by |
|---|---|---|---|
| runtime | `plowrt` | yes | platform triple |
| objects | `*.elf` / `*.cubin` + `build_defines.json` | yes | fingerprint × profile digest |
| model bundle | `model.pkt`, `build.json`, `weights.json`, `plow_config.h`, derived sidecar | yes | model × target × generation |
| checkpoint | HF safetensors, `tokenizer.json`, `chat_template.jinja` | **no** | a declared reference |

**plow distributes only what plow produced.** Anything the HuggingFace snapshot
already carries is linked from it at `prepare` time. That is why a tokenizer is
normally absent from a bundle: `plowc` does not generate one, it symlinks the
checkpoint's own file, so shipping it would redistribute HF content and risk it
diverging from the checkpoint it must match.

**The one exception is a tokenizer plow reconstructs.** Kimi-K3 ships
`tiktoken.model` and no `tokenizer.json`, so `scripts/kimi_k3_tokenizer.py`
rebuilds one — 163,584 tiktoken ranks recovered into HF BPE merges, verified for
exact id-sequence equality. That is a compiled artifact and it travels.
`bundle.json` states which case applies, and `prepare` refuses a
`source = "checkpoint"` tokenizer the snapshot does not have.

## Addressing

```
[<registry>/]<namespace>/<name>[:<label>][@g<n>]

plowrt load kimi-k3                           # → infervisor/kimi-k3, probe decides
plowrt load acme/kimi-k3-finetune             # a third-party namespace
plowrt load kimi-k3 --tp 4 --max-ctx 16384    # constrain axes
plowrt load kimi-k3:gfx942-mi325x-tp8-32k@g3  # fully pinned
plowrt load kimi-k3 --registry file:///mirror # a local mirror, no HTTP client needed
```

A bare name expands to `$PLOW_REGISTRY/infervisor/<name>` — the role ollama's
`library/` plays. A reference resolves **directly to a manifest URL**, with no
global catalog to scan, so publishing a model writes only under its own prefix.

A **variant** is one compiled build of a model, keyed on target × parallelism ×
context × features × build generation. `label` renders the first four;
`generation` counts builds of that label. An unpinned `load` takes the newest
compatible generation and **writes the pin it chose** — nothing moves it
afterwards except an explicit `plowrt upgrade`, so a served model does not drift
because a catalog refreshed.

## Store layout

Remote:

```
v1/<namespace>/<name>/index.json                # this model's variants
v1/<namespace>/<name>/manifests/<label>@g<n>    # one variant → bundle.json
v1/objsets/<objset-id>.json
v1/blobs/sha256/<hex>[.zst]                     # shared by everything
v1/runtime/index.json                           # released binaries, per triple
catalog.json                                    # model names only
```

Local (`$PLOW_HOME`, default `~/.plow`):

```
blobs/sha256/<hex>          content-addressed
refs/<digest>.json          {ref, variant} for one pin
bundles/<variant_id>/       hard links into blobs/ — what --assets receives
checkpoints/<org>--<repo>@<rev>/   the farm: links to the snapshot + derived shard
```

A blob's digest always names the **uncompressed** bytes, so a store may hold
`.zst`, plain, or both and every client stays correct.

## Why content addressing

A generation that changes only the code objects shares its packet, manifests and
derived sidecar with its predecessor, so an upgrade transfers ~40 MB of objects
rather than ~400 MB. The client reports what it will actually move, computed from
digests rather than from what kind of change it was:

```
$ plowrt ls --upgradable
infervisor/kimi-k3  a793020a5026  → gfx942-mi325x-tp8-32k@g3 available, 41.0 MiB to fetch
```

The rule that keeps this correct: a **specialised** object stamps the packet it
belongs to (`plow_packet_hash_{lo,hi}`) and pairs only with it; a **general**
object — every arm compiled — carries no stamp and pairs with any packet. So a
same-packet objset swap is legal only when the new objset is general or its stamp
still matches. `pack_bundle.py` refuses to publish a mismatch, which turns a
refusal deep in the loader into an obvious one at publish time.

## Selection

`load` probes the machine into a `HardwareFingerprint` and filters the published
variants. Refusals name both sides:

```
$ plowrt show kimi-k3
this machine: amd gfx942 MI300X, 304 units, 8 GPU(s), 192.0 GiB
infervisor/kimi-k3  hf:moonshotai/Kimi-K3 @…
  gfx942-mi325x-tp8-32k@g3  validated    8 GPU  ctx 32768   no — needs 274877906944 bytes …
  gfx942-mi300x-tp4-18k@g1  emits        4 GPU  ctx 18432   RUNS
```

Ranking: **generation first** — a newer generation of one label is the same model
on the same hardware, only built better — then `validated` over `emits`, then an
exact SKU match, then GPU count, then measured throughput. A tie is an error, not
a coin flip.

Two probe details worth knowing. On AMD the agent name *is* the ISA key
(`HSA_AGENT_INFO_NAME` returns `gfx942`, the same string spliced into the kernel
symbol the loader resolves). And capacity is what separates MI300X from MI325X —
same die, same gfx942, same 304 CUs, 192 vs 256 GiB — which is why the HSA memory
probe exists. A variant records the *datasheet* capacity while a probe reports
the *usable pool*, so they are compared with a tolerance rather than `>=`; on a
live MI300X the gap is 16 MiB.

## Serving is offline

`serve` performs **no network I/O**. `--model` resolves from the local store
only, and an unpulled model is an error naming the `load` that fixes it. The CLI
does the fetching, in a separate process; CI asserts no HTTP/TLS crate is present
in the serve-only feature set.

```
plowrt load  kimi-k3 --checkpoint /models/Kimi-K3
plowrt serve --model kimi-k3 --port 8080
```

## Producing a release

Producers use nix; the HTTPS store is generated from its outputs. Nothing on a
serving host runs nix, and a published manifest that references `/nix/store` is
refused at pack time — an exported bundle must be portable.

**Runtime binaries** (`<semver>+<git12>`, the semver saying what changed and the
commit saying which source produced the bytes):

```
nix build .#plowrt-release --no-link --print-out-paths
scripts/release_runtime.py --release <path> --store ./dist
```

The tarball is deterministic (sorted, fixed mtime, no gzip timestamp) and its
sha256 is computed at build time. A build from a modified checkout carries
`-dirty` and is refused unless `--allow-dirty`, because it names a commit it does
not correspond to.

**Model assets**, which advance independently — a new generation needs no new
binary:

```
scripts/release_dist.py --recipe recipes/infervisor/kimi-k3/<label>.toml --dry-run
scripts/release_dist.py --recipe … --commit <full-sha> --publish --store ./dist
```

Five stages, each refusing rather than degrading: resolve the source to a full
commit exactly once (a branch name is not provenance); refuse a compiler whose
source commit disagrees with the recipe, even when its version string matches;
validate the target, the pairing, and — for a `validated` claim — a measurement;
report the generation and what it supersedes; then publish under a per-model
lock, re-reading the index inside it so a concurrent release cannot reuse a
generation.

Publication order is the safety property: **blobs, then manifests, then the
index**. An interrupted publish leaves the previous generation completely
servable.

## Recipes

`recipes/<namespace>/<name>/<label>.toml` is consumed by builds and publishing;
the `.md` beside it is prose with generated command blocks. One file per label,
not per generation — its git history is the generation ledger.

```
scripts/check_recipe.py recipes/… --bundle <assets> --objects <objects> --strict
scripts/render_recipe.py recipes/… --check
```

`check_recipe.py` asserts the recipe against what the build recorded:
`[emit].env` against `build.json`'s `emit_config.replay`, `[objects].env` against
`build_defines.json`, `[target]` against both manifests, and every
`[artifacts]` digest against the file. `PLOW_DEFINES_ONLY=1 build_gfx942.sh`
resolves the `-D` set without compiling, so the check takes seconds.

`status` is `validated` (measured), `emits` (builds and loads, no measurement) or
`refused` (`plowc` cannot emit it; the recipe carries the file:line that refuses
it). A `refused` recipe is publishable and appears in `show`, but is never
selectable — which is how "can plow serve this?" becomes answerable from the
catalog instead of from a 2,700-line investigation log.

## Host requirements the distribution cannot supply

* `/dev/kfd` and `/dev/dri/renderD*` passed through, and the user in `render`
  (AMD).
* `ROCR_VISIBLE_DEVICES` for partial assignment — **not** `HIP_VISIBLE_DEVICES`:
  plowrt dlopens ROCr directly and never loads HIP.
* A driver ABI-compatible with the objects; `plowrt devices` prints the
  fingerprint and driver it resolved, which is the report to paste into a bug.
