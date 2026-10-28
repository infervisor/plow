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
v1/<namespace>/<name>/sigs/<sha256>.sig         # ed25519 over that version of index.json
v1/<namespace>/<name>/manifests/<label>@g<n>    # one variant → bundle.json
v1/objsets/<objset-id>.json
v1/blobs/sha256/<hex>[.zst]                     # shared by everything
v1/runtime/index.json                           # released binaries, per triple
catalog.json                                    # model names only
```

Local (`$PLOW_HOME`, default `~/.plow`):

```
blobs/sha256/<hex>          content-addressed
refs/<digest>.json          {ref, variant, previous[]} for one pin
bundles/<variant_id>/       hard links into blobs/ — what --assets receives
checkpoints/<org>--<repo>@<rev>/   the farm: links to the snapshot + derived shard
snapshots/<org>--<repo>@<rev>/     a snapshot `load --fetch-weights` downloaded
```

A blob's digest always names the **uncompressed** bytes, so a store may hold
`.zst`, plain, or both and every client stays correct.

A pulled bundle directory also holds its manifest as `bundle.json`, which is
where `plowrt ls --json` reads each pin's label and generation from. The
array it prints on stdout (logs go to stderr) is what a node reports:

```
[{"ref": "infervisor/kimi-k3:gfx942-mi325x-tp8-32k@g3", "namespace": "infervisor",
  "name": "kimi-k3", "label": "gfx942-mi325x-tp8-32k", "generation": 3,
  "variant_id": "a793020a5026", "bytes": 412316860,
  "dir": "/home/op/.plow/bundles/a793020a5026", "prepared": true}]
```

`bytes` counts the blobs the bundle links (files and objects, not the
checkpoint); `prepared` means the checkpoint farm is linked in, so `dir` serves
as is. A pin pulled before the manifest was recorded is skipped with a warning
until `plowrt load` re-materializes it.

### Generations on a host

A pin is a profile: `upgrade` moves it to the new generation and records the
one it left in `previous` (the last two). The old bundle stays in the store,
so `plowrt rollback <model>` is a pointer move, offline and instant, and a
second `rollback` moves forward again. `upgrade` never frees space;
`plowrt gc` removes the superseded generations of pinned models (never a
bundle of a model nothing pins) and then the blobs no bundle references. A
generation `gc` has taken cannot be rolled back to; `rollback` says so and
leaves the pin alone.

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
still matches. `plowrt pack` refuses to publish a mismatch, which turns a
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

## Runtime-compiled targets

Apple's Metal shaders are compiled on the serving machine from the packet, so
an Apple bundle (`target.vendor = "apple"`, `isa = "metal3"`) ships **no
objset**: the bundle's files are the whole build, `build.objset_id` is empty,
and there is no pairing hash. `plowrt target --json` on such a machine prints
the exact `target` a publisher records in the recipe's `[target]` — including
`mem_bytes`, which a unified-memory part must state because `hwspec` records the
smallest shipping configuration of each chip rather than this machine's memory;
`plowrt pack` takes no objset for it. Selection is
unchanged: the Metal device reports `apple`/`metal3`, the SKU hwspec names,
its core count and the memory Metal will let it use.

## Serving is offline

`serve` performs **no network I/O**. `--model` resolves from the local store
only, and an unpulled model is an error naming the `load` that fixes it. The CLI
does the fetching, in a separate process; CI asserts no HTTP/TLS crate is present
in the serve-only feature set.

```
plowrt load  kimi-k3 --checkpoint /models/Kimi-K3
plowrt serve --model kimi-k3 --port 8080
```

Without a snapshot on the host, `load --fetch-weights` downloads the whole
repository the bundle names (`hf:<org>/<repo>` at its revision) through
Hugging Face, with `HF_TOKEN` for a gated repo, into `snapshots/` in the
store, as `huggingface-cli download` would. Never implicit: a checkpoint can
be terabytes. The bundle's revision must be a full commit hash; a branch can
move after the build, so `--fetch-weights` refuses one.

The exception is a checkpoint with no Hugging Face repository to fetch from
(private, or for an air-gapped node): `checkpoint.source = "dist"` lists every
file as `files: [{name, sha256, bytes}]`, published under `v1/blobs/sha256/`
like a bundle file, and `revision` is a free version label. `pull` fetches those
blobs with the bundle and links them into `snapshots/dist+<ns>--<name>@<rev>-<hash>/`,
which `load` prepares without `--fetch-weights`. A recipe whose
`[checkpoint] source = "dist"` names the snapshot in `revision`, and
`plowrt pack --checkpoint-dir <dir>` both writes such a manifest and zips the
files as `checkpoint/<name>`.

## Producing a release

Producers use nix; the HTTPS store is generated from its outputs by `plowrt`
and the registry, never by a script. Nothing on a serving host runs nix, and a
published manifest that references `/nix/store` is refused at pack time — an
exported bundle must be portable.

**Runtime binaries** (`<semver>+<git12>`, the semver saying what changed and the
commit saying which source produced the bytes):

```
nix build .#plowrt-release --no-link --print-out-paths
```

The tarball is deterministic (sorted, fixed mtime, no gzip timestamp) and its
sha256 is computed at build time. A dynamic Linux release is re-pointed at
the host's loader (`/lib64/ld-linux-x86-64.so.2`) with its nix rpath removed
inside the build, so the digest is of the bytes a host runs; it needs glibc
2.38 or newer. A build from a modified checkout carries `-dirty` and is
refused at publish, because it names a commit it does not correspond to.

```
plowrt push-runtime --release <nix output dir> [--release <another triple's>] \
  [--install-script scripts/install.sh] [--keep-latest] [--dry-run] [--to r2://…]
```

records each output under `v1/runtime/<version>/<triple>/` (the tarball,
its `.sha256`, `release.json`; create-only, so a version is never
republished with other bytes), rewrites `v1/runtime/index.json` (every
release and triple, newest first, for a models page) and `v1/runtime/latest`
(the version string alone), and publishes `scripts/install.sh` at the
registry root when asked. The same credentials and conditional writes as
`push`. A client installs with:

```
curl -fsSL https://pdist.infervisor.ai/install.sh | sh     # PLOWRT_VERSION=… pins
```

which reads `latest`, then that version's `release.json` for the machine's
triple, verifies the tarball against its `.sha256`, and installs `plowrt`
into `/usr/local/bin` or `~/.local/bin` (`PLOWRT_INSTALL_DIR` overrides).

**Model assets**, which advance independently — a new generation needs no new
binary — are published by `plowrt` alone, straight into the bucket a client
reads. There is no service between the producer and the store, and no
wrapper script: the sequence is the recipe's build, its check, then `push`.

```
python3 scripts/campaign/campaign.py build recipes/<ns>/<name>/<label>.toml --out $OUT
scripts/check_recipe.py recipes/<ns>/<name>/<label>.toml --bundle $OUT/assets --objects $OUT/objects --strict
plowrt push --recipe recipes/<ns>/<name>/<label>.toml --assets $OUT/assets --objects $OUT/objects --dry-run
plowrt push --recipe … --assets … --objects …           # same, for real
plowrt show <name>                                       # from any machine: the generation is live
```

A measurement is not a condition of publishing: a build goes up as `emits`
and is re-pushed as `validated` with `--tok-s` when it has been gated and
measured. The registry carries what plow produced and nothing else; a
checkpoint is always a Hugging Face reference, pinned to a commit.

```
CLOUDFLARE_ACCOUNT_ID=… CLOUDFLARE_R2_BUCKET=… CLOUDFLARE_API_TOKEN=… \
plowrt push --recipe recipes/…/<label>.toml --assets <dir> \
  [--objects <dir>] [--checkpoint-dir <dir> --hf <org>/<repo>] \
  [--status validated --tok-s <n> --concurrency <c>] [--dry-run] \
  --key-file <secret> [--to r2://<account-id>/<bucket>]
```

Cloudflare R2 is the default destination: `--to` is `r2://` (account and
bucket from `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_R2_BUCKET`),
`r2://<bucket>` or `r2://<account-id>/<bucket>`. Writes go through
Cloudflare's REST API with `CLOUDFLARE_API_TOKEN`, or through R2's S3
endpoint with `CLOUDFLARE_R2_ACCESS_KEY_ID` + `CLOUDFLARE_R2_SECRET_ACCESS_KEY`
when no API token is set. `s3://<bucket>` is any S3 bucket from the `AWS_*`
environment (`AWS_ENDPOINT` for another S3-compatible service).
`file:///dir` writes the same tree, which is how a release is inspected or
mirrored by hand. The client is our fork of Apache `object_store`
(`storyvis/object-store`, pinned by revision); it is pure Rust, so `plowrt`
links nothing new. Both manifests are built from the recipe, which is the
only way to produce them: there is no flag that can disagree with what the
recipe declares. The recipe's `plow_git` is the provenance a published build
records; `--plow-git` exists only to reproduce a manifest whose recipe has
since moved. A `validated` status needs `--tok-s`. `plowrt pack --manifests
<dir>` writes the two JSON files without uploading, for inspecting them.

`push` numbers the build against the live index: the next generation of its
label, or nothing at all when the newest generation already carries exactly
these files, objset, status and measurement, so a re-run is a no-op. It then
uploads only the blobs the bucket lacks (a `HEAD` per digest; an objects-only
change moves the objects and nothing else) and records what the generation
supersedes and what changed (`packet`, `objset`, `both`). Publication order
is the safety property — **blobs, then manifests, then the signature, then
the index, then the catalog** — so an interrupted push leaves the previous
generation completely servable, and a generation is never reused. Every
immutable object is written with `If-None-Match: *` and the index and catalog
with `If-Match: <etag>`, so a generation someone else published meanwhile is
refused by the bucket, not overwritten; `push` re-reads the index and tries
the next number, up to four times. A manifest name already taken is adopted
when it holds this very build (a push that died before its index write) and
skipped when it holds another's. A push that died after the index is finished
by the next run of the same build: it writes the missing signature and catalog
row instead of reporting "unchanged". An objset name already holding other
bytes, or a rung override whose objset is not published, is refused before the
bundle is written. A bundle is also refused when the recipe says the model
serves with a file the build did not produce (`codec.pkt` for a `[checkpoint]
codec`, `cublaslt_algos.jsonl` for `PLOW_LT_RUNG_ALGOS`).

**Signing.** `plowrt keygen --name <key-name> --out <file>` makes an ed25519
key; its public half is printed as `<key-name>:<base64>`. The public registry
at `pdist.infervisor.ai` is signed with
`infervisor-2026:ESz+iXQToovy4GhRUgTHoUKUwMumUA/ls/4tAaEjL3M=`. `push --key-file`
(or `PLOW_SIGNING_KEY`) writes the signature under `sigs/<sha256 of the
index>.sig`, before the index itself: a reader fetches the signature of
exactly the bytes it read, so no reader sees an index without its signature
and two publishers' signatures cannot land on each other's index. A client
with `PLOW_TRUSTED_KEYS=<name>:<base64>[,…]` refuses an index that is
unsigned, signed by another key, or changed after signing; the index names
every manifest by digest and each manifest its blobs, so this one signature
covers the whole pull. A client with no trusted keys checks nothing, as
before. `push` applies the same `PLOW_TRUSTED_KEYS` to the index already in
the bucket, whose rows it carries into the new, signed index; with none set it
warns that they are carried unverified, which is the bootstrap for a registry
published before signing existed. Without a key, `push` to a bucket warns and
publishes unsigned. A `file://` registry enforces the same conditional writes
under `<dir>/.lock`.

`plowrt publish` (the registry's staged-upload API, `PLOW_PUBLISH_KEY`) is
the transition path, kept until every model has been re-published with
`push`; new publications use `push`.

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

A Hugging Face checkpoint is `source = "hf:<org>/<repo>"` with `revision` the
40-hex commit the build used, never a branch: a branch moves after the build
and hands a client other weights. `plowrt pack`, `push` and
`scripts/check_recipe.py` all refuse a recipe without it (a `refused` recipe
is exempt; it never packs). The published index carries that commit, and
`prepare` links exactly that snapshot into the bundle's checkpoint farm.

## Host requirements the distribution cannot supply

* `/dev/kfd` and `/dev/dri/renderD*` passed through, and the user in `render`
  (AMD).
* `ROCR_VISIBLE_DEVICES` for partial assignment — **not** `HIP_VISIBLE_DEVICES`:
  plowrt dlopens ROCr directly and never loads HIP.
* A driver ABI-compatible with the objects; `plowrt devices` prints the
  fingerprint and driver it resolved, which is the report to paste into a bug.
