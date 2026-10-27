# 27 — Content-addressed distribution

**Status: design (2026-10-27). Spike landed: `plowrt::dist::nar` serialises and hashes a tree
and reproduces the name `nix store add --dry-run` prints for it (unit test). Nothing else is
implemented; §12 is the plan.**

Replaces the mechanism `docs/DISTRIBUTION.md` describes: per-file digests in two hand-built
manifests, a zip through a registry's staged-upload API, generations numbered against a live
index, and a pin that `load` writes and nothing moves. What stays: the selection rule
(`plow_asset::dist::select`), the pairing check, the recipe as the only source of identity, the
Hugging Face snapshot as the only source of weights, and serving offline.

## Critique, before any phase past the spike

The design below is the Nix-faithful reading of the ask. Read this first; phase 2 (§12) is
gated on choosing between the full design and the hybrid at the end of this section.

1. **Identity and transport are separable, and the ask conflates them.** "Hash the whole
   directory" is an identity question; "a zip through an API" is a transport question. The
   NAR hash answers the first completely. For the second, today's per-file blob store is
   *finer-grained* than NAR-per-path: any unchanged file is reused, whereas a one-byte change
   to `bundle.json` (a `runtime_env` default) makes a new bundle path and re-moves everything
   in it. §2.3 recovers today's granularity only by splitting the bundle into parts along the
   lines that have changed so far (packet, objects, derived), which is a guess about the
   future. A **hybrid** — NAR hash as the name and the thing a signature covers, per-file
   blobs under `blobs/sha256/` as the transport, the narinfo listing the files — keeps
   `store.rs`, `fetch.rs` and the dedup, and drops only the manifest-digest chain and the
   generation counter. It is a much smaller migration and loses nothing the ask names except
   the literal `nar/<hash>.nar.zst` layout.
2. **The valuable parts do not need NAR at all.** Removing the registry API (bucket-only
   publishing with conditional PUT), profiles with generations and rollback, following the
   channel by default, and a gc with explicit roots can all be built on the existing blob
   store, keyed by `variant_id`. NAR adds: a name derived from bytes rather than a counter,
   stock-tool cross-checking, and a format other tooling can read. Those are real, but they
   are the smallest of the four gains and carry the whole migration cost.
3. **Following the channel by default moves risk to restart time.** A served process is
   stable (§5.5), but a `serve` *restart* after a cron `upgrade --all` loads a build the
   operator never validated on that host. The old frozen pin prevented exactly that. The
   design's answer is "pin production hosts", which is a policy a runbook has to carry; if
   production hosts will always be pinned, the default serves only lab boxes and the ask
   should be re-examined.
4. **Symlinks inside the served directory are a new contract for the loader.** Objects today
   are hard links inside the bundle; under §2.3 `hsaco/` is a symlink to another store path.
   Every loader path that opens objects, every `canonicalize`-then-`strip_prefix` check
   (`prepare.rs`, `store.rs`'s snapshot gc) and `no_build_host_paths` need auditing for
   symlinked directories, and the tree must be read-only, so nothing may write into it
   (`record_bundle`, `.plow-blobs`, `prepare`'s tokenizer link into the bundle all do today).
5. **The derived-sidecar farm (§2.4) is an exception to the design's own rule** ("the
   checkpoint is a symlink to the snapshot") that stays until the loader learns an override
   directory. A design that needs a loader change to be clean should say so up front: the
   loader change is small and should be in the plan, not deferred.
6. **Bucket-side garbage is not designed.** A push that dies between the narinfo and the
   index leaves a signed, reachable-by-nobody path; nothing collects it. Harmless, but it
   accumulates, and the same is true of superseded builds. A `plowrt push --gc` that walks
   every channel index is needed eventually.
7. **Compression cost is overstated for weights.** 406 MB → 57 MB was the packet.
   Safetensors do not compress; zstd level 19 over a 4.5 GB derived part or a `dist`
   checkpoint costs minutes per push for nothing. Compression must be per part
   (`Compression: none` is a valid narinfo value).
8. **Conditional PUT on multipart completion is unverified on R2.** `If-None-Match` on
   `PutObject` is documented; on `CompleteMultipartUpload` it is not. Phase 3 must test it
   against the real bucket; the fallback (HEAD, then complete) has a race that is harmless
   only because the key is content-named.
9. **The `/nix/store` literal in the name formula (§2.2)** buys a cross-check with stock
   tooling at the price of a string in every narinfo that is not a path plow uses. Readers
   will assume otherwise. If the cross-check is not valued, a plow-specific fingerprint
   (`plow:sha256:<hex>:<name>`) is cleaner and the spike's test changes by one line.
10. **A compiled-in trust key** means rotation needs a binary release. The `trusted-keys`
    file mitigates, but the first rotation is still a coordinated release.

**Recommendation.** Decide between (a) the full design below and (b) the hybrid of point 1
before phase 2. Points 2–3 argue for building profiles, channel-following, bucket-only push
and gc first, on whichever store wins, because they are the user-visible change; the NAR
identity can be introduced under either. The spike is unaffected by the choice: the hash and
the name are the same in both.

## 1. The problem

* **Identity is assembled, not measured.** A variant is `(label, generation)`, where the
  generation is a counter the publisher reads from the index and increments. Two publishers
  racing get a 409 and retry; a re-publish of the same bytes depends on `same_build` comparing
  manifests field by field; and `objset_id` must stay byte-compatible with a retired Python
  canonicalisation (`manifest.rs` header) because published ids are keyed by it.
* **The manifest, not the tree, is what is verified.** `pull` checks each file against the
  digest its manifest names and the manifest against the digest the index names. That is a
  chain of three documents, each hand-built by `manifest.rs`, each with a `deny_unknown_fields`
  schema the registry must match. A file the packer forgot (three quarters of a speech bundle,
  until `7374053d1`) publishes an immutable generation that is wrong.
* **The checkpoint is outside the identity.** `bundle.json` records `hf:<org>/<repo>` and a
  revision, but the farm `prepare` builds links the checkpoint into the bundle through an
  absolute symlink (`prepare.rs`: `link(&dir, &bundle_dir.join("checkpoint"))`), so the served
  directory is not reproducible from the published bytes, and a published generation with
  `revision: "main"` exists (`bin_dist.rs` test fixture `b49b5c9cf598`).
* **The pin is frozen, not followed.** `load` writes `refs/<digest>.json` and only `upgrade`
  moves it. The rule was written to stop a served model drifting, but it stops a *host* from
  following releases too, and a node that wants the newest compatible build has to script
  `upgrade --all`. There is no rollback: the previous bundle directory is gone the moment
  `rm` runs.
* **Two registries.** Bytes live in a bucket; numbering, catalog and upload staging live in an
  API (`POST …/api/v1/admin/registry/uploads`) a separate service runs. Publishing needs both
  to be up and agree.

Nix solved the same shape of problem: a store whose paths are named by the hash of their
contents, a binary cache that is a plain bucket of `<hash>.narinfo` + `nar/…`, profiles whose
generations are symlinks swapped atomically, a channel that resolves to the newest path, and a
garbage collector that walks references from roots. This design takes those four mechanisms
and nothing else from Nix: there is no `/nix/store`, no daemon, no derivations, no nix on a
serving host.

## 2. Store path identity

### 2.1 What is hashed

A bundle is a directory tree. Its identity is the **sha256 of its NAR serialisation**
(`nix-archive-1`, the format of the Nix thesis figure 5.2), which covers exactly what the
loader can observe:

| node | hashed | not hashed |
|---|---|---|
| regular file | bytes, executable bit (`mode & 0o111 != 0`) | mtime, owner, other mode bits, xattrs, hard-link count |
| directory | entries in **byte order** of their names, recursively | — |
| symlink | the **target string**, verbatim | what the target resolves to, whether it exists |

Byte order of names (not locale, not case-folded) and the symlink rule are what make the hash
stable across hosts and filesystems. The symlink rule is what makes the checkpoint reference
work (§2.4).

A NAR is streamed: `dist::nar::pack` writes it to any `Write`, and `nar_sha256` hashes it
without holding it, so a 4.5 GB derived sidecar costs one read.

### 2.2 The name

```
<hash>-<name>
hash = nixbase32( fold20( sha256( "source:sha256:" + hex(narhash) + ":/nix/store:" + name ) ) )
name = <namespace>-<model>-<label>[-<part>]        e.g. infervisor-kimi-k3-gfx942-mi325x-tp8-32k
```

This is Nix's own formula for `nix-store --add`, with the store directory fixed to the literal
`/nix/store`. The literal is not a path plow uses; it is kept so that stock tooling reproduces
the name: on any build host, `nix store add --dry-run --name <name> <dir>` prints
`/nix/store/<hash>-<name>` with the same hash `plowrt` computes, which gives an operator an
independent check with no plow binary (the spike test pins this: fixture →
`agwa6hp53s58zg7r3ghiwa2kzii1rs40-fixture`). Names are `[A-Za-z0-9+-._?=]{1,211}`, not starting
with `.`; namespace, model and label are already restricted to a subset of that
(`reference::segment_ok`).

The 32-character hash part is a fold of a second sha256, so the NAR hash travels separately in
the narinfo (§4) as `NarHash: sha256:<52 base32>`; verification after download compares the
NAR hash, and the path name is derived from it.

### 2.3 Parts: one bundle, several store paths

Today's objset dedup — an objset-only generation moves ~40 MB rather than ~400 MB — is kept by
making the parts of a build **separate store paths that the bundle references through
relative symlinks**, exactly as a Nix derivation output references its inputs:

```
$PLOW_HOME/store/
  <h1>-infervisor-kimi-k3-gfx942-mi325x-tp8-32k/        the bundle
    bundle.json  model.pkt  build.json  weights.json  plow_config.h  serve.json  …
    tokenizer.json         derived: a file   |   checkpoint-sourced: -> checkpoint/tokenizer.json
    config.json, tokenizer_config.json, generation_config.json, chat_template.jinja
                           -> checkpoint/<same>, when the snapshot has them
    hsaco       -> ../<h2>-infervisor-kimi-k3-gfx942-mi325x-tp8-32k-objects
    derived     -> ../<h3>-infervisor-kimi-k3-gfx942-mi325x-tp8-32k-derived     (optional)
    checkpoint  -> ../../snapshots/moonshotai--Kimi-K3@<40-hex commit>           (§2.4)
  <h2>-…-objects/          *.co / *.cubin, lowrung<max>/…, objects.json (arms, packet_hash, defines, env, toolchain)
  <h3>-…-derived/          model-idx-derived-*.safetensors
  <h4>-checkpoint-<ns>-<model>-<rev>/   a `dist` checkpoint: the weights themselves (§2.5)
```

A symlink hashes as its target string, so `<h1>` binds `<h2>` and `<h3>` — which bind their
contents — without containing them. A same-packet objset swap is a new `<h2>`, a new `<h1>`
(a few hundred KB of JSON and symlinks, plus the packet it shares nothing with… see below) and
nothing else moves. The pairing check (`ObjSet::pairs_with`) runs on `objects.json` at
`store add` and again at `pull`, as it does now.

The packet (`model.pkt`, hundreds of MB) sits in the bundle path, so a packet-unchanged objset
swap still re-transfers it. Two options, decided in phase 2 (§12) by measurement on the real
bundles: a fourth part `-packet` (one more symlink; `model.pkt -> ../<h5>-…-packet/model.pkt`),
or content-defined chunking of the NAR on the cache side. The first costs nothing and matches
today's granularity (packet, objset, derived), so it is the default.

### 2.4 The Hugging Face reference

`checkpoint` is a **relative symlink to the snapshot's commit-named directory**:

```
checkpoint -> ../../snapshots/<org>--<repo>@<commit>
```

* The revision is the repository's **40-hex commit**, never a branch. `weights::is_commit`
  already refuses anything else for `--fetch-weights`; `store add` refuses it for the symlink.
* The hash binds the string `../../snapshots/google--gemma-4-12b-it@842da379…`, not the
  weights. The bundle is reproducible — and verifiable against its narinfo — on a host with no
  checkpoint, and two bundles built against the same commit share the snapshot by construction
  (same target string).
* `bundle.json` repeats the reference as data (`checkpoint = {source = "hf:<org>/<repo>",
  revision = <commit>, shards, bytes}`) so `show` and `load --fetch-weights` do not have to
  read a symlink to know what to fetch. `store add` refuses a tree whose two copies disagree.
* `$PLOW_HOME/snapshots/<org>--<repo>@<commit>/` is the layout `weights::snapshot_path` and
  `--checkpoint` already use. An operator-supplied snapshot elsewhere is linked to that name,
  not the other way round — the store path never changes to suit a host.

**Derived sidecars.** The loader globs `*.safetensors` in one directory and takes the last
writer (`prepare.rs` header). A bundle with a derived sidecar therefore cannot point
`checkpoint` straight at the snapshot. For those bundles only, `load` builds the farm as
today, under a name that is a pure function of its inputs:

```
checkpoint -> ../../farms/<org>--<repo>@<commit>+<h3>        (h3 = the derived part's hash)
$PLOW_HOME/farms/<that name>/   symlinks to every snapshot shard, then the derived shards, sidecar files
```

The farm is host-built, rebuildable, and not a store path (it holds nothing). It is the one
piece of `prepare` that survives; `checkpoints/<key>@<rev>` goes. A later simplification is
for the loader to read `<assets>/derived/*.safetensors` as overrides after the checkpoint
directory, at which point every bundle points straight at the snapshot and `farms/` is
deleted; that is a loader change and is not part of this design's phases.

### 2.5 A checkpoint the distribution carries

Today `checkpoint.source = "dist"` lists the weights as blobs. Under this design the weights
are a store path of their own (`<h4>-checkpoint-<ns>-<model>-<rev>`, the files as the
snapshot would hold them) and the bundle's `checkpoint` symlink points at it
(`-> ../<h4>-…`). `pull` fetches it as part of the closure; nothing about `prepare` is special.
The `rev` in the name is the publisher's label; the hash is the identity, so a reused label
cannot alias two contents (what `weights::checkpoint_revision` guards today by hashing the
file list).

### 2.6 What is not in the tree

`status` (`validated` / `emits` / `refused`), `measured`, `released`, `description`: claims
*about* a build that change after it is built (a measurement arrives, a status is raised).
They live in the channel index (§4.3), keyed by store path, and updating them does not create
a new path. Today every such change mints a new generation of identical bytes.

The recipe path (`bundle.recipe`), `plow_git`, `target`, `parallel`, `max_ctx`, `features`,
`runtime_env`, `network`: part of the build's description, in `bundle.json`, in the hash.

## 3. The local store

```
$PLOW_HOME/                                        (--plow-home, PLOW_HOME, ~/.plow)
  store/<hash>-<name>/           immutable: chmod -R a-w after the NAR hash is verified
  store/.tmp/                    staging for unpack; renamed into place atomically
  narinfo/<hash>.narinfo         the record each store path arrived with (§4.2); the local "db"
  profiles/<ns>/<model>/
    generation-<n>  -> ../../../store/<hash>-<name>
    current         -> generation-<n>                  swapped by rename(2), never rewritten
    pin                                                present = pinned (§5.3)
  gcroots/auto/<pid>-<nonce> -> ../../store/<hash>-<name>     a running `serve` (§5.5)
  snapshots/<org>--<repo>@<commit>/                   Hugging Face checkpoints (unchanged)
  farms/<org>--<repo>@<commit>+<hash>/                §2.4, derived-sidecar bundles only
  channels/<registry>/<ns>/<model>/index.json         last fetched index, for `ls` and `show --offline`
```

`blobs/`, `refs/`, `bundles/`, `checkpoints/` and `.plow-blobs` go (phase 5).

A store path is written by unpacking the NAR into `store/.tmp/<hash>-<name>` while hashing the
decompressed stream, comparing with `NarHash`, making it read-only, and renaming into
`store/`. A path that exists is never written again: `pull` of a present path is a no-op
by name, and `store verify` re-hashes it on demand (`get`'s corruption check today, but
opt-in: re-hashing a 400 MB tree on every `ls` is what `bundle_bytes` was written to avoid).

Hard-link dedup across store paths (Nix's `optimise-store`) is not needed: the parts already
separate what changes from what does not.

## 4. The cache (bucket) layout

A Nix binary cache is a bucket of two object kinds plus a one-line descriptor. The same here,
with one channel index per model:

```
plow-cache-info                         StoreDir: /nix/store\nWantMassQuery: 1\nPriority: 40
<hash>.narinfo                          one per store path (§4.2)
nar/<narhash-base32>.nar.zst            the compressed NAR, named by the NAR hash
channels/<ns>/<model>/index.json        the model's variants → store paths (§4.3), signed
channels/<ns>/<model>/index.json.sig
runtime/index.json, runtime/<version>/<triple>/…      released binaries (unchanged)
catalog.json                            model names (unchanged)
```

Everything except `plow-cache-info`, the channel indexes and `catalog.json` is immutable and
content-named. `nar/` is keyed by NAR hash, not store hash, so two store paths with the same
content under different names share one object.

### 4.2 narinfo

Nix's text format, verbatim keys, so that the files are readable by anyone who has seen a
binary cache and so a future `nix copy --from` is not ruled out:

```
StorePath: /nix/store/<hash>-infervisor-kimi-k3-gfx942-mi325x-tp8-32k
URL: nar/<narhash-base32>.nar.zst
Compression: zstd
FileHash: sha256:<base32 of the .nar.zst>
FileSize: <bytes of the .nar.zst>
NarHash: sha256:<base32 of the NAR>
NarSize: <bytes of the NAR>
References: <hash2>-…-objects <hash3>-…-derived
Sig: infervisor-1:<base64 ed25519>
```

`References` is the closure `pull` fetches: every store path this tree's symlinks name
(§2.3). The snapshot symlink is not a reference — the snapshot is not a store path — and the
narinfo says nothing about it; `bundle.json` does.

### 4.3 The channel index

`plow.dist.model.v2`: today's `ModelIndex` with each variant naming a store path instead of a
manifest:

```json
{"schema": "plow.dist.model.v2", "namespace": "infervisor", "name": "kimi-k3",
 "hf": "moonshotai/Kimi-K3", "revision": "<commit>", "network": "kimi-k3",
 "variants": [
   {"store_path": "<hash>-infervisor-kimi-k3-gfx942-mi325x-tp8-32k",
    "nar_hash": "sha256:<base32>", "label": "gfx942-mi325x-tp8-32k", "seq": 3,
    "status": "validated", "target": {...}, "parallel": {...}, "max_ctx": 32768,
    "features": {...}, "build": {"plow_git": "...", "pairing_hash": "..."},
    "measured": {...}, "released": "2026-10-27T…", "supersedes": "<hash of seq 2>"}]}
```

`seq` is what `@g<n>` in a reference means: the position of the variant among its label's
publications, assigned by the publisher as `max(seq of label) + 1`, kept only so that a person
can say "the third build of that label". It is not part of any hash and is not what
`supersedes` or selection compares; selection ranks by `seq` where it ranked by `generation`.
The index is the only mutable object per model and is replaced with `If-Match` (§6).

`plow_asset::dist` keeps `Status`, `Target`, `Parallel`, `Measured`, `LiveTarget`,
`Constraints`, `Reject`, `assess`, `select` unchanged. `Variant` loses `variant_id`,
`generation`, `manifest`, `sha256`, `delta` and gains `store_path`, `nar_hash`, `seq`.
`Bundle` (`plow.dist.bundle.v2`) loses `files`, `generation`, `variant_id`, `objset.{manifest,
sha256}`, `checkpoint.files` and gains `parts: {objects, derived, packet, checkpoint}` as store
path names (the data form of the symlinks, so `pull` knows the closure before it has the
tree; the narinfo `References` must agree, checked at `store add`). `ObjSet` becomes
`objects.json` inside the objects part: `arms`, `packet_hash`, `defines`, `env`, `toolchain`,
`plow_git`, `script` stay (the loader's arm check reads them); per-object `sha256`/`bytes` go,
the tree hash covers them. `deny_unknown_fields` stays; v1 and v2 are told apart by `schema`,
and v1 parsing is deleted in phase 5.

## 5. Commands

### 5.1 Producer

```
plowrt store add --recipe recipes/infervisor/kimi-k3/gfx942-mi325x-tp8-32k.toml \
                 --assets <dir> [--objects <dir>] [--checkpoint-dir <dir>]
  → composes the parts (§2.3) into $PLOW_HOME/store, prints the bundle's store path
plowrt push <store path>|--recipe … --to r2://<bucket>|file:///mirror [--status validated --tok-s …]
  → uploads the closure's NARs and narinfos that are absent, signs, updates the channel index
```

`store add` replaces `pack`: it composes the tree from the recipe exactly as `manifest.rs`
does today (roles, object discovery, lowrung tiers, `no_build_host_paths`), writes
`bundle.json` v2 and `objects.json`, creates the symlinks, hashes each part, and refuses a tree
that references a build-host path or whose `checkpoint` revision is not a commit. The recipe
remains the only source of identity; `--plow-git` stays for the reproduce case. What
disappears is the zip, the manifest-digest chain and the generation counter. `pack
--manifests` becomes `store show <path>` (prints `bundle.json` and the narinfo).

`push` is idempotent by construction: an object named by its content is never overwritten
(§6); a second push of the same path transfers nothing and only re-asserts the index entry.
`status`/`measured` flags update the index entry for an existing path rather than minting a
generation.

### 5.2 Consumer: `load`, `upgrade`

```
plowrt load kimi-k3                       # resolve newest compatible → pull closure → new generation → current
plowrt load kimi-k3:gfx942-mi325x-tp8-32k@g3   # explicit; also pins (§5.3)
plowrt upgrade [kimi-k3 | --all]          # load, for profiles that are not pinned
```

`load` reads the channel index, runs `select` on this machine's `LiveTarget`, fetches the
closure (§7), writes `profiles/<ns>/<model>/generation-<n+1>` and swaps `current` with a
`symlink + rename`. A profile holds a reference string (`<registry>/<ns>/<model>`), as the pin
file does today, and **follows the channel**: every `load` or `upgrade` resolves afresh and
moves `current` to the newest compatible build. This is the change from "load writes the pin
it chose and nothing moves it". The served process does not move (§5.5), so the drift the
old rule prevented cannot happen; the new rule gives a host releases without a script.

`load` with a label and `@g<n>` — or with `--pin` — resolves exactly that variant and writes
`pin`. `upgrade` prints `pinned at …` for such a profile and leaves it unless `--unpin`.

### 5.3 `pin`, `unpin`, `rollback`

```
plowrt pin kimi-k3 [<store path> | :<label>@g<n>]     # default: whatever `current` points at
plowrt unpin kimi-k3
plowrt rollback kimi-k3 [--to <n>]                    # current -> generation-<n-1> (or <n>); pins it
plowrt ls [--generations]                             # per profile: current, pinned?, generations present
```

`rollback` pins, because a rolled-back profile that the next `upgrade --all` silently
re-advances is the failure rollback exists to prevent. `pin` writes the resolved store path
(never a label alone): a pin names bytes.

### 5.4 `gc`

Roots: every `profiles/*/*/generation-*` symlink, every `gcroots/auto/*` whose pid is alive
(dead ones are removed), and the transitive `References` of each root from `narinfo/`. Every
store path not reached is deleted (and its narinfo); every `snapshots/` and `farms/` entry no
reachable bundle's `checkpoint` names is reported, and deleted with `--snapshots`. `gc
--delete-older-than <days>` and `gc --keep-generations <n>` drop old generations of each
profile first, never `current` and never a pinned one. A path with no narinfo is treated as
reachable-unknown and left, with a warning, as `gc` does today for a bundle without a record.

### 5.5 Serve resolves at load, not during serve

`serve --model kimi-k3` calls `resolve_local`, which reads `profiles/<ns>/<model>/current`,
canonicalises it to `$PLOW_HOME/store/<hash>-<name>`, writes `gcroots/auto/<pid>-<nonce>` to
that path, and hands the loader the canonical path. From then on the process holds an
immutable directory by its content name; a `load`, `upgrade` or `rollback` in another process
swaps `current` and the server neither sees nor follows it. `/v1/models/load` resolves the same
way, once, at the request. The root is removed on clean exit and ignored by `gc` once the pid
is gone. `serve` opens nothing under `channels/` and no network — the invariant of
`docs/DISTRIBUTION.md` "Serving is offline" holds unchanged and CI's `serve is offline` tree
check is unchanged (§10).

## 6. Upload protocol

The bucket is Cloudflare R2 through its S3 API (endpoint
`https://<account>.r2.cloudflarestorage.com`, region `auto`, SigV4). Reads are plain HTTPS
GETs against the bucket's public hostname; only `push` signs requests.

For each store path in the closure, parts first (so an interrupted push never leaves a bundle
whose references are missing), then the bundle, then the index:

1. `HEAD <hash>.narinfo` → present: skip the path entirely.
2. Stream `nar::pack` → zstd (level 19, a one-time cost per path) → `PUT nar/<narhash>.nar.zst`
   with `If-None-Match: *`; a 412 means another publisher put identical bytes first and is
   success. Over 64 MiB the object goes as a multipart upload (64 MiB parts, each retried on
   its own); the multipart completion is preceded by a `HEAD`, and a race there writes identical
   bytes under a content name, which is harmless.
3. `PUT <hash>.narinfo` with `If-None-Match: *`, after the NAR is complete (publication order:
   bytes, then the record that names them — today's "blobs, then manifests, then the index").
4. `GET channels/<ns>/<model>/index.json` (ETag) → merge the entry → `PUT` with `If-Match:
   <etag>`; a 412 re-reads and merges again. This optimistic loop replaces the API's 409.
   The index is signed (§8) and its `.sig` is written after it.

Credentials: an R2 API token scoped to the bucket, from `PLOW_R2_ACCESS_KEY_ID` /
`PLOW_R2_SECRET_ACCESS_KEY` (never flags: `ps` shows flags, which is why `--key` was already
discouraged). `--to file:///mirror` writes the same tree to a directory, which is what tests use
and what an air-gapped mirror is: a bucket is exactly a directory tree, so `rsync` of one is a
complete registry, as today.

## 7. Download protocol

```
GET <base>/<hash>.narinfo               verify Sig against trusted keys; parse References
for each reference not in store/: recurse
GET <base>/nar/<narhash>.nar.zst        Range: bytes=<len of .part>-  when a partial exists
  → store/.tmp/<hash>.nar.zst.part      compared with FileHash/FileSize when complete
  → zstd decode → NAR hash as it unpacks into store/.tmp/<hash>-<name>
  → NarHash matches → chmod -R a-w → rename into store/ → narinfo/<hash>.narinfo written
```

Resumable because every object is immutable: a `Range` continuation of a content-named object
cannot splice two versions. The `Fetch` trait gains `open_at(path, offset)`; `FileFetch`
seeks, `HttpFetch` sends `Range`. The present `get`/`open` stay for small objects. Transfer
accounting (`Transfer`, `plan`) becomes per store path: "3 paths, 41.0 MiB to fetch, 2 paths
already local" — the honest number is still derived from what is absent, now at part
granularity.

Air-gapped: `--registry file:///mirror` as today; `plowrt push --to file://` produces the
mirror.

## 8. Signing

Nix's scheme, so that the cache can be checked by stock tooling and the format has a known
threat model:

* The signature is ed25519 over the fingerprint
  `1;/nix/store/<hash>-<name>;sha256:<narhash base32>;<narsize>;<comma-separated /nix/store/<ref> …>`
  — exactly Nix's string, store directory literal included (§2.2). `Sig: <keyname>:<base64>`;
  several `Sig` lines are allowed, which is how a key is rotated.
* The channel index is signed detached: `index.json.sig` holds `<keyname>:<base64>` over the
  index bytes. An unsigned or wrongly signed index is refused, because the index is what says
  "this path is a validated Kimi-K3 build for MI325X"; a narinfo signature alone proves only
  that the bytes are the bytes.
* Trust: `plowrt` carries the current public key (`infervisor-1:<base64>`) compiled in; the
  operator adds keys with `--trusted-key` / `$PLOW_HOME/trusted-keys` (one `name:base64` per
  line), which is also how a third-party namespace is trusted. `pull --no-check-sigs` exists
  for a mirror an operator built themselves and prints a warning on every path.
* Key custody is an open question (§11): who holds `infervisor-1`'s secret, whether CI signs
  or a release step on an operator's machine signs, and where a second key lives for rotation.
  `PLOW_SIGNING_KEY=<path>` names the secret key file (`<name>:<base64 64-byte key>`, the
  `nix-store --generate-binary-cache-key` layout) for `push`; a push without one is refused
  unless `--to file://`.

## 9. The recipe → store path

The recipe is unchanged. `store add --recipe` reads the same `[target]`, `[checkpoint]`,
`plow_git`, `label`, `network`, `[serve] env` it reads now and composes the tree; the store path
is the hash of that tree. Two consequences:

* **Reproducibility is of the tree, not of the recipe.** A re-emit of the same recipe at the
  same `plow_git` produces the same path only if every byte is the same; `nvcc` is not
  SASS-reproducible for some objects, so a re-emit is usually a new objects part and a new
  bundle path with the same packet part. That is content identity working as intended: two
  paths are the same bytes if and only if they have the same name, and the channel index says
  which of several same-recipe builds is the qualified one.
* **The checkpoint must be a commit.** `[checkpoint] source = "hf:<org>/<repo>"` and
  `revision = <40 hex>` become required for an `hf:` checkpoint. Of the 44 production recipes
  today, 17 satisfy that. The other 27: 15 name an `hf:` repository with no `revision`
  (DeepSeek-V4/V4.1, GLM-5.3 and -Flash, Kimi-K3, Qwen3.5-GDN, both Qwen3.8-27B, and the
  sm89 rows of Nemotron, Orpheus and Qwen3-ASR-0.6B); 6 name a bare `<org>/<repo>` with no
  prefix and no revision (the `xeon6-amx-bf16` rows and Gemma-4 26B BF16); 2 carry the commit
  inside the source string (`google/gemma-4-31B-it@842da37…`, `RedHatAI/…@d4ab4f5…`), which
  `store add` can accept by normalising; 2 name a locally re-keyed checkpoint that is no
  repository (`gemma-4-12b-it-fp8`, `gemma-4-26b-a4b-it-fp8`); one is `local:…`, one
  `pip:silero-vad==5.1.2`. Each either gains its commit or becomes a `dist` checkpoint part
  (§2.5); the re-keyed ones can only be the latter. This is the largest contradiction between
  the existing recipes and the ask (§11).

`check_recipe.py --bundle` keeps asserting the recipe against `build.json` /
`build_defines.json`; the `[artifacts]` digest check is replaced by comparing the recipe's
expected store path (recorded in the `.md` beside it when qualified) with `store add`'s.

## 10. Serve is offline — unchanged

`serve` reads `profiles/` and `store/`; `nar`, `narinfo`, `push` and `pull` are `dist`-only;
the S3 client (§13) is behind a new `push` feature that implies `dist`, so a consumer-only CLI
(`load` on a customer host) links no SigV4 or XML code. The CI step asserting that `cargo tree
-p plowrt --no-default-features --features cuda,hsa,cpu` contains none of `ureq rustls
hyper-tls native-tls reqwest openssl` stays green through every phase: the spike adds no
dependency at all (`sha2` is already unconditional), and every later crate is optional under
`dist` or `push`. A second assertion is added in phase 3: the `dist` tree without `push`
contains no `rusty-s3`.

## 11. Open questions

* **Bucket and account.** One bucket (`plow-cache`?) with a public custom hostname
  (`dist.infervisor.ai` is the compiled-in default registry today) and the S3 endpoint for
  push; or one per environment. Who provisions the R2 API token, and does CI get one.
* **Signing key custody.** Generated where, stored where, and whether the qualified-build push
  is done by a person or by CI. Rotation needs a second key name in the binary before the
  first is retired.
* **Retiring the registry API.** After every published model is re-pushed (§12 phase 5), the
  staged-upload API (`/api/v1/admin/registry/uploads`), generation numbering and the
  `catalog.json` writer have no caller. What the API still serves that the bucket does not:
  `description` (now in the index entry) and `v1/runtime/index.json` (written by the release
  step from `nix build .#plowrt-release`, which can write the bucket directly). The service's
  owners decide the shutdown date; this design only removes plow's dependence on it.
* **Recipes without a commit** (§9): 27 of 44 recipes have to be resolved per model before
  that model can be pushed; the two re-keyed FP8 checkpoints need a `dist` part each, which
  puts those weights in the bucket.
* **`ls --json` consumers.** A node's check-in reads `variant_id` and `generation`; they
  become `store_path` and `seq`. Whether anything parses the old fields outside this repo is
  unknown.
* **Packet part vs. chunking** (§2.3): decided by measuring the real upgrade transfer on the
  Kimi-K3 and Gemma-4 bundles in phase 2.

## 12. Phases

Each phase is one PR; each keeps the `serve is offline` tree assertion green and runs
`cargo test -p plowrt --no-default-features --features cpu,dist --lib dist::` and
`cargo test -p plow-asset`.

| # | PR | contents | tests |
|---|---|---|---|
| 1 | this branch | this document; `dist::nar` (NAR writer, NAR hash, nix base32, store path name) | fixture hash = `nix hash path`, store path = `nix store add --dry-run`; symlink target vs. contents; name validation |
| 2 | local store | `store add --recipe` composes the parts from `manifest.rs`'s discovery (roles, objects, lowrung, checkpoint symlink, refusal of non-commit revisions and build-host paths); `plow_asset::dist` v2 `Bundle`/`Variant`/`objects.json`; `store/`, `narinfo/` local records; `store ls`, `store show`, `store verify`; read-only trees. Measures the packet-part question on real bundles. | compose a fixture assets dir → expected store path; v2 schema goldens (`deny_unknown_fields`); pairing refused at `store add`; a `revision = "main"` recipe refused |
| 3 | cache + push | narinfo writer/parser (Nix text format), ed25519 signing and verification (`ed25519-dalek` 2, feature `dist`), zstd NAR stream, `Fetch::open_at`, `push --to file://` (complete), `push --to r2://` (`rusty-s3` + `ureq`, feature `push`), index merge with `If-Match` retry, channel index signing | file:// round trip; second push transfers nothing; a tampered narinfo and an unsigned index are refused; `If-Match` retry against a fake that returns 412 once; CI: `dist` tree without `push` has no `rusty-s3` |
| 4 | consumer | `pull`/`load` from narinfo + index v2 (file:// and https://), resumable ranged download, closure fetch, `profiles/` generations and the `current` swap, `pin`/`unpin`/`rollback`, `gc` with roots and `References`, `resolve_local` on profiles, `serve` writes its gcroot | generations and rollback; a resumed `.part` verifies; `gc` keeps a running server's path after `upgrade` swapped `current`; a pinned profile survives `upgrade --all`; selection on `seq` |
| 5 | migration | converter from a `v1/` manifest triple to store paths (`seq` = old generation), re-push of every published model, removal of `pack`, `publish`, the zip and the objset-id port in `manifest.rs`, `refs/`, `bundles/`, `blobs/`, `.plow-blobs`, v1 schema parsing; `docs/DISTRIBUTION.md` rewritten to this layout; `zip` leaves the lock | converter fixture = a known published variant; `ls --json` prints `store_path`/`seq`; the old commands are gone from `--help` |

Phase 2 and 3 do not touch `pull`/`load`, so a host on today's layout is unaffected until
phase 4 ships both readers; phase 5 removes the old reader only after the re-push.

## 13. Crates evaluated

Measured 2026-10-27 with one scratch package per crate outside the repo:
`cargo tree -e normal --prefix none | sort -u | wc -l` (the count includes the scratch package
itself). "New to lock" is against this workspace's `Cargo.lock`, which already carries `sha2`,
`url`, `idna`, `jiff` 0.2.28, `percent-encoding`, `base64`, `zeroize`, `data-encoding`,
`thiserror`, `zstd`, `ureq`.

### NAR

| crate | version | tree | new to lock | verdict |
|---|---|---|---|---|
| self-contained writer (`dist::nar`) | — | 0 | none (`sha2` is unconditional) | **chosen**: ~120 lines, streams, byte-order names, non-UTF-8 names, no fixed buffer; proven against nix 2.35.2 |
| `nix-nar` | 0.6.0 | 22 (with `nix-base32` + `sha2`) | `camino`, `symlink` (+ `is_executable` off unix) | sound encoder, but paths are UTF-8 (`camino`), it sizes an internal buffer by the longest file name and asserts on it, and it decodes too (unneeded) |
| `nix-compat` (snix / tvix) | crates.io `0.0.0-pre` placeholder; real crate git-only | not measurable from crates.io; defaults pull `tokio`, `nix-compat-derive` (path dep in a monorepo) | — | the most complete (NAR, narinfo, signatures, store paths), but a git dependency on a monorepo with a path-dependent proc-macro crate; unsuitable for a workspace that pins crates.io |
| `nar` | 0.0.8 | — | — | unrelated: a language tool (`pest`, `rustyline`); name collision only |
| `nix-base32` | 0.2.0 | 1 | none | correct and dependency-free, but the encoder is 15 lines and is inlined instead |

### S3 / R2

| crate | version | tree | new to lock | HTTP/TLS it brings | verdict |
|---|---|---|---|---|---|
| `rusty-s3` (no default features) | 0.11.0 | 49 | `rusty-s3`, `jiff-core` (bump of `jiff`) | none — sans-IO: builds signed requests, the caller sends them | **chosen**, with `rustcrypto` (+`hmac`, `md-5`) and `xml` (+`instant-xml`, `xmlparser`): 75 crates, ~7 new; requests go through the `ureq` agent `dist` already has |
| `object_store` (`aws`) | 0.14.2 | 169 | most of it | `reqwest`, `hyper`, `tokio`, `rustls`, `aws-lc-rs` (C build) | async runtime and a second HTTP stack for one PUT path; `aws-lc-sys` compiles C under nix for no gain |
| `aws-sdk-s3` + `aws-config` | 1.155.0 | 268 | most of it | `hyper`, `tokio`, `rustls`, `aws-lc-rs`, `ring` | the official client; largest by far, same objections, and its R2 use needs `region = auto` plus endpoint overrides either way |

R2 specifics the chosen client must cover and `rusty-s3` does: path-style or virtual-hosted
addressing against a custom endpoint, SigV4 with region `auto`, multipart upload
(`CreateMultipartUpload` / `UploadPart` / `CompleteMultipartUpload`), and passing arbitrary
headers (`If-None-Match`, `If-Match`), which R2 honours on `PutObject`.

### Signing

| crate | version | tree | new to lock | verdict |
|---|---|---|---|---|
| `ed25519-dalek` (`std`, no default) | 2.x | 24 | `curve25519-dalek` (+derive), `ed25519`, `signature`, `subtle` | **chosen**; the reference implementation of the scheme Nix uses |
| `ed25519-dalek` | 3.0.0 | — | — | published, but its dependency set did not resolve against crates.io on the measurement date; pin 2 |
| `ring` | — | — | C/asm build | would come free with `aws-lc`/`rustls` stacks rejected above; not otherwise justified |
