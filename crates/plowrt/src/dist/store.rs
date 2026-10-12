//! The local content-addressed store.
//!
//! ```text
//! ~/.plow/
//!   blobs/sha256/<hex>                                   every file, content-addressed
//!   refs/<digest>.json                                   {ref, variant} for one pin
//!   bundles/<variant_id>/                                links into blobs/ — what --assets gets
//! ```
//!
//! Content addressing is what makes an objset-only upgrade cheap: a new
//! generation that changes only the code objects shares the packet, the
//! manifests and the derived sidecar with its predecessor, so only the objects
//! move. It is also what lets two models on one GPU share a single objset.
//!
//! Writes are staged to a temporary file and renamed, because `cp` onto a live
//! path truncates it — the same hazard `scripts/install_hsaco.sh` exists for: a
//! reader mid-load gets a short read and a bogus code object.

use std::path::{Path, PathBuf};

use crate::{Result, RuntimeError};

/// A sha256 written as 64 lowercase hex characters.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest(String);

impl Digest {
    pub fn parse(s: &str) -> Result<Self> {
        if s.len() != 64
            || !s
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(RuntimeError::Dist(format!(
                "`{s}` is not a sha256 digest (64 lowercase hex characters)"
            )));
        }
        Ok(Digest(s.to_string()))
    }

    pub fn of(bytes: &[u8]) -> Self {
        Digest(plow_asset::decode_objects::image_sha256(bytes))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Written into each materialized bundle: the digests it links, one per line.
/// `gc` reads this instead of re-hashing the bundle.
const BUNDLE_BLOBS: &str = ".plow-blobs";

pub use plow_asset::dist::BUNDLE_MANIFEST;

/// One pin, as `plowrt ls --json` prints it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Pinned {
    /// `<namespace>/<name>:<label>@g<generation>`.
    #[serde(rename = "ref")]
    pub reference: String,
    pub namespace: String,
    pub name: String,
    pub label: String,
    pub generation: u32,
    pub variant_id: String,
    /// Every blob the bundle links, files and objects.
    pub bytes: u64,
    /// The materialized bundle: what `serve --assets` and `/v1/models/load` take.
    pub dir: PathBuf,
    /// The checkpoint farm is built and linked in, so the bundle serves as is.
    pub prepared: bool,
}

pub struct Store {
    root: PathBuf,
}

impl Store {
    /// `--plow-home` / `PLOW_HOME`, else `$HOME/.plow`.
    pub fn open_default() -> Result<Self> {
        let root = match crate::config::RuntimeConfig::get().plow_home.clone() {
            Some(p) => PathBuf::from(p),
            None => {
                let home = std::env::var_os("HOME").ok_or_else(|| {
                    RuntimeError::Dist(
                        "neither --plow-home nor HOME is set; pass an explicit store path".into(),
                    )
                })?;
                PathBuf::from(home).join(".plow")
            }
        };
        Self::open(root)
    }

    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        for sub in ["blobs/sha256", "manifests", "refs", "bundles"] {
            let p = root.join(sub);
            std::fs::create_dir_all(&p).map_err(|source| RuntimeError::Io {
                path: p.clone(),
                source,
            })?;
        }
        Ok(Store { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blob_path(&self, d: &Digest) -> PathBuf {
        self.root.join("blobs/sha256").join(d.as_str())
    }

    pub fn has(&self, d: &Digest) -> bool {
        self.blob_path(d).is_file()
    }

    /// Store `bytes`, verifying they hash to `want`.
    ///
    /// Returns `Ok(false)` when the blob was already present — the caller can
    /// report an upgrade's real cost by counting what it actually had to fetch.
    pub fn put(&self, want: &Digest, bytes: &[u8]) -> Result<bool> {
        let got = Digest::of(bytes);
        if &got != want {
            return Err(RuntimeError::Dist(format!(
                "digest mismatch: expected {want}, content hashes to {got} ({} bytes). \
                 The blob was corrupted in transit or the manifest is wrong; nothing was stored.",
                bytes.len()
            )));
        }
        let final_path = self.blob_path(want);
        if final_path.is_file() {
            return Ok(false);
        }
        // Stage-and-rename: a reader must never observe a partial blob under a
        // name that promises its full content.
        let tmp = final_path.with_extension(format!("part{}", std::process::id()));
        std::fs::write(&tmp, bytes).map_err(|source| RuntimeError::Io {
            path: tmp.clone(),
            source,
        })?;
        std::fs::rename(&tmp, &final_path).map_err(|source| RuntimeError::Io {
            path: final_path.clone(),
            source,
        })?;
        Ok(true)
    }

    /// [`Store::put`] for a stream: hashed as it is written to a staged file,
    /// renamed into place only when the digest matches.
    pub fn put_reader(&self, want: &Digest, r: &mut dyn std::io::Read) -> Result<bool> {
        use sha2::Digest as _;
        let final_path = self.blob_path(want);
        if final_path.is_file() {
            return Ok(false);
        }
        let tmp = final_path.with_extension(format!("part{}", std::process::id()));
        let io = |path: &Path| {
            let path = path.to_path_buf();
            move |source| RuntimeError::Io { path, source }
        };
        let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp).map_err(io(&tmp))?);
        let (mut h, mut n, mut buf) = (sha2::Sha256::new(), 0u64, vec![0u8; 1 << 20]);
        let copied = (|| -> std::io::Result<()> {
            loop {
                let k = r.read(&mut buf)?;
                if k == 0 {
                    return std::io::Write::flush(&mut out);
                }
                h.update(&buf[..k]);
                std::io::Write::write_all(&mut out, &buf[..k])?;
                n += k as u64;
            }
        })();
        drop(out);
        let got = format!("{:x}", h.finalize());
        if let Err(source) = copied {
            let _ = std::fs::remove_file(&tmp);
            return Err(RuntimeError::Dist(format!(
                "blob {want}: transfer failed: {source}"
            )));
        }
        if got != want.as_str() {
            let _ = std::fs::remove_file(&tmp);
            return Err(RuntimeError::Dist(format!(
                "digest mismatch: expected {want}, content hashes to {got} ({n} bytes). \
                 The blob was corrupted in transit or the manifest is wrong; nothing was stored."
            )));
        }
        std::fs::rename(&tmp, &final_path).map_err(io(&final_path))?;
        Ok(true)
    }

    pub fn get(&self, d: &Digest) -> Result<Vec<u8>> {
        let p = self.blob_path(d);
        let bytes = std::fs::read(&p).map_err(|source| RuntimeError::Io {
            path: p.clone(),
            source,
        })?;
        // A blob is named by its content, so a mismatch here is on-disk
        // corruption rather than a bad download, and silently serving it would
        // reproduce exactly the class of failure this store exists to stop.
        let got = Digest::of(&bytes);
        if &got != d {
            return Err(RuntimeError::Dist(format!(
                "stored blob {d} hashes to {got} — the store is corrupt; remove {} and re-pull",
                p.display()
            )));
        }
        Ok(bytes)
    }

    /// Materialize a bundle directory: one link per file, named as the runtime
    /// expects to find it.
    ///
    /// Hard links rather than copies, so a 380 MB packet shared by two variants
    /// costs one copy. Falls back to a real copy across filesystems.
    pub fn materialize(&self, variant_id: &str, files: &[(String, Digest)]) -> Result<PathBuf> {
        let dir = self.root.join("bundles").join(variant_id);
        std::fs::create_dir_all(&dir).map_err(|source| RuntimeError::Io {
            path: dir.clone(),
            source,
        })?;
        for (name, digest) in files {
            if !self.has(digest) {
                return Err(RuntimeError::Dist(format!(
                    "{name}: blob {digest} is not in the store — run `plowrt pull` first"
                )));
            }
            // A manifest is fetched over the network, so its file names are
            // untrusted input: a `..` or an absolute path would write outside
            // the bundle. Objects legitimately sit under `hsaco/`, so nested
            // paths are allowed, but only downward.
            if !is_safe_relative(name) {
                return Err(RuntimeError::Dist(format!(
                    "{name}: a bundle file name must be a relative path with no `..` component"
                )));
            }
            let dst = dir.join(name);
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent).map_err(|source| RuntimeError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
            if dst.exists() {
                std::fs::remove_file(&dst).map_err(|source| RuntimeError::Io {
                    path: dst.clone(),
                    source,
                })?;
            }
            let src = self.blob_path(digest);
            if std::fs::hard_link(&src, &dst).is_err() {
                std::fs::copy(&src, &dst).map_err(|source| RuntimeError::Io {
                    path: dst.clone(),
                    source,
                })?;
            }
        }
        // What this bundle references, so `gc` can compute reachability without
        // re-reading the bundle. Re-hashing every file would be O(total bytes) —
        // tens of GB for a K3 bundle — and walking the directory would have to
        // recurse into `hsaco/` to see the objects at all.
        let manifest: String = files.iter().map(|(_, d)| format!("{d}\n")).collect();
        let mpath = dir.join(BUNDLE_BLOBS);
        std::fs::write(&mpath, manifest).map_err(|source| RuntimeError::Io {
            path: mpath,
            source,
        })?;
        Ok(dir)
    }

    /// Add blobs to a materialized bundle's record without linking them into
    /// it. A `dist` checkpoint's files are linked into its snapshot instead,
    /// and `gc` reads only bundle records, so without this it would collect
    /// the weights of a bundle that still needs them.
    pub fn keep(&self, dir: &Path, digests: &[Digest]) -> Result<()> {
        use std::io::Write;
        let mpath = dir.join(BUNDLE_BLOBS);
        let lines: String = digests.iter().map(|d| format!("{d}\n")).collect();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&mpath)
            .and_then(|mut f| f.write_all(lines.as_bytes()))
            .map_err(|source| RuntimeError::Io {
                path: mpath,
                source,
            })
    }

    /// A pin's filename: a digest of the reference, not the reference itself.
    ///
    /// A reference contains `://` and `/`, and using it as a path both nests
    /// unpredictably and does not survive the round trip — `PathBuf::join`
    /// collapses `file:///store` to `file:/store`, so a listed pin could not be
    /// re-parsed and `upgrade --all` would fail on exactly the mirrors that
    /// motivate a local registry.
    fn pin_file(&self, reference: &str) -> PathBuf {
        let key = &plow_asset::decode_objects::image_sha256(reference.as_bytes())[..16];
        self.root.join("refs").join(format!("{key}.json"))
    }

    /// Record which variant a reference currently resolves to.
    ///
    /// A pin is written by `load` and moved only by `upgrade`: a served model
    /// must not drift under a running server because a catalog refreshed.
    pub fn pin(&self, reference: &str, variant_id: &str) -> Result<()> {
        let p = self.pin_file(reference);
        let body = serde_json::json!({ "ref": reference, "variant": variant_id });
        // Not `unwrap_or_default`: an empty pin file is an unparseable pin, and
        // silently writing one would lose the record rather than report it.
        let bytes = serde_json::to_vec(&body).map_err(|source| RuntimeError::Json {
            path: p.clone(),
            source,
        })?;
        std::fs::write(&p, bytes).map_err(|source| RuntimeError::Io {
            path: p.clone(),
            source,
        })
    }

    pub fn pinned(&self, reference: &str) -> Option<String> {
        let raw = std::fs::read(self.pin_file(reference)).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&raw).ok()?;
        v.get("variant")?.as_str().map(str::to_string)
    }

    pub fn unpin(&self, reference: &str) -> Result<()> {
        let p = self.pin_file(reference);
        match std::fs::remove_file(&p) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(source) => Err(RuntimeError::Io { path: p, source }),
        }
    }

    /// Every pin, as `(reference, variant_id)`.
    ///
    /// The reference comes back exactly as it was written, so a caller can
    /// re-parse it — which is what `upgrade --all` does.
    pub fn pins(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(self.root.join("refs")) {
            for e in rd.flatten() {
                let Ok(raw) = std::fs::read(e.path()) else {
                    continue;
                };
                let Ok(v) = serde_json::from_slice::<serde_json::Value>(&raw) else {
                    continue;
                };
                if let (Some(r), Some(id)) = (
                    v.get("ref").and_then(|x| x.as_str()),
                    v.get("variant").and_then(|x| x.as_str()),
                ) {
                    out.push((r.to_string(), id.to_string()));
                }
            }
        }
        out.sort();
        out
    }

    /// Write a pulled bundle's manifest beside its files, as `bundle.json` —
    /// the name a `plowc` assets directory already carries it under.
    ///
    /// A pin records only the variant id, which is a hash; the label and
    /// generation a person (or a node's check-in) names the model by live
    /// only in the manifest. Staged and renamed, because `dir` is full of hard
    /// links into `blobs/` and writing through one would rewrite a blob.
    pub fn record_bundle(&self, dir: &Path, bundle: &plow_asset::dist::Bundle) -> Result<()> {
        let path = dir.join(BUNDLE_MANIFEST);
        let bytes = serde_json::to_vec_pretty(bundle).map_err(|source| RuntimeError::Json {
            path: path.clone(),
            source,
        })?;
        let tmp = path.with_extension(format!("part{}", std::process::id()));
        std::fs::write(&tmp, bytes).map_err(|source| RuntimeError::Io {
            path: tmp.clone(),
            source,
        })?;
        std::fs::rename(&tmp, &path).map_err(|source| RuntimeError::Io { path, source })
    }

    /// Every pin with what a caller needs to serve it: `plowrt ls --json`.
    ///
    /// A pin whose bundle directory or manifest is missing is skipped with a
    /// warning rather than reported half-known: it cannot be served as it is,
    /// and `plowrt load` re-materializes it.
    pub fn list(&self) -> Vec<Pinned> {
        let mut out = Vec::new();
        for (reference, variant_id) in self.pins() {
            match self.pinned_bundle(&variant_id) {
                Ok(p) => out.push(p),
                Err(e) => tracing::warn!(%reference, %variant_id, "skipped: {e}"),
            }
        }
        out
    }

    fn pinned_bundle(&self, variant_id: &str) -> Result<Pinned> {
        let dir = self.root.join("bundles").join(variant_id);
        let path = dir.join(BUNDLE_MANIFEST);
        let raw = std::fs::read(&path).map_err(|source| RuntimeError::Io {
            path: path.clone(),
            source,
        })?;
        let b: plow_asset::dist::Bundle =
            serde_json::from_slice(&raw).map_err(|source| RuntimeError::Json {
                path: path.clone(),
                source,
            })?;
        let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
        Ok(Pinned {
            reference: format!("{}/{}:{}@g{}", b.namespace, b.name, b.label, b.generation),
            bytes: self.bundle_bytes(&dir),
            // `prepare::build` links the farm in as `checkpoint` last, so its
            // presence is what "ready to serve" means.
            prepared: dir.join("checkpoint").is_dir(),
            namespace: b.namespace,
            name: b.name,
            label: b.label,
            generation: b.generation,
            variant_id: variant_id.to_string(),
            dir,
        })
    }

    /// The bundle's size in the store: every blob its record names, files and
    /// objects alike. Read from the record rather than by walking the
    /// directory, which would also follow `checkpoint` into the weights.
    fn bundle_bytes(&self, dir: &Path) -> u64 {
        // `keep` appends a `dist` checkpoint's digests to the record: weights,
        // not the build (an `hf:` bundle's are not counted either). A digest
        // the record lists more often than the checkpoint does is also a build
        // file or object, and counts once.
        let mut count: std::collections::BTreeMap<String, i64> = Default::default();
        for l in std::fs::read_to_string(dir.join(BUNDLE_BLOBS))
            .unwrap_or_default()
            .lines()
        {
            *count.entry(l.trim().to_string()).or_default() += 1;
        }
        if let Some(b) = std::fs::read(dir.join(BUNDLE_MANIFEST))
            .ok()
            .and_then(|raw| serde_json::from_slice::<plow_asset::dist::Bundle>(&raw).ok())
        {
            for f in &b.checkpoint.files {
                if let Some(n) = count.get_mut(&f.sha256) {
                    *n -= 1;
                }
            }
        }
        count
            .into_iter()
            .filter(|(_, n)| *n > 0)
            .filter_map(|(l, _)| Digest::parse(&l).ok())
            .filter_map(|d| std::fs::metadata(self.blob_path(&d)).ok())
            .map(|m| m.len())
            .sum()
    }

    /// Remove blobs no live bundle directory references.
    ///
    /// Reachability is computed from the materialized bundles rather than from
    /// manifests, so a blob stays exactly as long as something can still open
    /// it. Returns the number of blobs removed and the bytes reclaimed.
    pub fn gc(&self) -> Result<(usize, u64)> {
        let mut live = std::collections::BTreeSet::new();
        let bundles = self.root.join("bundles");
        if let Ok(rd) = std::fs::read_dir(&bundles) {
            for bundle in rd.flatten() {
                if !bundle.path().is_dir() {
                    continue;
                }
                let record = bundle.path().join(BUNDLE_BLOBS);
                let Ok(list) = std::fs::read_to_string(&record) else {
                    // A bundle whose record is missing is opaque: refusing to
                    // collect at all is the only safe answer, because deleting a
                    // blob it still needs would leave it unservable.
                    return Err(RuntimeError::Dist(format!(
                        "{}: no {BUNDLE_BLOBS} record, so what it references is unknown and \
                         nothing can be collected safely. Re-materialize it with `plowrt pull`, \
                         or remove the directory.",
                        bundle.path().display()
                    )));
                };
                live.extend(
                    list.lines()
                        .map(str::trim)
                        .filter(|l| !l.is_empty())
                        .map(String::from),
                );
            }
        }
        let (mut n, mut freed) = (0usize, 0u64);
        let blobs = self.root.join("blobs/sha256");
        let rd = std::fs::read_dir(&blobs).map_err(|source| RuntimeError::Io {
            path: blobs.clone(),
            source,
        })?;
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if live.contains(&name) {
                continue;
            }
            // A blob still linked from a `dist` snapshot frees its space only
            // when the snapshot goes; `gc_dist_snapshots` counts it then.
            let size = entry
                .metadata()
                .map(|m| if last_link(&m) { m.len() } else { 0 })
                .unwrap_or(0);
            if std::fs::remove_file(entry.path()).is_ok() {
                n += 1;
                freed += size;
            }
        }
        let (sn, sfreed) = self.gc_dist_snapshots();
        Ok((n + sn, freed + sfreed))
    }

    /// Remove the `dist` snapshots no bundle uses. Their files are hard links
    /// to blobs, so the blobs `gc` just removed still hold their space until
    /// the snapshot goes too. A bundle with no recorded manifest might use any
    /// of them: then none is removed.
    fn gc_dist_snapshots(&self) -> (usize, u64) {
        let mut live = std::collections::BTreeSet::new();
        if let Ok(rd) = std::fs::read_dir(self.root.join("bundles")) {
            for dir in rd.flatten().filter(|d| d.path().is_dir()) {
                let Some(bundle) = std::fs::read(dir.path().join(BUNDLE_MANIFEST))
                    .ok()
                    .and_then(|raw| serde_json::from_slice::<plow_asset::dist::Bundle>(&raw).ok())
                else {
                    return (0, 0);
                };
                if bundle.checkpoint.source == plow_asset::dist::DIST_SOURCE {
                    if let Ok(p) = super::weights::dist_snapshot_path(&self.root, &bundle) {
                        live.insert(p);
                    }
                }
            }
        }
        // A bundle prepared with `--checkpoint` at some other snapshot links
        // into it through its farm without its manifest saying so: live too.
        let snapshots = self.root.join("snapshots");
        // Compared resolved: a `--checkpoint` may reach a snapshot through a
        // symlink, or the store root itself may be one.
        let resolved_root = std::fs::canonicalize(&snapshots).unwrap_or_else(|_| snapshots.clone());
        if let Ok(bundles) = std::fs::read_dir(self.root.join("bundles")) {
            for bundle in bundles.flatten() {
                let Ok(farm) = std::fs::read_link(bundle.path().join("checkpoint")) else {
                    continue;
                };
                let Ok(links) = std::fs::read_dir(bundle.path().join(&farm)) else {
                    continue;
                };
                for l in links.flatten() {
                    if let Ok(target) = std::fs::canonicalize(l.path()) {
                        if let Ok(rest) = target.strip_prefix(&resolved_root) {
                            if let Some(first) = rest.components().next() {
                                live.insert(snapshots.join(first));
                            }
                        }
                    }
                }
            }
        }
        let (mut n, mut freed) = (0usize, 0u64);
        let Ok(rd) = std::fs::read_dir(&snapshots) else {
            return (0, 0);
        };
        for snap in rd.flatten() {
            let path = snap.path();
            let name = snap.file_name().to_string_lossy().into_owned();
            if !name.starts_with("dist+") || !path.is_dir() || live.contains(&path) {
                continue;
            }
            // Space comes back only for files nothing else links to.
            // Its files are links to blobs `gc` counted already: only the
            // space comes back now.
            let (_, bytes) = last_links(&path);
            if std::fs::remove_dir_all(&path).is_ok() {
                freed += bytes;
            }
        }
        (n, freed)
    }
}

/// Whether removing this file frees its bytes: no other hard link to it.
fn last_link(m: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    return std::os::unix::fs::MetadataExt::nlink(m) == 1;
    #[cfg(not(unix))]
    return true;
}

/// Files under `dir` whose every hard link is inside it, and their size:
/// what removing `dir` frees. One file linked twice inside counts once.
fn last_links(dir: &Path) -> (usize, u64) {
    let mut inodes: std::collections::HashMap<(u64, u64), (u64, u64, u64)> = Default::default();
    collect_inodes(dir, &mut inodes);
    inodes
        .values()
        .filter(|(_, nlink, seen)| seen >= nlink)
        .fold((0, 0), |(n, b), (len, _, _)| (n + 1, b + len))
}

fn collect_inodes(dir: &Path, inodes: &mut std::collections::HashMap<(u64, u64), (u64, u64, u64)>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let Ok(m) = std::fs::symlink_metadata(e.path()) else {
            continue;
        };
        if m.is_dir() {
            collect_inodes(&e.path(), inodes);
        } else if m.is_file() {
            #[cfg(unix)]
            let (key, nlink) = {
                use std::os::unix::fs::MetadataExt;
                ((m.dev(), m.ino()), m.nlink())
            };
            #[cfg(not(unix))]
            let (key, nlink) = ((0, inodes.len() as u64), 1);
            inodes.entry(key).or_insert((m.len(), nlink, 0)).2 += 1;
        }
    }
}

/// A path that stays inside the bundle: relative, no `..`, no root, no prefix.
fn is_safe_relative(name: &str) -> bool {
    use std::path::Component;
    !name.is_empty()
        && std::path::Path::new(name)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
