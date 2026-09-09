//! The local content-addressed store.
//!
//! ```text
//! ~/.plow/
//!   blobs/sha256/<hex>                                   every file, content-addressed
//!   manifests/<registry>/<namespace>/<name>/<label>@g<n>
//!   refs/<registry>/<namespace>/<name>                   the pinned variant_id
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
            return Err(RuntimeError::Device(format!(
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
                    RuntimeError::Device(
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
            return Err(RuntimeError::Device(format!(
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
            return Err(RuntimeError::Device(format!(
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
                return Err(RuntimeError::Device(format!(
                    "{name}: blob {digest} is not in the store — run `plowrt pull` first"
                )));
            }
            // A manifest is fetched over the network, so its file names are
            // untrusted input: a `..` or an absolute path would write outside
            // the bundle. Objects legitimately sit under `hsaco/`, so nested
            // paths are allowed, but only downward.
            if !is_safe_relative(name) {
                return Err(RuntimeError::Device(format!(
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
        Ok(dir)
    }

    /// Record which variant a reference currently resolves to.
    ///
    /// A pin is written by `load` and moved only by `upgrade`: a served model
    /// must not drift under a running server because a catalog refreshed.
    pub fn pin(&self, ref_path: &str, variant_id: &str) -> Result<()> {
        let p = self.root.join("refs").join(ref_path);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|source| RuntimeError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        std::fs::write(&p, variant_id).map_err(|source| RuntimeError::Io {
            path: p.clone(),
            source,
        })
    }

    pub fn pinned(&self, ref_path: &str) -> Option<String> {
        std::fs::read_to_string(self.root.join("refs").join(ref_path))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn unpin(&self, ref_path: &str) -> Result<()> {
        let p = self.root.join("refs").join(ref_path);
        match std::fs::remove_file(&p) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(source) => Err(RuntimeError::Io { path: p, source }),
        }
    }

    /// Every pinned reference, as `(ref_path, variant_id)`.
    pub fn pins(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        collect_pins(&self.root.join("refs"), &mut String::new(), &mut out);
        out.sort();
        out
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
                let Ok(files) = std::fs::read_dir(bundle.path()) else {
                    continue;
                };
                for f in files.flatten() {
                    if let Ok(bytes) = std::fs::read(f.path()) {
                        live.insert(Digest::of(&bytes).0);
                    }
                }
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
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            if std::fs::remove_file(entry.path()).is_ok() {
                n += 1;
                freed += size;
            }
        }
        Ok((n, freed))
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

fn collect_pins(dir: &Path, prefix: &mut String, out: &mut Vec<(String, String)>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let path = e.path();
        if path.is_dir() {
            let saved = prefix.len();
            if !prefix.is_empty() {
                prefix.push('/');
            }
            prefix.push_str(&name);
            collect_pins(&path, prefix, out);
            prefix.truncate(saved);
        } else if let Ok(v) = std::fs::read_to_string(&path) {
            let key = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            out.push((key, v.trim().to_string()));
        }
    }
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
