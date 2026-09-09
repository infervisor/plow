//! Resolve a reference against this machine, fetch what is missing, and
//! materialize a directory `serve --assets` already knows how to load.
//!
//! Everything is verified against a digest before it is stored, and the digest
//! always names the UNCOMPRESSED bytes, so how a blob travelled is invisible
//! here. Nothing is written into the bundle directory until every blob it needs
//! is present and verified.

use plow_asset::dist::{Bundle, Constraints, LiveTarget, ModelIndex, ObjSet, Variant};

use super::{fetch, Digest, Fetch, Reference, Store};
use crate::{Result, RuntimeError};

/// What a pull would move, or did.
///
/// `fetched` counts only blobs that were actually absent, which is what makes an
/// objset-only upgrade honest: the number reported is the number transferred,
/// derived from digests rather than from what kind of change it was.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Transfer {
    pub fetched_blobs: usize,
    pub fetched_bytes: u64,
    pub reused_blobs: usize,
    pub reused_bytes: u64,
}

impl Transfer {
    pub fn total_bytes(&self) -> u64 {
        self.fetched_bytes + self.reused_bytes
    }
}

impl std::fmt::Display for Transfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} blob(s), {} to fetch, {} already local",
            self.fetched_blobs + self.reused_blobs,
            human(self.fetched_bytes),
            human(self.reused_bytes)
        )
    }
}

pub fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// A resolved variant and everything needed to place it on disk.
#[derive(Debug)]
pub struct Resolved {
    pub variant: Variant,
    pub bundle: Bundle,
    pub objsets: Vec<(Option<u32>, ObjSet)>,
}

impl Resolved {
    /// Every blob this variant needs, as `(path within the bundle, digest, bytes)`.
    ///
    /// Objects land under `hsaco/`, and a low-rung tier under
    /// `hsaco/lowrung<max>/` — the layout `scripts/glm53_serve_inner.sh` already
    /// wires `PLOW_HSACO_LOWRUNG` to, so the runtime finds them without an
    /// absolute path leaving the build host.
    pub fn files(&self) -> Result<Vec<(String, Digest, u64)>> {
        let mut out = Vec::new();
        for f in &self.bundle.files {
            out.push((f.name.clone(), Digest::parse(&f.sha256)?, f.bytes));
        }
        for (rung, set) in &self.objsets {
            let dir = match rung {
                None => "hsaco".to_string(),
                Some(max) => format!("hsaco/lowrung{max}"),
            };
            for o in &set.objects {
                out.push((
                    format!("{dir}/{}", o.name),
                    Digest::parse(&o.sha256)?,
                    o.bytes,
                ));
            }
        }
        Ok(out)
    }

    /// Split the file list by what the store already has. No network.
    pub fn plan(&self, store: &Store) -> Result<Transfer> {
        let mut t = Transfer::default();
        for (_, d, bytes) in self.files()? {
            if store.has(&d) {
                t.reused_blobs += 1;
                t.reused_bytes += bytes;
            } else {
                t.fetched_blobs += 1;
                t.fetched_bytes += bytes;
            }
        }
        Ok(t)
    }
}

fn parse_json<T: serde::de::DeserializeOwned>(what: &str, bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes)
        .map_err(|e| RuntimeError::Device(format!("{what}: not valid JSON for this schema: {e}")))
}

fn require(f: &dyn Fetch, path: &str) -> Result<Vec<u8>> {
    f.get(path)?
        .ok_or_else(|| RuntimeError::Device(format!("{}: {path} is not published", f.describe())))
}

/// Fetch a model's variant index.
pub fn index(f: &dyn Fetch, r: &Reference) -> Result<ModelIndex> {
    let raw = require(f, &r.index_path())?;
    let idx: ModelIndex = parse_json("index.json", &raw)?;
    idx.validate().map_err(RuntimeError::Device)?;
    Ok(idx)
}

/// Resolve a reference to a variant and fetch its manifests — but no blobs.
///
/// This is what `show`, `ls --upgradable` and `--dry-run` use: enough to report
/// exactly what a pull would move, without moving it.
pub fn resolve(
    f: &dyn Fetch,
    r: &Reference,
    live: &LiveTarget,
    extra: Constraints,
) -> Result<Resolved> {
    let idx = index(f, r)?;
    let variant = super::resolve(&idx, r, live, extra)
        .map_err(RuntimeError::Device)?
        .clone();

    let raw = require(
        f,
        &format!("v1/{}/{}/{}", r.namespace, r.name, variant.manifest),
    )?;
    // The index states the manifest's digest, so a manifest that does not match
    // it is a mismatched or tampered publication, not merely a stale cache.
    let got = Digest::of(&raw);
    if got.as_str() != variant.sha256 {
        return Err(RuntimeError::Device(format!(
            "{}@g{}: the index says its manifest hashes to {}, but the published manifest hashes \
             to {got}",
            variant.label, variant.generation, variant.sha256
        )));
    }
    let bundle: Bundle = parse_json("bundle.json", &raw)?;
    bundle.validate().map_err(RuntimeError::Device)?;

    let mut objsets = Vec::new();
    let mut want: Vec<(Option<u32>, String, String)> = vec![(
        None,
        bundle.objset.manifest.clone(),
        bundle.objset.sha256.clone(),
    )];
    for l in &bundle.objset.lowrung {
        want.push((
            Some(l.max),
            format!("v1/objsets/{}.json", l.objset_id),
            String::new(),
        ));
    }
    for (rung, path, sha) in want {
        let raw = require(f, &path)?;
        if !sha.is_empty() && Digest::of(&raw).as_str() != sha {
            return Err(RuntimeError::Device(format!(
                "{path}: objset manifest does not match the digest the bundle records"
            )));
        }
        let set: ObjSet = parse_json(&path, &raw)?;
        set.validate().map_err(RuntimeError::Device)?;
        // A specialised object pairs only with the packet that produced it.
        // Checking here means a mismatched publication is refused before a
        // single object byte moves, rather than twenty frames into the loader.
        set.pairs_with(bundle.pairing_hash.as_deref())
            .map_err(RuntimeError::Device)?;
        objsets.push((rung, set));
    }

    Ok(Resolved {
        variant,
        bundle,
        objsets,
    })
}

/// Fetch every missing blob, verify it, and materialize the bundle directory.
pub fn pull(
    store: &Store,
    f: &dyn Fetch,
    resolved: &Resolved,
) -> Result<(std::path::PathBuf, Transfer)> {
    let files = resolved.files()?;
    let mut t = Transfer::default();
    for (name, digest, bytes) in &files {
        if store.has(digest) {
            t.reused_blobs += 1;
            t.reused_bytes += bytes;
            continue;
        }
        let raw = fetch::blob(f, digest)?;
        // `put` re-hashes and refuses a mismatch, so a corrupted transfer never
        // reaches the store under a name that promises its content.
        store
            .put(digest, &raw)
            .map_err(|e| RuntimeError::Device(format!("{name}: {e}")))?;
        t.fetched_blobs += 1;
        t.fetched_bytes += raw.len() as u64;
    }
    let flat: Vec<(String, Digest)> = files.into_iter().map(|(n, d, _)| (n, d)).collect();
    let dir = store.materialize(&resolved.variant.variant_id, &flat)?;
    Ok((dir, t))
}

#[cfg(test)]
#[path = "pull_tests.rs"]
mod tests;
