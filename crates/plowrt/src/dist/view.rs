//! Checkpoint views: a checkpoint the recipe's prep built from the Hugging Face snapshot (renamed,
//! split, re-typed or new tensors, rewritten configs), carried in a bundle as only the bytes prep
//! derived. Every tensor and file whose bytes the snapshot already holds is referenced by its
//! snapshot path; `load` rebuilds the view byte for byte from the snapshot and the bundle's
//! blobs, so the packets bind exactly the checkpoint they were emitted against and serving reads
//! a plain directory.
//!
//! A safetensors file is rebuilt as written, not aliased: no prepared tensor keeps its snapshot
//! name, and every reader of a checkpoint directory (the weight loader, the speaker encoders)
//! takes names from the file headers.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use plow_asset::dist::{Assembled, Bundle, CheckpointView, ViewFile};

use crate::{Result, RuntimeError};

fn bad(msg: impl Into<String>) -> RuntimeError {
    RuntimeError::Dist(msg.into())
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> RuntimeError + '_ {
    move |source| RuntimeError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// One tensor of a safetensors header: its data range relative to the data section.
#[derive(Clone, Debug)]
pub(crate) struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub start: u64,
    pub end: u64,
}

/// A safetensors file's header: its exact bytes and its tensors in data order.
pub(crate) struct Header {
    pub text: Vec<u8>,
    pub tensors: Vec<Tensor>,
}

impl Header {
    /// Where the data section starts.
    pub fn data_start(&self) -> u64 {
        8 + self.text.len() as u64
    }

    fn parse(text: Vec<u8>, what: &Path) -> Result<Self> {
        let map: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&text)
            .map_err(|e| bad(format!("{}: bad safetensors header: {e}", what.display())))?;
        let mut tensors = Vec::with_capacity(map.len());
        for (name, v) in &map {
            if name == "__metadata__" {
                continue;
            }
            let field = |k: &str| v.get(k).ok_or_else(|| bad(format!("{name}: no `{k}`")));
            let offsets: Vec<u64> = serde_json::from_value(field("data_offsets")?.clone())
                .map_err(|e| bad(format!("{name}: data_offsets: {e}")))?;
            let [start, end] = offsets[..] else {
                return Err(bad(format!("{name}: data_offsets is not [start, end]")));
            };
            if end < start {
                return Err(bad(format!("{name}: data_offsets end before start")));
            }
            tensors.push(Tensor {
                name: name.clone(),
                dtype: field("dtype")?
                    .as_str()
                    .ok_or_else(|| bad(format!("{name}: dtype")))?
                    .to_string(),
                shape: serde_json::from_value(field("shape")?.clone())
                    .map_err(|e| bad(format!("{name}: shape: {e}")))?,
                start,
                end,
            });
        }
        tensors.sort_by_key(|t| (t.start, t.end));
        Ok(Header { text, tensors })
    }

    pub fn read(path: &Path) -> Result<Self> {
        let mut f = std::fs::File::open(path).map_err(io(path))?;
        let size = f.metadata().map_err(io(path))?.len();
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(io(path))?;
        let n = u64::from_le_bytes(len);
        if n > size.saturating_sub(8) || n > 512 << 20 {
            return Err(bad(format!(
                "{}: safetensors header length {n} out of range ({size} bytes)",
                path.display()
            )));
        }
        let mut text = vec![0u8; n as usize];
        f.read_exact(&mut text).map_err(io(path))?;
        let h = Self::parse(text, path)?;
        if let Some(t) = h.tensors.iter().find(|t| t.end > size - h.data_start()) {
            return Err(bad(format!(
                "{}: tensor {} ends past the file",
                path.display(),
                t.name
            )));
        }
        Ok(h)
    }

    /// Whether the tensors tile the data section exactly, so the file is header + parts.
    fn tiles(&self, file_len: u64) -> bool {
        let mut at = 0;
        for t in &self.tensors {
            if t.start != at {
                return false;
            }
            at = t.end;
        }
        self.data_start() + at == file_len
    }
}

fn sha256_file(p: &Path) -> Result<(String, u64)> {
    use sha2::Digest as _;
    let mut f = std::fs::File::open(p).map_err(io(p))?;
    let (mut h, mut n, mut buf) = (sha2::Sha256::new(), 0u64, vec![0u8; 1 << 22]);
    loop {
        let k = f.read(&mut buf).map_err(io(p))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((format!("{:x}", h.finalize()), n))
}

/// Every regular file under `dir` (following links), as `(relative path, path)`, sorted; hidden
/// entries are tool state, not the checkpoint. `repo_only`: a hub cache snapshot, whose
/// repository files are links into `blobs/`; anything else written there (a nested cache, a
/// `CACHEDIR.TAG`) is not in the repository a node fetches.
fn walk(dir: &Path, repo_only: bool) -> Result<Vec<(String, PathBuf)>> {
    fn go(
        root: &Path,
        dir: &Path,
        repo_only: bool,
        seen: &mut std::collections::BTreeSet<PathBuf>,
        out: &mut Vec<(String, PathBuf)>,
    ) -> Result<()> {
        let real = std::fs::canonicalize(dir).map_err(io(dir))?;
        if !seen.insert(real) {
            return Err(bad(format!("{}: a directory link cycle", dir.display())));
        }
        for e in std::fs::read_dir(dir).map_err(io(dir))? {
            let p = e.map_err(io(dir))?.path();
            let name = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            if name.starts_with('.') {
                continue;
            }
            let link = std::fs::symlink_metadata(&p).is_ok_and(|m| m.file_type().is_symlink());
            // A prepared directory may link a subdirectory in (`ecapa/`); a hub snapshot's
            // repository files are never directory links.
            if p.is_dir() && !(link && repo_only) {
                go(root, &p, repo_only, seen, out)?;
            } else if p.is_file() && (link || !repo_only) {
                let rel = p.strip_prefix(root).unwrap_or(&p);
                out.push((rel.to_string_lossy().replace('\\', "/"), p));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    go(dir, dir, repo_only, &mut Default::default(), &mut out)?;
    out.sort();
    Ok(out)
}

/// The digest a view's directory in the store is named by.
pub fn view_digest(view: &CheckpointView) -> Result<String> {
    let bytes = serde_json::to_vec(view).map_err(|e| bad(format!("view {}: {e}", view.dir)))?;
    Ok(super::Digest::of(&bytes).as_str()[..12].to_string())
}

/// Where a view is rebuilt in the store: beside the checkpoint's farm, named by its content, so
/// two bundles share one only when they bind the same bytes.
pub fn view_path(store_root: &Path, bundle: &Bundle, view: &CheckpointView) -> Result<PathBuf> {
    Ok(store_root.join("checkpoints").join(format!(
        "{}@{}+{}",
        super::weights::checkpoint_key(bundle),
        super::weights::checkpoint_revision(bundle),
        view_digest(view)?
    )))
}

/// Read-only maps of snapshot shards, opened once per shard.
struct Shards<'a> {
    snapshot: &'a Path,
    open: BTreeMap<String, (memmap2::Mmap, BTreeMap<String, Tensor>, u64)>,
}

impl<'a> Shards<'a> {
    fn tensor(&mut self, file: &str, name: &str) -> Result<(&[u8], &Tensor)> {
        if !self.open.contains_key(file) {
            if !super::weights::safe_repo_path(file) {
                return Err(bad(format!("{file}: not a snapshot path")));
            }
            let path = self.snapshot.join(file);
            let h = Header::read(&path)?;
            let f = std::fs::File::open(&path).map_err(io(&path))?;
            // SAFETY: a read-only map of a snapshot file nothing in this process writes.
            let map = unsafe { memmap2::Mmap::map(&f) }.map_err(io(&path))?;
            let base = h.data_start();
            let by_name = h.tensors.into_iter().map(|t| (t.name.clone(), t)).collect();
            self.open.insert(file.to_string(), (map, by_name, base));
        }
        let (map, by_name, base) = &self.open[file];
        let t = by_name
            .get(name)
            .ok_or_else(|| bad(format!("snapshot {file} has no tensor {name}")))?;
        let (s, e) = (base + t.start, base + t.end);
        let bytes = map
            .get(s as usize..e as usize)
            .ok_or_else(|| bad(format!("snapshot {file}: {name} runs past the file")))?;
        Ok((bytes, t))
    }
}

/// Rebuild one view into `dir` (which must not exist), from `snapshot` and the store's blobs.
fn build_into(
    store: &super::Store,
    view: &CheckpointView,
    snapshot: &Path,
    dir: &Path,
) -> Result<()> {
    use sha2::Digest as _;
    let mut shards = Shards {
        snapshot,
        open: BTreeMap::new(),
    };
    for f in &view.files {
        let dst = dir.join(&f.name);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).map_err(io(parent))?;
        }
        let refused = |why: String| {
            bad(format!(
                "checkpoint view {}: {}: {why}. The snapshot is not the one this bundle was built \
                 against, or a blob is damaged",
                view.dir, f.name
            ))
        };
        if let Some(hf) = &f.hf {
            if !super::weights::safe_repo_path(hf) {
                return Err(refused(format!("{hf} is not a snapshot path")));
            }
            let src = snapshot.join(hf);
            let (sha, n) = sha256_file(&src).map_err(|e| refused(e.to_string()))?;
            if sha != f.sha256 || n != f.bytes {
                return Err(refused(format!(
                    "snapshot {hf} hashes to {sha} ({n} bytes)"
                )));
            }
            std::os::unix::fs::symlink(&src, &dst).map_err(io(&dst))?;
            continue;
        }
        let Some(a) = &f.assembled else {
            let digest = super::Digest::parse(&f.sha256)?;
            let src = store.blob_path(&digest);
            if !store.has(&digest) {
                return Err(refused(format!("blob {digest} is not in the store")));
            }
            if std::fs::hard_link(&src, &dst).is_err() {
                std::fs::copy(&src, &dst).map_err(io(&dst))?;
            }
            continue;
        };
        let header = Header::parse(a.header.as_bytes().to_vec(), &dst)?;
        if header.tensors.len() != a.parts.len() {
            return Err(refused(format!(
                "{} parts for {} tensors",
                a.parts.len(),
                header.tensors.len()
            )));
        }
        let mut derived = match &a.derived {
            Some(d) => {
                let digest = super::Digest::parse(&d.sha256)?;
                if !store.has(&digest) {
                    return Err(refused(format!(
                        "derived blob {digest} is not in the store"
                    )));
                }
                let p = store.blob_path(&digest);
                Some(std::io::BufReader::new(
                    std::fs::File::open(&p).map_err(io(&p))?,
                ))
            }
            None => None,
        };
        let file = std::fs::File::create(&dst).map_err(io(&dst))?;
        let mut out = std::io::BufWriter::with_capacity(8 << 20, file);
        let mut h = sha2::Sha256::new();
        let mut n = 0u64;
        let mut put = |b: &[u8], out: &mut std::io::BufWriter<std::fs::File>| -> Result<()> {
            h.update(b);
            n += b.len() as u64;
            out.write_all(b).map_err(io(&dst))
        };
        put(&(a.header.len() as u64).to_le_bytes(), &mut out)?;
        put(a.header.as_bytes(), &mut out)?;
        let mut buf = vec![0u8; 8 << 20];
        for (t, p) in header.tensors.iter().zip(&a.parts) {
            let want = t.end - t.start;
            match (&p.hf, &p.tensor, p.derived) {
                (Some(file), Some(name), None) => {
                    let (bytes, src) = shards
                        .tensor(file, name)
                        .map_err(|e| refused(e.to_string()))?;
                    if src.dtype != t.dtype || src.shape != t.shape || bytes.len() as u64 != want {
                        return Err(refused(format!(
                            "{}: snapshot tensor {file}:{name} is {} {:?}, the view binds {} {:?}",
                            t.name, src.dtype, src.shape, t.dtype, t.shape
                        )));
                    }
                    put(bytes, &mut out)?;
                }
                (None, None, Some(len)) if len == want => {
                    let r = derived
                        .as_mut()
                        .ok_or_else(|| refused("derived part without a derived blob".into()))?;
                    let mut left = len;
                    while left > 0 {
                        let k = left.min(buf.len() as u64) as usize;
                        r.read_exact(&mut buf[..k])
                            .map_err(|e| refused(format!("derived blob: {e}")))?;
                        put(&buf[..k], &mut out)?;
                        left -= k as u64;
                    }
                }
                _ => {
                    return Err(refused(format!(
                        "{}: part does not match the tensor",
                        t.name
                    )))
                }
            }
        }
        out.flush().map_err(io(&dst))?;
        let got = format!("{:x}", h.finalize());
        if got != f.sha256 || n != f.bytes {
            return Err(refused(format!("rebuilt file hashes to {got} ({n} bytes)")));
        }
    }
    Ok(())
}

/// Rebuild a view (or reuse it: its path names its content) and return its directory.
/// Staged and renamed, so a reader never sees a partial view.
pub fn materialize(
    store: &super::Store,
    bundle: &Bundle,
    view: &CheckpointView,
    snapshot: &Path,
) -> Result<PathBuf> {
    view.validate().map_err(RuntimeError::Dist)?;
    let dir = view_path(store.root(), bundle, view)?;
    // Published by rename once whole; its snapshot links dangle if the HF cache was evicted.
    if dir.is_dir() && view.files.iter().all(|f| dir.join(&f.name).is_file()) {
        return Ok(dir);
    }
    let _ = std::fs::remove_dir_all(&dir);
    let staging = dir.with_extension(format!("partial-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(io(&staging))?;
    if let Err(e) = build_into(store, view, snapshot, &staging) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&staging, &dir) {
        let _ = std::fs::remove_dir_all(&staging);
        if !dir.is_dir() {
            return Err(io(&dir)(e));
        }
    }
    Ok(dir)
}

// --- publish side --------------------------------------------------------------------------

/// The blobs a view carries, digest → bytes: its derived files and derived tensor bytes.
pub fn blobs(view: &CheckpointView) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    for f in &view.files {
        match (&f.hf, &f.assembled) {
            (None, None) => out.push((f.sha256.clone(), f.bytes)),
            (
                _,
                Some(Assembled {
                    derived: Some(d), ..
                }),
            ) => out.push((d.sha256.clone(), d.bytes)),
            _ => {}
        }
    }
    out
}

/// Write the blob `digest` of `view` (bound at `bound`) to `out`: a derived file is copied, a
/// derived tensor blob is cut from the bound file's derived parts.
pub fn write_blob(view: &CheckpointView, bound: &Path, digest: &str, out: &Path) -> Result<()> {
    for f in &view.files {
        let src = bound.join(&f.name);
        if f.hf.is_none() && f.assembled.is_none() && f.sha256 == digest {
            std::fs::copy(&src, out).map_err(io(out))?;
            return Ok(());
        }
        let Some(a) = f
            .assembled
            .as_ref()
            .filter(|a| a.derived.as_ref().is_some_and(|d| d.sha256 == digest))
        else {
            continue;
        };
        let header = Header::read(&src)?;
        let mut r = std::fs::File::open(&src).map_err(io(&src))?;
        let mut w = std::io::BufWriter::with_capacity(
            8 << 20,
            std::fs::File::create(out).map_err(io(out))?,
        );
        let mut buf = vec![0u8; 8 << 20];
        for (t, p) in header.tensors.iter().zip(&a.parts) {
            if p.derived.is_none() {
                continue;
            }
            use std::io::Seek;
            r.seek(std::io::SeekFrom::Start(header.data_start() + t.start))
                .map_err(io(&src))?;
            let mut left = t.end - t.start;
            while left > 0 {
                let k = left.min(buf.len() as u64) as usize;
                r.read_exact(&mut buf[..k]).map_err(io(&src))?;
                w.write_all(&buf[..k]).map_err(io(out))?;
                left -= k as u64;
            }
        }
        w.flush().map_err(io(out))?;
        return Ok(());
    }
    Err(bad(format!("view {}: no blob {digest}", view.dir)))
}

/// The pinned snapshot in the local Hugging Face cache (`HF_HUB_CACHE`, else `HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`), which a view is computed against.
#[cfg(feature = "dist")]
pub fn local_snapshot(source: &str, revision: &str) -> Result<PathBuf> {
    let repo = super::weights::hf_repo(source)?;
    let cache = match std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        Some(p) => PathBuf::from(p),
        None => hf_hub::Cache::from_env().path().clone(),
    };
    let dir = cache
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(revision);
    if !dir.is_dir() {
        return Err(bad(format!(
            "{source}@{revision}: no snapshot at {}. A prepared checkpoint is published as a view \
             of the pinned snapshot, so that snapshot must be in the local Hugging Face cache \
             (set HF_HUB_CACHE or HF_HOME)",
            dir.display()
        )));
    }
    Ok(dir)
}

/// Compute the view of the bound checkpoint `bound` (published at `dir` in the bundle) against
/// `snapshot`: each file an HF file when its bytes are one, an assembled safetensors file when
/// its tensors tile its data, else a derived blob. Tensors match on dtype, shape and bytes,
/// preferring the same name.
#[cfg(feature = "dist")]
pub fn compute(dir: &str, bound: &Path, snapshot: &Path) -> Result<CheckpointView> {
    use sha2::Digest as _;
    // Snapshot files by size, hashed only when a bound file has that size.
    let hub = snapshot.parent().is_some_and(|s| s.ends_with("snapshots"))
        && snapshot.join("../../blobs").is_dir();
    let snap = walk(snapshot, hub)?;
    let mut by_size: BTreeMap<u64, Vec<(String, PathBuf)>> = BTreeMap::new();
    let mut tensors: BTreeMap<(String, Vec<u64>, u64), Vec<(String, Tensor)>> = BTreeMap::new();
    for (rel, p) in &snap {
        let len = std::fs::metadata(p).map_err(io(p))?.len();
        by_size
            .entry(len)
            .or_default()
            .push((rel.clone(), p.clone()));
        if rel.ends_with(".safetensors") {
            for t in Header::read(p)?.tensors {
                let key = (t.dtype.clone(), t.shape.clone(), t.end - t.start);
                tensors.entry(key).or_default().push((rel.clone(), t));
            }
        }
    }
    let mut hashed: BTreeMap<PathBuf, String> = BTreeMap::new();
    let mut tensor_sha: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut maps: BTreeMap<String, (memmap2::Mmap, u64)> = BTreeMap::new();
    let map_of = |rel: &str, maps: &mut BTreeMap<String, (memmap2::Mmap, u64)>| -> Result<()> {
        if !maps.contains_key(rel) {
            let p = snapshot.join(rel);
            let base = Header::read(&p)?.data_start();
            let f = std::fs::File::open(&p).map_err(io(&p))?;
            // SAFETY: a read-only map of a snapshot file nothing in this process writes.
            let m = unsafe { memmap2::Mmap::map(&f) }.map_err(io(&p))?;
            maps.insert(rel.to_string(), (m, base));
        }
        Ok(())
    };
    let sha = |b: &[u8]| format!("{:x}", sha2::Sha256::digest(b));

    let mut files = Vec::new();
    for (rel, p) in walk(bound, false)? {
        let (sha256, bytes) = sha256_file(&p)?;
        let mut same = None;
        for (srel, sp) in by_size.get(&bytes).into_iter().flatten() {
            let h = match hashed.get(sp) {
                Some(h) => h.clone(),
                None => {
                    let h = sha256_file(sp)?.0;
                    hashed.insert(sp.clone(), h.clone());
                    h
                }
            };
            if h == sha256 && (same.is_none() || *srel == rel) {
                same = Some(srel.clone());
            }
        }
        let mut file = ViewFile {
            name: rel.clone(),
            sha256,
            bytes,
            hf: same,
            assembled: None,
        };
        if file.hf.is_none() && rel.ends_with(".safetensors") {
            let header = Header::read(&p)?;
            // Shipped whole, its snapshot tensors would ride along as plow bytes.
            let text = String::from_utf8(header.text.clone())
                .ok()
                .filter(|_| header.tiles(bytes))
                .ok_or_else(|| {
                    bad(format!(
                        "{}: its tensors do not tile its data, so it cannot be published as \
                         snapshot tensors plus derived bytes",
                        p.display()
                    ))
                })?;
            let f = std::fs::File::open(&p).map_err(io(&p))?;
            // SAFETY: a read-only map of a build output nothing in this process writes.
            let m = unsafe { memmap2::Mmap::map(&f) }.map_err(io(&p))?;
            let base = header.data_start() as usize;
            let (mut parts, mut dh, mut dn) = (Vec::new(), sha2::Sha256::new(), 0u64);
            for t in &header.tensors {
                let b = &m[base + t.start as usize..base + t.end as usize];
                let key = (t.dtype.clone(), t.shape.clone(), t.end - t.start);
                let mut found: Option<(String, String)> = None;
                let mut want = None;
                for (srel, st) in tensors.get(&key).into_iter().flatten() {
                    let k = (srel.clone(), st.name.clone());
                    let h = match tensor_sha.get(&k) {
                        Some(h) => h.clone(),
                        None => {
                            map_of(srel, &mut maps)?;
                            let (sm, sbase) = &maps[srel];
                            let s = (sbase + st.start) as usize;
                            let h = sha(&sm[s..s + (st.end - st.start) as usize]);
                            tensor_sha.insert(k.clone(), h.clone());
                            h
                        }
                    };
                    let want = want.get_or_insert_with(|| sha(b));
                    if h == *want && (found.is_none() || st.name == t.name) {
                        found = Some(k);
                    }
                }
                parts.push(match found {
                    Some((hf, tensor)) => plow_asset::dist::Part {
                        hf: Some(hf),
                        tensor: Some(tensor),
                        derived: None,
                    },
                    None => {
                        dh.update(b);
                        dn += b.len() as u64;
                        plow_asset::dist::Part {
                            hf: None,
                            tensor: None,
                            derived: Some(b.len() as u64),
                        }
                    }
                });
            }
            file.assembled = Some(Assembled {
                header: text,
                parts,
                derived: (dn > 0).then(|| plow_asset::dist::CheckpointFile {
                    name: format!("{rel}.derived"),
                    sha256: format!("{:x}", dh.finalize()),
                    bytes: dn,
                }),
            });
        }
        files.push(file);
    }
    let view = CheckpointView {
        dir: dir.to_string(),
        files,
    };
    view.validate().map_err(RuntimeError::Dist)?;
    Ok(view)
}

#[cfg(all(test, feature = "dist"))]
#[path = "view_tests.rs"]
pub(crate) mod tests;
