//! One read, parse and hash of a packet file, shared by every consumer while any of them holds it.
//!
//! [`PacketImage::open`] returns the image already open for the same file (same device, inode,
//! size, mtime and ctime, read from the opened descriptor) instead of reading and parsing again.
//! The process keeps only weak references: memory goes when the last holder drops, so a load
//! sequence that wants one parse keeps one [`Arc`] alive across it. `build.json` is cached the
//! same way under its own identity, so a rewritten sidecar is never served stale.

use std::collections::HashMap;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use serde::Deserialize;

use crate::asset::devblob::DevBlob;
use crate::{Result, RuntimeError};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FileId {
    dev: u64,
    ino: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Stamp {
    len: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
}

fn identity(meta: &std::fs::Metadata) -> (FileId, Stamp) {
    (
        FileId {
            dev: meta.dev(),
            ino: meta.ino(),
        },
        Stamp {
            len: meta.len(),
            mtime: (meta.mtime(), meta.mtime_nsec()),
            ctime: (meta.ctime(), meta.ctime_nsec()),
        },
    )
}

type Memo<T> = Mutex<HashMap<FileId, (Stamp, Weak<T>)>>;

/// `Ok(None)` when `path` does not exist. Otherwise the value cached for the file `path` opens,
/// or `load` of its bytes when the cache has none (or a different version of the file).
fn memoized<T>(
    memo: &Memo<T>,
    path: &Path,
    load: impl FnOnce(Vec<u8>) -> Result<T>,
) -> Result<Option<Arc<T>>> {
    let io = |source| RuntimeError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io(e)),
    };
    let (id, stamp) = identity(&file.metadata().map_err(io)?);
    if let Some(hit) = memo
        .lock()
        .unwrap()
        .get(&id)
        .filter(|(s, _)| *s == stamp)
        .and_then(|(_, w)| w.upgrade())
    {
        return Ok(Some(hit));
    }
    let mut raw = Vec::with_capacity(stamp.len as usize);
    file.read_to_end(&mut raw).map_err(io)?;
    let value = Arc::new(load(raw)?);
    // A file rewritten in place while it was read is served once but never cached.
    if identity(&file.metadata().map_err(io)?).1 == stamp {
        let mut memo = memo.lock().unwrap();
        memo.retain(|_, (_, w)| w.strong_count() > 0);
        memo.insert(id, (stamp, Arc::downgrade(&value)));
    }
    Ok(Some(value))
}

/// A packet file's bytes and its parse. The parse carries no L2-placement verdict: every
/// executing consumer applies [`DevBlob::check_l2_dispatch`] with its own answer.
pub struct PacketImage {
    pub path: PathBuf,
    pub raw: Vec<u8>,
    pub blob: DevBlob,
    sha256: OnceLock<[u8; 32]>,
}

impl PacketImage {
    pub fn open(path: &Path) -> Result<Arc<PacketImage>> {
        static PACKETS: OnceLock<Memo<PacketImage>> = OnceLock::new();
        let memo = PACKETS.get_or_init(Default::default);
        memoized(memo, path, |raw| {
            let blob = DevBlob::parse_unguarded(&raw)?;
            Ok(PacketImage {
                path: path.to_path_buf(),
                raw,
                blob,
                sha256: OnceLock::new(),
            })
        })?
        .ok_or_else(|| RuntimeError::Io {
            path: path.to_path_buf(),
            source: std::io::ErrorKind::NotFound.into(),
        })
    }

    /// The packet in an assets directory (see [`DevBlob::find_in_dir`]), if it has one.
    pub fn open_in_dir(dir: &Path) -> Result<Option<Arc<PacketImage>>> {
        DevBlob::find_in_dir(dir)?
            .map(|p| Self::open(&p))
            .transpose()
    }

    /// SHA-256 of the packet bytes, computed once per image.
    pub fn sha256(&self) -> &[u8; 32] {
        self.sha256.get_or_init(|| {
            use sha2::Digest;
            sha2::Sha256::digest(&self.raw).into()
        })
    }

    pub fn sha256_hex(&self) -> String {
        self.sha256().iter().map(|b| format!("{b:02x}")).collect()
    }

    pub(crate) fn digest_cell(&self) -> &OnceLock<[u8; 32]> {
        &self.sha256
    }

    /// The `build.json` beside this packet; see [`BuildInfo::read`].
    pub fn build_info(&self) -> Result<Option<Arc<BuildInfo>>> {
        BuildInfo::read(&self.path.with_file_name(BUILD_FILE))
    }
}

pub const BUILD_FILE: &str = "build.json";

/// The top-level `build.json` blocks the runtime reads at load, kept as JSON values so a field
/// of an unexpected shape reads as it did from the whole document. The analysis records that make
/// up most of the file are skipped without being built.
#[derive(Debug, Default, Deserialize)]
pub struct BuildInfo {
    pub pairing: Option<serde_json::Value>,
    pub union: Option<serde_json::Value>,
    pub tuning: Option<serde_json::Value>,
    pub runtime_requires: Option<serde_json::Value>,
    pub knobs: Option<serde_json::Value>,
    pub rungs: Option<serde_json::Value>,
    pub shapes: Option<serde_json::Value>,
}

impl BuildInfo {
    /// `Ok(None)` when the file is absent; an unreadable or malformed file is an error, never
    /// an empty manifest.
    pub fn read(path: &Path) -> Result<Option<Arc<BuildInfo>>> {
        static BUILDS: OnceLock<Memo<BuildInfo>> = OnceLock::new();
        let memo = BUILDS.get_or_init(Default::default);
        memoized(memo, path, |raw| {
            serde_json::from_slice(&raw).map_err(|source| RuntimeError::Json {
                path: path.to_path_buf(),
                source,
            })
        })
    }

    pub fn in_dir(dir: &Path) -> Result<Option<Arc<BuildInfo>>> {
        Self::read(&dir.join(BUILD_FILE))
    }

    /// `pairing.hash` as the `u64` a specialised object is stamped with.
    pub fn pairing_hash(&self) -> Option<u64> {
        let h = self.pairing.as_ref()?.get("hash")?.as_str()?;
        u64::from_str_radix(h.trim_start_matches("0x"), 16).ok()
    }

    /// Op arms in the `union` the specialised objects compile.
    pub fn union_len(&self) -> usize {
        self.union
            .as_ref()
            .and_then(|u| u.as_array())
            .map_or(0, Vec::len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plow-pkimg-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn build_info_is_shared_while_held_and_reread_when_rewritten() {
        let d = dir("build");
        let path = d.join(BUILD_FILE);
        assert!(BuildInfo::read(&path).unwrap().is_none());
        std::fs::write(&path, r#"{"pairing":{"hash":"0x10"},"big":[1,2,3]}"#).unwrap();
        let a = BuildInfo::read(&path).unwrap().unwrap();
        let b = BuildInfo::read(&path).unwrap().unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.pairing_hash(), Some(16));
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&path, r#"{"pairing":{"hash":"0x11"}}"#).unwrap();
        let c = BuildInfo::read(&path).unwrap().unwrap();
        assert_eq!(c.pairing_hash(), Some(17));
        std::fs::write(&path, "{").unwrap();
        assert!(BuildInfo::read(&path).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn unreadable_build_json_is_an_error() {
        let d = dir("unreadable");
        std::fs::create_dir_all(d.join(BUILD_FILE)).unwrap();
        assert!(BuildInfo::in_dir(&d).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
