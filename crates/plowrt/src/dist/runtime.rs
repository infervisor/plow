//! `plowrt push-runtime`: publish released `plowrt` binaries into a registry.
//!
//! The producer is `nix build .#plowrt-release`: one directory per platform
//! triple holding the tarball, its `.sha256` (as `sha256sum` writes it) and
//! `release.json`. This module records those outputs under
//! `v1/runtime/<version>/<triple>/`, keeps `v1/runtime/index.json` (every
//! release, every triple: what a models page lists) and `v1/runtime/latest`
//! (the version string alone: what `install.sh` reads with `sed`), and
//! optionally publishes `install.sh` itself at the registry root.
//!
//! The same write discipline as `push`: a release's files are create-only,
//! so a version is never republished with other bytes — a second run of the
//! same output is adopted, a different output under the same name is
//! refused — and the index and the `latest` pointer are replaced only when
//! still the ones this run read.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::sink::{Cond, Put, Sink};
use crate::{Result, RuntimeError};

pub const RUNTIME_INDEX_SCHEMA: &str = "plow.dist.runtime.v1";
pub const RUNTIME_INDEX: &str = "v1/runtime/index.json";
pub const RUNTIME_LATEST: &str = "v1/runtime/latest";
pub const INSTALL_SCRIPT: &str = "install.sh";

fn dist(msg: String) -> RuntimeError {
    RuntimeError::Dist(msg)
}

/// `release.json`, as the flake writes it beside the tarball.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReleaseJson {
    /// `<semver>+<git12>`.
    pub version: String,
    pub semver: String,
    pub commit: String,
    pub triple: String,
    #[serde(default)]
    pub features: Vec<String>,
    /// The tarball's file name.
    pub file: String,
}

/// One released binary in the index.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RuntimeAsset {
    /// Key under the registry root.
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
    #[serde(default)]
    pub features: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RuntimeRelease {
    pub version: String,
    pub semver: String,
    pub commit: String,
    /// RFC 3339, when the version was first recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub released: Option<String>,
    pub triples: BTreeMap<String, RuntimeAsset>,
}

/// `v1/runtime/index.json`. Newest release first.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RuntimeIndex {
    pub schema: String,
    pub latest: String,
    pub releases: Vec<RuntimeRelease>,
}

impl Default for RuntimeIndex {
    fn default() -> Self {
        RuntimeIndex {
            schema: RUNTIME_INDEX_SCHEMA.into(),
            latest: String::new(),
            releases: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Record the release in the index without moving `latest` to it.
    pub keep_latest: bool,
    /// Publish this file as `install.sh` at the registry root.
    pub install_script: Option<PathBuf>,
    pub dry_run: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    pub version: String,
    /// `(triple, bytes)` uploaded by this run; an adopted asset is not listed.
    pub uploaded: Vec<(String, u64)>,
    pub adopted: Vec<String>,
    pub latest: String,
    pub install_script: bool,
}

/// A build output directory, checked: the tarball hashes to what its
/// `.sha256` says and `release.json` names it.
#[derive(Debug)]
pub struct ReleaseDir {
    pub json: ReleaseJson,
    pub tarball: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub release_json_bytes: Vec<u8>,
}

impl ReleaseDir {
    pub fn read(dir: &Path) -> Result<ReleaseDir> {
        let io = |p: &Path, e: std::io::Error| RuntimeError::Io {
            path: p.to_path_buf(),
            source: e,
        };
        let rj = dir.join("release.json");
        let release_json_bytes = std::fs::read(&rj).map_err(|e| io(&rj, e))?;
        let json: ReleaseJson = serde_json::from_slice(&release_json_bytes)
            .map_err(|e| dist(format!("{}: {e}", rj.display())))?;
        for (what, v) in [
            ("version", &json.version),
            ("triple", &json.triple),
            ("file", &json.file),
        ] {
            if v.is_empty() || v.contains('/') || v.contains("..") {
                return Err(dist(format!("{}: bad {what} {v:?}", rj.display())));
            }
        }
        if json.version.contains("-dirty") || json.version.ends_with("+unknown") {
            return Err(dist(format!(
                "{}: version {} was built from a checkout that names no commit; \
                 a release is built from a clean tree",
                rj.display(),
                json.version
            )));
        }
        let tarball = dir.join(&json.file);
        let (sha256, bytes) = sha256_file(&tarball)?;
        let sums = dir.join(format!("{}.sha256", json.file));
        let recorded = std::fs::read_to_string(&sums).map_err(|e| io(&sums, e))?;
        let recorded = recorded.split_whitespace().next().unwrap_or("").to_string();
        if recorded != sha256 {
            return Err(dist(format!(
                "{}: hashes to {sha256}, {} says {recorded}",
                tarball.display(),
                sums.display()
            )));
        }
        Ok(ReleaseDir {
            json,
            tarball,
            sha256,
            bytes,
            release_json_bytes,
        })
    }

    fn key_dir(&self) -> String {
        format!("v1/runtime/{}/{}", self.json.version, self.json.triple)
    }
}

fn sha256_file(p: &Path) -> Result<(String, u64)> {
    use sha2::Digest as _;
    use std::io::Read as _;
    let io = |e: std::io::Error| RuntimeError::Io {
        path: p.to_path_buf(),
        source: e,
    };
    let mut f = std::fs::File::open(p).map_err(io)?;
    let (mut h, mut n, mut buf) = (sha2::Sha256::new(), 0u64, vec![0u8; 1 << 20]);
    loop {
        let k = f.read(&mut buf).map_err(io)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((format!("{:x}", h.finalize()), n))
}

fn parse<T: serde::de::DeserializeOwned>(what: &str, bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes)
        .map_err(|e| dist(format!("{what}: not valid JSON for this schema: {e}")))
}

fn put_bytes(sink: &dyn Sink, key: &str, bytes: &[u8], ct: &str, cond: Cond) -> Result<Put> {
    sink.put(key, &mut &bytes[..], bytes.len() as u64, ct, cond)
}

/// Create-only put of a release file; a file already there holding the same
/// bytes (by `release.json` equality, which names the tarball's digest
/// indirectly through its `.sha256`) is adopted, another's is refused.
fn put_release_file(
    sink: &dyn Sink,
    key: &str,
    bytes: &[u8],
    ct: &str,
    same_as_published: impl Fn(&[u8]) -> bool,
) -> Result<bool> {
    match put_bytes(sink, key, bytes, ct, Cond::IfNoneMatchAny)? {
        Put::Stored(_) => Ok(true),
        Put::PreconditionFailed => {
            let (there, _) = sink
                .get(key)?
                .ok_or_else(|| dist(format!("{key}: refused as existing, then absent")))?;
            if same_as_published(&there) {
                Ok(false)
            } else {
                Err(dist(format!(
                    "{key}: already published with other contents; a release version is \
                     never rewritten, bump the version"
                )))
            }
        }
    }
}

/// Publish every release directory, then the index, `latest` and, when
/// given, `install.sh`. Order: files (create-only), index (`If-Match`),
/// `latest`, `install.sh` — a run that dies leaves the previous `latest`
/// pointing at a complete release.
pub fn push_runtime(sink: &dyn Sink, dirs: &[ReleaseDir], opts: &Options) -> Result<Outcome> {
    let first = dirs
        .first()
        .ok_or_else(|| dist("push-runtime: no release directory".into()))?;
    let version = first.json.version.clone();
    for d in dirs {
        if d.json.version != version {
            return Err(dist(format!(
                "{}: version {} differs from {}; one push publishes one version",
                d.tarball.display(),
                d.json.version,
                version
            )));
        }
        if d.json.semver != first.json.semver || d.json.commit != first.json.commit {
            return Err(dist(format!(
                "{}: semver/commit differ from {}'s under the same version",
                d.tarball.display(),
                first.tarball.display()
            )));
        }
    }
    let mut out = Outcome {
        version: version.clone(),
        ..Default::default()
    };
    if opts.dry_run {
        out.uploaded = dirs
            .iter()
            .map(|d| (d.json.triple.clone(), d.bytes))
            .collect();
        out.latest = if opts.keep_latest {
            String::new()
        } else {
            version.clone()
        };
        out.install_script = opts.install_script.is_some();
        return Ok(out);
    }

    // 1. The files, create-only. The tarball is compared through release.json:
    //    a bucket's entity tag is not its sha256, and the `.sha256` beside it
    //    is what install.sh trusts, so both are written before the tarball is
    //    considered published.
    for d in dirs {
        let dir = d.key_dir();
        let sums = format!("{}  {}\n", d.sha256, d.json.file);
        let same_sums = |there: &[u8]| {
            String::from_utf8_lossy(there).split_whitespace().next() == Some(d.sha256.as_str())
        };
        let fresh_sums = put_release_file(
            sink,
            &format!("{dir}/{}.sha256", d.json.file),
            sums.as_bytes(),
            "text/plain",
            same_sums,
        )?;
        let fresh_json = put_release_file(
            sink,
            &format!("{dir}/release.json"),
            &d.release_json_bytes,
            "application/json",
            |there| parse::<ReleaseJson>("release.json", there).ok().as_ref() == Some(&d.json),
        )?;
        let tar_key = format!("{dir}/{}", d.json.file);
        let fresh_tar = match sink.head(&tar_key)? {
            Some(_) if !fresh_sums && !fresh_json => false,
            Some(_) => {
                return Err(dist(format!(
                    "{tar_key}: exists while its release.json or .sha256 did not; remove it \
                     by hand before publishing this version"
                )))
            }
            None => {
                let mut f = std::fs::File::open(&d.tarball).map_err(|e| RuntimeError::Io {
                    path: d.tarball.clone(),
                    source: e,
                })?;
                match sink.put(&tar_key, &mut f, d.bytes, "application/gzip", Cond::IfNoneMatchAny)? {
                    Put::Stored(_) => true,
                    Put::PreconditionFailed => false,
                }
            }
        };
        if fresh_tar {
            out.uploaded.push((d.json.triple.clone(), d.bytes));
        } else {
            out.adopted.push(d.json.triple.clone());
        }
    }

    // 2. The index, replaced only if still the one read.
    let mut published_latest = String::new();
    for attempt in 0..4 {
        let (mut index, cond) = match sink.get(RUNTIME_INDEX)? {
            Some((raw, etag)) => (
                parse::<RuntimeIndex>(RUNTIME_INDEX, &raw)?,
                Cond::IfMatch(etag),
            ),
            None => (RuntimeIndex::default(), Cond::IfNoneMatchAny),
        };
        if index.schema != RUNTIME_INDEX_SCHEMA {
            return Err(dist(format!(
                "{RUNTIME_INDEX}: schema {} is not {RUNTIME_INDEX_SCHEMA}",
                index.schema
            )));
        }
        let pos = index.releases.iter().position(|r| r.version == version);
        let mut row = match pos {
            Some(i) => index.releases.remove(i),
            None => RuntimeRelease {
                version: version.clone(),
                semver: first.json.semver.clone(),
                commit: first.json.commit.clone(),
                released: Some(now_rfc3339()),
                triples: BTreeMap::new(),
            },
        };
        for d in dirs {
            row.triples.insert(
                d.json.triple.clone(),
                RuntimeAsset {
                    file: format!("{}/{}", d.key_dir(), d.json.file),
                    sha256: d.sha256.clone(),
                    bytes: d.bytes,
                    features: d.json.features.clone(),
                },
            );
        }
        index.releases.insert(0, row);
        if !opts.keep_latest || index.latest.is_empty() {
            index.latest = version.clone();
        }
        published_latest = index.latest.clone();
        let bytes = serde_json::to_vec_pretty(&index)
            .map_err(|e| dist(format!("{RUNTIME_INDEX}: serialize: {e}")))?;
        match put_bytes(sink, RUNTIME_INDEX, &bytes, "application/json", cond)? {
            Put::Stored(_) => break,
            Put::PreconditionFailed if attempt < 3 => continue,
            Put::PreconditionFailed => {
                return Err(dist(format!(
                    "{RUNTIME_INDEX} kept changing under this push; the files are published, \
                     run it again to record them"
                )))
            }
        }
    }
    out.latest = published_latest.clone();

    // 3. The pointer install.sh reads: the version string, nothing to parse.
    let cond = match sink.get(RUNTIME_LATEST)? {
        Some((raw, etag)) => {
            if String::from_utf8_lossy(&raw).trim() == published_latest {
                Cond::None
            } else {
                Cond::IfMatch(etag)
            }
        }
        None => Cond::IfNoneMatchAny,
    };
    if cond != Cond::None {
        if let Put::PreconditionFailed = put_bytes(
            sink,
            RUNTIME_LATEST,
            format!("{published_latest}\n").as_bytes(),
            "text/plain",
            cond,
        )? {
            return Err(dist(format!(
                "{RUNTIME_LATEST} changed under this push; the index is published, run it \
                 again to move the pointer"
            )));
        }
    }

    // 4. The installer. Not versioned: it reads `latest`.
    if let Some(script) = &opts.install_script {
        let bytes = std::fs::read(script).map_err(|e| RuntimeError::Io {
            path: script.clone(),
            source: e,
        })?;
        if !bytes.starts_with(b"#!/bin/sh") {
            return Err(dist(format!(
                "{}: an install script starts with #!/bin/sh",
                script.display()
            )));
        }
        put_bytes(sink, INSTALL_SCRIPT, &bytes, "text/x-shellscript", Cond::None)?;
        out.install_script = true;
    }
    Ok(out)
}

/// The current UTC time as RFC 3339 seconds, with no date crate: the civil
/// date from days since the epoch (Howard Hinnant's algorithm).
fn now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    rfc3339(secs)
}

fn rfc3339(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::sink::FileSink;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "plowrt-runtime-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn release(root: &Path, version: &str, triple: &str, body: &[u8]) -> ReleaseDir {
        let dir = root.join(format!("{version}-{triple}"));
        std::fs::create_dir_all(&dir).unwrap();
        let file = format!("plowrt-{version}-{triple}.tar.gz");
        std::fs::write(dir.join(&file), body).unwrap();
        let (sha, _) = sha256_file(&dir.join(&file)).unwrap();
        std::fs::write(dir.join(format!("{file}.sha256")), format!("{sha}  {file}\n")).unwrap();
        let (semver, commit) = version.split_once('+').unwrap();
        std::fs::write(
            dir.join("release.json"),
            serde_json::to_vec(&ReleaseJson {
                version: version.into(),
                semver: semver.into(),
                commit: commit.into(),
                triple: triple.into(),
                features: vec!["cpu".into(), "dist".into()],
                file,
            })
            .unwrap(),
        )
        .unwrap();
        ReleaseDir::read(&dir).unwrap()
    }

    fn read(sink: &dyn Sink, key: &str) -> Option<Vec<u8>> {
        sink.get(key).unwrap().map(|(b, _)| b)
    }

    #[test]
    fn a_release_is_recorded_and_a_rerun_adopts_it() {
        let root = tmp("rerun");
        let sink = FileSink::new(root.join("reg"));
        let x86 = release(&root, "0.1.3+abcdef123456", "x86_64-unknown-linux-gnu", b"x86 bytes");
        let out = push_runtime(&sink, &[x86], &Options::default()).unwrap();
        assert_eq!(out.uploaded, vec![("x86_64-unknown-linux-gnu".to_string(), 9)]);
        assert_eq!(out.latest, "0.1.3+abcdef123456");
        assert_eq!(
            String::from_utf8(read(&sink, RUNTIME_LATEST).unwrap()).unwrap(),
            "0.1.3+abcdef123456\n"
        );
        let index: RuntimeIndex = serde_json::from_slice(&read(&sink, RUNTIME_INDEX).unwrap()).unwrap();
        assert_eq!(index.releases.len(), 1);
        let asset = &index.releases[0].triples["x86_64-unknown-linux-gnu"];
        assert_eq!(
            asset.file,
            "v1/runtime/0.1.3+abcdef123456/x86_64-unknown-linux-gnu/plowrt-0.1.3+abcdef123456-x86_64-unknown-linux-gnu.tar.gz"
        );
        assert!(read(&sink, &asset.file).is_some());
        assert!(index.releases[0].released.is_some());

        let again = release(&root, "0.1.3+abcdef123456", "x86_64-unknown-linux-gnu", b"x86 bytes");
        let out = push_runtime(&sink, &[again], &Options::default()).unwrap();
        assert!(out.uploaded.is_empty());
        assert_eq!(out.adopted, vec!["x86_64-unknown-linux-gnu".to_string()]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn other_bytes_under_a_published_version_are_refused() {
        let root = tmp("refuse");
        let sink = FileSink::new(root.join("reg"));
        let a = release(&root, "0.1.3+abcdef123456", "x86_64-unknown-linux-gnu", b"first");
        push_runtime(&sink, &[a], &Options::default()).unwrap();
        let b = release(&root.join("b"), "0.1.3+abcdef123456", "x86_64-unknown-linux-gnu", b"second");
        let err = push_runtime(&sink, &[b], &Options::default()).unwrap_err().to_string();
        assert!(err.contains("never rewritten"), "{err}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn triples_join_one_row_and_keep_latest_leaves_the_pointer() {
        let root = tmp("triples");
        let sink = FileSink::new(root.join("reg"));
        let x86 = release(&root, "0.1.3+abcdef123456", "x86_64-unknown-linux-gnu", b"x86");
        push_runtime(&sink, &[x86], &Options::default()).unwrap();
        let mac = release(&root, "0.1.3+abcdef123456", "aarch64-apple-darwin", b"mac");
        push_runtime(&sink, &[mac], &Options::default()).unwrap();
        let old = release(&root, "0.1.2+000000000000", "x86_64-unknown-linux-gnu", b"old");
        let out = push_runtime(
            &sink,
            &[old],
            &Options {
                keep_latest: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(out.latest, "0.1.3+abcdef123456");
        let index: RuntimeIndex = serde_json::from_slice(&read(&sink, RUNTIME_INDEX).unwrap()).unwrap();
        assert_eq!(index.releases.len(), 2);
        assert_eq!(index.releases[0].version, "0.1.2+000000000000");
        let newest = index.releases.iter().find(|r| r.version == "0.1.3+abcdef123456").unwrap();
        assert_eq!(newest.triples.len(), 2);
        assert_eq!(index.latest, "0.1.3+abcdef123456");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_dirty_or_tampered_release_is_refused_before_any_write() {
        let root = tmp("dirty");
        let dir = root.join("rel");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("plowrt.tar.gz"), b"x").unwrap();
        std::fs::write(dir.join("plowrt.tar.gz.sha256"), "deadbeef  plowrt.tar.gz\n").unwrap();
        std::fs::write(
            dir.join("release.json"),
            r#"{"version":"0.1.3+abc-dirty","semver":"0.1.3","commit":"abc-dirty","triple":"x86_64-unknown-linux-gnu","file":"plowrt.tar.gz"}"#,
        )
        .unwrap();
        let err = ReleaseDir::read(&dir).unwrap_err().to_string();
        assert!(err.contains("clean tree"), "{err}");
        std::fs::write(
            dir.join("release.json"),
            r#"{"version":"0.1.3+abcdef123456","semver":"0.1.3","commit":"abcdef123456","triple":"x86_64-unknown-linux-gnu","file":"plowrt.tar.gz"}"#,
        )
        .unwrap();
        let err = ReleaseDir::read(&dir).unwrap_err().to_string();
        assert!(err.contains("says deadbeef"), "{err}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn install_script_lands_at_the_root_and_must_be_sh() {
        let root = tmp("install");
        let sink = FileSink::new(root.join("reg"));
        let rel = release(&root, "0.1.3+abcdef123456", "x86_64-unknown-linux-gnu", b"x");
        let script = root.join("install.sh");
        std::fs::write(&script, "#!/bin/sh\necho hi\n").unwrap();
        let out = push_runtime(
            &sink,
            &[rel],
            &Options {
                install_script: Some(script.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(out.install_script);
        assert_eq!(read(&sink, INSTALL_SCRIPT).unwrap(), b"#!/bin/sh\necho hi\n");
        std::fs::write(&script, "#!/bin/bash\n").unwrap();
        let rel = release(&root, "0.1.3+abcdef123456", "x86_64-unknown-linux-gnu", b"x");
        let err = push_runtime(
            &sink,
            &[rel],
            &Options {
                install_script: Some(script),
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("#!/bin/sh"), "{err}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rfc3339_is_right_at_known_instants() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(1_793_404_800), "2026-10-31T00:00:00Z");
    }
}
