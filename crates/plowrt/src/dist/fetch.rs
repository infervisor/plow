//! Fetching bytes from a registry.
//!
//! Two transports. `file://` is unconditional: it is what an air-gapped mirror
//! and every test uses, and it needs no dependency. HTTPS lives behind the
//! `dist` feature so the serving binary can be built without an HTTP client at
//! all — `serve` performs no network I/O, and CI pins that by asserting no
//! HTTP/TLS crate in the serve-only feature set.
//!
//! Blobs are published zstd-compressed. The digest in a manifest always names
//! the UNCOMPRESSED bytes, so content addressing, dedup and verification are
//! unaffected by how a blob travelled; a client that fetches `<blob>.zst`
//! decompresses it and checks it against the same digest. Measured on the
//! GLM-5.3 bundle, transport compression is 406 MB -> ~57 MB.

use crate::{Result, RuntimeError};

/// Where a registry's bytes come from.
pub trait Fetch: Send + Sync {
    /// Read one path relative to the registry root. `Ok(None)` for "not there",
    /// which is how an optional `.zst` sibling is probed.
    fn get(&self, path: &str) -> Result<Option<Vec<u8>>>;

    /// Human description, for error messages.
    fn describe(&self) -> String;
}

/// A registry served straight out of a directory.
///
/// Also the shape of an air-gapped mirror: `publish_dist.py` writes exactly this
/// tree, so an `rsync` of it is a complete registry.
pub struct FileFetch {
    root: std::path::PathBuf,
}

impl FileFetch {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        FileFetch { root: root.into() }
    }
}

impl Fetch for FileFetch {
    fn get(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let p = self.root.join(path);
        match std::fs::read(&p) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(RuntimeError::Io { path: p, source }),
        }
    }

    fn describe(&self) -> String {
        format!("file://{}", self.root.display())
    }
}

#[cfg(feature = "dist")]
pub struct HttpFetch {
    base: String,
    agent: ureq::Agent,
}

#[cfg(feature = "dist")]
impl HttpFetch {
    pub fn new(base: impl Into<String>) -> Self {
        let base = base.into();
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(20))
            // No read timeout: a 4.5 GB derived sidecar over a slow link is a
            // long single request, and killing it mid-transfer would leave the
            // caller retrying from zero.
            .build();
        HttpFetch { base, agent }
    }
}

#[cfg(feature = "dist")]
impl Fetch for HttpFetch {
    fn get(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let url = format!("{}/{path}", self.base.trim_end_matches('/'));
        match self.agent.get(&url).call() {
            Ok(resp) => {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf)
                    .map_err(|e| RuntimeError::Dist(format!("{url}: body read failed: {e}")))?;
                Ok(Some(buf))
            }
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(ureq::Error::Status(code, resp)) => Err(RuntimeError::Dist(format!(
                "{url}: HTTP {code} {}",
                resp.status_text()
            ))),
            Err(e) => Err(RuntimeError::Dist(format!("{url}: {e}"))),
        }
    }

    fn describe(&self) -> String {
        self.base.clone()
    }
}

/// Build a transport for a registry string.
///
/// `file:///path` or a bare absolute path selects the directory transport;
/// anything else is HTTPS, which needs the `dist` feature. The error when it is
/// absent names the build rather than the URL, because that is the actual fault.
pub fn transport(registry: &str) -> Result<Box<dyn Fetch>> {
    if let Some(path) = registry.strip_prefix("file://") {
        return Ok(Box::new(FileFetch::new(path)));
    }
    if registry.starts_with('/') {
        return Ok(Box::new(FileFetch::new(registry)));
    }
    #[cfg(feature = "dist")]
    {
        let base = if registry.starts_with("http://") || registry.starts_with("https://") {
            registry.to_string()
        } else {
            format!("https://{registry}")
        };
        Ok(Box::new(HttpFetch::new(base)))
    }
    #[cfg(not(feature = "dist"))]
    {
        Err(RuntimeError::Dist(format!(
            "cannot reach {registry}: this plowrt was built without the `dist` feature, so it has \
             no HTTP client. Serving reads only the local store; use a `file://` registry, or \
             fetch with a plowrt built with `--features dist`."
        )))
    }
}

/// Fetch one blob by digest, preferring the compressed sibling.
///
/// The digest names the uncompressed bytes either way, so a mirror may carry
/// `.zst`, plain, or both, and a client is correct against all three. The
/// caller verifies against the same digest regardless of how it travelled.
pub fn blob(f: &dyn Fetch, digest: &super::Digest) -> Result<Vec<u8>> {
    let plain = format!("v1/blobs/sha256/{digest}");
    let zst = format!("{plain}.zst");
    if let Some(compressed) = f.get(&zst)? {
        return decompress(&compressed, digest);
    }
    match f.get(&plain)? {
        Some(b) => Ok(b),
        None => Err(RuntimeError::Dist(format!(
            "{}: blob {digest} is not published (tried {zst} and {plain})",
            f.describe()
        ))),
    }
}

#[cfg(feature = "dist")]
fn decompress(compressed: &[u8], digest: &super::Digest) -> Result<Vec<u8>> {
    zstd::stream::decode_all(compressed)
        .map_err(|e| RuntimeError::Dist(format!("blob {digest}: zstd stream is damaged: {e}")))
}

#[cfg(not(feature = "dist"))]
fn decompress(_compressed: &[u8], digest: &super::Digest) -> Result<Vec<u8>> {
    Err(RuntimeError::Dist(format!(
        "blob {digest} is published zstd-compressed, but this plowrt was built without the \
         `dist` feature and cannot decompress it"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Digest;

    fn tmp() -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "plow-fetch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(p.join("v1/blobs/sha256")).unwrap();
        p
    }

    #[test]
    fn a_directory_is_a_registry() {
        let root = tmp();
        std::fs::write(root.join("v1/hello"), b"world").unwrap();
        let f = FileFetch::new(&root);
        assert_eq!(f.get("v1/hello").unwrap().as_deref(), Some(&b"world"[..]));
        assert!(
            f.get("v1/absent").unwrap().is_none(),
            "missing is None, not an error"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_plain_blob_is_fetched_by_digest() {
        let root = tmp();
        let bytes = b"the packet".to_vec();
        let d = Digest::of(&bytes);
        std::fs::write(root.join(format!("v1/blobs/sha256/{d}")), &bytes).unwrap();
        let f = FileFetch::new(&root);
        assert_eq!(blob(&f, &d).unwrap(), bytes);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_missing_blob_names_both_paths_it_tried() {
        let root = tmp();
        let d = Digest::of(b"absent");
        let err = blob(&FileFetch::new(&root), &d).unwrap_err().to_string();
        assert!(err.contains(".zst"), "{err}");
        assert!(err.contains(d.as_str()), "{err}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_file_registry_is_selected_by_prefix_or_absolute_path() {
        assert!(transport("file:///tmp/x")
            .unwrap()
            .describe()
            .starts_with("file://"));
        assert!(transport("/tmp/x")
            .unwrap()
            .describe()
            .starts_with("file://"));
    }

    #[cfg(feature = "dist")]
    #[test]
    fn a_zstd_blob_decompresses_to_its_digest() {
        let root = tmp();
        let bytes = vec![42u8; 100_000];
        let d = Digest::of(&bytes);
        let zst = zstd::stream::encode_all(&bytes[..], 19).unwrap();
        assert!(
            zst.len() < bytes.len() / 10,
            "the fixture should actually compress"
        );
        std::fs::write(root.join(format!("v1/blobs/sha256/{d}.zst")), &zst).unwrap();

        // The digest names the UNCOMPRESSED bytes, so the client is correct
        // whether or not the mirror compressed.
        assert_eq!(blob(&FileFetch::new(&root), &d).unwrap(), bytes);
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(feature = "dist")]
    #[test]
    fn a_damaged_zstd_stream_is_reported_not_returned() {
        let root = tmp();
        let d = Digest::of(b"whatever");
        std::fs::write(root.join(format!("v1/blobs/sha256/{d}.zst")), b"not zstd").unwrap();
        let err = blob(&FileFetch::new(&root), &d).unwrap_err().to_string();
        assert!(err.contains("zstd stream is damaged"), "{err}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(feature = "dist")]
    #[test]
    fn a_bare_host_becomes_https() {
        assert_eq!(
            transport("dist.infervisor.ai").unwrap().describe(),
            "https://dist.infervisor.ai"
        );
        assert_eq!(
            transport("https://mirror.internal").unwrap().describe(),
            "https://mirror.internal"
        );
    }

    #[cfg(not(feature = "dist"))]
    #[test]
    fn without_the_dist_feature_a_network_registry_names_the_build() {
        // Matched rather than `unwrap_err`, which would need `Debug` on
        // `Box<dyn Fetch>` — a bound the trait has no reason to carry.
        let err = match transport("dist.infervisor.ai") {
            Ok(_) => panic!("a build with no HTTP client reached a network registry"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("without the `dist` feature"), "{err}");
    }
}
