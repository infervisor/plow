//! Where `plowrt push` writes a registry: a directory, or an S3-compatible
//! bucket (Cloudflare R2). The bucket IS the registry — the same `v1/` tree
//! [`super::fetch`] reads — so publishing needs no service in between, and a
//! portal can read the same objects.
//!
//! Every write is conditional. A content-addressed object (a blob, an objset
//! manifest, a bundle manifest under its immutable `<label>@g<n>` name) is
//! written with `If-None-Match: *`, so it is never overwritten and two
//! publishers racing on one generation see the loser refused. A mutable
//! object (the model index, the catalog) is written with `If-Match: <etag>`
//! from the read it was derived from, so a concurrent update is never lost.
//!
//! SigV4 is implemented here rather than through an SDK: one PUT/GET/HEAD
//! path over the `ureq` agent `dist` already links, with `sha2` for the
//! digests. An S3 SDK would bring a second HTTP stack and an async runtime
//! into a publish-side tool.

use std::io::Read;

use crate::{Result, RuntimeError};

/// The precondition a write carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cond {
    None,
    /// Create only: refused when the key exists.
    IfNoneMatchAny,
    /// Replace only this version: refused when the current entity tag differs.
    IfMatch(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Put {
    /// Stored; the entity tag the store now reports for the key.
    Stored(String),
    /// The precondition did not hold; nothing was written.
    PreconditionFailed,
}

pub trait Sink: Send + Sync {
    /// The entity tag of `key`, or `None` when it is absent.
    fn head(&self, key: &str) -> Result<Option<String>>;
    /// The bytes and entity tag of `key`.
    fn get(&self, key: &str) -> Result<Option<(Vec<u8>, String)>>;
    fn put(
        &self,
        key: &str,
        body: &mut dyn Read,
        len: u64,
        content_type: &str,
        cond: Cond,
    ) -> Result<Put>;
    fn describe(&self) -> String;
}

fn dist(msg: String) -> RuntimeError {
    RuntimeError::Dist(msg)
}

/// A registry in a directory: an air-gapped mirror, or a test. The entity
/// tag is the content's sha256.
pub struct FileSink {
    root: std::path::PathBuf,
}

impl FileSink {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        FileSink { root: root.into() }
    }

    fn path(&self, key: &str) -> Result<std::path::PathBuf> {
        if !plow_asset::dist::is_safe_relative_path(key) {
            return Err(dist(format!("{key}: not a relative key")));
        }
        Ok(self.root.join(key))
    }
}

impl Sink for FileSink {
    fn head(&self, key: &str) -> Result<Option<String>> {
        Ok(self.get(key)?.map(|(_, etag)| etag))
    }

    fn get(&self, key: &str) -> Result<Option<(Vec<u8>, String)>> {
        let p = self.path(key)?;
        match std::fs::read(&p) {
            Ok(b) => {
                let etag = super::Digest::of(&b).as_str().to_string();
                Ok(Some((b, etag)))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(RuntimeError::Io { path: p, source }),
        }
    }

    fn put(
        &self,
        key: &str,
        body: &mut dyn Read,
        _len: u64,
        _content_type: &str,
        cond: Cond,
    ) -> Result<Put> {
        let p = self.path(key)?;
        match &cond {
            Cond::None => {}
            Cond::IfNoneMatchAny => {
                if p.exists() {
                    return Ok(Put::PreconditionFailed);
                }
            }
            Cond::IfMatch(etag) => {
                if self.head(key)?.as_deref() != Some(etag.as_str()) {
                    return Ok(Put::PreconditionFailed);
                }
            }
        }
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|source| RuntimeError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        // Staged and renamed: a reader of a mirror must never see a partial
        // object under a name that promises the whole thing.
        let tmp = p.with_extension(format!("part{}", std::process::id()));
        let io = |path: &std::path::Path| {
            let path = path.to_path_buf();
            move |source| RuntimeError::Io { path, source }
        };
        let mut out = std::fs::File::create(&tmp).map_err(io(&tmp))?;
        let mut h = <sha2::Sha256 as sha2::Digest>::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let k = body.read(&mut buf).map_err(io(&tmp))?;
            if k == 0 {
                break;
            }
            sha2::Digest::update(&mut h, &buf[..k]);
            std::io::Write::write_all(&mut out, &buf[..k]).map_err(io(&tmp))?;
        }
        drop(out);
        std::fs::rename(&tmp, &p).map_err(io(&p))?;
        Ok(Put::Stored(format!("{:x}", sha2::Digest::finalize(h))))
    }

    fn describe(&self) -> String {
        format!("file://{}", self.root.display())
    }
}

/// Static credentials for an S3-compatible endpoint.
#[derive(Clone)]
pub struct Credentials {
    pub access_key: String,
    pub secret_key: String,
}

/// An S3-compatible bucket addressed path-style: `<endpoint>/<bucket>/<key>`.
#[cfg(feature = "dist")]
pub struct S3Sink {
    endpoint: String,
    host: String,
    bucket: String,
    region: String,
    creds: Credentials,
    agent: ureq::Agent,
}

#[cfg(feature = "dist")]
impl S3Sink {
    pub fn new(endpoint: &str, bucket: &str, region: &str, creds: Credentials) -> Result<Self> {
        let endpoint = endpoint.trim_end_matches('/').to_string();
        let host = endpoint
            .strip_prefix("https://")
            .or_else(|| endpoint.strip_prefix("http://"))
            .ok_or_else(|| dist(format!("{endpoint}: an S3 endpoint is an http(s) URL")))?
            .to_string();
        if bucket.is_empty() || bucket.contains('/') {
            return Err(dist(format!("{bucket:?}: not a bucket name")));
        }
        Ok(S3Sink {
            endpoint,
            host,
            bucket: bucket.to_string(),
            region: region.to_string(),
            creds,
            agent: ureq::AgentBuilder::new()
                .timeout_connect(std::time::Duration::from_secs(20))
                .build(),
        })
    }

    fn path(&self, key: &str) -> Result<String> {
        if !plow_asset::dist::is_safe_relative_path(key) {
            return Err(dist(format!("{key}: not a relative key")));
        }
        Ok(format!("/{}/{key}", self.bucket))
    }

    /// A signed request: the URL and every header the signature covers.
    fn signed(&self, method: &str, path: &str, extra: &[(&str, &str)]) -> Vec<(String, String)> {
        let now = chrono::Utc::now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let mut headers: Vec<(String, String)> = vec![
            ("host".into(), self.host.clone()),
            ("x-amz-content-sha256".into(), UNSIGNED_PAYLOAD.into()),
            ("x-amz-date".into(), amz_date),
        ];
        for (k, v) in extra {
            headers.push((k.to_lowercase(), v.to_string()));
        }
        let auth = sign_v4(method, path, &headers, &self.region, &self.creds);
        headers.push(("authorization".into(), auth));
        headers
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.endpoint, uri_encode(path, true))
    }
}

#[cfg(feature = "dist")]
fn etag_of(resp: &ureq::Response) -> String {
    resp.header("etag").unwrap_or_default().to_string()
}

#[cfg(feature = "dist")]
fn status_err(url: &str, e: ureq::Error) -> RuntimeError {
    match e {
        ureq::Error::Status(code, resp) => dist(format!(
            "{url}: HTTP {code} {}",
            resp.into_string().unwrap_or_default().trim()
        )),
        ureq::Error::Transport(t) => dist(format!(
            "{url}: {} {}",
            t.kind(),
            t.message().unwrap_or_default()
        )),
    }
}

#[cfg(feature = "dist")]
impl Sink for S3Sink {
    fn head(&self, key: &str) -> Result<Option<String>> {
        let path = self.path(key)?;
        let url = self.url(&path);
        let mut req = self.agent.head(&url);
        for (k, v) in self.signed("HEAD", &path, &[]) {
            if k != "host" {
                req = req.set(&k, &v);
            }
        }
        match req.call() {
            Ok(resp) => Ok(Some(etag_of(&resp))),
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(e) => Err(status_err(&url, e)),
        }
    }

    fn get(&self, key: &str) -> Result<Option<(Vec<u8>, String)>> {
        let path = self.path(key)?;
        let url = self.url(&path);
        let mut req = self.agent.get(&url);
        for (k, v) in self.signed("GET", &path, &[]) {
            if k != "host" {
                req = req.set(&k, &v);
            }
        }
        match req.call() {
            Ok(resp) => {
                let etag = etag_of(&resp);
                let mut buf = Vec::new();
                resp.into_reader()
                    .read_to_end(&mut buf)
                    .map_err(|e| dist(format!("{url}: body read failed: {e}")))?;
                Ok(Some((buf, etag)))
            }
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(e) => Err(status_err(&url, e)),
        }
    }

    fn put(
        &self,
        key: &str,
        body: &mut dyn Read,
        len: u64,
        content_type: &str,
        cond: Cond,
    ) -> Result<Put> {
        let path = self.path(key)?;
        let url = self.url(&path);
        let extra: Vec<(&str, &str)> = match &cond {
            Cond::None => vec![],
            Cond::IfNoneMatchAny => vec![("if-none-match", "*")],
            Cond::IfMatch(etag) => vec![("if-match", etag.as_str())],
        };
        let mut req = self
            .agent
            .put(&url)
            .set("content-type", content_type)
            .set("content-length", &len.to_string());
        for (k, v) in self.signed("PUT", &path, &extra) {
            if k != "host" {
                req = req.set(&k, &v);
            }
        }
        match req.send(body) {
            Ok(resp) => Ok(Put::Stored(etag_of(&resp))),
            Err(ureq::Error::Status(412, _)) => Ok(Put::PreconditionFailed),
            Err(e) => Err(status_err(&url, e)),
        }
    }

    fn describe(&self) -> String {
        format!("{}/{}", self.endpoint, self.bucket)
    }
}

/// Build a sink for a destination string.
///
/// * `file:///dir` or `/dir` — a directory.
/// * `r2://<account-id>/<bucket>` — Cloudflare R2, region `auto`.
/// * `s3://<host>/<bucket>` — any S3-compatible endpoint over HTTPS; the
///   region is `PLOW_S3_REGION`, else `auto`.
///
/// Bucket credentials come from `PLOW_R2_ACCESS_KEY_ID` and
/// `PLOW_R2_SECRET_ACCESS_KEY` for both bucket forms.
pub fn sink(to: &str, creds: Option<Credentials>, region: Option<&str>) -> Result<Box<dyn Sink>> {
    if let Some(path) = to.strip_prefix("file://") {
        return Ok(Box::new(FileSink::new(path)));
    }
    if to.starts_with('/') {
        return Ok(Box::new(FileSink::new(to)));
    }
    let (endpoint, bucket) = if let Some(rest) = to.strip_prefix("r2://") {
        let (account, bucket) = rest
            .split_once('/')
            .ok_or_else(|| dist(format!("{to}: expected r2://<account-id>/<bucket>")))?;
        (
            format!("https://{account}.r2.cloudflarestorage.com"),
            bucket,
        )
    } else if let Some(rest) = to.strip_prefix("s3://") {
        let (host, bucket) = rest
            .split_once('/')
            .ok_or_else(|| dist(format!("{to}: expected s3://<host>/<bucket>")))?;
        (format!("https://{host}"), bucket)
    } else {
        return Err(dist(format!(
            "{to}: a push destination is file:///dir, r2://<account-id>/<bucket> or \
             s3://<host>/<bucket>"
        )));
    };
    let bucket = bucket.trim_end_matches('/');
    #[cfg(feature = "dist")]
    {
        let creds = creds.ok_or_else(|| {
            dist(
                "set PLOW_R2_ACCESS_KEY_ID and PLOW_R2_SECRET_ACCESS_KEY to a token allowed to \
                 write the bucket"
                    .into(),
            )
        })?;
        Ok(Box::new(S3Sink::new(
            &endpoint,
            bucket,
            region.unwrap_or("auto"),
            creds,
        )?))
    }
    #[cfg(not(feature = "dist"))]
    {
        let _ = (endpoint, bucket, creds, region);
        Err(dist(format!(
            "cannot push to {to}: this plowrt was built without the `dist` feature"
        )))
    }
}

// --- AWS Signature Version 4 ---------------------------------------------------

const UNSIGNED_PAYLOAD: &str = "UNSIGNED-PAYLOAD";

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    use sha2::Digest as _;
    let mut k = [0u8; 64];
    if key.len() > 64 {
        k[..32].copy_from_slice(&sha2::Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let ipad: Vec<u8> = k.iter().map(|b| b ^ 0x36).collect();
    let opad: Vec<u8> = k.iter().map(|b| b ^ 0x5c).collect();
    let inner = sha2::Sha256::new()
        .chain_update(&ipad)
        .chain_update(msg)
        .finalize();
    sha2::Sha256::new()
        .chain_update(&opad)
        .chain_update(inner)
        .finalize()
        .into()
}

/// RFC 3986 unreserved characters pass; everything else is `%XX`. `/` is
/// kept when the input is a path, as S3's canonical URI requires.
fn uri_encode(s: &str, keep_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        let keep = b.is_ascii_alphanumeric()
            || matches!(b, b'-' | b'_' | b'.' | b'~')
            || (keep_slash && b == b'/');
        if keep {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The `Authorization` value for one request. `headers` are the headers the
/// signature covers, lowercase names, in any order; `x-amz-date` must be
/// among them. Pure, so it is testable against the published AWS vector.
pub fn sign_v4(
    method: &str,
    path: &str,
    headers: &[(String, String)],
    region: &str,
    creds: &Credentials,
) -> String {
    let mut hs: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| (k.to_lowercase(), v.trim().to_string()))
        .collect();
    hs.sort();
    let signed_headers = hs
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let canonical_headers: String = hs.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let payload_hash = hs
        .iter()
        .find(|(k, _)| k == "x-amz-content-sha256")
        .map(|(_, v)| v.as_str())
        .unwrap_or(UNSIGNED_PAYLOAD);
    let amz_date = hs
        .iter()
        .find(|(k, _)| k == "x-amz-date")
        .map(|(_, v)| v.as_str())
        .unwrap_or_default();
    let date = &amz_date[..amz_date.len().min(8)];
    let canonical = format!(
        "{method}\n{}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}",
        uri_encode(path, true)
    );
    let scope = format!("{date}/{region}/s3/aws4_request");
    let to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        sha256_hex(canonical.as_bytes())
    );
    let k_date = hmac_sha256(
        format!("AWS4{}", creds.secret_key).as_bytes(),
        date.as_bytes(),
    );
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, b"s3");
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hmac_sha256(&k_signing, to_sign.as_bytes());
    let signature: String = signature.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, \
         Signature={signature}",
        creds.access_key
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "plow-sink-{}-{}-{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    fn put(s: &dyn Sink, key: &str, body: &[u8], cond: Cond) -> Put {
        s.put(key, &mut &body[..], body.len() as u64, "text/plain", cond)
            .unwrap()
    }

    // The example from "Signature Calculations for the Authorization Header"
    // in the S3 documentation (GET Object): the one public vector.
    #[test]
    fn sigv4_reproduces_the_aws_example() {
        let creds = Credentials {
            access_key: "AKIAIOSFODNN7EXAMPLE".into(),
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
        };
        let headers = vec![
            (
                "host".to_string(),
                "examplebucket.s3.amazonaws.com".to_string(),
            ),
            ("range".to_string(), "bytes=0-9".to_string()),
            (
                "x-amz-content-sha256".to_string(),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
            ),
            ("x-amz-date".to_string(), "20130524T000000Z".to_string()),
        ];
        let auth = sign_v4("GET", "/test.txt", &headers, "us-east-1", &creds);
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, \
             SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, \
             Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
        );
    }

    #[test]
    fn a_manifest_name_is_encoded_once_with_the_slashes_kept() {
        assert_eq!(
            uri_encode("/b/v1/infervisor/k3/manifests/gfx942-tp8@g3", true),
            "/b/v1/infervisor/k3/manifests/gfx942-tp8%40g3"
        );
        assert_eq!(uri_encode("a b/c", false), "a%20b%2Fc");
    }

    #[test]
    fn create_only_refuses_an_existing_key_and_if_match_needs_the_current_tag() {
        let root = tmp("cond");
        let s = FileSink::new(&root);
        assert!(s.head("v1/x").unwrap().is_none());
        let Put::Stored(tag) = put(&s, "v1/x", b"one", Cond::IfNoneMatchAny) else {
            panic!("first write is refused");
        };
        assert_eq!(s.head("v1/x").unwrap().as_deref(), Some(tag.as_str()));
        assert_eq!(
            put(&s, "v1/x", b"two", Cond::IfNoneMatchAny),
            Put::PreconditionFailed,
            "an immutable object is never overwritten"
        );
        assert_eq!(s.get("v1/x").unwrap().unwrap().0, b"one");
        assert_eq!(
            put(&s, "v1/x", b"two", Cond::IfMatch("stale".into())),
            Put::PreconditionFailed,
            "a stale read must not overwrite a newer write"
        );
        assert!(matches!(
            put(&s, "v1/x", b"two", Cond::IfMatch(tag)),
            Put::Stored(_)
        ));
        assert_eq!(s.get("v1/x").unwrap().unwrap().0, b"two");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_key_cannot_leave_the_registry() {
        let root = tmp("escape");
        let s = FileSink::new(&root);
        let e = s
            .put("../x", &mut &b"a"[..], 1, "text/plain", Cond::None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("not a relative key"), "{e}");
        let e = match sink("ftp://x/y", None, None) {
            Ok(_) => panic!("an unknown scheme is refused"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("file:///dir, r2://"), "{e}");
        let _ = std::fs::remove_dir_all(root);
    }
}
