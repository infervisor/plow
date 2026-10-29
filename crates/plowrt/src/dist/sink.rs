//! Where `plowrt push` writes a registry: a directory, or a bucket
//! (Cloudflare R2 by default; any S3 bucket through the same client). The
//! bucket IS the registry — the same `v1/` tree [`super::fetch`] reads — so
//! publishing needs no service in between, and a portal can read the same
//! objects.
//!
//! Every write is conditional. A content-addressed object (a blob, an objset
//! manifest, a bundle manifest under its immutable `<label>@g<n>` name) is
//! written with `If-None-Match: *`, so it is never overwritten and two
//! publishers racing on one generation see the loser refused. A mutable
//! object (the model index, the catalog) is written with `If-Match: <etag>`
//! from the read it was derived from, so a concurrent update is never lost.

use std::io::Read;

use crate::{Result, RuntimeError};

/// The precondition a write carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cond {
    None,
    /// The object must not exist.
    IfNoneMatchAny,
    /// The object must still carry this entity tag.
    IfMatch(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Put {
    /// Stored; the entity tag the store assigned.
    Stored(String),
    /// The precondition did not hold: someone else wrote first.
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
///
/// A directory has no conditional write, so one is built: every writer
/// stages under a name only it uses, then takes `<root>/.lock` for the
/// condition check and the commit together. Create-only commits are a
/// `hard_link`, which the filesystem refuses if the name exists, so even a
/// writer that skipped the lock cannot overwrite an immutable object.
pub struct FileSink {
    root: std::path::PathBuf,
}

/// Held across a condition check and its commit; see [`FileSink`].
struct DirLock(#[allow(dead_code)] std::fs::File);

static STAGE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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

    fn io(path: &std::path::Path, source: std::io::Error) -> RuntimeError {
        RuntimeError::Io {
            path: path.to_path_buf(),
            source,
        }
    }

    fn lock(&self) -> Result<DirLock> {
        std::fs::create_dir_all(&self.root).map_err(|e| Self::io(&self.root, e))?;
        let p = self.root.join(".lock");
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&p)
            .map_err(|e| Self::io(&p, e))?;
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // SAFETY: `flock` on a descriptor this process owns.
            if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) } != 0 {
                return Err(Self::io(&p, std::io::Error::last_os_error()));
            }
        }
        Ok(DirLock(f))
    }

    /// The sha256 of a file, streamed; `None` when it is absent.
    fn etag_of(p: &std::path::Path) -> Result<Option<String>> {
        let mut f = match std::fs::File::open(p) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(Self::io(p, e)),
        };
        let mut h = <sha2::Sha256 as sha2::Digest>::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let k = f.read(&mut buf).map_err(|e| Self::io(p, e))?;
            if k == 0 {
                break;
            }
            sha2::Digest::update(&mut h, &buf[..k]);
        }
        Ok(Some(format!("{:x}", sha2::Digest::finalize(h))))
    }
}

impl Sink for FileSink {
    fn head(&self, key: &str) -> Result<Option<String>> {
        Self::etag_of(&self.path(key)?)
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
        let parent = p.parent().unwrap_or(&self.root);
        std::fs::create_dir_all(parent).map_err(|e| Self::io(parent, e))?;
        let staged = parent.join(format!(
            ".{}.{}-{}.part",
            p.file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default(),
            std::process::id(),
            STAGE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staged)
            .map_err(|e| Self::io(&staged, e))?;
        let mut h = <sha2::Sha256 as sha2::Digest>::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let k = body
                .read(&mut buf)
                .map_err(|e| dist(format!("{key}: body read failed: {e}")))?;
            if k == 0 {
                break;
            }
            sha2::Digest::update(&mut h, &buf[..k]);
            std::io::Write::write_all(&mut f, &buf[..k]).map_err(|e| Self::io(&staged, e))?;
        }
        drop(f);
        let etag = format!("{:x}", sha2::Digest::finalize(h));

        let _lock = self.lock()?;
        let committed = match &cond {
            Cond::None => std::fs::rename(&staged, &p).map(|()| true),
            Cond::IfNoneMatchAny => match std::fs::hard_link(&staged, &p) {
                Ok(()) => Ok(true),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
                Err(e) => Err(e),
            },
            Cond::IfMatch(expected) => match Self::etag_of(&p)? {
                Some(current) if current == *expected => {
                    std::fs::rename(&staged, &p).map(|()| true)
                }
                _ => Ok(false),
            },
        };
        let _ = std::fs::remove_file(&staged);
        match committed.map_err(|e| Self::io(&p, e))? {
            true => Ok(Put::Stored(etag)),
            false => Ok(Put::PreconditionFailed),
        }
    }

    fn describe(&self) -> String {
        format!("file://{}", self.root.display())
    }
}

#[cfg(feature = "dist")]
mod remote {
    use super::*;
    use object_store::path::Path as Key;
    use object_store::{
        Attribute, Attributes, ObjectStore, ObjectStoreExt as _, PutMode, PutMultipartOptions,
        PutOptions, PutPayload, UpdateVersion, WriteMultipart,
    };
    use std::sync::Arc;

    /// Bodies above this go up in parts of this size, so a checkpoint shard
    /// never sits in memory whole. R2 wants every part but the last equal.
    const PART: usize = 64 << 20;

    /// The CLI runs `push` on its multi-thread runtime, which a nested
    /// `block_on` must step out of; a test calls in with no runtime at all.
    /// Decided once, so every connection the client pools lives on one
    /// runtime.
    enum Driver {
        Ambient(tokio::runtime::Handle),
        Own(tokio::runtime::Runtime),
    }

    /// A bucket behind the `object_store` interface.
    pub struct StoreSink {
        store: Arc<dyn ObjectStore>,
        driver: Driver,
        name: String,
        part: usize,
    }

    impl StoreSink {
        pub fn new(store: Arc<dyn ObjectStore>) -> Result<Self> {
            let driver = match tokio::runtime::Handle::try_current() {
                Ok(h) => Driver::Ambient(h),
                Err(_) => Driver::Own(
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| dist(format!("tokio runtime: {e}")))?,
                ),
            };
            let name = store.to_string();
            Ok(StoreSink {
                store,
                driver,
                name,
                part: PART,
            })
        }

        #[cfg(test)]
        fn with_part_size(mut self, part: usize) -> Self {
            self.part = part;
            self
        }

        fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
            match &self.driver {
                Driver::Ambient(h) => tokio::task::block_in_place(|| h.block_on(fut)),
                Driver::Own(rt) => rt.block_on(fut),
            }
        }

        fn key(&self, key: &str) -> Result<Key> {
            if !plow_asset::dist::is_safe_relative_path(key) {
                return Err(dist(format!("{key}: not a relative key")));
            }
            Key::parse(key).map_err(|e| dist(format!("{key}: {e}")))
        }

        fn fail(&self, key: &str, e: object_store::Error) -> RuntimeError {
            dist(format!("{}/{key}: {e}", self.name))
        }

        fn attributes(content_type: &str) -> Attributes {
            let mut a = Attributes::new();
            a.insert(Attribute::ContentType, content_type.to_string().into());
            a
        }

        fn put_whole(
            &self,
            key: &str,
            k: &Key,
            body: &mut dyn Read,
            len: u64,
            content_type: &str,
            cond: Cond,
        ) -> Result<Put> {
            let mut bytes = Vec::with_capacity(len as usize);
            body.read_to_end(&mut bytes)
                .map_err(|e| dist(format!("{key}: body read failed: {e}")))?;
            let mode = match cond {
                Cond::None => PutMode::Overwrite,
                Cond::IfNoneMatchAny => PutMode::Create,
                Cond::IfMatch(e_tag) => PutMode::Update(UpdateVersion {
                    e_tag: Some(e_tag),
                    version: None,
                }),
            };
            let opts = PutOptions {
                mode,
                attributes: Self::attributes(content_type),
                ..Default::default()
            };
            match self.block_on(self.store.put_opts(k, PutPayload::from(bytes), opts)) {
                Ok(r) => Ok(Put::Stored(r.e_tag.unwrap_or_default())),
                Err(object_store::Error::AlreadyExists { .. })
                | Err(object_store::Error::Precondition { .. }) => Ok(Put::PreconditionFailed),
                Err(e) => Err(self.fail(key, e)),
            }
        }

        /// A multipart upload carries no precondition, so an immutable
        /// object is checked for first; the bytes it could race with are,
        /// by its digest, the same bytes.
        fn put_parts(
            &self,
            key: &str,
            k: &Key,
            body: &mut dyn Read,
            content_type: &str,
            cond: Cond,
        ) -> Result<Put> {
            match cond {
                Cond::None => {}
                Cond::IfNoneMatchAny => {
                    if self.head(key)?.is_some() {
                        return Ok(Put::PreconditionFailed);
                    }
                }
                Cond::IfMatch(_) => {
                    return Err(dist(format!(
                        "{key}: a conditional replace of a multipart object is not supported"
                    )))
                }
            }
            let opts = PutMultipartOptions {
                attributes: Self::attributes(content_type),
                ..Default::default()
            };
            let part = self.part;
            let result = self.block_on(async {
                let upload = self.store.put_multipart_opts(k, opts).await?;
                let mut w = WriteMultipart::new_with_chunk_size(upload, part);
                let mut buf = vec![0u8; 1 << 20];
                loop {
                    let n = body
                        .read(&mut buf)
                        .map_err(|e| object_store::Error::Generic {
                            store: "push",
                            source: Box::new(e),
                        })?;
                    if n == 0 {
                        break;
                    }
                    w.write(&buf[..n]);
                    w.wait_for_capacity(4).await?;
                }
                w.finish().await
            });
            match result {
                Ok(r) => Ok(Put::Stored(r.e_tag.unwrap_or_default())),
                Err(e) => Err(self.fail(key, e)),
            }
        }
    }

    impl Sink for StoreSink {
        fn head(&self, key: &str) -> Result<Option<String>> {
            let k = self.key(key)?;
            match self.block_on(self.store.head(&k)) {
                Ok(m) => Ok(Some(m.e_tag.unwrap_or_default())),
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(e) => Err(self.fail(key, e)),
            }
        }

        fn get(&self, key: &str) -> Result<Option<(Vec<u8>, String)>> {
            let k = self.key(key)?;
            let got = self.block_on(async {
                let r = self.store.get(&k).await?;
                let etag = r.meta.e_tag.clone().unwrap_or_default();
                Ok::<_, object_store::Error>((Vec::from(r.bytes().await?), etag))
            });
            match got {
                Ok(v) => Ok(Some(v)),
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(e) => Err(self.fail(key, e)),
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
            let k = self.key(key)?;
            if len > self.part as u64 {
                self.put_parts(key, &k, body, content_type, cond)
            } else {
                self.put_whole(key, &k, body, len, content_type, cond)
            }
        }

        fn describe(&self) -> String {
            self.name.clone()
        }
    }

    fn env(name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.is_empty())
    }

    /// `r2://`, `r2://<bucket>` or `r2://<account-id>/<bucket>`; what the
    /// URL leaves out comes from `CLOUDFLARE_ACCOUNT_ID` and
    /// `CLOUDFLARE_R2_BUCKET`. With `CLOUDFLARE_API_TOKEN` set the writes go
    /// through Cloudflare's REST API; with `CLOUDFLARE_R2_ACCESS_KEY_ID` and
    /// `CLOUDFLARE_R2_SECRET_ACCESS_KEY` instead, through R2's S3 endpoint.
    pub fn r2(rest: &str) -> Result<Box<dyn Sink>> {
        let parts: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
        let (account, bucket) = match parts[..] {
            [] => (env("CLOUDFLARE_ACCOUNT_ID"), env("CLOUDFLARE_R2_BUCKET")),
            [b] => (env("CLOUDFLARE_ACCOUNT_ID"), Some(b.to_string())),
            [a, b] => (Some(a.to_string()), Some(b.to_string())),
            _ => {
                return Err(dist(format!(
                    "r2://{rest}: an R2 destination is r2://<account-id>/<bucket>, with no key \
                     prefix"
                )))
            }
        };
        let account = account.ok_or_else(|| {
            dist("no R2 account: r2://<account-id>/<bucket> or CLOUDFLARE_ACCOUNT_ID".into())
        })?;
        let bucket = bucket.ok_or_else(|| {
            dist("no R2 bucket: r2://<account-id>/<bucket> or CLOUDFLARE_R2_BUCKET".into())
        })?;
        let store: Arc<dyn ObjectStore> = if env("CLOUDFLARE_API_TOKEN")
            .or_else(|| env("CF_API_TOKEN"))
            .is_some()
        {
            Arc::new(
                object_store::cloudflare::CloudflareR2Builder::from_env()
                    .with_account_id(&account)
                    .with_bucket_name(&bucket)
                    .build()
                    .map_err(|e| dist(format!("r2://{account}/{bucket}: {e}")))?,
            )
        } else {
            let id = env("CLOUDFLARE_R2_ACCESS_KEY_ID").or_else(|| env("CF_R2_ACCESS_KEY_ID"));
            let secret =
                env("CLOUDFLARE_R2_SECRET_ACCESS_KEY").or_else(|| env("CF_R2_SECRET_ACCESS_KEY"));
            let (Some(id), Some(secret)) = (id, secret) else {
                return Err(dist(
                    "no R2 credentials: CLOUDFLARE_API_TOKEN (REST API), or \
                     CLOUDFLARE_R2_ACCESS_KEY_ID + CLOUDFLARE_R2_SECRET_ACCESS_KEY (S3 endpoint)"
                        .into(),
                ));
            };
            Arc::new(
                object_store::aws::AmazonS3Builder::new()
                    .with_endpoint(format!("https://{account}.r2.cloudflarestorage.com"))
                    .with_region("auto")
                    .with_bucket_name(&bucket)
                    .with_access_key_id(id)
                    .with_secret_access_key(secret)
                    .with_conditional_put(object_store::aws::S3ConditionalPut::ETagMatch)
                    .build()
                    .map_err(|e| dist(format!("r2://{account}/{bucket}: {e}")))?,
            )
        };
        Ok(Box::new(StoreSink::new(store)?))
    }

    /// `s3://<bucket>`, with the standard `AWS_*` environment (region,
    /// credentials, `AWS_ENDPOINT` for another S3-compatible service).
    pub fn s3(to: &str) -> Result<Box<dyn Sink>> {
        if to
            .trim_start_matches("s3://")
            .trim_end_matches('/')
            .contains('/')
        {
            return Err(dist(format!(
                "{to}: an S3 destination is s3://<bucket>, with no key prefix"
            )));
        }
        let store = object_store::aws::AmazonS3Builder::from_env()
            .with_url(to)
            .with_conditional_put(object_store::aws::S3ConditionalPut::ETagMatch)
            .build()
            .map_err(|e| dist(format!("{to}: {e}")))?;
        Ok(Box::new(StoreSink::new(Arc::new(store))?))
    }

    #[cfg(test)]
    mod tests {
        use super::super::tests::put;
        use super::*;

        fn memory(part: usize) -> StoreSink {
            StoreSink::new(Arc::new(object_store::memory::InMemory::new()))
                .unwrap()
                .with_part_size(part)
        }

        #[test]
        fn a_bucket_keeps_the_same_promises_as_a_directory() {
            let s = memory(PART);
            assert!(s.head("v1/x").unwrap().is_none());
            assert!(s.get("v1/x").unwrap().is_none());
            let Put::Stored(tag) = put(&s, "v1/x", b"one", Cond::IfNoneMatchAny) else {
                panic!("first write is refused");
            };
            assert_eq!(s.head("v1/x").unwrap().as_deref(), Some(tag.as_str()));
            assert_eq!(
                put(&s, "v1/x", b"two", Cond::IfNoneMatchAny),
                Put::PreconditionFailed
            );
            assert_eq!(s.get("v1/x").unwrap().unwrap().0, b"one");
            assert_eq!(
                put(&s, "v1/x", b"two", Cond::IfMatch("stale".into())),
                Put::PreconditionFailed
            );
            assert!(matches!(
                put(&s, "v1/x", b"two", Cond::IfMatch(tag)),
                Put::Stored(_)
            ));
            assert_eq!(s.get("v1/x").unwrap().unwrap().0, b"two");
            let e = s.head("../x").unwrap_err().to_string();
            assert!(e.contains("not a relative key"), "{e}");
        }

        #[test]
        fn a_large_body_goes_up_in_parts_and_is_still_never_overwritten() {
            let s = memory(1 << 20);
            let body: Vec<u8> = (0..(3 << 20) + 17).map(|i| (i % 251) as u8).collect();
            assert!(matches!(
                put(&s, "v1/big", &body, Cond::IfNoneMatchAny),
                Put::Stored(_)
            ));
            assert_eq!(s.get("v1/big").unwrap().unwrap().0, body);
            assert_eq!(
                put(&s, "v1/big", &body, Cond::IfNoneMatchAny),
                Put::PreconditionFailed
            );
            let e = s
                .put(
                    "v1/big",
                    &mut &body[..],
                    body.len() as u64,
                    "x",
                    Cond::IfMatch("t".into()),
                )
                .unwrap_err()
                .to_string();
            assert!(e.contains("not supported"), "{e}");
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn the_cli_runtime_is_reused_rather_than_nested() {
            let s = memory(PART);
            assert!(matches!(s.driver, Driver::Ambient(_)));
            assert!(matches!(
                put(&s, "v1/x", b"one", Cond::None),
                Put::Stored(_)
            ));
            assert_eq!(s.get("v1/x").unwrap().unwrap().0, b"one");
        }

        fn refused(r: Result<Box<dyn Sink>>) -> String {
            match r {
                Ok(s) => panic!("{} was accepted", s.describe()),
                Err(e) => e.to_string(),
            }
        }

        #[test]
        fn an_r2_destination_names_account_and_bucket_or_says_what_is_missing() {
            let e = refused(r2("a/b/c"));
            assert!(e.contains("no key prefix"), "{e}");
            // The test environment carries no Cloudflare variables.
            let e = refused(r2(""));
            assert!(
                e.contains("no R2 account") || e.contains("no R2 bucket"),
                "{e}"
            );
            let e = refused(r2("acct/bucket"));
            assert!(e.contains("no R2 credentials"), "{e}");
        }
    }
}

#[cfg(feature = "dist")]
pub use remote::StoreSink;

/// Build a sink for a destination string.
///
/// * `r2://`, `r2://<bucket>`, `r2://<account-id>/<bucket>` — Cloudflare R2,
///   the default; see [`remote::r2`] for the environment it reads.
/// * `s3://<bucket>` — any S3 bucket, from the `AWS_*` environment.
/// * `file:///dir` or `/dir` — a directory.
pub fn sink(to: &str) -> Result<Box<dyn Sink>> {
    if let Some(path) = to.strip_prefix("file://") {
        return Ok(Box::new(FileSink::new(path)));
    }
    if to.starts_with('/') {
        return Ok(Box::new(FileSink::new(to)));
    }
    #[cfg(feature = "dist")]
    {
        if let Some(rest) = to.strip_prefix("r2://") {
            return remote::r2(rest);
        }
        if to.starts_with("s3://") {
            return remote::s3(to);
        }
    }
    Err(dist(format!(
        "{to}: a destination is r2://<account-id>/<bucket>, s3://<bucket> or file:///dir"
    )))
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

    pub(super) fn put(s: &dyn Sink, key: &str, body: &[u8], cond: Cond) -> Put {
        s.put(key, &mut &body[..], body.len() as u64, "text/plain", cond)
            .unwrap()
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
    fn concurrent_writers_see_their_conditions_enforced() {
        let root = tmp("race");
        let s = std::sync::Arc::new(FileSink::new(&root));
        // Eight create-only writers of one key: exactly one lands, intact.
        let stored: Vec<Option<String>> = std::thread::scope(|sc| {
            (0..8u8)
                .map(|i| {
                    let s = std::sync::Arc::clone(&s);
                    sc.spawn(move || {
                        let body = vec![i; 300_000];
                        match put(&*s, "v1/one", &body, Cond::IfNoneMatchAny) {
                            Put::Stored(tag) => Some(tag),
                            Put::PreconditionFailed => None,
                        }
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        let winners: Vec<&String> = stored.iter().flatten().collect();
        assert_eq!(winners.len(), 1, "{stored:?}");
        let (bytes, tag) = s.get("v1/one").unwrap().unwrap();
        assert_eq!(&tag, winners[0]);
        assert_eq!(bytes.len(), 300_000);
        assert!(
            bytes.iter().all(|b| *b == bytes[0]),
            "no interleaved staging"
        );
        assert!(
            std::fs::read_dir(root.join("v1")).unwrap().count() == 1,
            "no staging file left behind"
        );

        // Eight updaters that all read the same tag: exactly one replaces it.
        put(&*s, "v1/idx", b"v0", Cond::None);
        let tag0 = s.head("v1/idx").unwrap().unwrap();
        let replaced = std::thread::scope(|sc| {
            (0..8u8)
                .map(|i| {
                    let (s, tag0) = (std::sync::Arc::clone(&s), tag0.clone());
                    sc.spawn(move || {
                        matches!(
                            put(&*s, "v1/idx", &[b'v', b'1', i], Cond::IfMatch(tag0)),
                            Put::Stored(_)
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .filter(|won| *won)
                .count()
        });
        assert_eq!(replaced, 1);
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
        let e = match sink("ftp://x/y") {
            Ok(_) => panic!("an unknown scheme is refused"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("r2://<account-id>/<bucket>"), "{e}");
        let _ = std::fs::remove_dir_all(root);
    }
}
