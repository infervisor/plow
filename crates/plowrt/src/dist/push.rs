//! `plowrt push`: publish a built variant straight into a registry.
//!
//! The registry is a bucket or a directory ([`super::sink`]); there is no
//! upload API and no archive. The tree written is exactly the one
//! [`super::pull`] reads, in the order that keeps it servable at every step:
//!
//! 1. blobs, under `v1/blobs/sha256/<digest>[.zst]` — content-addressed, so a
//!    blob already there is skipped and never rewritten;
//! 2. the objset manifest, under its content-derived id;
//! 3. the bundle manifest, under its immutable `<label>@g<n>` name, created
//!    only if absent — a second publisher numbering the same generation loses
//!    here and starts over;
//! 4. the model index, replaced only if it is still the one this run read;
//! 5. its signature, then the catalog.
//!
//! An interrupted push leaves the previous index intact and some unreferenced
//! blobs, which are harmless. A re-run of a push whose build is already the
//! newest generation writes nothing.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use plow_asset::dist::{
    Build, Bundle, Delta, Measured, ModelIndex, ObjSet, Status, Variant, INDEX_SCHEMA,
};

use super::sign::{SecretKey, Trust};
use super::sink::{Cond, Put, Sink};
use super::Digest;
use crate::{Result, RuntimeError};

fn dist(msg: String) -> RuntimeError {
    RuntimeError::Dist(msg)
}

/// What the recipe does not say about a publication.
#[derive(Clone, Debug, Default)]
pub struct Options<'a> {
    pub status: Option<Status>,
    pub description: Option<&'a str>,
    /// `<org>/<repo>` for the index; the bundle's `hf:` source when omitted.
    /// Required for a `dist` checkpoint, which names none.
    pub hf: Option<&'a str>,
    pub tok_s: Option<f64>,
    pub concurrency: Option<u32>,
    pub aliases: Vec<String>,
    /// Whether listings show the model: `Some(false)` publishes it unlisted
    /// (reachable by name, left out of `catalog.json` listings), `Some(true)`
    /// relists it, `None` keeps what the catalog row says (listed, for a new
    /// row).
    pub listed: Option<bool>,
    /// Report what would be written, and write nothing.
    pub dry_run: bool,
}

/// Where the bytes a manifest names are read from.
pub struct Sources<'a> {
    pub assets: &'a Path,
    pub objects: &'a Path,
    pub checkpoint: Option<&'a Path>,
}

/// What a push did.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub reference: String,
    pub generation: u32,
    pub variant_id: String,
    /// Blobs this run transferred, and their uncompressed size.
    pub uploaded_blobs: usize,
    pub uploaded_bytes: u64,
    pub supersedes: Option<String>,
    pub delta: Option<Delta>,
    /// Nothing was written: the newest generation is already this build.
    pub unchanged: bool,
}

/// The catalog, `catalog.json`: model names only, for discovery. `load`
/// never reads it; listings (the site, inferos, `plowrt catalog show`) do.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Catalog {
    pub models: Vec<CatalogEntry>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CatalogEntry {
    pub namespace: String,
    pub name: String,
    pub hf: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// `false` = unlisted: the model stays published and loads by name, but
    /// listings leave it out. Absent means listed, so a catalog written
    /// before the field existed, and a reader that predates it, agree.
    #[serde(default = "yes", skip_serializing_if = "is_yes")]
    pub listed: bool,
}

fn yes() -> bool {
    true
}

fn is_yes(b: &bool) -> bool {
    *b
}

/// Why `set_listed` wrote nothing.
#[derive(Clone, Debug, PartialEq)]
pub enum Listing {
    Changed,
    /// The row already said so.
    Unchanged,
}

/// Read the catalog. Absent (nothing published yet) is empty.
pub fn catalog(sink: &dyn Sink) -> Result<Catalog> {
    match sink.get("catalog.json")? {
        Some((raw, _)) => parse("catalog.json", &raw),
        None => Ok(Catalog::default()),
    }
}

/// List or unlist a published model: one conditional write on the catalog
/// row, nothing under `v1/`. A model with no row is refused — publish it
/// first; unlisting is not a way to reserve a name.
pub fn set_listed(sink: &dyn Sink, namespace: &str, name: &str, listed: bool) -> Result<Listing> {
    for _ in 0..4 {
        let (mut catalog, cond) = match sink.get("catalog.json")? {
            Some((raw, etag)) => (parse_to_rewrite::<Catalog>("catalog.json", &raw)?, Cond::IfMatch(etag)),
            None => (Catalog::default(), Cond::IfNoneMatchAny),
        };
        let Some(row) = catalog
            .models
            .iter_mut()
            .find(|m| m.namespace == namespace && m.name == name)
        else {
            return Err(dist(format!(
                "{namespace}/{name} is not in {}'s catalog: nothing published under that name",
                sink.describe()
            )));
        };
        if row.listed == listed {
            return Ok(Listing::Unchanged);
        }
        row.listed = listed;
        if let Put::Stored(_) = put_bytes(
            sink,
            "catalog.json",
            &canonical(&catalog)?,
            "application/json",
            cond,
        )? {
            return Ok(Listing::Changed);
        }
    }
    Err(dist(format!(
        "{}: catalog.json kept changing; run the command again",
        sink.describe()
    )))
}

/// Every blob a variant needs, digest → bytes.
fn inventory(bundle: &Bundle, objset: Option<&ObjSet>) -> BTreeMap<String, u64> {
    let mut inv = BTreeMap::new();
    for f in &bundle.files {
        inv.insert(f.sha256.clone(), f.bytes);
    }
    for o in objset.iter().flat_map(|s| &s.objects) {
        inv.insert(o.sha256.clone(), o.bytes);
    }
    for c in &bundle.checkpoint.files {
        inv.insert(c.sha256.clone(), c.bytes);
    }
    inv
}

/// Where one digest's bytes are on disk.
fn source_path(
    digest: &str,
    bundle: &Bundle,
    objset: Option<&ObjSet>,
    src: &Sources<'_>,
) -> Result<std::path::PathBuf> {
    if let Some(f) = bundle.files.iter().find(|f| f.sha256 == digest) {
        return Ok(src.assets.join(&f.name));
    }
    if let Some(o) = objset
        .iter()
        .flat_map(|s| &s.objects)
        .find(|o| o.sha256 == digest)
    {
        return Ok(src.objects.join(&o.name));
    }
    if let Some(c) = bundle.checkpoint.files.iter().find(|c| c.sha256 == digest) {
        let dir = src.checkpoint.ok_or_else(|| {
            dist(format!(
                "{}: the bundle carries its checkpoint: pass --checkpoint-dir",
                c.name
            ))
        })?;
        return Ok(dir.join(&c.name));
    }
    Err(dist(format!("no source file for blob {digest}")))
}

/// Whether two manifests describe the same build: equal in everything but
/// what publishing assigns (the generation, the variant id, and the objset
/// manifest's digest, which is recorded as it is stored).
pub fn same_build(published: &Bundle, local: &Bundle) -> bool {
    let mut p = published.clone();
    p.generation = local.generation;
    p.variant_id = local.variant_id.clone();
    if let (Some(po), Some(lo)) = (p.objset.as_mut(), local.objset.as_ref()) {
        po.sha256 = lo.sha256.clone();
    }
    p == *local
}

/// `variant_id` as every published generation is keyed: 12 hex of the sha256
/// over the full key, the build included, so two generations of one label
/// never collide.
pub fn variant_id(b: &Bundle) -> String {
    let parts = [
        b.namespace.clone(),
        b.name.clone(),
        b.label.clone(),
        b.generation.to_string(),
        b.target.isa.clone(),
        b.target.sku.clone(),
        b.target.units.to_string(),
        format!("{}{}", b.parallel.mode, b.parallel.n),
        b.max_ctx.to_string(),
        b.plow_git.clone(),
        b.objset
            .as_ref()
            .map(|o| o.objset_id.clone())
            .unwrap_or_default(),
    ];
    let digest = Digest::of(parts.join("\x1f").as_bytes());
    digest.as_str()[..12].to_string()
}

/// Sorted keys, no whitespace: the bytes the index's digests and signature
/// are computed over, and what the retired publisher wrote.
fn canonical<T: serde::Serialize>(v: &T) -> Result<Vec<u8>> {
    super::manifest::canonical_json(v)
}

fn parse<T: serde::de::DeserializeOwned>(what: &str, bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes)
        .map_err(|e| dist(format!("{what}: not valid JSON for this schema: {e}")))
}

/// Parse a document this publisher is about to rewrite (an index, the
/// catalog, the runtime index). Readers are lenient, so a client a version
/// behind still reads what a newer publisher wrote. A writer must not be: it
/// would write the document back without the fields it does not know. Such a
/// field is refused by name instead.
pub(crate) fn parse_to_rewrite<T>(what: &str, bytes: &[u8]) -> Result<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let typed: T = parse(what, bytes)?;
    let whole: serde_json::Value = parse(what, bytes)?;
    let again = serde_json::to_value(&typed)
        .map_err(|e| dist(format!("{what}: serialize: {e}")))?;
    match dropped(&whole, &again, "") {
        Some(field) => Err(dist(format!(
            "{what}: carries `{field}`, which this plowrt does not know and would drop on \
             rewrite; publish with the plowrt that wrote it, or a newer one"
        ))),
        None => Ok(typed),
    }
}

/// The first field of `was` that `now` lost, as a path. Losing a value the
/// writer omits as a default (null, empty, a `listed: true`) loses nothing.
fn dropped(was: &serde_json::Value, now: &serde_json::Value, at: &str) -> Option<String> {
    use serde_json::Value;
    let path = |k: &str| if at.is_empty() { k.to_string() } else { format!("{at}.{k}") };
    match (was, now) {
        (Value::Object(a), Value::Object(b)) => a.iter().find_map(|(k, x)| match b.get(k) {
            Some(y) => dropped(x, y, &path(k)),
            None => {
                let default = match x {
                    Value::Null => true,
                    Value::Array(v) => v.is_empty(),
                    Value::Object(m) => m.is_empty(),
                    Value::Bool(true) => k == "listed",
                    _ => false,
                };
                (!default).then(|| path(k))
            }
        }),
        (Value::Array(a), Value::Array(b)) => a
            .iter()
            .zip(b)
            .enumerate()
            .find_map(|(i, (x, y))| dropped(x, y, &format!("{at}[{i}]"))),
        _ => None,
    }
}

/// Compress for transport: the published blob is `.zst` when that is smaller,
/// plain otherwise. Safetensors barely compress, so a large file gets the fast
/// level; everything else the thorough one. Staged in `dir`, hashed as it is
/// read so what is published is exactly what the manifest names.
fn stage_blob(
    path: &Path,
    digest: &str,
    bytes: u64,
    dir: &Path,
) -> Result<(std::path::PathBuf, u64, bool)> {
    use sha2::Digest as _;
    let io = |p: &Path| {
        let p = p.to_path_buf();
        move |source| RuntimeError::Io { path: p, source }
    };
    let mut src = std::fs::File::open(path).map_err(io(path))?;
    let plain = dir.join(format!("{digest}.plain"));
    let zst = dir.join(format!("{digest}.zst"));
    let level = if bytes > 64 << 20 { 3 } else { 19 };
    let mut out_plain = std::fs::File::create(&plain).map_err(io(&plain))?;
    let mut enc =
        zstd::stream::write::Encoder::new(std::fs::File::create(&zst).map_err(io(&zst))?, level)
            .map_err(|e| dist(format!("zstd: {e}")))?;
    let (mut h, mut n, mut buf) = (sha2::Sha256::new(), 0u64, vec![0u8; 1 << 20]);
    loop {
        let k = src.read(&mut buf).map_err(io(path))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        std::io::Write::write_all(&mut out_plain, &buf[..k]).map_err(io(&plain))?;
        std::io::Write::write_all(&mut enc, &buf[..k]).map_err(io(&zst))?;
        n += k as u64;
    }
    drop(out_plain);
    enc.finish().map_err(|e| dist(format!("zstd: {e}")))?;
    let got = format!("{:x}", h.finalize());
    if got != digest {
        return Err(dist(format!(
            "{}: hashes to {got}, the manifest says {digest}",
            path.display()
        )));
    }
    let zst_len = std::fs::metadata(&zst).map_err(io(&zst))?.len();
    if zst_len < n {
        let _ = std::fs::remove_file(&plain);
        Ok((zst, zst_len, true))
    } else {
        let _ = std::fs::remove_file(&zst);
        Ok((plain, n, false))
    }
}

fn put_file(
    sink: &dyn Sink,
    key: &str,
    path: &Path,
    len: u64,
    content_type: &str,
    cond: Cond,
) -> Result<Put> {
    let mut f = std::fs::File::open(path).map_err(|source| RuntimeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    sink.put(key, &mut f, len, content_type, cond)
}

fn put_bytes(
    sink: &dyn Sink,
    key: &str,
    bytes: &[u8],
    content_type: &str,
    cond: Cond,
) -> Result<Put> {
    sink.put(key, &mut &bytes[..], bytes.len() as u64, content_type, cond)
}

/// Where the signature over one version of the index lives: keyed by the
/// version's digest, written before the index (see `push`).
fn signature_key(model_dir: &str, index_bytes: &[u8]) -> String {
    format!("{model_dir}/sigs/{}.sig", Digest::of(index_bytes).as_str())
}

/// The catalog row for this model, present and current. Names only, so a
/// lost race is retried without touching the index; a row already right is
/// left alone, which is what makes a re-run after a failed push a repair.
fn ensure_catalog(
    sink: &dyn Sink,
    bundle: &Bundle,
    hf: &str,
    description: Option<&str>,
    listed: Option<bool>,
) -> Result<()> {
    for _ in 0..4 {
        let (mut catalog, cond) = match sink.get("catalog.json")? {
            Some((raw, etag)) => (parse_to_rewrite::<Catalog>("catalog.json", &raw)?, Cond::IfMatch(etag)),
            None => (Catalog::default(), Cond::IfNoneMatchAny),
        };
        let mine = |m: &CatalogEntry| m.namespace == bundle.namespace && m.name == bundle.name;
        let existing = catalog.models.iter().find(|m| mine(m)).cloned();
        let entry = CatalogEntry {
            namespace: bundle.namespace.clone(),
            name: bundle.name.clone(),
            hf: hf.to_string(),
            description: description
                .map(str::to_string)
                .or_else(|| existing.as_ref().and_then(|m| m.description.clone())),
            listed: listed.unwrap_or(existing.as_ref().map_or(true, |m| m.listed)),
        };
        if existing.as_ref() == Some(&entry) {
            return Ok(());
        }
        catalog.models.retain(|m| !mine(m));
        catalog.models.push(entry);
        catalog
            .models
            .sort_by(|a, b| (&a.namespace, &a.name).cmp(&(&b.namespace, &b.name)));
        if let Put::Stored(_) = put_bytes(
            sink,
            "catalog.json",
            &canonical(&catalog)?,
            "application/json",
            cond,
        )? {
            return Ok(());
        }
    }
    Err(dist(format!(
        "{}: catalog.json kept changing under this push; the generation is published, run \
         the push again to list it",
        sink.describe()
    )))
}

/// Publish `bundle` (the manifests `generate` produced, not yet numbered)
/// into `sink`. `key` signs the index; without one the index is published
/// unsigned, which a client with trusted keys refuses. `trust` is applied to
/// the index already there: its rows are carried into the new, signed index,
/// so they are believed on the same terms a client applies — with no trusted
/// keys, on none.
///
/// Order of writes: blobs, objset, bundle manifest, signature, index,
/// catalog. Everything before the index is content-addressed or create-only,
/// so a push that dies anywhere leaves the previous generation servable and
/// the next run finishes the job: an identical manifest left behind is
/// adopted, a missing signature or catalog row behind a published index is
/// written on the re-run that would otherwise report "unchanged".
pub fn push(
    sink: &dyn Sink,
    mut bundle: Bundle,
    objset: Option<(ObjSet, Vec<u8>)>,
    src: &Sources<'_>,
    key: Option<&SecretKey>,
    trust: &Trust,
    opts: &Options<'_>,
) -> Result<Outcome> {
    if opts.status == Some(Status::Validated) && opts.tok_s.is_none() {
        return Err(dist(
            "--status validated needs the measurement: pass --tok-s".into(),
        ));
    }
    let hf = match (opts.hf, bundle.checkpoint.source.strip_prefix("hf:")) {
        (Some(h), _) => h.to_string(),
        (None, Some(repo)) => repo.to_string(),
        (None, None) => {
            return Err(dist(
                "the bundle's checkpoint is `dist`: pass --hf <org>/<repo> for the index".into(),
            ))
        }
    };
    let (objset, objset_bytes) = match objset {
        Some((s, b)) => (Some(s), Some(b)),
        None => (None, None),
    };
    if let (Some(os), Some(b)) = (bundle.objset.as_mut(), objset_bytes.as_ref()) {
        os.sha256 = Digest::of(b).as_str().to_string();
    }
    let model_dir = format!("v1/{}/{}", bundle.namespace, bundle.name);
    let index_key = format!("{model_dir}/index.json");
    let inv = inventory(&bundle, objset.as_ref());

    // A stale index between attempts means another publisher moved it; the
    // numbering, the manifest name and the index write all restart from a
    // fresh read. Blobs and objsets are content-addressed, so a second attempt
    // finds them present and uploads nothing twice.
    for attempt in 0..4 {
        let index: Option<(ModelIndex, String, Vec<u8>)> = match sink.get(&index_key)? {
            Some((raw, etag)) => {
                if !trust.is_empty() {
                    let sig = sink
                        .get(&signature_key(&model_dir, &raw))?
                        .map(|(b, _)| String::from_utf8_lossy(&b).trim().to_string());
                    super::sign::verify(&raw, sig.as_deref(), trust)
                        .map_err(|e| dist(format!("{index_key}: {e}")))?;
                }
                Some((parse_to_rewrite(&index_key, &raw)?, etag, raw))
            }
            None => None,
        };
        let newest = index
            .iter()
            .flat_map(|(i, _, _)| &i.variants)
            .filter(|v| v.label == bundle.label)
            .max_by_key(|v| v.generation)
            .cloned();
        if let Some(v) = &newest {
            let (raw, _) = sink
                .get(&format!("{model_dir}/{}", v.manifest))?
                .ok_or_else(|| {
                    dist(format!(
                        "{}: {model_dir}/{} is missing",
                        sink.describe(),
                        v.manifest
                    ))
                })?;
            let published: Bundle = parse(&v.manifest, &raw)?;
            let status_kept = opts.status.is_none_or(|s| v.status == s);
            let measure_kept = opts
                .tok_s
                .is_none_or(|t| v.measured.as_ref().is_some_and(|m| m.tok_s == t))
                && opts.concurrency.is_none_or(|c| {
                    v.measured
                        .as_ref()
                        .is_some_and(|m| m.concurrency == Some(c))
                });
            let hf_kept = index.as_ref().is_some_and(|(i, _, _)| i.hf == hf);
            if same_build(&published, &bundle) && status_kept && measure_kept && hf_kept {
                // Published; finish what a push that died after the index
                // may have left undone.
                if let Some((_, _, raw)) = index.as_ref() {
                    if let Some(k) = key {
                        let sig_key = signature_key(&model_dir, raw);
                        if sink.head(&sig_key)?.is_none() {
                            put_bytes(
                                sink,
                                &sig_key,
                                k.sign(raw).to_string().as_bytes(),
                                "text/plain",
                                Cond::None,
                            )?;
                        }
                    }
                }
                ensure_catalog(sink, &bundle, &hf, opts.description, opts.listed)?;
                return Ok(Outcome {
                    reference: format!(
                        "{}/{}:{}@g{}",
                        bundle.namespace, bundle.name, v.label, v.generation
                    ),
                    generation: v.generation,
                    variant_id: v.variant_id.clone(),
                    uploaded_blobs: 0,
                    uploaded_bytes: 0,
                    supersedes: None,
                    delta: None,
                    unchanged: true,
                });
            }
        }
        bundle.generation = newest.as_ref().map_or(1, |v| v.generation + 1);
        bundle.variant_id = variant_id(&bundle);

        // What a client upgrading from the predecessor will transfer, from
        // digests rather than from what kind of change this was.
        let (supersedes, delta) = match &newest {
            None => (None, None),
            Some(prev) => {
                let prior: Option<Bundle> = sink
                    .get(&format!("{model_dir}/{}", prev.manifest))?
                    .map(|(raw, _)| parse(&prev.manifest, &raw))
                    .transpose()?;
                let prior_objset: Option<ObjSet> =
                    match prior.as_ref().and_then(|p| p.objset.as_ref()) {
                        Some(o) => sink
                            .get(&format!("v1/objsets/{}.json", o.objset_id))?
                            .map(|(raw, _)| parse(&o.objset_id, &raw))
                            .transpose()?,
                        None => None,
                    };
                let prior_inv = prior
                    .as_ref()
                    .map(|p| inventory(p, prior_objset.as_ref()))
                    .unwrap_or_default();
                let packet_changed = bundle
                    .files
                    .iter()
                    .any(|f| f.role == "packet" && !prior_inv.contains_key(&f.sha256));
                (
                    Some(prev.variant_id.clone()),
                    Some(if packet_changed {
                        Delta::Packet
                    } else {
                        Delta::Objset
                    }),
                )
            }
        };

        let missing: Vec<(&String, &u64)> = {
            let mut m = Vec::new();
            for (d, n) in &inv {
                let present = sink.head(&format!("v1/blobs/sha256/{d}.zst"))?.is_some()
                    || sink.head(&format!("v1/blobs/sha256/{d}"))?.is_some();
                if !present {
                    m.push((d, n));
                }
            }
            m
        };
        if opts.dry_run {
            return Ok(Outcome {
                reference: format!(
                    "{}/{}:{}@g{}",
                    bundle.namespace, bundle.name, bundle.label, bundle.generation
                ),
                generation: bundle.generation,
                variant_id: bundle.variant_id.clone(),
                uploaded_blobs: missing.len(),
                uploaded_bytes: missing.iter().map(|(_, n)| **n).sum(),
                supersedes,
                delta,
                unchanged: false,
            });
        }

        // 1. Blobs. Each is staged once, hashed as it is read, and created
        // only if absent: a racing publisher of the same bytes is harmless.
        let staging = private_temp_dir()?;
        let uploaded = (|| -> Result<(usize, u64)> {
            let (mut n, mut bytes) = (0usize, 0u64);
            for (d, len) in &missing {
                let path = source_path(d, &bundle, objset.as_ref(), src)?;
                let (staged, staged_len, compressed) = stage_blob(&path, d, **len, &staging)?;
                let key = if compressed {
                    format!("v1/blobs/sha256/{d}.zst")
                } else {
                    format!("v1/blobs/sha256/{d}")
                };
                put_file(
                    sink,
                    &key,
                    &staged,
                    staged_len,
                    "application/octet-stream",
                    Cond::IfNoneMatchAny,
                )?;
                let _ = std::fs::remove_file(&staged);
                n += 1;
                bytes += **len;
            }
            Ok((n, bytes))
        })();
        let _ = std::fs::remove_dir_all(&staging);
        let (uploaded_blobs, uploaded_bytes) = uploaded?;

        // 2. The objset manifest, under its content-derived id. The id covers
        // the objects, not the script or env, so a name already taken must
        // hold these bytes: the bundle names their digest.
        if let (Some(os), Some(b)) = (bundle.objset.as_ref(), objset_bytes.as_ref()) {
            let key = format!("v1/objsets/{}.json", os.objset_id);
            if let Put::PreconditionFailed =
                put_bytes(sink, &key, b, "application/json", Cond::IfNoneMatchAny)?
            {
                let existing = sink.get(&key)?.map(|(b, _)| b);
                if existing.as_deref() != Some(&b[..]) {
                    return Err(dist(format!(
                        "{}: {key} already holds an objset manifest with different bytes; the \
                         id covers the objects only, so a build that shares them but differs \
                         in script or env collides with it and its bundle would name a digest \
                         the registry does not serve",
                        sink.describe()
                    )));
                }
            }
        }
        // A rung override names another objset by digest; the index must not
        // expose a bundle whose dependencies the registry lacks.
        for l in bundle.objset.iter().flat_map(|o| &o.lowrung) {
            let key = format!("v1/objsets/{}.json", l.objset_id);
            let Some((published, _)) = sink.get(&key)? else {
                return Err(dist(format!(
                    "{}: rung override {} names {key}, which is not published; a rung's \
                     objset is published before the bundle that depends on it",
                    sink.describe(),
                    l.max
                )));
            };
            // A client checks the rung against the bytes served, so the
            // bundle must name their digest, not that of a local copy.
            let got = Digest::of(&published);
            if got.as_str() != l.sha256 {
                return Err(dist(format!(
                    "{}: rung override {} records {} for {key}, the registry serves {got}",
                    sink.describe(),
                    l.max,
                    l.sha256
                )));
            }
        }

        // 3. The bundle manifest: immutable, so create-only. A name already
        // taken holds either this very build — left by a push that died
        // before its index write, and adopted — or someone else's generation,
        // which is skipped, never overwritten.
        let first = bundle.generation;
        let (manifest_rel, bundle_bytes) = loop {
            let manifest_rel = format!("manifests/{}@g{}", bundle.label, bundle.generation);
            let key = format!("{model_dir}/{manifest_rel}");
            let bundle_bytes = canonical(&bundle)?;
            match put_bytes(
                sink,
                &key,
                &bundle_bytes,
                "application/json",
                Cond::IfNoneMatchAny,
            )? {
                Put::Stored(_) => break (manifest_rel, bundle_bytes),
                Put::PreconditionFailed => {
                    let existing = sink.get(&key)?.map(|(b, _)| b);
                    if existing.as_deref() == Some(&bundle_bytes[..]) {
                        tracing::info!("{key} already holds this build; adopting it");
                        break (manifest_rel, bundle_bytes);
                    }
                    if bundle.generation - first >= 64 {
                        return Err(dist(format!(
                            "{}: {key} and the 63 names before it are taken by other builds",
                            sink.describe()
                        )));
                    }
                    tracing::warn!(attempt, "{key} is taken by another build; renumbering");
                    bundle.generation += 1;
                    bundle.variant_id = variant_id(&bundle);
                }
            }
        };
        let reference = format!(
            "{}/{}:{}@g{}",
            bundle.namespace, bundle.name, bundle.label, bundle.generation
        );

        // 4. The index, replaced only if it is still the one read above.
        let row = Variant {
            variant_id: bundle.variant_id.clone(),
            label: bundle.label.clone(),
            generation: bundle.generation,
            status: opts.status.unwrap_or(Status::Emits),
            target: bundle.target.clone(),
            parallel: bundle.parallel.clone(),
            max_ctx: bundle.max_ctx,
            features: BTreeMap::new(),
            build: Build {
                plow_git: bundle.plow_git.clone(),
                plowc: env!("CARGO_PKG_VERSION").to_string(),
                objset_id: bundle
                    .objset
                    .as_ref()
                    .map(|o| o.objset_id.clone())
                    .unwrap_or_default(),
                pairing_hash: bundle.pairing_hash.clone(),
            },
            manifest: manifest_rel.clone(),
            sha256: Digest::of(&bundle_bytes).as_str().to_string(),
            released: Some(chrono::Utc::now().format("%Y-%m-%d").to_string()),
            supersedes: supersedes.clone(),
            delta,
            measured: opts.tok_s.map(|tok_s| Measured {
                tok_s,
                p50_tpot_ms: None,
                concurrency: opts.concurrency,
            }),
            refusal: None,
        };
        let mut variants: Vec<Variant> = index
            .iter()
            .flat_map(|(i, _, _)| i.variants.iter().cloned())
            .filter(|v| !(v.label == row.label && v.generation == row.generation))
            .collect();
        variants.push(row);
        variants.sort_by(|a, b| (&a.label, a.generation).cmp(&(&b.label, b.generation)));
        let mut aliases: Vec<String> = index
            .iter()
            .flat_map(|(i, _, _)| i.aliases.iter().cloned())
            .chain(opts.aliases.iter().cloned())
            .collect();
        aliases.sort();
        aliases.dedup();
        let new_index = ModelIndex {
            schema: INDEX_SCHEMA.into(),
            namespace: bundle.namespace.clone(),
            name: bundle.name.clone(),
            hf: hf.clone(),
            revision: bundle.checkpoint.revision.clone(),
            network: bundle.network.clone(),
            aliases,
            checkpoint: plow_asset::dist::CheckpointRef {
                shards: bundle.checkpoint.shards,
                layout: bundle.checkpoint.layout.clone(),
                bytes: bundle.checkpoint.bytes,
            },
            variants,
        };
        new_index.validate().map_err(dist)?;
        let index_bytes = canonical(&new_index)?;
        super::manifest::no_build_host_paths(&index_bytes, "index.json")?;
        // 4. The signature, keyed by the index's digest, BEFORE the index: a
        // reader fetches the signature of the bytes it read, so no reader
        // ever sees this index unsigned, and two publishers' signatures
        // cannot land on each other's index. An orphan signature (index write
        // lost) is inert.
        if let Some(k) = key {
            put_bytes(
                sink,
                &signature_key(&model_dir, &index_bytes),
                k.sign(&index_bytes).to_string().as_bytes(),
                "text/plain",
                Cond::None,
            )?;
        }

        // 5. The index, replaced only if it is still the one read above.
        let cond = match &index {
            Some((_, etag, _)) => Cond::IfMatch(etag.clone()),
            None => Cond::IfNoneMatchAny,
        };
        match put_bytes(sink, &index_key, &index_bytes, "application/json", cond)? {
            Put::Stored(_) => {}
            Put::PreconditionFailed => {
                tracing::warn!(attempt, "{index_key} changed meanwhile; re-reading");
                continue;
            }
        }

        // 6. The catalog.
        ensure_catalog(sink, &bundle, &hf, opts.description, opts.listed)?;

        return Ok(Outcome {
            reference,
            generation: bundle.generation,
            variant_id: bundle.variant_id.clone(),
            uploaded_blobs,
            uploaded_bytes,
            supersedes,
            delta,
            unchanged: false,
        });
    }
    Err(dist(format!(
        "{}: the index kept changing under this push; another publisher is racing it",
        sink.describe()
    )))
}

/// A staging directory only this user can read: a `dist` checkpoint's shards
/// pass through it.
fn private_temp_dir() -> Result<std::path::PathBuf> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    for attempt in 0..16u32 {
        let dir = std::env::temp_dir().join(format!(
            "plowrt-push-{}-{stamp}-{attempt}",
            std::process::id()
        ));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        match builder.create(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(RuntimeError::Io { path: dir, source }),
        }
    }
    Err(dist("could not create a private staging directory".into()))
}

#[cfg(test)]
#[path = "push_tests.rs"]
mod tests;
