//! The `plowrt` distribution subcommands: `pull`, `load`, `show`, `ls`,
//! `upgrade`, `prepare`, `rm`, `pack`, `publish`.
//!
//! These run the network; `serve` does not. `resolve_local` is the one entry
//! point `serve` uses, and it reads only the local store — an unpulled model is
//! an error naming the `load` that fixes it, never a silent fetch.

use std::io::Write;
use std::path::PathBuf;

use plowrt::config::RuntimeConfig;
use plowrt::device;
use plowrt::dist::{self, prepare, pull, reference, store::Store, Reference};

type Err = Box<dyn std::error::Error>;

/// Print a distribution failure and exit non-zero.
///
/// Returning the error from `main` would render it through `Debug`, which wraps
/// it in `Dist("…")` and escapes every newline — and the most important thing
/// these commands print on failure is the multi-line table of variants and why
/// each one was rejected. That has to arrive as text a person can read.
pub fn report(r: Result<(), Err>) -> Result<(), Err> {
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    Ok(())
}

fn store() -> Result<Store, Err> {
    Ok(Store::open_default()?)
}

fn parse(model: &str) -> Result<Reference, Err> {
    let mut r = reference::parse(model)?;
    // A bare name takes the configured registry rather than the compiled-in
    // default, so `--registry file:///mirror` works without retyping the path.
    if r.registry == reference::DEFAULT_REGISTRY {
        r.registry = RuntimeConfig::get().registry.clone();
    }
    Ok(r)
}

/// Describe this machine for selection.
///
/// Probing opens the driver, which is exactly what `serve` will do later, so a
/// machine that cannot host the model fails here rather than after a download.
fn live() -> plow_asset::dist::LiveTarget {
    let backends = device::select_all(1);
    dist::live_target(&backends)
}

/// `plowrt target`: the live target, as a line or as the `target` object a
/// bundle manifest carries (`mem_bytes` is the usable pool the probe reports,
/// which is what selection compares against).
pub fn cmd_target(json: bool) -> Result<(), Err> {
    let t = live();
    if json {
        let sku = t
            .sku
            .clone()
            .ok_or("this machine has no GPU fingerprint; nothing to publish for")?;
        let target = plow_asset::dist::Target {
            vendor: t.vendor.clone(),
            isa: t.isa.clone(),
            sku,
            units: t.units,
            mem_bytes: t.mem_bytes,
        };
        println!("{}", serde_json::to_string_pretty(&target)?);
    } else {
        print_live(&t);
    }
    Ok(())
}

fn print_live(t: &plow_asset::dist::LiveTarget) {
    match &t.sku {
        Some(sku) => println!(
            "this machine: {} {} {sku}, {} units, {} GPU(s), {}",
            t.vendor,
            t.isa,
            t.units,
            t.gpus,
            pull::human(t.mem_bytes)
        ),
        None => println!(
            "this machine: {} (no GPU fingerprint), {} GPU(s)",
            t.vendor, t.gpus
        ),
    }
}

pub fn cmd_pull(model: &str, c: plow_asset::dist::Constraints, dry_run: bool) -> Result<(), Err> {
    let r = parse(model)?;
    let f = dist::transport(&r.registry)?;
    let live = live();
    print_live(&live);
    let resolved = pull::resolve(&*f, &r, &live, c)?;
    let v = &resolved.variant;
    println!(
        "selected {}@g{} ({:?}), {} on {} x {}",
        v.label,
        v.generation,
        v.status,
        v.parallel.label(),
        v.target.units,
        v.target.sku
    );

    let st = store()?;
    let plan = resolved.plan(&st)?;
    if dry_run {
        println!("would transfer: {plan}");
        return Ok(());
    }
    let (dir, t) = pull::pull(&st, &*f, &resolved)?;
    st.pin(&r.pin_key(), &v.variant_id)?;
    println!("transferred: {t}");
    println!("bundle: {}", dir.display());
    Ok(())
}

pub fn cmd_load(
    model: &str,
    c: plow_asset::dist::Constraints,
    checkpoint: Option<PathBuf>,
    fetch_weights: bool,
) -> Result<(), Err> {
    let r = parse(model)?;
    let f = dist::transport(&r.registry)?;
    let live = live();
    print_live(&live);
    let resolved = pull::resolve(&*f, &r, &live, c)?;
    let v = &resolved.variant;
    println!(
        "selected {}@g{} ({:?}) — {} on {} x {}",
        v.label,
        v.generation,
        v.status,
        v.parallel.label(),
        v.target.units,
        v.target.sku
    );
    if let Some(m) = &v.measured {
        println!("  measured: {:.1} tok/s", m.tok_s);
    }

    let st = store()?;
    let (dir, t) = pull::pull(&st, &*f, &resolved)?;
    println!("transferred: {t}");

    // A `dist` checkpoint came with the pull, linked into its snapshot. An
    // `hf:` one is not distributed: resolve it, or say precisely what to do.
    let dist_snapshot = (resolved.bundle.checkpoint.source == plow_asset::dist::DIST_SOURCE)
        .then(|| dist::weights::dist_snapshot_path(st.root(), &resolved.bundle))
        .transpose()?;
    let ckpt = checkpoint
        .or_else(|| RuntimeConfig::get().checkpoint.clone().map(PathBuf::from))
        .or(dist_snapshot)
        .unwrap_or_else(|| prepare::farm_path(st.root(), &resolved.bundle));
    let ckpt = if ckpt.is_dir() {
        ckpt
    } else if fetch_weights {
        let c = &resolved.bundle.checkpoint;
        println!(
            "fetching {} @{} ({})",
            c.source,
            c.revision,
            pull::human(c.bytes)
        );
        dist::weights::fetch(st.root(), &c.source, &c.revision, c.shards)?
    } else {
        return Err(format!(
            "{} needs the {} checkpoint at revision {} ({}), which the distribution does not \
             carry.\n  Point at a local snapshot:  --checkpoint <dir>\n  Or consent to \
             downloading it:      --fetch-weights",
            v.label,
            resolved.bundle.checkpoint.source,
            resolved.bundle.checkpoint.revision,
            pull::human(resolved.bundle.checkpoint.bytes),
        )
        .into());
    };

    let farm = prepare::build(st.root(), &resolved.bundle, &dir, &ckpt)?;
    st.pin(&r.pin_key(), &v.variant_id)?;
    println!(
        "checkpoint: {} ({} shards{})",
        farm.dir.display(),
        farm.shards,
        match &farm.derived {
            Some(d) => format!(", derived sidecar {d}"),
            None => String::new(),
        }
    );
    println!("\nserve it with:\n  plowrt serve --model {r} --port 8080");
    Ok(())
}

pub fn cmd_show(model: &str) -> Result<(), Err> {
    let r = parse(model)?;
    let f = dist::transport(&r.registry)?;
    let live = live();
    print_live(&live);

    let idx = pull::index(&*f, &r)?;
    println!(
        "{}/{}  hf:{} @{}",
        idx.namespace, idx.name, idx.hf, idx.revision
    );
    let c = plow_asset::dist::Constraints {
        oversub: RuntimeConfig::get().amd.oversub,
        ..Default::default()
    };
    // The same table a failed `load` prints: every variant and the rule that
    // rejected it, so "what can this box run" is answerable without a download.
    for a in plow_asset::dist::assess(&idx.variants, &live, &c) {
        let v = a.variant;
        let verdict = match &a.reject {
            None => "RUNS".to_string(),
            Some(rej) => format!("no — {rej}"),
        };
        println!(
            "  {}@g{}  {:<9} {:>3} GPU  ctx {:<6}  {}",
            v.label,
            v.generation,
            format!("{:?}", v.status).to_lowercase(),
            v.parallel.n,
            v.max_ctx,
            verdict
        );
        for w in &a.warns {
            println!("        warning: {w}");
        }
    }
    Ok(())
}

pub fn cmd_ls(upgradable: bool, json: bool) -> Result<(), Err> {
    let st = store()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&st.list())?);
        return Ok(());
    }
    let pins = st.pins();
    if pins.is_empty() {
        println!("no models pulled — try `plowrt load <model>`");
        return Ok(());
    }
    let live = upgradable.then(live);
    for (reference, variant_id) in pins {
        print!("{reference}  {variant_id}");
        if let Some(live) = &live {
            match check_newer(&reference, variant_id.as_str(), live) {
                Ok(Some((label, gen, t))) => {
                    print!("  → {label}@g{gen} available, {} to fetch", pull::human(t))
                }
                Ok(None) => print!("  (current)"),
                Err(e) => print!("  (registry unreachable: {e})"),
            }
        }
        println!();
    }
    Ok(())
}

fn check_newer(
    reference: &str,
    pinned: &str,
    live: &plow_asset::dist::LiveTarget,
) -> Result<Option<(String, u32, u64)>, Err> {
    let r = parse(reference)?;
    let f = dist::transport(&r.registry)?;
    let resolved = pull::resolve(&*f, &r, live, Default::default())?;
    if resolved.variant.variant_id == pinned {
        return Ok(None);
    }
    let plan = resolved.plan(&store()?)?;
    Ok(Some((
        resolved.variant.label.clone(),
        resolved.variant.generation,
        plan.fetched_bytes,
    )))
}

pub fn cmd_upgrade(model: Option<&str>, all: bool, dry_run: bool) -> Result<(), Err> {
    let st = store()?;
    let targets: Vec<String> = match (model, all) {
        (Some(m), false) => vec![parse(m)?.pin_key()],
        (None, true) => st.pins().into_iter().map(|(k, _)| k).collect(),
        _ => return Err("give a model reference, or --all".into()),
    };
    if targets.is_empty() {
        println!("nothing pinned");
        return Ok(());
    }
    let live = live();
    for pin in targets {
        let r = parse(&pin)?;
        let f = dist::transport(&r.registry)?;
        let resolved = pull::resolve(&*f, &r, &live, Default::default())?;
        let pinned = st.pinned(&pin);
        if pinned.as_deref() == Some(resolved.variant.variant_id.as_str()) {
            println!("{pin}: current at g{}", resolved.variant.generation);
            continue;
        }
        let plan = resolved.plan(&st)?;
        if dry_run {
            println!(
                "{pin}: g{} available, would transfer {}",
                resolved.variant.generation,
                pull::human(plan.fetched_bytes)
            );
            continue;
        }
        let (_, t) = pull::pull(&st, &*f, &resolved)?;
        // The pin moves only after every blob is present and verified, so an
        // interrupted upgrade leaves the previous generation servable.
        st.pin(&pin, &resolved.variant.variant_id)?;
        println!("{pin}: → g{} ({t})", resolved.variant.generation);
    }
    Ok(())
}

/// Package a built variant into one `.zip`: `bundle.json`, `objset.json`,
/// and every file they reference, under `assets/` and `objects/`
/// respectively so a bundle file and an objset object can never collide on
/// name — and a `dist` checkpoint's files under `checkpoint/`. Does not build
/// the manifests — `pack_bundle.py`/`pack_objset.py` still do that; this only
/// packages what they already produced.
#[cfg(feature = "dist")]
pub fn cmd_pack(
    bundle_path: &std::path::Path,
    objset_path: Option<&std::path::Path>,
    assets: &std::path::Path,
    objects: Option<&std::path::Path>,
    checkpoint_dir: Option<&std::path::Path>,
    out: &std::path::Path,
) -> Result<(), Err> {
    use plow_asset::dist::{Bundle, ObjSet};

    let bundle_bytes = std::fs::read(bundle_path)?;
    let bundle: Bundle = serde_json::from_slice(&bundle_bytes)
        .map_err(|e| format!("{}: {e}", bundle_path.display()))?;
    bundle
        .validate()
        .map_err(|e| format!("{}: {e}", bundle_path.display()))?;

    // A bundle that names an objset must ship it; a runtime-compiled one
    // (no objset in the bundle) must not be handed one.
    let objset = match (&bundle.objset, objset_path) {
        (Some(_), None) => return Err("the bundle names an objset: pass --objset".into()),
        (None, Some(p)) => {
            return Err(format!("{}: the bundle has no objset to pair it with", p.display()).into())
        }
        (None, None) => None,
        (Some(_), Some(objset_path)) => {
            let objset_bytes = std::fs::read(objset_path)?;
            let objset: ObjSet = serde_json::from_slice(&objset_bytes)
                .map_err(|e| format!("{}: {e}", objset_path.display()))?;
            objset
                .validate()
                .map_err(|e| format!("{}: {e}", objset_path.display()))?;
            objset
                .pairs_with(bundle.pairing_hash.as_deref())
                .map_err(|e| format!("{}: {e}", objset_path.display()))?;
            Some((objset, objset_bytes))
        }
    };

    // Every argument is checked before --out is created or truncated.
    let objects_in = match (&objset, objects) {
        (Some(_), None) => return Err("the objset's objects need --objects".into()),
        (Some((objset, _)), Some(dir)) => objset.objects.iter().map(|o| (o, dir)).collect(),
        (None, _) => Vec::new(),
    };
    let checkpoint_in = match (bundle.checkpoint.files.is_empty(), checkpoint_dir) {
        (false, None) => {
            return Err("the bundle carries its checkpoint: pass --checkpoint-dir".into())
        }
        (true, Some(p)) => {
            return Err(format!("{}: the bundle's checkpoint lists no files", p.display()).into())
        }
        (_, dir) => dir,
    };

    let file = std::fs::File::create(out)?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("bundle.json", options)?;
    zip.write_all(&bundle_bytes)?;
    let mut total_bytes: u64 = bundle_bytes.len() as u64;
    let mut total_files: usize = 1;
    if let Some((_, objset_bytes)) = &objset {
        zip.start_file("objset.json", options)?;
        zip.write_all(objset_bytes)?;
        total_bytes += objset_bytes.len() as u64;
        total_files += 1;
    }

    for f in &bundle.files {
        let path = assets.join(&f.name);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let got = dist::Digest::of(&bytes);
        if got.as_str() != f.sha256 {
            return Err(format!(
                "{}: hashes to {got}, bundle.json says {}",
                path.display(),
                f.sha256
            )
            .into());
        }
        zip.start_file(format!("assets/{}", f.name), options)?;
        zip.write_all(&bytes)?;
        total_bytes += bytes.len() as u64;
        total_files += 1;
    }

    for (o, objects) in objects_in {
        let path = objects.join(&o.name);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let got = dist::Digest::of(&bytes);
        if got.as_str() != o.sha256 {
            return Err(format!(
                "{}: hashes to {got}, objset.json says {}",
                path.display(),
                o.sha256
            )
            .into());
        }
        zip.start_file(format!("objects/{}", o.name), options)?;
        zip.write_all(&bytes)?;
        total_bytes += bytes.len() as u64;
        total_files += 1;
    }

    // Weights are stored, not deflated: safetensors barely compress, and a
    // shard can pass 4 GiB, which needs a zip64 entry.
    if let Some(dir) = checkpoint_in {
        for c in &bundle.checkpoint.files {
            let path = dir.join(&c.name);
            let stored = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored)
                .large_file(c.bytes >= u32::MAX as u64);
            zip.start_file(format!("checkpoint/{}", c.name), stored)?;
            let (got, n) = copy_hashed(&path, &mut zip)?;
            if got != c.sha256 {
                return Err(format!(
                    "{}: hashes to {got}, bundle.json says {}",
                    path.display(),
                    c.sha256
                )
                .into());
            }
            total_bytes += n;
            total_files += 1;
        }
    }

    zip.finish()?;
    println!(
        "{}: {} file(s), {} packaged",
        out.display(),
        total_files,
        pull::human(total_bytes)
    );
    Ok(())
}

/// Stream a file into `w`, returning its sha256 and length. A checkpoint
/// shard is gigabytes; reading it whole, as the bundle files are, would hold
/// all of it in memory.
#[cfg(feature = "dist")]
fn copy_hashed(path: &std::path::Path, w: &mut impl Write) -> Result<(String, u64), Err> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut h, mut n, mut buf) = (Sha256::new(), 0u64, vec![0u8; 1 << 20]);
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        w.write_all(&buf[..k])?;
        n += k as u64;
    }
    Ok((format!("{:x}", h.finalize()), n))
}

#[cfg(feature = "dist")]
/// Where `publish` sends a packed bundle.
pub struct PublishTo<'a> {
    pub url: &'a str,
    pub api_key: &'a str,
    pub status: Option<&'a str>,
    pub description: Option<&'a str>,
    pub tok_s: Option<f64>,
    pub concurrency: Option<u32>,
}

#[cfg(feature = "dist")]
/// `pack` into a temporary archive, then publish it through the registry's
/// staged upload: ask for an upload target, `PUT` the zip there, publish it.
/// The target is a presigned bucket URL when the registry has one, so a large
/// archive never passes the API's request limit.
pub fn cmd_publish(
    bundle_path: &std::path::Path,
    objset_path: Option<&std::path::Path>,
    assets: &std::path::Path,
    objects: Option<&std::path::Path>,
    checkpoint_dir: Option<&std::path::Path>,
    to: &PublishTo,
) -> Result<(), Err> {
    // The registry's index would hold a variant clients refuse to read.
    if to.status == Some("validated") && to.tok_s.is_none() {
        return Err("--status validated needs the measurement: pass --tok-s".into());
    }
    let dir = private_temp_dir()?;
    let sent = (|| {
        let Some(bundle) = next_generation(bundle_path, to.url)? else {
            return Ok(());
        };
        let bundle_out = dir.join("bundle.json");
        std::fs::write(&bundle_out, serde_json::to_vec_pretty(&bundle)?)?;
        let zip_path = dir.join("bundle.zip");
        cmd_pack(
            &bundle_out,
            objset_path,
            assets,
            objects,
            checkpoint_dir,
            &zip_path,
        )?;
        upload(&zip_path, to, &bundle)
    })();
    let _ = std::fs::remove_dir_all(&dir);
    sent
}

/// The bundle as the registry's next generation of its label, as
/// `publish_dist.py` numbers one; `None` when the newest generation already
/// carries exactly these files and objset (a re-run publishes nothing).
#[cfg(feature = "dist")]
fn next_generation(
    bundle_path: &std::path::Path,
    registry: &str,
) -> Result<Option<plow_asset::dist::Bundle>, Err> {
    use plow_asset::dist::{Bundle, ModelIndex};
    let mut bundle: Bundle = serde_json::from_slice(&std::fs::read(bundle_path)?)
        .map_err(|e| format!("{}: {e}", bundle_path.display()))?;
    let f = dist::fetch::transport(registry)?;
    let model_dir = format!("v1/{}/{}", bundle.namespace, bundle.name);
    let index: Option<ModelIndex> = f
        .get(&format!("{model_dir}/index.json"))?
        .map(|raw| serde_json::from_slice(&raw))
        .transpose()
        .map_err(|e| format!("{}: {model_dir}/index.json: {e}", f.describe()))?;
    let newest = index
        .iter()
        .flat_map(|i| &i.variants)
        .filter(|v| v.label == bundle.label)
        .max_by_key(|v| v.generation);
    if let Some(v) = newest {
        let raw = f
            .get(&format!("{model_dir}/{}", v.manifest))?
            .ok_or_else(|| format!("{}: {model_dir}/{} is missing", f.describe(), v.manifest))?;
        let published: Bundle = serde_json::from_slice(&raw)
            .map_err(|e| format!("{}: {}: {e}", f.describe(), v.manifest))?;
        if same_build(&published, &bundle) {
            println!(
                "already published: {}/{}:{}@g{} is this build",
                bundle.namespace, bundle.name, bundle.label, v.generation
            );
            return Ok(None);
        }
    }
    bundle.generation = newest.map_or(1, |v| v.generation + 1);
    bundle.variant_id = variant_id(&bundle);
    Ok(Some(bundle))
}

/// Whether two manifests describe the same build: equal in everything but
/// what publishing assigns (the generation, the variant id, and the objset
/// manifest's digest, which the registry records as it stores it).
#[cfg(feature = "dist")]
fn same_build(published: &plow_asset::dist::Bundle, local: &plow_asset::dist::Bundle) -> bool {
    let mut p = published.clone();
    p.generation = local.generation;
    p.variant_id = local.variant_id.clone();
    if let (Some(po), Some(lo)) = (p.objset.as_mut(), local.objset.as_ref()) {
        po.sha256 = lo.sha256.clone();
    }
    p == *local
}

/// `publish_dist.py`'s `variant_id`: 12 hex of the sha256 over the full key,
/// the build included, so two generations of one label never collide.
#[cfg(feature = "dist")]
fn variant_id(b: &plow_asset::dist::Bundle) -> String {
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
    let digest = dist::Digest::of(parts.join("\x1f").as_bytes());
    digest.as_str()[..12].to_string()
}

/// A new directory only this user can enter. `create_dir` refuses a path
/// that exists, symlink or not, so nothing another user planted is written
/// through.
#[cfg(feature = "dist")]
fn private_temp_dir() -> Result<PathBuf, Err> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    for attempt in 0..16u32 {
        let dir = std::env::temp_dir().join(format!(
            "plowrt-publish-{}-{stamp}-{attempt}",
            std::process::id()
        ));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        match builder.create(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("{}: {e}", dir.display()).into()),
        }
    }
    Err("could not create a private temporary directory".into())
}

#[cfg(feature = "dist")]
fn upload(
    zip_path: &std::path::Path,
    to: &PublishTo,
    bundle: &plow_asset::dist::Bundle,
) -> Result<(), Err> {
    #[derive(serde::Deserialize)]
    struct UploadTarget {
        id: String,
        upload_url: String,
    }
    let base = to.url.trim_end_matches('/');
    let auth = format!("Bearer {}", to.api_key);
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(20))
        .build();
    let failed = |url: &str, e: ureq::Error| -> Err {
        match e {
            ureq::Error::Status(code, resp) => {
                format!("{url}: {code}: {}", resp.into_string().unwrap_or_default()).into()
            }
            // Not ureq's own text: it repeats the URL, query string and all.
            ureq::Error::Transport(t) => {
                format!("{url}: {} {}", t.kind(), t.message().unwrap_or_default()).into()
            }
        }
    };

    // 1. Where to put it.
    let uploads = format!("{base}/api/v1/admin/registry/uploads");
    let created = agent
        .post(&uploads)
        .set("Authorization", &auth)
        .call()
        .map_err(|e| failed(&uploads, e))?;
    let target: UploadTarget =
        serde_json::from_reader(created.into_reader()).map_err(|e| format!("{uploads}: {e}"))?;

    // 2. The zip. An absolute URL is a presigned bucket and takes no
    // credentials; a relative one is the registry itself.
    let file = std::fs::File::open(zip_path)?;
    let len = file.metadata()?.len().to_string();
    let put_url = if target.upload_url.starts_with('/') {
        format!("{base}{}", target.upload_url)
    } else {
        target.upload_url.clone()
    };
    let mut put = agent
        .put(&put_url)
        .set("Content-Type", "application/zip")
        .set("Content-Length", &len);
    if target.upload_url.starts_with('/') {
        put = put.set("Authorization", &auth);
    }
    // A presigned URL's query string is a credential: never print it.
    let shown = put_url.split('?').next().unwrap_or_default();
    put.send(file).map_err(|e| failed(shown, e))?;

    // 3. Publish it.
    let publish = format!("{uploads}/{}/publish", target.id);
    let mut req = agent.post(&publish).set("Authorization", &auth);
    if let Some(s) = to.status {
        req = req.query("status", s);
    }
    if let Some(d) = to.description {
        req = req.query("description", d);
    }
    if let Some(t) = to.tok_s {
        req = req.query("tok_s", &t.to_string());
    }
    if let Some(c) = to.concurrency {
        req = req.query("concurrency", &c.to_string());
    }
    match req.call() {
        Ok(resp) => {
            println!("published: {}", resp.into_string()?);
            Ok(())
        }
        // Generations are immutable: another publish took this one first.
        Err(ureq::Error::Status(409, resp)) => Err(format!(
            "{}@g{} was published by someone else while this ran; run again for the next \
             generation ({})",
            bundle.label,
            bundle.generation,
            resp.into_string().unwrap_or_default()
        )
        .into()),
        Err(e) => Err(failed(&publish, e)),
    }
}

pub fn cmd_prepare(model: &str, checkpoint: Option<PathBuf>) -> Result<(), Err> {
    let r = parse(model)?;
    let st = store()?;
    let dir = resolve_local(model)?;
    let bundle_json = dir.join("bundle.json");
    let bundle: plow_asset::dist::Bundle = serde_json::from_slice(&std::fs::read(&bundle_json)?)
        .map_err(|e| format!("{}: {e}", bundle_json.display()))?;
    let ckpt = checkpoint
        .or_else(|| RuntimeConfig::get().checkpoint.clone().map(PathBuf::from))
        .ok_or("pass --checkpoint <dir> (or set PLOW_CHECKPOINT)")?;
    let farm = prepare::build(st.root(), &bundle, &dir, &ckpt)?;
    println!("{}: {} shards at {}", r, farm.shards, farm.dir.display());
    Ok(())
}

pub fn cmd_rm(model: &str, gc: bool) -> Result<(), Err> {
    let r = parse(model)?;
    let st = store()?;
    let Some(variant_id) = st.pinned(&r.pin_key()) else {
        return Err(format!("{r} is not pinned").into());
    };
    st.unpin(&r.pin_key())?;
    let bundle = st.root().join("bundles").join(&variant_id);
    let _ = std::fs::remove_dir_all(&bundle);
    println!("unpinned {r}");
    if gc {
        let (n, freed) = st.gc()?;
        println!("collected {n} blob(s), {} reclaimed", pull::human(freed));
    } else {
        println!("blobs kept; `plowrt rm {model} --gc` reclaims what nothing references");
    }
    Ok(())
}

/// Resolve a reference to a materialized bundle directory using ONLY the local
/// store. This is what `serve --model` calls, and why serving needs no network.
pub fn resolve_local(model: &str) -> Result<PathBuf, Err> {
    let r = parse(model)?;
    let st = store()?;
    let variant_id = st.pinned(&r.pin_key()).ok_or_else(|| {
        format!("{r} has not been pulled on this machine — run `plowrt load {model}` first")
    })?;
    let dir = st.root().join("bundles").join(&variant_id);
    if !dir.is_dir() {
        return Err(format!(
            "{r} is pinned to {variant_id}, but its bundle directory is missing — \
             run `plowrt load {model}` to re-materialize it"
        )
        .into());
    }
    Ok(dir)
}

#[cfg(all(test, feature = "dist"))]
mod tests {
    #[test]
    fn variant_id_is_publish_dists() {
        // `publish_dist.py`'s `variant_id(bundle, 1)` for this bundle.
        let b: plow_asset::dist::Bundle = serde_json::from_value(serde_json::json!({
            "schema": "plow.dist.bundle.v1", "namespace": "infervisor", "name": "smollm2-360m",
            "label": "metal3-apple-m4-pro-tp1-0k", "generation": 1, "variant_id": "",
            "plow_git": "fa87c478", "network": "smollm2-360m-instruct",
            "target": {"vendor": "apple", "isa": "metal3", "sku": "Apple M4 Pro", "units": 16, "mem_bytes": 19069665280u64},
            "parallel": {"mode": "tp", "n": 1}, "max_ctx": 0, "files": [],
            "tokenizer": {"file": "tokenizer.json", "generator": "plow", "sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c", "source": "derived", "verified": false},
            "checkpoint": {"source": "hf:HuggingFaceTB/SmolLM2-360M-Instruct", "revision": "main", "shards": 0, "layout": "hf-safetensors", "bytes": 0}
        }))
        .unwrap();
        assert_eq!(super::variant_id(&b), "b49b5c9cf598");
    }

    #[test]
    fn same_build_ignores_only_what_publishing_assigns() {
        let local: plow_asset::dist::Bundle = serde_json::from_value(serde_json::json!({
            "schema": "plow.dist.bundle.v1", "namespace": "infervisor", "name": "m",
            "label": "l", "generation": 0, "variant_id": "", "plow_git": "abc", "network": "m",
            "target": {"vendor": "apple", "isa": "metal3", "sku": "Apple M4 Pro", "units": 16, "mem_bytes": 1},
            "parallel": {"mode": "tp", "n": 1}, "max_ctx": 2048, "files": [],
            "tokenizer": {"file": "t", "generator": "plow", "sha256": "0", "source": "derived", "verified": false},
            "checkpoint": {"source": "hf:o/m", "revision": "main", "shards": 0, "layout": "x", "bytes": 0}
        }))
        .unwrap();
        let mut published = local.clone();
        published.generation = 3;
        published.variant_id = "abcdef012345".into();
        assert!(super::same_build(&published, &local));
        published.max_ctx = 4096;
        assert!(
            !super::same_build(&published, &local),
            "a metadata change is a new build"
        );
    }
}
