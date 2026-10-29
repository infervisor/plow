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
    let resolved = pull::resolve_trusted(&*f, &r, &live, c, &trust_for(&r.registry)?)?;
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
    let _lock = st.lock()?;
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
    let resolved = pull::resolve_trusted(&*f, &r, &live, c, &trust_for(&r.registry)?)?;
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
    let _lock = st.lock()?;
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

    let idx = pull::index_trusted(&*f, &r, &trust_for(&r.registry)?)?;
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
    let resolved =
        pull::resolve_trusted(&*f, &r, live, Default::default(), &trust_for(&r.registry)?)?;
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
    let _lock = st.lock()?;
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
        let resolved =
            pull::resolve_trusted(&*f, &r, &live, Default::default(), &trust_for(&r.registry)?)?;
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

/// What `pack` and `publish` build the two manifests from: the recipe that
/// declares the variant, plus the facts about one build that a recipe cannot
/// fix. Shared by both so they cannot drift on how a variant is identified.
#[cfg(feature = "dist")]
pub struct Recipe<'a> {
    pub path: &'a std::path::Path,
    pub over: plowrt::dist::manifest::Overrides<'a>,
    pub args: plowrt::dist::manifest::BundleArgs<'a>,
}

/// Package a built variant into one `.zip`: `bundle.json`, `objset.json`,
/// and every file they reference, under `assets/` and `objects/`
/// respectively so a bundle file and an objset object can never collide on
/// name — and a `dist` checkpoint's files under `checkpoint/`.
///
/// `manifests` writes just the two JSON files, for a caller that validates or
/// publishes them separately from the archive.
#[cfg(feature = "dist")]
pub fn cmd_pack<'a>(
    recipe: Recipe<'a>,
    assets: &std::path::Path,
    objects: Option<&std::path::Path>,
    checkpoint_dir: Option<&'a std::path::Path>,
    out: Option<&std::path::Path>,
    manifests: Option<&std::path::Path>,
) -> Result<(), Err> {
    if out.is_none() && manifests.is_none() {
        return Err("give --out (the archive), --manifests (the two JSON files), or both".into());
    }
    // Every argument is checked before anything is created or truncated.
    let args = pack_args(recipe, assets, objects, checkpoint_dir)?;
    if let Some(dir) = manifests {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("bundle.json"), &args.bundle_bytes)?;
        if let Some((_, bytes)) = &args.objset {
            std::fs::write(dir.join("objset.json"), bytes)?;
        }
    }
    match out {
        Some(out) => write_zip(&args, assets, objects, out),
        None => Ok(()),
    }
}

/// Write the archive. Separate from [`cmd_pack`] because `publish` packs a
/// bundle it has renumbered in memory, and staging that to a file only to read
/// it back is a round trip that can disagree with itself.
#[cfg(feature = "dist")]
fn write_zip(
    args: &PackArgs<'_>,
    assets: &std::path::Path,
    objects: Option<&std::path::Path>,
    out: &std::path::Path,
) -> Result<(), Err> {
    let PackArgs {
        bundle_bytes,
        bundle,
        objset,
        checkpoint_in,
    } = args;
    let objects_in = match (&objset, objects) {
        (Some((objset, _)), Some(dir)) => objset.objects.iter().map(|o| (o, dir)).collect(),
        // A recipe's objects live in the asset directory unless the recipe's
        // build script put them elsewhere, exactly as `generate` reads them.
        (Some((objset, _)), None) => objset.objects.iter().map(|o| (o, assets)).collect(),
        _ => Vec::new(),
    };

    let file = std::fs::File::create(out)?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("bundle.json", options)?;
    zip.write_all(bundle_bytes)?;
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

/// What `pack` reads from its arguments, every one of them checked.
#[cfg(feature = "dist")]
struct PackArgs<'a> {
    bundle_bytes: Vec<u8>,
    bundle: plow_asset::dist::Bundle,
    objset: Option<(plow_asset::dist::ObjSet, Vec<u8>)>,
    checkpoint_in: Option<&'a std::path::Path>,
}

/// Check `pack`'s arguments against the bundle: `publish` runs this before it
/// asks the registry anything, so a wrong argument never reads as "already
/// published".
#[cfg(feature = "dist")]
fn pack_args<'a>(
    recipe: Recipe<'a>,
    assets: &std::path::Path,
    objects: Option<&std::path::Path>,
    checkpoint_dir: Option<&'a std::path::Path>,
) -> Result<PackArgs<'a>, Err> {
    let Recipe {
        path,
        over,
        mut args,
    } = recipe;
    // A `dist` checkpoint's `files[]` is enumerated from this directory, so the
    // generator needs it and not only the zip writer. Hardcoding `None` here
    // left no way at all to pack a carried checkpoint from a recipe: the bundle
    // came out with an empty `files[]` and the refusal blamed the caller.
    args.checkpoint_dir = checkpoint_dir;
    let g = plowrt::dist::manifest::generate(path, assets, objects, over, &args)?;
    let (bundle_bytes, bundle, objset) = (g.bundle_bytes, g.bundle, g.objset);
    let where_ = path.display();
    // Generated, but gated exactly as a hand-written manifest was: the
    // generator is not trusted to have produced a publishable manifest.
    bundle.validate().map_err(|e| format!("{where_}: {e}"))?;
    if let Some((objset, _)) = &objset {
        objset.validate().map_err(|e| format!("{where_}: {e}"))?;
        objset
            .pairs_with(bundle.pairing_hash.as_deref())
            .map_err(|e| format!("{where_}: {e}"))?;
    }
    let checkpoint_in = checkpoint_arg(&bundle, checkpoint_dir)?;
    Ok(PackArgs {
        bundle_bytes,
        bundle,
        objset,
        checkpoint_in,
    })
}

/// `--checkpoint-dir`, given exactly when the bundle carries its checkpoint.
#[cfg(feature = "dist")]
fn checkpoint_arg<'a>(
    bundle: &plow_asset::dist::Bundle,
    checkpoint_dir: Option<&'a std::path::Path>,
) -> Result<Option<&'a std::path::Path>, Err> {
    match (bundle.checkpoint.files.is_empty(), checkpoint_dir) {
        (false, None) => Err("the bundle carries its checkpoint: pass --checkpoint-dir".into()),
        (true, Some(p)) => {
            Err(format!("{}: the bundle's checkpoint lists no files", p.display()).into())
        }
        (_, dir) => Ok(dir),
    }
}

#[cfg(feature = "dist")]
/// Where `publish` sends a packed bundle.
pub struct PublishTo<'a> {
    pub url: &'a str,
    pub api_key: &'a str,
    pub status: Option<&'a str>,
    pub description: Option<&'a str>,
    pub hf: Option<&'a str>,
    pub tok_s: Option<f64>,
    pub concurrency: Option<u32>,
}

#[cfg(feature = "dist")]
/// `pack` into a temporary archive, then publish it through the registry's
/// staged upload: ask for an upload target, `PUT` the zip there, publish it.
/// The target is a presigned bucket URL when the registry has one, so a large
/// archive never passes the API's request limit.
pub fn cmd_publish<'a>(
    recipe: Recipe<'a>,
    assets: &std::path::Path,
    objects: Option<&std::path::Path>,
    checkpoint_dir: Option<&'a std::path::Path>,
    to: &PublishTo,
) -> Result<(), Err> {
    // The registry's index would hold a variant clients refuse to read.
    if to.status == Some("validated") && to.tok_s.is_none() {
        return Err("--status validated needs the measurement: pass --tok-s".into());
    }
    // Every argument is checked before the registry is asked anything: a
    // wrong one must fail, not read as "already published".
    let mut args = pack_args(recipe, assets, objects, checkpoint_dir)?;
    if args.bundle.checkpoint.source == "dist" && to.hf.is_none() {
        return Err(
            "the bundle's checkpoint is `dist`: pass --hf <org>/<repo> for the index".into(),
        );
    }
    let dir = private_temp_dir()?;
    let sent = (|| {
        let Some(bundle) = next_generation(args.bundle.clone(), to)? else {
            return Ok(());
        };
        // The archive carries the bundle as the registry will store it.
        args.bundle_bytes = serde_json::to_vec_pretty(&bundle)?;
        args.bundle = bundle;
        let zip_path = dir.join("bundle.zip");
        write_zip(&args, assets, objects, &zip_path)?;
        upload(&zip_path, to, &args.bundle)
    })();
    let _ = std::fs::remove_dir_all(&dir);
    sent
}

/// The bundle as the registry's next generation of its label; `None` when
/// the newest generation is already
/// this build at the status and measurement asked for (a re-run publishes
/// nothing). A new status or measurement for the same build is a new
/// generation, since a published one is immutable.
#[cfg(feature = "dist")]
fn next_generation(
    mut bundle: plow_asset::dist::Bundle,
    to: &PublishTo,
) -> Result<Option<plow_asset::dist::Bundle>, Err> {
    let registry = to.url;
    use plow_asset::dist::{Bundle, ModelIndex};
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
        let status_kept = to.status.is_none_or(|s| {
            serde_json::to_value(v.status)
                .ok()
                .and_then(|j| j.as_str().map(|j| j == s))
                == Some(true)
        });
        let measure_kept = to
            .tok_s
            .is_none_or(|t| v.measured.as_ref().is_some_and(|m| m.tok_s == t))
            && to.concurrency.is_none_or(|c| {
                v.measured
                    .as_ref()
                    .is_some_and(|m| m.concurrency == Some(c))
            });
        let hf_kept = to
            .hf
            .is_none_or(|h| index.as_ref().is_some_and(|i| i.hf == h));
        if same_build(&published, &bundle) && status_kept && measure_kept && hf_kept {
            if to.description.is_some() {
                println!(
                    "note: --description was not applied; it is recorded with a new generation, \
                     and this build is already published"
                );
            }
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

#[cfg(feature = "dist")]
use plowrt::dist::push::{same_build, variant_id};

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
    if let Some(h) = to.hf {
        req = req.query("hf", h);
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

/// The keys this machine accepts a model index from: `PLOW_TRUSTED_KEYS`.
/// The keys `registry` is verified with: see [`dist::sign::Trust::for_registry`].
fn trust_for(registry: &str) -> Result<dist::sign::Trust, Err> {
    Ok(dist::sign::Trust::for_registry(
        RuntimeConfig::signing_env().1.as_deref(),
        registry,
    )?)
}

/// Write a built variant straight into a registry: blobs, manifests, index,
/// signature, catalog, each write conditional so a racing publisher is
/// refused rather than overwritten.
#[cfg(feature = "dist")]
#[allow(clippy::too_many_arguments)]
pub fn cmd_push<'a>(
    recipe: Recipe<'a>,
    assets: &std::path::Path,
    objects: Option<&std::path::Path>,
    checkpoint_dir: Option<&'a std::path::Path>,
    to: &str,
    key_file: Option<&std::path::Path>,
    opts: &dist::push::Options<'_>,
) -> Result<(), Err> {
    // Every argument is checked before the registry is asked anything.
    let args = pack_args(recipe, assets, objects, checkpoint_dir)?;
    let sink = dist::sink::sink(to)?;
    let key = match key_file {
        Some(p) => Some(dist::sign::SecretKey::read(p)?),
        None => None,
    };
    if key.is_none() && !opts.dry_run && !to.starts_with("file://") && !to.starts_with('/') {
        println!(
            "warning: no signing key (--key-file or PLOW_SIGNING_KEY): the index goes up \
             unsigned, and a client with PLOW_TRUSTED_KEYS set will refuse it"
        );
    }
    let objects_dir = objects.unwrap_or(assets);
    let src = dist::push::Sources {
        assets,
        objects: objects_dir,
        checkpoint: checkpoint_dir,
    };
    let trust = trust_for(to)?;
    if key.is_some() && trust.is_empty() && !opts.dry_run {
        println!(
            "warning: PLOW_TRUSTED_KEYS is not set: the rows of the index already published \
             are carried into the signed index unverified"
        );
    }
    let out = dist::push::push(
        &*sink,
        args.bundle,
        args.objset,
        &src,
        key.as_ref(),
        &trust,
        opts,
    )?;
    if out.unchanged {
        println!(
            "already published: {} is this build at {}",
            out.reference,
            sink.describe()
        );
        return Ok(());
    }
    if opts.dry_run {
        println!(
            "would publish {} to {}: {} blob(s), {} to upload{}",
            out.reference,
            sink.describe(),
            out.uploaded_blobs,
            pull::human(out.uploaded_bytes),
            match (&out.supersedes, out.delta) {
                (Some(s), Some(d)) => format!(", supersedes {s} ({d:?})"),
                _ => String::new(),
            }
        );
        return Ok(());
    }
    println!(
        "published {} ({}) to {}: {} blob(s) uploaded, {}{}",
        out.reference,
        out.variant_id,
        sink.describe(),
        out.uploaded_blobs,
        pull::human(out.uploaded_bytes),
        match (&out.supersedes, out.delta) {
            (Some(s), Some(d)) => format!(", supersedes {s} ({d:?})"),
            _ => String::new(),
        }
    );
    Ok(())
}

/// Generate a signing key. The secret goes to `out` (mode 0600); the public
/// half is printed, which is what every client's `PLOW_TRUSTED_KEYS` takes.
#[cfg(feature = "dist")]
pub fn cmd_keygen(name: &str, out: &std::path::Path) -> Result<(), Err> {
    if out.exists() {
        return Err(format!("{}: exists; a key is never overwritten", out.display()).into());
    }
    let key = dist::sign::SecretKey::generate(name)?;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    let mut f = opts.open(out)?;
    f.write_all(key.to_file_contents().as_bytes())?;
    println!("secret key: {}", out.display());
    println!("public key: {}", key.public());
    println!("clients trust it with: PLOW_TRUSTED_KEYS={}", key.public());
    Ok(())
}

/// Move a pin back to the generation it served before the current one.
pub fn cmd_rollback(model: &str) -> Result<(), Err> {
    let r = parse(model)?;
    let st = store()?;
    let _lock = st.lock()?;
    let (from, to) = st.rollback(&r.pin_key())?;
    println!("{r}: {from} → {to}");
    if let Some(p) = st.list().into_iter().find(|p| p.variant_id == to) {
        println!("now serving {}", p.reference);
    }
    Ok(())
}

/// Collect the bundles no pin can serve or roll back to, and the blobs no
/// bundle references.
pub fn cmd_gc() -> Result<(), Err> {
    let st = store()?;
    let _lock = st.lock()?;
    let pruned = st.prune()?;
    let (n, freed) = st.gc()?;
    println!(
        "removed {pruned} superseded generation(s); collected {n} blob(s), {} reclaimed",
        pull::human(freed)
    );
    Ok(())
}

pub fn cmd_prepare(model: &str, checkpoint: Option<PathBuf>) -> Result<(), Err> {
    let r = parse(model)?;
    let st = store()?;
    let _lock = st.lock()?;
    let dir = resolve_local(model)?;
    let bundle_json = dir.join("bundle.json");
    let bundle: plow_asset::dist::Bundle = serde_json::from_slice(&std::fs::read(&bundle_json)?)
        .map_err(|e| format!("{}: {e}", bundle_json.display()))?;
    // A `dist` checkpoint came with the pull, as for `load`.
    let dist_snapshot = (bundle.checkpoint.source == plow_asset::dist::DIST_SOURCE)
        .then(|| dist::weights::dist_snapshot_path(st.root(), &bundle))
        .transpose()?;
    let ckpt = checkpoint
        .or_else(|| RuntimeConfig::get().checkpoint.clone().map(PathBuf::from))
        .or(dist_snapshot)
        .ok_or("pass --checkpoint <dir> (or set PLOW_CHECKPOINT)")?;
    let farm = prepare::build(st.root(), &bundle, &dir, &ckpt)?;
    println!("{}: {} shards at {}", r, farm.shards, farm.dir.display());
    Ok(())
}

pub fn cmd_rm(model: &str, gc: bool) -> Result<(), Err> {
    let r = parse(model)?;
    let st = store()?;
    let _lock = st.lock()?;
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
    fn variant_id_matches_published() {
        // `variant_id(bundle, 1)` as every published generation is keyed.
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
        assert_eq!(plowrt::dist::push::variant_id(&b), "b49b5c9cf598");
    }

    /// `publish` against a registry that is not there: what fails must be
    /// the argument, never the registry (nor "already published"). The
    /// recipe and asset directory are the smallest `generate` accepts.
    fn publish_refuses(source: &str, ckpt_dir: bool, hf: Option<&str>) -> String {
        let dir = std::env::temp_dir().join(format!(
            "plowrt-publish-args-{}-{}{}",
            std::process::id(),
            source.replace([':', '/'], "_"),
            ckpt_dir
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let assets = dir.join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        for (file, body) in [
            ("model.pkt", "PLOWDEV"),
            ("plow_config.h", "#define X 1"),
            ("weights.json", r#"{"network":"m"}"#),
            (
                "build.json",
                r#"{"pairing":{"hash":"0x541009da5f9890c1"},
                    "features":{"fp8_kv":false,"mxfp4_weights":false,"w8a8":false}}"#,
            ),
        ] {
            std::fs::write(assets.join(file), body).unwrap();
        }
        let recipe = dir.join("r.toml");
        std::fs::write(
            &recipe,
            format!(
                r#"
schema = "plow.recipe.v1"
namespace = "infervisor"
name = "m"
label = "l"
plow_git = "1972fa37ac7a6f632cd5ece0b32fe37ab76eec75"
network = "m"

[target]
vendor = "apple"
isa = "m4"
sku = "M4 Max"
units = 132
mem_bytes = 85899345920
num_gpus = 1
parallel = "tp"
toolchain = "cuda-13.4+gcc-13"

[checkpoint]
source = "{source}"
revision = "8b770f9e69e6b35ef320d4cd70a99a4ab6dd022f"
shards = 1

[cell]
max_ctx = 2048
"#
            ),
        )
        .unwrap();
        let ckpt = dir.join("ckpt");
        std::fs::create_dir_all(&ckpt).unwrap();
        std::fs::write(ckpt.join("model-00001.safetensors"), b"shard").unwrap();
        let to = super::PublishTo {
            url: "http://127.0.0.1:9",
            api_key: "k",
            status: None,
            description: None,
            hf,
            tok_s: None,
            concurrency: None,
        };
        let r = super::Recipe {
            path: &recipe,
            over: Default::default(),
            args: Default::default(),
        };
        let err = super::cmd_publish(r, &assets, None, ckpt_dir.then_some(ckpt.as_path()), &to)
            .unwrap_err();
        std::fs::remove_dir_all(&dir).ok();
        err.to_string()
    }

    #[test]
    fn publish_checks_its_arguments_before_the_registry() {
        // A `dist` checkpoint travels as blobs, and the index needs `--hf`
        // to name the model it is.
        let e = publish_refuses("dist", true, None);
        assert!(e.contains("pass --hf"), "dist bundle without --hf: {e}");
        // `--checkpoint-dir` is what makes a checkpoint travel, whatever the
        // recipe says its source is.
        let e = publish_refuses("hf:o/m", true, None);
        assert!(e.contains("pass --hf"), "hf recipe + --checkpoint-dir: {e}");
        // A `dist` checkpoint with nothing to carry is not publishable.
        let e = publish_refuses("dist", false, Some("o/m"));
        assert!(
            e.contains("lists the files it carries"),
            "dist bundle without its checkpoint: {e}"
        );
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

/// `plowrt catalog show`: every row of the registry's catalog.
#[cfg(feature = "dist")]
pub fn cmd_catalog_show(to: &str) -> Result<(), Err> {
    let sink = dist::sink::sink(to)?;
    let catalog = dist::push::catalog(&*sink)?;
    if catalog.models.is_empty() {
        println!("{}: no models published", sink.describe());
        return Ok(());
    }
    for m in &catalog.models {
        println!(
            "{}/{:<28} {:<9} {}{}",
            m.namespace,
            m.name,
            if m.listed { "listed" } else { "unlisted" },
            m.hf,
            m.description
                .as_deref()
                .map(|d| format!("  — {d}"))
                .unwrap_or_default()
        );
    }
    Ok(())
}

/// `plowrt catalog list|unlist <model>`: one write on the catalog row.
#[cfg(feature = "dist")]
pub fn cmd_catalog_set(to: &str, model: &str, listed: bool) -> Result<(), Err> {
    let r = parse(model)?;
    if r.label.is_some() || r.generation.is_some() {
        return Err(format!(
            "{model}: listing is per model; give `[<namespace>/]<name>` without a label or generation"
        )
        .into());
    }
    let sink = dist::sink::sink(to)?;
    let verb = if listed { "listed" } else { "unlisted" };
    match dist::push::set_listed(&*sink, &r.namespace, &r.name, listed)? {
        dist::push::Listing::Changed => {
            println!(
                "{}/{}: now {verb} at {}",
                r.namespace,
                r.name,
                sink.describe()
            )
        }
        dist::push::Listing::Unchanged => {
            println!(
                "{}/{}: already {verb} at {}",
                r.namespace,
                r.name,
                sink.describe()
            )
        }
    }
    Ok(())
}

/// `plowrt push-runtime`: see [`dist::runtime`].
#[cfg(feature = "dist")]
pub fn cmd_push_runtime(
    releases: &[std::path::PathBuf],
    to: &str,
    opts: &dist::runtime::Options,
) -> Result<(), Err> {
    // Every directory is read and checked before the registry is touched.
    let dirs = releases
        .iter()
        .map(|d| dist::runtime::ReleaseDir::read(d))
        .collect::<Result<Vec<_>, _>>()?;
    let sink = dist::sink::sink(to)?;
    let out = dist::runtime::push_runtime(&*sink, &dirs, opts)?;
    let verb = if opts.dry_run {
        "would publish"
    } else {
        "published"
    };
    for (triple, bytes) in &out.uploaded {
        println!(
            "{verb} plowrt {} {triple} ({} MiB)",
            out.version,
            bytes >> 20
        );
    }
    for triple in &out.adopted {
        println!("already published: plowrt {} {triple}", out.version);
    }
    if !out.latest.is_empty() {
        println!("latest: {}", out.latest);
    }
    if out.install_script {
        println!("{verb} install.sh");
    }
    println!("registry: {}", sink.describe());
    Ok(())
}
