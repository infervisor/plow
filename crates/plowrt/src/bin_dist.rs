//! The `plowrt` distribution subcommands: `pull`, `load`, `show`, `ls`,
//! `upgrade`, `prepare`, `rm`.
//!
//! These run the network; `serve` does not. `resolve_local` is the one entry
//! point `serve` uses, and it reads only the local store — an unpulled model is
//! an error naming the `load` that fixes it, never a silent fetch.

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

    // Weights are never distributed. Resolve them, or say precisely what to do.
    let ckpt = checkpoint
        .or_else(|| RuntimeConfig::get().checkpoint.clone().map(PathBuf::from))
        .unwrap_or_else(|| {
            prepare::farm_path(
                st.root(),
                &resolved.bundle.checkpoint.source,
                &resolved.bundle.checkpoint.revision,
            )
        });
    if !ckpt.is_dir() {
        if !fetch_weights {
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
        }
        return Err(format!(
            "--fetch-weights is not implemented yet; fetch {} at revision {} with \
             `huggingface-cli download` and pass --checkpoint <dir>",
            resolved.bundle.checkpoint.source, resolved.bundle.checkpoint.revision
        )
        .into());
    }

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

pub fn cmd_ls(upgradable: bool) -> Result<(), Err> {
    let st = store()?;
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
