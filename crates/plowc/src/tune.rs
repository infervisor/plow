//! `plowc tune` — inspect and calibrate kernel selection for a hardware target.
//!
//! Separate from `compile` on purpose. `compile` may *read* qualified tuning
//! records and must never write them; if the same command could do both, the
//! thing being measured and the thing doing the measuring stop being separable
//! and a build could quietly calibrate itself against its own output.
//!
//! What this command does today:
//!
//! * **inventory** — for a target, list the kernels the registry says are
//!   executable, group opcodes that share one implementation, and name the
//!   declared-but-undispatched ones. This is the answer to "what could tuning
//!   possibly choose between here", and on NVIDIA the answer for dense GEMM is
//!   "one kernel under three names".
//! * **select** — resolve one shape to a kernel and print the rationale and
//!   calibration tier, so a selection can be explained without rerunning a
//!   compile.
//! * **status** — report what the tuning database holds for the target, and
//!   what is stale and why.
//!
//! What it does not do yet is run benchmarks. That needs a per-op harness with
//! a correctness oracle; measurements taken without one cannot be published
//! (`tunedb` refuses to qualify them), so a benchmark subcommand that could not
//! reach `qualified` would only produce numbers that look authoritative and are
//! not selectable. `runtime/bench/interp_dispatch_floor_nv.cu` is the worked
//! example of a measurement done properly, under `gpulease`, and its record is
//! in `tuning/`.

use std::path::PathBuf;

use kernelcaps::{
    dense_gemm_inventory, select_kernel, HardwareFingerprint, Inventory, NoMeasurements,
    OpSignature, Phase, ProfileId, Rationale,
};
use tunedb::{Digests, TuneStore};

/// What the user asked `tune` to do.
#[derive(Debug, Clone)]
pub enum TuneAction {
    Inventory,
    Select { m: i64, n: i64, k: i64 },
    Status,
}

#[derive(Debug, Clone)]
pub struct TuneOptions {
    /// Repository root, used to locate the interpreter sources to probe.
    pub root: PathBuf,
    /// GPU spec name, resolved through `hwspec::registry`.
    pub gpu: String,
    pub profile: ProfileId,
    pub db: PathBuf,
    pub action: TuneAction,
}

pub fn run(opts: &TuneOptions) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hwspec::registry::lookup(&opts.gpu)
        .ok_or_else(|| format!("unknown GPU {:?}; see hwspec::registry", opts.gpu))?;
    let hw = HardwareFingerprint::from_spec(spec)
        .ok_or_else(|| format!("{} has no ISA level mapping", spec.name))?;

    println!("target      : {} ({})", hw.sku, hw.isa.arch_flag());
    println!("fingerprint : {}", hw.tuning_path());
    println!("profile     : {}", opts.profile.label());
    let caps = hw.caps();
    println!(
        "capabilities: mma_sync={} wgmma={} tcgen05={} tmem={} tma={} mfma={} lanes={}",
        caps.mma_sync, caps.wgmma, caps.tcgen05, caps.tmem, caps.tma, caps.mfma, caps.warp_lanes
    );
    println!();

    // Derived, never declared: the inventory is probed out of the interpreter
    // object this target actually builds. A failure here is reported rather
    // than papered over, because "no kernels" and "could not look" call for
    // very different responses.
    let reg = match dense_gemm_inventory(&opts.root, hw.isa) {
        Ok(inv) => inv,
        Err(e) => {
            println!("could not derive an inventory for {}:", hw.isa.arch_flag());
            println!("  {e}");
            println!();
            println!("an inventory is read out of a built object, so probing needs the");
            println!("toolchain that builds it. There is deliberately no hand-written");
            println!("fallback: a declared kernel list is exactly what drifts from the");
            println!("object being compiled for.");
            return Ok(());
        }
    };
    println!("inventory   : probed from {}", reg.build().label());
    println!("             defines: {}", reg.build().defines.join(" "));
    println!();

    match &opts.action {
        TuneAction::Inventory => inventory(&reg, &hw),
        TuneAction::Select { m, n, k } => select(&reg, &hw, opts.profile, *m, *n, *k)?,
        TuneAction::Status => status(&opts.db, &hw)?,
    }
    Ok(())
}

fn inventory(reg: &Inventory, hw: &HardwareFingerprint) {
    println!("executable kernels ({}):", reg.len());
    for k in reg.iter() {
        let tile = k
            .tile
            .map(|t| format!("{}x{}x{}", t.bm, t.bn, t.bk))
            .unwrap_or_else(|| "-".into());
        println!(
            "  {:>3}  {:<28} tile {:<14} {}",
            k.id.raw(),
            k.id.c_name(),
            tile,
            if k.dispatched { "dispatched" } else { "NO DISPATCH ARM" }
        );
    }

    let groups = reg.alias_groups();
    println!();
    if groups.is_empty() {
        println!("aliases     : none — every opcode is a distinct implementation.");
    } else {
        println!("aliases     : opcodes reaching the SAME body. Ranking within a group");
        println!("              measures dispatch noise, not kernels.");
        for (hash, members) in &groups {
            let names: Vec<&str> = members.iter().map(|k| k.id.c_name()).collect();
            println!("  {hash}  <-  {}", names.join(", "));
        }
        if hw.isa.vendor() == hwspec::Vendor::Nvidia {
            println!();
            println!("              on NVIDIA the tile is a compile-time macro per interpreter");
            println!("              object, so the real tuning axis here is which object is");
            println!("              built, not which opcode is emitted.");
        }
    }

    let absent = reg.declared_but_absent();
    if !absent.is_empty() {
        println!();
        println!("declared but not dispatched in this build:");
        for k in absent {
            println!("  {:>3}  {}", k.id.raw(), k.id.c_name());
        }
    }
}

fn select(
    reg: &Inventory,
    hw: &HardwareFingerprint,
    profile: ProfileId,
    m: i64,
    n: i64,
    k: i64,
) -> Result<(), Box<dyn std::error::Error>> {
    let op = OpSignature::gemm(Phase::Prefill, m, n, k);
    println!("op          : dense matmul {m}x{n}x{k} ({:?})", op.shape.class());

    // The SAME ranking the compiler uses. A tune command that reports a
    // different kernel than the build would emit is worse than no command: it
    // looks authoritative and quietly disagrees.
    let spec = hwspec::registry::lookup(&hw.sku).expect("fingerprint came from this spec");
    let n_units = hw.units;
    let realization = select_kernel(reg, &op, hw, profile, &NoMeasurements, |kernel| {
        tile_cost(spec, kernel, m, n, k, n_units)
    })?;

    println!("selected    : {} ({})", realization.kernel.c_name(), realization.kernel.raw());
    if let Some(t) = realization.tile {
        println!("tile        : {}x{}x{}", t.bm, t.bn, t.bk);
    }
    match &realization.rationale {
        Rationale::Measured { median_ns } => {
            println!("rationale   : measured, median {median_ns:.1} ns")
        }
        Rationale::Analytical { cost } => {
            println!("rationale   : analytical cold start (cost {cost})");
            println!("              no measurement matched this hardware and build");
        }
        Rationale::OnlyCandidate => println!("rationale   : the only legal candidate"),
        Rationale::AliasCollapsed { members } => {
            println!("rationale   : {members} opcodes share one implementation");
            println!("              returning the canonical opcode; there is nothing to rank");
        }
    }
    println!("tier        : {}", realization.rationale.tier().label());
    if !realization.fallbacks.is_empty() {
        let names: Vec<&str> = realization.fallbacks.iter().map(|f| f.c_name()).collect();
        println!("fallbacks   : {}", names.join(", "));
    }
    Ok(())
}

fn status(db: &PathBuf, hw: &HardwareFingerprint) -> Result<(), Box<dyn std::error::Error>> {
    let store = TuneStore::new(db.clone());
    let cell = hw.tuning_path();
    let all = store.load_kernels(&cell)?;

    println!("database    : {}", db.display());
    println!("cell        : {cell}");
    if all.is_empty() {
        println!();
        println!("no kernel measurements for this cell.");
        println!("selection will use the analytical model and report tier `portable`.");
        return Ok(());
    }

    let qualified = all.iter().filter(|r| r.state.is_selectable()).count();
    println!("records     : {} ({qualified} qualified)", all.len());

    // Report against the digests of the records themselves, so this shows what
    // the store holds rather than asserting a build identity this command does
    // not have.
    if let Some(first) = all.first() {
        let want = Digests { ..first.digests.clone() };
        let (best, stale) = store.best_for(&cell, &want)?;
        println!("selectable  : {} op cases", best.len());
        for (case, rec) in &best {
            println!(
                "  {case:<24} {:<24} median {:.1} ns  ({} samples)",
                rec.kernel_name, rec.stats.median_ns, rec.stats.samples
            );
        }
        if !stale.is_empty() {
            println!();
            println!("stale       : {} record(s) exist but cannot be used", stale.len());
            for s in &stale {
                println!("  {:<24} {:<24} changed: {}", s.op_case, s.kernel_name, s.changed.join(", "));
            }
        }
    }
    Ok(())
}

// The tile cost model and its LDS-budget helper live in `devgen`, the crate
// that owns the device-blob emitters — they are one ranking shared by `plowc
// tune` and the emitters' `pick_tile`, and `devgen` sits below both. Re-exported
// here so `plowc tune`'s callers and any external users keep the same paths.
pub use devgen::{gemm_lds_bytes, tile_cost};

/// Parse a `--profile` value.
pub fn parse_profile(s: &str) -> Result<ProfileId, String> {
    Ok(match s {
        "decode_dense" => ProfileId::DecodeDense,
        "decode_moe" => ProfileId::DecodeMoe,
        "decode_latent" => ProfileId::DecodeLatent,
        "prefill_dense" => ProfileId::PrefillDense,
        "prefill_moe" => ProfileId::PrefillMoe,
        "recurrent_mamba" => ProfileId::RecurrentMamba,
        "portable_reference" => ProfileId::PortableReference,
        other => return Err(format!("unknown profile {other:?}")),
    })
}

/// Parse an `M,N,K` triple.
pub fn parse_shape(s: &str) -> Result<(i64, i64, i64), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err(format!("expected M,N,K — got {s:?}"));
    }
    let v: Result<Vec<i64>, _> = parts.iter().map(|p| p.trim().parse::<i64>()).collect();
    let v = v.map_err(|e| format!("bad shape {s:?}: {e}"))?;
    if v.iter().any(|d| *d <= 0) {
        return Err(format!("shape dimensions must be positive: {s:?}"));
    }
    Ok((v[0], v[1], v[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_parsing_rejects_malformed_input() {
        assert_eq!(parse_shape("4096,4096,4096").unwrap(), (4096, 4096, 4096));
        assert!(parse_shape("4096,4096").is_err());
        assert!(parse_shape("4096,0,4096").is_err(), "zero is not a shape");
        assert!(parse_shape("a,b,c").is_err());
    }

    #[test]
    fn profile_parsing_round_trips_every_variant() {
        for p in [
            ProfileId::DecodeDense,
            ProfileId::DecodeMoe,
            ProfileId::DecodeLatent,
            ProfileId::PrefillDense,
            ProfileId::PrefillMoe,
            ProfileId::RecurrentMamba,
            ProfileId::PortableReference,
        ] {
            assert_eq!(parse_profile(p.label()).unwrap(), p);
        }
        assert!(parse_profile("nope").is_err());
    }

    /// An unknown GPU must fail rather than fall back to a default spec.
    #[test]
    fn an_unknown_target_is_an_error() {
        let opts = TuneOptions {
            root: PathBuf::from("."),
            gpu: "Totally Fake GPU".into(),
            profile: ProfileId::PrefillDense,
            db: PathBuf::from("tuning"),
            action: TuneAction::Inventory,
        };
        assert!(run(&opts).is_err());
    }

    /// Every registered SKU either has declared kernels or reports that it has
    /// none — neither path may panic.
    #[test]
    fn every_registered_sku_is_handled() {
        for spec in hwspec::registry::ALL {
            let opts = TuneOptions {
                root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
                gpu: spec.name.to_string(),
                profile: ProfileId::PrefillDense,
                db: PathBuf::from("tuning"),
                action: TuneAction::Inventory,
            };
            run(&opts).unwrap_or_else(|e| panic!("{} failed: {e}", spec.name));
        }
    }
}
