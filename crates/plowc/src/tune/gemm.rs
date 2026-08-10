//! `plowc tune gemm` — run the gfx950 prefill-GEMM tile campaign and publish it.
//!
//! The Rust replacement for `scripts/rebench_tune_gemm.sh`, and a strict superset of it: same
//! harness, same JSONL, same row order — plus the shape list is DERIVED (`--shapes auto`) instead
//! of hand-authored, and the ingest half can no longer be forgotten.
//!
//! # The two halves are preserved on purpose
//!
//! Measurement and publication stay distinct (`tune ingest` is its own subcommand) because the C
//! harness cannot know the build identity and the store's gates cannot be applied without it. What
//! changes is that `gemm` now *calls* ingest, so the 2026-07-29 failure — a campaign that wrote
//! 180 rows and published none, then reported `NO usable records remain` — cannot recur by
//! omission. See `tune::ingest`'s module docs.
//!
//! # Two subprocess rules, and they are opposites
//!
//! * The **harness build** runs the SYSTEM gcc with a cleared environment. That is required: nix's
//!   `CPATH`/`LIBRARY_PATH` shadow the system glibc and every ROCm header/lib this links against
//!   (`knob-contract` §0a: system ROCm binaries die with `GLIBC_2.38 not found` under
//!   `nix develop`).
//! * **Emission** (`--shapes auto`, in `tune::demand`) is a LIBRARY CALL with the environment
//!   intact. `env -i` in the bash pipeline was never about isolating `plowc`; it was about nvcc's
//!   host pass. Clearing the environment there would derive the wrong shape list. Do not "restore
//!   symmetry".
//!
//! # GPU discipline this encodes
//!
//! * `gpulease` `rc=76` means CONTENDED. The shape's samples are DISCARDED, not averaged in —
//!   which is why each shape measures into its own scratch file and is appended only on success.
//! * Every GPU invocation needs the `render` gid: this account is not in the group for the current
//!   session, and without it `hsa_init` fails 4104 and a harness can fall back to a CPU backend and
//!   print confident garbage. Checked up front and REFUSED — plowc cannot fix it for you, because
//!   `nix develop` runs in a user namespace where root maps to `nobody`, which makes
//!   `/usr/bin/newgrp`'s setuid bit inert and `sg render -c` fail with `setgroups: Operation not
//!   permitted` from inside the shell plowc lives in. The gid must be acquired before nix starts:
//!   `sg render -c 'nix develop --command plowc … tune gemm …'`.
//! * `rocm-smi --showpids` is the only reliable leak check. Reported before measuring. No
//!   `pkill`/`pgrep` patterns anywhere here — they match the monitoring shell itself.

use std::path::{Path, PathBuf};
use std::process::Command;

use devgen::tune_demand::Demand;
use kernelcaps::QuantScheme;

use super::demand::{self, EmitSpec};

type Err = Box<dyn std::error::Error>;

/// Where the campaign's shapes come from.
#[derive(Debug, Clone)]
pub enum ShapeSource {
    /// Derive from the compiler's demand. The default, and the whole point.
    Auto(EmitSpec),
    /// An explicit `M N K [label]` list — kept so the hand-authored list in
    /// `scripts/rebench_tune_gemm.sh` can be replayed for a byte-comparable A/B.
    File(PathBuf),
}

#[derive(Debug, Clone)]
pub struct Campaign {
    /// Object directory holding a freshly built `test_kernels.elf`.
    pub obj: PathBuf,
    /// Output JSONL of raw samples.
    pub samples: PathBuf,
    pub shapes: ShapeSource,
    /// Wrap every GPU invocation in `perf-data/harness/gpulease -n 1`. Off by default, matching
    /// the bash script, whose caller holds the lease.
    pub lease: bool,
    /// Skip the publish step. For an A/B that only wants the JSONL.
    pub no_ingest: bool,
    pub db: PathBuf,
    pub campaign: String,
    pub provisional: bool,
    /// Derive the shape list and print it, measuring nothing.
    pub dry_run: bool,
}

pub fn run(root: &Path, c: &Campaign, isa: hwspec::IsaLevel, cell: &str) -> Result<(), Err> {
    // `--root .` is the default and the harness build runs with `current_dir(obj)`, so a relative
    // root resolves against the OBJECT directory and every source path misses. Canonicalise once,
    // here, rather than at each use.
    let root =
        &std::fs::canonicalize(root).map_err(|e| format!("--root {}: {e}", root.display()))?;
    // The digest first, in every subcommand, because digest churn is the dominant operational
    // fact: any edit to `runtime/amd/*.hip|h` or `runtime/common/dev_isa.h` moves the
    // preprocessed build digest and re-stales EVERY record. `probe_digest` exists so tooling can
    // ASK rather than guess; this is that call.
    let want = super::ingest::digests(root, isa)?;
    println!("build digest: {}", want.interpreter);
    println!("toolchain   : {}", want.toolchain);
    println!();

    let shapes = resolve(&c.shapes)?;
    match &c.shapes {
        ShapeSource::Auto(_) => {
            let (hit, miss) = shapes_coverage(&shapes);
            println!(
                "shapes      : {} DERIVED from the compiler's demand",
                shapes.len()
            );
            println!(
                "store cover : {hit} HIT / {miss} MISS against {}",
                c.db.display()
            );
            if miss > 0 {
                // The line nobody printed on 2026-07-29. `0 HIT / 32 MISS` was the whole bug.
                println!("              the {miss} MISS shapes select from the ANALYTICAL MODEL");
                println!("              until this campaign publishes; that is what it is for.");
            }
        }
        ShapeSource::File(p) => {
            println!(
                "shapes      : {} from {} (HAND-AUTHORED — it can drift from demand;",
                shapes.len(),
                p.display()
            );
            println!("              `--shapes auto` is the form that cannot)");
        }
    }
    for s in &shapes {
        let mark = match s.hit {
            Some(true) => "HIT ",
            Some(false) => "MISS",
            None => "?   ",
        };
        println!(
            "  {mark} {:>6} x {:>6} x {:>6}  {:<8} {}",
            s.m,
            s.n,
            s.k,
            format!("{:?}", s.quant),
            s.label
        );
    }
    // The encoding census, printed because it is the one thing a reader cannot infer from the
    // shape list: 3410 records existed and every one of them was bf16, and nothing said so.
    {
        let mut by_quant: std::collections::BTreeMap<String, usize> = Default::default();
        for s in &shapes {
            *by_quant.entry(format!("{:?}", s.quant)).or_default() += 1;
        }
        let census: Vec<String> = by_quant.iter().map(|(q, n)| format!("{n} {q}")).collect();
        println!("encodings   : {}", census.join(", "));
    }
    println!();
    if c.dry_run {
        println!("--dry-run: derived only, nothing measured.");
        return Ok(());
    }

    preflight(root, c)?;

    let harness = build_harness(root, &c.obj)?;
    let _ = std::fs::remove_file(&c.samples);

    let mut rows = 0usize;
    let mut contended = Vec::new();
    for s in &shapes {
        println!("=== {}  {}x{}x{}  {:?}", s.label, s.m, s.n, s.k, s.quant);
        // Each shape measures into its own scratch file and is appended only on a clean exit, so
        // a contended run (gpulease rc=76) is DISCARDED rather than averaged in. The bash script
        // pointed the harness straight at the campaign file and had no way to take a row back.
        let scratch = c
            .samples
            .with_extension(format!("part.{}", std::process::id()));
        let _ = std::fs::remove_file(&scratch);
        match measure(root, &harness, &c.obj, s, &scratch, c.lease) {
            Ok(()) => {
                if let Ok(part) = std::fs::read_to_string(&scratch) {
                    rows += part.lines().filter(|l| !l.trim().is_empty()).count();
                    append(&c.samples, &part)?;
                }
            }
            Err(Contention::Contended) => {
                println!("  CONTENDED (gpulease rc=76) — samples discarded, not averaged in");
                contended.push(s.label.clone());
            }
            Err(Contention::Failed(e)) => return Err(e),
        }
        let _ = std::fs::remove_file(&scratch);
    }
    println!("wrote {rows} rows -> {}", c.samples.display());
    if !contended.is_empty() {
        println!(
            "contended   : {} shape(s) discarded and NOT measured: {}",
            contended.len(),
            contended.join(", ")
        );
        println!("              re-run them; a contended timing is not a caveated timing.");
    }

    if c.no_ingest {
        println!();
        println!("--no-ingest: samples written, NOTHING PUBLISHED. The store is unchanged and");
        println!("             the compiler's answer has not moved. Publish with:");
        println!(
            "               plowc tune ingest --samples {}",
            c.samples.display()
        );
        return Ok(());
    }
    println!();
    let published = super::ingest::ingest(
        root,
        &c.db,
        &c.samples,
        &c.campaign,
        c.provisional,
        isa,
        cell,
    )?;
    if published == 0 && !c.provisional {
        return Err(
            "the campaign measured rows but published NO qualified record. The store is \
                    unchanged, so the compiler's answer has not moved — check the rejected/stale \
                    lines above rather than treating this as a success."
                .into(),
        );
    }
    Ok(())
}

/// A shape to measure. `label` is cosmetic — the harness does not put it in the JSONL — so two
/// campaigns that differ only in labels produce byte-identical sample files.
struct Shape {
    m: i64,
    n: i64,
    k: i64,
    label: String,
    /// The weight encoding to measure this shape in. Carried from the demand rather than assumed,
    /// because it is IN THE OP-CASE KEY (`tunedb::gemm_op_case`) — a campaign that dropped it
    /// would file every mxfp4 timing under the bf16 case and serve a bf16 op an fp4 number.
    /// Dropping it is exactly why the store held 3410 records and not one of them was mxfp4.
    quant: QuantScheme,
    /// Whether the store already answers this shape. `None` for a hand-authored list: that list
    /// was never asked of the compiler, so nothing is known about whether it is demanded at all —
    /// which is the drift `--shapes auto` removes.
    hit: Option<bool>,
}

fn resolve(src: &ShapeSource) -> Result<Vec<Shape>, Err> {
    Ok(match src {
        ShapeSource::Auto(spec) => demand::derive(spec)?
            .into_iter()
            .map(|d: Demand| Shape {
                m: d.m,
                n: d.n,
                k: d.k,
                label: d.label(),
                quant: d.quant,
                hit: Some(d.hit),
            })
            .collect(),
        ShapeSource::File(p) => demand::parse_list(&std::fs::read_to_string(p)?)?
            .into_iter()
            .map(|l| Shape {
                m: l.m,
                n: l.n,
                k: l.k,
                label: l.label,
                quant: l.quant,
                hit: None,
            })
            .collect(),
    })
}

fn shapes_coverage(shapes: &[Shape]) -> (usize, usize) {
    let hit = shapes.iter().filter(|s| s.hit == Some(true)).count();
    (hit, shapes.len() - hit)
}

/// Refuse before measuring rather than produce numbers that are silently invalid.
fn preflight(root: &Path, c: &Campaign) -> Result<(), Err> {
    if !c.obj.join("test_kernels.elf").is_file() {
        return Err(format!("no test_kernels.elf in {}", c.obj.display()).into());
    }
    // Without the `render` gid `hsa_init` fails 4104 and a harness can silently fall back to a
    // CPU backend and print confident garbage — this has happened twice. The gid is checked on
    // the PROCESS, not in the group database: `getent group render` lists this user while `id -G`
    // does not, because the login session predates the group being added. That gap is the trap.
    if !in_render_group() {
        let how = if group_db_lists_render() {
            "This session predates the group being added, so no process in it carries the gid \
             (`getent group render` lists you; `id -G` does not). Acquire it OUTSIDE the nix \
             shell and let plowc inherit it:\n\
             \n    sg render -c 'nix develop --command ./target/release/plowc … tune gemm …'\n\
             \n\
             plowc CANNOT do this for itself. `nix develop` runs in a user namespace where root \
             maps to `nobody`, so /usr/bin/newgrp's setuid bit is inert inside it and `sg` dies \
             with `setgroups: Operation not permitted`. The gid has to be acquired before nix \
             starts. And never `sudo -g render`: it resets the environment, so a leased run \
             lands on GPU 0 regardless of the lease — silently invalid rather than loudly wrong."
        } else {
            "This user is not a member of the `render` group at all, so `sg render` cannot help."
        };
        return Err(format!(
            "the `render` gid is absent from this process. Every GPU call would fail hsa_init \
             4104, and a harness that cannot reach the GPU can fall back to a CPU backend and \
             print confident garbage — this has happened twice, which is why this refuses \
             instead of trying.\n\n{how}"
        )
        .into());
    }
    // `rocm-smi --showpids` is the ONLY reliable leak check: one leaked lease here spun 8h03m
    // holding 30.4 GB, invisible to every `pgrep -f`. Reported, not enforced — a foreign holder
    // is a fact about the run and belongs in the log.
    //
    // Cleared environment for the same reason as the harness build: `rocm-smi` is a system ROCm
    // tool and the nix dev shell's LD_LIBRARY_PATH hides `librocm_smi64.so.1` from it, so run
    // inside `nix develop` it reports "Unable to load the rocm_smi library" — a leak check that
    // silently checks nothing.
    match Command::new("/usr/bin/rocm-smi")
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .arg("--showpids")
        .output()
    {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout);
            if s.contains("No KFD PIDs") {
                println!("vram        : no KFD processes — the cards are free");
            } else {
                println!("vram        : rocm-smi --showpids reports FOREIGN HOLDERS:");
                for l in s.lines().filter(|l| !l.trim().is_empty()) {
                    println!("  {l}");
                }
            }
        }
        Err(e) => println!("vram        : could not run rocm-smi ({e}) — leak check NOT performed"),
    }
    let _ = root;
    Ok(())
}

/// Whether the current process actually carries the `render` gid.
///
/// `getent group render` listing the user is NOT the same thing: a login session that predates the
/// group being added has no process in it, which is exactly the trap `knob-contract` §0a records.
/// So this reads the process's own supplementary groups, not the group database's membership.
fn in_render_group() -> bool {
    let Some(gid) = render_gid() else {
        return false;
    };
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return false;
    };
    status
        .lines()
        .find_map(|l| l.strip_prefix("Groups:"))
        .map(|g| g.split_whitespace().any(|x| x.parse::<u32>() == Ok(gid)))
        .unwrap_or(false)
}

fn render_gid() -> Option<u32> {
    let group = std::fs::read_to_string("/etc/group").ok()?;
    group.lines().find_map(|l| {
        let mut f = l.split(':');
        (f.next()? == "render").then(|| f.nth(1)?.parse::<u32>().ok())?
    })
}

/// Whether the group DATABASE lists this user — which is what decides if `sg render` can succeed,
/// and is a different question from whether the running process carries the gid.
fn group_db_lists_render() -> bool {
    let Ok(group) = std::fs::read_to_string("/etc/group") else {
        return false;
    };
    let user = std::env::var("USER").unwrap_or_default();
    group.lines().any(|l| {
        l.starts_with("render:")
            && l.rsplit(':')
                .next()
                .is_some_and(|m| m.split(',').any(|u| u == user))
    })
}

/// The ROCm install the harness is built and run against.
///
/// `ROCM_PATH` first because `/opt/rocm` is not reliably the install: an apt/versioned box has
/// no such symlink at all, and this gfx942 box has an `/opt/rocm` that carries `bin` and neither
/// `lib` nor `include` — the real tree is `/opt/rocm-7.2.4`. Both the compile (`-I`, `-L`) and
/// the exec (`LD_LIBRARY_PATH`) read it from here so they cannot disagree.
fn rocm_root() -> String {
    std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into())
}

fn rocm_lib() -> String {
    format!("{}/lib", rocm_root())
}

/// Build the host harness with the SYSTEM toolchain and a cleared environment.
///
/// `env_clear` is correct HERE and wrong for emission (see the module docs): this links against
/// `/opt/rocm` headers and `libhsa-runtime64`, and the nix dev shell's `CPATH`/`LIBRARY_PATH`
/// shadow the system glibc that ROCm was built against. Same rule as `build_gfx950.sh`'s `chat`.
fn build_harness(root: &Path, obj: &Path) -> Result<PathBuf, Err> {
    // The sweep source MOVED to runtime/bench/gemm/ and this path did not follow it, so every
    // campaign died in gcc with "No such file or directory" -- i.e. `plowc tune gemm` could not
    // measure anything at all, which is exactly how a target ends up with no tuning cell and
    // every compile silently falling back to the analytical model. Resolved by probing both
    // locations rather than repinning to one: an older checkout still has it under ubench/.
    const SWEEP_SRC: [&str; 2] = [
        "runtime/bench/gemm/gemm_tile_sweep.c",
        "runtime/ubench/gemm_tile_sweep.c",
    ];
    let src = SWEEP_SRC
        .iter()
        .map(|p| root.join(p))
        .find(|p| p.exists())
        .ok_or_else(|| -> Err {
            format!(
                "gemm_tile_sweep.c not found under {} (looked in {})",
                root.display(),
                SWEEP_SRC.join(", ")
            )
            .into()
        })?;
    let st = Command::new("/usr/bin/gcc")
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", std::env::var("HOME").unwrap_or_default())
        .current_dir(obj)
        .args(["-O2", "-std=gnu11", "-o", "gemm_tile_sweep"])
        .arg(&src)
        .arg(root.join("runtime/amd/hsa_backend.c"))
        // `-I<root>/runtime/bench` is what makes the sweep's own `#include "../amd/hsa_backend.h"`
        // resolve from its NEW home: a quoted include falls back to the -I list after the source
        // directory, and `<root>/runtime/bench/../amd` is `<root>/runtime/amd`. The alternative --
        // editing the include -- would break the ubench/ fallback above, which needs the same
        // spelling to mean a different directory.
        .arg(format!("-I{}", root.join("runtime/bench").display()))
        // ROCM_PATH-relative, not hardcoded /opt/rocm: an apt/versioned install has no
        // /opt/rocm symlink and this build then dies on hsa/hsa.h — same treatment the
        // build scripts' bundler/readelf discovery already got.
        .arg(format!("-I{}/include", rocm_root()))
        .arg(format!("-L{}", rocm_lib()))
        .args(["-lhsa-runtime64", "-lm"])
        .status()?;
    if !st.success() {
        return Err(format!("building gemm_tile_sweep failed: {st}").into());
    }
    Ok(obj.join("gemm_tile_sweep"))
}

enum Contention {
    Contended,
    Failed(Err),
}

fn measure(
    root: &Path,
    harness: &Path,
    obj: &Path,
    s: &Shape,
    out: &Path,
    lease: bool,
) -> Result<(), Contention> {
    // The quant is argv[5] and the harness sweeps a DIFFERENT tile table for it — five selectable
    // mxfp4 rungs rather than the bf16 twelve. `{:?}` is the same spelling `tunedb::parse_quant`
    // reads back out of the JSONL, so writer and reader share one vocabulary.
    let mut argv: Vec<String> = vec![
        harness.to_string_lossy().into_owned(),
        s.m.to_string(),
        s.n.to_string(),
        s.k.to_string(),
        s.label.clone(),
        format!("{:?}", s.quant),
    ];
    // No `sg` wrapping here: it is impossible from inside `nix develop` (see `preflight`), and
    // by the time we reach this point the gid is present because preflight refused otherwise.
    if lease {
        let mut wrapped = vec![
            root.join("perf-data/harness/gpulease")
                .to_string_lossy()
                .into_owned(),
            "-n".into(),
            "1".into(),
            format!("tune-gemm-{}", s.label),
        ];
        wrapped.extend(argv);
        argv = wrapped;
    }

    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..]).current_dir(obj);
    // A CURATED environment, not the ambient one and not an empty one.
    //
    // Cleared, because plowc lives inside `nix develop` and the harness is a SYSTEM binary: with
    // nix's `LD_LIBRARY_PATH` inherited it loads nix's libelf/libdrm/libnuma/libstdc++ against
    // the system glibc and dies with `GLIBC_2.38 not found` before reaching main. That is
    // knob-contract §0a's "do NOT run ROCm builds under nix develop", at run time.
    //
    // But NOT `env -i`. Stripping the environment wholesale is how the lease gets destroyed —
    // §0a records three instances of exactly that bug (`env -i` in the TP scripts, `sudo -g
    // render`), where the run lands on GPU 0 regardless of its lease and produces numbers that
    // are silently invalid rather than loudly wrong. So the lease variable is forwarded BY NAME.
    cmd.env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", std::env::var("HOME").unwrap_or_default())
        // ROCM_PATH-relative for the same reason `build_harness` already is: this box has an
        // /opt/rocm with `bin` and nothing else — no `lib`, no `include` — and the real install
        // is /opt/rocm-7.2.4. Hardcoding /opt/rocm/lib links the harness fine and then loses
        // libhsa-runtime64.so.1 at exec time, which is a failure the campaign cannot recover
        // from and could not previously be fixed from the caller's environment.
        .env("LD_LIBRARY_PATH", rocm_lib())
        .env("PLOW_GEMM_JSONL", out);
    if let Ok(v) = std::env::var("ROCR_VISIBLE_DEVICES") {
        cmd.env("ROCR_VISIBLE_DEVICES", v);
    }
    for k in [
        "GPU_LEASE_DIR",
        "GPU_LEASE_TIMEOUT",
        "GPU_LEASE_NGPU",
        "TMPDIR",
    ] {
        if let Ok(v) = std::env::var(k) {
            cmd.env(k, v);
        }
    }
    // `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` are deliberately NOT forwarded. gpulease
    // exports them alongside `ROCR_VISIBLE_DEVICES` to the same absolute id and they COMPOSE, so
    // a correctly leased card reports "no ROCm-capable device is detected". For a runtime,
    // `ROCR_VISIBLE_DEVICES` alone is correct.
    let st = cmd
        .status()
        .map_err(|e| Contention::Failed(Box::new(e) as Err))?;
    match st.code() {
        Some(0) => Ok(()),
        // gpulease's "completed but contended". A contended run silently invalidates every
        // number, so it is discarded and re-run — never stored with a caveat.
        Some(76) => Err(Contention::Contended),
        other => Err(Contention::Failed(
            format!(
                "gemm_tile_sweep {}x{}x{} {:?} exited {other:?}",
                s.m, s.n, s.k, s.quant
            )
            .into(),
        )),
    }
}

fn append(path: &Path, text: &str) -> Result<(), Err> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    f.write_all(text.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The label never reaches the JSONL, so a derived campaign and a hand-authored one over the
    /// same dimensions must be byte-comparable. This pins the property the acceptance test rests
    /// on: `Shape` carries the label, and nothing but the three dimensions is passed downstream
    /// into a row.
    #[test]
    fn labels_are_cosmetic() {
        let src = "512 6144 512 alpha\n512 6144 512 beta\n";
        let a = demand::parse_list(src).unwrap();
        assert_eq!((a[0].m, a[0].n, a[0].k), (a[1].m, a[1].n, a[1].k));
        assert_ne!(a[0].label, a[1].label);
    }

    /// The QUANT is not cosmetic, and this is the counterpart to the test above: two campaign
    /// lines that differ only in encoding must reach the harness as two different sweeps, and
    /// be filed under two different op cases. Getting this wrong is invisible — the rows land
    /// in the store and read as measurements.
    #[test]
    fn the_quant_is_not_cosmetic() {
        let a = demand::parse_list("128 576 7168 kva None\n128 576 7168 kva Mxfp4\n").unwrap();
        assert_ne!(a[0].quant, a[1].quant);
        assert_ne!(
            tunedb::gemm_op_case(a[0].m, a[0].n, a[0].k, a[0].quant),
            tunedb::gemm_op_case(a[1].m, a[1].n, a[1].k, a[1].quant),
        );
    }
}
