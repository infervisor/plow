//! `--shapes auto`: derive a campaign's shape list from the compiler's own demand.
//!
//! # Why `auto` is the default rather than an option
//!
//! `scripts/rebench_tune_gemm.sh` carries its shape list as a shell string **authored by hand**.
//! A hand-maintained list drifts from the compiler's real demand, and no guard can catch the
//! drift: a guard reads the records that exist, never the lookups that missed, so
//! `tuned_tile_selection` stays green while a whole model's prefill selects from the analytical
//! model and the calibration tier still reads `measured`.
//!
//! The case where that costs real time is **Gemma-31B**, whose census reads **100 HIT / 1073
//! MISS** and where dense GEMM genuinely dominates prefill. It was *discovered* on GLM-5.2 (32
//! shapes asked, 32 MISS), but the GLM impact was subsequently measured at **|Δ| < 0.3% of TTFT**
//! — GLM prefill is MLA flash plus a 256-expert MoE, and dense GEMM is ~1.3% of its 4k TTFT. The
//! mechanism is general; the price tag is Gemma's. `devgen::tune_demand` carries the full record.
//!
//! Deriving the list from the lookups is the only thing that closes the drift, so `auto` is the
//! default here.
//!
//! # And it is not the first thing to check
//!
//! Coverage is the *second* question. The first is whether the store's records are keyed to the
//! object about to ship at all — see `tune::status`, which reports the digest census and treats
//! "records exist, none selectable" as a failure. A 100%-stale store reads as 0 HIT on every
//! shape regardless of how well the campaign covered them, and that has happened twice.
//!
//! # Why emission is a LIBRARY CALL and not a subprocess
//!
//! The bash pipeline shells out to `plowc --emit devblob` under `env -i`, and it is tempting to
//! read that as isolation this code should preserve. **It is not.** `env -i` exists in these
//! scripts for exactly one reason: nix's `CPATH` points **nvcc's host pass** at glibc headers that
//! collide with the CUDA math headers, so the *compiler-driver* subprocesses have to be run with a
//! cleared environment. `plowc` is a nix-linked Rust binary that reads a checkpoint and writes a
//! packet; it has never had that problem, and clearing its environment would in fact *break* it —
//! `PLOW_MLA_PREFILL`, `PLOW_FP8`, `GLM_*`, `PLOW_TUNEDB` are all read from the environment and
//! they are precisely the knobs that decide which shapes get asked about.
//!
//! Stated here because "restore symmetry — wrap emission in `env_clear()` too" is a plausible
//! future cleanup that would silently derive the WRONG shape list: a cleared environment emits the
//! default ladder, not the one being shipped.
//!
//! `kernelcaps::probe` (which does drive `hipcc`) already calls `env_clear()` at its own call
//! site, which is where that rule belongs.

use std::path::{Path, PathBuf};

use devgen::tune_demand::{self, Demand};

/// The compile whose demand is being observed.
///
/// Deliberately the same fields as a `plowc --emit devblob` invocation, populated from the same
/// top-level flags: `plowc --hf-dir X --max-ctx C --n-cu N --num-gpus G tune gemm` derives the
/// demand of *that* compile. A shape list is only correct for one configuration — the ladder is
/// part of the demand — so the configuration has to be stated, not defaulted.
#[derive(Debug, Clone)]
pub struct EmitSpec {
    pub hf_dir: PathBuf,
    pub ctx: u32,
    pub n_cu: u32,
    pub tp: u32,
    pub gpu: String,
    pub arch: String,
    /// Tuning store the emit reads. Passed through so the HIT/MISS column reports against the
    /// store the campaign is about to write to.
    pub db: PathBuf,
}

/// Run the emit and return the distinct dense-GEMM shapes it asked the store about.
///
/// The packet is written to a scratch path and discarded: this is an *observation* of a compile,
/// not a build. Anything else would make `tune` a second emitter that can disagree with the first.
pub fn derive(spec: &EmitSpec) -> Result<Vec<Demand>, Box<dyn std::error::Error>> {
    if !spec.hf_dir.join("config.json").is_file() {
        return Err(format!(
            "--shapes auto needs a checkpoint to derive demand from: no config.json under {}. \
             Pass `--hf-dir <checkpoint>` before the `tune` subcommand, or `--shapes <file>` to \
             supply a list explicitly.",
            spec.hf_dir.display()
        )
        .into());
    }

    // Same environment carriage as `run_devblob`: `devgen::pick_tile` reads the store through
    // `PLOW_TUNEDB`, so the HIT/MISS column reflects the real store rather than a guess.
    std::env::set_var("PLOW_TUNEDB", spec.db.as_os_str());

    let scratch = scratch_pkt();
    tune_demand::start_recording();
    // The Lean gates are skipped rather than run: they are read-only with respect to the emitted
    // bytes (`run_devblob` says so), they cost minutes, and nothing here consumes a certificate.
    devgen::run_verified(
        devgen::EmitArgs {
            dir: spec.hf_dir.clone(),
            ctx: spec.ctx,
            out: scratch.to_str().ok_or("non-UTF8 scratch path")?.to_string(),
            n_cu: spec.n_cu,
            tp: spec.tp.max(1),
            block_spec: std::env::var("PLOW_BLOCK").ok().filter(|s| !s.is_empty()),
            embed_cubin: None,
            embed_hsaco: None,
            rope_gen: true,
            l2_layout: None,
            gpu: spec.gpu.clone(),
            arch: spec.arch.clone(),
            emit_cfg: None,
        },
        Some(devgen::skip_hook(
            "plowc tune: deriving shape demand, not building an asset",
        )),
    );
    let log = tune_demand::take();
    cleanup(&scratch);

    if log.is_empty() {
        return Err(
            "the emit asked the tuning store about no dense GEMM at all. Either this \
                    model has no prefill GEMM on this path, or the emitter reached `pick_tile` \
                    through a route that does not consult the store — which is the failure this \
                    command exists to detect, so it is an error and not an empty campaign."
                .into(),
        );
    }
    Ok(tune_demand::distinct(log))
}

/// Summary line: how much of the derived demand the store can already answer.
///
/// The number that mattered on 2026-07-29 was `0 HIT / 32 MISS`, and it was invisible because
/// nothing printed it.
pub fn coverage(shapes: &[Demand]) -> (usize, usize) {
    let hit = shapes.iter().filter(|d| d.hit).count();
    (hit, shapes.len() - hit)
}

/// Render the derived demand in the whitespace format `--shapes <file>` reads, so a derived list
/// can be diffed against the hand-authored one in `scripts/rebench_tune_gemm.sh`.
pub fn render(shapes: &[Demand]) -> String {
    let mut s = String::new();
    for d in shapes {
        s.push_str(&format!(
            "{} {} {}    {}    {:?}\n",
            d.m,
            d.n,
            d.k,
            d.label(),
            d.quant
        ));
    }
    s
}

/// One line of a `--shapes <file>` list.
pub struct Listed {
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub label: String,
    /// The weight encoding to measure. Absent means `None` (bf16) — which is what every
    /// hand-authored list in the tree means, since bf16 was the only ladder that existed.
    pub quant: kernelcaps::QuantScheme,
}

/// Parse a `--shapes <file>` list: `M N K [label [quant]]` per line, `#` comments, blanks ignored.
///
/// A superset of the `SHAPES` heredoc grammar in `scripts/rebench_tune_gemm.sh`, so that list can
/// still be fed to this command verbatim — which is what makes an A/B against the bash campaign
/// possible. The `quant` column is the addition: a shape list is only a campaign once it says
/// which ENCODING each shape is to be measured in, and a file that omits it means bf16 because
/// that is what every file in the tree predating mxfp4 meant.
pub fn parse_list(text: &str) -> Result<Vec<Listed>, String> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 3 {
            return Err(format!(
                "line {}: expected `M N K [label [quant]]`, got {raw:?}",
                i + 1
            ));
        }
        let p = |s: &str| {
            s.parse::<i64>()
                .map_err(|e| format!("line {}: {s:?}: {e}", i + 1))
        };
        let (m, n, k) = (p(f[0])?, p(f[1])?, p(f[2])?);
        if m <= 0 || n <= 0 || k <= 0 {
            return Err(format!("line {}: shape dimensions must be positive", i + 1));
        }
        let quant = match f.get(4) {
            // Rejected, not defaulted. A typo'd encoding that silently measured bf16 would
            // publish a bf16 timing under whatever key the typo resolved to.
            Some(q) => tunedb::gemm::parse_quant(q)
                .ok_or_else(|| format!("line {}: unknown quant {q:?}", i + 1))?,
            None => kernelcaps::QuantScheme::None,
        };
        out.push(Listed {
            m,
            n,
            k,
            label: f
                .get(3)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "shape".into()),
            quant,
        });
    }
    Ok(out)
}

fn scratch_pkt() -> PathBuf {
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
    Path::new(&base).join(format!("plowc-tune-demand-{}.pkt", std::process::id()))
}

/// The emit may drop sidecars beside the packet (`block.json`, a weights map). Remove what we can
/// and stay quiet about what we cannot: a leftover file in `TMPDIR` is not worth failing a
/// campaign over.
fn cleanup(pkt: &Path) {
    let _ = std::fs::remove_file(pkt);
    if let Some(stem) = pkt.file_stem().and_then(|s| s.to_str()) {
        if let Some(dir) = pkt.parent() {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for e in rd.flatten() {
                    let name = e.file_name();
                    if name.to_string_lossy().starts_with(stem) {
                        let _ = std::fs::remove_file(e.path());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernelcaps::QuantScheme;

    /// The hand-authored list in `scripts/rebench_tune_gemm.sh` must parse with this grammar,
    /// because an A/B against the bash campaign feeds exactly that text to `--shapes`.
    #[test]
    fn the_bash_campaigns_shape_grammar_round_trips() {
        let sample = "\
512 6144 512    glm52-kvb-up-M512
128 128 2816   gemma26b-router

# a comment
2048 21504 5376 gemma31b-gateup-M2048
";
        let got = parse_list(sample).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!((got[0].m, got[0].n, got[0].k), (512, 6144, 512));
        assert_eq!(got[0].label, "glm52-kvb-up-M512");
        assert_eq!(got[2].k, 5376);
        // A list with no quant column is a bf16 list — every one in the tree predates mxfp4.
        assert!(got.iter().all(|l| l.quant == QuantScheme::None));
    }

    /// The quant column is what makes a hand-authored list able to describe an mxfp4 campaign,
    /// and an unknown value must fail rather than quietly measure bf16 under an mxfp4 key.
    #[test]
    fn the_quant_column_is_parsed_and_a_bad_one_is_rejected() {
        let got = parse_list("128 576 7168 kimi-kva Mxfp4\n128 576 7168 kimi-kva None\n").unwrap();
        assert_eq!(got[0].quant, QuantScheme::Mxfp4);
        assert_eq!(got[1].quant, QuantScheme::None);
        assert!(parse_list("128 576 7168 kimi-kva Mxfp5").is_err());
    }

    #[test]
    fn a_malformed_shape_line_is_an_error_not_a_skip() {
        assert!(parse_list("512 6144").is_err());
        assert!(parse_list("512 0 6144").is_err());
        assert!(parse_list("a b c").is_err());
    }

    /// `render` -> `parse_list` is the loop a derived campaign goes round when it is written out
    /// and replayed, so it has to be lossless in the dimensions (the label is cosmetic — the
    /// harness does not put it in the JSONL).
    #[test]
    fn rendered_demand_parses_back_to_the_same_shapes() {
        let shapes = vec![
            Demand {
                m: 512,
                n: 6144,
                k: 512,
                quant: QuantScheme::None,
                hit: false,
            },
            Demand {
                m: 8192,
                n: 64,
                k: 6144,
                quant: QuantScheme::Mxfp4,
                hit: true,
            },
        ];
        let back = parse_list(&render(&shapes)).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!((back[0].m, back[0].n, back[0].k), (512, 6144, 512));
        assert_eq!((back[1].m, back[1].n, back[1].k), (8192, 64, 6144));
        // The quant survives the round trip, or a derived mxfp4 campaign replays as bf16.
        assert_eq!(back[0].quant, QuantScheme::None);
        assert_eq!(back[1].quant, QuantScheme::Mxfp4);
    }

    #[test]
    fn coverage_counts_the_number_that_was_invisible() {
        let shapes: Vec<Demand> = (0..32)
            .map(|i| Demand {
                m: 512,
                n: i,
                k: 6144,
                quant: QuantScheme::None,
                hit: false,
            })
            .collect();
        assert_eq!(coverage(&shapes), (0, 32));
    }
}
