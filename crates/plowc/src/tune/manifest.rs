//! Derive a [`TuneManifest`] from any model (`--config`, `--hf-dir`,
//! `--model`, or `--net`) and check/report its coverage against `TuneStore`.

use std::collections::BTreeSet;
use std::path::PathBuf;

use kernelcaps::{dense_gemm_tuning_build, HardwareFingerprint, QuantScheme};
use schedule::Phase;
use tunedb::{
    check_strict, format_report, from_config_json, infer_digests_for_cell, Digests, NetworkBlock,
    NetworkBlockDefinition, NetworkOp, TuneStore,
};

use super::TuneOptions;
use crate::net::{NetConfig, NetOp};

/// Where the model topology is read from.
#[derive(Debug, Clone)]
pub enum ManifestSource {
    /// Local HuggingFace directory containing `config.json` (weights not required).
    HfDir(PathBuf),
    /// Direct path to a HuggingFace `config.json` file.
    ConfigJson(PathBuf),
    /// HuggingFace Hub model id (metadata-only fetch of `config.json`).
    HubModel(String),
    /// Plow-native network JSON ([`NetConfig`]).
    NetJson(PathBuf),
}

/// Request parameters for `plowc tune manifest`.
#[derive(Debug, Clone)]
pub struct ManifestRequest {
    pub source: ManifestSource,
    pub batches: Vec<i64>,
    pub seqs: Vec<i64>,
    pub phases: Vec<Phase>,
    pub image_size: i64,
    pub strict: bool,
    pub json: bool,
    pub out: Option<PathBuf>,
}

pub fn run(
    opts: &TuneOptions,
    spec: &hwspec::GpuSpec,
    hw: &HardwareFingerprint,
    req: &ManifestRequest,
) -> Result<(), Box<dyn std::error::Error>> {
    let cell = if matches!(
        hw.isa,
        hwspec::IsaLevel::Gfx942 | hwspec::IsaLevel::Gfx950 | hwspec::IsaLevel::Gfx1201
    ) {
        tunedb::amd_tuning_cell(spec)
    } else {
        hw.tuning_path()
    };

    let def = build_network_block_definition(&req.source, req)?;
    let store = TuneStore::new(&opts.db);
    let (want, digest_note) = resolve_digests(opts, hw, &store, &cell)?;
    let manifest = def.compile_for(&store, &cell, &want)?;

    if let Some(out_path) = &req.out {
        manifest.save_json(out_path)?;
    }

    if req.json {
        if !digest_note.starts_with("probed ") {
            eprintln!("note: {digest_note}");
        }
        println!("{}", manifest.to_json_pretty()?);
    } else {
        print!(
            "{}",
            format_report(&manifest, &cell, &digest_note, req.out.as_deref())
        );
    }

    if req.strict {
        check_strict(&manifest, &cell)?;
    }
    Ok(())
}

/// Build a [`NetworkBlockDefinition`] from any supported model source without
/// requiring checkpoint weight shards.
pub fn build_network_block_definition(
    source: &ManifestSource,
    req: &ManifestRequest,
) -> Result<NetworkBlockDefinition, Box<dyn std::error::Error>> {
    let asset_phases: Vec<plow_asset::Phase> = req
        .phases
        .iter()
        .map(|p| match p {
            Phase::Prefill => plow_asset::Phase::Prefill,
            Phase::Decode => plow_asset::Phase::Decode,
        })
        .collect();

    match source {
        ManifestSource::NetJson(path) => {
            let json = std::fs::read_to_string(path)?;
            let net: NetConfig = serde_json::from_str(&json)?;
            net.validate()?;
            Ok(build_from_net_config(
                &net,
                &req.batches,
                &req.seqs,
                &req.phases,
            ))
        }
        ManifestSource::ConfigJson(path) => {
            let json = std::fs::read_to_string(path)?;
            let model_name = path
                .parent()
                .and_then(|p| p.file_name())
                .map(|s| s.to_string_lossy().into_owned())
                .filter(|s| !s.is_empty() && s != ".")
                .unwrap_or_else(|| "config".to_string());
            from_config_json(
                &json,
                &model_name,
                &req.batches,
                &req.seqs,
                &asset_phases,
                req.image_size,
            )
        }
        ManifestSource::HfDir(dir) => {
            let cfg_path = dir.join("config.json");
            let json = std::fs::read_to_string(&cfg_path).map_err(|e| {
                format!(
                    "cannot read {}: {e} (use `--config <file>` for a standalone config.json)",
                    cfg_path.display()
                )
            })?;
            let model_name = dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| dir.display().to_string());
            from_config_json(
                &json,
                &model_name,
                &req.batches,
                &req.seqs,
                &asset_phases,
                req.image_size,
            )
        }
        ManifestSource::HubModel(model_id) => {
            let json = nn_graph::hub::fetch_file_at_revision(model_id, "config.json", None)
                .map_err(|e| format!("failed to fetch config.json for {model_id:?}: {e}"))?;
            from_config_json(
                &json,
                model_id,
                &req.batches,
                &req.seqs,
                &asset_phases,
                req.image_size,
            )
        }
    }
}

fn build_from_net_config(
    net: &NetConfig,
    batches: &[i64],
    seqs: &[i64],
    phases: &[Phase],
) -> NetworkBlockDefinition {
    let bs: Vec<i64> = if batches.is_empty() {
        vec![1]
    } else {
        batches.to_vec()
    };
    let ss: Vec<i64> = if seqs.is_empty() {
        vec![512]
    } else {
        seqs.to_vec()
    };
    let ps: Vec<Phase> = if phases.is_empty() {
        vec![Phase::Prefill, Phase::Decode]
    } else {
        phases.to_vec()
    };

    let mut seen = BTreeSet::new();
    let mut ops = Vec::new();
    for phase in ps {
        let pairs: Vec<(&str, i64, i64)> = match phase {
            Phase::Prefill => bs
                .iter()
                .flat_map(|&b| ss.iter().map(move |&s| ("prefill", b, s)))
                .collect(),
            Phase::Decode => bs.iter().map(|&b| ("decode", b, 1)).collect(),
        };
        for (phase_label, b, s) in pairs {
            if !seen.insert((phase_label, b, s)) {
                continue;
            }
            let tokens = b.saturating_mul(s);
            let mut running = net.hidden;
            for net_op in &net.ops {
                match net_op {
                    NetOp::Norm { feat } => {
                        let k = feat.unwrap_or(running);
                        ops.push(NetworkOp::new("rmsnorm", phase_label, tokens, k, k, "bf16"));
                        running = k;
                    }
                    NetOp::Gemm { n, k } => {
                        let in_k = k.unwrap_or(running);
                        ops.push(NetworkOp::gemm(
                            phase_label,
                            tokens,
                            *n,
                            in_k,
                            QuantScheme::None,
                        ));
                        running = *n;
                    }
                    NetOp::Act => {
                        ops.push(NetworkOp::new(
                            "act",
                            phase_label,
                            tokens,
                            running,
                            running,
                            "bf16",
                        ));
                    }
                    NetOp::Flash { heads, head_dim } => {
                        let width = heads * head_dim;
                        ops.push(NetworkOp::new(
                            "attention",
                            phase_label,
                            tokens,
                            width,
                            *head_dim,
                            "bf16",
                        ));
                        running = width;
                    }
                }
            }
        }
    }

    NetworkBlockDefinition::new("plow_net", vec![NetworkBlock::new(0, "net", ops)])
        .with_model(&net.name)
}

fn resolve_digests(
    opts: &TuneOptions,
    hw: &HardwareFingerprint,
    store: &TuneStore,
    cell: &str,
) -> Result<(Digests, String), Box<dyn std::error::Error>> {
    match dense_gemm_tuning_build(&opts.root, hw.isa) {
        Ok(build) => {
            let label = build.label();
            let want = Digests {
                execution: None,
                implementation: label.clone(),
                interpreter: label.clone(),
                toolchain: build.toolchain,
                oracle: tunedb::gemm_oracle(hw.isa).to_string(),
            };
            Ok((want, format!("probed {label}")))
        }
        Err(err) => {
            if let Some(inferred) = infer_digests_for_cell(store, cell)? {
                let note = format!(
                    "inferred from {cell} ({}; local probe skipped: {err})",
                    inferred.implementation
                );
                Ok((inferred, note))
            } else {
                Ok((
                    Digests {
                        execution: None,
                        implementation: String::new(),
                        interpreter: String::new(),
                        toolchain: String::new(),
                        oracle: tunedb::gemm_oracle(hw.isa).to_string(),
                    },
                    format!("unavailable ({err}); checking existence only"),
                ))
            }
        }
    }
}
