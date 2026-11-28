//! Build and resolve a `TuneManifest` from any model `config.json` (or
//! HuggingFace directory containing `config.json`) against a `TuneStore`.

use std::path::PathBuf;
use std::process::ExitCode;

use kernelcaps::HardwareFingerprint;
use plow_asset::Phase;
use tunedb::{
    amd_tuning_cell, check_strict, format_report, from_config_json, gemm_oracle,
    infer_digests_for_cell, Digests, TuneStore,
};

fn usage() -> &'static str {
    "usage: tunedb-manifest --config <CONFIG.json | HF_DIR> [--model NAME] [--gpu GPU] [--db DIR] \
     [--batch 1,4,8] [--seq 512,2048,8192] [--phase both|prefill|decode] \
     [--impl DIGEST] [--interp DIGEST] [--toolchain TC] [--oracle ORACLE] \
     [--out FILE] [--json] [--strict]"
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage());
        return Ok(());
    }

    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let cfg_arg = PathBuf::from(value("--config").ok_or_else(|| usage().to_string())?);
    let cfg_path = if cfg_arg.is_dir() {
        cfg_arg.join("config.json")
    } else {
        cfg_arg
    };
    let config_json = std::fs::read_to_string(&cfg_path)
        .map_err(|e| format!("cannot read {}: {e}", cfg_path.display()))?;

    let model_name = value("--model").unwrap_or_else(|| {
        cfg_path
            .parent()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty() && s != ".")
            .unwrap_or_else(|| "model".to_string())
    });

    let gpu_name = value("--gpu").unwrap_or_else(|| "H100 SXM5".to_string());
    let spec = hwspec::registry::lookup(&gpu_name)
        .ok_or_else(|| format!("unknown GPU {gpu_name:?}; see hwspec::registry"))?;
    let hw = HardwareFingerprint::from_spec(spec)
        .ok_or_else(|| format!("{} has no ISA level mapping", spec.name))?;
    let cell = if matches!(
        hw.isa,
        hwspec::IsaLevel::Gfx942 | hwspec::IsaLevel::Gfx950 | hwspec::IsaLevel::Gfx1201
    ) {
        amd_tuning_cell(spec)
    } else {
        hw.tuning_path()
    };

    let batches = parse_i64_list(&value("--batch").unwrap_or_else(|| "1,4,8".into()))?;
    let seqs = parse_i64_list(&value("--seq").unwrap_or_else(|| "512,2048,8192".into()))?;
    let phases = match value("--phase").as_deref().unwrap_or("both") {
        "prefill" => vec![Phase::Prefill],
        "decode" => vec![Phase::Decode],
        "both" => vec![Phase::Prefill, Phase::Decode],
        other => {
            return Err(format!("unknown --phase {other:?} (expected prefill|decode|both)").into())
        }
    };
    let image_size = value("--image-size")
        .map(|s| s.parse::<i64>())
        .transpose()?
        .unwrap_or(1024);

    let db = PathBuf::from(value("--db").unwrap_or_else(|| "tuning".to_string()));
    let store = TuneStore::new(&db);

    let (want, digest_note) = match (value("--impl"), value("--interp")) {
        (Some(imp), Some(intp)) => {
            let note = format!("explicit {imp}");
            (
                Digests {
                    execution: None,
                    implementation: imp,
                    interpreter: intp,
                    toolchain: value("--toolchain").unwrap_or_default(),
                    oracle: value("--oracle").unwrap_or_else(|| gemm_oracle(hw.isa).to_string()),
                },
                note,
            )
        }
        (imp_opt, intp_opt) => {
            let inferred = infer_digests_for_cell(&store, &cell)?;
            if let Some(mut d) = inferred {
                if let Some(imp) = imp_opt {
                    d.implementation = imp;
                }
                if let Some(intp) = intp_opt {
                    d.interpreter = intp;
                }
                if let Some(tc) = value("--toolchain") {
                    d.toolchain = tc;
                }
                if let Some(or) = value("--oracle") {
                    d.oracle = or;
                }
                let note = format!("inferred from {cell} ({})", d.implementation);
                (d, note)
            } else {
                (
                    Digests {
                        execution: None,
                        implementation: imp_opt.unwrap_or_default(),
                        interpreter: intp_opt.unwrap_or_default(),
                        toolchain: value("--toolchain").unwrap_or_default(),
                        oracle: value("--oracle")
                            .unwrap_or_else(|| gemm_oracle(hw.isa).to_string()),
                    },
                    format!("none in {} for {cell}", db.display()),
                )
            }
        }
    };

    let def = from_config_json(
        &config_json,
        &model_name,
        &batches,
        &seqs,
        &phases,
        image_size,
    )?;
    let manifest = def.compile_for(&store, &cell, &want)?;

    let out_path = value("--out").map(PathBuf::from);
    if let Some(ref p) = out_path {
        manifest.save_json(p)?;
    }

    if flag("--json") {
        println!("{}", manifest.to_json_pretty()?);
    } else {
        print!(
            "{}",
            format_report(&manifest, &cell, &digest_note, out_path.as_deref())
        );
    }

    if flag("--strict") {
        check_strict(&manifest, &cell)?;
    }
    Ok(())
}

fn parse_i64_list(raw: &str) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    raw.split(',')
        .map(|part| part.trim().parse::<i64>().map_err(Into::into))
        .collect()
}
