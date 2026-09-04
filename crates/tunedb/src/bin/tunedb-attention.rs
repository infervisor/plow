//! Publish model-neutral attention split measurements through TuneStore's gates.
//!
//! Usage: tunedb-attention publish --db tuning --results RAW.jsonl
//!
//! RAW rows carry timing samples rather than precomputed statistics. This tool constructs
//! AttentionMeasurement, applies the common sample/correctness gates, and serializes the
//! canonical store schema.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

use tunedb::{
    AttentionAlgorithm, AttentionCell, AttentionMeasurement, Correctness, Digests, RecordState,
    Stats, TuneStore,
};

#[derive(serde::Deserialize)]
struct RawAttentionMeasurement {
    cell: AttentionCell,
    algorithm: AttentionAlgorithm,
    nsplit: u32,
    digests: Digests,
    samples_ns: Vec<f64>,
    correctness: Correctness,
    campaign: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("tunedb-attention: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("publish") {
        return Err("usage: tunedb-attention publish --db ROOT --results RAW.jsonl".into());
    }
    let option = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|index| args.get(index + 1))
            .cloned()
    };
    let db = PathBuf::from(option("--db").unwrap_or_else(|| "tuning".into()));
    let results = PathBuf::from(option("--results").ok_or("publish needs --results RAW.jsonl")?);
    let file = std::fs::File::open(results)?;
    let mut by_hardware = std::collections::BTreeMap::<String, Vec<_>>::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: RawAttentionMeasurement = serde_json::from_str(&line)?;
        let hardware = raw.cell.hardware.clone();
        by_hardware
            .entry(hardware)
            .or_default()
            .push(AttentionMeasurement {
                cell: raw.cell,
                algorithm: raw.algorithm,
                nsplit: raw.nsplit,
                digests: raw.digests,
                stats: Stats::from_samples(raw.samples_ns)?,
                correctness: raw.correctness,
                state: RecordState::Provisional,
                campaign: raw.campaign,
            });
    }
    if by_hardware.is_empty() {
        return Err("results contained no records".into());
    }
    let store = TuneStore::new(db);
    for (hardware, records) in by_hardware {
        let count = store.publish_attention(&hardware, records)?;
        println!("published {count} attention record(s) to {hardware}");
    }
    Ok(())
}
