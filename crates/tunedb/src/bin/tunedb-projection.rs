use std::{
    io::{BufRead, BufReader},
    path::PathBuf,
};
use tunedb::projection::{ProjectionCell, ProjectionMeasurement, ProjectionTiming};
use tunedb::{Correctness, Digests, RecordState, TuneStore};
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    cell: ProjectionCell,
    split: u32,
    digests: Digests,
    stats: ProjectionTiming,
    baseline: ProjectionTiming,
    baseline_object: plow_asset::decode_objects::DecodeObject,
    candidate_object: plow_asset::decode_objects::DecodeObject,
    baseline_registers: u32,
    candidate_registers: u32,
    native_blocks: Vec<tunedb::projection::NativeBlockGuard>,
    correctness: Correctness,
    campaign: String,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("publish") {
        return Err("usage: tunedb-projection publish --db ROOT --results RAW.jsonl".into());
    }
    let value = |name: &str| {
        args.iter()
            .position(|v| v == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let root = PathBuf::from(value("--db").ok_or("missing --db")?);
    let input = std::fs::File::open(value("--results").ok_or("missing --results")?)?;
    let mut records = Vec::new();
    for line in BufReader::new(input).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let r: Raw = serde_json::from_str(&line)?;
        records.push(ProjectionMeasurement {
            cell: r.cell,
            split: r.split,
            digests: r.digests,
            stats: r.stats,
            baseline: r.baseline,
            baseline_object: r.baseline_object,
            candidate_object: r.candidate_object,
            baseline_registers: r.baseline_registers,
            candidate_registers: r.candidate_registers,
            native_blocks: r.native_blocks,
            correctness: r.correctness,
            state: RecordState::Provisional,
            campaign: r.campaign,
        });
    }
    let hardware = records
        .first()
        .ok_or("empty publication")?
        .cell
        .hardware
        .clone();
    let count = TuneStore::new(root).publish_projection(&hardware, records)?;
    println!("published {count} projection records to {hardware}");
    Ok(())
}
