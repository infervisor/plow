//! One compiled shape bucket: the decoded packet stream plus its address map
//! and sidecars.

use std::path::Path;

use packet::Program;
use plow_asset::{Blocks, BucketStat, Experts, MemoryMap, Phase, RequestIo};

use crate::asset::read_json;
use crate::{Result, RuntimeError};

/// Scheduler dispatch key: `(phase, batch, seq)`. `Copy` + hashable so bucket
/// selection is a cheap map lookup on the hot path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BucketKey {
    pub phase: Phase,
    pub batch: i64,
    pub seq: i64,
}

impl BucketKey {
    pub fn new(phase: &str, batch: i64, seq: i64) -> Self {
        BucketKey {
            phase: Phase::from_str_loose(phase),
            batch,
            seq,
        }
    }
}

/// A decoded bucket. The `Program` is owned (decoded once); the map/sidecars are
/// consulted at request-marshal and allocation time, not per token.
pub struct Bucket {
    pub key: BucketKey,
    pub program: Program,
    pub map: MemoryMap,
    pub request_io: Option<RequestIo>,
    pub blocks: Option<Blocks>,
    pub experts: Option<Experts>,
    /// Compiler estimate (cycles) — feeds the watchdog progress curve (§J/§K).
    pub makespan: u64,
    pub ideal_makespan: u64,
}

impl Bucket {
    pub(crate) fn load(dir: &Path, stat: &BucketStat) -> Result<Self> {
        let pkt_path = dir.join(&stat.packet_file);
        // mmap the packet bytes and decode once. The file is small (KBs–MBs) and
        // read-only; decoding up front keeps the hot path free of parsing.
        let file = std::fs::File::open(&pkt_path).map_err(|source| RuntimeError::Io {
            path: pkt_path.clone(),
            source,
        })?;
        // SAFETY: the .pkt file is a read-only compiled artifact we own for the
        // process lifetime; we copy out of the mapping immediately via decode.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| RuntimeError::Io {
            path: pkt_path.clone(),
            source,
        })?;
        let program = Program::decode(&mmap).map_err(|reason| RuntimeError::Packet {
            path: pkt_path.clone(),
            reason: reason.to_string(),
        })?;

        let map: MemoryMap = read_json(&dir.join(&stat.memory_file))?;
        map.validate().map_err(RuntimeError::AddressMap)?;

        // Sidecars share the packet stem; absence is fine (not every bucket has
        // MoE layers or a fully-tagged request-io surface).
        let stem = stat.packet_file.trim_end_matches(".pkt");
        let request_io = read_json_opt(dir, &format!("{stem}.request_io.json"))?;
        let blocks = read_json_opt(dir, &format!("{stem}.blocks.json"))?;
        let experts = read_json_opt(dir, &format!("{stem}.experts.json"))?;

        Ok(Bucket {
            key: BucketKey::new(&stat.phase, stat.batch, stat.seq),
            program,
            map,
            request_io,
            blocks,
            experts,
            makespan: stat.makespan,
            ideal_makespan: stat.ideal_makespan,
        })
    }
}

/// Read an optional sidecar: `Ok(None)` if the file is absent, `Err` if present
/// but malformed.
fn read_json_opt<T: serde::de::DeserializeOwned>(dir: &Path, name: &str) -> Result<Option<T>> {
    let path = dir.join(name);
    if !path.exists() {
        return Ok(None);
    }
    read_json(&path).map(Some)
}
