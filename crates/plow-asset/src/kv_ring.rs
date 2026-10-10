//! The `kv_ring.v1` obligation (lean-plow `Plow/KvRing.lean`): the K/V rows each packed prefill
//! launch writes into a sliding ring, derived from the packet's `live_kv` and `packed_prefill`
//! sections by running the runtime planner (`plan_with_limit`, `stage_slots`) on boundary
//! request mixes. A span is a maximal run of rows with one slot and consecutive positions, which
//! is what the kernel writes; Lean rejects two spans on one slot.

use std::collections::BTreeSet;

use serde_json::{json, Value};

use crate::live_kv;
use crate::packed_prefill::{self, Request};

pub const ENDPOINT: &str = "kv_ring.v1";

/// The ring geometry Lean checks: `2^ring_log` rows, attention `window`, `capacity` positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ring {
    pub ring_log: u32,
    pub window: u32,
    pub capacity: u32,
}

/// Sliding caches of `live` (a ring narrower than the context), deduplicated by geometry.
pub fn rings(live: &live_kv::Manifest) -> Result<Vec<Ring>, String> {
    let mut out = BTreeSet::new();
    for c in live.caches.iter().filter(|c| c.window > 0 && c.stride < live.max_ctx) {
        if !c.stride.is_power_of_two() || c.mask != c.stride - 1 {
            return Err(format!("cache {:?}: ring stride {} / mask {:#x} is not a power-of-two ring", c.pair, c.stride, c.mask));
        }
        out.insert(Ring { ring_log: c.stride.trailing_zeros(), window: c.window, capacity: live.max_ctx });
    }
    Ok(out.into_iter().collect())
}

/// One launch from the per-row slot map and positions the runtime binds (`-1` = not written).
pub fn launch(ring: Ring, slots: &[i32], positions: &[i32]) -> Result<Value, String> {
    if slots.len() != positions.len() {
        return Err(format!("slot map has {} rows, positions {}", slots.len(), positions.len()));
    }
    let mut spans: Vec<(u32, u32, u32)> = Vec::new();
    for (&slot, &pos) in slots.iter().zip(positions) {
        if slot < 0 {
            continue;
        }
        let pos = u32::try_from(pos).map_err(|_| format!("negative position {pos} for slot {slot}"))?;
        match spans.last_mut() {
            Some((s, start, len)) if *s == slot as u32 && *start + *len == pos => *len += 1,
            _ => spans.push((slot as u32, pos, 1)),
        }
    }
    let spans: Vec<Value> = spans.iter().map(|&(slot, start, len)| json!({"slot": slot, "start": start, "len": len})).collect();
    Ok(json!({"ring_log": ring.ring_log, "window": ring.window, "capacity": ring.capacity, "spans": spans}))
}

/// Boundary request mixes for a `rows`-row bucket whose requests write at most `limit` rows:
/// one full request ending at the context, `n` staggered requests that wrap the ring, and a
/// one-row request (on an unmasked plan its padding writes the rest of the bucket).
fn mixes(rows: usize, limit: usize, batch: usize, max_ctx: usize) -> Vec<Vec<(usize, usize, usize)>> {
    let full = limit.min(rows).min(max_ctx);
    let tail = max_ctx.saturating_sub(rows);
    let n = batch.min(4).min(rows).max(1);
    let each = (rows / n).min(limit).max(1);
    vec![
        vec![(0, tail.min(max_ctx - full), full)],
        (0..n).map(|i| (i, tail * i / n, each)).collect(),
        vec![(0, 0, 1)],
    ]
}

/// The `kv_ring.v1` request for a packet: every prefill rung, every sliding ring, every mix
/// and (when staged) every stage. `None` when the packet has no sliding ring.
pub fn request(pf: &packed_prefill::Manifest, live: &live_kv::Manifest, rungs: &[u32]) -> Result<Option<Value>, String> {
    let rings = rings(live)?;
    if rings.is_empty() {
        return Ok(None);
    }
    let shape = Shape {
        rungs,
        max_request_rows: pf.max_request_rows,
        stage_rows: pf.stage_rows,
        batch: live.batch as usize,
        max_ctx: live.max_ctx as usize,
    };
    Ok(Some(json!({"schema": 1, "launches": launches(&rings, &shape)?})))
}

/// What the planner needs from a packet's `packed_prefill`/`live_kv` sections.
pub struct Shape<'a> {
    pub rungs: &'a [u32],
    pub max_request_rows: Option<u32>,
    pub stage_rows: Option<u32>,
    pub batch: usize,
    pub max_ctx: usize,
}

/// Launches the runtime planner binds for every rung, mix and stage of `shape`, on every ring.
pub fn launches(rings: &[Ring], shape: &Shape<'_>) -> Result<Vec<Value>, String> {
    let widest = shape.rungs.iter().copied().max().ok_or("no prefill rungs")?;
    let limit = shape.max_request_rows.unwrap_or(widest) as usize;
    let mut out = Vec::new();
    for &rows in shape.rungs.iter().collect::<BTreeSet<_>>() {
        let rows = rows as usize;
        for mix in mixes(rows, limit, shape.batch, shape.max_ctx) {
            let mut frontiers = vec![0u32; shape.batch];
            let requests: Vec<Request> = mix
                .iter()
                .map(|&(slot, start, len)| {
                    frontiers[slot] = start as u32;
                    Request { slot, start, len, prompt: start + len }
                })
                .collect();
            let plan = packed_prefill::plan_with_limit(&requests, &frontiers, rows, shape.max_ctx, shape.max_request_rows)
                .map_err(|e| format!("rung {rows}, mix {mix:?}: {e}"))?;
            let stages = match shape.stage_rows {
                Some(stage_rows) => {
                    let sr = stage_rows as usize;
                    (0..packed_prefill::stages_needed(&plan, sr)?)
                        .map(|s| packed_prefill::stage_slots(&plan, s, sr))
                        .collect::<Result<Vec<_>, _>>()?
                }
                None => vec![plan.slots.clone()],
            };
            for slots in &stages {
                for &ring in rings {
                    out.push(launch(ring, slots, &plan.positions)?);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> Ring {
        Ring { ring_log: 11, window: 1024, capacity: 16384 }
    }

    #[test]
    fn launch_splits_rows_into_slot_runs() {
        let v = launch(ring(), &[2, 2, 2, 0, 0, -1, -1], &[10, 11, 12, 0, 1, 0, 0]).unwrap();
        assert_eq!(
            v["spans"],
            json!([{"slot": 2, "start": 10, "len": 3}, {"slot": 0, "start": 0, "len": 2}])
        );
        let split = launch(ring(), &[1, 1, 1], &[5, 7, 8]).unwrap();
        assert_eq!(split["spans"].as_array().unwrap().len(), 2, "a position gap is a second span");
        assert!(launch(ring(), &[1], &[-3]).unwrap_err().contains("negative"));
        assert!(launch(ring(), &[1, 1], &[0]).is_err());
    }

    #[test]
    fn mixes_are_valid_plans() {
        for (rows, limit, batch, ctx) in [(4096, 1024, 32, 16384), (8192, 8192, 1, 8192), (512, 512, 2, 4096), (64, 64, 8, 256), (64, 16, 8, 32)] {
            for mix in mixes(rows, limit, batch, ctx) {
                let mut frontiers = vec![0u32; batch];
                let requests: Vec<Request> = mix
                    .iter()
                    .map(|&(slot, start, len)| {
                        frontiers[slot] = start as u32;
                        Request { slot, start, len, prompt: start + len }
                    })
                    .collect();
                let limit = (limit < rows).then_some(limit as u32);
                packed_prefill::plan_with_limit(&requests, &frontiers, rows, ctx, limit)
                    .unwrap_or_else(|e| panic!("{rows} {mix:?}: {e}"));
            }
        }
    }
}
