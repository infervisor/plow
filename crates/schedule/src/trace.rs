//! Chrome Trace Event Format dump of a compile-time [`Schedule`].
//!
//! Each scheduled task is emitted as one duration event (`ph:"X"`) with:
//!   * `tid` = per-resource lane (SM / DMA / DPU / Host)
//!   * `ts`  = start cycle
//!   * `dur` = task duration in cycles
//!   * `cat` = task class (`compute` / `dma_in` / `dma_out` / `host`)
//!   * `name`= op name + task id
//!
//! Load the resulting JSON in `chrome://tracing` or `ui.perfetto.dev` to view
//! the per-SM timeline of a bucket. Also emits `thread_name` metadata events
//! so each lane is labeled (`GPU0/SM12`, `GPU0/DMA0`, `DPU0`, `Host0`).
//!
//! Timestamps are cycles — the Chrome viewer will label the axis as µs, but
//! the shape and relative widths are what matter for scheduler debugging.

use crate::expand::TaskGraph;
use crate::machine::Machine;
use crate::passes::{PacketKind, Schedule};
use crate::resource::ResourceId;
use std::collections::BTreeMap;

/// Emit the Chrome Trace Event Format JSON for a scheduled bucket.
pub fn to_chrome_json(tasks: &TaskGraph, sched: &Schedule, machine: &Machine) -> String {
    let mut events: Vec<String> = Vec::new();

    // Sort lanes deterministically for stable output.
    let mut lanes: BTreeMap<u64, String> = BTreeMap::new();

    for (&res, packets) in &sched.packets {
        let tid = tid_of(res);
        lanes.entry(tid).or_insert_with(|| lane_name(res, machine));
        for p in packets {
            // Synthetic packets (injected host ops) may carry task ids past
            // the task graph; default their duration rather than panicking.
            let dur = tasks.tasks.get(p.task).map(|t| t.dur).unwrap_or(1);
            let cat = category(p.kind);
            let name = escape_json(&format!("{}#{}", p.op, p.task));
            events.push(format!(
                "{{\"name\":\"{name}\",\"cat\":\"{cat}\",\"ph\":\"X\",\"pid\":0,\"tid\":{tid},\"ts\":{ts},\"dur\":{dur},\"args\":{{\"task\":{tid_task},\"kind\":\"{cat}\"}}}}",
                name = name,
                cat = cat,
                tid = tid,
                ts = p.start,
                dur = dur.max(1),
                tid_task = p.task,
            ));
        }
    }

    // Thread-name metadata events so the viewer labels each lane.
    for (tid, name) in &lanes {
        events.push(format!(
            "{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":{tid},\"args\":{{\"name\":\"{name}\"}}}}",
            tid = tid,
            name = escape_json(name),
        ));
    }
    events.push(
        "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":0,\"args\":{\"name\":\"plow schedule\"}}"
            .to_string(),
    );

    format!(
        "{{\"traceEvents\":[{}],\"displayTimeUnit\":\"ns\"}}",
        events.join(",")
    )
}

/// Pack a `ResourceId` into a monotone `tid` so lanes group cleanly:
/// SMs at the top, then DMAs, then DPUs, then Host.
fn tid_of(r: ResourceId) -> u64 {
    // Assume < 10_000 SMs per unit; < 100 DMAs; < 100 DPUs.
    const SM_BASE: u64 = 0;
    const DMA_BASE: u64 = 1_000_000;
    const DPU_BASE: u64 = 2_000_000;
    const HOST_BASE: u64 = 3_000_000;
    match r {
        ResourceId::Sm(u, s) => SM_BASE + (u as u64) * 10_000 + s as u64,
        ResourceId::Dma(u, i) => DMA_BASE + (u as u64) * 100 + i as u64,
        ResourceId::Dpu(i) => DPU_BASE + i as u64,
        ResourceId::Host(i) => HOST_BASE + i as u64,
    }
}

fn lane_name(r: ResourceId, _m: &Machine) -> String {
    match r {
        ResourceId::Sm(u, s) => format!("GPU{u}/SM{s:03}"),
        ResourceId::Dma(u, i) => format!("GPU{u}/DMA{i}"),
        ResourceId::Dpu(i) => format!("DPU{i}"),
        ResourceId::Host(i) => format!("Host{i}"),
    }
}

fn category(k: PacketKind) -> &'static str {
    match k {
        PacketKind::Compute => "compute",
        PacketKind::TmaIn => "dma_in",
        PacketKind::TmaOut => "dma_out",
        PacketKind::Rdma => "rdma",
        PacketKind::HostCoord => "host",
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tid_lanes_are_monotone_by_class() {
        // SMs < DMAs < DPUs < Hosts so lanes group cleanly in the viewer.
        let sm = tid_of(ResourceId::Sm(0, 200));
        let dma = tid_of(ResourceId::Dma(0, 0));
        let dpu = tid_of(ResourceId::Dpu(0));
        let host = tid_of(ResourceId::Host(0));
        assert!(sm < dma && dma < dpu && dpu < host);
    }

    #[test]
    fn escape_json_handles_quotes_and_control() {
        assert_eq!(escape_json("a\"b"), "a\\\"b");
        assert_eq!(escape_json("x\ny"), "x\\ny");
    }

    #[test]
    fn category_covers_all_kinds() {
        assert_eq!(category(PacketKind::Compute), "compute");
        assert_eq!(category(PacketKind::TmaIn), "dma_in");
        assert_eq!(category(PacketKind::TmaOut), "dma_out");
        assert_eq!(category(PacketKind::Rdma), "rdma");
        assert_eq!(category(PacketKind::HostCoord), "host");
    }
}
