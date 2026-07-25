//! §K Per-executor/per-task timeline.
//!
//! Every SM/CU logs a **start and end timestamp for each task** into a
//! per-executor trace ring in device memory; the interpreter's hot path only
//! does two clock reads + a ring append, emitted as `CHECKPOINT` OOB events. A
//! background task drains the rings and reconstructs a Chrome-trace timeline.

use crate::exec::oob::{OobKind, OobMsg};

/// One task's execution slice: `(executor, task, opcode, start, end)`.
#[derive(Clone, Copy, Debug)]
pub struct TaskSpan {
    pub exec: u32,
    pub task: u32,
    pub opcode: u16,
    pub t_start: u64,
    pub t_end: u64,
}

/// Hard cap on retained spans. Once full, recording stops (new spans are
/// dropped) so a long-lived `--trace` server can't grow without bound; the
/// earliest spans are the ones kept, matching "trace the run from the start".
/// Dropping the oldest instead would cost an O(n) shift per span on the hot
/// path — not worth it for a debugging aid.
pub const MAX_TRACE_SPANS: usize = 100_000;

/// Reconstructs spans from drained OOB `CHECKPOINT` events and emits Chrome
/// trace JSON (each task a duration slice for a flame/timeline view).
#[derive(Default)]
pub struct Timeline {
    spans: Vec<TaskSpan>,
}

impl Timeline {
    pub fn new() -> Self {
        Timeline::default()
    }

    /// Ingest drained OOB events; `CHECKPOINT` carries `arg0 = start<<32|task`,
    /// `arg1 = end<<16|opcode` (skeleton packing — the device ABI fixes the real
    /// layout).
    pub fn ingest(&mut self, events: &[OobMsg]) {
        for e in events {
            if e.kind == OobKind::Checkpoint as u16 {
                self.push(TaskSpan {
                    exec: e.exec,
                    task: (e.arg0 & 0xffff_ffff) as u32,
                    opcode: (e.arg1 & 0xffff) as u16,
                    t_start: e.arg0 >> 32,
                    t_end: e.arg1 >> 16,
                });
            }
        }
    }

    /// Append a span directly (the §O simulator and the CPU reference run feed
    /// spans here without going through the OOB ring). Silently dropped once
    /// [`MAX_TRACE_SPANS`] is reached.
    pub fn push(&mut self, span: TaskSpan) {
        if self.spans.len() < MAX_TRACE_SPANS {
            self.spans.push(span);
        }
    }

    pub fn spans(&self) -> &[TaskSpan] {
        &self.spans
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Chrome trace JSON (`chrome://tracing` / Perfetto compatible).
    pub fn to_chrome_json(&self) -> String {
        let events: Vec<String> = self
            .spans
            .iter()
            .map(|s| {
                format!(
                    "{{\"name\":\"op{:#06x}\",\"ph\":\"X\",\"pid\":0,\"tid\":{},\"ts\":{},\"dur\":{}}}",
                    s.opcode,
                    s.exec,
                    s.t_start,
                    s.t_end.saturating_sub(s.t_start)
                )
            })
            .collect();
        format!("{{\"traceEvents\":[{}]}}", events.join(","))
    }
}
