//! Pass — unified HBM memory allocation.
//!
//! Assigns every buffer (weights, request I/O, scratch activations, growable KV)
//! a concrete byte **offset** within an arena and emits an [`AddressMap`] the
//! runtime allocates once and **rebases** to (the compile-time counterpart of the
//! indirection table in `gpu_hypervisor_architecture.md §7.2`). This realizes the
//! "layouts are static, addressing is dynamic" principle
//! (`infervisor_compiler_scheduler_design.md §10.6`): the packet stream references
//! logical slots; this map fixes where each slot lives.
//!
//! The arena is laid out by lifetime class so the growable region sits last and
//! can extend into free HBM as the context/sequence length grows:
//!
//! ```text
//! [ Persistent (weights) | RequestIo | Scratch (reused by liveness) | Growable… → ]
//!                                                                    ^ growable_base
//! ```
//!
//! Scratch is packed with a greedy linear-scan first-fit planner — the same
//! interval-conflict reuse the SRAM allocator ([`crate::resource`]) does, but in
//! bytes rather than page slots: two scratch buffers share storage iff their live
//! intervals are disjoint.

use crate::interval::Cycle;
use std::collections::HashMap;

pub use plow_asset::BufClass;
pub use plow_asset::Segment;

/// DMA/TMA-friendly alignment for every buffer offset (bytes).
pub const MEM_ALIGN: u64 = 256;

/// A buffer to place.
#[derive(Clone, Debug)]
pub struct BufReq {
    pub name: String,
    pub size: u64,
    pub class: BufClass,
    /// Live interval `[start, end)` in schedule cycles. Drives [`BufClass::Scratch`]
    /// reuse; ignored for the other classes (treated as live for the whole program).
    pub live: Option<(Cycle, Cycle)>,
    /// Extra bytes reserved past `size` for in-place growth ([`BufClass::Growable`]).
    pub growth_reserve: u64,
    /// Which device (GPU/unit) owns this buffer. Buffers are placed in their
    /// device's segment of the global address space (Phase C5 / PGAS).
    pub device: u8,
    /// If non-empty, this is a **replicated** tensor: one physical copy is placed
    /// per listed device (each in that device's segment). A consumer reads the
    /// copy local to its device. Empty ⇒ a single copy on `device` (Phase C5 / TP).
    pub replicas: Vec<u8>,
    /// This buffer is a zero-copy **view** placed inside another's storage at
    /// `alias_off` bytes from that target's base (Phase C). `alias_off == 0` is a
    /// reshape (shares the whole buffer); a positive offset places it as a
    /// sub-region — e.g. a concat input written directly into its slice of the
    /// output, so the concat moves no bytes. Liveness of the shared buffer is the
    /// union of all members; disjoint sub-regions coexist without conflict.
    pub alias_of: Option<String>,
    pub alias_off: u64,
}

impl BufReq {
    /// A fixed-size buffer of `class` with no growth and (optionally) a live interval.
    pub fn new(name: impl Into<String>, size: u64, class: BufClass) -> Self {
        BufReq {
            name: name.into(),
            size,
            class,
            live: None,
            growth_reserve: 0,
            device: 0,
            replicas: Vec::new(),
            alias_of: None,
            alias_off: 0,
        }
    }
    /// Place this buffer in `device`'s segment of the global address space.
    pub fn on_device(mut self, device: u8) -> Self {
        self.device = device;
        self
    }
    /// Replicate this tensor — one copy per device in `devices`.
    pub fn replicated_on(mut self, devices: impl IntoIterator<Item = u8>) -> Self {
        self.replicas = devices.into_iter().collect();
        self
    }
    pub fn with_live(mut self, start: Cycle, end: Cycle) -> Self {
        self.live = Some((start, end));
        self
    }
    pub fn with_growth(mut self, reserve: u64) -> Self {
        self.growth_reserve = reserve;
        self
    }
    /// Mark this buffer as a zero-copy view sharing `target`'s base address.
    pub fn alias(self, target: impl Into<String>) -> Self {
        self.alias_at(target, 0)
    }
    /// Place this buffer inside `target` at byte offset `off` (a concat sub-region).
    pub fn alias_at(mut self, target: impl Into<String>, off: u64) -> Self {
        self.alias_of = Some(target.into());
        self.alias_off = off;
        self
    }
}

/// One placed buffer in the arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddrEntry {
    /// Logical slot id (stable: the buffer's index in the input request list).
    pub slot: u32,
    pub name: String,
    pub class: BufClass,
    /// Byte offset in the **global** address space (segment base + local offset).
    pub offset: u64,
    /// Bytes reserved at `offset` (`size` + `growth_reserve` for Growable).
    pub reserved: u64,
    /// Whether the runtime may extend this region in place.
    pub growable: bool,
    /// Owning device (its segment of the global address space).
    pub device: u8,
}

/// The compile-time address map the runtime rebases to.
#[derive(Clone, Debug, Default)]
pub struct AddressMap {
    pub entries: Vec<AddrEntry>,
    /// Total bytes across all device segments.
    pub arena_bytes: u64,
    /// Device 0's growable base (single-device convenience; see `segments`).
    pub growable_base: u64,
    /// Per-device segments of the global address space (one entry per device).
    pub segments: Vec<Segment>,
}

impl AddressMap {
    pub fn get(&self, name: &str) -> Option<&AddrEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// The copy of `name` resident on `device` (for replicated tensors, the local
    /// replica a consumer on that device reads).
    pub fn get_on_device(&self, name: &str, device: u8) -> Option<&AddrEntry> {
        self.entries
            .iter()
            .find(|e| e.name == name && e.device == device)
    }

    /// Every physical copy of `name` (one per device for a replicated tensor).
    pub fn replicas<'a>(&'a self, name: &str) -> impl Iterator<Item = &'a AddrEntry> + 'a {
        let name = name.to_string();
        self.entries.iter().filter(move |e| e.name == name)
    }

    /// Structural sanity the runtime loader can assert before allocating: every
    /// buffer fits inside the arena and slot ids are unique. (Aliases overlap by
    /// design, so this does *not* check byte-disjointness — that invariant is
    /// liveness-dependent and checked at allocation time.)
    pub fn validate(&self) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for e in &self.entries {
            if e.offset + e.reserved > self.arena_bytes {
                return Err(format!(
                    "buffer '{}' [{}, {}) exceeds arena {}",
                    e.name,
                    e.offset,
                    e.offset + e.reserved,
                    self.arena_bytes
                ));
            }
            if !seen.insert(e.slot) {
                return Err(format!("duplicate slot id {}", e.slot));
            }
        }
        Ok(())
    }
}

/// Debug-time invariant: two buffers that are live at the same time and are not in
/// an alias relationship must never share bytes. Guards the scratch-reuse and
/// concat-group placement — a placement bug fails the test suite rather than
/// silently corrupting data.
fn debug_assert_disjoint(merged: &[BufReq], moff: &HashMap<usize, u64>) {
    if cfg!(debug_assertions) {
        for i in 0..merged.len() {
            for j in (i + 1)..merged.len() {
                let (oi, oj) = (moff[&i], moff[&j]);
                let (ei, ej) = (oi + merged[i].size, oj + merged[j].size);
                let bytes_overlap = oi < ej && oj < ei;
                let live_overlap = match (merged[i].live, merged[j].live) {
                    (Some((s1, e1)), Some((s2, e2))) => s1 < e2 && s2 < e1,
                    _ => true, // no interval ⇒ live for the whole program
                };
                debug_assert!(
                    !(bytes_overlap && live_overlap),
                    "buffers '{}' and '{}' overlap in both bytes and liveness",
                    merged[i].name,
                    merged[j].name
                );
            }
        }
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) / a * a
}

/// Greedy linear-scan first-fit for the scratch region: place buffers in
/// live-start order at the lowest byte offset (≥ `base`) free of any
/// time-overlapping buffer. Returns each buffer's offset and the region's peak end.
fn plan_scratch(reqs: &[BufReq], base: u64) -> (HashMap<usize, u64>, u64) {
    struct Placed {
        off: u64,
        size: u64,
        start: Cycle,
        end: Cycle,
    }
    let mut idx: Vec<usize> = (0..reqs.len())
        .filter(|&i| reqs[i].class == BufClass::Scratch)
        .collect();
    idx.sort_by_key(|&i| reqs[i].live.map(|(s, _)| s).unwrap_or(0));

    let mut placed: Vec<Placed> = Vec::new();
    let mut offsets = HashMap::new();
    let mut peak = base;
    for i in idx {
        let (s, e) = reqs[i].live.unwrap_or((0, u64::MAX));
        let size = align_up(reqs[i].size, MEM_ALIGN);
        // Buffers whose interval overlaps [s, e) still occupy storage.
        let mut active: Vec<&Placed> = placed.iter().filter(|p| p.start < e && s < p.end).collect();
        active.sort_by_key(|p| p.off);
        // Lowest offset (≥ base) with no overlap: walk active in offset order.
        let mut off = base;
        for p in &active {
            if p.off + p.size <= off {
                continue; // entirely below the candidate
            }
            if p.off >= off + size {
                break; // [off, off+size) fits in the gap before p
            }
            off = align_up(p.off + p.size, MEM_ALIGN); // push above p
        }
        offsets.insert(i, off);
        placed.push(Placed {
            off,
            size,
            start: s,
            end: e,
        });
        peak = peak.max(off + size);
    }
    (offsets, peak)
}

/// Place a set of (already alias-merged) buffers into one arena, segregated by
/// lifetime class. Returns each buffer's index→offset, the arena size, and the
/// growable base.
fn place(reqs: &[BufReq]) -> (HashMap<usize, u64>, u64, u64) {
    let mut cursor = 0u64;
    let mut offsets: HashMap<usize, u64> = HashMap::new();
    // 1. Persistent (weights), packed in input order.
    for (i, r) in reqs.iter().enumerate() {
        if r.class == BufClass::Persistent {
            cursor = align_up(cursor, MEM_ALIGN);
            offsets.insert(i, cursor);
            cursor += r.size;
        }
    }
    // 1b. Static (compile-time-computed constants), adjacent to Persistent.
    for (i, r) in reqs.iter().enumerate() {
        if r.class == BufClass::Static {
            cursor = align_up(cursor, MEM_ALIGN);
            offsets.insert(i, cursor);
            cursor += r.size;
        }
    }
    // 2. Per-request I/O.
    for (i, r) in reqs.iter().enumerate() {
        if r.class == BufClass::RequestIo {
            cursor = align_up(cursor, MEM_ALIGN);
            offsets.insert(i, cursor);
            cursor += r.size;
        }
    }
    // 3. Scratch, reused by liveness.
    let scratch_base = align_up(cursor, MEM_ALIGN);
    let (scratch_off, scratch_peak) = plan_scratch(reqs, scratch_base);
    offsets.extend(scratch_off);
    cursor = scratch_peak;
    // 4. Growable, last so it can extend into free HBM.
    let growable_base = align_up(cursor, MEM_ALIGN);
    cursor = growable_base;
    for (i, r) in reqs.iter().enumerate() {
        if r.class == BufClass::Growable {
            cursor = align_up(cursor, MEM_ALIGN);
            offsets.insert(i, cursor);
            cursor += r.size + r.growth_reserve;
        }
    }
    (offsets, cursor, growable_base)
}

fn union_live(a: Option<(Cycle, Cycle)>, b: Option<(Cycle, Cycle)>) -> Option<(Cycle, Cycle)> {
    match (a, b) {
        (Some((s1, e1)), Some((s2, e2))) => Some((s1.min(s2), e1.max(e2))),
        (x, None) | (None, x) => x,
    }
}

/// Allocate every buffer into one arena. Slot ids are the buffer's index in
/// `reqs` (stable). A buffer with `alias_of` is a zero-copy view: it shares its
/// target's offset and contributes only its size/liveness to that shared buffer,
/// so no extra bytes are reserved. Growable buffers go last.
pub fn allocate(reqs: &[BufReq]) -> AddressMap {
    // Expand replicated tensors into one single-device buffer per device — each a
    // distinct slot placed in its device's segment, all sharing the tensor name.
    let reqs: Vec<BufReq> = reqs
        .iter()
        .flat_map(|r| {
            if r.replicas.is_empty() {
                vec![r.clone()]
            } else {
                r.replicas
                    .iter()
                    .map(|&d| BufReq {
                        device: d,
                        replicas: Vec::new(),
                        ..r.clone()
                    })
                    .collect()
            }
        })
        .collect();
    let reqs = &reqs[..];

    // Alias lookup is by (name, device): a view aliases the copy of its target on
    // its *own* device, so aliases stay intra-device even with replicas present.
    let idx: HashMap<(&str, u8), usize> = reqs
        .iter()
        .enumerate()
        .map(|(i, r)| ((r.name.as_str(), r.device), i))
        .collect();
    // Follow alias_of to a terminal root, accumulating the byte offset along the
    // chain (capped to avoid cycles).
    let root_and_off = |start: usize| -> (usize, u64) {
        let mut cur = start;
        let mut off = 0u64;
        for _ in 0..reqs.len() {
            let dev = reqs[cur].device;
            match reqs[cur]
                .alias_of
                .as_deref()
                .and_then(|t| idx.get(&(t, dev)))
            {
                Some(&n) if n != cur => {
                    off += reqs[cur].alias_off;
                    cur = n;
                }
                _ => break,
            }
        }
        (cur, off)
    };

    // Build merged roots: each root keeps its own class/growth; members fold in
    // their liveness (union) and extend the root to cover their sub-region
    // (`off + size`). Preserve `reqs` order for stability.
    let mut merged: Vec<BufReq> = Vec::new();
    let mut root_pos: HashMap<usize, usize> = HashMap::new(); // root index → merged slot
    for i in 0..reqs.len() {
        let (r, off) = root_and_off(i);
        let pos = *root_pos.entry(r).or_insert_with(|| {
            merged.push(reqs[r].clone());
            merged.len() - 1
        });
        // The root must span every member's [off, off+size).
        merged[pos].size = merged[pos].size.max(off + reqs[i].size);
        if i != r {
            merged[pos].live = union_live(merged[pos].live, reqs[i].live);
        }
    }

    // Place each device's roots in its own segment of the global address space;
    // lay the segments contiguously (`[seg0 | seg1 | …]`). One device ⇒ one
    // segment at base 0, so global offsets equal the single-arena layout.
    let devices: Vec<u8> = merged
        .iter()
        .map(|m| m.device)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut moff: HashMap<usize, u64> = HashMap::new();
    let mut segments = Vec::new();
    let mut global_base = 0u64;
    let mut growable_base = 0u64;
    for (k, &d) in devices.iter().enumerate() {
        let sub_idx: Vec<usize> = (0..merged.len())
            .filter(|&i| merged[i].device == d)
            .collect();
        let sub: Vec<BufReq> = sub_idx.iter().map(|&i| merged[i].clone()).collect();
        let (local_off, size, gbase_local) = place(&sub);
        for (li, &mi) in sub_idx.iter().enumerate() {
            moff.insert(mi, global_base + local_off[&li]);
        }
        let seg_growable = global_base + gbase_local;
        segments.push(Segment {
            device: d,
            global_base,
            size,
            growable_base: seg_growable,
        });
        if k == 0 {
            growable_base = seg_growable;
        }
        global_base += align_up(size, MEM_ALIGN);
    }
    let arena_bytes = global_base;
    debug_assert_disjoint(&merged, &moff);

    let entries = reqs
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let (root, off) = root_and_off(i);
            let pos = root_pos[&root];
            let m = &merged[pos];
            AddrEntry {
                slot: i as u32,
                name: r.name.clone(),
                class: m.class,
                offset: moff[&pos] + off,
                // each entry reserves only the bytes it occupies; aliases overlap
                // their root's span by design.
                reserved: if r.class == BufClass::Growable {
                    r.size + r.growth_reserve
                } else {
                    r.size
                },
                growable: m.class == BufClass::Growable,
                device: m.device,
            }
        })
        .collect();

    AddressMap {
        entries,
        arena_bytes,
        growable_base,
        segments,
    }
}

/// Derive buffer requests from a scheduled task graph and place them.
///
/// Classification by data-flow direction (writers vs readers of each tensor):
/// a tensor written by some Compute/DmaOut task is an intermediate ([`BufClass::Scratch`],
/// reused by liveness); a tensor that is only ever read (a leaf — weight or model
/// input) is [`BufClass::Persistent`]. Live intervals come from the schedule's
/// per-task `starts`: `[first writer start, last reader start + 1)`.
///
/// KV/growable classification is not derivable from the task graph alone (it needs
/// the bucket's `KvLayout`); those buffers are added by the caller. This covers
/// weights + activations, which is what the task graph names.
pub fn plan_from_schedule(
    tasks: &crate::expand::TaskGraph,
    sched: &crate::passes::Schedule,
    cons: &rewrite::ConstraintSet,
) -> AddressMap {
    plan_from_schedule_with_task_sets(tasks, sched, cons).0
}

/// Per-tensor writer / reader task-id sets (`plans/lean-formal-verification-analysis.md`
/// §6.2.5). Keyed by tensor name (matches `AddrEntry.name`). Callers who need
/// to feed the Lean verifier build a `ScheduleRequest` from these plus the
/// `AddressMap`.
pub type TensorTaskSets = HashMap<String, (Vec<crate::expand::TaskId>, Vec<crate::expand::TaskId>)>;

/// Like `plan_from_schedule` but also returns the writer/reader task-id set
/// for each tensor. Used by the Lean-verifier bridge; regular callers should
/// prefer `plan_from_schedule`.
pub fn plan_from_schedule_with_task_sets(
    tasks: &crate::expand::TaskGraph,
    sched: &crate::passes::Schedule,
    cons: &rewrite::ConstraintSet,
) -> (AddressMap, TensorTaskSets) {
    use crate::expand::{TaskId, TaskKind};
    use rewrite::OpKind;

    // Zero-storage aliases (Phase C): a buffer placed inside another at a byte
    // offset. Reshape → output aliases input 0 at offset 0 (identical bytes).
    let mut alias_of: HashMap<String, (String, u64)> = HashMap::new();
    for task in &tasks.tasks {
        if task.kind != TaskKind::Compute {
            continue;
        }
        if let Some(desc) = cons.op_io.get(&task.node) {
            if let OpKind::Layout(s) = desc.kind {
                if s.alias {
                    if let Some(in0) = desc.inputs.first() {
                        alias_of.insert(desc.output.clone(), (in0.clone(), 0));
                    }
                }
            }
        }
    }

    // Per tensor: max byte size, whether it has a writer, writer/reader cycles, and
    // the device it lives on (its writer's unit, else the first task touching it).
    // The writer/reader task lists back the Lean verifier's reclamation check
    // (§6.2.2): overlapping bytes are safe iff every reader of one is
    // happens-before every writer of the other.
    struct Acc {
        size: u64,
        has_writer: bool,
        first_write: Cycle,
        last_read: Cycle,
        device: u8,
        writers: Vec<TaskId>,
        readers: Vec<TaskId>,
    }
    let mut acc: HashMap<String, Acc> = HashMap::new();
    let start = |t: TaskId| sched.starts.get(t).copied().unwrap_or(0);

    for (tid, task) in tasks.tasks.iter().enumerate() {
        let Some(name) = &task.tensor else { continue };
        let e = acc.entry(name.clone()).or_insert(Acc {
            size: 0,
            has_writer: false,
            first_write: Cycle::MAX,
            last_read: 0,
            device: task.unit as u8,
            writers: Vec::new(),
            readers: Vec::new(),
        });
        // Size buffers from the op-level tensor size: under PerTile granularity
        // `task.bytes` is one tile's slice, and packing full tensors into
        // tile-sized regions would overlap adjacent buffers at runtime.
        e.size = e.size.max(task.bytes.max(task.tensor_bytes));
        match task.kind {
            TaskKind::Compute | TaskKind::DmaOut => {
                e.has_writer = true;
                e.first_write = e.first_write.min(start(tid));
                e.device = task.unit as u8; // a tensor lives where it is produced
                e.writers.push(tid);
            }
            TaskKind::DmaIn => {
                e.last_read = e.last_read.max(start(tid) + 1);
                e.readers.push(tid);
            }
            TaskKind::Host => {}
        }
    }

    // Contiguous concats: lay the parts end-to-end inside the output. Part 0 is
    // the group root; later parts alias it at their cumulative byte offset; the
    // output aliases the root at offset 0 (it spans the whole group).
    for grp in &cons.concat_groups {
        if grp.parts.len() < 2 || !acc.contains_key(&grp.parts[0]) {
            continue;
        }
        let root = &grp.parts[0];
        let mut off = acc[root].size;
        for p in &grp.parts[1..] {
            if acc.contains_key(p) {
                alias_of.insert(p.clone(), (root.clone(), off));
            }
            off += acc.get(p).map(|a| a.size).unwrap_or(0);
        }
        if acc.contains_key(&grp.output) {
            alias_of.insert(grp.output.clone(), (root.clone(), 0));
        }
    }

    // Stable order: by name, so slot ids are deterministic across runs.
    let mut names: Vec<&String> = acc.keys().collect();
    names.sort();
    let reqs: Vec<BufReq> = names
        .iter()
        .map(|name| {
            let a = &acc[*name];
            let mut req = if a.has_writer {
                let s = if a.first_write == Cycle::MAX {
                    0
                } else {
                    a.first_write
                };
                let e = a.last_read.max(s + 1);
                BufReq::new((*name).clone(), a.size, BufClass::Scratch).with_live(s, e)
            } else {
                BufReq::new((*name).clone(), a.size, BufClass::Persistent)
            }
            .on_device(a.device);
            // Only alias when the target is itself a known buffer in this map.
            if let Some((target, off)) = alias_of.get(name.as_str()) {
                if acc.contains_key(target) {
                    req = req.alias_at(target.clone(), *off);
                }
            }
            req
        })
        .collect();

    let map = allocate(&reqs);
    let task_sets: TensorTaskSets = acc
        .into_iter()
        .map(|(name, a)| (name, (a.writers, a.readers)))
        .collect();
    (map, task_sets)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn off(m: &AddressMap, name: &str) -> u64 {
        m.get(name).unwrap().offset
    }
    /// Byte ranges of two entries overlap.
    fn overlaps(a: &AddrEntry, b: &AddrEntry) -> bool {
        a.offset < b.offset + b.reserved && b.offset < a.offset + a.reserved
    }

    #[test]
    fn classes_are_segregated_and_growable_is_last() {
        let reqs = vec![
            BufReq::new("w0", 1000, BufClass::Persistent),
            BufReq::new("io", 64, BufClass::RequestIo),
            BufReq::new("s0", 500, BufClass::Scratch).with_live(0, 10),
            BufReq::new("kv", 2048, BufClass::Growable).with_growth(8192),
        ];
        let m = allocate(&reqs);
        assert_eq!(off(&m, "w0"), 0, "weights pack first");
        assert!(off(&m, "io") >= 1000);
        assert!(off(&m, "s0") > off(&m, "io"));
        assert_eq!(
            off(&m, "kv"),
            m.growable_base,
            "growable starts the growable region"
        );
        // growable reserves size + growth and is the top of the arena
        assert_eq!(m.get("kv").unwrap().reserved, 2048 + 8192);
        assert_eq!(m.arena_bytes, off(&m, "kv") + 2048 + 8192);
        // every offset is aligned
        for e in &m.entries {
            assert_eq!(e.offset % MEM_ALIGN, 0, "{} unaligned", e.name);
        }
    }

    #[test]
    fn disjoint_scratch_reuses_storage_overlapping_does_not() {
        // a and b are disjoint in time → share an offset; c overlaps both → distinct.
        let reqs = vec![
            BufReq::new("a", 4096, BufClass::Scratch).with_live(0, 10),
            BufReq::new("b", 4096, BufClass::Scratch).with_live(10, 20),
            BufReq::new("c", 4096, BufClass::Scratch).with_live(0, 20),
        ];
        let m = allocate(&reqs);
        assert_eq!(
            off(&m, "a"),
            off(&m, "b"),
            "disjoint-in-time scratch reuses bytes"
        );
        assert_ne!(
            off(&m, "a"),
            off(&m, "c"),
            "time-overlapping scratch is distinct"
        );
        assert_ne!(off(&m, "b"), off(&m, "c"));
        // peak arena holds only two 4 KiB buffers, not three.
        assert_eq!(m.arena_bytes, off(&m, "a") + 2 * 4096);
    }

    #[test]
    fn no_two_time_overlapping_buffers_share_bytes() {
        let reqs = vec![
            BufReq::new("a", 300, BufClass::Scratch).with_live(0, 5),
            BufReq::new("b", 700, BufClass::Scratch).with_live(2, 8),
            BufReq::new("c", 200, BufClass::Scratch).with_live(4, 9),
            BufReq::new("d", 900, BufClass::Scratch).with_live(0, 9),
        ];
        let m = allocate(&reqs);
        for (i, x) in m.entries.iter().enumerate() {
            for y in &m.entries[i + 1..] {
                let (lx, ly) = (
                    reqs.iter()
                        .find(|r| r.name == x.name)
                        .unwrap()
                        .live
                        .unwrap(),
                    reqs.iter()
                        .find(|r| r.name == y.name)
                        .unwrap()
                        .live
                        .unwrap(),
                );
                let time_overlap = lx.0 < ly.1 && ly.0 < lx.1;
                if time_overlap {
                    assert!(
                        !overlaps(x, y),
                        "{} and {} overlap in both time and bytes",
                        x.name,
                        y.name
                    );
                }
            }
        }
    }

    #[test]
    fn empty_input_is_empty_arena() {
        let m = allocate(&[]);
        assert_eq!(m.arena_bytes, 0);
        assert!(m.entries.is_empty());
    }

    #[test]
    fn validate_accepts_real_maps_and_rejects_out_of_bounds() {
        let reqs = vec![
            BufReq::new("w", 1000, BufClass::Persistent),
            BufReq::new("s", 500, BufClass::Scratch).with_live(0, 10),
        ];
        let m = allocate(&reqs);
        assert!(m.validate().is_ok(), "a freshly allocated map validates");

        // An entry past the arena end is rejected.
        let mut bad = m.clone();
        bad.arena_bytes = 100;
        assert!(bad.validate().is_err(), "out-of-bounds buffer is caught");
    }

    #[test]
    fn alias_shares_offset_and_reserves_no_extra_bytes() {
        // `view` is a reshape of `src`: it shares src's offset and adds no bytes.
        let reqs = vec![
            BufReq::new("src", 4096, BufClass::Scratch).with_live(0, 10),
            BufReq::new("view", 4096, BufClass::Scratch)
                .with_live(10, 20)
                .alias("src"),
            BufReq::new("other", 4096, BufClass::Scratch).with_live(0, 25),
        ];
        let m = allocate(&reqs);
        assert_eq!(
            off(&m, "view"),
            off(&m, "src"),
            "alias shares the target's offset"
        );
        // src and other overlap 'view's extended liveness → distinct; arena holds
        // only two 4 KiB buffers (src/view shared, + other), not three.
        assert_ne!(off(&m, "src"), off(&m, "other"));
        assert_eq!(m.arena_bytes, off(&m, "src").max(off(&m, "other")) + 4096);
    }

    #[test]
    fn concat_adjacency_places_parts_contiguously_under_output() {
        // out = concat(a, b): a at the group base, b right after a, out covers both.
        // Realized by offset-aliasing — zero bytes moved.
        let reqs = vec![
            BufReq::new("a", 256, BufClass::Scratch).with_live(0, 10),
            BufReq::new("b", 256, BufClass::Scratch)
                .with_live(0, 10)
                .alias_at("a", 256),
            BufReq::new("out", 512, BufClass::Scratch)
                .with_live(5, 20)
                .alias_at("a", 0),
        ];
        let m = allocate(&reqs);
        let base = off(&m, "a");
        assert_eq!(off(&m, "out"), base, "output starts at the group base");
        assert_eq!(off(&m, "b"), base + 256, "b is placed right after a");
        // One 512-byte group, not three separate buffers.
        assert_eq!(m.arena_bytes, base + 512);
    }

    #[test]
    fn devices_form_contiguous_global_segments() {
        // Two devices: each gets a segment; global offsets are contiguous and each
        // buffer lands in its own device's segment.
        let reqs = vec![
            BufReq::new("d0a", 256, BufClass::Persistent).on_device(0),
            BufReq::new("d0b", 256, BufClass::Persistent).on_device(0),
            BufReq::new("d1a", 256, BufClass::Persistent).on_device(1),
        ];
        let m = allocate(&reqs);
        assert_eq!(m.segments.len(), 2, "one segment per device");
        let seg1 = m.segments.iter().find(|s| s.device == 1).unwrap();
        // device 0 holds two 256B buffers ⇒ device 1's segment starts at 512.
        assert_eq!(seg1.global_base, 512);
        assert_eq!(
            m.get("d1a").unwrap().offset,
            512,
            "d1a lives in device 1's segment"
        );
        assert_eq!(m.get("d1a").unwrap().device, 1);
        assert_eq!(m.get("d0a").unwrap().device, 0);
        assert_eq!(m.arena_bytes, 512 + 256);
        assert!(m.validate().is_ok());
    }

    #[test]
    fn replicated_tensor_gets_one_copy_per_device() {
        // An all-gathered activation replicated on devices 0 and 1: two physical
        // copies, one in each device's segment, both addressable by name+device.
        let reqs = vec![
            BufReq::new("w0", 256, BufClass::Persistent).on_device(0),
            BufReq::new("act", 512, BufClass::Persistent).replicated_on([0u8, 1u8]),
        ];
        let m = allocate(&reqs);
        let r: Vec<_> = m.replicas("act").collect();
        assert_eq!(r.len(), 2, "one copy per device");
        let a0 = m.get_on_device("act", 0).unwrap();
        let a1 = m.get_on_device("act", 1).unwrap();
        assert_eq!(a0.device, 0);
        assert_eq!(a1.device, 1);
        assert_ne!(a0.slot, a1.slot, "each replica is its own slot");
        // Each replica lives in its device's segment.
        let seg1 = m.segments.iter().find(|s| s.device == 1).unwrap();
        assert!(a1.offset >= seg1.global_base && a1.offset < seg1.global_base + seg1.size);
        assert!(m.validate().is_ok());
    }

    #[test]
    fn single_device_is_one_segment_at_base_zero() {
        let reqs = vec![BufReq::new("x", 100, BufClass::Persistent)];
        let m = allocate(&reqs);
        assert_eq!(m.segments.len(), 1);
        assert_eq!(m.segments[0].global_base, 0);
        assert_eq!(
            m.get("x").unwrap().offset,
            0,
            "global == local for one device"
        );
    }

    #[test]
    fn alias_chain_resolves_to_root() {
        // c → b → a : all three share a's offset.
        let reqs = vec![
            BufReq::new("a", 256, BufClass::Persistent),
            BufReq::new("b", 256, BufClass::Persistent).alias("a"),
            BufReq::new("c", 256, BufClass::Persistent).alias("b"),
        ];
        let m = allocate(&reqs);
        assert_eq!(off(&m, "a"), off(&m, "b"));
        assert_eq!(off(&m, "b"), off(&m, "c"));
        assert_eq!(
            m.arena_bytes, 256,
            "one buffer's worth of bytes for the whole chain"
        );
    }

    #[test]
    fn concat_group_lays_parts_inside_output_via_schedule() {
        use crate::expand::{Task, TaskGraph, TaskKind};
        use crate::passes::Schedule;
        use rewrite::{ConcatGroup, ConstraintSet};
        use std::collections::HashMap;

        let mk = |kind, tensor: &str, bytes| Task {
            node: 0,
            op: "x".into(),
            unit: 0,
            kind,
            coord: vec![],
            dur: 1,
            bytes,
            tensor_bytes: 0,
            sram_pages: 0,
            out_pages: 0,
            tmem_cols: 0,
            tensor: Some(tensor.into()),
            cross_unit: false,
        };
        // a, b produced; out = concat(a, b) produced then read.
        let mut tasks = TaskGraph::default();
        tasks.tasks = vec![
            mk(TaskKind::Compute, "a", 256),
            mk(TaskKind::Compute, "b", 256),
            mk(TaskKind::Compute, "out", 512),
            mk(TaskKind::DmaIn, "out", 512),
        ];
        let sched = Schedule {
            streams: HashMap::new(),
            packets: HashMap::new(),
            counters: vec![],
            placement: HashMap::new(),
            starts: vec![0, 0, 5, 9],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 10,
        };
        let mut cons = ConstraintSet::default();
        cons.concat_groups = vec![ConcatGroup {
            output: "out".into(),
            parts: vec!["a".into(), "b".into()],
        }];

        let m = plan_from_schedule(&tasks, &sched, &cons);
        let base = off(&m, "a");
        assert_eq!(
            off(&m, "out"),
            base,
            "concat output starts at the group base"
        );
        assert_eq!(
            off(&m, "b"),
            base + 256,
            "second part placed right after the first"
        );
    }

    #[test]
    fn classifies_leaves_persistent_and_intermediates_scratch() {
        use crate::expand::{Task, TaskGraph, TaskKind};
        use crate::passes::Schedule;
        use std::collections::HashMap;

        let mk = |op: &str, kind, tensor: &str, bytes| Task {
            node: 0,
            op: op.into(),
            unit: 0,
            kind,
            coord: vec![],
            dur: 1,
            bytes,
            tensor_bytes: 0,
            sram_pages: 0,
            out_pages: 0,
            tmem_cols: 0,
            tensor: Some(tensor.into()),
            cross_unit: false,
        };
        // t0: load weight "w" (read-only leaf). t1: compute writes "act". t2: load "act".
        let mut tasks = TaskGraph::default();
        tasks.tasks = vec![
            mk("dma", TaskKind::DmaIn, "w", 1024),
            mk("gemm", TaskKind::Compute, "act", 512),
            mk("dma", TaskKind::DmaIn, "act", 512),
        ];
        let sched = Schedule {
            streams: HashMap::new(),
            packets: HashMap::new(),
            counters: vec![],
            placement: HashMap::new(),
            starts: vec![0, 5, 9],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 10,
        };
        let m = plan_from_schedule(&tasks, &sched, &rewrite::ConstraintSet::default());
        assert_eq!(
            m.get("w").unwrap().class,
            BufClass::Persistent,
            "weight leaf is persistent"
        );
        let act = m.get("act").unwrap();
        assert_eq!(act.class, BufClass::Scratch, "produced tensor is scratch");
        // weight packed first (offset 0), activation lands after it.
        assert_eq!(m.get("w").unwrap().offset, 0);
        assert!(act.offset >= 1024);
    }
}
