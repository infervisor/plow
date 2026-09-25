use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct Lease {
    id: u64,
    live: AtomicBool,
}

impl Lease {
    fn new() -> Arc<Self> {
        let id = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .expect("allocation provenance identity exhausted");
        Arc::new(Self {
            id,
            live: AtomicBool::new(true),
        })
    }

    fn invalidate(&self) {
        self.live.store(false, Ordering::Release);
    }
}

#[derive(Clone, Debug)]
struct Span {
    va: u64,
    bytes: u64,
    physical: Arc<Lease>,
    physical_offset: u64,
    mapping: Arc<Lease>,
}

/// Process-local allocation and mapping identities, not GPU retirement evidence.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct AllocationRange {
    pub virtual_base: u64,
    pub bytes: u64,
    pub physical_id: u64,
    pub physical_offset: u64,
    pub binding_generation: u64,
}

pub(crate) fn ranges_disjoint(a: &[AllocationRange], b: &[AllocationRange]) -> bool {
    a.iter().all(|a| {
        b.iter().all(|b| {
            a.physical_id != b.physical_id
                || a.physical_offset
                    .checked_add(a.bytes)
                    .is_some_and(|end| end <= b.physical_offset)
                || b.physical_offset
                    .checked_add(b.bytes)
                    .is_some_and(|end| end <= a.physical_offset)
        })
    })
}

/// Metadata pins identities only; it never keeps device storage alive.
#[derive(Clone, Debug)]
pub struct MemoryRegion {
    spans: Arc<[Span]>,
    base: u64,
    bytes: u64,
}

impl MemoryRegion {
    pub(crate) fn matches_range(&self, base: u64, bytes: u64) -> bool {
        self.base == base && self.bytes == bytes
    }

    pub(crate) fn owned(base: u64, bytes: u64) -> Self {
        let physical = Lease::new();
        Self {
            spans: vec![Span {
                va: base,
                bytes,
                physical: physical.clone(),
                physical_offset: 0,
                mapping: physical,
            }]
            .into(),
            base,
            bytes,
        }
    }

    pub(crate) fn subrange(&self, base: u64, bytes: u64) -> Option<Self> {
        if base < self.base || base.checked_add(bytes)? > self.base.checked_add(self.bytes)? {
            return None;
        }
        Some(Self {
            spans: self.spans.clone(),
            base,
            bytes,
        })
    }

    pub(crate) fn invalidate_owner(&self) {
        for span in self.spans.iter() {
            span.physical.invalidate();
        }
    }

    pub(crate) fn is_live(&self) -> bool {
        let Some(end) = self.base.checked_add(self.bytes) else {
            return false;
        };
        let mut cursor = self.base;
        for span in self.spans.iter() {
            let Some(span_end) = span.va.checked_add(span.bytes) else {
                return false;
            };
            let lo = self.base.max(span.va);
            let hi = end.min(span_end);
            if hi <= lo {
                continue;
            }
            if lo != cursor
                || !span.physical.live.load(Ordering::Acquire)
                || !span.mapping.live.load(Ordering::Acquire)
            {
                return false;
            }
            cursor = hi;
        }
        self.bytes != 0 && cursor == end
    }

    /// Call at admission/quiescent audit points, not concurrently with rebinding.
    /// A successful snapshot is not a lease authorizing subsequent GPU execution.
    pub fn evidence(&self, base: u64, bytes: u64) -> Option<Vec<AllocationRange>> {
        if base != self.base || bytes != self.bytes || bytes == 0 {
            return None;
        }
        let end = base.checked_add(bytes)?;
        let mut cursor = base;
        let mut out = Vec::new();
        for span in self.spans.iter() {
            let lo = base.max(span.va);
            let hi = end.min(span.va.checked_add(span.bytes)?);
            if hi <= lo {
                continue;
            }
            if lo != cursor
                || !span.physical.live.load(Ordering::Acquire)
                || !span.mapping.live.load(Ordering::Acquire)
            {
                return None;
            }
            out.push(AllocationRange {
                virtual_base: lo,
                bytes: hi - lo,
                physical_id: span.physical.id,
                physical_offset: span.physical_offset.checked_add(lo - span.va)?,
                binding_generation: span.mapping.id,
            });
            cursor = hi;
        }
        (cursor == end).then_some(out)
    }
}

#[derive(Default)]
pub(crate) struct VmmProvenance {
    physical: BTreeMap<u64, (Arc<Lease>, u64)>,
    mapped: BTreeMap<u64, Span>,
    ordinary: BTreeMap<u64, MemoryRegion>,
    invalid: bool,
}

impl VmmProvenance {
    // Only successful driver allocations/maps enter the inventory. Unexpected
    // driver/caller state fails evidence closed without changing driver ownership.
    pub(crate) fn created(&mut self, handle: u64, bytes: u64) {
        if let Some((old, _)) = self.physical.insert(handle, (Lease::new(), bytes)) {
            old.invalidate();
            self.invalidate();
        }
    }

    pub(crate) fn released(&mut self, handle: u64) {
        if let Some((lease, _)) = self.physical.remove(&handle) {
            lease.invalidate();
        }
    }

    pub(crate) fn mapped(&mut self, va: u64, bytes: u64, handle: u64) {
        let Some((physical, capacity)) = self.physical.get(&handle) else {
            self.invalidate();
            return;
        };
        if bytes == 0
            || bytes > *capacity
            || va.checked_add(bytes).is_none()
            || self
                .mapped
                .range(..va.saturating_add(bytes))
                .next_back()
                .is_some_and(|(_, m)| m.va + m.bytes > va)
        {
            self.invalidate();
            return;
        }
        self.mapped.insert(
            va,
            Span {
                va,
                bytes,
                physical: physical.clone(),
                physical_offset: 0,
                mapping: Lease::new(),
            },
        );
    }

    pub(crate) fn unmapped(&mut self, va: u64, bytes: u64) {
        match self.mapped.get(&va) {
            Some(m) if m.bytes == bytes => {
                self.mapped.remove(&va).unwrap().mapping.invalidate();
            }
            // Partial unmaps are not modeled: never leave stale evidence live.
            _ => self.invalidate(),
        }
    }

    fn invalidate(&mut self) {
        self.invalid = true;
        for m in self.mapped.values() {
            m.mapping.invalidate();
        }
        for r in self.ordinary.values() {
            r.invalidate_owner();
        }
    }

    pub(crate) fn allocated(&mut self, va: u64, bytes: u64) {
        if let Some(old) = self.ordinary.insert(va, MemoryRegion::owned(va, bytes)) {
            old.invalidate_owner();
            self.invalidate();
        }
    }

    pub(crate) fn freed(&mut self, va: u64) {
        if let Some(r) = self.ordinary.remove(&va) {
            r.invalidate_owner();
        }
    }

    pub(crate) fn region(&self, va: u64, bytes: u64) -> Option<MemoryRegion> {
        if self.invalid || bytes == 0 {
            return None;
        }
        let end = va.checked_add(bytes)?;
        if let Some((_, r)) = self.ordinary.range(..=va).next_back() {
            if let Some(sub) = r.subrange(va, bytes) {
                sub.evidence(va, bytes)?;
                return Some(sub);
            }
        }
        let first = self
            .mapped
            .range(..=va)
            .next_back()
            .map_or(va, |(&start, _)| start);
        let spans: Vec<_> = self
            .mapped
            .range(first..end)
            .filter(|(_, m)| m.va + m.bytes > va)
            .map(|(_, m)| m.clone())
            .collect();
        let region = MemoryRegion {
            spans: spans.into(),
            base: va,
            bytes,
        };
        region.evidence(va, bytes)?;
        Some(region)
    }
}

impl Drop for VmmProvenance {
    fn drop(&mut self) {
        self.invalidate();
        for (lease, _) in self.physical.values() {
            lease.invalidate();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_aliases_and_mapping_generations() {
        let mut p = VmmProvenance::default();
        p.created(7, 128);
        p.mapped(1024, 128, 7);
        p.mapped(2048, 128, 7);
        let a = p.region(1024, 128).unwrap();
        let b = p.region(2080, 32).unwrap();
        let ae = a.evidence(1024, 128).unwrap();
        let be = b.evidence(2080, 32).unwrap();
        assert_eq!(ae[0].physical_id, be[0].physical_id);
        assert_eq!(be[0].physical_offset, 32);
        assert!(
            !ranges_disjoint(&ae, &be),
            "disjoint VAs may alias physically"
        );
        let prefix = p.region(1024, 32).unwrap().evidence(1024, 32).unwrap();
        assert!(ranges_disjoint(&prefix, &be));
        assert_ne!(ae[0].binding_generation, be[0].binding_generation);
        p.unmapped(1024, 128);
        assert!(a.evidence(1024, 128).is_none());
        assert!(b.evidence(2080, 32).is_some());
        p.mapped(1024, 128, 7);
        let new = p.region(1024, 128).unwrap().evidence(1024, 128).unwrap();
        assert_eq!(new[0].physical_id, ae[0].physical_id);
        assert_ne!(new[0].binding_generation, ae[0].binding_generation);
        p.released(7);
        assert!(b.evidence(2080, 32).is_none());
        p.unmapped(1024, 128);
        p.unmapped(2048, 128);
        p.created(7, 128);
        p.mapped(1024, 128, 7);
        assert_ne!(
            p.region(1024, 128).unwrap().evidence(1024, 128).unwrap()[0].physical_id,
            ae[0].physical_id
        );
    }

    #[test]
    fn spanning_carves_require_complete_live_backing() {
        let mut p = VmmProvenance::default();
        p.created(1, 64);
        p.created(2, 64);
        p.mapped(1024, 64, 1);
        assert!(p.region(1056, 64).is_none());
        p.mapped(1088, 64, 2);
        let r = p.region(1056, 64).unwrap();
        let e = r.evidence(1056, 64).unwrap();
        assert_eq!(e.len(), 2);
        assert_ne!(e[0].physical_id, e[1].physical_id);
        assert!(r.evidence(1057, 64).is_none());
        assert!(r.subrange(1055, 1).is_none());
        assert!(r.subrange(1056, u64::MAX).is_none());
        p.unmapped(1088, 64);
        assert!(r.evidence(1056, 64).is_none());
        assert!(r.subrange(1056, 32).unwrap().evidence(1056, 32).is_some());
    }

    #[test]
    fn malformed_mapping_and_tracker_drop_invalidate_snapshots() {
        for bad in 0..6 {
            let mut p = VmmProvenance::default();
            p.created(1, 64);
            p.mapped(1024, 64, 1);
            let r = p.region(1024, 64).unwrap();
            match bad {
                0 => p.mapped(2048, 64, 99),
                1 => p.mapped(2048, 65, 1),
                2 => p.mapped(1025, 32, 1),
                3 => p.unmapped(1024, 32),
                4 => p.created(1, 64),
                _ => drop(p),
            }
            assert!(r.evidence(1024, 64).is_none());
        }
    }

    #[test]
    fn snapshot_allocations_reject_freed_and_reused_addresses() {
        let mut p = VmmProvenance::default();
        p.allocated(1024, 64);
        let r = p.region(1040, 32).unwrap();
        let old = r.evidence(1040, 32).unwrap();
        assert_eq!(old[0].physical_offset, 16);
        p.freed(1024);
        p.allocated(1024, 64);
        assert!(r.evidence(1040, 32).is_none());
        assert_ne!(
            old[0].physical_id,
            p.region(1040, 32).unwrap().evidence(1040, 32).unwrap()[0].physical_id
        );
        assert!(p.region(1024, 65).is_none());
    }

    #[test]
    #[ignore = "CPU metadata microbenchmark; run optimized, no GPU or speedup qualification"]
    fn provenance_metadata_cost() {
        use std::hint::black_box;
        use std::time::Instant;
        const N: u64 = 100_000;
        fn median(mut values: Vec<f64>) -> f64 {
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        }
        let mut allocation = Vec::new();
        let mut subview = Vec::new();
        let mut mapping = Vec::new();
        for _ in 0..7 {
            let start = Instant::now();
            for _ in 0..N {
                black_box(Arc::new(MemoryRegion::owned(1024, 1024)));
            }
            allocation.push(start.elapsed().as_nanos() as f64 / N as f64);
            let owner = MemoryRegion::owned(1024, 1024);
            let start = Instant::now();
            for _ in 0..N {
                black_box(Arc::new(owner.subrange(1056, 64).unwrap()));
            }
            subview.push(start.elapsed().as_nanos() as f64 / N as f64);
            let p = parking_lot::Mutex::new(VmmProvenance::default());
            let start = Instant::now();
            for i in 0..N {
                p.lock().created(i, 64);
                p.lock().mapped(1024, 64, i);
                p.lock().unmapped(1024, 64);
                p.lock().released(i);
            }
            mapping.push(start.elapsed().as_nanos() as f64 / N as f64);
        }
        eprintln!(
            "provenance-only medians ns: allocation+drop={:.2} subview+drop={:.2} \
            create+map+unmap+release+4_uncontended_locks={:.2}; no driver calls",
            median(allocation),
            median(subview),
            median(mapping)
        );
    }
}
