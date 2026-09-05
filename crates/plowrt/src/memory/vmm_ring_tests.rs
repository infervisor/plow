use super::*;
use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    Reserve,
    Create,
    Map,
    Access,
}
#[derive(Default)]
struct Ledger {
    calls: Vec<Call>,
    fail: Option<(Call, usize)>,
    next_va: u64,
    next_handle: u64,
    reserved: BTreeMap<u64, u64>,
    handles: BTreeMap<u64, u64>,
    mapped: BTreeMap<u64, (u64, u64, bool)>,
}
#[derive(Default)]
struct Mock(Mutex<Ledger>);
impl Mock {
    fn call(&self, call: Call) -> Result<()> {
        let mut s = self.0.lock().unwrap();
        s.calls.push(call);
        if let Some((stage, remaining)) = s.fail.as_mut() {
            if *stage == call {
                *remaining -= 1;
                if *remaining == 0 {
                    s.fail = None;
                    return Err(RuntimeError::Oom("injected ring allocation failure".into()));
                }
            }
        }
        Ok(())
    }
    fn fail(&self, call: Call, after: usize) {
        self.0.lock().unwrap().fail = Some((call, after));
    }
    fn calls(&self) -> usize {
        self.0.lock().unwrap().calls.len()
    }
    fn empty(&self) {
        let s = self.0.lock().unwrap();
        assert!(s.reserved.is_empty() && s.handles.is_empty() && s.mapped.is_empty());
    }
}
impl VmmOps for Mock {
    fn granularity(&self) -> Result<u64> {
        Ok(64)
    }
    fn reserve(&self, bytes: u64) -> Result<u64> {
        self.call(Call::Reserve)?;
        let mut s = self.0.lock().unwrap();
        let va = 0x10000000 + s.next_va;
        s.next_va += bytes;
        assert_eq!(s.reserved.insert(va, bytes), None);
        Ok(va)
    }
    fn address_free(&self, va: u64, bytes: u64) {
        let mut s = self.0.lock().unwrap();
        assert!(!s.mapped.keys().any(|&p| p >= va && p < va + bytes));
        assert_eq!(s.reserved.remove(&va), Some(bytes));
    }
    fn create(&self, bytes: u64) -> Result<u64> {
        self.call(Call::Create)?;
        let mut s = self.0.lock().unwrap();
        s.next_handle += 1;
        let handle = s.next_handle;
        assert_eq!(s.handles.insert(handle, bytes), None);
        Ok(handle)
    }
    fn release(&self, handle: u64) {
        let mut s = self.0.lock().unwrap();
        assert!(!s.mapped.values().any(|&(_, h, _)| h == handle));
        assert!(s.handles.remove(&handle).is_some());
    }
    fn map(&self, va: u64, bytes: u64, handle: u64) -> Result<()> {
        self.call(Call::Map)?;
        let mut s = self.0.lock().unwrap();
        assert_eq!(s.handles.get(&handle), Some(&bytes));
        assert!(s
            .reserved
            .iter()
            .any(|(&p, &n)| va >= p && va + bytes <= p + n));
        assert!(!s
            .mapped
            .iter()
            .any(|(&p, &(n, h, _))| h == handle || (va < p + n && p < va + bytes)));
        assert_eq!(s.mapped.insert(va, (bytes, handle, false)), None);
        Ok(())
    }
    fn unmap(&self, va: u64, bytes: u64) {
        let mut s = self.0.lock().unwrap();
        let (n, _, _) = s.mapped.remove(&va).expect("exact mapped range");
        assert_eq!(n, bytes);
    }
    fn set_access(&self, va: u64, bytes: u64) -> Result<()> {
        self.call(Call::Access)?;
        let mut s = self.0.lock().unwrap();
        let m = s.mapped.get_mut(&va).expect("mapped before access");
        assert_eq!(m.0, bytes);
        m.2 = true;
        Ok(())
    }
    fn alloc(&self, _: u64) -> Result<u64> {
        panic!("rings cannot allocate snapshots")
    }
    fn free(&self, _: u64) {
        panic!("rings cannot free snapshots")
    }
    fn copy_dtod(&self, _: u64, _: u64, _: u64) -> Result<()> {
        panic!("rings cannot copy prefixes")
    }
}

fn tensors() -> [LiveRingTensor; 2] {
    [
        LiveRingTensor {
            tensor: 3,
            slot_bytes: 128,
        },
        LiveRingTensor {
            tensor: 8,
            slot_bytes: 256,
        },
    ]
}

#[test]
fn cold_prefill_holes_expand_to_actual_prefix_and_retain_without_driver_calls() {
    let ops = Arc::new(Mock::default());
    let mut rings = VmmRings::new(ops.clone(), &tensors(), 16).unwrap();
    let bases = [rings.tensor_va(3).unwrap(), rings.tensor_va(8).unwrap()];
    assert_eq!(rings.stats().reserved_bytes, 16 * 384);
    for (slot, rows, prefill_count) in [(0, 1, 1), (3, 4, 2), (15, 16, 5)] {
        rings.ensure_slot(slot).unwrap();
        assert_eq!(rings.stats().mapped_slots, prefill_count);
        rings.ensure_prefix(rows).unwrap();
        assert_eq!(rings.stats().mapped_prefix, rows);
        assert_eq!(rings.stats().resident_bytes, rows as u64 * 384);
    }
    let calls = ops.calls();
    for _ in 0..16 {
        for slot in 0..16 {
            rings.ensure_slot(slot).unwrap();
            rings.ensure_prefix(slot + 1).unwrap();
        }
    }
    assert_eq!(
        ops.calls(),
        calls,
        "retained reset/re-entry must not call the driver"
    );
    assert_eq!(
        bases,
        [rings.tensor_va(3).unwrap(), rings.tensor_va(8).unwrap()]
    );
    assert!(rings.ensure_slot(16).is_err() && rings.ensure_prefix(17).is_err());
    assert_eq!(ops.calls(), calls);
    let s = ops.0.lock().unwrap();
    assert_eq!(s.mapped.len(), 32);
    assert!(s.mapped.values().all(|m| m.2));
    drop(s);
    drop(rings);
    ops.empty();
}

#[test]
fn failed_new_slot_unwinds_every_partial_stage_without_disturbing_old_slots() {
    for stage in [Call::Create, Call::Map, Call::Access] {
        for nth in 1..=2 {
            let ops = Arc::new(Mock::default());
            let mut rings = VmmRings::new(ops.clone(), &tensors(), 16).unwrap();
            rings.ensure_slot(0).unwrap();
            let stats = rings.stats();
            let maps = ops.0.lock().unwrap().mapped.clone();
            ops.fail(stage, nth);
            assert!(rings.ensure_slot(3).is_err(), "{stage:?} {nth}");
            assert_eq!(rings.stats(), stats);
            assert_eq!(ops.0.lock().unwrap().mapped, maps);
            assert_eq!(ops.0.lock().unwrap().handles.len(), 2);
            rings.ensure_prefix(4).unwrap();
            assert_eq!(rings.stats().mapped_slots, 4);
            assert!(ops.0.lock().unwrap().mapped.values().all(|m| m.2));
            drop(rings);
            ops.empty();
        }
    }
}

#[test]
fn constructor_failure_and_invalid_geometry_leak_no_reservations() {
    for nth in 1..=2 {
        let ops = Arc::new(Mock::default());
        ops.fail(Call::Reserve, nth);
        assert!(VmmRings::new(ops.clone(), &tensors(), 16).is_err());
        ops.empty();
    }
    let cases = [
        (
            vec![LiveRingTensor {
                tensor: 0,
                slot_bytes: 0,
            }],
            16,
        ),
        (
            vec![LiveRingTensor {
                tensor: 0,
                slot_bytes: 65,
            }],
            16,
        ),
        (vec![tensors()[0], tensors()[0]], 16),
        (
            vec![LiveRingTensor {
                tensor: 0,
                slot_bytes: u64::MAX - 63,
            }],
            16,
        ),
        (tensors().to_vec(), 0),
    ];
    for (t, b) in cases {
        let ops = Arc::new(Mock::default());
        assert!(VmmRings::new(ops.clone(), &t, b).is_err());
        assert_eq!(ops.calls(), 0);
        ops.empty();
    }
}

#[test]
fn serialized_concurrent_ensure_calls_do_not_duplicate_backing() {
    let ops = Arc::new(Mock::default());
    let rings = Mutex::new(VmmRings::new(ops.clone(), &tensors(), 16).unwrap());
    std::thread::scope(|scope| {
        for rows in [4, 16, 1, 8, 16, 4] {
            let rings = &rings;
            scope.spawn(move || {
                rings.lock().unwrap().ensure_prefix(rows).unwrap();
            });
        }
    });
    assert_eq!(rings.lock().unwrap().stats().mapped_slots, 16);
    assert_eq!(ops.0.lock().unwrap().handles.len(), 32);
    drop(rings);
    ops.empty();
}

#[test]
fn failed_prefix_expansion_exposes_only_complete_accessible_slots() {
    let ops = Arc::new(Mock::default());
    let mut rings = VmmRings::new(ops.clone(), &tensors(), 16).unwrap();
    ops.fail(Call::Access, 5);
    assert!(rings.ensure_prefix(4).is_err());
    assert_eq!(rings.stats().mapped_prefix, 2);
    assert_eq!(rings.stats().mapped_slots, 2);
    assert_eq!(rings.stats().resident_bytes, 2 * 384);
    assert!(ops.0.lock().unwrap().mapped.values().all(|m| m.2));
    assert_eq!(ops.0.lock().unwrap().handles.len(), 4);
    rings.ensure_prefix(4).unwrap();
    assert_eq!(rings.stats().mapped_prefix, 4);
    drop(rings);
    ops.empty();
}

#[test]
#[ignore = "CPU actual ring packet; set TEST_LIVE_RING_PACKET"]
fn actual_packet_ring_prefix_reservations() {
    let path = std::env::var("TEST_LIVE_RING_PACKET").unwrap();
    let blob = crate::asset::devblob::DevBlob::parse(&std::fs::read(&path).unwrap()).unwrap();
    let layout = LiveKvLayout::from_blob(&blob).unwrap();
    assert_eq!(layout.geometry.batch, 16);
    assert!(!layout.ring_tensors.is_empty());
    let per_slot: u64 = layout.ring_tensors.iter().map(|t| t.slot_bytes).sum();
    for t in &layout.ring_tensors {
        assert_eq!(t.slot_bytes % (2 << 20), 0);
        assert_eq!(t.slot_bytes * 16, blob.tensors[t.tensor].bytes);
    }
    let ops = Arc::new(Mock::default());
    let mut rings = VmmRings::new(ops.clone(), &layout.ring_tensors, 16).unwrap();
    for (slot, prefix) in [(0, 1), (3, 4), (15, 16)] {
        rings.ensure_slot(slot).unwrap();
        rings.ensure_prefix(prefix).unwrap();
        assert_eq!(rings.stats().resident_bytes, per_slot * prefix as u64);
        assert_eq!(rings.stats().mapped_slots, prefix);
    }
    eprintln!(
        "{path}: {} ring tensors, {per_slot} bytes per physical slot, retained prefix1/4/16 passed",
        layout.ring_tensors.len()
    );
    drop(rings);
    ops.empty();
}
