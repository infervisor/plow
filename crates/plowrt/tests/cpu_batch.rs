//! Batched-decode plumbing that needs no model: the ladder rung rule and the
//! shared tensor table a `kv_rebase` edits in place.
#![cfg(feature = "cpu")]

use std::ffi::c_void;

use plowrt::exec::cpu::engine::{rung_for, TensorTable};

#[test]
fn rung_rule_picks_narrowest_covering_rung() {
    let rungs = [1u32, 2, 4, 8];
    assert_eq!(rung_for(&rungs, 1), 0);
    assert_eq!(rung_for(&rungs, 2), 1);
    assert_eq!(rung_for(&rungs, 3), 2);
    assert_eq!(rung_for(&rungs, 4), 2);
    assert_eq!(rung_for(&rungs, 5), 3);
    assert_eq!(rung_for(&rungs, 8), 3);
    // Past the widest rung: the widest (the caller rejects the slot table).
    assert_eq!(rung_for(&rungs, 9), 3);
    // Single-rung blob (batch 1).
    assert_eq!(rung_for(&[1], 1), 0);
    assert_eq!(rung_for(&[], 1), 0);
}

#[test]
fn tensor_table_rebase_is_visible_through_the_kernel_view() {
    let mut backing = vec![0u8; 4096];
    let base = backing.as_mut_ptr() as *mut c_void;
    let t = TensorTable::new(vec![base, base.wrapping_add(1024)]);
    assert_eq!(t.len(), 2);
    // Kernels read the table as a `*mut c_void[]`.
    let view = t.as_ptr();
    assert_eq!(unsafe { *view.add(1) }, base.wrapping_add(1024));
    // Rebase slot 1 onto its second block; the kernel view follows in place.
    unsafe { t.set(1, base.wrapping_add(1024 + 512)) };
    assert_eq!(t.get(1), base.wrapping_add(1536));
    assert_eq!(unsafe { *view.add(1) }, base.wrapping_add(1536));
    assert_eq!(t.get(0), base);
}
