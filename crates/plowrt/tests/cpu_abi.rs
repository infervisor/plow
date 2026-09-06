//! ABI lock for the CPU kernel library bindings.
//!
//! `exec::cpu::ffi` mirrors `runtime/cpu/dev/cpu_dev.h` by hand. build.rs
//! compiles a probe TU against the real header that reports the C compiler's
//! `sizeof`/`offsetof`; this test compares them to Rust's so a field added on
//! either side fails here instead of as a kernel reading a stale word.
//!
//! Passes against both the no-op stub library (no kernel sources yet) and the
//! real one: functional checks branch on `has(op)`.
#![cfg(feature = "cpu")]

use std::ffi::c_void;
use std::mem::{offset_of, size_of};

use plowrt::exec::cpu::ffi::{self, abi, DevInst64, Isa, KernelTable, PlowCpuCtx};

const NOP: u16 = 0;

#[test]
fn ctx_layout_matches_c() {
    // SAFETY: probes are pure functions returning constants.
    unsafe {
        assert_eq!(abi::plow_cpu_abi_sizeof_ctx(), size_of::<PlowCpuCtx>());
        assert_eq!(size_of::<PlowCpuCtx>(), 64, "PlowCpuCtx is one cache line");
        assert_eq!(abi::plow_cpu_abi_offsetof_ctx_scratch(), offset_of!(PlowCpuCtx, scratch));
        assert_eq!(
            abi::plow_cpu_abi_offsetof_ctx_scratch_bytes(),
            offset_of!(PlowCpuCtx, scratch_bytes)
        );
        assert_eq!(abi::plow_cpu_abi_offsetof_ctx_worker(), offset_of!(PlowCpuCtx, worker));
        assert_eq!(abi::plow_cpu_abi_offsetof_ctx_node(), offset_of!(PlowCpuCtx, node));
        assert_eq!(abi::plow_cpu_abi_offsetof_ctx_isa(), offset_of!(PlowCpuCtx, isa));
        assert_eq!(abi::plow_cpu_abi_offsetof_ctx_reserved(), offset_of!(PlowCpuCtx, reserved));
    }
}

#[test]
fn inst_layout_matches_c() {
    // SAFETY: probes are pure functions returning constants.
    unsafe {
        assert_eq!(abi::plow_cpu_abi_sizeof_inst(), size_of::<DevInst64>());
        assert_eq!(size_of::<DevInst64>(), 64);
        assert_eq!(abi::plow_cpu_abi_offsetof_inst_op(), offset_of!(DevInst64, op));
        assert_eq!(abi::plow_cpu_abi_offsetof_inst_blocks(), offset_of!(DevInst64, blocks));
        assert_eq!(abi::plow_cpu_abi_offsetof_inst_fj(), offset_of!(DevInst64, fj));
        assert_eq!(abi::plow_cpu_abi_offsetof_inst_t(), offset_of!(DevInst64, t));
        assert_eq!(abi::plow_cpu_abi_offsetof_inst_i(), offset_of!(DevInst64, i));
    }
}

#[test]
fn constants_match_c() {
    // SAFETY: probes are pure functions returning constants.
    unsafe {
        assert_eq!(abi::plow_cpu_abi_isa_scalar(), Isa::Scalar as i32);
        assert_eq!(abi::plow_cpu_abi_isa_avx512(), Isa::Avx512 as i32);
        assert_eq!(abi::plow_cpu_abi_isa_amx(), Isa::Amx as i32);
        assert_eq!(abi::plow_cpu_abi_dop_table() as usize, ffi::DOP_TABLE);
    }
}

#[test]
fn init_reports_a_tier_within_cap() {
    let tier = ffi::init(Isa::Scalar).expect("scalar init");
    assert_eq!(tier, Isa::Scalar, "cap is a ceiling");
    assert_eq!(ffi::isa(), Some(tier));
    // Idempotent; a higher cap may or may not raise the tier on this host, but
    // must never exceed it.
    let tier2 = ffi::init(Isa::Amx).expect("amx-capped init");
    assert!(tier2 <= Isa::Amx);
    assert_eq!(ffi::isa(), Some(tier2));
}

#[test]
fn nop_dispatch_is_consistent_with_has() {
    ffi::init(Isa::Scalar).expect("init");
    let mut ctx = PlowCpuCtx::new(0, 0);
    ffi::thread_init(&mut ctx).expect("thread init");
    // `scratch_bytes` is what a worker must provide; keep the kernel honest
    // even under the stub (0).
    let need = ffi::scratch_bytes() as usize;
    let mut scratch = vec![0u8; need.max(64)];
    ctx.scratch = scratch.as_mut_ptr() as *mut c_void;
    ctx.scratch_bytes = scratch.len() as u32;

    let inst = DevInst64 {
        op: NOP,
        blocks: 1,
        ..Default::default()
    };
    let tensors: [*mut c_void; 0] = [];

    if ffi::has(NOP) {
        let f = ffi::kernel(NOP).expect("has() implies kernel()");
        // SAFETY: NOP names no tensor handles; ctx is thread-initialised.
        unsafe { f(&inst, 0, 1, tensors.as_ptr(), &mut ctx) };
        // SAFETY: as above.
        unsafe { ffi::exec(&inst, 0, 1, &tensors, &mut ctx) }.expect("exec NOP");
        let table = KernelTable::resolve([NOP].into_iter()).expect("resolve NOP");
        assert!(table.get(NOP).is_some());
    } else {
        assert!(ffi::kernel(NOP).is_none());
        // SAFETY: exec must reject without touching operands.
        assert!(unsafe { ffi::exec(&inst, 0, 1, &tensors, &mut ctx) }.is_err());
        assert_eq!(KernelTable::resolve([NOP].into_iter()).unwrap_err(), vec![NOP]);
    }
    // Unknown ops are never present, in either library.
    assert!(!ffi::has(0xFFFE));
    assert!(KernelTable::resolve([0xFFFEu16].into_iter()).is_err());
}
