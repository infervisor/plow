//! Rust bindings to the CPU kernel library (`runtime/cpu/dev/cpu_dev.h`).
//!
//! Hand-written rather than bindgen'd: six functions and one struct do not
//! justify libclang in the nix build. The layout is locked the same way the
//! device ISA is (`packet/tests/dev_abi.rs`): `tests/cpu_abi.rs` asks the C
//! compiler for `sizeof`/`offsetof` through `plow_cpu_abi_*` probes emitted by
//! build.rs and compares them to Rust's. The instruction record itself is
//! [`DevInst64`], already ABI-locked, so it is reused rather than redefined.
//!
//! The interpreter never sees C types beyond this module: kernels are resolved
//! once per program into a [`KernelTable`], and a missing op is a typed `None`,
//! never a null call.

use std::ffi::c_void;

pub use packet::dev::DevInst64;

use crate::{Result, RuntimeError};

/// Kernel tier (`PLOW_CPU_ISA_*`), ordered. `init(cap)` never activates above `cap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum Isa {
    Scalar = 0,
    Avx512 = 1,
    Amx = 2,
}

impl Isa {
    pub fn from_i32(v: i32) -> Option<Isa> {
        match v {
            0 => Some(Isa::Scalar),
            1 => Some(Isa::Avx512),
            2 => Some(Isa::Amx),
            _ => None,
        }
    }
}

/// Mirror of `PlowCpuCtx`: the per-worker-thread context every kernel receives.
/// Fixed 64 bytes. Owned by the worker, zero-initialised, `scratch` points at a
/// 64-byte-aligned arena of at least [`scratch_bytes`] bytes.
#[derive(Debug)]
#[repr(C)]
pub struct PlowCpuCtx {
    pub scratch: *mut c_void,
    pub scratch_bytes: u32,
    pub worker: u32,
    pub node: u32,
    /// Active tier for this thread; written by [`thread_init`].
    pub isa: u32,
    pub reserved: [u64; 5],
}

impl PlowCpuCtx {
    /// A context with no scratch. Give it scratch before running kernels that
    /// need it ([`scratch_bytes`] > 0).
    pub fn new(worker: u32, node: u32) -> Self {
        PlowCpuCtx {
            scratch: std::ptr::null_mut(),
            scratch_bytes: 0,
            worker,
            node,
            isa: 0,
            reserved: [0; 5],
        }
    }
}

// SAFETY: the raw `scratch` pointer is owned by exactly one worker thread; the
// struct is moved to that thread once and never shared.
unsafe impl Send for PlowCpuCtx {}

/// `plow_cpu_kernel_fn`: compute the `slice`-th of `nblk` shares of `inst`
/// over host pointers `tensors[handle]`.
pub type KernelFn = unsafe extern "C" fn(
    inst: *const DevInst64,
    slice: u32,
    nblk: u32,
    tensors: *const *mut c_void,
    ctx: *mut PlowCpuCtx,
);

extern "C" {
    fn plow_cpu_init(isa_cap: i32) -> i32;
    fn plow_cpu_isa() -> i32;
    fn plow_cpu_thread_init(ctx: *mut PlowCpuCtx) -> i32;
    fn plow_cpu_scratch_bytes() -> u32;
    fn plow_cpu_has(op: u16) -> i32;
    fn plow_cpu_kernel(op: u16) -> Option<KernelFn>;
    fn plow_cpu_exec(
        inst: *const DevInst64,
        slice: u32,
        nblk: u32,
        tensors: *const *mut c_void,
        ctx: *mut PlowCpuCtx,
    ) -> i32;
    fn plow_cpu_prepack_bf16_b_bytes(n: u32, k: u32) -> usize;
    fn plow_cpu_prepack_bf16_b(dst: *mut c_void, src: *const c_void, n: u32, k: u32) -> i32;
}

/// Process-wide init (cpuid, AMX permission, dispatch table). Idempotent.
/// Returns the tier actually activated, which is `<= cap`.
pub fn init(cap: Isa) -> Result<Isa> {
    // SAFETY: plain FFI, no pointers.
    let rc = unsafe { plow_cpu_init(cap as i32) };
    if rc < 0 {
        return Err(RuntimeError::Device(format!(
            "plow_cpu_init(cap={cap:?}) failed: {rc}"
        )));
    }
    Isa::from_i32(rc).ok_or_else(|| {
        RuntimeError::Device(format!("plow_cpu_init returned unknown tier {rc}"))
    })
}

/// Active tier, `None` before [`init`].
pub fn isa() -> Option<Isa> {
    // SAFETY: plain FFI, no pointers.
    Isa::from_i32(unsafe { plow_cpu_isa() })
}

/// Per-thread init (AMX tile config, `ctx.isa`). Call on the worker thread,
/// after [`init`], before its first kernel.
pub fn thread_init(ctx: &mut PlowCpuCtx) -> Result<()> {
    // SAFETY: `ctx` is a valid, exclusively borrowed PlowCpuCtx.
    let rc = unsafe { plow_cpu_thread_init(ctx) };
    if rc != 0 {
        return Err(RuntimeError::Device(format!(
            "plow_cpu_thread_init failed: {rc}"
        )));
    }
    Ok(())
}

/// Scratch bytes a worker must hand to kernels via `PlowCpuCtx::scratch`.
pub fn scratch_bytes() -> u32 {
    // SAFETY: plain FFI.
    unsafe { plow_cpu_scratch_bytes() }
}

/// Whether `op` has a kernel at the active tier.
pub fn has(op: u16) -> bool {
    // SAFETY: plain FFI.
    unsafe { plow_cpu_has(op) != 0 }
}

/// Resolve `op` to its kernel. Load-time only — see [`KernelTable`] for the
/// per-program resolution the interpreter uses.
pub fn kernel(op: u16) -> Option<KernelFn> {
    // SAFETY: plain FFI; a NULL return maps to `None` via the Option<fn> ABI.
    unsafe { plow_cpu_kernel(op) }
}

/// Lookup + call in one FFI hop. Convenience for tests and one-off ops; the
/// interpreter calls resolved [`KernelFn`]s directly.
///
/// # Safety
/// `tensors` must hold a valid host pointer for every handle `inst` names,
/// sized for the op's extent; `ctx` must have been through [`thread_init`] on
/// this thread and carry adequate scratch.
pub unsafe fn exec(
    inst: &DevInst64,
    slice: u32,
    nblk: u32,
    tensors: &[*mut c_void],
    ctx: &mut PlowCpuCtx,
) -> Result<()> {
    let rc = plow_cpu_exec(inst, slice, nblk, tensors.as_ptr(), ctx);
    if rc != 0 {
        return Err(RuntimeError::Device(format!(
            "no CPU kernel for op {}",
            inst.op
        )));
    }
    Ok(())
}

/// Per-program kernel table: `op → KernelFn`, resolved once at load so the
/// per-packet path is an indexed load, never an FFI lookup.
#[derive(Debug)]
pub struct KernelTable {
    fns: Vec<Option<KernelFn>>,
}

impl KernelTable {
    /// Resolve every distinct op in `ops`. `Err` lists the ops with no kernel
    /// at the active tier (deduplicated, ascending) so the loader can name them.
    pub fn resolve(ops: impl Iterator<Item = u16>) -> std::result::Result<Self, Vec<u16>> {
        let mut fns: Vec<Option<KernelFn>> = Vec::new();
        let mut missing = Vec::new();
        for op in ops {
            let i = op as usize;
            if i >= fns.len() {
                fns.resize(i + 1, None);
            }
            if fns[i].is_some() {
                continue;
            }
            match kernel(op) {
                Some(f) => fns[i] = Some(f),
                None => {
                    if !missing.contains(&op) {
                        missing.push(op);
                    }
                }
            }
        }
        if missing.is_empty() {
            Ok(KernelTable { fns })
        } else {
            missing.sort_unstable();
            Err(missing)
        }
    }

    #[inline]
    pub fn get(&self, op: u16) -> Option<KernelFn> {
        self.fns.get(op as usize).copied().flatten()
    }
}

/// Bytes the AMX/VNNI-packed copy of a bf16 `[n][k]` weight occupies.
pub fn prepack_bf16_b_bytes(n: u32, k: u32) -> usize {
    // SAFETY: plain FFI.
    unsafe { plow_cpu_prepack_bf16_b_bytes(n, k) }
}

/// Repack a bf16 weight `src[n][k]` into the AMX/VNNI B layout in `dst`.
/// `dst.len() * 2` must be at least [`prepack_bf16_b_bytes`].
pub fn prepack_bf16_b(dst: &mut [u16], src: &[u16], n: u32, k: u32) -> Result<()> {
    let need = prepack_bf16_b_bytes(n, k);
    if src.len() * 2 < (n as usize) * (k as usize) * 2 || dst.len() * 2 < need {
        return Err(RuntimeError::Device(format!(
            "prepack_bf16_b: n={n} k={k} needs {need} B dst, {} B src; got {} / {}",
            (n as usize) * (k as usize) * 2,
            dst.len() * 2,
            src.len() * 2
        )));
    }
    // SAFETY: both slices are bounds-checked above for the op's extent.
    let rc = unsafe {
        plow_cpu_prepack_bf16_b(
            dst.as_mut_ptr() as *mut c_void,
            src.as_ptr() as *const c_void,
            n,
            k,
        )
    };
    if rc != 0 {
        return Err(RuntimeError::Device(format!(
            "plow_cpu_prepack_bf16_b(n={n}, k={k}) failed: {rc}"
        )));
    }
    Ok(())
}

/// Layout probes emitted by build.rs (`abi_probe.c`) for `tests/cpu_abi.rs`.
pub mod abi {
    extern "C" {
        pub fn plow_cpu_abi_sizeof_ctx() -> usize;
        pub fn plow_cpu_abi_sizeof_inst() -> usize;
        pub fn plow_cpu_abi_offsetof_ctx_scratch() -> usize;
        pub fn plow_cpu_abi_offsetof_ctx_scratch_bytes() -> usize;
        pub fn plow_cpu_abi_offsetof_ctx_worker() -> usize;
        pub fn plow_cpu_abi_offsetof_ctx_node() -> usize;
        pub fn plow_cpu_abi_offsetof_ctx_isa() -> usize;
        pub fn plow_cpu_abi_offsetof_ctx_reserved() -> usize;
        pub fn plow_cpu_abi_offsetof_inst_op() -> usize;
        pub fn plow_cpu_abi_offsetof_inst_blocks() -> usize;
        pub fn plow_cpu_abi_offsetof_inst_fj() -> usize;
        pub fn plow_cpu_abi_offsetof_inst_t() -> usize;
        pub fn plow_cpu_abi_offsetof_inst_i() -> usize;
        pub fn plow_cpu_abi_isa_scalar() -> i32;
        pub fn plow_cpu_abi_isa_avx512() -> i32;
        pub fn plow_cpu_abi_isa_amx() -> i32;
        pub fn plow_cpu_abi_dop_table() -> i32;
    }
}

/// `PLOW_CPU_DOP_TABLE`: dispatch-table extent on the C side.
pub const DOP_TABLE: usize = 256;
