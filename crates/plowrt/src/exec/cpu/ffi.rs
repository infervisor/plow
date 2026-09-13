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

pub use packet::dev::{DevInst64, TokenBatch};

use crate::{Result, RuntimeError};

static INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    fn plow_cpu_tier_of(op: u16) -> i32;
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
    fn plow_token_batch_validate_host(tb: *const TokenBatch) -> i32;
    fn plow_token_row_host(tb: *const TokenBatch, row: u32, out: *mut TokenRowFlat) -> i32;
    fn plow_token_sample_row_host(tb: *const TokenBatch, s: u32, out: *mut u32) -> i32;
    fn plow_token_batch_descriptor_version() -> u32;
}

/// Mirror of `PlowTokenRowFlat` (`runtime/cpu/dev/cpu_dev.h`): one resolved packed row, with
/// the span reported as an INDEX because the device struct's span POINTER does not survive the
/// FFI boundary. `span == SPAN_NONE` is a padding row, which is also the only case where
/// `active == 0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct TokenRowFlat {
    pub span: u32,
    pub local_row: u32,
    pub slot: u32,
    pub state_slot: u32,
    pub position: u32,
    pub active: u32,
}

impl TokenRowFlat {
    /// `PLOW_TB_SPAN_NONE`: this row is padding and belongs to no span.
    pub const SPAN_NONE: u32 = u32::MAX;
}

/// `PLOW_TB_*` refusal codes from `runtime/common/token_batch.h`. Each names ONE invariant,
/// because a refusal that does not say what it refused is indistinguishable from a crash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenBatchRefusal(pub i32);

impl std::fmt::Display for TokenBatchRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0 {
            -1 => "descriptor or a required array is null",
            -2 => "descriptor version this build does not implement",
            -3 => "reserved flags set",
            -4 => "real_rows exceeds row_capacity, or capacity is zero",
            -5 => "span count disagrees with the live row count",
            -6 => "spans do not cover [0, M) exactly (gap, overlap or zero length)",
            -7 => "positions[] disagrees with the span's arithmetic",
            -8 => "park mask disagrees with the span cover",
            -9 => "kv_len != kv_row0 + n_rows",
            -10 => "a sample index is not a live row",
            -11 => "row index outside row_capacity",
            _ => "unknown token-batch refusal",
        })
    }
}

/// Validate a token-batch descriptor through the SHARED resolver source, before any launch.
///
/// # Safety
/// `tb`'s pointer fields must be valid HOST pointers for the extents its counts declare. This
/// is the host twin: on a device the same header traps instead of returning.
pub unsafe fn token_batch_validate(tb: &TokenBatch) -> std::result::Result<(), TokenBatchRefusal> {
    match plow_token_batch_validate_host(tb) {
        0 => Ok(()),
        rc => Err(TokenBatchRefusal(rc)),
    }
}

/// Resolve one packed row through the shared resolver.
///
/// # Safety
/// As [`token_batch_validate`].
pub unsafe fn token_row(
    tb: &TokenBatch,
    row: u32,
) -> std::result::Result<TokenRowFlat, TokenBatchRefusal> {
    let mut out = TokenRowFlat::default();
    match plow_token_row_host(tb, row, &mut out) {
        0 => Ok(out),
        rc => Err(TokenBatchRefusal(rc)),
    }
}

/// The `s`-th selected hidden row, as an index into the body's rows.
///
/// # Safety
/// As [`token_batch_validate`].
pub unsafe fn token_sample_row(
    tb: &TokenBatch,
    s: u32,
) -> std::result::Result<u32, TokenBatchRefusal> {
    let mut out = 0u32;
    match plow_token_sample_row_host(tb, s, &mut out) {
        0 => Ok(out),
        rc => Err(TokenBatchRefusal(rc)),
    }
}

/// `PLOW_TOKEN_BATCH_VERSION` this library was compiled against. Compared against
/// [`packet::dev::TOKEN_BATCH_VERSION`] so a half-rebuilt tree fails loudly.
pub fn token_batch_descriptor_version() -> u32 {
    // SAFETY: plain FFI, no pointers.
    unsafe { plow_token_batch_descriptor_version() }
}

/// Process-wide init (cpuid, AMX permission, dispatch table). Idempotent.
/// Returns the tier actually activated, which is `<= cap`.
pub fn init(cap: Isa) -> Result<Isa> {
    let _guard = INIT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // SAFETY: serialized plain FFI, no pointers.
    let rc = unsafe { plow_cpu_init(cap as i32) };
    if rc < 0 {
        return Err(RuntimeError::Device(format!(
            "plow_cpu_init(cap={cap:?}) failed: {rc}"
        )));
    }
    let tier = Isa::from_i32(rc)
        .ok_or_else(|| RuntimeError::Device(format!("plow_cpu_init returned unknown tier {rc}")))?;
    if tier > cap {
        return Err(RuntimeError::Device(format!(
            "CPU kernels were already initialized at {tier:?}, above requested cap {cap:?}"
        )));
    }
    Ok(tier)
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

/// The tier `op` resolves to, or `None` when no kernel is registered.
pub fn tier_of(op: u16) -> Option<Isa> {
    // SAFETY: pure table read.
    Isa::from_i32(unsafe { plow_cpu_tier_of(op) })
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
        pub fn plow_cpu_abi_sizeof_token_row() -> usize;
        pub fn plow_cpu_abi_sizeof_token_batch() -> usize;
        pub fn plow_cpu_abi_offsetof_token_row_active() -> usize;
        pub fn plow_cpu_abi_tb_span_none() -> i32;
    }
}

/// `PLOW_CPU_DOP_TABLE`: dispatch-table extent on the C side.
pub const DOP_TABLE: usize = 256;

#[cfg(test)]
mod f32_packet_tests {
    use super::*;
    use packet::dev::{DevOp, TENSOR_NONE16};

    fn inst(op: DevOp) -> DevInst64 {
        DevInst64 {
            op: op as u16,
            blocks: 1,
            fj: [0; 3],
            t: [TENSOR_NONE16; 8],
            i: [0; 8],
        }
    }

    #[test]
    fn q8_projection_and_layer_norm_execute_as_packets() {
        init(Isa::Scalar).unwrap();
        let input: Vec<f32> = (0..64).map(|i| i as f32 / 16.0 - 2.0).collect();
        let mut weight = Vec::new();
        for row in 0..2 {
            weight.extend_from_slice(&0x3c00u16.to_le_bytes());
            weight.extend((0..32).map(|i| (i as i8 - 16 + row) as u8));
        }
        let mut projected = vec![0.0f32; 4];
        let mut table = vec![std::ptr::null_mut(); 4];
        table[0] = projected.as_mut_ptr().cast();
        table[1] = input.as_ptr().cast_mut().cast();
        table[2] = weight.as_ptr().cast_mut().cast();
        let mut q8 = inst(DevOp::Q8GemmF32);
        q8.t[..4].copy_from_slice(&[0, 1, 2, TENSOR_NONE16]);
        q8.i[..4].copy_from_slice(&[2, 2, 32, 0]);
        let mut ctx = PlowCpuCtx::new(0, 0);
        unsafe { kernel(q8.op).unwrap()(&q8, 0, 1, table.as_ptr(), &mut ctx) };
        for row in 0..2 {
            for column in 0..2 {
                let expected = (0..32)
                    .map(|i| input[row * 32 + i] * (i as f32 - 16.0 + column as f32))
                    .sum::<f32>();
                assert!((projected[row * 2 + column] - expected).abs() < 1e-5);
            }
        }

        let gamma = [1.0f32, 1.0];
        let beta = [0.0f32, 0.0];
        let mut normalized = [0.0f32; 4];
        table[0] = normalized.as_mut_ptr().cast();
        table[1] = projected.as_ptr().cast_mut().cast();
        table[2] = gamma.as_ptr().cast_mut().cast();
        table[3] = beta.as_ptr().cast_mut().cast();
        let mut norm = inst(DevOp::LayerNormF32);
        norm.t[..4].copy_from_slice(&[0, 1, 2, 3]);
        norm.i[..3].copy_from_slice(&[2, 2, 3]);
        norm.fj[0] = 1e-5f32.to_bits();
        unsafe { kernel(norm.op).unwrap()(&norm, 0, 1, table.as_ptr(), &mut ctx) };
        assert!(normalized.iter().all(|value| value.is_finite()));
        assert!(normalized.iter().all(|value| value.to_bits() & 0xffff == 0));
        assert!((normalized[0] + normalized[1]).abs() < 1e-5);
        assert!((normalized[2] + normalized[3]).abs() < 1e-5);
    }

    #[test]
    fn rnnt_f32_primitives_execute_as_packets() {
        init(Isa::Scalar).unwrap();
        let mut ctx = PlowCpuCtx::new(0, 0);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [1.0f32, 0.0, 0.0, 1.0];
        let bias = [-2.0f32, -1.0];
        let mut projected = [0.0f32; 4];
        let mut table = vec![std::ptr::null_mut(); 5];
        table[0] = projected.as_mut_ptr().cast();
        table[1] = input.as_ptr().cast_mut().cast();
        table[2] = weight.as_ptr().cast_mut().cast();
        table[3] = bias.as_ptr().cast_mut().cast();
        let mut dense = inst(DevOp::DenseGemmF32);
        dense.t[..4].copy_from_slice(&[0, 1, 2, 3]);
        dense.i[..4].copy_from_slice(&[2, 2, 2, 1]);
        unsafe { kernel(dense.op).unwrap()(&dense, 0, 1, table.as_ptr(), &mut ctx) };
        assert_eq!(projected, [0.0, 1.0, 1.0, 3.0]);

        let strided_weight = [1.0f32, 0.0, 10.0, 0.0, 1.0, 20.0];
        table[2] = strided_weight.as_ptr().cast_mut().cast();
        dense.i[5] = 3;
        dense.i[6] = 2;
        unsafe { kernel(dense.op).unwrap()(&dense, 0, 1, table.as_ptr(), &mut ctx) };
        assert_eq!(projected, [9.0, 21.0, 11.0, 23.0]);

        let bf16_weight = [0x3f80u16, 0, 0, 0x3f80];
        table[2] = bf16_weight.as_ptr().cast_mut().cast();
        dense.i[5] = 0;
        dense.i[6] = 0;
        dense.i[7] = 4;
        unsafe { kernel(dense.op).unwrap()(&dense, 0, 1, table.as_ptr(), &mut ctx) };
        assert_eq!(projected, [0.0, 1.0, 1.0, 3.0]);

        let embedding = [0x3f80u16, 0x4000, 0x4040, 0x4080];
        let tokens = [1u32, 0, 0];
        let overlay = [1.001f32, -2.25];
        let overlay_index = [u32::MAX, 0, u32::MAX];
        let mut spliced = [0u16; 6];
        table[0] = spliced.as_mut_ptr().cast();
        table[1] = embedding.as_ptr().cast_mut().cast();
        table[2] = tokens.as_ptr().cast_mut().cast();
        table[3] = overlay.as_ptr().cast_mut().cast();
        table[4] = overlay_index.as_ptr().cast_mut().cast();
        let mut splice = inst(DevOp::EmbedOverlayBf16);
        splice.t[..5].copy_from_slice(&[0, 1, 2, 3, 4]);
        splice.i[..4].copy_from_slice(&[3, 2, 2, 1]);
        unsafe { kernel(splice.op).unwrap()(&splice, 0, 1, table.as_ptr(), &mut ctx) };
        let bits = overlay[0].to_bits();
        let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
        assert_eq!(spliced, [0x4040, 0x4080, rounded, 0xc010, 0x3f80, 0x4000]);

        let convolution_weight = [0x3c00u16, 0xbc00];
        let convolution_bias = [0.0f32; 2];
        let mut convolution_output = [f32::NAN; 8];
        table[0] = convolution_output.as_mut_ptr().cast();
        table[1] = input.as_ptr().cast_mut().cast();
        table[2] = convolution_weight.as_ptr().cast_mut().cast();
        table[3] = convolution_bias.as_ptr().cast_mut().cast();
        let mut convolution = inst(DevOp::Conv2dF32);
        convolution.t[..4].copy_from_slice(&[0, 1, 2, 3]);
        convolution.i.copy_from_slice(&[2, 2, 1, 2, 1, 1, 0, 0]);
        convolution.fj[1] = 2 | 4;
        unsafe { kernel(convolution.op).unwrap()(&convolution, 0, 1, table.as_ptr(), &mut ctx) };
        assert_eq!(convolution_output, [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);

        let convolution_weight_f32 = [1.0f32, -1.0];
        table[2] = convolution_weight_f32.as_ptr().cast_mut().cast();
        convolution_output.fill(f32::NAN);
        convolution.fj[1] = (2 << 2) | (2 << 4) | (1 << 6);
        convolution.fj[2] = 2;
        convolution.i[..4].copy_from_slice(&[1, 2, 1, 2]);
        unsafe { kernel(convolution.op).unwrap()(&convolution, 0, 1, table.as_ptr(), &mut ctx) };
        assert_eq!(
            convolution_output,
            [1.0, 2.0, -1.0, -2.0, 3.0, 4.0, -3.0, -4.0]
        );

        let packed_input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut packed_rows = [f32::NAN; 6];
        table[0] = packed_rows.as_mut_ptr().cast();
        table[1] = packed_input.as_ptr().cast_mut().cast();
        let mut pack = inst(DevOp::PackNcfwRowsF32);
        pack.t[..2].copy_from_slice(&[0, 1]);
        pack.i[..5].copy_from_slice(&[3, 2, 1, 2, 2]);
        unsafe { kernel(pack.op).unwrap()(&pack, 0, 1, table.as_ptr(), &mut ctx) };
        assert_eq!(packed_rows, [1.0, 3.0, 2.0, 4.0, 5.0, 7.0]);
        packed_rows.fill(f32::NAN);
        pack.i[0] = 5;
        unsafe { kernel(pack.op).unwrap()(&pack, 0, 1, table.as_ptr(), &mut ctx) };
        assert!(packed_rows.iter().all(|value| value.is_nan()));

        let attention_query = [0.0f32; 8];
        let attention_key = [0.0f32; 8];
        let attention_value = [1.0f32, 10.0, 3.0, 30.0, 100.0, 1000.0, 300.0, 3000.0];
        let mut attention_output = [f32::NAN; 8];
        table[0] = attention_output.as_mut_ptr().cast();
        table[1] = attention_query.as_ptr().cast_mut().cast();
        table[2] = attention_key.as_ptr().cast_mut().cast();
        table[3] = attention_value.as_ptr().cast_mut().cast();
        let mut attention = inst(DevOp::GroupedAttentionF32);
        attention.t[..4].copy_from_slice(&[0, 1, 2, 3]);
        attention.i[..5].copy_from_slice(&[4, 2, 1, 2, 0]);
        unsafe { kernel(attention.op).unwrap()(&attention, 0, 1, table.as_ptr(), &mut ctx) };
        assert_eq!(
            attention_output,
            [2.0, 20.0, 2.0, 20.0, 200.0, 2000.0, 200.0, 2000.0]
        );
        let valid_rows = 1u32;
        table[4] = (&valid_rows as *const u32).cast_mut().cast();
        attention.t[4] = 4;
        attention_output.fill(f32::NAN);
        unsafe { kernel(attention.op).unwrap()(&attention, 0, 1, table.as_ptr(), &mut ctx) };
        assert_eq!(&attention_output[..2], &[1.0, 10.0]);
        assert!(attention_output[2..].iter().all(|value| value.is_nan()));

        let dense_input = [1.001f32];
        let dense_weight = [1.0f32];
        let mut rounded = [f32::NAN];
        table[0] = rounded.as_mut_ptr().cast();
        table[1] = dense_input.as_ptr().cast_mut().cast();
        table[2] = dense_weight.as_ptr().cast_mut().cast();
        let mut dense_bf16 = inst(DevOp::DenseGemmF32);
        dense_bf16.t[..4].copy_from_slice(&[0, 1, 2, TENSOR_NONE16]);
        dense_bf16.i[..4].copy_from_slice(&[1, 1, 1, 0]);
        dense_bf16.i[7] = 1;
        unsafe { kernel(dense_bf16.op).unwrap()(&dense_bf16, 0, 1, table.as_ptr(), &mut ctx) };
        let bits = dense_input[0].to_bits();
        let expected = f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000);
        assert_eq!(rounded, [expected]);

        let addend = [0.0f32];
        table[0] = rounded.as_mut_ptr().cast();
        table[1] = dense_input.as_ptr().cast_mut().cast();
        table[2] = addend.as_ptr().cast_mut().cast();
        let mut scaled_add = inst(DevOp::ScaledAddF32);
        scaled_add.t[..3].copy_from_slice(&[0, 1, 2]);
        scaled_add.i[..2].copy_from_slice(&[1, 1]);
        scaled_add.fj[0] = 1.0f32.to_bits();
        unsafe { kernel(scaled_add.op).unwrap()(&scaled_add, 0, 1, table.as_ptr(), &mut ctx) };
        assert_eq!(rounded, [expected]);

        let query = [1.0f32, 0.0, 0.0, 1.0];
        let key = query;
        let value = [1.0f32, 2.0, 3.0, 4.0];
        let mut attended = [f32::NAN; 4];
        table[0] = attended.as_mut_ptr().cast();
        table[1] = query.as_ptr().cast_mut().cast();
        table[2] = key.as_ptr().cast_mut().cast();
        table[3] = value.as_ptr().cast_mut().cast();
        let mut attention = inst(DevOp::GroupedAttentionF32);
        attention.t[..4].copy_from_slice(&[0, 1, 2, 3]);
        attention.i[..5].copy_from_slice(&[2, 2, 2, 2, 0]);
        unsafe { kernel(attention.op).unwrap()(&attention, 0, 1, table.as_ptr(), &mut ctx) };
        assert!((attended[0] - 1.6604769).abs() < 1e-6);
        assert!((attended[1] - 2.660477).abs() < 1e-6);
        assert!((attended[2] - 2.339523).abs() < 1e-6);
        assert!((attended[3] - 3.339523).abs() < 1e-6);

        let gates = [0.0f32; 8];
        let previous = [1.0f32; 2];
        let mut hidden = [0.0f32; 2];
        let mut cell = [0.0f32; 2];
        table[0] = hidden.as_mut_ptr().cast();
        table[1] = cell.as_mut_ptr().cast();
        table[2] = gates.as_ptr().cast_mut().cast();
        table[3] = previous.as_ptr().cast_mut().cast();
        let mut lstm = inst(DevOp::LstmCellF32);
        lstm.t[..4].copy_from_slice(&[0, 1, 2, 3]);
        lstm.i[0] = 2;
        unsafe { kernel(lstm.op).unwrap()(&lstm, 0, 1, table.as_ptr(), &mut ctx) };
        assert_eq!(cell, [0.5, 0.5]);
        let expected = 0.5 * 0.5f32.tanh();
        assert!(hidden.iter().all(|value| (*value - expected).abs() < 1e-7));

        let scores = [1.0f32, 5.0, 5.0, -1.0, 2.0, 3.0];
        let mut ids = [u32::MAX; 2];
        table[0] = ids.as_mut_ptr().cast();
        table[1] = scores.as_ptr().cast_mut().cast();
        let mut argmax = inst(DevOp::ArgmaxF32);
        argmax.t[..2].copy_from_slice(&[0, 1]);
        argmax.i[..2].copy_from_slice(&[2, 3]);
        unsafe { kernel(argmax.op).unwrap()(&argmax, 0, 1, table.as_ptr(), &mut ctx) };
        assert_eq!(ids, [1, 2]);
    }
}
