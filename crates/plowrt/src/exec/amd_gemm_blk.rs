//! Native gfx942 block-scale FP8 projection through AITER's pre-shuffled assembly.
//!
//! The BF16 twin of this module is [`super::amd_gemm_lt`], and the shape of the two is deliberately
//! the same: a hash-pinned object, geometry validated at load, refused by name when anything does
//! not match. What differs is the numerics. GLM-5.3's checkpoint declares
//! `activation_scheme: dynamic` over a `[128,128]` `weight_scale_inv` grid, so W8A8 with a
//! per-token per-128-group activation scale IS this model's reference serving arithmetic — the
//! BF16 route plow ships today is *more* precise than the reference, not less.
//!
//! # Why assembly and not hipBLASLt
//!
//! hipBLASLt on gfx942 carries per-TENSOR FP8 scaling only; a `[128,128]` grid needs either CK's
//! `a8w8_blockscale_*` instances or these assembly objects. AITER's gfx942 tuning table selects CK
//! at every M=8192 shape; these objects measured 706-783 TF/s there, 84-94% of that table.
//!
//! # ABI
//!
//! Transcribed from `/workspace/aiter/csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`
//! (`KernelArgs` + the launch at the bottom) and checked against the object's own
//! `NT_AMDGPU_METADATA`: 256-byte kernarg, 256 threads, 64 KiB static LDS, no private segment.
//! Every argument occupies a 16-byte slot. `ldc` is in BYTES; `lda`/`ldb` are in elements.
//! `A` is `[M,K]` e4m3-fnuz row-major, `B` is `[N,K]` e4m3-fnuz pre-shuffled by
//! [`super::amd::shuffle_moe_weight_16x32`] (AITER's `shuffle_weight(layout=(16,16))`),
//! `a_scale` is `[K/128][M]` f32 (group-major — the transposed layout AITER's own test passes),
//! `b_scale` is `[N/128][K/128]` f32, and `bias` must be a real `[N]` f32 pointer even when zero.
//!
//! # Route
//!
//! `GemmBlkPf` (`PLOW_GLM_GEMM_BLK`) sits alone in its segment. With `i3=1` the segment first runs
//! `plow_gemm_blk_quant` (the MoE route's activation quant with K as a parameter), then the GEMM;
//! `i3=0` reuses the scratch the previous quantizing instruction wrote from the same `x`, which
//! [`routes`] checks in segment order. The bound weight and scale grid are the checkpoint's bytes
//! transformed once at bind ([`prepare_weight`], [`prepare_scales`]), so only this route may read
//! them ([`bound_weights`]).

use std::collections::HashMap;
use std::path::Path;

use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};

use super::amd_gemm_lt::segment_owners;
use super::device_api::EngineDevice;
use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::Module;
use crate::{Result, RuntimeError};

/// The six pre-shuffled tiles AITER ships for gfx942. All are `tile_n = 128`; the entry is
/// `(file, sha256, tile_m)`. `tile_n` is a constant of the family, not a per-object field:
/// `validate_inputs` in AITER's launcher refuses `N % 128 != 0` for every one of them.
const OBJECTS: [(&str, &str, u32); 6] = [
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_32x128.co",
        "3d1acfd1e5bf6f16816334d8aede37f93af0cbb3f0d6030dddb1deacf7e38d8f",
        32,
    ),
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_48x128.co",
        "f9a85f3cca7df1d2fea8b73f717a71674595b650bb4d9bafbb45b1298f62d841",
        48,
    ),
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_64x128.co",
        "b3b0814cdfc6be1cc838ba7e9065aa763dc2dcd72eb0c26afda31efab2884736",
        64,
    ),
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_80x128.co",
        "9f79c4151eab0d216595010507d671945ef8c1abc5e8271f026d21b155a1c5f1",
        80,
    ),
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_96x128.co",
        "6cda76fcdafd257f73d9cb9cbcf22b4730077b8d41c5bc9c6b28942799c9e5af",
        96,
    ),
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_128x128.co",
        "de905e406509c62f78d0760a36e762f7de89ded5d39fd3f98062eb58a8078b99",
        128,
    ),
];

/// Per rank, TP8: `q_a_proj`, `kv_a_latent`, `o_proj`, indexer `wq_b`. Each is a whole checkpoint
/// block-FP8 tensor or a block-row-aligned slice of one. `q_absorb` shares `wq_b`'s 4096x2048 but
/// is a prep product with no grid; the emitter, not this list, keeps it off the route.
const PINNED: [(u32, u32); 4] = [(2048, 6144), (512, 6144), (6144, 2048), (4096, 2048)];
const TILE_N: u32 = 128;
/// The checkpoint's `weight_block_size`, and the group length of the activation quant.
const BLOCK_K: u32 = 128;
const WORKGROUP: u32 = 256;
const LDS_BYTES: u32 = 65536;
const KERNARG_BYTES: u32 = 256;
/// File offset of `KERNARG_SIZE` in the kernel descriptor (`.kd` at 0x12c0, field at +8).
const KD_KERNARG_SIZE: usize = 0x12c8;
const ADAPTER: &str = "gemm_blk_adapter_gfx942.elf";
const ADAPTER_MARKER: &str = "plow_gemm_blk_abi_1";
const QUANT_ARGS_BYTES: u32 = 32;
/// The MoE prepare pass's launch shape: four workgroups per CU.
const QUANT_GRID: u32 = 304 * 4;

/// Kernel-name prefix; the suffix is `{tile_m}x128E`, Itanium-mangled as `aiter::<name>`.
fn symbol(tile_m: u32) -> String {
    let stem = format!("fp8gemm_bf16_blockscale_BpreShuffle_{tile_m}x128");
    format!("_ZN5aiter{}{stem}E", stem.len())
}

/// AITER's own heuristic, restricted to `splitK = 1`.
///
/// `get_heuristic_fp8_kernel` minimises the number of CU rounds
/// `ceil(tiles / num_cu)`, breaking ties on spare CUs and then on the tile's compute-to-memory
/// ratio `tile_m*tile_n/(tile_m+tile_n)` — which is monotone in `tile_m` at fixed `tile_n`, so the
/// tie-break always prefers the larger tile. `splitK > 1` accumulates atomically into `C` and would
/// oblige this route to zero the output first, which a native segment has no packet to do.
fn tile_choice(m: u32, n: u32, n_cu: u32) -> Option<usize> {
    if m == 0 || n == 0 || !n.is_multiple_of(TILE_N) || n_cu == 0 {
        return None;
    }
    let tiles_n = n / TILE_N;
    let mut best: Option<(u32, u32, usize)> = None;
    for (index, &(_, _, tile_m)) in OBJECTS.iter().enumerate() {
        let tiles = m.div_ceil(tile_m).checked_mul(tiles_n)?;
        let round = tiles.div_ceil(n_cu);
        let spare = round * n_cu - tiles;
        // Ordering is (fewest rounds, then most spare CUs, then largest tile_m).
        if best.is_none_or(|(r, s, _)| round < r || (round == r && spare >= s)) {
            best = Some((round, spare, index));
        }
    }
    best.map(|(_, _, index)| index)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    inst: DevInst64,
    rows: u32,
    index: usize,
    n_cu: u32,
}

impl Route {
    fn new(inst: DevInst64, n_cu: u32) -> Result<Self> {
        let index = tile_choice(inst.i[0], inst.i[1], n_cu).ok_or_else(|| {
            RuntimeError::Device("block-scale FP8 projection: no tile covers it".into())
        })?;
        Ok(Self {
            inst,
            rows: inst.i[0],
            index,
            n_cu,
        })
    }

    /// Narrow to the live rows of this chunk. The tile is re-chosen, because the row count is
    /// what picks it. A quantizing route and the routes reusing its scratch are rebased to the
    /// same rows, so the group-major scale stride (`rows`) agrees between them.
    pub fn rebase(&mut self, rows: u32) -> Result<()> {
        if rows == 0 || rows > self.inst.i[0] {
            return Err(RuntimeError::Device(
                "block-scale FP8 chunk exceeds row capacity".into(),
            ));
        }
        self.index = tile_choice(rows, self.inst.i[1], self.n_cu)
            .ok_or_else(|| RuntimeError::Device("block-scale FP8 rebase found no tile".into()))?;
        self.rows = rows;
        Ok(())
    }

    pub fn launches(self) -> usize {
        1 + usize::from(self.quantizes())
    }

    fn quantizes(self) -> bool {
        self.inst.i[3] == 1
    }

    fn grid(&self) -> [u32; 3] {
        [
            self.inst.i[1] / TILE_N,
            self.rows.div_ceil(OBJECTS[self.index].2),
            1,
        ]
    }
}

pub(super) fn routes(
    prog: &DevProg,
    tensors: &[DevTensor],
    segments: usize,
    n_cu: u32,
) -> Result<Vec<Option<Route>>> {
    let mut routes = vec![None; segments];
    let native = |i: &DevInst64| i.op == DevOp::GemmBlkPf as u16;
    if !prog.insts.iter().any(native) {
        return Ok(routes);
    }
    let owners = segment_owners(prog, segments, native)?;
    let mut order = Vec::new();
    for (ix, inst) in prog.insts.iter().enumerate() {
        if native(inst) {
            let seg = owners[ix].ok_or_else(|| {
                RuntimeError::Device(format!("block-scale FP8 instruction {ix} has no segment"))
            })?;
            order.push((seg, ix));
        }
    }
    order.sort_unstable();
    let mut quantized = None;
    for (seg, ix) in order {
        let inst = prog.insts[ix];
        let err = |s: &str| RuntimeError::Device(format!("block-scale FP8 instruction {ix}: {s}"));
        let [m, n, k, quantize] = [inst.i[0], inst.i[1], inst.i[2], inst.i[3]];
        if !(2048..=8192).contains(&prog.t)
            || m != prog.t
            || !PINNED.contains(&(n, k))
            || quantize > 1
            || inst.i[4..] != [0; 4]
            || inst.fj != [0; 3]
            || inst.t[7] != TENSOR_NONE16
        {
            return Err(err("requires a qualified block-scale FP8 projection"));
        }
        let t = &inst.t[..7];
        if t.contains(&TENSOR_NONE16) {
            return Err(err("operand is missing"));
        }
        if (1..t.len()).any(|a| t[..a].contains(&t[a])) {
            return Err(err("operands alias"));
        }
        let [m, n, k] = [u64::from(m), u64::from(n), u64::from(k)];
        let need = [
            m * n * 2,
            m * k * 2,
            n * k,
            n / 128 * (k / 128) * 4,
            m * k,
            k / 128 * m * 4,
            n * 4,
        ];
        for (&handle, bytes) in t.iter().zip(need) {
            if tensors.get(handle as usize).is_none_or(|d| d.bytes < bytes) {
                return Err(err("operand capacity is insufficient"));
            }
        }
        let key = (t[1], t[4], t[5], k);
        if quantize == 1 {
            quantized = Some(key);
        } else if quantized != Some(key) {
            return Err(err(
                "reuses a quantized activation no earlier instruction wrote",
            ));
        }
        routes[seg] = Some(Route::new(inst, n_cu)?);
    }
    Ok(routes)
}

/// What the bind loop must do to a tensor `GemmBlkPf` reads as its weight (`t2`) or scale grid
/// (`t3`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Bound {
    Weight { n: u32, k: u32 },
    Scales,
}

/// Tensor handles whose bound bytes this route transforms, across every program. Refused when
/// any other instruction names one of them: the pre-shuffled weight and doubled scales are
/// readable only by this route.
pub(super) fn bound_weights(progs: &[DevProg]) -> Result<HashMap<u16, Bound>> {
    let mut bound = HashMap::new();
    for inst in progs.iter().flat_map(|p| &p.insts) {
        if inst.op != DevOp::GemmBlkPf as u16 {
            continue;
        }
        for (handle, want) in [
            (
                inst.t[2],
                Bound::Weight {
                    n: inst.i[1],
                    k: inst.i[2],
                },
            ),
            (inst.t[3], Bound::Scales),
        ] {
            if *bound.entry(handle).or_insert(want) != want {
                return Err(RuntimeError::Device(
                    "block-scale FP8 weight bound with two geometries".into(),
                ));
            }
        }
    }
    for inst in progs.iter().flat_map(|p| &p.insts) {
        if inst.op != DevOp::GemmBlkPf as u16 && inst.t.iter().any(|h| bound.contains_key(h)) {
            return Err(RuntimeError::Device(format!(
                "{:?} reads a block-scale FP8 weight bound in the route's layout",
                DevOp::from_u16(inst.op)
            )));
        }
    }
    Ok(bound)
}

/// The bound layout of a `GemmBlkPf` weight: the checkpoint's `[N,K]` e4m3 bytes pre-shuffled for
/// the kernel, with OCP `-0` (FNUZ's only NaN) scrubbed to zero.
pub(super) fn prepare_weight(src: &[u8], n: u32, k: u32) -> Result<Vec<u8>> {
    let mut weight = super::amd::shuffle_moe_weight_16x32(src, n as usize, k as usize)?;
    for byte in &mut weight {
        if *byte == 0x80 {
            *byte = 0;
        }
    }
    Ok(weight)
}

/// The checkpoint's `[N/128][K/128]` f32 `weight_scale_inv`, doubled: on gfx942 the same e4m3
/// byte pattern denotes half its OCP value (bias 8, not 7).
pub(super) fn prepare_scales(src: &[u8]) -> Result<Vec<u8>> {
    if !src.len().is_multiple_of(4) {
        return Err(RuntimeError::Device(
            "block-scale FP8 scale grid is not f32".into(),
        ));
    }
    let mut out = Vec::with_capacity(src.len());
    for chunk in src.chunks_exact(4) {
        let scale = 2.0 * f32::from_le_bytes(chunk.try_into().unwrap());
        if !scale.is_finite() || scale <= 0.0 {
            return Err(RuntimeError::Device(
                "block-scale FP8 scale is not a finite positive value".into(),
            ));
        }
        out.extend_from_slice(&scale.to_le_bytes());
    }
    Ok(out)
}

/// The kernel's 256-byte argument block. Each field owns a 16-byte slot; the padding is the
/// object's own (`.name: pad` in its metadata), not alignment slack, so it is spelled out.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    c: u64,
    _p0: u64,
    a: u64,
    _p1: u64,
    b: u64,
    _p2: u64,
    a_scale: u64,
    _p3: u64,
    b_scale: u64,
    _p4: u64,
    bias: u64,
    _p5: u64,
    m: u32,
    _p6: [u32; 3],
    n: u32,
    _p7: [u32; 3],
    k: u32,
    _p8: [u32; 3],
    lda: u32,
    _p9: [u32; 3],
    ldb: u32,
    _p10: [u32; 3],
    /// BYTES per output row, not elements — AITER passes `stride(0) * sizeof(uint16_t)`.
    ldc: u32,
    _p11: [u32; 3],
    ks: u32,
    _p12: [u32; 3],
    scale_m: u32,
    _p13: [u32; 3],
    scale_n: u32,
    _p14: [u32; 3],
    scale_k: u32,
    _p15: [u32; 3],
}
const _: () = assert!(std::mem::size_of::<Args>() == KERNARG_BYTES as usize);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantArgs {
    q: u64,
    qs: u64,
    x: u64,
    rows: u32,
    k: u32,
}
const _: () = assert!(std::mem::size_of::<QuantArgs>() == QUANT_ARGS_BYTES as usize);

/// `[out, a, b, a_scale, b_scale, bias]` device addresses, in that order.
fn arguments(route: Route, tensors: [u64; 6]) -> Args {
    let [c, a, b, a_scale, b_scale, bias] = tensors;
    let (n, k) = (route.inst.i[1], route.inst.i[2]);
    Args {
        c,
        a,
        b,
        a_scale,
        b_scale,
        bias,
        m: route.rows,
        n,
        k,
        lda: k,
        ldb: k,
        ldc: n * 2,
        ks: 1,
        // `block_shape_m` is 1: one activation scale per token per K-group.
        scale_m: route.rows,
        scale_n: n.div_ceil(BLOCK_K),
        scale_k: k.div_ceil(BLOCK_K),
        ..Default::default()
    }
}

pub(super) struct GemmBlk {
    kernels: Vec<HsaKernel>,
    quant: HsaKernel,
}

impl GemmBlk {
    pub fn load(be: &HsaBackend, dir: &Path, modules: &mut Vec<Module>) -> Result<Self> {
        let mut kernels = Vec::with_capacity(OBJECTS.len());
        for &(file, hash, tile_m) in &OBJECTS {
            let path = dir.join(file);
            let mut image = std::fs::read(&path)
                .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
            if plow_asset::decode_objects::image_sha256(&image) != hash {
                return Err(RuntimeError::Device(format!(
                    "{file} does not match the qualified block-scale FP8 ABI"
                )));
            }
            // Same omission as the AITER MoE objects: HIP reads the kernarg size from metadata,
            // ROCr from the descriptor, and these descriptors leave it zero. All six place
            // `.kd` at 0x12c0.
            let field = &mut image[KD_KERNARG_SIZE..KD_KERNARG_SIZE + 4];
            if field != [0; 4] {
                return Err(RuntimeError::Device(format!(
                    "{file} descriptor differs from the qualified ABI"
                )));
            }
            field.copy_from_slice(&KERNARG_BYTES.to_le_bytes());
            let module = EngineDevice::module_load(be, &image)?;
            let kernel = EngineDevice::get_function(be, &module, &symbol(tile_m))?;
            if kernel.kernarg_size() != KERNARG_BYTES
                || kernel.private_segment_size() != 0
                || HsaBackend::kernel_lds_bytes(&kernel) != LDS_BYTES
            {
                return Err(RuntimeError::Device(format!(
                    "{file} resource ABI mismatch: kernarg={} private={} lds={}",
                    kernel.kernarg_size(),
                    kernel.private_segment_size(),
                    HsaBackend::kernel_lds_bytes(&kernel),
                )));
            }
            modules.push(module);
            kernels.push(kernel);
        }
        let path = dir.join(ADAPTER);
        let image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        if !super::amd::elf_symbol_names(&image).contains(&ADAPTER_MARKER) {
            return Err(RuntimeError::Device(
                "block-scale FP8 adapter lacks ABI marker".into(),
            ));
        }
        let module = EngineDevice::module_load(be, &image)?;
        let quant = EngineDevice::get_function(be, &module, "plow_gemm_blk_quant")?;
        if ![QUANT_ARGS_BYTES, QUANT_ARGS_BYTES + 256].contains(&quant.kernarg_size())
            || quant.private_segment_size() != 0
        {
            return Err(RuntimeError::Device(
                "block-scale FP8 adapter resource ABI mismatch".into(),
            ));
        }
        modules.push(module);
        tracing::info!(tiles = OBJECTS.len(), "block-scale FP8 projections loaded");
        Ok(Self { kernels, quant })
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, tensor_table: &[u8]) -> Result<()> {
        let addr = |slot: usize| {
            let at = usize::from(route.inst.t[slot]) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        };
        if route.quantizes() {
            let args = QuantArgs {
                q: addr(4),
                qs: addr(5),
                x: addr(1),
                rows: route.rows,
                k: route.inst.i[2],
            };
            be.launch(
                self.quant,
                QUANT_GRID,
                WORKGROUP,
                0,
                bytemuck::bytes_of(&args),
            )?;
        }
        let args = arguments(
            route,
            [addr(0), addr(4), addr(2), addr(5), addr(3), addr(6)],
        );
        be.launch_3d(
            self.kernels[route.index],
            route.grid(),
            WORKGROUP as u16,
            bytemuck::bytes_of(&args),
        )
    }
}

/// gfx942's FP8 is e4m3 **fnuz**: bias 8, no infinities, `0x80` is the only NaN, max 240.
#[cfg(test)]
fn fnuz_decode(b: u8) -> f64 {
    if b == 0 || b == 0x80 {
        return 0.0;
    }
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((b >> 3) & 0xf);
    let man = f64::from(b & 7);
    if exp == 0 {
        sign * (man / 8.0) * 2f64.powi(-7)
    } else {
        sign * (1.0 + man / 8.0) * 2f64.powi(exp - 8)
    }
}

/// The checkpoint's e4m3 is OCP: bias 7, max 448.
#[cfg(test)]
fn ocp_decode(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((b >> 3) & 0xf);
    let man = f64::from(b & 7);
    if exp == 0 {
        sign * (man / 8.0) * 2f64.powi(-6)
    } else {
        sign * (1.0 + man / 8.0) * 2f64.powi(exp - 7)
    }
}

#[cfg(test)]
fn fnuz_encode(x: f32) -> u8 {
    if !x.is_finite() || x == 0.0 {
        return 0;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= 240.0 {
        return sign | 0x7f;
    }
    let bits = a.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let man = bits & 0x7f_ffff;
    if exp + 8 >= 1 {
        let (mut m, rem) = (man >> 20, man & 0xf_ffff);
        let mut code_exp = (exp + 8) as u32;
        if rem > 0x8_0000 || (rem == 0x8_0000 && m & 1 == 1) {
            m += 1;
        }
        if m == 8 {
            m = 0;
            code_exp += 1;
        }
        if code_exp > 15 {
            return sign | 0x7f;
        }
        sign | ((code_exp as u8) << 3) | (m as u8)
    } else {
        // Subnormal: value = m/8 * 2^-7. m == 8 has rounded up into the smallest normal.
        let m = (f64::from(a) * 1024.0).round() as u32;
        if m == 0 {
            0
        } else {
            sign | (m.min(8) as u8)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::StreamEnt;

    fn inst(n: u32, k: u32, quantize: u32, x: u16) -> DevInst64 {
        DevInst64 {
            op: DevOp::GemmBlkPf as u16,
            t: [0, x, 2, 3, 4, 5, 6, TENSOR_NONE16],
            i: [8192, n, k, quantize, 0, 0, 0, 0],
            ..Default::default()
        }
    }

    fn fixture(insts: Vec<DevInst64>) -> (DevProg, Vec<DevTensor>) {
        let stream = (0..insts.len())
            .map(|ix| StreamEnt {
                inst: ix as u32,
                seg: ix as u16,
                ..Default::default()
            })
            .collect();
        let prog = DevProg {
            t: 8192,
            role: packet::devbuild::ProgramRole::PrefillBucket { rows: 8192 },
            n_counter: 0,
            insts,
            stream,
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_domains: 0,
        };
        let tensors = (0..8)
            .map(|i| DevTensor {
                name: i.to_string(),
                bytes: 1 << 30,
                init: None,
            })
            .collect();
        (prog, tensors)
    }

    #[test]
    fn blk_routes_pin_the_checkpoint_shapes_and_refuse_the_rest() {
        for (n, k) in PINNED {
            let (p, t) = fixture(vec![inst(n, k, 1, 1)]);
            let route = routes(&p, &t, 1, 304).unwrap()[0].unwrap();
            assert_eq!(route.launches(), 2);
        }
        // k_rope / indexer weights_proj are narrower than a tile, q_rope is a non-block-aligned
        // slice, the shared expert belongs to the MoE fold, indexer wk is out of scope.
        for (n, k) in [
            (64, 6144),
            (32, 6144),
            (512, 2048),
            (256, 6144),
            (6144, 256),
            (128, 6144),
        ] {
            let (p, t) = fixture(vec![inst(n, k, 1, 1)]);
            assert!(routes(&p, &t, 1, 304).is_err(), "{n}x{k}");
        }
        for bad in 0..9 {
            let (mut p, mut t) = fixture(vec![inst(2048, 6144, 1, 1)]);
            match bad {
                0 => p.t = 1024,
                1 => p.insts[0].i[0] = 4096,
                2 => p.insts[0].i[3] = 2,
                3 => p.insts[0].i[4] = 1,
                4 => p.insts[0].t[6] = TENSOR_NONE16,
                5 => p.insts[0].t[4] = 1,
                6 => t[5].bytes = 16,
                7 => p.insts[0].i[3] = 0,
                _ => p.stream[0].wait_len = 1,
            }
            assert!(routes(&p, &t, 1, 304).is_err(), "case {bad}");
        }
    }

    #[test]
    fn blk_reuse_must_follow_its_own_quantizer() {
        let q_a = inst(2048, 6144, 1, 1);
        let kv_a = inst(512, 6144, 0, 1);
        let (p, t) = fixture(vec![q_a, kv_a]);
        let r = routes(&p, &t, 2, 304).unwrap();
        assert_eq!(r[0].unwrap().launches(), 2);
        assert_eq!(r[1].unwrap().launches(), 1);
        let (p, t) = fixture(vec![kv_a, q_a]);
        assert!(routes(&p, &t, 2, 304).is_err());
        // A different activation in between overwrites the shared scratch.
        let (p, t) = fixture(vec![q_a, inst(6144, 2048, 1, 7), kv_a]);
        assert!(routes(&p, &t, 3, 304).is_err());
    }

    #[test]
    fn blk_rebase_tracks_live_rows_and_reselects_the_tile() {
        let (p, t) = fixture(vec![inst(6144, 2048, 1, 1)]);
        let mut route = routes(&p, &t, 1, 304).unwrap()[0].unwrap();
        // 8192 rows x 48 N-tiles: the 128-row tile is 3072 groups = 11 rounds of 304 CUs, the
        // 96-row tile 4128 = 14. Largest tile wins outright.
        assert_eq!(OBJECTS[route.index].2, 128);
        assert_eq!(route.grid(), [48, 64, 1]);
        route.rebase(464).unwrap();
        assert_eq!(route.rows, 464);
        assert_eq!(
            route.grid(),
            [48, 464u32.div_ceil(OBJECTS[route.index].2), 1]
        );
        assert!(route.rebase(0).is_err());
        assert!(route.rebase(8193).is_err());
    }

    #[test]
    fn blk_arguments_carry_the_aiter_abi() {
        let (p, t) = fixture(vec![inst(6144, 2048, 1, 1)]);
        let mut route = routes(&p, &t, 1, 304).unwrap()[0].unwrap();
        route.rebase(4096).unwrap();
        let args = arguments(route, [1, 2, 3, 4, 5, 6]);
        let raw = bytemuck::bytes_of(&args);
        // Pointer slots are 16 bytes apart; the scalars start at 96 and are 16 bytes apart too.
        for (slot, want) in [1u64, 2, 3, 4, 5, 6].into_iter().enumerate() {
            let at = slot * 16;
            assert_eq!(
                u64::from_le_bytes(raw[at..at + 8].try_into().unwrap()),
                want
            );
        }
        for (slot, want) in [4096u32, 6144, 2048, 2048, 2048, 6144 * 2, 1, 4096, 48, 16]
            .into_iter()
            .enumerate()
        {
            let at = 96 + slot * 16;
            assert_eq!(
                u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()),
                want,
                "scalar slot {slot}"
            );
        }
    }

    #[test]
    fn blk_bound_weights_are_shuffled_scrubbed_doubled_and_exclusive() {
        let src: Vec<u8> = (0..256 * 128).map(|i| (i % 251) as u8).collect();
        let mut expect = super::super::amd::shuffle_moe_weight_16x32(&src, 256, 128).unwrap();
        for b in &mut expect {
            if *b == 0x80 {
                *b = 0;
            }
        }
        assert_eq!(prepare_weight(&src, 256, 128).unwrap(), expect);
        assert!(expect.iter().all(|&b| b != 0x80));
        let scales = prepare_scales(bytemuck::cast_slice(&[0.25f32, 3.0])).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, f32>(&scales), &[0.5, 6.0]);
        assert!(prepare_scales(bytemuck::cast_slice(&[f32::NAN])).is_err());
        assert!(prepare_scales(bytemuck::cast_slice(&[0.0f32])).is_err());

        let (p, _) = fixture(vec![inst(2048, 6144, 1, 1)]);
        let bound = bound_weights(std::slice::from_ref(&p)).unwrap();
        assert_eq!(bound[&2], Bound::Weight { n: 2048, k: 6144 });
        assert_eq!(bound[&3], Bound::Scales);
        let (mut shared, _) = fixture(vec![inst(2048, 6144, 1, 1)]);
        shared.insts.push(DevInst64 {
            op: DevOp::GemvFp8Blk as u16,
            t: [
                9,
                1,
                2,
                TENSOR_NONE16,
                3,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
            ],
            ..Default::default()
        });
        assert!(bound_weights(&[shared]).is_err());
    }

    #[test]
    fn blk_symbols_match_the_object_names() {
        assert_eq!(
            symbol(128),
            "_ZN5aiter43fp8gemm_bf16_blockscale_BpreShuffle_128x128E"
        );
        assert_eq!(
            symbol(32),
            "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E"
        );
    }

    #[test]
    fn blk_fnuz_codec_round_trips_every_representable_byte() {
        for b in 0..=255u8 {
            if b == 0x80 {
                continue;
            }
            let v = fnuz_decode(b);
            assert_eq!(fnuz_encode(v as f32), b, "byte {b:#04x}");
            // The same byte is half its OCP value, which is what `prepare_scales` compensates.
            if b & 0x7f != 0x7f {
                assert_eq!(ocp_decode(b), 2.0 * v, "byte {b:#04x}");
            }
        }
        assert_eq!(fnuz_encode(1000.0), 0x7f);
        assert_eq!(fnuz_decode(0x7f), 240.0);
    }

    #[cfg(feature = "hsa")]
    fn bf16(x: f32) -> u16 {
        let bits = x.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    #[cfg(feature = "hsa")]
    fn unbf16(x: u16) -> f64 {
        f64::from(f32::from_bits(u32::from(x) << 16))
    }

    /// One deterministic pseudo-Gaussian stream, so repeated runs see identical inputs.
    #[cfg(feature = "hsa")]
    fn gaussian(seed: &mut u64) -> f32 {
        let mut sum = 0i64;
        for _ in 0..4 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            sum += ((*seed >> 33) & 0xffff) as i64 - 0x8000;
        }
        sum as f32 / 65536.0
    }

    /// Device buffers for one projection, and the tensor table the route reads them through.
    #[cfg(feature = "hsa")]
    struct Bench {
        mem: Vec<crate::device::DeviceMem>,
        table: Vec<u8>,
    }

    #[cfg(feature = "hsa")]
    impl Bench {
        /// `x` bf16 `[m,k]`, `weight` bound layout, `scales` bound f32 grid.
        fn new(
            be: &HsaBackend,
            m: usize,
            n: usize,
            k: usize,
            x: &[u16],
            weight: &[u8],
            scales: &[u8],
        ) -> Self {
            let upload = |bytes: &[u8]| {
                let mem = EngineDevice::alloc(be, bytes.len() as u64).unwrap();
                EngineDevice::upload(be, &mem, 0, bytes).unwrap();
                mem
            };
            let mem = vec![
                EngineDevice::alloc(be, (m * n * 2) as u64).unwrap(),
                upload(bytemuck::cast_slice(x)),
                upload(weight),
                upload(scales),
                EngineDevice::alloc(be, (m * k) as u64).unwrap(),
                EngineDevice::alloc(be, (k / 128 * m * 4) as u64).unwrap(),
                upload(&vec![0u8; n * 4]),
            ];
            let table = mem.iter().flat_map(|d| d.base.to_le_bytes()).collect();
            Self { mem, table }
        }

        fn download(&self, be: &HsaBackend, slot: usize, bytes: usize) -> Vec<u8> {
            let mut out = vec![0u8; bytes];
            EngineDevice::download(be, &self.mem[slot], 0, &mut out).unwrap();
            out
        }
    }

    #[cfg(feature = "hsa")]
    fn device_inst(m: u32, n: u32, k: u32, quantize: u32) -> DevInst64 {
        DevInst64 {
            op: DevOp::GemmBlkPf as u16,
            t: [0, 1, 2, 3, 4, 5, 6, TENSOR_NONE16],
            i: [m, n, k, quantize, 0, 0, 0, 0],
            ..Default::default()
        }
    }

    #[cfg(feature = "hsa")]
    fn time_us(be: &HsaBackend, blk: &GemmBlk, route: Route, table: &[u8]) -> f64 {
        for _ in 0..3 {
            blk.enqueue(be, route, table).unwrap();
        }
        be.synchronize().unwrap();
        let iters = 20;
        let started = std::time::Instant::now();
        for _ in 0..iters {
            blk.enqueue(be, route, table).unwrap();
        }
        be.synchronize().unwrap();
        started.elapsed().as_secs_f64() * 1e6 / f64::from(iters)
    }

    /// Known answer on synthetic data through the whole route: the adapter's activation quant
    /// against a host transcription of it, the GEMM against FP64 on the operands it actually read,
    /// and the W8A8 floor against FP64 on the BF16 operands. Times quant+GEMM and GEMM alone, so
    /// the activation quant's own cost is measured rather than estimated.
    #[test]
    #[cfg(feature = "hsa")]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR (scripts/build_gemm_blk.sh)"]
    fn blk_matches_fp64_and_reports_throughput() {
        let dir = std::env::var("PLOW_TEST_AITER_DIR").expect("PLOW_TEST_AITER_DIR");
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let blk = GemmBlk::load(&be, Path::new(&dir), &mut modules).unwrap();
        let m = 8192usize;
        let probe = [0usize, 1, 1023, 4096, 8191];
        for (label, n, k) in [
            ("q_a_proj", 2048usize, 6144usize),
            ("kv_a_latent", 512, 6144),
            ("o_proj", 6144, 2048),
            ("indexer_wq_b", 4096, 2048),
        ] {
            let mut seed = 0x5eed_0000_0000_0001 ^ (n as u64) << 20 ^ k as u64;
            let x: Vec<u16> = (0..m * k)
                .map(|i| bf16(gaussian(&mut seed) * (0.5 + (i / k % 7) as f32 * 0.3)))
                .collect();
            let w: Vec<u16> = (0..n * k)
                .map(|_| bf16(gaussian(&mut seed) * 0.05))
                .collect();
            let gk = k / 128;
            let (mut w_q, mut w_s) = (vec![0u8; n * k], vec![0f32; (n / 128) * gk]);
            for bn in 0..n / 128 {
                for bg in 0..gk {
                    let at = |r: usize, c: usize| (bn * 128 + r) * k + bg * 128 + c;
                    let mut amax = 1e-10f32;
                    for r in 0..128 {
                        for c in 0..128 {
                            amax = amax.max(unbf16(w[at(r, c)]).abs() as f32);
                        }
                    }
                    let s = amax / 240.0;
                    w_s[bn * gk + bg] = s;
                    for r in 0..128 {
                        for c in 0..128 {
                            w_q[at(r, c)] = fnuz_encode(unbf16(w[at(r, c)]) as f32 / s);
                        }
                    }
                }
            }
            let shuffled = super::super::amd::shuffle_moe_weight_16x32(&w_q, n, k).unwrap();
            let bench = Bench::new(&be, m, n, k, &x, &shuffled, bytemuck::cast_slice(&w_s));
            let quant = Route::new(device_inst(m as u32, n as u32, k as u32, 1), 304).unwrap();
            blk.enqueue(&be, quant, &bench.table).unwrap();
            be.synchronize().unwrap();
            let c = bench.download(&be, 0, m * n * 2);
            let c: &[u16] = bytemuck::cast_slice(&c);
            let xq = bench.download(&be, 4, m * k);
            let xs = bench.download(&be, 5, gk * m * 4);
            let xs: &[f32] = bytemuck::cast_slice(&xs);

            // The adapter against a host transcription of the same arithmetic.
            let (mut scale_miss, mut byte_miss) = (0usize, 0usize);
            for row in 0..m {
                for g in 0..gk {
                    let span = &x[row * k + g * 128..row * k + g * 128 + 128];
                    let amax = span
                        .iter()
                        .fold(1e-10f32, |a, &v| a.max(unbf16(v).abs() as f32));
                    let s = amax * (1.0 / 240.0);
                    scale_miss += usize::from(xs[g * m + row] != s);
                    let inverse = 1.0 / s;
                    for (j, &v) in span.iter().enumerate() {
                        let want = fnuz_encode(unbf16(v) as f32 * inverse);
                        byte_miss += usize::from(xq[row * k + g * 128 + j] != want);
                    }
                }
            }

            let (mut e_q, mut e_b, mut norm) = (0f64, 0f64, 0f64);
            for &row in &probe {
                for col in 0..n {
                    let (mut acc_q, mut acc_b) = (0f64, 0f64);
                    for g in 0..gk {
                        let (mut dot, wsc) = (0f64, f64::from(w_s[(col / 128) * gk + g]));
                        for j in 0..128 {
                            let (at_a, at_w) = (row * k + g * 128 + j, col * k + g * 128 + j);
                            dot += fnuz_decode(xq[at_a]) * fnuz_decode(w_q[at_w]);
                            acc_b += unbf16(x[at_a]) * unbf16(w[at_w]);
                        }
                        acc_q += dot * f64::from(xs[g * m + row]) * wsc;
                    }
                    let got = unbf16(c[row * n + col]);
                    e_q += (got - acc_q).powi(2);
                    e_b += (got - acc_b).powi(2);
                    norm += acc_b * acc_b;
                }
            }
            let (rel_q, rel_b) = ((e_q / norm).sqrt(), (e_b / norm).sqrt());
            let with_quant = time_us(&be, &blk, quant, &bench.table);
            let reuse = Route::new(device_inst(m as u32, n as u32, k as u32, 0), 304).unwrap();
            let gemm_only = time_us(&be, &blk, reuse, &bench.table);
            println!(
                "{label:14} {m}x{n}x{k} tile_m={:3} gemm {gemm_only:7.1} us {:6.1} TF/s, \
                 quant+gemm {with_quant:7.1} us (quant {:5.1} us); rel-L2 vs FP64(fp8) {rel_q:.3e} \
                 vs FP64(bf16) {rel_b:.3e}; quant scale misses {scale_miss}, byte misses {byte_miss}",
                OBJECTS[quant.index].2,
                2.0 * (m * n * k) as f64 / (gemm_only * 1e6),
                with_quant - gemm_only,
            );
            assert_eq!(
                scale_miss, 0,
                "{label}: adapter scales differ from the transcription"
            );
            assert!(
                (byte_miss as f64) < 1e-4 * (m * k) as f64,
                "{label}: adapter quantized {byte_miss} bytes differently"
            );
            assert!(
                rel_q < 5e-3,
                "{label}: kernel disagrees with its own operands: {rel_q:.3e}"
            );
            assert!(
                rel_b < 8e-2,
                "{label}: W8A8 floor {rel_b:.3e} is worse than the MoE route's"
            );
        }
    }

    /// Per-row numerics on CAPTURED activations: rank 0's layer-77 `act.xn` / `act.qlat` /
    /// `act.oat` from one 8192-row chunk of real text, against the checkpoint's own FP8 weights
    /// run through the bind-time transform ([`prepare_weight`], [`prepare_scales`]).
    ///
    /// `PLOW_TEST_BLK_CAPTURE` holds `xn.prefill.bin`, `qlat.prefill.bin`, `oat.prefill.bin` (bf16
    /// `[8192][K]`) and, per projection, `w_<label>.bin` (checkpoint e4m3 `[N][K]`, rank 0's
    /// shard) and `s_<label>.bin` (its f32 `weight_scale_inv` grid, as stored). Two references
    /// per row: FP64 over the dequantised weight (the ideal), and FP64 over that weight rounded
    /// to bf16 (what the prep uploads for the BF16 path today).
    #[test]
    #[cfg(feature = "hsa")]
    #[ignore = "requires a gfx942 GPU lease, PLOW_TEST_AITER_DIR and PLOW_TEST_BLK_CAPTURE"]
    fn blk_captured_activations_match_the_bf16_path() {
        let dir = std::env::var("PLOW_TEST_AITER_DIR").expect("PLOW_TEST_AITER_DIR");
        let cap = std::path::PathBuf::from(
            std::env::var("PLOW_TEST_BLK_CAPTURE").expect("PLOW_TEST_BLK_CAPTURE"),
        );
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let blk = GemmBlk::load(&be, Path::new(&dir), &mut modules).unwrap();
        let m = 8192usize;
        let read =
            |name: &str| std::fs::read(cap.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        for (label, act, n, k) in [
            ("q_a_proj", "xn", 2048usize, 6144usize),
            ("kv_a_latent", "xn", 512, 6144),
            ("o_proj", "oat", 6144, 2048),
            ("indexer_wq_b", "qlat", 4096, 2048),
        ] {
            let x_raw = read(&format!("{act}.prefill.bin"));
            assert_eq!(x_raw.len(), m * k * 2, "{act}");
            let x: &[u16] = bytemuck::cast_slice(&x_raw);
            let w = read(&format!("w_{label}.bin"));
            let s_raw = read(&format!("s_{label}.bin"));
            assert_eq!(w.len(), n * k, "{label} weight");
            assert_eq!(s_raw.len(), n / 128 * (k / 128) * 4, "{label} scales");
            let s: &[f32] = bytemuck::cast_slice(&s_raw);
            let bench = Bench::new(
                &be,
                m,
                n,
                k,
                x,
                &prepare_weight(&w, n as u32, k as u32).unwrap(),
                &prepare_scales(&s_raw).unwrap(),
            );
            let route = Route::new(device_inst(m as u32, n as u32, k as u32, 1), 304).unwrap();
            blk.enqueue(&be, route, &bench.table).unwrap();
            be.synchronize().unwrap();
            let c = bench.download(&be, 0, m * n * 2);
            let c: &[u16] = bytemuck::cast_slice(&c);
            assert!(
                c.iter().all(|&v| unbf16(v).is_finite()),
                "{label}: non-finite output"
            );

            let gk = k / 128;
            let ideal: Vec<f64> = (0..n * k)
                .map(|i| ocp_decode(w[i]) * f64::from(s[(i / k / 128) * gk + (i % k) / 128]))
                .collect();
            let bf16_path: Vec<f64> = ideal.iter().map(|&v| unbf16(bf16(v as f32))).collect();
            let rows: Vec<usize> = (0..64).map(|i| i * (m - 1) / 63).collect();
            let (mut per_ideal, mut per_bf16, mut floor) = (Vec::new(), Vec::new(), Vec::new());
            let (mut e_i, mut e_b, mut n_i, mut n_b) = (0f64, 0f64, 0f64, 0f64);
            for &row in &rows {
                let xr = &x[row * k..row * k + k];
                let (mut ei, mut eb, mut ni, mut nb, mut ef) = (0f64, 0f64, 0f64, 0f64, 0f64);
                for col in 0..n {
                    let (wi, wb) = (
                        &ideal[col * k..col * k + k],
                        &bf16_path[col * k..col * k + k],
                    );
                    let (mut ri, mut rb) = (0f64, 0f64);
                    for j in 0..k {
                        let a = unbf16(xr[j]);
                        ri += a * wi[j];
                        rb += a * wb[j];
                    }
                    let got = unbf16(c[row * n + col]);
                    ei += (got - ri).powi(2);
                    eb += (got - rb).powi(2);
                    ef += (rb - ri).powi(2);
                    ni += ri * ri;
                    nb += rb * rb;
                }
                per_ideal.push((ei / ni).sqrt());
                per_bf16.push((eb / nb).sqrt());
                floor.push((ef / ni).sqrt());
                (e_i, e_b, n_i, n_b) = (e_i + ei, e_b + eb, n_i + ni, n_b + nb);
            }
            let stats = |v: &mut Vec<f64>| {
                v.sort_by(f64::total_cmp);
                (v[v.len() / 2], v[v.len() * 9 / 10], v[v.len() - 1])
            };
            let (i50, i90, imax) = stats(&mut per_ideal);
            let (b50, b90, bmax) = stats(&mut per_bf16);
            let (f50, _, fmax) = stats(&mut floor);
            println!(
                "{label:14} {} rows: rel-L2 vs BF16 path {:.3e} (row median {b50:.3e} p90 {b90:.3e} \
                 max {bmax:.3e}); vs FP64 ideal {:.3e} (median {i50:.3e} p90 {i90:.3e} max {imax:.3e}); \
                 BF16 path vs ideal median {f50:.3e} max {fmax:.3e}",
                rows.len(),
                (e_b / n_b).sqrt(),
                (e_i / n_i).sqrt(),
            );
            assert!(
                (e_i / n_i).sqrt() < 8e-2,
                "{label}: W8A8 floor on captured rows"
            );
        }
    }
}
