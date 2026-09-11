use std::collections::BTreeSet;
use std::path::Path;

use packet::dev::{DevInst64, DevOp, SE_XCTR, TENSOR_NONE16};

use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::{DeviceMem, Module};
use crate::exec::device_api::EngineDevice;
use crate::{Result, RuntimeError};

const OBJECT_HASH: &str = "cd8fa62e18abada15beeeedd49357bcc1e9e2eee7353d038533f30cac93c3607";
const OBJECT: &str = "mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co";

pub(super) fn union_handle(inst: &DevInst64) -> Option<u32> {
    if inst.op == DevOp::FlashMlaPrefill as u16 && inst.t[7] != TENSOR_NONE16 {
        Some(u32::from(inst.t[7]))
    } else if inst.op == DevOp::FlashMlaPrefillFp8 as u16 {
        inst.fj[1].checked_sub(1)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    inst: DevInst64,
    index: u16,
    scale: Option<u16>,
    rows: u32,
    kv_len: u32,
    pub active: bool,
}

impl Route {
    pub fn rebase(&mut self, rows: u32, prior: u32) -> Result<()> {
        if rows > self.inst.i[4] || prior.checked_add(rows).is_none_or(|n| n > self.inst.i[2]) {
            return Err(RuntimeError::Device(
                "sparse MLA chunk exceeds its query/KV capacity".into(),
            ));
        }
        self.rows = rows;
        self.kv_len = prior + rows;
        // Fixed-width CSR is valid only when every row has all 2048 causal keys.
        self.active = rows != 0 && prior >= 2047;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::StreamEnt;

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR"]
    fn sparse_mla_hsa_dispatch() {
        sparse_mla_hsa(true);
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and a default PLOW_TEST_AITER_DIR adapter"]
    fn sparse_mla_legacy_hsa_dispatch() {
        sparse_mla_hsa(false);
    }

    fn sparse_mla_hsa(single_supported: bool) {
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        const ROWS: u32 = 513;
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        for (fp8, single) in [(false, false), (true, false), (true, true)] {
            if single && !single_supported {
                continue;
            }
            let mut kernel =
                SparseMla::load(&be, Path::new(&dir), ROWS, 4096, fp8, &mut modules).unwrap();
            assert_eq!(kernel.pack_fp8_single.is_some(), fp8 && single_supported);
            if !single {
                kernel.pack_fp8_single = None;
            }
            let sizes = [
                u64::from(ROWS) * 8 * 512 * 4 + 1024,
                u64::from(ROWS) * 8 * 2 * 4 + 1024,
                u64::from(ROWS) * 8 * 512 * 2,
                u64::from(ROWS) * 8 * 64 * 2,
                2 * 4096 * 512 * 2,
                2 * 4096 * 64 * 2,
                4,
                4,
                u64::from(ROWS) * 2048 * 4,
                2 * 4096 * 4,
            ];
            let mut buffers = Vec::new();
            for bytes in sizes {
                let buf = EngineDevice::alloc(&be, bytes).unwrap();
                EngineDevice::upload(&be, &buf, 0, &vec![0; bytes as usize]).unwrap();
                buffers.push(buf);
            }
            let mut ck = vec![0x3f80u16; 2 * 4096 * 512];
            ck[4096 * 512..].fill(0x4000);
            if fp8 {
                let bytes: Vec<u8> = (0..2 * 4096 * 512)
                    .map(|i| if i % 2 == 0 { 0x38 } else { 0x40 })
                    .collect();
                EngineDevice::upload(&be, &buffers[4], 0, &bytes).unwrap();
                let scales: Vec<f32> = (0..2 * 4096)
                    .map(|i| (1 + i / 4096) as f32 * (1.0 + (i % 128) as f32 / 128.0))
                    .collect();
                EngineDevice::upload(&be, &buffers[9], 0, bytemuck::cast_slice(&scales)).unwrap();
            } else {
                EngineDevice::upload(&be, &buffers[4], 0, bytemuck::cast_slice(&ck)).unwrap();
            }
            let idx: Vec<u32> = (0..ROWS * 2048).map(|i| i % 2048).collect();
            EngineDevice::upload(&be, &buffers[8], 0, bytemuck::cast_slice(&idx)).unwrap();
            let mut route = Route {
                inst: DevInst64 {
                    t: [0, 1, 2, 3, 4, 5, 6, 7],
                    i: [1, 8, 4096, 0, ROWS, u32::MAX, 0, 0],
                    fj: [0.0625f32.to_bits(), 0, 0],
                    ..Default::default()
                },
                index: 8,
                scale: fp8.then_some(9),
                rows: ROWS,
                kv_len: 0,
                active: false,
            };
            for slot in 0..2u64 {
                let mut table: Vec<u64> = buffers.iter().map(|m| m.base).collect();
                table[0] += 512;
                table[1] += 512;
                table[4] += slot * 4096 * 512 * if fp8 { 1 } else { 2 };
                table[9] += slot * 4096 * 4;
                table[5] += slot * 4096 * 64 * 2;
                for rows in [1, 129, 511, 512, ROWS] {
                    for b in [0, 1] {
                        let poison = vec![f32::NAN; buffers[b].len as usize / 4];
                        EngineDevice::upload(&be, &buffers[b], 0, bytemuck::cast_slice(&poison))
                            .unwrap();
                    }
                    route.rebase(rows, 2048).unwrap();
                    kernel
                        .enqueue(&be, route, bytemuck::cast_slice(&table))
                        .unwrap();
                    be.synchronize().unwrap();
                    let mut out = vec![0f32; rows as usize * 8 * 512];
                    EngineDevice::download(
                        &be,
                        &buffers[0],
                        512,
                        bytemuck::cast_slice_mut(&mut out),
                    )
                    .unwrap();
                    assert!(out.iter().enumerate().all(|(i, &x)| {
                        let factor = if fp8 {
                            1.49609375 * (1 + i % 2) as f32
                        } else {
                            1.0
                        };
                        (x - (slot + 1) as f32 * factor).abs() < 1e-5
                    }));
                    if fp8 {
                        let mut packed = vec![0u16; (2048 + rows) as usize * 576];
                        EngineDevice::download(
                            &be,
                            &kernel._scratch,
                            kernel.kv - kernel._scratch.base,
                            bytemuck::cast_slice_mut(&mut packed),
                        )
                        .unwrap();
                        for (i, &bits) in packed.iter().enumerate() {
                            let (row, col) = (i / 576, i % 576);
                            let expected = if col < 512 {
                                (slot + 1) as f32
                                    * (1.0 + (row % 128) as f32 / 128.0)
                                    * (1 + col % 2) as f32
                            } else {
                                0.0
                            };
                            assert_eq!(bits, (expected.to_bits() >> 16) as u16);
                        }
                    }
                    let mut ml = vec![0f32; rows as usize * 8 * 2];
                    EngineDevice::download(
                        &be,
                        &buffers[1],
                        512,
                        bytemuck::cast_slice_mut(&mut ml),
                    )
                    .unwrap();
                    assert!(ml.chunks_exact(2).all(|x| x == [0.0, 1.0]));
                    for (b, width) in [(0, 512), (1, 2)] {
                        for offset in [0, 512 + u64::from(rows) * 8 * width * 4] {
                            let mut guard = vec![0f32; 128];
                            EngineDevice::download(
                                &be,
                                &buffers[b],
                                offset,
                                bytemuck::cast_slice_mut(&mut guard),
                            )
                            .unwrap();
                            assert!(
                                guard.iter().all(|x| x.is_nan()),
                                "rows={rows} fp8={fp8} buffer={b}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn fixture() -> (DevProg, Vec<DevTensor>) {
        let union = DevInst64 {
            op: DevOp::IndexUnionPf as u16,
            t: [
                7,
                9,
                8,
                6,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
            ],
            i: [8192, 2048, 81920, 16384, 8, 0, 0, 0],
            ..Default::default()
        };
        let flash = DevInst64 {
            op: DevOp::FlashMlaPrefill as u16,
            t: [0, 1, 2, 3, 4, 5, 6, 7],
            i: [1, 8, 81920, 0, 8192, u32::MAX, 16384, 4],
            fj: [0.0625f32.to_bits(), 0, 0],
            ..Default::default()
        };
        let prog = DevProg {
            t: 8192,
            packed_prefill_only: false,
            n_counter: 0,
            insts: vec![union, flash],
            stream: vec![StreamEnt {
                inst: 1,
                seg: 1,
                ..Default::default()
            }],
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_domains: 0,
        };
        let tensors = (0..10)
            .map(|i| DevTensor {
                name: i.to_string(),
                bytes: 256 << 20,
                init: None,
            })
            .collect();
        (prog, tensors)
    }

    #[test]
    fn sparse_mla_fp8_routes_require_scales_and_selected_union() {
        let (mut prog, mut tensors) = fixture();
        prog.insts[1].op = DevOp::FlashMlaPrefillFp8 as u16;
        prog.insts[1].fj[1] = 8;
        prog.insts[1].t[7] = 9;
        let route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        assert_eq!(route.scale, Some(9));
        assert_eq!(route.index, 8);
        tensors[9].bytes = 81920 * 4 - 1;
        assert!(routes(&prog, &tensors, 2).is_err());
        tensors[9].bytes = 81920 * 4;
        prog.insts[1].fj[1] = u32::MAX;
        assert!(routes(&prog, &tensors, 2).is_err());
    }

    #[test]
    fn sparse_mla_rebases_ragged_rows_and_restores_early_fallback() {
        let (prog, tensors) = fixture();
        let mut route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        route.rebase(129, 65536).unwrap();
        assert!(route.active);
        assert_eq!(route.rows, 129);
        route.rebase(8192, 0).unwrap();
        assert!(!route.active);
        route.rebase(1, 2046).unwrap();
        assert!(!route.active);
        route.rebase(1, 2047).unwrap();
        assert!(route.active);
        assert!(route.rebase(8193, 0).is_err());
        assert!(route.rebase(129, 81920).is_err());
    }

    #[test]
    fn sparse_mla_rejects_counter_obligations_in_both_streams() {
        for global in [false, true] {
            for kind in 0..3 {
                let (mut prog, tensors) = fixture();
                prog.gq_stream = prog.stream.clone();
                let entry = if global {
                    &mut prog.gq_stream[0]
                } else {
                    &mut prog.stream[0]
                };
                match kind {
                    0 => entry.wait_len = 1,
                    1 => entry.succ_len = 1,
                    _ => entry.flags = SE_XCTR,
                }
                assert!(routes(&prog, &tensors, 2).is_err());
            }
        }
    }

    #[test]
    fn sparse_mla_rejects_mixed_segments_and_wrong_geometry() {
        for kind in 0..6 {
            let (mut prog, mut tensors) = fixture();
            match kind {
                0 => prog.stream.push(StreamEnt {
                    inst: 0,
                    seg: 1,
                    ..Default::default()
                }),
                1 => prog.stream.push(StreamEnt {
                    inst: 1,
                    seg: 0,
                    ..Default::default()
                }),
                2 => prog.insts[0].i[1] = 1024,
                3 => prog.insts[1].i[1] = 16,
                4 => tensors[4].bytes = 16,
                _ => prog.insts[1].t[2] = TENSOR_NONE16,
            }
            assert!(routes(&prog, &tensors, 2).is_err());
        }
    }
}

pub(super) fn routes(
    prog: &DevProg,
    tensors: &[DevTensor],
    segments: usize,
) -> Result<Vec<Option<Route>>> {
    let mut routes = vec![None; segments];
    if prog.packed_prefill_only || prog.t < 2048 {
        return Ok(routes);
    }
    for (ix, inst) in prog.insts.iter().enumerate() {
        let Some(union_handle) = union_handle(inst) else {
            continue;
        };
        let fp8 = inst.op == DevOp::FlashMlaPrefillFp8 as u16;
        let err = |s: &str| RuntimeError::Device(format!("sparse AITER MLA instruction {ix}: {s}"));
        if inst.i[0] != 1
            || inst.i[1] != 8
            || inst.i[2] > 81920
            || inst.i[2] < 2048
            || inst.i[3] != 0
            || inst.i[4] != prog.t
            || inst.i[5] != u32::MAX
            || prog.t > 8192
            || (!fp8 && inst.fj[1] != 0)
            || inst.fj[2] != 0
            || !f32::from_bits(inst.fj[0]).is_finite()
            || f32::from_bits(inst.fj[0]) <= 0.0
        {
            return Err(err(
                "requires BF16 QH8 latent512/rope64, rows<=8192, ctx<=81920",
            ));
        }
        let union = prog.insts[..ix]
            .iter()
            .rev()
            .find(|d| d.op == DevOp::IndexUnionPf as u16 && u32::from(d.t[0]) == union_handle)
            .ok_or_else(|| err("no preceding index union"))?;
        if union.i[..5] != [prog.t, 2048, inst.i[2], inst.i[6], 8] || union.t[3] != inst.t[6] {
            return Err(err(
                "index union does not describe 2048 selected keys per query",
            ));
        }
        let ctx = u64::from(inst.i[2]);
        let rows = u64::from(prog.t);
        for (handle, bytes) in [
            (inst.t[0], rows * 8 * 512 * 4),
            (inst.t[1], rows * 8 * 2 * 4),
            (inst.t[2], rows * 8 * 512 * 2),
            (inst.t[3], rows * 8 * 64 * 2),
            (inst.t[4], ctx * 512 * if fp8 { 1 } else { 2 }),
            (inst.t[5], ctx * 64 * 2),
            (union.t[2], rows * 2048 * 4),
        ] {
            if handle == TENSOR_NONE16
                || tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes)
            {
                return Err(err("operand capacity is insufficient"));
            }
        }
        if fp8
            && (inst.t[7] == TENSOR_NONE16
                || tensors
                    .get(inst.t[7] as usize)
                    .is_none_or(|t| t.bytes < ctx * 4))
        {
            return Err(err("FP8 scale capacity is insufficient"));
        }
        let mut owner = BTreeSet::new();
        for entry in prog
            .stream
            .iter()
            .chain(&prog.gq_stream)
            .filter(|e| e.inst as usize == ix)
        {
            if entry.wait_len != 0 || entry.succ_len != 0 || entry.flags & SE_XCTR != 0 {
                return Err(err(
                    "counter obligations remain; re-emit with PLOW_MLA_PF_AITER=1",
                ));
            }
            owner.insert(entry.seg as usize);
        }
        if owner.len() != 1 {
            return Err(err("requires exactly one segment owner"));
        }
        let seg = *owner.first().unwrap();
        if seg >= segments
            || prog
                .stream
                .iter()
                .chain(&prog.gq_stream)
                .any(|e| e.seg as usize == seg && e.inst as usize != ix)
        {
            return Err(err("segment contains other interpreter work"));
        }
        routes[seg] = Some(Route {
            inst: *inst,
            index: union.t[2],
            scale: fp8.then_some(inst.t[7]),
            rows: prog.t,
            kv_len: 0,
            active: false,
        });
    }
    Ok(routes)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackArgs {
    q: u64,
    kv: u64,
    qa: u64,
    qr: u64,
    ck: u64,
    kr: u64,
    qp: u64,
    kp: u64,
    last: u64,
    splits: u64,
    rows: u32,
    ctx: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackFp8Args {
    base: PackArgs,
    scale: u64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackSingleArgs {
    base: PackFp8Args,
    ml: u64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceArgs {
    out: u64,
    ml: u64,
    part: u64,
    lse: u64,
    rows: u32,
    pad: u32,
}

const _: () = assert!(std::mem::size_of::<PackArgs>() == 88);
const _: () = assert!(std::mem::size_of::<PackFp8Args>() == 96);
const _: () = assert!(std::mem::size_of::<PackSingleArgs>() == 104);
const _: () = assert!(std::mem::size_of::<ReduceArgs>() == 40);

pub(super) struct SparseMla {
    pack: HsaKernel,
    pack_fp8: Option<HsaKernel>,
    pack_fp8_single: Option<HsaKernel>,
    attention: HsaKernel,
    reduce: HsaKernel,
    _scratch: DeviceMem,
    q: u64,
    kv: u64,
    part: u64,
    lse: u64,
    qp: u64,
    kp: u64,
    last: u64,
    splits: u64,
}

fn load_attention(be: &HsaBackend, dir: &Path, modules: &mut Vec<Module>) -> Result<HsaKernel> {
    let path = dir.join(OBJECT);
    let mut image = std::fs::read(&path)
        .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
    if plow_asset::decode_objects::image_sha256(&image) != OBJECT_HASH {
        return Err(RuntimeError::Device(
            "sparse AITER MLA object hash does not match qualified ABI".into(),
        ));
    }
    // This exact object declares 320 bytes in metadata but leaves KERNARG_SIZE
    // unspecified in its descriptor. ROCr reports the descriptor's zero, unlike HIP.
    // The hash fixes the descriptor at file offset 0x1000; only its size is normalized.
    image[0x1008..0x100c].copy_from_slice(&320u32.to_le_bytes());
    let module = EngineDevice::module_load(be, &image)?;
    let attention = EngineDevice::get_function(
        be,
        &module,
        "_ZN5aiter36mla_a16w16_qh8_qseqlen1_gqaratio8_v3E",
    )?;
    if attention.kernarg_size() != 320
        || attention.private_segment_size() != 0
        || HsaBackend::kernel_lds_bytes(&attention) != 65536
    {
        return Err(RuntimeError::Device(format!(
            "sparse AITER MLA resource ABI mismatch: kernarg={}, private={}, LDS={}",
            attention.kernarg_size(),
            attention.private_segment_size(),
            HsaBackend::kernel_lds_bytes(&attention)
        )));
    }
    modules.push(module);
    Ok(attention)
}

impl SparseMla {
    pub fn load(
        be: &HsaBackend,
        dir: &Path,
        rows: u32,
        ctx: u32,
        fp8: bool,
        modules: &mut Vec<Module>,
    ) -> Result<Self> {
        let attention = load_attention(be, dir, modules)?;
        let path = dir.join("mla_sparse_adapter_gfx942.elf");
        let image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        if !super::amd::elf_symbol_names(&image).contains(&"plow_mla_sparse_adapter_abi_1") {
            return Err(RuntimeError::Device(
                "sparse MLA adapter lacks ABI marker".into(),
            ));
        }
        let module = EngineDevice::module_load(be, &image)?;
        let pack = EngineDevice::get_function(be, &module, "plow_mla_sparse_pack")?;
        let pack_fp8 = if fp8 {
            if !super::amd::elf_symbol_names(&image).contains(&"plow_mla_sparse_adapter_fp8_abi_1")
            {
                return Err(RuntimeError::Device(
                    "sparse MLA adapter lacks FP8 ABI marker".into(),
                ));
            }
            let kernel = EngineDevice::get_function(be, &module, "plow_mla_sparse_pack_fp8")?;
            if ![96, 352].contains(&kernel.kernarg_size()) || kernel.private_segment_size() != 0 {
                return Err(RuntimeError::Device(
                    "sparse MLA FP8 pack resource ABI mismatch".into(),
                ));
            }
            Some(kernel)
        } else {
            None
        };
        let reduce = EngineDevice::get_function(be, &module, "plow_mla_sparse_reduce")?;
        let pack_fp8_single = if fp8
            && super::amd::elf_symbol_names(&image).contains(&"plow_mla_sparse_single_abi_1")
        {
            let kernel =
                EngineDevice::get_function(be, &module, "plow_mla_sparse_pack_fp8_single")?;
            if ![104, 360].contains(&kernel.kernarg_size()) || kernel.private_segment_size() != 0 {
                return Err(RuntimeError::Device(
                    "sparse MLA single-pass pack resource ABI mismatch".into(),
                ));
            }
            Some(kernel)
        } else {
            None
        };
        for (kernel, size) in [(pack, 88), (reduce, 40)] {
            if ![size, size + 256].contains(&kernel.kernarg_size())
                || kernel.private_segment_size() != 0
            {
                return Err(RuntimeError::Device(
                    "sparse MLA adapter resource ABI mismatch".into(),
                ));
            }
        }
        modules.push(module);
        let rows = u64::from(rows);
        let sizes = [
            rows * 8 * 576 * 2,
            u64::from(ctx) * 576 * 2,
            rows * 2 * 8 * 512 * 4,
            rows * 2 * 8 * 4,
            (rows + 1) * 4,
            (rows + 1) * 4,
            (rows + 1) * 4,
            (rows + 1) * 4,
        ];
        let bytes = sizes.iter().map(|s| s.div_ceil(256) * 256).sum();
        let scratch = EngineDevice::alloc(be, bytes)?;
        let mut ptr = scratch.base;
        let offsets = sizes.map(|size| {
            let p = ptr;
            ptr += size.div_ceil(256) * 256;
            p
        });
        let [q, kv, part, lse, qp, kp, last, splits] = offsets;
        tracing::info!(
            bytes,
            single_pass = pack_fp8_single.is_some(),
            "allocated sparse AITER MLA workspace"
        );
        Ok(Self {
            pack,
            pack_fp8,
            pack_fp8_single,
            attention,
            reduce,
            _scratch: scratch,
            q,
            kv,
            part,
            lse,
            qp,
            kp,
            last,
            splits,
        })
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, tensor_table: &[u8]) -> Result<()> {
        let addr = |handle: u16| {
            let at = usize::from(handle) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        };
        let t = route.inst.t;
        let single = route.rows >= 512 && route.scale.is_some() && self.pack_fp8_single.is_some();
        let splits = if single { 1 } else { 2 };
        let pack = PackArgs {
            q: self.q,
            kv: self.kv,
            qa: addr(t[2]),
            qr: addr(t[3]),
            ck: addr(t[4]),
            kr: addr(t[5]),
            qp: self.qp,
            kp: self.kp,
            last: self.last,
            splits: self.splits,
            rows: route.rows,
            // VMM may leave the capacity beyond the live prefix unmapped.
            ctx: route.kv_len,
        };
        if let Some(scale) = route.scale {
            let kernel = self
                .pack_fp8
                .ok_or_else(|| RuntimeError::Device("FP8 sparse MLA pack was not loaded".into()))?;
            let args = PackFp8Args {
                base: pack,
                scale: addr(scale),
            };
            if single {
                let args = PackSingleArgs {
                    base: args,
                    ml: addr(t[1]),
                };
                be.launch(
                    self.pack_fp8_single.unwrap(),
                    304,
                    256,
                    0,
                    bytemuck::bytes_of(&args),
                )?;
            } else {
                be.launch(kernel, 304, 256, 0, bytemuck::bytes_of(&args))?;
            }
        } else {
            be.launch(self.pack, 304, 256, 0, bytemuck::bytes_of(&pack))?;
        }
        let mut args = [0u64; 40];
        args[0] = if single { addr(t[0]) } else { self.part };
        args[2] = self.lse;
        args[4] = self.q;
        args[6] = self.kv;
        args[8] = self.kp;
        args[10] = addr(route.index);
        args[12] = self.last;
        args[14] = u64::from(route.inst.fj[0]);
        args[16] = 8;
        args[18] = splits;
        args[20] = 8 * 576 * 2;
        args[22] = 576 * 2;
        args[26] = self.qp;
        args[28] = if single { self.qp } else { self.splits };
        // The pinned kernel writes normalized FP32 at one split when out_16_nosplit stays zero.
        be.launch_3d(
            self.attention,
            [1, route.rows, splits as u32],
            256,
            bytemuck::cast_slice(&args),
        )?;
        if single {
            return Ok(());
        }
        let reduce = ReduceArgs {
            out: addr(t[0]),
            ml: addr(t[1]),
            part: self.part,
            lse: self.lse,
            rows: route.rows,
            pad: 0,
        };
        be.launch(self.reduce, 304, 256, 0, bytemuck::bytes_of(&reduce))
    }
}

/// Decode twin of [`Route`]: one isolated sparse FP8 `FlashMlaDecodeFp8` per segment, every
/// row with its own 2048-key selection, `nsplit` partials per (row, head) for the merge.
#[derive(Clone, Copy, Debug)]
pub(super) struct DecodeRoute {
    inst: DevInst64,
    index: u16,
    rows: u32,
    pub active: bool,
}

impl DecodeRoute {
    /// The fixed-width CSR needs every row to hold all 2048 keys; shorter rows run the
    /// interpreter arm for the whole step.
    pub fn arm(&mut self, kvlen: &[u32]) {
        let rows = self.rows as usize;
        self.active = kvlen.len() >= rows && kvlen[..rows].iter().all(|&k| k >= 2048);
    }
}

pub(super) fn sparse_decode_inst(inst: &DevInst64) -> bool {
    inst.op == DevOp::FlashMlaDecodeFp8 as u16 && inst.fj[1] != 0
}

pub(super) fn decode_routes(
    prog: &DevProg,
    tensors: &[DevTensor],
    segments: usize,
) -> Result<Vec<Option<DecodeRoute>>> {
    let mut routes = vec![None; segments];
    for (ix, inst) in prog.insts.iter().enumerate() {
        if !sparse_decode_inst(inst) {
            continue;
        }
        // Only a pure, counter-free segment is a native boundary; the ordinary emit keeps the
        // op inside an interpreter segment and is not this route's business.
        let mut owner = BTreeSet::new();
        let mut obligations = false;
        for entry in prog
            .stream
            .iter()
            .chain(&prog.gq_stream)
            .filter(|e| e.inst as usize == ix)
        {
            obligations |= entry.wait_len != 0 || entry.succ_len != 0 || entry.flags & SE_XCTR != 0;
            owner.insert(entry.seg as usize);
        }
        let Some(&seg) = owner.first().filter(|_| owner.len() == 1 && !obligations) else {
            continue;
        };
        if seg >= segments
            || prog
                .stream
                .iter()
                .chain(&prog.gq_stream)
                .any(|e| e.seg as usize == seg && e.inst as usize != ix)
        {
            continue;
        }
        let err = |s: &str| {
            RuntimeError::Device(format!("sparse AITER MLA decode instruction {ix}: {s}"))
        };
        let ns = inst.i[4];
        if inst.i[0] != prog.t
            || prog.t > 20
            || inst.i[1] != 8
            || !(2048..=81920).contains(&inst.i[2])
            || inst.i[3] != 0
            || !(1..=16).contains(&ns)
            || inst.i[5] != u32::MAX
            || inst.i[6] != 2048
            || inst.i[7] != 4
            || inst.fj[2] != 0
            || inst.fj[1] > u32::from(TENSOR_NONE16)
            || !f32::from_bits(inst.fj[0]).is_finite()
            || f32::from_bits(inst.fj[0]) <= 0.0
        {
            return Err(err(
                "requires QH8 latent512/rope64 GF4 top-2048 rows<=20, nsplit<=16, ctx 2048..81920",
            ));
        }
        let index = (inst.fj[1] - 1) as u16;
        let rows = u64::from(prog.t);
        let ctx = u64::from(inst.i[2]);
        let ns = u64::from(ns);
        for (handle, bytes) in [
            (inst.t[0], rows * 8 * ns * 512 * 4),
            (inst.t[1], rows * 8 * ns * 2 * 4),
            (inst.t[2], rows * 8 * 512 * 2),
            (inst.t[3], rows * 8 * 64 * 2),
            (inst.t[4], rows * ctx * 512),
            (inst.t[5], rows * ctx * 64 * 2),
            (inst.t[6], rows * 4),
            (inst.t[7], rows * ctx * 4),
            (index, rows * 2048 * 4),
        ] {
            if handle == TENSOR_NONE16
                || tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes)
            {
                return Err(err("operand capacity is insufficient"));
            }
        }
        if inst.t[1] == inst.t[0] {
            return Err(err("partial outputs alias"));
        }
        routes[seg] = Some(DecodeRoute {
            inst: *inst,
            index,
            rows: prog.t,
            active: false,
        });
    }
    Ok(routes)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DecodePackArgs {
    q: u64,
    kv: u64,
    kvi: u64,
    qa: u64,
    qr: u64,
    ck: u64,
    kr: u64,
    idx: u64,
    kvlen: u64,
    scale: u64,
    qp: u64,
    kp: u64,
    last: u64,
    splits: u64,
    rows: u32,
    ctx: u32,
    nsplit: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RelayoutArgs {
    opart: u64,
    ml: u64,
    part: u64,
    lse: u64,
    rows: u32,
    nsplit: u32,
}

const _: () = assert!(std::mem::size_of::<DecodePackArgs>() == 128);
const _: () = assert!(std::mem::size_of::<RelayoutArgs>() == 40);

pub(super) struct SparseMlaDecode {
    pack: HsaKernel,
    attention: HsaKernel,
    relayout: HsaKernel,
    _scratch: DeviceMem,
    q: u64,
    kv: u64,
    kvi: u64,
    part: u64,
    lse: u64,
    qp: u64,
    kp: u64,
    last: u64,
    splits: u64,
}

impl SparseMlaDecode {
    pub fn load(be: &HsaBackend, dir: &Path, rows: u32, modules: &mut Vec<Module>) -> Result<Self> {
        if !(1..=20).contains(&rows) {
            return Err(RuntimeError::Device(
                "sparse AITER MLA decode requires 1..20 rows".into(),
            ));
        }
        let attention = load_attention(be, dir, modules)?;
        let path = dir.join("mla_sparse_adapter_gfx942.elf");
        let image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        let syms = super::amd::elf_symbol_names(&image);
        if !syms.contains(&"plow_mla_sparse_adapter_abi_1")
            || !syms.contains(&"plow_mla_sparse_decode_abi_1")
        {
            return Err(RuntimeError::Device(
                "sparse MLA adapter lacks decode ABI marker".into(),
            ));
        }
        let module = EngineDevice::module_load(be, &image)?;
        let pack = EngineDevice::get_function(be, &module, "plow_mla_sparse_decode_pack")?;
        let relayout = EngineDevice::get_function(be, &module, "plow_mla_sparse_decode_relayout")?;
        for (kernel, size) in [(pack, 128), (relayout, 40)] {
            if ![size, size + 256].contains(&kernel.kernarg_size())
                || kernel.private_segment_size() != 0
            {
                return Err(RuntimeError::Device(
                    "sparse MLA decode adapter resource ABI mismatch".into(),
                ));
            }
        }
        modules.push(module);
        let rows = u64::from(rows);
        let sizes = [
            rows * 8 * 576 * 2,
            rows * 2048 * 576 * 2,
            rows * 2048 * 4,
            rows * 16 * 8 * 512 * 4,
            rows * 16 * 8 * 4,
            (rows + 1) * 4,
            (rows + 1) * 4,
            (rows + 1) * 4,
            (rows + 1) * 4,
        ];
        let bytes = sizes.iter().map(|s| s.div_ceil(256) * 256).sum();
        let scratch = EngineDevice::alloc(be, bytes)?;
        let mut ptr = scratch.base;
        let offsets = sizes.map(|size| {
            let p = ptr;
            ptr += size.div_ceil(256) * 256;
            p
        });
        let [q, kv, kvi, part, lse, qp, kp, last, splits] = offsets;
        tracing::info!(bytes, rows, "allocated sparse AITER MLA decode workspace");
        Ok(Self {
            pack,
            attention,
            relayout,
            _scratch: scratch,
            q,
            kv,
            kvi,
            part,
            lse,
            qp,
            kp,
            last,
            splits,
        })
    }

    pub fn enqueue(&self, be: &HsaBackend, route: DecodeRoute, tensor_table: &[u8]) -> Result<()> {
        let addr = |handle: u16| {
            let at = usize::from(handle) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        };
        let t = route.inst.t;
        let ns = route.inst.i[4];
        let pack = DecodePackArgs {
            q: self.q,
            kv: self.kv,
            kvi: self.kvi,
            qa: addr(t[2]),
            qr: addr(t[3]),
            ck: addr(t[4]),
            kr: addr(t[5]),
            idx: addr(route.index),
            kvlen: addr(t[6]),
            scale: addr(t[7]),
            qp: self.qp,
            kp: self.kp,
            last: self.last,
            splits: self.splits,
            rows: route.rows,
            ctx: route.inst.i[2],
            nsplit: ns,
            pad: 0,
        };
        be.launch(self.pack, 304, 256, 0, bytemuck::bytes_of(&pack))?;
        let mut args = [0u64; 40];
        args[0] = self.part;
        args[2] = self.lse;
        args[4] = self.q;
        args[6] = self.kv;
        args[8] = self.kp;
        args[10] = self.kvi;
        args[12] = self.last;
        args[14] = u64::from(route.inst.fj[0]);
        args[16] = 8;
        args[18] = u64::from(ns);
        args[20] = 8 * 576 * 2;
        args[22] = 576 * 2;
        args[26] = self.qp;
        args[28] = self.splits;
        be.launch_3d(
            self.attention,
            [1, route.rows, ns],
            256,
            bytemuck::cast_slice(&args),
        )?;
        let relayout = RelayoutArgs {
            opart: addr(t[0]),
            ml: addr(t[1]),
            part: self.part,
            lse: self.lse,
            rows: route.rows,
            nsplit: ns,
        };
        be.launch(self.relayout, 304, 256, 0, bytemuck::bytes_of(&relayout))
    }
}

#[cfg(test)]
mod decode_tests {
    use super::*;
    use packet::dev::StreamEnt;

    fn fixture(rows: u32) -> (DevProg, Vec<DevTensor>) {
        let flash = DevInst64 {
            op: DevOp::FlashMlaDecodeFp8 as u16,
            t: [0, 1, 2, 3, 4, 5, 6, 7],
            i: [rows, 8, 4096, 0, 16, u32::MAX, 2048, 4],
            fj: [0.0625f32.to_bits(), 9, 0],
            ..Default::default()
        };
        let prog = DevProg {
            t: rows,
            packed_prefill_only: false,
            n_counter: 0,
            insts: vec![
                DevInst64 {
                    op: DevOp::Nop as u16,
                    ..Default::default()
                },
                flash,
            ],
            stream: vec![
                StreamEnt {
                    inst: 0,
                    seg: 0,
                    ..Default::default()
                },
                StreamEnt {
                    inst: 1,
                    seg: 1,
                    ..Default::default()
                },
            ],
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_domains: 0,
        };
        let tensors = (0..9)
            .map(|i| DevTensor {
                name: i.to_string(),
                bytes: 64 << 20,
                init: None,
            })
            .collect();
        (prog, tensors)
    }

    #[test]
    fn sparse_decode_routes_require_pure_segments_geometry_and_capacity() {
        let (prog, tensors) = fixture(20);
        let routes = decode_routes(&prog, &tensors, 2).unwrap();
        assert!(routes[0].is_none());
        let mut route = routes[1].unwrap();
        assert_eq!(route.index, 8);
        assert!(!route.active);
        route.arm(&[2048; 20]);
        assert!(route.active);
        route.arm(&[2048; 19]);
        assert!(!route.active);
        let mut short = [4096u32; 20];
        short[7] = 2047;
        route.arm(&short);
        assert!(!route.active);
        // Mixed or counter-bearing segments are the ordinary emit: no route, no error.
        let (mut mixed, tensors) = fixture(20);
        mixed.stream[1].seg = 0;
        assert!(decode_routes(&mixed, &tensors, 1).unwrap()[0].is_none());
        let (mut waits, tensors) = fixture(20);
        waits.stream[1].wait_len = 1;
        assert!(decode_routes(&waits, &tensors, 2).unwrap()[1].is_none());
        for bad in 0..9 {
            let (mut prog, mut tensors) = fixture(20);
            match bad {
                0 => prog.insts[1].i[0] = 8,
                1 => prog.insts[1].i[1] = 16,
                2 => prog.insts[1].i[4] = 17,
                3 => prog.insts[1].i[6] = 1024,
                4 => prog.insts[1].i[7] = 2,
                5 => prog.insts[1].fj[2] = 1,
                6 => tensors[8].bytes = 20 * 2048 * 4 - 1,
                7 => tensors[7].bytes = 20 * 4096 * 4 - 1,
                _ => prog.insts[1].t[7] = TENSOR_NONE16,
            }
            assert!(decode_routes(&prog, &tensors, 2).is_err(), "case {bad}");
        }
        let (mut wide, tensors) = fixture(21);
        wide.insts[1].i[0] = 21;
        assert!(decode_routes(&wide, &tensors, 2).is_err());
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with the decode adapter"]
    fn sparse_mla_decode_hsa_dispatch() {
        const CTX: usize = 4096;
        const NS: usize = 16;
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let kernel = SparseMlaDecode::load(&be, Path::new(&dir), 20, &mut modules).unwrap();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let bf16 = |v: f32| {
            let b = v.to_bits();
            ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
        };
        let unbf16 = |b: u16| f64::from(f32::from_bits(u32::from(b) << 16));
        let decode = |byte: u8| -> f64 {
            let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
            let exp = i32::from((byte >> 3) & 0xf);
            let man = f64::from(byte & 7) / 8.0;
            if exp == 0 {
                sign * man * 2f64.powi(-6)
            } else {
                sign * (1.0 + man) * 2f64.powi(exp - 7)
            }
        };
        for rows in [1usize, 8, 20] {
            // Random FP8 latent (magnitudes below 2, no NaN pattern), per-row scales, BF16 rope,
            // BF16 queries, and a distinct random 2048-key selection per row.
            let ck: Vec<u8> = (0..rows * CTX * 512)
                .map(|_| {
                    let r = next();
                    (((r >> 8) & 1) << 7) as u8 | (((r >> 4) % 8) << 3) as u8 | (r % 8) as u8
                })
                .collect();
            let scales: Vec<f32> = (0..rows * CTX)
                .map(|_| 0.5 + (next() % 1000) as f32 / 1000.0)
                .collect();
            let kr: Vec<u16> = (0..rows * CTX * 64)
                .map(|_| bf16((next() % 2001) as f32 / 1000.0 - 1.0))
                .collect();
            let qa: Vec<u16> = (0..rows * 8 * 512)
                .map(|_| bf16((next() % 2001) as f32 / 2000.0 - 0.5))
                .collect();
            let qr: Vec<u16> = (0..rows * 8 * 64)
                .map(|_| bf16((next() % 2001) as f32 / 2000.0 - 0.5))
                .collect();
            let kvlen: Vec<i32> = (0..rows).map(|b| (2048 + b * 97).min(CTX) as i32).collect();
            let mut idx = vec![0i32; rows * 2048];
            for b in 0..rows {
                let len = kvlen[b] as usize;
                let mut keys: Vec<i32> = (0..len as i32).collect();
                for i in (1..len).rev() {
                    keys.swap(i, (next() % (i as u64 + 1)) as usize);
                }
                idx[b * 2048..(b + 1) * 2048].copy_from_slice(&keys[..2048]);
            }
            let upload = |bytes: &[u8]| {
                let m = EngineDevice::alloc(&be, bytes.len() as u64 + 1024).unwrap();
                EngineDevice::upload(&be, &m, 0, bytes).unwrap();
                m
            };
            let opart_bytes = rows * 8 * NS * 512 * 4;
            let ml_bytes = rows * 8 * NS * 2 * 4;
            let opart = upload(&vec![0xa5u8; opart_bytes + 512]);
            let mlpart = upload(&vec![0xa5u8; ml_bytes + 512]);
            let d_qa = upload(bytemuck::cast_slice(&qa));
            let d_qr = upload(bytemuck::cast_slice(&qr));
            let d_ck = upload(&ck);
            let d_kr = upload(bytemuck::cast_slice(&kr));
            let d_kvlen = upload(bytemuck::cast_slice(&kvlen));
            let d_scale = upload(bytemuck::cast_slice(&scales));
            let d_idx = upload(bytemuck::cast_slice(&idx));
            let table = [
                opart.base + 256,
                mlpart.base + 256,
                d_qa.base,
                d_qr.base,
                d_ck.base,
                d_kr.base,
                d_kvlen.base,
                d_scale.base,
                d_idx.base,
            ];
            let (prog, tensors) = fixture(rows as u32);
            let mut route = decode_routes(&prog, &tensors, 2).unwrap()[1].unwrap();
            route.arm(&kvlen.iter().map(|&k| k as u32).collect::<Vec<_>>());
            assert!(route.active);
            kernel
                .enqueue(&be, route, bytemuck::cast_slice(&table))
                .unwrap();
            be.synchronize().unwrap();
            let mut ob = vec![0u8; opart_bytes + 512];
            let mut mb = vec![0u8; ml_bytes + 512];
            EngineDevice::download(&be, &opart, 0, &mut ob).unwrap();
            EngineDevice::download(&be, &mlpart, 0, &mut mb).unwrap();
            for (bytes, len) in [(&ob, opart_bytes), (&mb, ml_bytes)] {
                assert!(bytes[..256]
                    .iter()
                    .chain(&bytes[256 + len..])
                    .all(|&x| x == 0xa5));
            }
            let op: &[f32] = bytemuck::cast_slice(&ob[256..256 + opart_bytes]);
            let ml: &[f32] = bytemuck::cast_slice(&mb[256..256 + ml_bytes]);
            let (mut num, mut den) = (0f64, 0f64);
            for b in 0..rows {
                for h in 0..8 {
                    // Host copy of d_mla_merge_fold's combine over the NS partials.
                    let base = (b * 8 + h) * NS;
                    let gm = (0..NS)
                        .map(|s| ml[(base + s) * 2])
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut merged = vec![0f64; 512];
                    let mut gl = 0f64;
                    for s in 0..NS {
                        let (m, l) = (ml[(base + s) * 2], ml[(base + s) * 2 + 1]);
                        assert!(
                            m.is_finite() && l == 1.0,
                            "rows={rows} b={b} h={h} s={s}: ml=({m},{l})"
                        );
                        let w = f64::from(l) * f64::from((m - gm).exp());
                        gl += w;
                        for d in 0..512 {
                            merged[d] += f64::from(op[(base + s) * 512 + d]) * w;
                        }
                    }
                    for v in &mut merged {
                        *v /= gl;
                    }
                    // FP64 reference over the row's selected keys on the BF16-packed values.
                    let q: Vec<f64> = (0..576)
                        .map(|d| {
                            if d < 512 {
                                unbf16(qa[(b * 8 + h) * 512 + d])
                            } else {
                                unbf16(qr[(b * 8 + h) * 64 + d - 512])
                            }
                        })
                        .collect();
                    let mut scores = Vec::with_capacity(2048);
                    let mut keys = Vec::with_capacity(2048);
                    for j in 0..2048 {
                        let key = idx[b * 2048 + j] as usize;
                        let src = b * CTX + key;
                        let scale = f64::from(scales[src]);
                        let mut k: Vec<f64> = (0..512)
                            .map(|d| {
                                let v = (decode(ck[src * 512 + d]) * scale) as f32;
                                unbf16(bf16(v))
                            })
                            .collect();
                        k.extend((0..64).map(|d| unbf16(kr[src * 64 + d])));
                        scores.push(0.0625 * q.iter().zip(&k).map(|(a, b)| a * b).sum::<f64>());
                        keys.push(k);
                    }
                    let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let ws: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
                    let total: f64 = ws.iter().sum();
                    for d in 0..512 {
                        let r: f64 =
                            ws.iter().zip(&keys).map(|(w, k)| w * k[d]).sum::<f64>() / total;
                        num += (merged[d] - r).powi(2);
                        den += r.powi(2);
                    }
                }
            }
            let rel = (num / den).sqrt();
            // Warm route latency (pack + attention + relayout, host-timed through a drain).
            for _ in 0..3 {
                kernel
                    .enqueue(&be, route, bytemuck::cast_slice(&table))
                    .unwrap();
            }
            be.synchronize().unwrap();
            let mut samples: Vec<f64> = (0..15)
                .map(|_| {
                    let start = std::time::Instant::now();
                    kernel
                        .enqueue(&be, route, bytemuck::cast_slice(&table))
                        .unwrap();
                    be.synchronize().unwrap();
                    start.elapsed().as_secs_f64() * 1e6
                })
                .collect();
            samples.sort_by(|a, b| a.total_cmp(b));
            eprintln!(
                "rows={rows} rel_l2 vs fp64 = {rel:.3e}; route warm median {:.1}us",
                samples[samples.len() / 2]
            );
            assert!(rel < 5e-3, "rows={rows}: rel L2 {rel}");
        }
    }
}
