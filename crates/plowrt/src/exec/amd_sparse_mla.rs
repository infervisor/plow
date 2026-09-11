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
            token_batch_body: false,
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

impl SparseMla {
    pub fn load(
        be: &HsaBackend,
        dir: &Path,
        rows: u32,
        ctx: u32,
        fp8: bool,
        modules: &mut Vec<Module>,
    ) -> Result<Self> {
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
