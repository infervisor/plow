use std::collections::BTreeSet;
use std::path::Path;

use packet::dev::{DevInst64, DevOp, SE_XCTR, TENSOR_NONE16};

use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::{DeviceMem, Module};
use crate::exec::device_api::EngineDevice;
use crate::{Result, RuntimeError};

const OBJECT: &str = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co";
const OBJECT_HASH: &str = "65b4c0a0b290dd83039047c18e0bb86f4253790e926dce324ddb6b45a7b28650";

const FLAT_OBJECT: &str = "fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3.co";
const FLAT_OBJECT_HASH: &str = "be7052284094e7cedeb266afb24d4d6723bdf4234ac391b2e5b29473d8ee8f06";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Sorted,
    Flat,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    inst: DevInst64,
    rows: u32,
    mode: Mode,
}

impl Route {
    pub fn launches(self) -> u64 {
        match self.mode {
            Mode::Sorted => 4,
            Mode::Flat => 2,
        }
    }

    pub fn rebase(&mut self, rows: u32) -> Result<()> {
        if rows == 0 || rows > self.inst.i[0] {
            return Err(RuntimeError::Device(
                "AITER MoE chunk exceeds row capacity".into(),
            ));
        }
        self.rows = rows;
        Ok(())
    }
}

pub(super) fn routes(
    prog: &DevProg,
    tensors: &[DevTensor],
    segments: usize,
) -> Result<Vec<Option<Route>>> {
    let mut routes = vec![None; segments];
    for (ix, inst) in prog.insts.iter().enumerate() {
        if inst.op != DevOp::MoeAiterFp8Pf as u16 {
            continue;
        }
        let err = |s: &str| RuntimeError::Device(format!("AITER MoE instruction {ix}: {s}"));
        let mode = if inst.i[6] == 1 {
            Mode::Flat
        } else {
            Mode::Sorted
        };
        let flat = mode == Mode::Flat;
        let geometry = if flat {
            [prog.t, 6144, 256, 256, 8, 0, 1, 0]
        } else {
            [prog.t, 6144, 256, 256, 8, 64, 0, 0]
        };
        if prog.packed_prefill_only
            || !(if flat {
                matches!(prog.t, 2 | 4 | 8)
            } else {
                (128..=8192).contains(&prog.t)
            })
            || inst.i != geometry
            || inst.fj != [0; 3]
        {
            return Err(err("requires H6144/I256/E256/top8; sorted prefill rows128..8192 or flat decode rows2/4/8"));
        }
        if flat {
            let router = prog.insts[..ix]
                .iter()
                .rev()
                .find(|d| d.op == DevOp::MoeRouterTopkPf as u16 && d.t[0] == inst.t[4])
                .ok_or_else(|| err("no preceding raw routing table"))?;
            if router.i[1..3] != [256, 8] || router.i[4] != prog.t {
                return Err(err("raw routing geometry does not match"));
            }
            let combine = prog.insts[ix + 1..]
                .iter()
                .find(|d| d.op == DevOp::MoeCombinePf as u16 && d.t[3] == inst.t[0])
                .ok_or_else(|| err("flat output has no BF16 combine consumer"))?;
            if combine.i[..5] != [6144, 1, prog.t, 0, 0] || combine.i[7] != 1 {
                return Err(err("flat output requires one BF16 partial per row"));
            }
        } else {
            let align = prog.insts[..ix]
                .iter()
                .rev()
                .find(|d| d.op == DevOp::MoeAlignPf as u16 && d.t[0] == inst.t[4])
                .ok_or_else(|| err("no preceding aligned routing table"))?;
            if align.i[..3] != [prog.t, 256, 8]
                || ![0, 4].contains(&align.i[3])
                || align.t[2..5] != inst.t[5..8]
            {
                return Err(err("aligned routing geometry does not match"));
            }
        }
        let rows = u64::from(prog.t);
        let capacity = rows * 8 + 256 * 63;
        let sizes = if flat {
            [
                rows * 6144 * 2 + 8,
                rows * 6144 * 2,
                256 * 3 * 8,
                256 * 3 * 8,
                rows * 8 * 8,
                0,
                0,
                0,
            ]
        } else {
            [
                rows * 6144 * 4,
                rows * 6144 * 2,
                256 * 3 * 8,
                256 * 3 * 8,
                (3 * 256 + 1) * 4,
                capacity * 4,
                capacity * 4,
                capacity * 4,
            ]
        };
        for (handle, bytes) in inst.t.into_iter().zip(sizes) {
            if bytes == 0 {
                if handle != TENSOR_NONE16 {
                    return Err(err("unused flat operand must be absent"));
                }
            } else if handle == TENSOR_NONE16
                || tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes)
            {
                return Err(err("operand capacity is insufficient"));
            }
        }
        let mut owners = BTreeSet::new();
        for entry in prog
            .stream
            .iter()
            .chain(&prog.gq_stream)
            .filter(|e| e.inst as usize == ix)
        {
            if entry.wait_len != 0 || entry.succ_len != 0 || entry.flags & SE_XCTR != 0 {
                return Err(err(
                    "native segment retains interpreter counter obligations",
                ));
            }
            owners.insert(entry.seg as usize);
        }
        if owners.len() != 1 {
            return Err(err("requires exactly one segment owner"));
        }
        let seg = *owners.first().unwrap();
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
            rows: prog.t,
            mode,
        });
    }
    Ok(routes)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PrepareArgs {
    pointers: [u64; 12],
    rows: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlatPackArgs {
    pointers: [u64; 9],
    rows: u32,
    pad: u32,
}

const _: () = assert!(std::mem::size_of::<FlatPackArgs>() == 80);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StoreArgs {
    out: u64,
    src: u64,
    rows: u32,
    pad: u32,
}

const _: () = assert!(std::mem::size_of::<PrepareArgs>() == 104);
const _: () = assert!(std::mem::size_of::<StoreArgs>() == 24);

pub(super) struct MoeAiter {
    pack: HsaKernel,
    flat: Option<(HsaKernel, HsaKernel)>,
    prepare: HsaKernel,
    moe: HsaKernel,
    store: HsaKernel,
    _scratch: DeviceMem,
    buffers: [u64; 11],
}

impl MoeAiter {
    pub fn load(
        be: &HsaBackend,
        dir: &Path,
        rows: u32,
        flat_decode: bool,
        modules: &mut Vec<Module>,
    ) -> Result<Self> {
        if !(128..=8192).contains(&rows) {
            return Err(RuntimeError::Device(
                "AITER MoE workspace requires rows 128..8192".into(),
            ));
        }
        let path = dir.join(OBJECT);
        let mut image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        if plow_asset::decode_objects::image_sha256(&image) != OBJECT_HASH {
            return Err(RuntimeError::Device(
                "AITER MoE object hash does not match qualified ABI".into(),
            ));
        }
        // The pinned descriptor at 0x1d00 omits KERNARG_SIZE; HIP uses metadata,
        // but ROCr reads the descriptor. Normalize only this field after hashing.
        image[0x1d08..0x1d0c].copy_from_slice(&448u32.to_le_bytes());
        let module = EngineDevice::module_load(be, &image)?;
        let moe = EngineDevice::get_function(
            be,
            &module,
            "_ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256E",
        )?;
        if moe.kernarg_size() != 448
            || moe.private_segment_size() != 0
            || HsaBackend::kernel_lds_bytes(&moe) != 65536
        {
            return Err(RuntimeError::Device(
                "AITER MoE resource ABI mismatch".into(),
            ));
        }
        modules.push(module);
        let path = dir.join("moe_aiter_adapter_gfx942.elf");
        let image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        if !super::amd::elf_symbol_names(&image).contains(&"plow_moe_aiter_fp8_abi_1") {
            return Err(RuntimeError::Device(
                "AITER MoE adapter lacks ABI marker".into(),
            ));
        }
        let flat = if flat_decode {
            if !super::amd::elf_symbol_names(&image).contains(&"plow_moe_flat_fp8_abi_1") {
                return Err(RuntimeError::Device(
                    "flat MoE adapter lacks ABI marker".into(),
                ));
            }
            let path = dir.join(FLAT_OBJECT);
            let mut image = std::fs::read(&path)
                .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
            if plow_asset::decode_objects::image_sha256(&image) != FLAT_OBJECT_HASH {
                return Err(RuntimeError::Device(
                    "flat MoE object hash does not match qualified ABI".into(),
                ));
            }
            // The pinned descriptor omits KERNARG_SIZE although metadata declares 448 bytes.
            image[0x1208..0x120c].copy_from_slice(&448u32.to_le_bytes());
            let module = EngineDevice::module_load(be, &image)?;
            let moe = EngineDevice::get_function(
                be,
                &module,
                "_ZN5aiter60fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3E",
            )?;
            if moe.kernarg_size() != 448
                || moe.private_segment_size() != 0
                || HsaBackend::kernel_lds_bytes(&moe) != 65536
            {
                return Err(RuntimeError::Device(
                    "flat MoE resource ABI mismatch".into(),
                ));
            }
            modules.push(module);
            tracing::info!(object = %path.display(), sha256 = FLAT_OBJECT_HASH, "flat MoE decode kernels loaded");
            Some(moe)
        } else {
            None
        };
        let module = EngineDevice::module_load(be, &image)?;
        let flat = flat
            .map(|moe| -> Result<_> {
                let pack = EngineDevice::get_function(be, &module, "plow_moe_aiter_flat_pack")?;
                if ![80, 336].contains(&pack.kernarg_size()) || pack.private_segment_size() != 0 {
                    return Err(RuntimeError::Device("flat MoE packing ABI mismatch".into()));
                }
                Ok((pack, moe))
            })
            .transpose()?;
        let pack = EngineDevice::get_function(be, &module, "pack_moe")?;
        let prepare = EngineDevice::get_function(be, &module, "plow_moe_aiter_prepare")?;
        let store = EngineDevice::get_function(be, &module, "plow_moe_aiter_store")?;
        for (kernel, size) in [(pack, 48), (prepare, 104), (store, 24)] {
            if ![size, size + 256].contains(&kernel.kernarg_size())
                || kernel.private_segment_size() != 0
            {
                return Err(RuntimeError::Device(format!("AITER MoE adapter resource ABI mismatch: kernarg={} expected={size}, private={}",
                    kernel.kernarg_size(), kernel.private_segment_size())));
            }
        }
        modules.push(module);
        let rows = u64::from(rows);
        let capacity = rows * 8 + 256 * 63;
        let sizes = [
            256 * 512 * 6144,
            256 * 6144 * 256,
            256 * 192 * 4,
            256 * 96 * 4,
            rows * 6144,
            rows * 48 * 4,
            rows * 6144 * 2,
            capacity * 4,
            capacity * 4,
            capacity.div_ceil(32) * 4,
            8,
        ];
        let bytes = sizes.iter().map(|s| s.div_ceil(256) * 256).sum();
        let scratch = EngineDevice::alloc(be, bytes)?;
        let mut ptr = scratch.base;
        let buffers = sizes.map(|size| {
            let p = ptr;
            ptr += size.div_ceil(256) * 256;
            p
        });
        tracing::info!(bytes, "allocated reusable AITER MoE workspace");
        Ok(Self {
            pack,
            flat,
            prepare,
            moe,
            store,
            _scratch: scratch,
            buffers,
        })
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, tensor_table: &[u8]) -> Result<()> {
        let addr = |handle: u16| {
            let at = usize::from(handle) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        };
        let t = route
            .inst
            .t
            .map(|h| if h == TENSOR_NONE16 { 0 } else { addr(h) });
        let [gu, down, gs, ds, q, qs, out, ids, weights, experts, valid] = self.buffers;
        if route.mode == Mode::Flat {
            let (pack, moe) = self.flat.ok_or_else(|| {
                RuntimeError::Device("flat MoE route has no loaded kernels".into())
            })?;
            let pack_args = FlatPackArgs {
                pointers: [gu, down, gs, ds, t[2], t[3], t[4], ids, weights],
                rows: route.rows,
                pad: 0,
            };
            be.launch(pack, 2048, 256, 0, bytemuck::bytes_of(&pack_args))?;
            let args = flat_moe_args(t[0], t[1], gu, down, gs, ds, ids, weights, route.rows);
            return be.launch_3d(moe, [2, 8, route.rows], 256, bytemuck::cast_slice(&args));
        }
        be.launch(
            self.pack,
            2048,
            256,
            0,
            bytemuck::cast_slice(&[gu, down, gs, ds, t[2], t[3]]),
        )?;
        let prepare = PrepareArgs {
            pointers: [
                q, qs, out, ids, weights, experts, valid, t[1], t[4], t[5], t[6], t[7],
            ],
            rows: route.rows,
            pad: 0,
        };
        be.launch(self.prepare, 304 * 4, 256, 0, bytemuck::bytes_of(&prepare))?;
        let mut args = [0u64; 56];
        for (i, value) in [
            out,
            q,
            gu,
            valid,
            down,
            qs,
            gs,
            ds,
            0,
            ids,
            weights,
            experts,
            6144,
            256,
            u64::from(route.rows),
            256,
            6144,
            6144,
            256,
            12288,
            512 * 6144,
            256 * 6144,
            192 * 4,
            96 * 4,
            256 * 4,
            8,
            304,
            1,
        ]
        .into_iter()
        .enumerate()
        {
            args[i * 2] = value;
        }
        be.launch(self.moe, 304, 256, 0, bytemuck::cast_slice(&args))?;
        let store = StoreArgs {
            out: t[0],
            src: out,
            rows: route.rows,
            pad: 0,
        };
        be.launch(self.store, 304, 256, 0, bytemuck::bytes_of(&store))
    }
}

#[allow(clippy::too_many_arguments)]
fn flat_moe_args(
    out: u64,
    x: u64,
    gu: u64,
    down: u64,
    gs: u64,
    ds: u64,
    ids: u64,
    weights: u64,
    rows: u32,
) -> [u64; 56] {
    let mut args = [0; 56];
    for (i, value) in [
        out,
        x,
        gu,
        ids,
        down,
        0,
        gs,
        ds,
        0,
        ids,
        weights,
        ids,
        6144,
        256,
        u64::from(rows),
        256,
        12288,
        6144,
        256,
        12288,
        512 * 6144,
        256 * 6144,
        192 * 4,
        96 * 4,
        256 * 4,
        8,
        0,
        2,
    ]
    .into_iter()
    .enumerate()
    {
        args[i * 2] = value;
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::StreamEnt;

    fn fixture() -> (DevProg, Vec<DevTensor>) {
        let align = DevInst64 {
            op: DevOp::MoeAlignPf as u16,
            t: [4, 8, 5, 6, 7, TENSOR_NONE16, TENSOR_NONE16, TENSOR_NONE16],
            i: [8192, 256, 8, 4, 64, 0, 0, 0],
            ..Default::default()
        };
        let moe = DevInst64 {
            op: DevOp::MoeAiterFp8Pf as u16,
            t: [0, 1, 2, 3, 4, 5, 6, 7],
            i: [8192, 6144, 256, 256, 8, 64, 0, 0],
            ..Default::default()
        };
        let prog = DevProg {
            t: 8192,
            packed_prefill_only: false,
            n_counter: 0,
            insts: vec![align, moe],
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
        let tensors = (0..9)
            .map(|i| DevTensor {
                name: i.to_string(),
                bytes: 256 << 20,
                init: None,
            })
            .collect();
        (prog, tensors)
    }

    #[test]
    fn moe_aiter_routes_require_geometry_capacity_and_isolation() {
        let (prog, tensors) = fixture();
        let mut route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        route.rebase(1).unwrap();
        route.rebase(8192).unwrap();
        assert!(route.rebase(0).is_err());
        assert!(route.rebase(8193).is_err());
        for bad in 0..10 {
            let (mut p, mut t) = fixture();
            match bad {
                0 => p.packed_prefill_only = true,
                1 => p.insts[1].i[2] = 512,
                2 => p.insts[0].t[2] = 6,
                3 => p.insts[0].i[3] = 1,
                4 => t[0].bytes = 16,
                5 => p.stream[0].wait_len = 1,
                6 => p.stream[0].succ_len = 1,
                7 => p.stream[0].flags |= SE_XCTR,
                8 => p.stream.push(StreamEnt {
                    inst: 0,
                    seg: 1,
                    ..Default::default()
                }),
                _ => p.gq_stream.push(StreamEnt {
                    inst: 1,
                    seg: 0,
                    ..Default::default()
                }),
            }
            assert!(routes(&p, &t, 2).is_err(), "case {bad}");
        }
    }

    fn flat_fixture() -> (DevProg, Vec<DevTensor>) {
        let (mut prog, tensors) = fixture();
        prog.t = 8;
        prog.insts = vec![
            DevInst64 {
                op: DevOp::MoeRouterTopkPf as u16,
                t: [
                    4,
                    8,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                ],
                i: [0, 256, 8, 0, 8, 0, 0, 0],
                ..Default::default()
            },
            DevInst64 {
                op: DevOp::MoeAiterFp8Pf as u16,
                t: [0, 1, 2, 3, 4, TENSOR_NONE16, TENSOR_NONE16, TENSOR_NONE16],
                i: [8, 6144, 256, 256, 8, 0, 1, 0],
                ..Default::default()
            },
            DevInst64 {
                op: DevOp::MoeCombinePf as u16,
                t: [
                    8,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    0,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                ],
                i: [6144, 1, 8, 0, 0, 0, 0, 1],
                ..Default::default()
            },
        ];
        (prog, tensors)
    }

    #[test]
    fn flat_route_requires_raw_routing_bf16_consumer_and_protocol_space() {
        let (prog, tensors) = flat_fixture();
        let mut route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        assert_eq!(route.mode, Mode::Flat);
        assert_eq!(route.launches(), 2);
        route.rebase(3).unwrap();
        assert!(route.rebase(9).is_err());
        for bad in 0..8 {
            let (mut p, mut t) = flat_fixture();
            match bad {
                0 => p.insts[0].i[4] = 7,
                1 => p.insts[2].i[7] = 0,
                2 => p.insts[2].i[1] = 8,
                3 => p.insts[2].i[4] = 1,
                4 => t[0].bytes = 8 * 6144 * 2 + 7,
                5 => p.insts[1].t[7] = 0,
                6 => p.insts[1].i[6] = 0,
                _ => p.stream[0].succ_len = 1,
            }
            assert!(routes(&p, &t, 2).is_err(), "case {bad}");
        }
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with flat objects"]
    fn moe_aiter_flat_hsa_dispatch() {
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let kernel = MoeAiter::load(&be, Path::new(&dir), 128, true, &mut modules).unwrap();
        let upload = |bytes: &[u8]| {
            let m = EngineDevice::alloc(&be, bytes.len() as u64).unwrap();
            EngineDevice::upload(&be, &m, 0, bytes).unwrap();
            m
        };
        let weight = upload(&vec![0x38; 256 * 6144]);
        let scales: Vec<f32> = (0..256 * 96)
            .map(|i| (1 + (i / 96) % 8) as f32 * 0.001)
            .collect();
        let scale = upload(bytemuck::cast_slice(&scales));
        let wt = upload(bytemuck::cast_slice(&vec![weight.base; 256 * 3]));
        let scale_ptrs: Vec<u64> = (0..256 * 3)
            .map(|i| scale.base + (i / 3 * 96 * 4) as u64)
            .collect();
        let st = upload(bytemuck::cast_slice(&scale_ptrs));
        let out = upload(&vec![0xa5; 8 * 6144 * 2 + 8 + 512]);
        for (iteration, rows) in [8u32, 3, 2, 1, 4, 8].into_iter().enumerate() {
            let x: Vec<u16> = (0..rows as usize * 6144)
                .map(|i| [0x3f80, 0x3f00, 0x3e80][i / 6144 % 3])
                .collect();
            let mut routes = Vec::new();
            let mut expected = Vec::new();
            for row in 0..rows as usize {
                let activation = f32::from_bits(u32::from(x[row * 6144]) << 16);
                let mut sum = 0.0f32;
                for slot in 0..8 {
                    let expert = (iteration * 37 + row * 11 + slot) % 256;
                    let gate = (slot + 1) as f32 / 36.0;
                    routes.extend([expert as u32, gate.to_bits()]);
                    let scale = (expert % 8 + 1) as f32 * 0.001;
                    let gu = 6144.0 * activation * scale;
                    sum += gu * gu / (1.0 + (-gu).exp()) * 256.0 * scale * gate;
                }
                expected.push(sum);
            }
            let x = upload(bytemuck::cast_slice(&x));
            let routes = upload(bytemuck::cast_slice(&routes));
            let poison = vec![0xa5; out.len as usize];
            EngineDevice::upload(&be, &out, 0, &poison).unwrap();
            let table = [out.base, x.base, wt.base, st.base, routes.base];
            let route = Route {
                inst: DevInst64 {
                    t: [0, 1, 2, 3, 4, TENSOR_NONE16, TENSOR_NONE16, TENSOR_NONE16],
                    ..Default::default()
                },
                rows,
                mode: Mode::Flat,
            };
            kernel
                .enqueue(&be, route, bytemuck::cast_slice(&table))
                .unwrap();
            be.synchronize().unwrap();
            let mut actual = poison.clone();
            EngineDevice::download(&be, &out, 0, &mut actual).unwrap();
            let end = rows as usize * 6144 * 2;
            assert_eq!(&actual[end + 8..], &poison[end + 8..]);
            for (i, bits) in actual[..end].chunks_exact(2).enumerate() {
                let value = f32::from_bits(u32::from(u16::from_le_bytes([bits[0], bits[1]])) << 16);
                let want = expected[i / 6144];
                assert!(
                    value.is_finite() && (value / want - 1.0).abs() < 0.04,
                    "iteration={iteration} rows={rows} element={i} expected={want} actual={value}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR"]
    fn moe_aiter_hsa_dispatch() {
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let kernel = MoeAiter::load(&be, Path::new(&dir), 129, false, &mut modules).unwrap();
        let upload = |bytes: &[u8]| {
            let m = EngineDevice::alloc(&be, bytes.len() as u64).unwrap();
            EngineDevice::upload(&be, &m, 0, bytes).unwrap();
            m
        };
        let weight = upload(&vec![0x38; 256 * 6144]);
        let scales: Vec<f32> = (0..256 * 96)
            .map(|i| (1 + (i / 96) % 8) as f32 * 0.001)
            .collect();
        let scale = upload(bytemuck::cast_slice(&scales));
        let wt = upload(bytemuck::cast_slice(&vec![weight.base; 256 * 3]));
        let scale_ptrs: Vec<u64> = (0..256 * 3)
            .map(|i| scale.base + (i / 3 * 96 * 4) as u64)
            .collect();
        let st = upload(bytemuck::cast_slice(&scale_ptrs));
        for rows in [1u32, 129, 1] {
            let stride = rows.div_ceil(64) * 64;
            let mut meta = vec![0u32; 769];
            let mut rt = vec![u32::MAX; 8 * stride as usize];
            let mut rp = rt.clone();
            let mut rg = vec![0f32; rt.len()];
            for e in 0..256 {
                meta[e] = e.min(8) as u32 * stride;
                meta[e + 256] = if e < 8 { rows } else { 0 };
                meta[e + 512] = e.min(8) as u32 * stride / 64;
                if e < 8 {
                    for row in 0..rows as usize {
                        let i = e * stride as usize + row;
                        rt[i] = row as u32;
                        rp[i] = row as u32 * 8 + e as u32;
                        rg[i] = (e + 1) as f32 / 36.0;
                    }
                }
            }
            meta[768] = 8 * stride / 64;
            let x = upload(bytemuck::cast_slice(&vec![0x3f80u16; rows as usize * 6144]));
            let out = upload(bytemuck::cast_slice(&vec![f32::NAN; rows as usize * 6144]));
            let meta = upload(bytemuck::cast_slice(&meta));
            let rt = upload(bytemuck::cast_slice(&rt));
            let rp = upload(bytemuck::cast_slice(&rp));
            let rg = upload(bytemuck::cast_slice(&rg));
            let table = [
                out.base, x.base, wt.base, st.base, meta.base, rt.base, rp.base, rg.base,
            ];
            let route = Route {
                inst: DevInst64 {
                    t: [0, 1, 2, 3, 4, 5, 6, 7],
                    ..Default::default()
                },
                rows,
                mode: Mode::Sorted,
            };
            kernel
                .enqueue(&be, route, bytemuck::cast_slice(&table))
                .unwrap();
            be.synchronize().unwrap();
            let expected: f32 = (1..=8)
                .map(|e| {
                    let scale = e as f32 * 0.001;
                    let gu = 6144.0 * scale;
                    gu * gu / (1.0 + (-gu).exp()) * 256.0 * scale * e as f32 / 36.0
                })
                .sum();
            let mut actual = vec![0f32; rows as usize * 6144];
            EngineDevice::download(&be, &out, 0, bytemuck::cast_slice_mut(&mut actual)).unwrap();
            assert!(
                actual
                    .iter()
                    .all(|x| x.is_finite() && (x / expected - 1.0).abs() < 0.04),
                "rows={rows} expected={expected} first={}",
                actual[0]
            );
        }
    }
}
