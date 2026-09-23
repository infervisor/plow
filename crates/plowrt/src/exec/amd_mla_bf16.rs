use std::path::Path;

use packet::dev::{DevOp, SE_XCTR, TENSOR_NONE16};

use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::{DeviceMem, Module};
use crate::exec::device_api::EngineDevice;
use crate::{Result, RuntimeError};

const STAGE: &str = "mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps.co";
const METADATA: &str = "mla_metadata_gfx950.elf";
const REDUCE: &str = "mla_reduce_gfx950.elf";
const ADAPTER: &str = "mla_sparse_adapter_gfx950.elf";

fn invalid(message: &str) -> RuntimeError {
    RuntimeError::Device(format!("BF16 persistent MLA: {message}"))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    handles: [u16; 7],
    rows: u32,
    ctx: u32,
    max_live: u32,
}

impl Route {
    pub fn arm(&mut self, lengths: &[u32]) {
        self.max_live = lengths
            .get(..self.rows as usize)
            .filter(|v| v.iter().all(|&n| n > 0 && n <= self.ctx))
            .and_then(|v| v.iter().max().copied())
            .unwrap_or(0);
    }
}

pub(super) fn pair_index(prog: &DevProg, seg: usize) -> Result<Option<[usize; 2]>> {
    let mut pair = [usize::MAX; 2];
    for e in prog
        .stream
        .iter()
        .chain(&prog.gq_stream)
        .filter(|e| e.seg as usize == seg)
    {
        let Some(inst) = prog.insts.get(e.inst as usize) else {
            return Err(invalid("instruction index"));
        };
        let slot = match DevOp::from_u16(inst.op) {
            Some(DevOp::FlashMlaDecode | DevOp::FlashGatherDecode) => 0,
            Some(DevOp::FlashMerge) => 1,
            _ => return Ok(None),
        };
        if pair[slot] != usize::MAX && pair[slot] != e.inst as usize {
            return Ok(None);
        }
        pair[slot] = e.inst as usize;
    }
    if pair.contains(&usize::MAX) {
        return Ok(None);
    }
    if pair[1] != pair[0] + 1 {
        return Err(invalid("nonadjacent flash/merge"));
    }
    for e in prog
        .stream
        .iter()
        .chain(&prog.gq_stream)
        .filter(|e| pair.contains(&(e.inst as usize)))
    {
        if e.seg as usize != seg || e.wait_len != 0 || e.succ_len != 0 || e.flags & SE_XCTR != 0 {
            return Err(invalid(
                "raw pair has counter obligations or multiple owners",
            ));
        }
    }
    Ok(Some(pair))
}

pub(super) fn route(prog: &DevProg, tensors: &[DevTensor], seg: usize) -> Result<Option<Route>> {
    let Some(pair) = pair_index(prog, seg)? else {
        return Ok(None);
    };
    let f = &prog.insts[pair[0]];
    let merge = &prog.insts[pair[1]];
    let sparse = f.op == DevOp::FlashGatherDecode as u16;
    if !(1..32).contains(&prog.t)
        || f.i[0] != prog.t
        || f.i[1] != 8
        || !(1..=131072).contains(&f.i[2])
        || f.i[3] != 0
        || f.i[4] == 0
        || f.i[5] != u32::MAX
        || f.fj != [0.0625f32.to_bits(), 0, 0]
        || (sparse && (f.i[6] != 2048 || f.t[7] == TENSOR_NONE16))
        || (!sparse && (f.i[2] > 2048 || f.t[7] != TENSOR_NONE16 || f.i[6] != 0))
        || merge.i[..4] != [prog.t, 8, f.i[4], 512]
        || merge.t[1..3] != f.t[..2]
    {
        return Err(invalid(
            "requires QH8 latent512/rope64, BF16, top2048 or short dense, M<32",
        ));
    }
    let handles = [merge.t[0], f.t[2], f.t[3], f.t[4], f.t[5], f.t[6], f.t[7]];
    let rows = u64::from(prog.t);
    let sizes = [
        rows * 8 * 512 * 2,
        rows * 8 * 512 * 2,
        rows * 8 * 64 * 2,
        rows * u64::from(f.i[2]) * 512 * 2,
        rows * u64::from(f.i[2]) * 64 * 2,
        rows * 4,
        if sparse { rows * 2048 * 4 } else { 0 },
    ];
    for (&handle, bytes) in handles.iter().zip(sizes) {
        if bytes != 0
            && (handle == TENSOR_NONE16
                || tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes))
        {
            return Err(invalid("operand capacity"));
        }
    }
    if handles[1..7].contains(&handles[0]) {
        return Err(invalid("output aliases an input"));
    }
    Ok(Some(Route {
        handles,
        rows: prog.t,
        ctx: f.i[2],
        max_live: 0,
    }))
}

fn load<'a>(
    be: &HsaBackend,
    dir: &Path,
    file: &str,
    hash: &str,
    modules: &'a mut Vec<Module>,
) -> Result<&'a Module> {
    let path = dir.join(file);
    let mut image =
        std::fs::read(&path).map_err(|e| invalid(&format!("{}: {e}", path.display())))?;
    if plow_asset::decode_objects::image_sha256(&image) != hash {
        return Err(invalid(&format!("unqualified object {file}")));
    }
    if file == STAGE {
        if image.get(0x1708..0x170c) != Some(&[0, 0, 0, 0]) {
            return Err(invalid("stage descriptor"));
        }
        image[0x1708..0x170c].copy_from_slice(&384u32.to_le_bytes());
    }
    let module = EngineDevice::module_load(be, &image)?;
    modules.push(module);
    Ok(modules.last().unwrap())
}

fn kernel(
    be: &HsaBackend,
    module: &Module,
    symbol: &str,
    args: u32,
    lds: u32,
) -> Result<HsaKernel> {
    let k = EngineDevice::get_function(be, module, symbol)?;
    if k.kernarg_size() != args || k.private_segment_size() != 0 || k.group_segment_size() != lds {
        return Err(invalid(&format!("resource ABI for {symbol}")));
    }
    Ok(k)
}

pub(super) struct PersistentMla {
    pack: HsaKernel,
    metadata: HsaKernel,
    stage: HsaKernel,
    reduce: HsaKernel,
    unpad: HsaKernel,
    _scratch: DeviceMem,
    ptr: [u64; 14],
    rows_cap: u32,
}

impl PersistentMla {
    pub fn load(be: &HsaBackend, dir: &Path, rows: u32, modules: &mut Vec<Module>) -> Result<Self> {
        if be.arch() != "gfx950"
            || be.sm_count() != 256
            || be.lds_bytes() != 163840
            || !(1..32).contains(&rows)
        {
            return Err(invalid("requires gfx950/256CU/160KiB LDS and M<32"));
        }
        let m = load(
            be,
            dir,
            STAGE,
            "b6d4181c3ed19750b22a02dc0d290727272ed53091678c5e75c39f45d9832cfd",
            modules,
        )?;
        let stage = kernel(
            be,
            &m,
            "_ZN5aiter41mla_a16w16_qh64_qseqlen1_gqaratio64_v3_psE",
            384,
            163840,
        )?;
        let m = load(
            be,
            dir,
            ADAPTER,
            "ed1426279c1b8228334315570830fac9bd57d4c43adc3066466afcda0649a35b",
            modules,
        )?;
        let pack = kernel(be, &m, "plow_mla_bf16_pack", 368, 260)?;
        let unpad = kernel(be, &m, "plow_mla_bf16_unpad", 280, 0)?;
        let m = load(
            be,
            dir,
            METADATA,
            "3ade36a825eb249bc7dd8168e2338119173c595f1a1a427622733b73d9bd4081",
            modules,
        )?;
        let metadata = kernel(be, &m, "_Z33kn_get_mla_metadata_v1_2_parallelI20MlaMetadataV12TraitsILi128ELb0ELi1ELb1ELb0EEEv28MlaMetadataV1KernelParameter", 392, 0)?;
        let m = load(
            be,
            dir,
            REDUCE,
            "401e7dd9c9714650361a87bba36b216e5b491d90fa10e8fc9cda72712e62f383",
            modules,
        )?;
        let reduce = kernel(be, &m, "_Z16kn_mla_reduce_v1I23MlaReduceKernelV1TraitsILi512ELi16ELi1EEfDF16bEv23MlaReduceKernelV1Params24MlaReduceKernelV1Configs", 84, 0)?;
        let r = u64::from(rows);
        let cap = 256 + r - 1;
        let sizes = [
            r * 16 * 576 * 2,
            r * 2048 * 576 * 2,
            r * 2048 * 4,
            (r + 1) * 4,
            (r + 1) * 4,
            r * 4,
            80,
            257 * 4,
            cap * 8 * 4,
            (r + 1) * 4,
            r * 2 * 4,
            cap * 4,
            cap * 16 * 513 * 4,
            r * 16 * 512 * 2,
        ];
        let scratch = EngineDevice::alloc(be, sizes.iter().map(|n| n.div_ceil(256) * 256).sum())?;
        let mut at = scratch.base;
        let ptr = sizes.map(|n| {
            let p = at;
            at += n.div_ceil(256) * 256;
            p
        });
        Ok(Self {
            pack,
            metadata,
            stage,
            reduce,
            unpad,
            _scratch: scratch,
            ptr,
            rows_cap: rows,
        })
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, table: &[u8]) -> Result<()> {
        if route.rows > self.rows_cap || route.max_live == 0 {
            return Err(invalid("unarmed or invalid live rows"));
        }
        let mut a = [0u64; 7];
        for (dst, &handle) in a.iter_mut().zip(&route.handles) {
            if handle == TENSOR_NONE16 {
                continue;
            }
            let offset = usize::from(handle) * 8;
            *dst = u64::from_le_bytes(
                table
                    .get(offset..offset + 8)
                    .ok_or_else(|| invalid("tensor table"))?
                    .try_into()
                    .unwrap(),
            );
        }
        let [q, kv, kvi, qp, kp, last, metadata, workptr, info, reduceptr, finalmap, partialmap, part, out] =
            self.ptr;
        let lse = part + u64::from(256 + self.rows_cap - 1) * 16 * 512 * 4;
        let r = route.rows;
        let mut pack = [
            q, kv, kvi, a[1], a[2], a[3], a[4], a[6], a[5], qp, kp, last, 0, 0,
        ];
        pack[12] = u64::from(r) | u64::from(route.ctx) << 32;
        pack[13] = if route.handles[6] == TENSOR_NONE16 {
            0
        } else {
            2048
        };
        be.launch(self.pack, 256, 256, 0, bytemuck::cast_slice(&pack))?;
        let mut meta = [0u32; 34];
        for (slot, ptr) in meta.chunks_exact_mut(2).zip([
            metadata, workptr, info, reduceptr, finalmap, partialmap, qp, kp, last,
        ]) {
            slot[0] = ptr as u32;
            slot[1] = (ptr >> 32) as u32;
        }
        let splits = ((route.max_live.min(2048) / 128).max(1).next_power_of_two() * r).min(256);
        meta[18..31].copy_from_slice(&[r, 0, 16, 256, r + 1, 1, 16, 4, 1, 1, u32::MAX, 1, splits]);
        meta[31..].copy_from_slice(&[1, 16, 1]);
        be.launch(self.metadata, 1, 512, 163840, bytemuck::cast_slice(&meta))?;
        let mut stage = [0u64; 48];
        for (i, value) in [
            (0, part),
            (2, lse),
            (4, q),
            (6, kv),
            (8, kp),
            (10, kvi),
            (12, last),
            (14, u64::from(0.0625f32.to_bits())),
            (16, 16),
            (18, 1),
            (20, 16 * 576 * 2),
            (22, 576 * 2),
            (26, qp),
            (28, metadata),
            (30, out),
            (36, 1),
            (42, 1),
        ] {
            stage[i] = value;
        }
        be.launch(self.stage, 256, 256, 0, bytemuck::cast_slice(&stage))?;
        let mut reduce = [0u32; 21];
        for (slot, ptr) in reduce
            .chunks_exact_mut(2)
            .zip([reduceptr, finalmap, partialmap, 0, out, lse, part])
        {
            slot[0] = ptr as u32;
            slot[1] = (ptr >> 32) as u32;
        }
        reduce[14..].copy_from_slice(&[16 * 512, 512, 256, r, 256, 0, 256]);
        be.launch_3d_lds(
            self.reduce,
            [16, 1, r],
            128,
            2048,
            bytemuck::cast_slice(&reduce),
        )?;
        be.launch(
            self.unpad,
            16,
            256,
            0,
            bytemuck::cast_slice(&[a[0], out, u64::from(r)]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::{DevInst64, StreamEnt};

    fn fixture(rows: u32, ctx: u32) -> (DevProg, Vec<DevTensor>) {
        let f = DevInst64 {
            op: DevOp::FlashGatherDecode as u16,
            t: [7, 8, 1, 2, 3, 4, 5, 6],
            i: [rows, 8, ctx, 0, 8, u32::MAX, 2048, 4],
            fj: [0.0625f32.to_bits(), 0, 0],
            ..Default::default()
        };
        let merge = DevInst64 {
            op: DevOp::FlashMerge as u16,
            t: [
                0,
                7,
                8,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
            ],
            i: [rows, 8, 8, 512, 0, 0, 0, 0],
            ..Default::default()
        };
        (
            DevProg {
                t: rows,
                role: packet::devbuild::ProgramRole::DecodeRung { rows },
                n_counter: 0,
                insts: vec![f, merge],
                stream: vec![
                    StreamEnt {
                        inst: 0,
                        ..Default::default()
                    },
                    StreamEnt {
                        inst: 1,
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
            },
            (0..9)
                .map(|i| DevTensor {
                    name: i.to_string(),
                    bytes: 4 << 30,
                    init: None,
                })
                .collect(),
        )
    }

    #[test]
    fn bf16_route_guards_geometry_counters_capacity_and_live_lengths() {
        for rows in [1, 8, 16, 31] {
            let (p, t) = fixture(rows, 71680);
            let mut r = route(&p, &t, 0).unwrap().unwrap();
            r.arm(&vec![512; rows as usize]);
            assert_eq!(r.max_live, 512);
            for invalid in [0, 71681] {
                r.arm(&vec![invalid; rows as usize]);
                assert_eq!(r.max_live, 0);
            }
        }
        for bad in 0..10 {
            let (mut p, mut t) = fixture(16, 8192);
            match bad {
                0 => p.t = 32,
                1 => p.insts[0].i[1] = 16,
                2 => p.insts[0].i[6] = 1024,
                3 => p.insts[0].fj[0] = 0.125f32.to_bits(),
                4 => p.insts[1].t[1] = 0,
                5 => t[3].bytes = 1,
                6 => p.stream[0].wait_len = 1,
                7 => {
                    p.gq_stream = p.stream.clone();
                    p.gq_stream[0].flags = SE_XCTR;
                }
                8 => {
                    p.gq_stream = p.stream.clone();
                    p.gq_stream[1].seg = 1;
                }
                9 => p.insts[1].t[0] = 1,
                _ => unreachable!(),
            }
            assert!(route(&p, &t, 0).is_err(), "case {bad}");
        }
        let (mut p, t) = fixture(8, 512);
        p.insts[0].op = DevOp::FlashMlaDecode as u16;
        p.insts[0].t[7] = TENSOR_NONE16;
        p.insts[0].i[6] = 0;
        assert!(route(&p, &t, 0).unwrap().is_some());
        p.insts[0].i[2] = 8192;
        assert!(route(&p, &t, 0).is_err());
        p.insts.push(DevInst64::default());
        p.stream.push(StreamEnt {
            inst: 2,
            ..Default::default()
        });
        assert!(route(&p, &t, 0).unwrap().is_none());
    }

    #[test]
    #[ignore = "requires queued gfx950 GPU and PLOW_TEST_AITER_DIR with pinned objects/reference fixtures"]
    fn bf16_persistent_hsa_reference_replay() {
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let dir = Path::new(&dir);
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let kernel = PersistentMla::load(&be, dir, 31, &mut modules).unwrap();
        let report: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("reference/attention-comparison.json")).unwrap(),
        )
        .unwrap();
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 16);
        for case in cases {
            let bytes =
                std::fs::read(dir.join("reference").join(case["file"].as_str().unwrap())).unwrap();
            assert_eq!(
                plow_asset::decode_objects::image_sha256(&bytes),
                case["sha256"].as_str().unwrap()
            );
            let word =
                |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
            let (m, ctx, nw, cap) = (word(1), word(2), word(4), word(6));
            let ragged = word(0) == 0x41505332;
            assert!(ragged || word(0) == 0x41505331);
            let lengths: Vec<u32> = (0..m)
                .map(|b| {
                    if ragged {
                        word(7 + b) as u32
                    } else {
                        ctx as u32
                    }
                })
                .collect();
            let mut at = 28 + if ragged { m * 4 } else { 0 };
            let q = &bytes[at..at + m * 16 * 576 * 2];
            at += q.len();
            let kv = &bytes[at..at + m * ctx * 576 * 2];
            at += kv.len();
            let total: usize = lengths.iter().map(|&n| n.min(2048) as usize).sum();
            let indices = &bytes[at..at + total * 4];
            at += indices.len() + 257 * 4 + cap * 8 * 4;
            let part = &bytes[at..at + nw * 16 * 512 * 4];
            at += part.len();
            let lse = &bytes[at..];
            assert_eq!(lse.len(), nw * 16 * 4);
            let mut inputs = [
                vec![255; m * 8 * 512 * 2],
                vec![0; m * 8 * 512 * 2],
                vec![0; m * 8 * 64 * 2],
                vec![0; m * ctx * 512 * 2],
                vec![0; m * ctx * 64 * 2],
                bytemuck::cast_slice(&lengths).to_vec(),
                vec![255; m * 2048 * 4],
            ];
            for row in 0..m * 8 {
                let src = row * 2 * 576 * 2;
                assert_eq!(&q[src..src + 1152], &q[src + 1152..src + 2304]);
                inputs[1][row * 1024..(row + 1) * 1024].copy_from_slice(&q[src..src + 1024]);
                inputs[2][row * 128..(row + 1) * 128].copy_from_slice(&q[src + 1024..src + 1152]);
            }
            for (row, src) in kv.chunks_exact(1152).enumerate() {
                inputs[3][row * 1024..(row + 1) * 1024].copy_from_slice(&src[..1024]);
                inputs[4][row * 128..(row + 1) * 128].copy_from_slice(&src[1024..]);
            }
            let mut at = 0;
            for (b, &len) in lengths.iter().enumerate() {
                for j in 0..len.min(2048) as usize {
                    let idx = u32::from_le_bytes(indices[at..at + 4].try_into().unwrap())
                        - (b * ctx) as u32;
                    assert!(idx < len);
                    let off = (b * 2048 + j) * 4;
                    inputs[6][off..off + 4].copy_from_slice(&idx.to_le_bytes());
                    at += 4;
                }
            }
            let reduction = std::fs::read(
                dir.join("reference")
                    .join(case["reduce_file"].as_str().unwrap()),
            )
            .unwrap();
            assert_eq!(
                plow_asset::decode_objects::image_sha256(&reduction),
                case["reduce_sha256"].as_str().unwrap()
            );
            let padded = &reduction[reduction.len() - m * 16 * 512 * 2..];
            let expected: Vec<u8> = padded
                .chunks_exact(2048)
                .flat_map(|v| v[..1024].iter().copied())
                .collect();
            let devices: Vec<_> = inputs
                .iter()
                .map(|v| {
                    let d = EngineDevice::alloc(&be, v.len() as u64).unwrap();
                    be.memcpy_htod(d.base, v).unwrap();
                    d
                })
                .collect();
            let table: Vec<u64> = devices.iter().map(|d| d.base).collect();
            let poison = vec![255; kernel._scratch.len as usize];
            for dense in [false, true] {
                if dense && ctx > 2048 {
                    continue;
                }
                let (mut p, t) = fixture(m as u32, ctx as u32);
                if dense {
                    // Preserve the reference's key order while replacing selection with identity.
                    for (handle, width) in [(3, 1024), (4, 128)] {
                        let mut reordered = inputs[handle].clone();
                        for (b, &len) in lengths.iter().enumerate() {
                            for j in 0..len as usize {
                                let off = (b * 2048 + j) * 4;
                                let key =
                                    u32::from_le_bytes(inputs[6][off..off + 4].try_into().unwrap())
                                        as usize;
                                let src = (b * ctx + key) * width;
                                let dst = (b * ctx + j) * width;
                                reordered[dst..dst + width]
                                    .copy_from_slice(&inputs[handle][src..src + width]);
                            }
                        }
                        be.memcpy_htod(devices[handle].base, &reordered).unwrap();
                    }
                    p.insts[0].op = DevOp::FlashMlaDecode as u16;
                    p.insts[0].t[7] = TENSOR_NONE16;
                    p.insts[0].i[6] = 0;
                }
                let mut r = route(&p, &t, 0).unwrap().unwrap();
                r.arm(&lengths);
                for _ in 0..2 {
                    be.memcpy_htod(kernel._scratch.base, &poison).unwrap();
                    be.memcpy_htod(devices[0].base, &inputs[0]).unwrap();
                    be.begin_dispatch_chain(5).unwrap();
                    kernel
                        .enqueue(&be, r, bytemuck::cast_slice(&table))
                        .unwrap();
                    be.commit_dispatch_chain().unwrap();
                    be.synchronize().unwrap();
                    let mut output = vec![0; expected.len()];
                    EngineDevice::download(&be, &devices[0], 0, &mut output).unwrap();
                    assert_eq!(
                        output, expected,
                        "{} dense={dense} final BF16",
                        case["file"]
                    );
                    let offset = kernel.ptr[12] - kernel._scratch.base;
                    let mut actual = vec![0; part.len()];
                    EngineDevice::download(&be, &kernel._scratch, offset, &mut actual).unwrap();
                    assert_eq!(actual, part, "stage partials");
                    let mut actual = vec![0; lse.len()];
                    EngineDevice::download(
                        &be,
                        &kernel._scratch,
                        offset + (256 + 31 - 1) * 16 * 512 * 4,
                        &mut actual,
                    )
                    .unwrap();
                    assert_eq!(actual, lse, "stage LSE");
                }
                println!(
                    "BF16 Rust AQL chain bitwise PASS {} dense={dense}",
                    case["file"]
                );
            }
        }
    }
}
