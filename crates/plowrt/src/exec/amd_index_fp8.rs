use std::path::Path;

use packet::dev::{DevOp, SE_XCTR};

use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::{DeviceMem, Module};
use crate::exec::device_api::EngineDevice;
use crate::{Result, RuntimeError};

const PREPARE_HASH: &str = "5fcef6ced52c737acfb436a2e64c06fa592922d1f96bf0bc1c21ddffa4c59133";
const SCORE_HASH: &str = "b8dfada77eb90e79572ba057c31245dfa2b3cf86f6dbb5620d7c6d0f15f24182";

fn invalid(message: &str) -> RuntimeError {
    RuntimeError::Device(format!("native FP8 indexer: {message}"))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    handles: [u16; 8],
    rows: u32,
    ctx: u32,
    armed: bool,
    invalid_rows: u64,
    parked: Option<u64>,
}

impl Route {
    pub fn check_domain(&self) -> Result<()> {
        if !self.armed
            || self
                .parked
                .is_none_or(|mask| self.invalid_rows & !mask != 0)
        {
            return Err(invalid("unqualified live score domain"));
        }
        Ok(())
    }

    pub fn arm(&mut self, positions: &[u32], lengths: &[u32]) {
        self.invalid_rows = 0;
        self.armed = matches!(self.rows, 1 | 8 | 16 | 32 | 64)
            && matches!(self.ctx, 8192 | 71680 | 81920 | 131072)
            && positions
                .get(..self.rows as usize)
                .zip(lengths.get(..self.rows as usize))
                .is_some_and(|(p, n)| {
                    for (row, (&pos, &len)) in p.iter().zip(n).enumerate() {
                        if len > self.ctx || (len != 0 && pos.checked_add(1) != Some(len)) {
                            self.invalid_rows |= 1u64 << row;
                        }
                    }
                    true
                });
    }

    pub fn set_parked(&mut self, parked: &[u32]) {
        self.parked = parked
            .get(..self.rows as usize)
            .filter(|_| self.rows <= 64)
            .map(|p| {
                p.iter().enumerate().fold(0, |mask, (row, &value)| {
                    mask | (u64::from(value != 0) << row)
                })
            });
    }
}

pub(super) fn routes(
    prog: &DevProg,
    tensors: &[DevTensor],
    memory: &[DeviceMem],
    segments: usize,
) -> Result<Vec<Option<Route>>> {
    let mut result = vec![None; segments];
    for (ix, inst) in prog
        .insts
        .iter()
        .enumerate()
        .filter(|(_, d)| d.op == DevOp::IndexFp8Decode as u16)
    {
        let rows = inst.i[0];
        let ctx = inst.i[1];
        if !prog.role.is_decode_rung()
            || prog.t != rows
            || !matches!(rows, 1 | 8 | 16 | 32 | 64)
            || !(16..=131072).contains(&ctx)
            || ctx % 16 != 0
            || inst.i[2..] != [0; 6]
            || inst.fj != [0; 3]
        {
            return Err(invalid("decode geometry or reserved operands"));
        }
        let handles = inst.t;
        let r = u64::from(rows);
        let sizes = [
            r * u64::from(ctx) * 4,
            r * 8192,
            r * 256,
            r * 64,
            r * u64::from(ctx) * 132,
            r * 4,
            r * 4,
            r * 4,
        ];
        let mut ranges = [(0, 0); 8];
        for (j, (&handle, size)) in handles.iter().zip(sizes).enumerate() {
            let tensor = tensors
                .get(handle as usize)
                .ok_or_else(|| invalid("tensor handle"))?;
            let mem = memory
                .get(handle as usize)
                .ok_or_else(|| invalid("device memory handle"))?;
            if tensor.bytes < size || mem.len < tensor.bytes || mem.base == 0 || mem.base % 16 != 0
            {
                return Err(invalid("operand capacity/alignment"));
            }
            ranges[j] = (
                mem.base,
                mem.base
                    .checked_add(tensor.bytes)
                    .ok_or_else(|| invalid("address overflow"))?,
            );
            if ranges[..j]
                .iter()
                .any(|&(lo, hi)| ranges[j].0 < hi && lo < ranges[j].1)
            {
                return Err(invalid("operands alias"));
            }
        }
        let name = &tensors[handles[4] as usize].name;
        let owned = name
            .strip_prefix("kv.")
            .and_then(|s| s.strip_suffix(".kidx_fp8"))
            .is_some_and(|l| l.parse::<u32>().is_ok());
        let slot_bytes = u64::from(ctx) * 132;
        let cache_bytes = tensors[handles[4] as usize].bytes;
        if !owned
            || cache_bytes % slot_bytes != 0
            || cache_bytes / slot_bytes > 64
            || tensors[handles[5] as usize].name != "in.pos"
            || tensors[handles[6] as usize].name != "in.kvlen"
            || tensors[handles[7] as usize].name != "in.parked"
            || tensors[handles[6] as usize].bytes != cache_bytes / slot_bytes * 4
            || tensors[handles[7] as usize].bytes != cache_bytes / slot_bytes * 4
        {
            return Err(invalid("cache ownership or scalar bindings"));
        }
        let mut owner = None;
        for e in prog
            .stream
            .iter()
            .chain(&prog.gq_stream)
            .filter(|e| e.inst as usize == ix)
        {
            if e.wait_len != 0
                || e.succ_len != 0
                || e.flags & SE_XCTR != 0
                || owner.is_some_and(|s| s != e.seg as usize)
            {
                return Err(invalid("native counter obligations or multiple owners"));
            }
            owner = Some(e.seg as usize);
        }
        let seg = owner.ok_or_else(|| invalid("unowned native instruction"))?;
        if seg >= segments
            || result[seg].is_some()
            || prog
                .stream
                .iter()
                .chain(&prog.gq_stream)
                .any(|e| e.seg as usize == seg && e.inst as usize != ix)
        {
            return Err(invalid("native segment is not pure"));
        }
        result[seg] = Some(Route {
            handles,
            rows,
            ctx,
            armed: false,
            invalid_rows: 0,
            parked: None,
        });
    }
    Ok(result)
}

pub(super) struct Indexer {
    prepare: HsaKernel,
    score: HsaKernel,
    scratch: DeviceMem,
    rows: u32,
    ctx: u32,
}

impl Indexer {
    pub fn load(
        be: &HsaBackend,
        dir: &Path,
        rows: u32,
        ctx: u32,
        modules: &mut Vec<Module>,
    ) -> Result<Self> {
        if be.arch() != "gfx950"
            || be.sm_count() != 256
            || !matches!(rows, 1 | 8 | 16 | 32 | 64)
            || !(16..=131072).contains(&ctx)
            || ctx % 16 != 0
        {
            return Err(invalid("requires gfx950/256CU and aligned decode geometry"));
        }
        let mut kernels = Vec::new();
        for (file, hash, symbol, args) in [
            (
                "indexer_decode_gfx950.elf",
                PREPARE_HASH,
                "glm_indexer_decode_prepare",
                352,
            ),
            (
                "indexer_score_gfx950.elf",
                SCORE_HASH,
                "_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle",
                152,
            ),
        ] {
            let path = dir.join(file);
            let image =
                std::fs::read(&path).map_err(|e| invalid(&format!("{}: {e}", path.display())))?;
            if plow_asset::decode_objects::image_sha256(&image) != hash {
                return Err(invalid("unqualified code object"));
            }
            let module = EngineDevice::module_load(be, &image)?;
            let kernel = EngineDevice::get_function(be, &module, symbol)?;
            modules.push(module);
            if kernel.kernarg_size() != args
                || kernel.private_segment_size() != 0
                || kernel.group_segment_size() != 0
            {
                return Err(invalid("kernel ABI/resource envelope"));
            }
            kernels.push(kernel);
        }
        let table: Vec<u32> = (0..rows * (ctx / 16)).collect();
        let scratch = EngineDevice::alloc(be, u64::from(rows) * 4356 + (table.len() * 4) as u64)?;
        EngineDevice::upload(
            be,
            &scratch,
            u64::from(rows) * 4352,
            bytemuck::cast_slice(&table),
        )?;
        Ok(Self {
            prepare: kernels[0],
            score: kernels[1],
            scratch,
            rows,
            ctx,
        })
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, table: &[u8]) -> Result<()> {
        route.check_domain()?;
        if route.rows > self.rows || route.ctx != self.ctx {
            return Err(invalid("unqualified score geometry"));
        }
        let mut a = [0u64; 8];
        for (dst, handle) in a.iter_mut().zip(route.handles) {
            let offset = usize::from(handle) * 8;
            *dst = u64::from_le_bytes(
                table
                    .get(offset..offset + 8)
                    .ok_or_else(|| invalid("tensor address table"))?
                    .try_into()
                    .unwrap(),
            );
        }
        let q = self.scratch.base;
        let scales = q + u64::from(self.rows) * 4096;
        let weights = scales + u64::from(self.rows) * 128;
        let pages = q + u64::from(self.rows) * 4352;
        let effective_lengths = pages + u64::from(self.rows) * u64::from(self.ctx / 16) * 4;
        let rows = route.rows;
        let prep = [
            q,
            scales,
            weights,
            a[4],
            a[1],
            a[3],
            a[2],
            a[5],
            a[6],
            a[7],
            effective_lengths,
            u64::from(rows) | u64::from(self.ctx) << 32,
        ];
        be.launch(
            self.prepare,
            (rows * 33).div_ceil(4),
            256,
            0,
            bytemuck::cast_slice(&prep),
        )?;
        let splits = (256 / rows).div_ceil(5) * 10;
        let score = [
            u64::from(rows) | 1 << 32,
            32,
            q,
            4096 | 4096 << 32,
            128,
            a[4],
            2112,
            a[4] + 2048,
            528,
            effective_lengths,
            pages,
            weights,
            32,
            a[0],
            u64::from(self.ctx),
            u64::from(self.ctx) | u64::from(self.ctx / 16) << 32,
            u64::from(rows * (self.ctx / 16)) | u64::from(splits) << 32,
            0,
            0,
        ];
        be.launch(
            self.score,
            rows * splits,
            256,
            4096,
            bytemuck::cast_slice(&score),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::{DevInst64, StreamEnt};
    use packet::devbuild::ProgramRole;

    fn fixture() -> (DevProg, Vec<DevTensor>, Vec<DeviceMem>) {
        let prog = DevProg {
            t: 16,
            role: ProgramRole::DecodeRung { rows: 16 },
            n_counter: 0,
            insts: vec![DevInst64 {
                op: DevOp::IndexFp8Decode as u16,
                t: [0, 1, 2, 3, 4, 5, 6, 7],
                i: [16, 8192, 0, 0, 0, 0, 0, 0],
                ..Default::default()
            }],
            stream: vec![StreamEnt::default()],
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_domains: 0,
        };
        let names = [
            "act.iscore",
            "act.qidx",
            "act.kidx_normed",
            "act.widx",
            "kv.6.kidx_fp8",
            "in.pos",
            "in.kvlen",
            "in.parked",
        ];
        let sizes = [
            16 * 8192 * 4,
            16 * 8192,
            16 * 256,
            16 * 64,
            16 * 8192 * 132,
            16 * 4,
            16 * 4,
            16 * 4,
        ];
        let tensors = names
            .into_iter()
            .zip(sizes)
            .map(|(name, bytes)| DevTensor {
                name: name.into(),
                bytes,
                init: None,
            })
            .collect();
        let memory = sizes
            .into_iter()
            .enumerate()
            .map(|(i, bytes)| DeviceMem::view((i as u64 + 1) << 30, bytes))
            .collect();
        (prog, tensors, memory)
    }

    #[test]
    fn index_fp8_route_checks_geometry_ownership_aliases_and_counters() {
        let (p, t, memory) = fixture();
        assert!(routes(&p, &t, &memory, 1).unwrap()[0].is_some());
        for bad in 0..19 {
            let (mut p, mut t, mut memory) = fixture();
            match bad {
                0 => p.t = 8,
                1 => p.insts[0].i[1] = 8191,
                2 => p.insts[0].i[2] = 1,
                3 => p.insts[0].fj[0] = 1,
                4 => p.insts[0].t[7] = 0,
                5 => t[4].name = "kv.6.kidx".into(),
                6 => t[4].bytes -= 1,
                7 => memory[4].len -= 1,
                8 => memory[4].base = memory[0].base,
                9 => p.stream[0].wait_len = 1,
                10 => {
                    p.gq_stream = p.stream.clone();
                    p.gq_stream[0].seg = 1;
                }
                11 => {
                    p.insts.push(DevInst64::default());
                    p.stream.push(StreamEnt {
                        inst: 1,
                        ..Default::default()
                    });
                }
                12 => t[5].name = "act.pos".into(),
                13 => p.stream.clear(),
                14 => t[7].name = "act.parked".into(),
                15 => p.insts[0].t[7] = packet::dev::TENSOR_NONE16,
                16 => t[7].bytes -= 1,
                17 => {
                    t[4].bytes += 8192 * 132;
                    memory[4].len = t[4].bytes;
                }
                18 => {
                    t[7].bytes += 16;
                    memory[7].len = t[7].bytes;
                }
                _ => unreachable!(),
            }
            assert!(routes(&p, &t, &memory, 2).is_err(), "case {bad}");
        }
    }

    #[test]
    fn index_fp8_admission_excludes_unqualified_score_domains() {
        let (p, t, memory) = fixture();
        let mut route = routes(&p, &t, &memory, 1).unwrap()[0].unwrap();
        assert!(!route.armed);
        assert!(route.check_domain().is_err());
        route.arm(&[8191; 16], &[8192; 16]);
        assert!(route.armed);
        assert!(route.check_domain().is_err());
        route.set_parked(&[0; 64]);
        assert!(route.check_domain().is_ok());
        for (pos, len) in [
            (8190, 8192),
            (8192, 8193),
            (u32::MAX, 1),
            (u32::MAX, u32::MAX),
        ] {
            route.arm(&[pos; 16], &[len; 16]);
            assert!(route.check_domain().is_err());
        }
        route.arm(&[8191; 8], &[8192; 8]);
        assert!(!route.armed);
        for rows in [1, 8, 16, 32, 64] {
            route.rows = rows;
            for ctx in [8192, 71680, 81920, 131072] {
                route.ctx = ctx;
                for len in [
                    0,
                    1,
                    15,
                    16,
                    17,
                    255,
                    256,
                    257,
                    2047,
                    2048,
                    2049,
                    ctx - 1,
                    ctx,
                ] {
                    route.arm(
                        &vec![len.wrapping_sub(1); rows as usize],
                        &vec![len; rows as usize],
                    );
                    assert!(route.check_domain().is_ok(), "M={rows} ctx={ctx} len={len}");
                }
                route.arm(&vec![0; rows as usize], &vec![0; rows as usize]);
                assert!(route.check_domain().is_ok());
                route.arm(&vec![0; rows as usize - 1], &vec![0; rows as usize]);
                assert!(route.check_domain().is_err());
            }
        }
        route.ctx = 8191;
        route.arm(&[0; 64], &[1; 64]);
        assert!(route.check_domain().is_err());
    }

    #[test]
    fn index_fp8_parking_binds_admission_and_unparking_rechecks_rows() {
        let (p, t, memory) = fixture();
        let mut route = routes(&p, &t, &memory, 1).unwrap()[0].unwrap();
        route.arm(&[4095; 16], &[1; 16]);
        route.set_parked(&[1; 16]);
        assert!(route.check_domain().is_ok());
        route.set_parked(&[0; 16]);
        assert!(route.check_domain().is_err());
        route.arm(&[4095; 16], &[4096; 16]);
        assert!(route.check_domain().is_ok());
        route.set_parked(&[1; 15]);
        assert!(route.check_domain().is_err());
        route.rows = 64;
        let mut pos = [0; 64];
        pos[63] = 4095;
        route.arm(&pos, &[1; 64]);
        let mut parked = [0; 64];
        parked[63] = u32::MAX;
        route.set_parked(&parked);
        assert!(route.check_domain().is_ok());
        parked[63] = 0;
        route.set_parked(&parked);
        assert!(route.check_domain().is_err());
    }

    #[test]
    #[ignore = "requires queued gfx950 GPU and PLOW_TEST_AITER_DIR with pinned native-chain fixtures"]
    fn index_fp8_native_chain_reference_replay() {
        let root = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let root = Path::new(&root);
        let reference = root.join("reference");
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference.join("reference.json")).unwrap())
                .unwrap();
        assert_eq!(report["vllm_version"], "0.29.0");
        assert_eq!(report["passed"], true);
        let cases = report["cases"].as_array().unwrap();
        let sweep = report["sweep"].as_bool().unwrap_or(false);
        let headroom = report["headroom"].as_bool().unwrap_or(false);
        assert!(!headroom || sweep);
        let contexts = if headroom {
            [81920, 131072]
        } else {
            [8192, 71680]
        };
        let profiles: &[&str] = if headroom {
            &["full", "ragged", "mixed", "inactive", "headroom"]
        } else {
            &["full", "ragged", "mixed", "inactive"]
        };
        let mut coverage = std::collections::BTreeSet::new();
        for case in cases {
            let m = case["rows"].as_u64().unwrap();
            let ctx = case["ctx"].as_u64().unwrap();
            let profile = case["profile"].as_str().unwrap_or("full");
            assert!(coverage.insert((m, ctx, profile)));
        }
        let expected_coverage: std::collections::BTreeSet<_> = if sweep {
            [1, 8, 16, 32, 64]
                .into_iter()
                .flat_map(|m| {
                    contexts.into_iter().flat_map(move |ctx| {
                        profiles
                            .iter()
                            .copied()
                            .map(move |profile| (m, ctx, profile))
                    })
                })
                .collect()
        } else {
            [(16, 8192, "full"), (16, 71680, "full")].into()
        };
        assert_eq!(coverage, expected_coverage);
        let be = HsaBackend::new(0).unwrap();
        for case in cases {
            let bytes = std::fs::read(reference.join(case["file"].as_str().unwrap())).unwrap();
            assert_eq!(
                plow_asset::decode_objects::image_sha256(&bytes),
                case["sha256"].as_str().unwrap()
            );
            assert_eq!(case["repeat_bitwise"], true);
            let ctx = case["ctx"].as_u64().unwrap() as usize;
            let m = case["rows"].as_u64().unwrap() as usize;
            let header: Vec<u32> = bytes[..16]
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(header, [0x494e4431, m as u32, ctx as u32, 0]);
            let sizes = [
                m * ctx * 4,
                m * 8192,
                m * 256,
                m * 64,
                m * ctx * 132,
                m * 4,
                m * 4,
                m * 4,
            ];
            assert_eq!(
                bytes.len(),
                16 + m * 8520 + 2 * sizes[4] + m * 4352 + sizes[0]
            );
            let mut inputs: Vec<&[u8]> = vec![&[]; 7];
            let mut at = 16;
            for handle in [1, 3, 2, 5, 6, 4] {
                inputs[handle] = &bytes[at..at + sizes[handle]];
                at += sizes[handle];
            }
            assert_eq!(
                plow_asset::decode_objects::image_sha256(&bytes[at..]),
                case["expected_sha256"].as_str().unwrap()
            );
            let expected_prep = &bytes[at..at + m * 4352];
            at += expected_prep.len();
            let expected_cache = &bytes[at..at + sizes[4]];
            at += expected_cache.len();
            let expected_scores = &bytes[at..];
            let (mut prog, mut tensors, _) = fixture();
            prog.t = m as u32;
            prog.role = ProgramRole::DecodeRung { rows: m as u32 };
            prog.insts[0].i[0] = m as u32;
            prog.insts[0].i[1] = ctx as u32;
            for (tensor, &size) in tensors.iter_mut().zip(&sizes) {
                tensor.bytes = size as u64;
            }
            let memory: Vec<_> = sizes
                .iter()
                .map(|&size| EngineDevice::alloc(&be, size as u64 + 256).unwrap())
                .collect();
            for handle in 1..7 {
                EngineDevice::upload(&be, &memory[handle], 0, inputs[handle]).unwrap();
            }
            let positions: Vec<_> = inputs[5]
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            let lengths: Vec<_> = inputs[6]
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            let mut route = routes(&prog, &tensors, &memory, 1).unwrap()[0].unwrap();
            route.arm(&positions, &lengths);
            if sweep {
                assert_eq!(case["lengths"], serde_json::json!(lengths));
                let boundaries = [
                    ctx as u32 - 1,
                    1,
                    15,
                    16,
                    17,
                    255,
                    256,
                    257,
                    2047,
                    2048,
                    2049,
                    ctx as u32 / 2,
                ];
                let profile = case["profile"].as_str().unwrap();
                let expected_lengths: Vec<_> = (0..m)
                    .map(|row| match profile {
                        "full" => ctx as u32,
                        "inactive" => 0,
                        "mixed" if row % 3 == 0 => 0,
                        "headroom" if headroom => {
                            [8192, 8193, 8447, 8448, 71680, 71681, 71935, 71936][row % 8]
                        }
                        "ragged" | "mixed" => boundaries[row % boundaries.len()],
                        _ => panic!("unsupported reference profile"),
                    })
                    .collect();
                assert_eq!(lengths, expected_lengths);
                for (&pos, &len) in positions.iter().zip(&lengths) {
                    assert!(len <= ctx as u32);
                    assert_eq!(pos, len.wrapping_sub(1));
                }
            }
            assert!(route.armed);
            let mut modules = Vec::new();
            let kernel = Indexer::load(&be, root, 64, ctx as u32, &mut modules).unwrap();
            let table: Vec<_> = memory.iter().map(|mem| mem.base).collect();
            for (poison, parked_length) in [(0x55, 0), (0xaa, 1), (0x55, ctx as u32)] {
                let mut live_positions = positions.clone();
                let mut staged_lengths = lengths.clone();
                let mut parked = vec![0u32; m];
                for row in 0..m {
                    if lengths[row] == 0 && parked_length != 0 {
                        parked[row] = if row % 2 == 0 { 1 } else { u32::MAX };
                        staged_lengths[row] = parked_length;
                        live_positions[row] = parked_length - 1;
                    }
                }
                route.arm(&live_positions, &staged_lengths);
                route.set_parked(&parked);
                assert!(route.check_domain().is_ok());
                EngineDevice::upload(&be, &memory[5], 0, bytemuck::cast_slice(&live_positions))
                    .unwrap();
                EngineDevice::upload(&be, &memory[6], 0, bytemuck::cast_slice(&staged_lengths))
                    .unwrap();
                EngineDevice::upload(&be, &memory[7], 0, bytemuck::cast_slice(&parked)).unwrap();
                let effective_offset = 64 * 4352 + 64 * (ctx / 16) * 4;
                EngineDevice::upload(
                    &be,
                    &kernel.scratch,
                    effective_offset as u64,
                    &vec![poison; 64 * 4],
                )
                .unwrap();
                EngineDevice::upload(&be, &kernel.scratch, 0, &vec![poison; 64 * 4352]).unwrap();
                EngineDevice::upload(&be, &memory[0], 0, &vec![poison; sizes[0]]).unwrap();
                EngineDevice::upload(&be, &memory[4], 0, inputs[4]).unwrap();
                for (mem, &size) in memory.iter().zip(&sizes) {
                    EngineDevice::upload(&be, mem, size as u64, &[poison; 256]).unwrap();
                }
                be.begin_dispatch_chain(2).unwrap();
                kernel
                    .enqueue(&be, route, bytemuck::cast_slice(&table))
                    .unwrap();
                be.commit_dispatch_chain().unwrap();
                be.synchronize().unwrap();
                let mut effective = vec![0; m * 4];
                EngineDevice::download(
                    &be,
                    &kernel.scratch,
                    effective_offset as u64,
                    &mut effective,
                )
                .unwrap();
                assert_eq!(effective, bytemuck::cast_slice::<u32, u8>(&lengths));
                for (offset, expected) in [
                    (0, &expected_prep[..m * 4096]),
                    (64 * 4096, &expected_prep[m * 4096..m * 4224]),
                    (64 * 4224, &expected_prep[m * 4224..]),
                ] {
                    let mut actual = vec![0; expected.len()];
                    EngineDevice::download(&be, &kernel.scratch, offset, &mut actual).unwrap();
                    assert_eq!(
                        actual.iter().zip(expected).position(|(a, b)| a != b),
                        None,
                        "M={m} ctx={ctx} poison={poison} preparation offset={offset}"
                    );
                }
                for (name, mem, expected) in [
                    ("cache", &memory[4], expected_cache),
                    ("scores", &memory[0], expected_scores),
                ] {
                    let mut actual = vec![0; expected.len()];
                    EngineDevice::download(&be, mem, 0, &mut actual).unwrap();
                    if name == "scores" {
                        for (row, &len) in lengths.iter().enumerate() {
                            let tail = (row * ctx + len as usize) * 4..(row + 1) * ctx * 4;
                            assert!(expected[tail.clone()].iter().all(|&v| v == 0));
                            actual[tail].fill(0);
                        }
                    }
                    if actual != expected {
                        let first = actual
                            .iter()
                            .zip(expected)
                            .position(|(a, b)| a != b)
                            .unwrap();
                        panic!(
                            "M={m} ctx={ctx} poison={poison} {name}: first differing byte {first}"
                        );
                    }
                }
                for (mem, &size) in memory.iter().zip(&sizes) {
                    let mut guard = [0; 256];
                    EngineDevice::download(&be, mem, size as u64, &mut guard).unwrap();
                    assert_eq!(guard, [poison; 256], "operand end guard");
                }
                eprintln!(
                    "native indexer M={m} ctx={ctx} profile={} poison={poison} parked_length={parked_length}: preparation/cache/live scores bitwise, guards intact",
                    case["profile"]
                );
            }
        }
    }
}
