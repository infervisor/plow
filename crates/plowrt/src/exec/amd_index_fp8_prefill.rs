use std::path::Path;

use packet::dev::{DevOp, SE_XCTR, TENSOR_NONE16};

use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::{DeviceMem, Module};
use crate::exec::device_api::EngineDevice;
use crate::{Result, RuntimeError};

fn invalid(message: &str) -> RuntimeError {
    RuntimeError::Device(format!("native FP8 prefill indexer: {message}"))
}

fn supported_rows(rows: u32) -> bool {
    matches!(rows, 128 | 512 | 1024 | 2048 | 4096 | 8192)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    handles: [u16; 8],
    addresses: [u64; 8],
    rows: u32,
    ctx: u32,
    slots: u32,
    append_only: bool,
    chunk: Option<(u32, u32)>,
}

impl Route {
    pub fn arm_for_program(&mut self, base: u32, live: u32, executed_rows: u32) -> Result<()> {
        self.chunk = None;
        if executed_rows != live {
            return Err(invalid(
                "prefill consumers must use the live chunk row count",
            ));
        }
        self.arm(base, live)
    }

    fn arm(&mut self, base: u32, live: u32) -> Result<()> {
        self.chunk = None;
        if live > self.rows
            || base
                .checked_add(live)
                .is_none_or(|n| n == 0 || n > self.ctx)
        {
            return Err(invalid("chunk exceeds capacity"));
        }
        self.chunk = Some((base, live));
        Ok(())
    }

    pub fn check_domain(&self) -> Result<()> {
        if self.chunk.is_none() || !supported_rows(self.rows) || self.ctx != 131072 {
            return Err(invalid("unqualified prefill shape or unstaged chunk"));
        }
        Ok(())
    }

    pub fn launches(&self) -> usize {
        if self.append_only {
            1
        } else {
            1 + self
                .chunk
                .map_or(self.rows, |(_, live)| live)
                .div_ceil(self.score_chunk_rows()) as usize
        }
    }

    fn large_scores(&self) -> bool {
        self.chunk
            .is_some_and(|(base, live)| u64::from(live) * u64::from(base + live) * 4 >= (1 << 31))
    }

    fn score_chunk_rows(&self) -> u32 {
        if self.large_scores() {
            self.rows
        } else {
            ((1u64 << 31) / (u64::from(self.ctx) * 4)) as u32
        }
    }

    fn bindings(&self, table: &[u8]) -> Result<[u64; 8]> {
        self.check_domain()?;
        let mut addresses = [0; 8];
        for (i, &h) in self.handles.iter().enumerate() {
            if h == TENSOR_NONE16 {
                continue;
            }
            let offset = usize::from(h) * 8;
            let address = u64::from_le_bytes(
                table
                    .get(offset..offset + 8)
                    .ok_or_else(|| invalid("tensor address table"))?
                    .try_into()
                    .unwrap(),
            );
            if i == 4 {
                let slot_bytes = u64::from(self.ctx) * 132;
                let delta = address
                    .checked_sub(self.addresses[i])
                    .ok_or_else(|| invalid("cache rebase"))?;
                if delta % slot_bytes != 0 || delta / slot_bytes >= u64::from(self.slots) {
                    return Err(invalid("cache rebase is outside an owned slot"));
                }
            } else if address != self.addresses[i] {
                return Err(invalid("unexpected operand rebinding"));
            }
            addresses[i] = address;
        }
        Ok(addresses)
    }
}

pub(super) fn routes(
    prog: &DevProg,
    tensors: &[DevTensor],
    memory: &[DeviceMem],
    segments: usize,
) -> Result<Vec<Option<Route>>> {
    let mut routes = vec![None; segments];
    for (ix, inst) in prog
        .insts
        .iter()
        .enumerate()
        .filter(|(_, d)| d.op == DevOp::IndexFp8Prefill as u16)
    {
        let rows = inst.i[0];
        let ctx = inst.i[1];
        let append_only = inst.i[2] == 1;
        if !prog.role.is_prefill_bucket()
            || prog.t != rows
            || !supported_rows(rows)
            || ctx != 131072
            || inst.i[2] > 1
            || inst.i[3..] != [0; 5]
            || inst.fj != [0; 3]
            || inst.t[7] != TENSOR_NONE16
            || (append_only && [inst.t[0], inst.t[1], inst.t[3]] != [TENSOR_NONE16; 3])
        {
            return Err(invalid("prefill role, geometry or reserved operands"));
        }
        let r = u64::from(rows);
        let sizes = [
            r * u64::from(ctx) * 4,
            r * 8192,
            r * 256,
            r * 64,
            u64::from(ctx) * 132,
            u64::from(ctx) * 4,
            4,
        ];
        let mut addresses = [0; 8];
        let mut ranges = [(0, 0); 8];
        for (i, (&h, size)) in inst.t.iter().zip(sizes).enumerate() {
            if append_only && matches!(i, 0 | 1 | 3) {
                continue;
            }
            let tensor = tensors
                .get(h as usize)
                .ok_or_else(|| invalid("tensor handle"))?;
            let mem = memory
                .get(h as usize)
                .ok_or_else(|| invalid("memory handle"))?;
            if tensor.bytes < size || mem.len < tensor.bytes || mem.base == 0 || mem.base % 16 != 0
            {
                return Err(invalid("operand capacity/alignment"));
            }
            let end = mem
                .base
                .checked_add(tensor.bytes)
                .ok_or_else(|| invalid("address overflow"))?;
            if ranges[..i]
                .iter()
                .any(|&(lo, hi)| mem.base < hi && lo < end)
            {
                return Err(invalid("operands alias"));
            }
            addresses[i] = mem.base;
            ranges[i] = (mem.base, end);
        }
        let cache = &tensors[inst.t[4] as usize];
        let owned = cache
            .name
            .strip_prefix("kv.")
            .and_then(|s| s.strip_suffix(".kidx_fp8"))
            .is_some_and(|layer| layer.parse::<u32>().is_ok());
        let slot_bytes = u64::from(ctx) * 132;
        let slots = cache.bytes / slot_bytes;
        if !owned
            || cache.bytes % slot_bytes != 0
            || !(1..=64).contains(&slots)
            || tensors[inst.t[5] as usize].name != "in.pos"
            || tensors[inst.t[6] as usize].name != "in.kvlen"
            || tensors[inst.t[6] as usize].bytes != slots * 4
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
                || owner.is_some_and(|seg| seg != e.seg as usize)
            {
                return Err(invalid("native counter obligations or multiple owners"));
            }
            owner = Some(e.seg as usize);
        }
        let seg = owner.ok_or_else(|| invalid("unowned native instruction"))?;
        if seg >= segments
            || routes[seg].is_some()
            || prog
                .stream
                .iter()
                .chain(&prog.gq_stream)
                .any(|e| e.seg as usize == seg && e.inst as usize != ix)
        {
            return Err(invalid("native segment is not pure"));
        }
        routes[seg] = Some(Route {
            handles: inst.t,
            addresses,
            rows,
            ctx,
            slots: slots as u32,
            append_only,
            chunk: None,
        });
    }
    Ok(routes)
}

pub(super) struct Indexer {
    prepare: HsaKernel,
    append: HsaKernel,
    scores: [HsaKernel; 2],
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
        if be.arch() != "gfx950" || be.sm_count() != 256 || !supported_rows(rows) || ctx != 131072 {
            return Err(invalid(
                "requires gfx950/256CU and qualified prefill geometry",
            ));
        }
        let mut kernels = Vec::new();
        for (file, hash, symbol, args) in [
            (
                "indexer_prefill_gfx950.elf",
                "38a75b92793716ff023d9bee7011aa680697a5e76563897cfbebb3ce1921c803",
                "glm_indexer_prefill_prepare",
                360,
            ),
            (
                "indexer_prefill_append_gfx950.elf",
                "0a2a5ad29a788ed0171ac8fd03afc94d738773c52f1437ad17ec11fb9934554d",
                "glm_indexer_prefill_append",
                288,
            ),
            (
                "indexer_prefill_score_gfx950.elf",
                "279e8359e2f5002ab46bd13388598c194f7dbc1984a7babc60f624553ad4cd55",
                "_gluon_fp8_mqa_logits_kernel",
                360,
            ),
            (
                "indexer_prefill_score_large_gfx950.elf",
                "87862e1204f8c16c533917f2e6ab9070467a3192f1e1c2264acb7795f19699f1",
                "_gluon_fp8_mqa_logits_kernel",
                360,
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
        let scratch = EngineDevice::alloc(be, u64::from(rows) * 4360 + u64::from(ctx) * 132)?;
        Ok(Self {
            prepare: kernels[0],
            append: kernels[1],
            scores: [kernels[2], kernels[3]],
            scratch,
            rows,
            ctx,
        })
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, table: &[u8]) -> Result<()> {
        let a = route.bindings(table)?;
        if route.rows > self.rows || route.ctx != self.ctx {
            return Err(invalid("scratch capacity or cache stride mismatch"));
        }
        let (base, live) = route.chunk.ok_or_else(|| invalid("unstaged chunk"))?;
        let rows = route.rows;
        if route.append_only {
            let args = [
                a[4],
                a[2],
                u64::from(rows) | u64::from(base) << 32,
                u64::from(live),
            ];
            return be.launch(
                self.append,
                rows.div_ceil(4),
                256,
                0,
                bytemuck::cast_slice(&args),
            );
        }
        let q = self.scratch.base;
        let scales = q + u64::from(self.rows) * 4096;
        let weights = scales + u64::from(self.rows) * 128;
        let starts = weights + u64::from(self.rows) * 128;
        let ends = starts + u64::from(self.rows) * 4;
        let keys = ends + u64::from(self.rows) * 4;
        let key_scales = keys + u64::from(self.ctx) * 128;
        let prep = [
            q,
            scales,
            weights,
            a[4],
            keys,
            key_scales,
            starts,
            ends,
            a[1],
            a[3],
            a[2],
            u64::from(rows) | u64::from(base) << 32,
            u64::from(live),
        ];
        be.launch(
            self.prepare,
            (rows * 33 + base).div_ceil(4),
            256,
            0,
            bytemuck::cast_slice(&prep),
        )?;
        let chunk_rows = route.score_chunk_rows();
        for row in (0..live).step_by(chunk_rows as usize) {
            let count = (live - row).min(chunk_rows);
            let offset = u64::from(row);
            let score = [
                q + offset * 4096,
                keys,
                key_scales,
                weights + offset * 128,
                starts + offset * 4,
                ends + offset * 4,
                a[0] + offset * u64::from(self.ctx) * 4,
                u64::from(count) | u64::from(base + live) << 32,
                4096 | 128 << 32,
                32 | u64::from(self.ctx) << 32,
                1,
                0,
                0,
            ];
            be.launch(
                self.scores[usize::from(route.large_scores())],
                count,
                64,
                12416,
                bytemuck::cast_slice(&score),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::{DevInst64, StreamEnt};
    use packet::devbuild::ProgramRole;

    fn fixture(append_only: bool) -> (DevProg, Vec<DevTensor>, Vec<DeviceMem>) {
        let mut inst = DevInst64 {
            op: DevOp::IndexFp8Prefill as u16,
            t: [0, 1, 2, 3, 4, 5, 6, TENSOR_NONE16],
            i: [128, 131072, u32::from(append_only), 0, 0, 0, 0, 0],
            ..Default::default()
        };
        if append_only {
            for i in [0, 1, 3] {
                inst.t[i] = TENSOR_NONE16;
            }
        }
        let prog = DevProg {
            t: 128,
            role: ProgramRole::PrefillBucket { rows: 128 },
            n_counter: 0,
            insts: vec![inst],
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
            "act.iscore_pf",
            "act.qidx_pf",
            "act.kidx_pf",
            "act.widx_pf",
            "kv.6.kidx_fp8",
            "in.pos",
            "in.kvlen",
        ];
        let sizes = [
            128 * 131072 * 4,
            128 * 8192,
            128 * 256,
            128 * 64,
            3 * 131072 * 132,
            131072 * 4,
            3 * 4,
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
            .map(|(i, bytes)| DeviceMem::view((i as u64 + 1) << 34, bytes))
            .collect();
        (prog, tensors, memory)
    }

    #[test]
    fn prefill_indexer_checks_both_modes_and_fail_closed_rearming() {
        for append_only in [false, true] {
            let (prog, tensors, memory) = fixture(append_only);
            let mut route = routes(&prog, &tensors, &memory, 1).unwrap()[0].unwrap();
            assert!(route.check_domain().is_err());
            assert_eq!(route.launches(), if append_only { 1 } else { 2 });
            for (base, live) in [(0, 128), (15, 125), (15, 0), (131071, 1)] {
                route.arm(base, live).unwrap();
                route.check_domain().unwrap();
                assert_eq!(
                    route.launches(),
                    if append_only || live == 0 { 1 } else { 2 }
                );
            }
            for (base, live) in [(0, 0), (0, 129), (131072, 1), (u32::MAX, 1)] {
                route.arm(15, 127).unwrap();
                assert!(route.arm(base, live).is_err());
                assert!(route.check_domain().is_err());
            }
            route.arm_for_program(71680, 127, 127).unwrap();
            assert!(route.arm_for_program(71680, 127, 128).is_err());
            assert!(route.check_domain().is_err());
            route.arm_for_program(71680, 128, 128).unwrap();
        }
    }

    #[test]
    fn prefill_indexer_rejects_bad_roles_aliases_capacity_and_ownership() {
        for append_only in [false, true] {
            for bad in 0..20 {
                let (mut prog, mut tensors, mut memory) = fixture(append_only);
                match bad {
                    0 => prog.role = ProgramRole::DecodeRung { rows: 128 },
                    1 => prog.role = ProgramRole::PackedSibling { of_rows: 128 },
                    2 => prog.role = ProgramRole::RowSplitSibling { of_rows: 128 },
                    3 => prog.t = 127,
                    4 => prog.insts[0].i[1] = 71680,
                    5 => prog.insts[0].i[2] = 2,
                    6 => prog.insts[0].i[3] = 1,
                    7 => prog.insts[0].fj[0] = 1,
                    8 => prog.insts[0].t[7] = 0,
                    9 => tensors[2].bytes -= 1,
                    10 => memory[2].len -= 1,
                    11 => memory[2].base += 1,
                    12 => memory[2].base = memory[4].base,
                    13 => tensors[4].name = "kv.6.kidx".into(),
                    14 => tensors[4].bytes -= 16,
                    15 => tensors[6].bytes -= 4,
                    16 => prog.stream[0].wait_len = 1,
                    17 => prog.stream.clear(),
                    18 => prog.stream.push(StreamEnt {
                        seg: 1,
                        ..Default::default()
                    }),
                    19 => {
                        prog.insts.push(DevInst64::default());
                        prog.stream.push(StreamEnt {
                            inst: 1,
                            ..Default::default()
                        });
                    }
                    _ => unreachable!(),
                }
                assert!(
                    routes(&prog, &tensors, &memory, 2).is_err(),
                    "mode={append_only} case={bad}"
                );
            }
        }
    }

    #[test]
    fn prefill_indexer_binds_only_owned_slot_rebases() {
        let (prog, tensors, memory) = fixture(false);
        let mut route = routes(&prog, &tensors, &memory, 1).unwrap()[0].unwrap();
        let addresses: Vec<u64> = memory.iter().map(|m| m.base).collect();
        let table = bytemuck::cast_slice(&addresses);
        assert!(route.bindings(table).is_err());
        route.arm(71680, 127).unwrap();
        for slot in 0..3 {
            let mut addresses = addresses.clone();
            addresses[4] += slot * 131072 * 132;
            assert!(route.bindings(bytemuck::cast_slice(&addresses)).is_ok());
        }
        for delta in [16, 3 * 131072 * 132, u64::MAX] {
            let mut addresses = addresses.clone();
            addresses[4] = addresses[4].wrapping_add(delta);
            assert!(route.bindings(bytemuck::cast_slice(&addresses)).is_err());
        }
        let mut moved = addresses.clone();
        moved[2] += 16;
        assert!(route.bindings(bytemuck::cast_slice(&moved)).is_err());
        assert!(route.bindings(&table[..8]).is_err());
    }

    #[test]
    fn prefill_indexer_keeps_reference_math_when_output_stride_changes() {
        let (prog, tensors, memory) = fixture(false);
        let mut route = routes(&prog, &tensors, &memory, 1).unwrap()[0].unwrap();
        route.rows = 8192;
        route.arm(0, 8192).unwrap();
        assert!(!route.large_scores());
        assert_eq!(route.score_chunk_rows(), 4096);
        assert_eq!(route.launches(), 3);
        route.arm(71680, 8191).unwrap();
        assert!(route.large_scores());
        assert_eq!(route.score_chunk_rows(), 8192);
        assert_eq!(route.launches(), 2);
        for live in [1, 4096, 4097] {
            route.arm(71680, live).unwrap();
            assert!(!route.large_scores());
            assert_eq!(route.launches(), 1 + live.div_ceil(4096) as usize);
        }
        route.arm(71680, 0).unwrap();
        assert_eq!(route.launches(), 1);
        route.arm(57344, 8192).unwrap();
        assert!(route.large_scores());
        route.arm(57345, 8191).unwrap();
        assert!(!route.large_scores());
        route.rows = 4096;
        route.arm(0, 4096).unwrap();
        assert!(!route.large_scores());
        assert_eq!(route.launches(), 2);
    }

    #[test]
    #[ignore = "requires queued gfx950 GPU and PLOW_TEST_AITER_DIR with pinned prefill fixtures"]
    fn prefill_indexer_native_chain_reference_replay() {
        let root = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let root = Path::new(&root);
        let reference = root.join("reference");
        let manifest = std::fs::read(reference.join("reference.json")).unwrap();
        assert_eq!(
            plow_asset::decode_objects::image_sha256(&manifest),
            "1f38abdd159bd3a3e821cc974ad69204abbcd1e7e0ccd5786100c73c85f47b54"
        );
        let report: serde_json::Value = serde_json::from_slice(&manifest).unwrap();
        assert_eq!(report["passed"], true);
        assert_eq!(report["vllm_version"], "0.29.0");
        assert_eq!(report["live_query_rows"], true);
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 21);
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let kernel = Indexer::load(&be, root, 8192, 131072, &mut modules).unwrap();
        for case in cases {
            let bytes = std::fs::read(reference.join(case["file"].as_str().unwrap())).unwrap();
            assert_eq!(
                plow_asset::decode_objects::image_sha256(&bytes),
                case["sha256"].as_str().unwrap()
            );
            let hdr: Vec<_> = bytes[..32]
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            let (rows, ctx, base, live, slots, slot, stride) =
                (hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7]);
            assert_eq!((hdr[0], ctx, slots, slot), (0x49504631, 131072, 3, 1));
            assert!(supported_rows(rows));
            assert_eq!(stride, (base + live).div_ceil(256) * 256);
            let m = rows as usize;
            let c = ctx as usize;
            let sizes = [
                m * c * 4,
                m * 8192,
                m * 256,
                m * 64,
                slots as usize * c * 132,
                c * 4,
                slots as usize * 4,
            ];
            let mut inputs = [&[][..]; 7];
            let mut at = 32;
            for i in [1, 3, 2, 4] {
                inputs[i] = &bytes[at..at + sizes[i]];
                at += sizes[i];
            }
            let expected_sizes = [
                m * 4096,
                m * 128,
                m * 128,
                sizes[4],
                (base + live) as usize * 128,
                (base + live) as usize * 4,
                m * 4,
                m * 4,
                m * stride as usize * 4,
            ];
            let mut expected = [&[][..]; 9];
            for (i, size) in expected_sizes.into_iter().enumerate() {
                expected[i] = &bytes[at..at + size];
                at += size;
                assert_eq!(
                    plow_asset::decode_objects::image_sha256(expected[i]),
                    case["output_sha256"][i].as_str().unwrap()
                );
            }
            assert_eq!(at, bytes.len());
            let memory: Vec<_> = sizes
                .iter()
                .map(|&size| EngineDevice::alloc(&be, size as u64 + 256).unwrap())
                .collect();
            for i in [1, 2, 3] {
                EngineDevice::upload(&be, &memory[i], 0, inputs[i]).unwrap();
            }
            let mut table: Vec<_> = memory.iter().map(|m| m.base).collect();
            table[4] += u64::from(slot) * u64::from(ctx) * 132;
            for append_only in [false, true] {
                let (mut prog, mut tensors, _) = fixture(append_only);
                prog.t = rows;
                prog.role = ProgramRole::PrefillBucket { rows };
                prog.insts[0].i[0] = rows;
                for (t, &size) in tensors.iter_mut().zip(&sizes) {
                    t.bytes = size as u64;
                }
                let mut route = routes(&prog, &tensors, &memory, 1).unwrap()[0].unwrap();
                route.arm(base, live).unwrap();
                if live == 0 {
                    assert!(case["kernel"].is_null());
                } else {
                    assert_eq!(case["kernel"]["grid"], serde_json::json!([live]));
                    assert_eq!(case["kernel"]["buffer_store"], !route.large_scores());
                }
                for poison in [0x55, 0xaa] {
                    EngineDevice::upload(&be, &memory[4], 0, inputs[4]).unwrap();
                    if !append_only {
                        EngineDevice::upload(&be, &memory[0], 0, &vec![poison; sizes[0]]).unwrap();
                    }
                    for (mem, &size) in memory.iter().zip(&sizes) {
                        EngineDevice::upload(&be, mem, size as u64, &[poison; 256]).unwrap();
                    }
                    be.begin_dispatch_chain(route.launches()).unwrap();
                    kernel
                        .enqueue(&be, route, bytemuck::cast_slice(&table))
                        .unwrap();
                    be.commit_dispatch_chain().unwrap();
                    be.synchronize().unwrap();
                    let mut actual = vec![0; sizes[4]];
                    EngineDevice::download(&be, &memory[4], 0, &mut actual).unwrap();
                    assert!(
                        actual == expected[3],
                        "cache T={rows} base={base} append={append_only}"
                    );
                    if !append_only {
                        let r = u64::from(kernel.rows);
                        let scratch_offsets = [
                            0,
                            r * 4096,
                            r * 4224,
                            0,
                            r * 4360,
                            r * 4360 + u64::from(ctx) * 128,
                            r * 4352,
                            r * 4356,
                        ];
                        for part in [0, 1, 2, 4, 5, 6, 7] {
                            let mut actual = vec![0; expected[part].len()];
                            EngineDevice::download(
                                &be,
                                &kernel.scratch,
                                scratch_offsets[part],
                                &mut actual,
                            )
                            .unwrap();
                            assert!(
                                actual == expected[part],
                                "prep part={part} T={rows} base={base}"
                            );
                        }
                        let mut actual = vec![0; sizes[0]];
                        EngineDevice::download(&be, &memory[0], 0, &mut actual).unwrap();
                        let poison_row = vec![poison; c * 4];
                        for row in 0..m {
                            let len = if row < live as usize {
                                base as usize + row + 1
                            } else {
                                0
                            };
                            assert!(
                                actual[row * c * 4..(row * c + len) * 4]
                                    == expected[8][row * stride as usize * 4
                                        ..(row * stride as usize + len) * 4],
                                "scores T={rows} base={base} row={row}"
                            );
                            assert!(
                                actual[(row * c + len) * 4..(row + 1) * c * 4]
                                    == poison_row[..(c - len) * 4]
                            );
                        }
                    }
                    for (mem, &size) in memory.iter().zip(&sizes) {
                        let mut guard = [0; 256];
                        EngineDevice::download(&be, mem, size as u64, &mut guard).unwrap();
                        assert_eq!(guard, [poison; 256]);
                    }
                    for i in [1, 2, 3] {
                        let mut actual = vec![0; sizes[i]];
                        EngineDevice::download(&be, &memory[i], 0, &mut actual).unwrap();
                        assert!(actual == inputs[i]);
                    }
                    eprintln!("prefill native T={rows} base={base} live={live} append={append_only} poison={poison} packets={} PASS", route.launches());
                }
            }
        }
        drop(kernel);
        for module in modules {
            be.module_unload(&module).unwrap();
        }
    }
}
