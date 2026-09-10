use std::collections::BTreeSet;
use std::path::Path;

use packet::dev::{DevInst64, DevOp, SE_XCTR, TENSOR_NONE16};

use super::amd::TpBind;
use super::device_api::EngineDevice;
use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::Module;
use crate::{Result, RuntimeError};

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    inst: DevInst64,
    rows: u32,
}

impl Route {
    pub fn rebase(&mut self, rows: u32) -> Result<()> {
        if rows == 0 || rows > self.inst.i[0] {
            return Err(RuntimeError::Device(
                "TP indexer chunk exceeds row capacity".into(),
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
    tp: TpBind,
) -> Result<Vec<Option<Route>>> {
    let mut routes = vec![None; segments];
    for (ix, inst) in prog.insts.iter().enumerate() {
        if inst.op != DevOp::IndexTpPf as u16 {
            continue;
        }
        let err = |s: &str| RuntimeError::Device(format!("TP indexer instruction {ix}: {s}"));
        if prog.packed_prefill_only
            || !(2048..=8192).contains(&prog.t)
            || prog.t % 8 != 0
            || inst.i[0] != prog.t
            || !(2048..=131072).contains(&inst.i[1])
            || inst.i[1] < prog.t
            || inst.i[2..4] != [2048, 8]
            || tp.n_gpu != 8
            || tp.rank >= 8
            || u64::from(inst.i[4]) != tp.slot_b
            || inst.i[5].checked_add(2) != Some(inst.i[6])
            || inst.i[6] >= tp.xstatus_id
            || inst.i[7] != 0
            || inst.fj != [0x3c7fffff, 0, 0]
            || inst.t[7] != TENSOR_NONE16
        {
            return Err(err(
                "requires unpacked TP8 HI32/DI128/top2048 prefill and three valid gates",
            ));
        }
        for (other_ix, other) in prog.insts.iter().enumerate() {
            if other_ix == ix {
                continue;
            }
            for gate in inst.i[5]..=inst.i[6] {
                let overlap = match DevOp::from_u16(other.op) {
                    Some(
                        DevOp::XReduce
                        | DevOp::XReduceAddNorm
                        | DevOp::XReduceScatter
                        | DevOp::XAllGather,
                    ) => gate == other.i[3],
                    Some(DevOp::XReduceTwoShot) => gate == other.i[3] || gate == other.i[4],
                    Some(DevOp::IndexTpPf) => (other.i[5]..=other.i[6]).contains(&gate),
                    Some(DevOp::XArgmaxFin) => {
                        gate == other.i[3]
                            || (gate >= other.i[4]
                                && gate - other.i[4]
                                    < packet::devbuild::xargmax_value_lines(other.i[1])
                                        .unwrap_or(u32::MAX))
                    }
                    _ => false,
                };
                if overlap {
                    return Err(err("arrival gate overlaps another collective"));
                }
            }
        }
        if inst.t[..7].iter().collect::<BTreeSet<_>>().len() != 7 {
            return Err(err("operands alias"));
        }
        let rows = u64::from(prog.t);
        let ctx = u64::from(inst.i[1]);
        for (&handle, bytes) in inst.t[..7].iter().zip([
            rows * 2048 * 4,
            rows * ctx * 4,
            rows * 32 * 128 * 2,
            ctx * 128 * 2,
            rows * 32 * 2,
            4,
            rows * 2048 * 4,
        ]) {
            if handle == TENSOR_NONE16
                || tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes)
            {
                return Err(err("operand capacity is insufficient"));
            }
        }
        if tensors[inst.t[6] as usize].name != "act.dg_tp" || tp.slot_b < rows * 2048 * 4 {
            return Err(err("selected rows must fit the bound TP scratch slot"));
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
        });
    }
    Ok(routes)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScoreArgs {
    pointers: [u64; 5],
    rows: u32,
    stride: u32,
    rank: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SelectArgs {
    pointers: [u64; 3],
    peers: u64,
    xctr_offset: u64,
    status: u64,
    deadline: u64,
    rows: u32,
    stride: u32,
    rank: u32,
    gate: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherArgs {
    out: u64,
    peers: u64,
    xctr_offset: u64,
    slot_offset: u64,
    status: u64,
    deadline: u64,
    rows: u32,
    rank: u32,
    gate: u32,
    pad: u32,
}

const _: () = assert!(std::mem::size_of::<ScoreArgs>() == 56);
const _: () = assert!(std::mem::size_of::<SelectArgs>() == 72);
const _: () = assert!(std::mem::size_of::<GatherArgs>() == 64);

pub(super) struct IndexTp {
    kernels: [HsaKernel; 4],
}

impl IndexTp {
    pub fn load(be: &HsaBackend, dir: &Path, modules: &mut Vec<Module>) -> Result<Self> {
        let path = dir.join("dsa_tp_adapter_gfx942.elf");
        let image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        if !super::amd::elf_symbol_names(&image).contains(&"plow_dsa_tp_abi_1") {
            return Err(RuntimeError::Device(
                "TP indexer adapter lacks ABI marker".into(),
            ));
        }
        let module = EngineDevice::module_load(be, &image)?;
        let mut kernels = Vec::new();
        for (name, bytes) in [
            ("plow_dsa_tp_score", 56),
            ("plow_dsa_tp_select", 72),
            ("plow_dsa_tp_gather", 64),
            ("plow_dsa_tp_complete", 64),
        ] {
            let kernel = EngineDevice::get_function(be, &module, name)?;
            if ![bytes, bytes + 256].contains(&kernel.kernarg_size())
                || kernel.private_segment_size() != 0
            {
                return Err(RuntimeError::Device(format!(
                    "TP indexer resource ABI mismatch for {name}: kernarg={}, private={}",
                    kernel.kernarg_size(),
                    kernel.private_segment_size()
                )));
            }
            kernels.push(kernel);
        }
        modules.push(module);
        Ok(Self {
            kernels: kernels.try_into().ok().unwrap(),
        })
    }

    pub fn enqueue(
        &self,
        be: &HsaBackend,
        route: Route,
        tensor_table: &[u8],
        tp: TpBind,
    ) -> Result<()> {
        let addr = |handle: u16| {
            let at = usize::from(handle) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        };
        let t: [u64; 7] = std::array::from_fn(|i| addr(route.inst.t[i]));
        if t[6] != tp.scratch_base + tp.slot_b {
            return Err(RuntimeError::Device(
                "TP indexer scratch binding changed".into(),
            ));
        }
        let score = ScoreArgs {
            pointers: [t[1], t[2], t[3], t[4], t[5]],
            rows: route.rows,
            stride: route.inst.i[1],
            rank: tp.rank,
            pad: 0,
        };
        let select = SelectArgs {
            pointers: [t[6], t[1], t[5]],
            peers: tp.peer_table,
            xctr_offset: tp.xctr - tp.scratch_base,
            status: tp.xctr + u64::from(tp.xstatus_id) * 128,
            deadline: 1_000_000_000,
            rows: route.rows,
            stride: route.inst.i[1],
            rank: tp.rank,
            gate: route.inst.i[5],
        };
        let gather = GatherArgs {
            out: t[0],
            peers: tp.peer_table,
            xctr_offset: tp.xctr - tp.scratch_base,
            slot_offset: tp.slot_b,
            status: tp.xctr + u64::from(tp.xstatus_id) * 128,
            deadline: 1_000_000_000,
            rows: route.rows,
            rank: tp.rank,
            gate: route.inst.i[5] + 1,
            pad: 0,
        };
        let [ks, ki, kg, kc] = self.kernels;
        be.launch(ks, 304, 512, 0, bytemuck::bytes_of(&score))?;
        be.launch(ki, 304, 512, 0, bytemuck::bytes_of(&select))?;
        be.launch(kg, 304, 256, 0, bytemuck::bytes_of(&gather))?;
        be.launch(kc, 1, 64, 0, bytemuck::bytes_of(&gather))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::StreamEnt;

    fn fixture() -> (DevProg, Vec<DevTensor>, TpBind) {
        let prog = DevProg {
            t: 8192,
            packed_prefill_only: false,
            n_counter: 0,
            insts: vec![DevInst64 {
                op: DevOp::IndexTpPf as u16,
                t: [0, 1, 2, 3, 4, 5, 6, TENSOR_NONE16],
                i: [8192, 81920, 2048, 8, 96 << 20, 0, 2, 0],
                fj: [0x3c7fffff, 0, 0],
                ..Default::default()
            }],
            stream: vec![StreamEnt {
                inst: 0,
                seg: 0,
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
        let tensors = (0..7)
            .map(|i| DevTensor {
                name: if i == 6 {
                    "act.dg_tp".into()
                } else {
                    i.to_string()
                },
                bytes: 4 << 30,
                init: None,
            })
            .collect();
        let tp = TpBind {
            rank: 0,
            n_gpu: 8,
            peer_table: 0,
            xctr: 0,
            xstatus_id: 3,
            scratch_base: 0,
            slot_b: 96 << 20,
        };
        (prog, tensors, tp)
    }

    #[test]
    fn index_tp_routes_reject_invalid_geometry_bindings_and_segments() {
        let (p, t, tp) = fixture();
        let mut route = routes(&p, &t, 1, tp).unwrap()[0].unwrap();
        for rows in [1, 129, 4464, 8192] {
            route.rebase(rows).unwrap();
        }
        assert!(route.rebase(0).is_err());
        assert!(route.rebase(8193).is_err());
        for bad in 0..19 {
            let (mut p, mut t, mut tp) = fixture();
            match bad {
                0 => p.packed_prefill_only = true,
                1 => p.insts[0].i[0] = 4096,
                2 => p.insts[0].i[2] = 1024,
                3 => p.insts[0].fj[0] = 0x3c800000,
                4 => p.insts[0].i[6] = 0,
                5 => tp.xstatus_id = 1,
                6 => tp.slot_b = 16,
                7 => tp.n_gpu = 4,
                8 => tp.rank = 8,
                9 => p.insts[0].t[2] = 1,
                10 => t[6].name = "unmapped".into(),
                11 => p.stream[0].wait_len = 1,
                12 => p.stream[0].succ_len = 1,
                13 => p.stream[0].flags |= SE_XCTR,
                14 => p.gq_stream.push(StreamEnt {
                    inst: 0,
                    seg: 1,
                    ..Default::default()
                }),
                15 => p.stream.push(StreamEnt {
                    inst: 1,
                    seg: 0,
                    ..Default::default()
                }),
                16 => t[1].bytes = 16,
                17 => t[6].bytes = 16,
                _ => p.insts[0].t[5] = TENSOR_NONE16,
            }
            assert!(routes(&p, &t, 1, tp).is_err(), "case {bad}");
        }
        let (mut p, t, tp) = fixture();
        p.insts.push(DevInst64 {
            op: DevOp::XReduce as u16,
            i: [0, 8, 0, 2, 0, 0, 0, 0],
            ..Default::default()
        });
        p.stream.push(StreamEnt {
            inst: 1,
            seg: 1,
            ..Default::default()
        });
        assert!(routes(&p, &t, 2, tp)
            .unwrap_err()
            .to_string()
            .contains("overlaps"));
    }
}
