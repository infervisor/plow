use std::collections::BTreeSet;
use std::io::ErrorKind;
use std::path::Path;

use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};

use super::amd_gemm_lt::segment_owners;
use super::device_api::EngineDevice;
use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::Module;
use crate::{Result, RuntimeError};

const OBJECT: &str = "gemma4_gemm_glu_gfx942.elf";
const C2_LDS: u32 = 55_296;
const C5_LDS: u32 = 64_512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    Gemma12,
    Gemma31,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tile {
    C2,
    C5,
}

fn model(inst: &DevInst64) -> Option<Model> {
    match (inst.i[1], inst.i[2]) {
        (15_360, 3_840) => Some(Model::Gemma12),
        (21_504, 5_376) => Some(Model::Gemma31),
        _ => None,
    }
}

fn candidate(inst: &DevInst64) -> bool {
    inst.op == DevOp::GemmGlu as u16 && model(inst).is_some()
}

fn native(inst: &DevInst64) -> bool {
    candidate(inst)
        && inst.i[3..5] == [0; 2]
        && inst.i[5] == 0
        && inst.i[6..] == [0; 2]
        && inst.fj == [0; 3]
        && inst.t[3..5] == [TENSOR_NONE16; 2]
        && inst.t[6..] == [TENSOR_NONE16; 2]
}

pub(super) fn program_candidate(prog: &DevProg) -> bool {
    if prog.l2_domains != 0 || !matches!(prog.t, 1024 | 2048 | 4096 | 8192) {
        return false;
    }
    let segments = prog
        .stream
        .iter()
        .chain(&prog.gq_stream)
        .map(|entry| usize::from(entry.seg) + 1)
        .max()
        .unwrap_or(0);
    segment_owners(prog, segments, native)
        .is_ok_and(|owners| owners.into_iter().any(|owner| owner.is_some()))
}

fn tile(model: Model, rows: u32) -> Tile {
    match model {
        Model::Gemma12 if rows <= 4096 => Tile::C2,
        Model::Gemma31 if rows <= 1024 => Tile::C2,
        Model::Gemma12 | Model::Gemma31 => Tile::C5,
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    inst: DevInst64,
    capacity: u32,
    rows: u32,
    model: Model,
    tile: Tile,
}

impl Route {
    pub fn rebase(&mut self, rows: u32) -> Result<()> {
        if rows == 0 || rows > self.capacity {
            return Err(RuntimeError::Device(
                "native Gemma-4 GemmGlu exceeds row capacity".into(),
            ));
        }
        self.rows = rows;
        self.tile = tile(self.model, rows);
        Ok(())
    }
}

pub(super) fn routes(
    prog: &DevProg,
    tensors: &[DevTensor],
    segments: usize,
) -> Result<Vec<Option<Route>>> {
    let mut out = vec![None; segments];
    if prog.l2_domains != 0
        || !matches!(prog.t, 1024 | 2048 | 4096 | 8192)
        || !prog.insts.iter().any(native)
    {
        return Ok(out);
    }
    let Ok(owners) = segment_owners(prog, segments, native) else {
        return Ok(out);
    };
    for (ix, inst) in prog.insts.iter().enumerate().filter(|(_, i)| native(i)) {
        let err = |s: &str| RuntimeError::Device(format!("native Gemma-4 GemmGlu {ix}: {s}"));
        let model = model(inst).unwrap();
        if inst.i[0] != prog.t {
            return Err(err(
                "requires an exact 1K/2K/4K/8K bias-free BF16 Gemma gate/up projection",
            ));
        }
        let mut handles = BTreeSet::new();
        let (m, n, k) = (
            u64::from(prog.t),
            u64::from(inst.i[1]),
            u64::from(inst.i[2]),
        );
        for (&handle, bytes) in [inst.t[0], inst.t[1], inst.t[2], inst.t[5]].iter().zip([
            m * n * 2,
            m * k * 2,
            n * k * 2,
            n * k * 2,
        ]) {
            if handle == TENSOR_NONE16
                || !handles.insert(handle)
                || tensors
                    .get(usize::from(handle))
                    .is_none_or(|tensor| tensor.bytes < bytes)
            {
                return Err(err("aliased or undersized operand"));
            }
        }
        let seg = owners[ix].ok_or_else(|| err("requires exactly one segment owner"))?;
        out[seg] = Some(Route {
            inst: *inst,
            capacity: prog.t,
            rows: prog.t,
            model,
            tile: tile(model, prog.t),
        });
    }
    Ok(out)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    out: u64,
    input: u64,
    gate: u64,
    up: u64,
    rows: u32,
    n: u32,
    k: u32,
    act: u32,
}
const _: () = assert!(std::mem::size_of::<Args>() == 48);

pub(super) struct Gemma4Glu {
    c2: HsaKernel,
    c5_12: HsaKernel,
    c5_31: HsaKernel,
    n_cu: u32,
}

impl Gemma4Glu {
    pub fn load(
        be: &HsaBackend,
        dir: &Path,
        n_cu: u32,
        modules: &mut Vec<Module>,
    ) -> Result<Option<Self>> {
        let path = dir.join(OBJECT);
        let image = match std::fs::read(&path) {
            Ok(image) => image,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(RuntimeError::Device(format!("{}: {error}", path.display()))),
        };
        let symbols = super::amd::elf_symbol_names(&image);
        for marker in [
            "plow_gemma4_glu_abi_1",
            "plow_gemma4_glu_wg512_1",
            "plow_gemma4_glu_nospill_1",
            "plow_gemma4_glu_c2_128x256x64_1",
            "plow_gemma4_glu_c5_192x256x64_1",
            "plow_gemma4_glu_c5_exact_shape_1",
        ] {
            if !symbols.contains(&marker) {
                return Err(RuntimeError::Device(format!(
                    "{} lacks required marker `{marker}`",
                    path.display()
                )));
            }
        }
        let module = EngineDevice::module_load(be, &image)?;
        let c2 = EngineDevice::get_function(be, &module, "plow_gemma4_gemm_glu_c2_gfx942")?;
        let c5_12 = EngineDevice::get_function(be, &module, "plow_gemma4_12b_gemm_glu_c5_gfx942")?;
        let c5_31 = EngineDevice::get_function(be, &module, "plow_gemma4_31b_gemm_glu_c5_gfx942")?;
        for (name, kernel, lds) in [
            ("C2", c2, C2_LDS),
            ("12B C5", c5_12, C5_LDS),
            ("31B C5", c5_31, C5_LDS),
        ] {
            let kernarg = kernel.kernarg_size();
            if ![48, 304].contains(&kernarg)
                || kernel.private_segment_size() != 0
                || HsaBackend::kernel_lds_bytes(&kernel) != lds
            {
                return Err(RuntimeError::Device(format!(
                    "Gemma-4 GemmGlu {name} resource ABI mismatch: kernarg={kernarg}, LDS={}, private={}",
                    HsaBackend::kernel_lds_bytes(&kernel),
                    kernel.private_segment_size()
                )));
            }
        }
        modules.push(module);
        Ok(Some(Self {
            c2,
            c5_12,
            c5_31,
            n_cu,
        }))
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, tensor_table: &[u8]) -> Result<()> {
        let [out, input, gate, up] = [
            route.inst.t[0],
            route.inst.t[1],
            route.inst.t[2],
            route.inst.t[5],
        ]
        .map(|handle| {
            let at = usize::from(handle) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        });
        let args = Args {
            out,
            input,
            gate,
            up,
            rows: route.rows,
            n: route.inst.i[1],
            k: route.inst.i[2],
            act: route.inst.i[5],
        };
        let kernel = match (route.model, route.tile) {
            (Model::Gemma12 | Model::Gemma31, Tile::C2) => self.c2,
            (Model::Gemma12, Tile::C5) => self.c5_12,
            (Model::Gemma31, Tile::C5) => self.c5_31,
        };
        be.launch(kernel, self.n_cu, 512, 0, bytemuck::bytes_of(&args))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::{StreamEnt, SE_XCTR};

    fn fixture(rows: u32, n: u32, k: u32) -> (DevProg, Vec<DevTensor>) {
        let inst = DevInst64 {
            op: DevOp::GemmGlu as u16,
            t: [
                0,
                1,
                2,
                TENSOR_NONE16,
                TENSOR_NONE16,
                3,
                TENSOR_NONE16,
                TENSOR_NONE16,
            ],
            i: [rows, n, k, 0, 0, 0, 0, 0],
            ..Default::default()
        };
        let prog = DevProg {
            t: rows,
            role: packet::devbuild::ProgramRole::PrefillBucket { rows },
            n_counter: 0,
            insts: vec![inst],
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
        let bytes = [
            u64::from(rows) * u64::from(n) * 2,
            u64::from(rows) * u64::from(k) * 2,
            u64::from(n) * u64::from(k) * 2,
            u64::from(n) * u64::from(k) * 2,
        ];
        let tensors = bytes
            .into_iter()
            .enumerate()
            .map(|(i, bytes)| DevTensor {
                name: i.to_string(),
                bytes,
                init: None,
            })
            .collect();
        (prog, tensors)
    }

    #[test]
    fn exact_gemma_shapes_select_qualified_tiles() {
        for (n, k, model) in [
            (15_360, 3_840, Model::Gemma12),
            (21_504, 5_376, Model::Gemma31),
        ] {
            for rows in [1024, 2048, 4096, 8192] {
                let (prog, tensors) = fixture(rows, n, k);
                assert!(program_candidate(&prog));
                let route = routes(&prog, &tensors, 1).unwrap()[0].unwrap();
                assert_eq!(route.model, model);
                assert_eq!(
                    route.tile,
                    if (model == Model::Gemma12 && rows <= 4096)
                        || (model == Model::Gemma31 && rows <= 1024)
                    {
                        Tile::C2
                    } else {
                        Tile::C5
                    }
                );
            }
        }
    }

    #[test]
    fn route_rebases_ragged_rows_and_rejects_bad_abi() {
        let (prog, tensors) = fixture(8192, 15_360, 3_840);
        let mut route = routes(&prog, &tensors, 1).unwrap()[0].unwrap();
        route.rebase(4096).unwrap();
        assert_eq!((route.rows, route.tile), (4096, Tile::C2));
        assert!(route.rebase(0).is_err());
        assert!(route.rebase(8193).is_err());

        let (mut l2_prog, l2_tensors) = fixture(8192, 15_360, 3_840);
        l2_prog.l2_domains = 8;
        assert!(!program_candidate(&l2_prog));
        assert!(routes(&l2_prog, &l2_tensors, 1).unwrap()[0].is_none());

        for case in 0..10 {
            let (mut prog, mut tensors) = fixture(8192, 15_360, 3_840);
            match case {
                0 => prog.t = 512,
                1 => prog.insts[0].i[0] = 4096,
                2 => prog.insts[0].i[5] = 1,
                3 => prog.insts[0].fj[0] = 1,
                4 => prog.insts[0].t[6] = 0,
                5 => prog.insts[0].t[5] = 2,
                6 => tensors[3].bytes = 1,
                7 => prog.stream[0].wait_len = 1,
                8 => prog.stream[0].flags = SE_XCTR,
                _ => prog.stream.clear(),
            }
            let result = routes(&prog, &tensors, 1);
            if matches!(case, 1 | 5 | 6) {
                assert!(result.is_err(), "case {case}");
            } else {
                assert!(result.unwrap()[0].is_none(), "case {case}");
            }
        }
    }
}
