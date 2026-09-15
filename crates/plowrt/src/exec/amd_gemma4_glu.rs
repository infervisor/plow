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
enum Precision {
    Bf16,
    Fp8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Glu,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tile {
    C2,
    C3,
    C5,
    C6,
}

fn classify(inst: &DevInst64) -> Option<(Model, Kind)> {
    match (inst.i[1], inst.i[2]) {
        (15_360, 3_840) => Some((Model::Gemma12, Kind::Glu)),
        (21_504, 5_376) => Some((Model::Gemma31, Kind::Glu)),
        (3_840, 15_360) => Some((Model::Gemma12, Kind::Down)),
        (5_376, 21_504) => Some((Model::Gemma31, Kind::Down)),
        _ => None,
    }
}

fn precision(inst: &DevInst64) -> Option<Precision> {
    match DevOp::from_u16(inst.op) {
        Some(
            DevOp::Gemm
            | DevOp::GemmMed
            | DevOp::GemmSmall
            | DevOp::GemmWide
            | DevOp::GemmC5
            | DevOp::GemmGlu,
        ) => Some(Precision::Bf16),
        Some(
            DevOp::GemmFp8
            | DevOp::GemmMedFp8
            | DevOp::GemmSmallFp8
            | DevOp::GemmWideFp8
            | DevOp::GemmC5Fp8
            | DevOp::GemmGluFp8,
        ) => Some(Precision::Fp8),
        _ => None,
    }
}

fn native(inst: &DevInst64) -> bool {
    let Some((_, kind)) = classify(inst) else {
        return false;
    };
    match (kind, precision(inst)) {
        (Kind::Glu, Some(Precision::Bf16)) => {
            inst.op == DevOp::GemmGlu as u16
                && inst.i[3..] == [0; 5]
                && inst.fj == [0; 3]
                && inst.t[3..5] == [TENSOR_NONE16; 2]
                && inst.t[6..] == [TENSOR_NONE16; 2]
        }
        (Kind::Glu, Some(Precision::Fp8)) => {
            inst.op == DevOp::GemmGluFp8 as u16
                && inst.i[3..] == [0; 5]
                && inst.fj == [0; 3]
                && inst.t[..7].iter().all(|&tensor| tensor != TENSOR_NONE16)
                && inst.t[7] == TENSOR_NONE16
        }
        (Kind::Down, Some(Precision::Bf16)) => {
            inst.i[3..] == [0; 5]
                && inst.t[..3].iter().all(|&tensor| tensor != TENSOR_NONE16)
                && inst.t[3..] == [TENSOR_NONE16; 5]
        }
        (Kind::Down, Some(Precision::Fp8)) => {
            inst.i[3..] == [0; 5]
                && inst.t[..5].iter().all(|&tensor| tensor != TENSOR_NONE16)
                && inst.t[5..] == [TENSOR_NONE16; 3]
        }
        _ => false,
    }
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

pub(super) fn program_fp8_candidate(prog: &DevProg) -> bool {
    program_candidate(prog)
        && prog
            .insts
            .iter()
            .any(|inst| native(inst) && precision(inst) == Some(Precision::Fp8))
}

pub(super) fn program_down_candidate(prog: &DevProg) -> bool {
    program_candidate(prog)
        && prog
            .insts
            .iter()
            .any(|inst| native(inst) && classify(inst).is_some_and(|(_, k)| k == Kind::Down))
}

fn tile(model: Model, kind: Kind, precision: Precision, rows: u32) -> Tile {
    match (model, kind, precision, rows) {
        (Model::Gemma12, Kind::Down, Precision::Bf16, 8192) => Tile::C6,
        (Model::Gemma12, Kind::Down, _, _) => Tile::C2,
        (Model::Gemma31, Kind::Down, Precision::Bf16, _) => Tile::C5,
        (Model::Gemma31, Kind::Down, Precision::Fp8, 4096) => Tile::C3,
        (Model::Gemma31, Kind::Down, Precision::Fp8, rows) if rows <= 2048 => Tile::C5,
        (Model::Gemma31, Kind::Down, _, _) => Tile::C2,
        (Model::Gemma12, Kind::Glu, _, rows) if rows <= 4096 => Tile::C2,
        (Model::Gemma31, Kind::Glu, _, rows) if rows <= 1024 => Tile::C2,
        (Model::Gemma12 | Model::Gemma31, Kind::Glu, _, _) => Tile::C5,
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    inst: DevInst64,
    capacity: u32,
    rows: u32,
    model: Model,
    precision: Precision,
    kind: Kind,
    tile: Tile,
}

impl Route {
    pub fn rebase(&mut self, rows: u32) -> Result<()> {
        if rows == 0 || rows > self.capacity {
            return Err(RuntimeError::Device(
                "native Gemma-4 dense GEMM exceeds row capacity".into(),
            ));
        }
        self.rows = rows;
        self.tile = tile(self.model, self.kind, self.precision, rows);
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
        let err = |s: &str| RuntimeError::Device(format!("native Gemma-4 dense GEMM {ix}: {s}"));
        let (model, kind) = classify(inst).unwrap();
        if inst.i[0] != prog.t {
            return Err(err(
                "requires an exact 1K/2K/4K/8K bias-free Gemma projection",
            ));
        }
        let mut handles = BTreeSet::new();
        let precision = precision(inst).unwrap();
        let (m, n, k) = (
            u64::from(prog.t),
            u64::from(inst.i[1]),
            u64::from(inst.i[2]),
        );
        let operands: Vec<(u16, u64)> = match (kind, precision) {
            (Kind::Glu, Precision::Bf16) => vec![
                (inst.t[0], m * n * 2),
                (inst.t[1], m * k * 2),
                (inst.t[2], n * k * 2),
                (inst.t[5], n * k * 2),
            ],
            (Kind::Glu, Precision::Fp8) => vec![
                (inst.t[0], m * n * 2),
                (inst.t[1], m * k),
                (inst.t[2], n * k),
                (inst.t[3], m * 4),
                (inst.t[4], n * 4),
                (inst.t[5], n * k),
                (inst.t[6], n * 4),
            ],
            (Kind::Down, Precision::Bf16) => vec![
                (inst.t[0], m * n * 2),
                (inst.t[1], m * k * 2),
                (inst.t[2], n * k * 2),
            ],
            (Kind::Down, Precision::Fp8) => vec![
                (inst.t[0], m * n * 2),
                (inst.t[1], m * k),
                (inst.t[2], n * k),
                (inst.t[3], m * 4),
                (inst.t[4], n * 4),
            ],
        };
        for (handle, bytes) in operands {
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
            precision,
            kind,
            tile: tile(model, kind, precision, prog.t),
        });
    }
    Ok(out)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GluArgs {
    out: u64,
    input: u64,
    gate: u64,
    up: u64,
    rows: u32,
    n: u32,
    k: u32,
    act: u32,
}
const _: () = assert!(std::mem::size_of::<GluArgs>() == 48);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LinearArgs {
    out: u64,
    input: u64,
    weight: u64,
    rows: u32,
    n: u32,
    k: u32,
    reserved: u32,
}
const _: () = assert!(std::mem::size_of::<LinearArgs>() == 40);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Fp8Args {
    out: u64,
    input: u64,
    gate: u64,
    up: u64,
    input_scale: u64,
    gate_scale: u64,
    up_scale: u64,
    rows: u32,
    n: u32,
    k: u32,
    act: u32,
}
const _: () = assert!(std::mem::size_of::<Fp8Args>() == 72);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Fp8LinearArgs {
    out: u64,
    input: u64,
    weight: u64,
    input_scale: u64,
    weight_scale: u64,
    rows: u32,
    n: u32,
    k: u32,
    reserved: u32,
}
const _: () = assert!(std::mem::size_of::<Fp8LinearArgs>() == 56);

pub(super) struct Gemma4Glu {
    c2: HsaKernel,
    c5_12: HsaKernel,
    c5_31: HsaKernel,
    fp8_c2_12: Option<HsaKernel>,
    fp8_c5_12: Option<HsaKernel>,
    fp8_c2_31: Option<HsaKernel>,
    fp8_c5_31: Option<HsaKernel>,
    down_c2_12: Option<HsaKernel>,
    down_c5_12: Option<HsaKernel>,
    down_c6_12: Option<HsaKernel>,
    down_c2_31: Option<HsaKernel>,
    down_c5_31: Option<HsaKernel>,
    down_fp8_c2_12: Option<HsaKernel>,
    down_fp8_c5_12: Option<HsaKernel>,
    down_fp8_c2_31: Option<HsaKernel>,
    down_fp8_c3_31: Option<HsaKernel>,
    down_fp8_c5_31: Option<HsaKernel>,
    n_cu: u32,
}

impl Gemma4Glu {
    pub fn load(
        be: &HsaBackend,
        dir: &Path,
        n_cu: u32,
        need_fp8: bool,
        need_down: bool,
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
        if need_fp8 {
            for marker in [
                "plow_gemma4_glu_fp8_128x256x128_1",
                "plow_gemma4_glu_fp8_192x256x128_1",
            ] {
                if !symbols.contains(&marker) {
                    return Err(RuntimeError::Device(format!(
                        "{} lacks required marker `{marker}`",
                        path.display()
                    )));
                }
            }
        }
        if need_down {
            for marker in [
                "plow_gemma4_down_128x256x64_1",
                "plow_gemma4_down_192x256x64_1",
                "plow_gemma4_down_320x128x64_1",
                "plow_gemma4_down_fp8_128x256x128_1",
                "plow_gemma4_down_fp8_128x128x128_1",
                "plow_gemma4_down_fp8_192x256x128_1",
            ] {
                if !symbols.contains(&marker) {
                    return Err(RuntimeError::Device(format!(
                        "{} lacks required marker `{marker}`",
                        path.display()
                    )));
                }
            }
        }
        let module = EngineDevice::module_load(be, &image)?;
        let c2 = EngineDevice::get_function(be, &module, "plow_gemma4_gemm_glu_c2_gfx942")?;
        let c5_12 = EngineDevice::get_function(be, &module, "plow_gemma4_12b_gemm_glu_c5_gfx942")?;
        let c5_31 = EngineDevice::get_function(be, &module, "plow_gemma4_31b_gemm_glu_c5_gfx942")?;
        let load_fp8 = |name| {
            need_fp8
                .then(|| EngineDevice::get_function(be, &module, name))
                .transpose()
        };
        let fp8_c2_12 = load_fp8("plow_gemma4_12b_gemm_glu_fp8_c2_gfx942")?;
        let fp8_c5_12 = load_fp8("plow_gemma4_12b_gemm_glu_fp8_c5_gfx942")?;
        let fp8_c2_31 = load_fp8("plow_gemma4_31b_gemm_glu_fp8_c2_gfx942")?;
        let fp8_c5_31 = load_fp8("plow_gemma4_31b_gemm_glu_fp8_c5_gfx942")?;
        let load_down = |name| {
            need_down
                .then(|| EngineDevice::get_function(be, &module, name))
                .transpose()
        };
        let down_c2_12 = load_down("plow_gemma4_12b_down_c2_gfx942")?;
        let down_c5_12 = load_down("plow_gemma4_12b_down_c5_gfx942")?;
        let down_c6_12 = load_down("plow_gemma4_12b_down_c6_gfx942")?;
        let down_c2_31 = load_down("plow_gemma4_31b_down_c2_gfx942")?;
        let down_c5_31 = load_down("plow_gemma4_31b_down_c5_gfx942")?;
        let down_fp8_c2_12 = load_down("plow_gemma4_12b_down_fp8_c2_gfx942")?;
        let down_fp8_c5_12 = load_down("plow_gemma4_12b_down_fp8_c5_gfx942")?;
        let down_fp8_c2_31 = load_down("plow_gemma4_31b_down_fp8_c2_gfx942")?;
        let down_fp8_c3_31 = load_down("plow_gemma4_31b_down_fp8_c3_gfx942")?;
        let down_fp8_c5_31 = load_down("plow_gemma4_31b_down_fp8_c5_gfx942")?;
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
        for (name, kernel, lds, kernargs) in [
            ("12B down C2", down_c2_12, C2_LDS, [40, 296]),
            ("12B down C5", down_c5_12, C5_LDS, [40, 296]),
            ("12B down C6", down_c6_12, C5_LDS, [40, 296]),
            ("31B down C2", down_c2_31, C2_LDS, [40, 296]),
            ("31B down C5", down_c5_31, C5_LDS, [40, 296]),
            ("12B down FP8 C2", down_fp8_c2_12, 49_152, [56, 312]),
            ("12B down FP8 C5", down_fp8_c5_12, 57_344, [56, 312]),
            ("31B down FP8 C2", down_fp8_c2_31, 49_152, [56, 312]),
            ("31B down FP8 C3", down_fp8_c3_31, 32_768, [56, 312]),
            ("31B down FP8 C5", down_fp8_c5_31, 57_344, [56, 312]),
        ] {
            let Some(kernel) = kernel else { continue };
            let kernarg = kernel.kernarg_size();
            if !kernargs.contains(&kernarg)
                || kernel.private_segment_size() != 0
                || HsaBackend::kernel_lds_bytes(&kernel) != lds
            {
                return Err(RuntimeError::Device(format!(
                    "Gemma-4 {name} resource ABI mismatch: kernarg={kernarg}, LDS={}, private={}",
                    HsaBackend::kernel_lds_bytes(&kernel),
                    kernel.private_segment_size()
                )));
            }
        }
        for (name, kernel, lds) in [
            ("12B FP8 C2", fp8_c2_12, 49_152),
            ("12B FP8 C5", fp8_c5_12, 57_344),
            ("31B FP8 C2", fp8_c2_31, 49_152),
            ("31B FP8 C5", fp8_c5_31, 57_344),
        ] {
            let Some(kernel) = kernel else { continue };
            let kernarg = kernel.kernarg_size();
            if ![72, 328].contains(&kernarg)
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
            fp8_c2_12,
            fp8_c5_12,
            fp8_c2_31,
            fp8_c5_31,
            down_c2_12,
            down_c5_12,
            down_c6_12,
            down_c2_31,
            down_c5_31,
            down_fp8_c2_12,
            down_fp8_c5_12,
            down_fp8_c2_31,
            down_fp8_c3_31,
            down_fp8_c5_31,
            n_cu,
        }))
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, tensor_table: &[u8]) -> Result<()> {
        let pointer = |handle: u16| {
            let at = usize::from(handle) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        };
        match (route.kind, route.precision) {
            (Kind::Glu, Precision::Bf16) => {
                let args = GluArgs {
                    out: pointer(route.inst.t[0]),
                    input: pointer(route.inst.t[1]),
                    gate: pointer(route.inst.t[2]),
                    up: pointer(route.inst.t[5]),
                    rows: route.rows,
                    n: route.inst.i[1],
                    k: route.inst.i[2],
                    act: route.inst.i[5],
                };
                let kernel = match (route.model, route.tile) {
                    (Model::Gemma12 | Model::Gemma31, Tile::C2) => self.c2,
                    (_, Tile::C3) => unreachable!("C3 is only qualified for FP8 down"),
                    (_, Tile::C6) => unreachable!("C6 is only qualified for BF16 down"),
                    (Model::Gemma12, Tile::C5) => self.c5_12,
                    (Model::Gemma31, Tile::C5) => self.c5_31,
                };
                be.launch(kernel, self.n_cu, 512, 0, bytemuck::bytes_of(&args))?;
            }
            (Kind::Glu, Precision::Fp8) => {
                let args = Fp8Args {
                    out: pointer(route.inst.t[0]),
                    input: pointer(route.inst.t[1]),
                    gate: pointer(route.inst.t[2]),
                    up: pointer(route.inst.t[5]),
                    input_scale: pointer(route.inst.t[3]),
                    gate_scale: pointer(route.inst.t[4]),
                    up_scale: pointer(route.inst.t[6]),
                    rows: route.rows,
                    n: route.inst.i[1],
                    k: route.inst.i[2],
                    act: route.inst.i[5],
                };
                let kernel = match (route.model, route.tile) {
                    (Model::Gemma12, Tile::C2) => self.fp8_c2_12,
                    (Model::Gemma12, Tile::C5) => self.fp8_c5_12,
                    (Model::Gemma31, Tile::C2) => self.fp8_c2_31,
                    (_, Tile::C3) => unreachable!("C3 is only qualified for FP8 down"),
                    (_, Tile::C6) => unreachable!("C6 is only qualified for BF16 down"),
                    (Model::Gemma31, Tile::C5) => self.fp8_c5_31,
                }
                .ok_or_else(|| {
                    RuntimeError::Device("native Gemma-4 FP8 GLU kernel was not loaded".into())
                })?;
                be.launch(kernel, self.n_cu, 512, 0, bytemuck::bytes_of(&args))?;
            }
            (Kind::Down, Precision::Bf16) => {
                let args = LinearArgs {
                    out: pointer(route.inst.t[0]),
                    input: pointer(route.inst.t[1]),
                    weight: pointer(route.inst.t[2]),
                    rows: route.rows,
                    n: route.inst.i[1],
                    k: route.inst.i[2],
                    reserved: 0,
                };
                let kernel = match (route.model, route.tile) {
                    (Model::Gemma12, Tile::C2) => self.down_c2_12,
                    (Model::Gemma12, Tile::C5) => self.down_c5_12,
                    (Model::Gemma12, Tile::C6) => self.down_c6_12,
                    (Model::Gemma31, Tile::C2) => self.down_c2_31,
                    (_, Tile::C3) => unreachable!("C3 is only qualified for FP8 down"),
                    (Model::Gemma31, Tile::C5) => self.down_c5_31,
                    (Model::Gemma31, Tile::C6) => unreachable!("31B BF16 down uses C5"),
                }
                .ok_or_else(|| {
                    RuntimeError::Device("native Gemma-4 down kernel was not loaded".into())
                })?;
                be.launch(kernel, self.n_cu, 512, 0, bytemuck::bytes_of(&args))?;
            }
            (Kind::Down, Precision::Fp8) => {
                let args = Fp8LinearArgs {
                    out: pointer(route.inst.t[0]),
                    input: pointer(route.inst.t[1]),
                    weight: pointer(route.inst.t[2]),
                    input_scale: pointer(route.inst.t[3]),
                    weight_scale: pointer(route.inst.t[4]),
                    rows: route.rows,
                    n: route.inst.i[1],
                    k: route.inst.i[2],
                    reserved: 0,
                };
                let kernel = match (route.model, route.tile) {
                    (Model::Gemma12, Tile::C2) => self.down_fp8_c2_12,
                    (Model::Gemma12, Tile::C5) => self.down_fp8_c5_12,
                    (Model::Gemma31, Tile::C2) => self.down_fp8_c2_31,
                    (Model::Gemma31, Tile::C3) => self.down_fp8_c3_31,
                    (Model::Gemma31, Tile::C5) => self.down_fp8_c5_31,
                    (Model::Gemma12, Tile::C3) => unreachable!("12B down uses C2"),
                    (_, Tile::C6) => unreachable!("C6 is only qualified for BF16 down"),
                }
                .ok_or_else(|| {
                    RuntimeError::Device("native Gemma-4 FP8 down kernel was not loaded".into())
                })?;
                be.launch(kernel, self.n_cu, 512, 0, bytemuck::bytes_of(&args))?;
            }
        }
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

    fn fp8_fixture(rows: u32, n: u32, k: u32) -> (DevProg, Vec<DevTensor>) {
        let (mut prog, _) = fixture(rows, n, k);
        prog.insts[0].op = DevOp::GemmGluFp8 as u16;
        prog.insts[0].t = [0, 1, 2, 3, 4, 5, 6, TENSOR_NONE16];
        prog.insts[0].i = [rows, n, k, 0, 0, 0, 0, 0];
        let bytes = [
            u64::from(rows) * u64::from(n) * 2,
            u64::from(rows) * u64::from(k),
            u64::from(n) * u64::from(k),
            u64::from(rows) * 4,
            u64::from(n) * 4,
            u64::from(n) * u64::from(k),
            u64::from(n) * 4,
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

    fn down_fixture(rows: u32, n: u32, k: u32, precision: Precision) -> (DevProg, Vec<DevTensor>) {
        let (mut prog, _) = fixture(rows, n, k);
        let fp8 = precision == Precision::Fp8;
        prog.insts[0].op = if fp8 {
            DevOp::GemmFp8 as u16
        } else {
            DevOp::Gemm as u16
        };
        prog.insts[0].t = if fp8 {
            [0, 1, 2, 3, 4, TENSOR_NONE16, TENSOR_NONE16, TENSOR_NONE16]
        } else {
            [
                0,
                1,
                2,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
            ]
        };
        let bytes = if fp8 {
            vec![
                u64::from(rows) * u64::from(n) * 2,
                u64::from(rows) * u64::from(k),
                u64::from(n) * u64::from(k),
                u64::from(rows) * 4,
                u64::from(n) * 4,
            ]
        } else {
            vec![
                u64::from(rows) * u64::from(n) * 2,
                u64::from(rows) * u64::from(k) * 2,
                u64::from(n) * u64::from(k) * 2,
            ]
        };
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
                type Fixture = fn(u32, u32, u32) -> (DevProg, Vec<DevTensor>);
                for (precision, fixture) in [
                    (Precision::Bf16, fixture as Fixture),
                    (Precision::Fp8, fp8_fixture as Fixture),
                ] {
                    let (prog, tensors) = fixture(rows, n, k);
                    assert!(program_candidate(&prog));
                    assert_eq!(program_fp8_candidate(&prog), precision == Precision::Fp8);
                    let route = routes(&prog, &tensors, 1).unwrap()[0].unwrap();
                    assert_eq!((route.model, route.precision), (model, precision));
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

    #[test]
    fn exact_down_shapes_select_measured_tiles() {
        for (n, k, model) in [
            (3_840, 15_360, Model::Gemma12),
            (5_376, 21_504, Model::Gemma31),
        ] {
            for precision in [Precision::Bf16, Precision::Fp8] {
                for rows in [1024, 2048, 4096, 8192] {
                    let (prog, tensors) = down_fixture(rows, n, k, precision);
                    assert!(program_down_candidate(&prog));
                    assert_eq!(program_fp8_candidate(&prog), precision == Precision::Fp8);
                    let route = routes(&prog, &tensors, 1).unwrap()[0].unwrap();
                    assert_eq!(
                        (route.model, route.kind, route.precision),
                        (model, Kind::Down, precision)
                    );
                    let expected = match (model, precision, rows) {
                        (Model::Gemma12, Precision::Bf16, 8192) => Tile::C6,
                        (Model::Gemma12, _, _) => Tile::C2,
                        (Model::Gemma31, Precision::Bf16, _) => Tile::C5,
                        (Model::Gemma31, Precision::Fp8, 4096) => Tile::C3,
                        (Model::Gemma31, Precision::Fp8, 8192) => Tile::C2,
                        (Model::Gemma31, Precision::Fp8, _) => Tile::C5,
                    };
                    assert_eq!(route.tile, expected);
                }
            }
        }

        let (mut prog, tensors) = down_fixture(4096, 3_840, 15_360, Precision::Fp8);
        prog.insts[0].t[4] = TENSOR_NONE16;
        assert!(routes(&prog, &tensors, 1).unwrap()[0].is_none());
    }
}
