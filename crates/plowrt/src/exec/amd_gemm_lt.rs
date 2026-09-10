use std::path::Path;

use packet::dev::{DevInst64, DevOp, SE_XCTR, TENSOR_NONE16};
use serde::Deserialize;

use super::device_api::EngineDevice;
use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::Module;
use crate::{Result, RuntimeError};

const OBJECT_HASH: &str = "efa5b0365bedc2effa52265c85eded14fb63febd9c067bab37138d99db607db5";

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    inst: DevInst64,
    rows: u32,
}

impl Route {
    pub fn rebase(&mut self, rows: u32) -> Result<()> {
        if rows == 0 || rows > self.inst.i[0] || (self.inst.i[3] == 1 && rows != self.inst.i[0]) {
            return Err(RuntimeError::Device(
                "hipBLASLt chunk exceeds row capacity".into(),
            ));
        }
        self.rows = rows;
        Ok(())
    }

    fn kernel_choice(self) -> (usize, u32) {
        if self.inst.i[3] == 1 {
            return decode_choice(self.rows, self.inst.i[1], self.inst.i[2]).unwrap();
        }
        let tail = self.rows <= 4464;
        let kv = self.inst.i[1] == 512;
        let index = usize::from(kv) * 2 + usize::from(tail);
        let mapping = if kv {
            4
        } else if self.inst.i[1] == 2048 && tail {
            8
        } else {
            6
        };
        (index, (8 << 16) | mapping)
    }
}

fn decode_choice(rows: u32, n: u32, k: u32) -> Option<(usize, u32)> {
    Some(match (rows, n, k) {
        (16, 512, 6144) => (5, 524289),
        (16, 2048, 6144) => (6, 524289),
        (16, 4096, 2048) => (8, 524289),
        (16, 6144, 2048) => (11, 524289),
        (20, 512, 6144) => (5, 524294),
        (20, 2048, 6144) => (5, 524296),
        (20, 4096, 2048) => (9, 524294),
        (20, 6144, 2048) => (10, 524289),
        _ => return None,
    })
}

pub(super) fn routes(
    prog: &DevProg,
    tensors: &[DevTensor],
    segments: usize,
) -> Result<Vec<Option<Route>>> {
    let mut routes = vec![None; segments];
    if !prog.insts.iter().any(|i| i.op == DevOp::GemmLtPf as u16) {
        return Ok(routes);
    }
    let mut owners = vec![None; prog.insts.len()];
    let mut segment_first = vec![None; segments];
    let mut mixed = vec![false; segments];
    for entry in prog.stream.iter().chain(&prog.gq_stream) {
        let ix = entry.inst as usize;
        let seg = entry.seg as usize;
        if let Some(first) = segment_first.get_mut(seg) {
            if first.is_some_and(|prev| prev != ix) {
                mixed[seg] = true;
            }
            first.get_or_insert(ix);
        }
        if prog
            .insts
            .get(ix)
            .is_none_or(|i| i.op != DevOp::GemmLtPf as u16)
        {
            continue;
        }
        let err = |s: &str| RuntimeError::Device(format!("hipBLASLt instruction {ix}: {s}"));
        if seg >= segments {
            return Err(err("segment is outside program"));
        }
        if entry.wait_len != 0 || entry.succ_len != 0 || entry.flags & SE_XCTR != 0 {
            return Err(err(
                "native segment retains interpreter counter obligations",
            ));
        }
        if owners[ix].is_some_and(|owner| owner != seg) {
            return Err(err("requires exactly one segment owner"));
        }
        owners[ix] = Some(seg);
    }
    for (ix, inst) in prog.insts.iter().enumerate() {
        if inst.op != DevOp::GemmLtPf as u16 {
            continue;
        }
        let err = |s: &str| RuntimeError::Device(format!("hipBLASLt instruction {ix}: {s}"));
        let shape_ok = match inst.i[3] {
            0 => {
                (2048..=8192).contains(&prog.t)
                    && matches!(
                        (inst.i[1], inst.i[2]),
                        (2048, 6144) | (512, 6144) | (4096, 2048)
                    )
            }
            1 => decode_choice(prog.t, inst.i[1], inst.i[2]).is_some(),
            _ => false,
        };
        if prog.packed_prefill_only
            || !shape_ok
            || inst.i[0] != prog.t
            || inst.i[4..] != [0; 4]
            || inst.fj != [0; 3]
            || inst.t[3..] != [TENSOR_NONE16; 5]
        {
            return Err(err("requires an unpacked qualified BF16 projection"));
        }
        if inst.t[0] == inst.t[1] || inst.t[0] == inst.t[2] || inst.t[1] == inst.t[2] {
            return Err(err("operands alias"));
        }
        let [m, n, k] = [
            u64::from(prog.t),
            u64::from(inst.i[1]),
            u64::from(inst.i[2]),
        ];
        for (&handle, bytes) in inst.t[..3].iter().zip([m * n * 2, m * k * 2, n * k * 2]) {
            if handle == TENSOR_NONE16
                || tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes)
            {
                return Err(err("operand capacity is insufficient"));
            }
        }
        let seg = owners[ix].ok_or_else(|| err("requires exactly one segment owner"))?;
        if mixed[seg] {
            return Err(err("segment contains other interpreter work"));
        }
        routes[seg] = Some(Route {
            inst: *inst,
            rows: prog.t,
        });
    }
    Ok(routes)
}

#[derive(Deserialize)]
struct KernelSpec {
    name: String,
    kernarg_offset: usize,
    lds: u32,
    mt_i: u32,
    mt_j: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    dims: [u32; 8],
    pointers: [u64; 4],
    strides: [u32; 8],
    alpha: f32,
    beta: f32,
    epilogue: [u32; 14],
}
const _: () = assert!(std::mem::size_of::<Args>() == 160);

fn arguments(route: Route, tensors: [u64; 3], spec: &KernelSpec, info1: u32) -> Args {
    let [out, x, weight] = tensors;
    let (m, n, k) = (route.rows, route.inst.i[1], route.inst.i[2]);
    let grid = n.div_ceil(spec.mt_i) * m.div_ceil(spec.mt_j);
    let mut epilogue = [0; 14];
    // The optional dstD address begins at byte 140, without eight-byte alignment.
    epilogue[9] = out as u32;
    epilogue[10] = (out >> 32) as u32;
    Args {
        dims: [1, 1, info1, grid, n, m, 1, k],
        pointers: [out, out, weight, x],
        strides: [n, n * m, n, n * m, k, k * n, k, k * m],
        alpha: 1.0,
        beta: 0.0,
        epilogue,
    }
}

pub(super) struct GemmLt {
    kernels: Vec<HsaKernel>,
    specs: Vec<KernelSpec>,
}

impl GemmLt {
    pub fn load(be: &HsaBackend, dir: &Path, modules: &mut Vec<Module>) -> Result<Self> {
        let mut specs: Vec<KernelSpec> =
            serde_json::from_str(include_str!("../../../../runtime/amd/glm_lt_gfx942.json"))
                .map_err(|e| {
                    RuntimeError::Device(format!("hipBLASLt kernel specification: {e}"))
                })?;
        let decode_specs: Vec<KernelSpec> = serde_json::from_str(include_str!(
            "../../../../runtime/amd/glm_lt_decode_gfx942.json"
        ))
        .map_err(|e| RuntimeError::Device(format!("hipBLASLt decode specification: {e}")))?;
        specs.extend(decode_specs);
        let path = dir.join("glm_lt_gfx942.elf");
        let mut image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        if plow_asset::decode_objects::image_sha256(&image) != OBJECT_HASH {
            return Err(RuntimeError::Device(
                "hipBLASLt object hash does not match qualified ABI".into(),
            ));
        }
        for spec in &specs {
            // The pinned descriptors omit KERNARG_SIZE; the inspected notes declare 160 bytes.
            let field = image
                .get_mut(spec.kernarg_offset..spec.kernarg_offset + 4)
                .ok_or_else(|| {
                    RuntimeError::Device("hipBLASLt descriptor is outside object".into())
                })?;
            if field != [0; 4] {
                return Err(RuntimeError::Device(
                    "hipBLASLt descriptor differs from qualified ABI".into(),
                ));
            }
            field.copy_from_slice(&160u32.to_le_bytes());
        }
        let module = EngineDevice::module_load(be, &image)?;
        let mut kernels = Vec::new();
        for spec in &specs {
            let kernel = EngineDevice::get_function(be, &module, &spec.name)?;
            if kernel.kernarg_size() != 160
                || kernel.private_segment_size() != 0
                || HsaBackend::kernel_lds_bytes(&kernel) != spec.lds
            {
                return Err(RuntimeError::Device(
                    "hipBLASLt resource ABI mismatch".into(),
                ));
            }
            kernels.push(kernel);
        }
        modules.push(module);
        Ok(Self { kernels, specs })
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, tensor_table: &[u8]) -> Result<()> {
        let tensors = std::array::from_fn(|i| {
            let at = usize::from(route.inst.t[i]) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        });
        let (index, info1) = route.kernel_choice();
        let args = arguments(route, tensors, &self.specs[index], info1);
        be.launch(
            self.kernels[index],
            args.dims[3],
            256,
            0,
            bytemuck::bytes_of(&args),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::StreamEnt;

    fn fixture() -> (DevProg, Vec<DevTensor>) {
        let prog = DevProg {
            t: 8192,
            packed_prefill_only: false,
            n_counter: 0,
            insts: vec![DevInst64 {
                op: DevOp::GemmLtPf as u16,
                t: [
                    0,
                    1,
                    2,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                ],
                i: [8192, 2048, 6144, 0, 0, 0, 0, 0],
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
        let tensors = (0..3)
            .map(|i| DevTensor {
                name: i.to_string(),
                bytes: 1 << 30,
                init: None,
            })
            .collect();
        (prog, tensors)
    }

    #[test]
    fn lt_route_rejects_invalid_shapes_bindings_and_segments() {
        let (p, t) = fixture();
        assert!(routes(&p, &t, 1).unwrap()[0].is_some());
        for bad in 0..16 {
            let (mut p, mut t) = fixture();
            match bad {
                0 => p.packed_prefill_only = true,
                1 => p.insts[0].i[0] = 4096,
                2 => p.insts[0].i[1] = 256,
                3 => p.insts[0].i[4] = 1,
                4 => p.insts[0].fj[0] = 1,
                5 => p.insts[0].t[3] = 0,
                6 => p.insts[0].t[1] = 0,
                7 => t[2].bytes = 16,
                8 => p.stream[0].wait_len = 1,
                9 => p.stream[0].succ_len = 1,
                10 => p.stream[0].flags = SE_XCTR,
                11 => p.stream.push(StreamEnt {
                    inst: 1,
                    seg: 0,
                    ..Default::default()
                }),
                12 => p.stream.clear(),
                13 => p.stream[0].seg = 1,
                14 => p.gq_stream.push(StreamEnt {
                    inst: 1,
                    seg: 0,
                    ..Default::default()
                }),
                _ => p.gq_stream.push(StreamEnt {
                    inst: 0,
                    seg: 1,
                    ..Default::default()
                }),
            }
            assert!(
                routes(&p, &t, if bad == 15 { 2 } else { 1 }).is_err(),
                "case {bad}"
            );
        }
    }

    #[test]
    fn lt_arguments_preserve_live_rows_and_inline_abi() {
        let mut specs: Vec<KernelSpec> =
            serde_json::from_str(include_str!("../../../../runtime/amd/glm_lt_gfx942.json"))
                .unwrap();
        for (n, k) in [(2048, 6144), (512, 6144), (4096, 2048)] {
            let (mut p, t) = fixture();
            p.insts[0].i[1] = n;
            p.insts[0].i[2] = k;
            let mut route = routes(&p, &t, 1).unwrap()[0].unwrap();
            for rows in [1, 129, 2048, 4096, 4464, 4465, 8192] {
                route.rebase(rows).unwrap();
                let (index, info1) = route.kernel_choice();
                let args = arguments(route, [0x1122334455667788, 32, 64], &specs[index], info1);
                let raw = bytemuck::bytes_of(&args);
                assert_eq!(u32::from_le_bytes(raw[20..24].try_into().unwrap()), rows);
                assert_eq!(
                    u64::from_le_bytes(raw[140..148].try_into().unwrap()),
                    0x1122334455667788
                );
                assert_eq!(
                    args.pointers,
                    [0x1122334455667788, 0x1122334455667788, 64, 32]
                );
                assert_eq!(index, usize::from(n == 512) * 2 + usize::from(rows <= 4464));
                assert_eq!(
                    args.dims[3],
                    n.div_ceil(specs[index].mt_i) * rows.div_ceil(specs[index].mt_j)
                );
            }
            assert!(route.rebase(0).is_err());
            assert!(route.rebase(8193).is_err());
        }
    }
    #[test]
    fn decode_routes_check_mode_geometry_and_rebasing() {
        for rows in [16, 20] {
            for (n, k) in [(2048, 6144), (512, 6144), (4096, 2048), (6144, 2048)] {
                let (mut p, t) = fixture();
                p.t = rows;
                p.insts[0].i = [rows, n, k, 1, 0, 0, 0, 0];
                let mut route = routes(&p, &t, 1).unwrap()[0].unwrap();
                let (index, _) = route.kernel_choice();
                assert!((4..12).contains(&index));
                route.rebase(rows).unwrap();
                assert!(route.rebase(rows - 1).is_err());
                p.insts[0].i[3] = 0;
                assert!(routes(&p, &t, 1).is_err());
                p.insts[0].i[3] = 1;
                p.t = 8;
                p.insts[0].i[0] = 8;
                assert!(routes(&p, &t, 1).is_err());
            }
        }
    }

}
