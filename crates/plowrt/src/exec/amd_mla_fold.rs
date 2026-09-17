use std::collections::BTreeSet;
use std::path::Path;

use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};

use super::amd_gemm_lt::{load_kernels, segment_owners, KernelSpec};
use super::device_api::EngineDevice;
use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::{DeviceMem, Module};
use crate::{Result, RuntimeError};

const OBJECT_HASH: &str = "209f4165a46672d5289816f0df6b41cf5a3f44d78452ff2a28c8cd7a0e1b77d1";
const WEIGHT_BYTES: u64 = 8 * 512 * 256 * 4;

/// FP32 W_uv bytes for `heads` heads.
fn weight_bytes(heads: u32) -> u64 {
    u64::from(heads) * 512 * 256 * 4
}

pub(super) fn native(inst: &DevInst64) -> bool {
    inst.op == DevOp::MlaMergeFold as u16 && inst.i[5] != 0
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    inst: DevInst64,
    rows: u32,
    heads: u32,
    /// Heads per attention launch of a group-major row-band fold; 0 for token-major partials.
    group: u32,
}

impl Route {
    pub fn rebase(&mut self, rows: u32) -> Result<()> {
        if self.group != 0 {
            // Row-band fold: this rank's fixed `T/8` rows, selected only for a full chunk.
            if rows != self.inst.i[0] * 8 {
                return Err(RuntimeError::Device(
                    "row-band MLA fold runs only a full chunk".into(),
                ));
            }
            return Ok(());
        }
        if rows == 0 || rows > self.inst.i[0] {
            return Err(RuntimeError::Device(
                "native MLA fold exceeds row capacity".into(),
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
    let mut out = vec![None; segments];
    if !prog.insts.iter().any(native) {
        return Ok(out);
    }
    let owners = segment_owners(prog, segments, native)?;
    for (ix, inst) in prog.insts.iter().enumerate().filter(|(_, d)| native(d)) {
        let err = |s: &str| RuntimeError::Device(format!("native MLA fold {ix}: {s}"));
        // Packed-prefill siblings carry the fold unchanged: it consumes per-row partials and
        // reads no per-row position (class A).
        // Row-band sibling (`PLOW_GLM_ROWBAND_ATTN`): one fold over this rank's `T/8` rows at every
        // head, reading the group-major partials of 16-head attention launches.
        let (heads, group) = (inst.i[1], inst.i[6]);
        let band = prog.role.is_rowsplit_sibling() && heads == 64 && group == 16;
        let band_rows = if band { prog.t / 8 } else { prog.t };
        if !(2048..=8192).contains(&prog.t)
            || inst.i[0] != band_rows
            || !(band || (heads == 8 && group == 0))
            || inst.i[2..4] != [256, 0]
            || !(1..8).contains(&inst.i[4])
            || (band && inst.i[4] != 1)
            || inst.i[5] != 1
            || inst.i[7] != 0
            || inst.fj != [0; 3]
            || inst.t[4..] != [TENSOR_NONE16; 4]
        {
            return Err(err(
                "requires prefill of 2048..8192 rows, eight heads (or a 64-head row band), latent512/value256 and splits<8",
            ));
        }
        let rows = u64::from(band_rows);
        let splits = u64::from(inst.i[4]);
        let h = u64::from(heads);
        let sizes = [
            rows * h * 256 * 2,
            rows * h * splits * 512 * 4,
            rows * h * splits * 2 * 4,
            weight_bytes(heads) / 2,
        ];
        let mut handles = BTreeSet::new();
        for (&handle, bytes) in inst.t[..4].iter().zip(sizes) {
            if handle == TENSOR_NONE16
                || !handles.insert(handle)
                || tensors
                    .get(usize::from(handle))
                    .is_none_or(|t| t.bytes < bytes)
            {
                return Err(err("aliased or undersized operand"));
            }
        }
        out[owners[ix].unwrap()] = Some(Route {
            inst: *inst,
            rows: band_rows,
            heads,
            group,
        });
    }
    Ok(out)
}

fn kernel_index(rows: u32) -> usize {
    match rows {
        0..=128 => 0,
        129..=512 => 1,
        513..=2048 => 2,
        2049..=4464 => 3,
        _ => 4,
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NormalizeArgs {
    out: u64,
    partial: u64,
    ml: u64,
    rows: u32,
    heads: u32,
    splits: u32,
    group: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvertArgs {
    out: u64,
    input: u64,
    rows: u32,
    heads: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmArgs {
    dims: [u32; 8],
    pointers: [u64; 4],
    strides: [u32; 8],
    alpha: f32,
    beta: f32,
    epilogue: [u32; 10],
}
const _: () = assert!(std::mem::size_of::<GemmArgs>() == 144);

fn arguments(rows: u32, heads: u32, weight: u64, input: u64, output: u64, spec: &KernelSpec) -> GemmArgs {
    let grid = 256u32.div_ceil(spec.mt_i) * rows.div_ceil(spec.mt_j) * heads;
    GemmArgs {
        dims: [1, 0x20104001, 0x4c080006, grid, 256, rows, heads, 512],
        pointers: [output, output, weight, input],
        strides: [
            256,
            rows * 256,
            256,
            rows * 256,
            256,
            512 * 256,
            512,
            rows * 512,
        ],
        alpha: 1.0,
        beta: 0.0,
        epilogue: [0; 10],
    }
}

pub(super) struct MlaFold {
    kernels: Vec<HsaKernel>,
    specs: Vec<KernelSpec>,
    normalize: HsaKernel,
    /// `plow_glm_fold_normalize_grouped`, when the adapter carries it.
    normalize_grouped: Option<HsaKernel>,
    convert: HsaKernel,
    _scratch: DeviceMem,
    _weights: DeviceMem,
    normalized: u64,
    product: u64,
    weights: Vec<u64>,
}

impl MlaFold {
    pub fn load<'a>(
        be: &HsaBackend,
        dir: &Path,
        progs: impl IntoIterator<Item = &'a DevProg>,
        tensors: &[DevTensor],
        device: &[DeviceMem],
        modules: &mut Vec<Module>,
    ) -> Result<Self> {
        let specs: Vec<KernelSpec> = serde_json::from_str(include_str!(
            "../../../../runtime/amd/glm_fold_lt_gfx942.json"
        ))
        .map_err(|e| RuntimeError::Device(format!("native MLA fold kernel specification: {e}")))?;
        let kernels = load_kernels(
            be,
            dir,
            "glm_fold_lt_gfx942.elf",
            OBJECT_HASH,
            &specs,
            144,
            modules,
        )?;
        let path = dir.join("glm_fold_adapter.elf");
        let image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        if !super::amd::elf_symbol_names(&image).contains(&"plow_glm_fold_adapter_abi_1") {
            return Err(RuntimeError::Device(
                "native MLA fold adapter lacks ABI marker".into(),
            ));
        }
        let module = EngineDevice::module_load(be, &image)?;
        let normalize = EngineDevice::get_function(be, &module, "plow_glm_fold_normalize")?;
        let convert = EngineDevice::get_function(be, &module, "plow_glm_fold_convert")?;
        let weight_convert = EngineDevice::get_function(be, &module, "plow_glm_fold_weight")?;
        let normalize_grouped = if super::amd::elf_symbol_names(&image).contains(&"plow_glm_fold_grouped_abi_1") {
            Some(EngineDevice::get_function(be, &module, "plow_glm_fold_normalize_grouped")?)
        } else {
            None
        };
        for (k, size) in [(normalize, 36), (convert, 24), (weight_convert, 16)]
            .into_iter()
            .chain(normalize_grouped.map(|k| (k, 40)))
        {
            if ![size, size + 256].contains(&k.kernarg_size())
                || k.private_segment_size() != 0
                || HsaBackend::kernel_lds_bytes(&k) != 0
            {
                return Err(RuntimeError::Device(
                    "native MLA fold helper ABI mismatch".into(),
                ));
            }
        }
        modules.push(module);
        let mut handles = std::collections::BTreeMap::new();
        let mut max_elems = 0u64;
        for prog in progs {
            let segments = prog
                .stream
                .iter()
                .map(|e| usize::from(e.seg) + 1)
                .max()
                .unwrap_or(0);
            for route in routes(prog, tensors, segments)?.into_iter().flatten() {
                if route.group != 0 && normalize_grouped.is_none() {
                    return Err(RuntimeError::Device(
                        "row-band MLA fold needs plow_glm_fold_normalize_grouped \
                         (glm_fold_adapter.elf lacks plow_glm_fold_grouped_abi_1)"
                            .into(),
                    ));
                }
                handles.insert(usize::from(route.inst.t[3]), route.heads);
                max_elems = max_elems.max(u64::from(route.rows) * u64::from(route.heads));
            }
        }
        if handles.is_empty() {
            return Err(RuntimeError::Device("native MLA fold has no routes".into()));
        }
        let norm_bytes = max_elems * 512 * 4;
        let scratch = EngineDevice::alloc(be, norm_bytes + norm_bytes / 2)?;
        let weight_storage =
            EngineDevice::alloc(be, handles.values().map(|&h| weight_bytes(h)).sum::<u64>())?;
        let mut weights = vec![0; tensors.len()];
        let mut at = weight_storage.base;
        for (&handle, &heads) in &handles {
            weights[handle] = at;
            at += weight_bytes(heads);
            let args = [weights[handle], device[handle].base];
            be.launch(
                weight_convert,
                (weight_bytes(heads) / 4 / 256) as u32,
                256,
                0,
                bytemuck::bytes_of(&args),
            )?;
        }
        EngineDevice::synchronize(be)?;
        tracing::info!(
            layers = handles.len(),
            scratch_bytes = scratch.len,
            weight_bytes = weight_storage.len,
            "loaded native FP32 MLA prefill fold"
        );
        Ok(Self {
            kernels,
            specs,
            normalize,
            normalize_grouped,
            convert,
            normalized: scratch.base,
            product: scratch.base + norm_bytes,
            _scratch: scratch,
            _weights: weight_storage,
            weights,
        })
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, table: &[u8]) -> Result<()> {
        let [out, partial, ml] = std::array::from_fn(|i| {
            let at = usize::from(route.inst.t[i]) * 8;
            u64::from_le_bytes(table[at..at + 8].try_into().unwrap())
        });
        let norm = NormalizeArgs {
            out: self.normalized,
            partial,
            ml,
            rows: route.rows,
            heads: route.heads,
            splits: route.inst.i[4],
            group: route.group,
        };
        let (normalize, args_bytes) = match (route.group, self.normalize_grouped) {
            (0, _) => (self.normalize, 36),
            (_, Some(k)) => (k, 40),
            (_, None) => {
                return Err(RuntimeError::Device(
                    "row-band MLA fold has no grouped normalize kernel".into(),
                ))
            }
        };
        be.launch(
            normalize,
            route.rows * route.heads * 2,
            256,
            0,
            &bytemuck::bytes_of(&norm)[..args_bytes],
        )?;
        let index = kernel_index(route.rows);
        let gemm = arguments(
            route.rows,
            route.heads,
            self.weights[usize::from(route.inst.t[3])],
            self.normalized,
            self.product,
            &self.specs[index],
        );
        be.launch(
            self.kernels[index],
            gemm.dims[3],
            256,
            0,
            bytemuck::bytes_of(&gemm),
        )?;
        let convert = ConvertArgs {
            out,
            input: self.product,
            rows: route.rows,
            heads: route.heads,
        };
        be.launch(
            self.convert,
            route.rows * route.heads,
            256,
            0,
            bytemuck::bytes_of(&convert),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::{StreamEnt, SE_XCTR};

    fn fixture() -> (DevProg, Vec<DevTensor>) {
        let prog = DevProg {
            t: 8192,
            role: packet::devbuild::ProgramRole::PrefillBucket { rows: 8192 },
            n_counter: 0,
            insts: vec![DevInst64 {
                op: DevOp::MlaMergeFold as u16,
                t: [
                    0,
                    1,
                    2,
                    3,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                ],
                i: [8192, 8, 256, 0, 7, 1, 0, 0],
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
        let tensors = [
            8192 * 8 * 256 * 2 + 512,
            8192 * 8 * 7 * 512 * 4,
            8192 * 8 * 7 * 2 * 4,
            WEIGHT_BYTES / 2,
        ]
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
    fn native_fold_rejects_bad_routes() {
        let (p, t) = fixture();
        let mut route = routes(&p, &t, 1).unwrap()[0].unwrap();
        for rows in 1..=8192 {
            route.rebase(rows).unwrap();
        }
        assert!(route.rebase(0).is_err());
        assert!(route.rebase(8193).is_err());
        for bad in 0..15 {
            let (mut p, mut t) = fixture();
            match bad {
                0 => p.insts[0].fj[0] = 1,
                1 => p.t = 20,
                2 => p.insts[0].i[0] = 8191,
                3 => p.insts[0].i[1] = 16,
                4 => p.insts[0].i[2] = 128,
                5 => p.insts[0].i[4] = 0,
                6 => p.insts[0].i[4] = 8,
                7 => p.insts[0].i[5] = 2,
                8 => p.insts[0].t[1] = 0,
                9 => t[3].bytes = 16,
                10 => p.stream[0].wait_len = 1,
                11 => p.stream[0].succ_len = 1,
                12 => p.stream[0].flags = SE_XCTR,
                13 => p.stream.clear(),
                _ => {
                    p.insts.push(DevInst64::default());
                    p.stream.push(StreamEnt {
                        inst: 1,
                        seg: 0,
                        ..Default::default()
                    });
                }
            }
            assert!(routes(&p, &t, 1).is_err(), "case {bad}");
        }
    }

    #[test]
    fn row_band_fold_takes_sixty_four_grouped_heads_on_the_sibling_only() {
        let band = |role, i: [u32; 8]| {
            let (mut p, mut t) = fixture();
            p.role = role;
            p.insts[0].i = i;
            let sizes = [1024 * 64 * 256 * 2, 1024 * 64 * 512 * 4, 1024 * 64 * 2 * 4, weight_bytes(64) / 2];
            for (tensor, bytes) in t.iter_mut().zip(sizes) {
                tensor.bytes = bytes;
            }
            routes(&p, &t, 1)
        };
        let sibling = packet::devbuild::ProgramRole::RowSplitSibling { of_rows: 8192 };
        let bucket = packet::devbuild::ProgramRole::PrefillBucket { rows: 8192 };
        let mut route = band(sibling, [1024, 64, 256, 0, 1, 1, 16, 0]).unwrap()[0].unwrap();
        assert_eq!((route.rows, route.heads, route.group), (1024, 64, 16));
        route.rebase(8192).unwrap();
        assert_eq!(route.rows, 1024, "a band keeps its own rows");
        assert!(route.rebase(4096).is_err(), "a band runs only a full chunk");
        assert!(band(bucket, [1024, 64, 256, 0, 1, 1, 16, 0]).is_err(), "64 heads only on the sibling");
        assert!(band(sibling, [1024, 64, 256, 0, 1, 1, 0, 0]).is_err(), "64 heads need the group layout");
        assert!(band(sibling, [1024, 64, 256, 0, 2, 1, 16, 0]).is_err(), "one split per head");
        assert!(band(sibling, [8192, 64, 256, 0, 1, 1, 16, 0]).is_err(), "band rows are T/8");
        let a = arguments(1024, 64, 0, 0, 0, &serde_json::from_str::<Vec<KernelSpec>>(include_str!(
            "../../../../runtime/amd/glm_fold_lt_gfx942.json"
        ))
        .unwrap()[kernel_index(1024)]);
        assert_eq!(a.dims[6], 64);
        assert_eq!(a.dims[3] % 64, 0);
    }

    #[test]
    fn native_fold_arguments_match_library_exports() {
        let specs: Vec<KernelSpec> = serde_json::from_str(include_str!(
            "../../../../runtime/amd/glm_fold_lt_gfx942.json"
        ))
        .unwrap();
        let selected: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../runtime/bench/amd/glm_fold_tail/gemm-selected.json"
        ))
        .unwrap();
        for record in selected["records"].as_array().unwrap() {
            let rows = record["rows"].as_u64().unwrap() as u32;
            let spec = &specs[kernel_index(rows)];
            assert_eq!(spec.name, record["name"].as_str().unwrap());
            let args = arguments(rows, 8, 0, 0, 0, spec);
            let text = record["args_hex"].as_str().unwrap();
            let mut expected: Vec<u8> = text
                .as_bytes()
                .chunks_exact(2)
                .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
                .collect();
            expected.extend([0; 4]);
            assert_eq!(bytemuck::bytes_of(&args), expected);
        }
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_FOLD_DIR"]
    fn native_fold_hsa_pipeline() {
        let be = HsaBackend::new(0).unwrap();
        let (prog, tensors) = fixture();
        let device: Vec<_> = tensors
            .iter()
            .map(|t| EngineDevice::alloc(&be, t.bytes).unwrap())
            .collect();
        let mut w = vec![0u16; (WEIGHT_BYTES / 4) as usize];
        for h in 0..8 {
            for v in 0..256 {
                w[(h * 512 + v) * 256 + v] = (((h + 1) as f32 * 0.5).to_bits() >> 16) as u16;
                w[(h * 512 + v + 256) * 256 + v] =
                    ((-(h as f32 + 1.0) * 0.25).to_bits() >> 16) as u16;
            }
        }
        EngineDevice::upload(&be, &device[3], 0, bytemuck::cast_slice(&w)).unwrap();
        let mut modules = Vec::new();
        let dir = std::env::var("PLOW_TEST_FOLD_DIR").unwrap();
        let kernel = MlaFold::load(
            &be,
            Path::new(&dir),
            std::slice::from_ref(&prog),
            &tensors,
            &device,
            &mut modules,
        )
        .unwrap();
        let table: Vec<_> = device.iter().map(|m| m.base).collect();
        let mut route = routes(&prog, &tensors, 1).unwrap()[0].unwrap();
        let cases = [
            1, 7, 127, 128, 129, 511, 512, 513, 2047, 2048, 2049, 4463, 4464, 4465, 8191, 8192,
        ]
        .into_iter()
        .map(|r| (r, 1))
        .chain(
            [2, 7]
                .into_iter()
                .flat_map(|s| [128, 2048, 8191].map(|r| (r, s))),
        );
        let poison = vec![0x7fu8; device[0].len as usize];
        for (rows, splits) in cases {
            let value = |row: usize, s: usize, k: usize| {
                ((row * 17 + s * 11 + k * 3) % 257) as f32 / 32.0 - 4.0
            };
            let mut partial = vec![0f32; rows * 8 * splits * 512];
            let mut ml = vec![0f32; rows * 8 * splits * 2];
            for row in 0..rows * 8 {
                for s in 0..splits {
                    let dead = row % 13 == 0 || (row % 7 == 0 && s + 1 == splits);
                    ml[(row * splits + s) * 2] = if dead {
                        f32::NEG_INFINITY
                    } else {
                        (s % 3) as f32 - 1.0
                    };
                    ml[(row * splits + s) * 2 + 1] = if dead { 0.0 } else { (1 + row % 3) as f32 };
                    for k in 0..512 {
                        partial[(row * splits + s) * 512 + k] =
                            if dead { 0.0 } else { value(row, s, k) };
                    }
                }
            }
            EngineDevice::upload(&be, &device[1], 0, bytemuck::cast_slice(&partial)).unwrap();
            EngineDevice::upload(&be, &device[2], 0, bytemuck::cast_slice(&ml)).unwrap();
            route.inst.i[4] = splits as u32;
            route.rebase(rows as u32).unwrap();
            for _ in 0..3 {
                EngineDevice::upload(&be, &device[0], 0, &poison).unwrap();
                kernel
                    .enqueue(&be, route, bytemuck::cast_slice(&table))
                    .unwrap();
                be.synchronize().unwrap();
                let mut actual = poison.clone();
                EngineDevice::download(&be, &device[0], 0, &mut actual).unwrap();
                let end = rows * 8 * 256 * 2;
                assert_eq!(&actual[end..], &poison[end..]);
                let mut error = 0.0;
                let mut norm = 0.0;
                for row in 0..rows * 8 {
                    let gm = (0..splits)
                        .map(|s| ml[(row * splits + s) * 2])
                        .fold(f32::NEG_INFINITY, f32::max) as f64;
                    let ex: Vec<_> = (0..splits)
                        .map(|s| {
                            let m = ml[(row * splits + s) * 2] as f64;
                            if m.is_finite() {
                                (m - gm).exp2()
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let sum: f64 = (0..splits)
                        .map(|s| ex[s] * ml[(row * splits + s) * 2 + 1] as f64)
                        .sum();
                    for v in 0..256 {
                        let want = if sum == 0.0 {
                            0.0
                        } else {
                            (0..splits)
                                .map(|s| {
                                    ex[s]
                                        * (0.5 * value(row, s, v) as f64
                                            - 0.25 * value(row, s, v + 256) as f64)
                                })
                                .sum::<f64>()
                                * (row % 8 + 1) as f64
                                / sum
                        };
                        let i = (row * 256 + v) * 2;
                        let got = f32::from_bits(
                            u32::from(u16::from_le_bytes([actual[i], actual[i + 1]])) << 16,
                        ) as f64;
                        assert!(got.is_finite());
                        error += (got - want).powi(2);
                        norm += want * want;
                    }
                }
                assert!(
                    (error / norm).sqrt() < 0.003,
                    "rows={rows} splits={splits} rel={}",
                    (error / norm).sqrt()
                );
            }
            eprintln!("native fold HSA rows={rows} splits={splits} passed");
        }
    }
}
