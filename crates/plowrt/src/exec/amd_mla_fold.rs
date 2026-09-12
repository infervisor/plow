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

pub(super) fn native(inst: &DevInst64) -> bool {
    inst.op == DevOp::MlaMergeFold as u16 && inst.i[5] != 0
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    inst: DevInst64,
    rows: u32,
}

impl Route {
    pub fn rebase(&mut self, rows: u32) -> Result<()> {
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
        let shape_ok = if prog.role.is_decode_rung() {
            matches!(prog.t, 16 | 20) && (1..=64).contains(&inst.i[4])
        } else {
            (2048..=8192).contains(&prog.t) && (1..8).contains(&inst.i[4])
        };
        if !shape_ok
            || inst.i[0] != prog.t
            || inst.i[1..4] != [8, 256, 0]
            || inst.i[5..] != [1, 0, 0]
            || inst.fj != [0; 3]
            || inst.t[4..] != [TENSOR_NONE16; 4]
        {
            return Err(err(
                "requires prefill of 2048..8192 rows with splits<8 or decode rungs 16/20 with \
                 splits<=64, eight heads and latent512/value256",
            ));
        }
        let rows = u64::from(prog.t);
        let splits = u64::from(inst.i[4]);
        let sizes = [
            rows * 8 * 256 * 2,
            rows * 8 * splits * 512 * 4,
            rows * 8 * splits * 2 * 4,
            WEIGHT_BYTES / 2,
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
            rows: prog.t,
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
    pad: u32,
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

fn arguments(rows: u32, weight: u64, input: u64, output: u64, spec: &KernelSpec) -> GemmArgs {
    let grid = 256u32.div_ceil(spec.mt_i) * rows.div_ceil(spec.mt_j) * 8;
    GemmArgs {
        dims: [1, 0x20104001, 0x4c080006, grid, 256, rows, 8, 512],
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
        for (k, size) in [(normalize, 36), (convert, 24), (weight_convert, 16)] {
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
        let mut handles = BTreeSet::new();
        let mut max_rows = 0;
        for prog in progs {
            let segments = prog
                .stream
                .iter()
                .map(|e| usize::from(e.seg) + 1)
                .max()
                .unwrap_or(0);
            for route in routes(prog, tensors, segments)?.into_iter().flatten() {
                handles.insert(usize::from(route.inst.t[3]));
                max_rows = max_rows.max(route.rows);
            }
        }
        if handles.is_empty() {
            return Err(RuntimeError::Device("native MLA fold has no routes".into()));
        }
        let norm_bytes = u64::from(max_rows) * 8 * 512 * 4;
        let scratch = EngineDevice::alloc(be, norm_bytes + norm_bytes / 2)?;
        let weight_storage = EngineDevice::alloc(be, handles.len() as u64 * WEIGHT_BYTES)?;
        let mut weights = vec![0; tensors.len()];
        for (index, handle) in handles.iter().copied().enumerate() {
            weights[handle] = weight_storage.base + index as u64 * WEIGHT_BYTES;
            let args = [weights[handle], device[handle].base];
            be.launch(
                weight_convert,
                (WEIGHT_BYTES / 4 / 256) as u32,
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
            heads: 8,
            splits: route.inst.i[4],
            pad: 0,
        };
        be.launch(
            self.normalize,
            route.rows * 16,
            256,
            0,
            &bytemuck::bytes_of(&norm)[..36],
        )?;
        let index = kernel_index(route.rows);
        let gemm = arguments(
            route.rows,
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
            heads: 8,
        };
        be.launch(
            self.convert,
            route.rows * 8,
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
    fn native_fold_accepts_decode_rungs_16_and_20() {
        let decode = |rows: u32, splits: u32| {
            let (mut p, _) = fixture();
            p.t = rows;
            p.role = packet::devbuild::ProgramRole::DecodeRung { rows };
            p.insts[0].i[0] = rows;
            p.insts[0].i[4] = splits;
            let (r, s) = (u64::from(rows), u64::from(splits.max(1)));
            let tensors: Vec<DevTensor> = [
                r * 8 * 256 * 2,
                r * 8 * s * 512 * 4,
                r * 8 * s * 2 * 4,
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
            (p, tensors)
        };
        for rows in [16, 20] {
            for splits in [1, 16, 64] {
                let (p, t) = decode(rows, splits);
                assert!(routes(&p, &t, 1).unwrap()[0].is_some(), "rows={rows} splits={splits}");
            }
        }
        for (rows, splits) in [(8, 16), (1, 16), (20, 0), (20, 65)] {
            let (p, t) = decode(rows, splits);
            assert!(routes(&p, &t, 1).is_err(), "rows={rows} splits={splits}");
        }
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
            let args = arguments(rows, 0, 0, 0, spec);
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

    /// Tier 2 for `PLOW_GLM_FOLD_LT_DECODE`: per-layer time of the native fold (normalize, FP32
    /// GEMM, convert) against the interpreter's `d_mla_merge_fold<512, 256>` body at the rung-20
    /// geometry (160 workgroups x 512 threads; `runtime/bench/amd/glm53_kbench_fold_decode.hip`),
    /// over 78 distinct layers of W_uv so every layer's weights are cold, as in a decode step.
    /// Both outputs are checked against FP64. PLOW_TEST_FOLD_ROWS (16 or 20, default 20) and
    /// PLOW_TEST_FOLD_SPLITS (default 16; the production packet runs 4 at rung 20, 8 at rung 16).
    #[test]
    #[ignore = "requires a gfx942 GPU lease, PLOW_TEST_FOLD_DIR and PLOW_TEST_FOLD_KBENCH"]
    fn native_fold_decode_rung20_timing() {
        const LAYERS: usize = 78;
        let rows: u32 = std::env::var("PLOW_TEST_FOLD_ROWS")
            .map(|v| v.parse().unwrap())
            .unwrap_or(20);
        const REPS: usize = 30;
        let splits: u32 = std::env::var("PLOW_TEST_FOLD_SPLITS")
            .map(|v| v.parse().unwrap())
            .unwrap_or(16);
        let be = HsaBackend::new(0).unwrap();
        let (r, s) = (rows as usize, splits as usize);
        let mut sizes = vec![
            (r * 8 * 256 * 2) as u64,
            (r * 8 * s * 512 * 4) as u64,
            (r * 8 * s * 2 * 4) as u64,
        ];
        sizes.extend([WEIGHT_BYTES / 2; LAYERS]);
        sizes.push((r * 8 * 256 * 2) as u64);
        let tensors: Vec<DevTensor> = sizes
            .iter()
            .enumerate()
            .map(|(i, &bytes)| DevTensor {
                name: i.to_string(),
                bytes,
                init: None,
            })
            .collect();
        let device: Vec<_> = tensors
            .iter()
            .map(|t| EngineDevice::alloc(&be, t.bytes).unwrap())
            .collect();
        let interp_out = LAYERS + 3;
        let prog = DevProg {
            t: rows,
            role: packet::devbuild::ProgramRole::DecodeRung { rows: rows },
            n_counter: 0,
            insts: (0..LAYERS)
                .map(|l| DevInst64 {
                    op: DevOp::MlaMergeFold as u16,
                    t: [
                        0,
                        1,
                        2,
                        3 + l as u16,
                        TENSOR_NONE16,
                        TENSOR_NONE16,
                        TENSOR_NONE16,
                        TENSOR_NONE16,
                    ],
                    i: [rows, 8, 256, 0, splits, 1, 0, 0],
                    ..Default::default()
                })
                .collect(),
            stream: (0..LAYERS)
                .map(|l| StreamEnt {
                    inst: l as u32,
                    seg: l as u16,
                    ..Default::default()
                })
                .collect(),
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_domains: 0,
        };
        let mut seed = 0x9e3779b9u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) & 0xffff) as f32 / 32768.0 - 1.0
        };
        let bf = |x: f32| ((x.to_bits() + 0x7fff + ((x.to_bits() >> 16) & 1)) >> 16) as u16;
        let unbf = |h: u16| f32::from_bits(u32::from(h) << 16);
        let weights: Vec<Vec<u16>> = (0..LAYERS)
            .map(|_| (0..WEIGHT_BYTES / 4).map(|_| bf(rnd() * 0.03)).collect())
            .collect();
        for (l, w) in weights.iter().enumerate() {
            EngineDevice::upload(&be, &device[3 + l], 0, bytemuck::cast_slice(w)).unwrap();
        }
        let mut ml = vec![0f32; r * 8 * s * 2];
        let mut partial = vec![0f32; r * 8 * s * 512];
        for row in 0..r * 8 {
            for sp in 0..s {
                let dead = sp > 0 && (row * 7 + sp) % 19 == 0;
                let l = 1.0 + 20.0 * (rnd() + 1.0);
                ml[(row * s + sp) * 2] = if dead { f32::NEG_INFINITY } else { 2.0 * rnd() };
                ml[(row * s + sp) * 2 + 1] = if dead { 0.0 } else { l };
                for k in 0..512 {
                    partial[(row * s + sp) * 512 + k] = if dead { 0.0 } else { l * 0.35 * rnd() };
                }
            }
        }
        EngineDevice::upload(&be, &device[1], 0, bytemuck::cast_slice(&partial)).unwrap();
        EngineDevice::upload(&be, &device[2], 0, bytemuck::cast_slice(&ml)).unwrap();
        let mut modules = Vec::new();
        let dir = std::env::var("PLOW_TEST_FOLD_DIR").unwrap();
        let fold = MlaFold::load(
            &be,
            Path::new(&dir),
            std::slice::from_ref(&prog),
            &tensors,
            &device,
            &mut modules,
        )
        .unwrap();
        let routes: Vec<Route> = routes(&prog, &tensors, LAYERS)
            .unwrap()
            .into_iter()
            .map(Option::unwrap)
            .collect();
        let image = std::fs::read(std::env::var("PLOW_TEST_FOLD_KBENCH").unwrap()).unwrap();
        let module = EngineDevice::module_load(&be, &image).unwrap();
        let kbench = EngineDevice::get_function(&be, &module, "plow_kb_fold_decode").unwrap();
        let adapter = EngineDevice::module_load(
            &be,
            &std::fs::read(Path::new(&dir).join("glm_fold_adapter.elf")).unwrap(),
        )
        .unwrap();
        let noop = EngineDevice::get_function(&be, &adapter, "plow_glm_fold_convert").unwrap();
        let table: Vec<u64> = device.iter().map(|m| m.base).collect();
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct KbArgs {
            out: u64,
            partial: u64,
            ml: u64,
            wuv: u64,
            n_batch: u32,
            n_head: u32,
            v: u32,
            nsplit: u32,
            nblk: u32,
            pad: u32,
        }
        let nblk = rows * 8;
        let interp = |l: usize| {
            let a = KbArgs {
                out: device[interp_out].base,
                partial: device[1].base,
                ml: device[2].base,
                wuv: device[3 + l].base,
                n_batch: rows,
                n_head: 8,
                v: 256,
                nsplit: splits,
                nblk,
                pad: 0,
            };
            // The kernel declares 52 bytes (no hidden arguments); the struct pads to 56.
            be.launch(kbench, nblk, 512, 0, &bytemuck::bytes_of(&a)[..52]).unwrap();
        };
        let floor_args = ConvertArgs {
            out: device[0].base,
            input: fold.product,
            rows: 1,
            heads: 8,
        };
        let mut time = |name: &str, f: &mut dyn FnMut(usize)| -> f64 {
            for l in 0..LAYERS {
                f(l);
            }
            be.synchronize().unwrap();
            let mut per_layer = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                let t0 = std::time::Instant::now();
                for l in 0..LAYERS {
                    f(l);
                }
                be.synchronize().unwrap();
                per_layer.push(t0.elapsed().as_secs_f64() * 1e6 / LAYERS as f64);
            }
            per_layer.sort_by(f64::total_cmp);
            let median = per_layer[REPS / 2];
            eprintln!(
                "TIER2 {name}: median {median:.2} us/layer (min {:.2}, max {:.2}) rows={rows} splits={splits}",
                per_layer[0],
                per_layer[REPS - 1]
            );
            median
        };
        let t_native = time("native", &mut |l| {
            fold.enqueue(&be, routes[l], bytemuck::cast_slice(&table)).unwrap()
        });
        let t_interp = time("interp", &mut |l| interp(l));
        let t_floor = time("dispatch_floor", &mut |_| {
            be.launch(noop, 8, 256, 0, bytemuck::bytes_of(&floor_args)).unwrap()
        });
        eprintln!(
            "TIER2 summary: interp {t_interp:.2} - floor {t_floor:.2} = {:.2} us body; native {t_native:.2} us (3 launches) -> {:+.2} us/layer, x78 = {:+.2} ms/step",
            t_interp - t_floor,
            t_native - (t_interp - t_floor),
            (t_native - (t_interp - t_floor)) * 78.0 / 1000.0
        );
        for (name, run) in [("native", 0usize), ("interp", 1)] {
            if run == 0 {
                fold.enqueue(&be, routes[0], bytemuck::cast_slice(&table)).unwrap();
            } else {
                interp(0);
            }
            be.synchronize().unwrap();
            let handle = if run == 0 { 0 } else { interp_out };
            let mut out = vec![0u16; r * 8 * 256];
            EngineDevice::download(&be, &device[handle], 0, bytemuck::cast_slice_mut(&mut out))
                .unwrap();
            let (mut err, mut norm) = (0f64, 0f64);
            for row in 0..r * 8 {
                let m = &ml[row * s * 2..(row + 1) * s * 2];
                let gm = (0..s).map(|i| m[i * 2] as f64).fold(f64::NEG_INFINITY, f64::max);
                let wt: Vec<f64> = (0..s)
                    .map(|i| if m[i * 2].is_finite() { (m[i * 2] as f64 - gm).exp2() } else { 0.0 })
                    .collect();
                let gl: f64 = (0..s).map(|i| wt[i] * m[i * 2 + 1] as f64).sum();
                let olat: Vec<f64> = (0..512)
                    .map(|k| {
                        (0..s).map(|i| wt[i] * partial[(row * s + i) * 512 + k] as f64).sum::<f64>()
                            / gl
                    })
                    .collect();
                let w = &weights[0];
                let h = row % 8;
                for v in 0..256 {
                    let want: f64 = (0..512)
                        .map(|k| olat[k] * unbf(w[(h * 512 + k) * 256 + v]) as f64)
                        .sum();
                    let got = unbf(out[row * 256 + v]) as f64;
                    err += (got - want).powi(2);
                    norm += want * want;
                }
            }
            let rel = (err / norm).sqrt();
            eprintln!("TIER2 {name} rel-L2 vs FP64 {rel:.3e}");
            assert!(rel < 0.003, "{name} rel={rel}");
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
