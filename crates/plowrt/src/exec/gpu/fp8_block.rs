use super::*;
use plow_asset::segment_roles::{fp8_block_cfg, NATIVE_DECODE_FP8_BLOCK_MAX_ROWS};
use std::collections::BTreeMap;

const TMAP_UINT8: u32 = 0;
const SWIZZLE_128B: u32 = 3;
const L2_PROMOTION_256B: u32 = 3;
const GROUP: u32 = 128;

#[repr(C, align(128))]
struct TensorMap([u8; 128]);

struct Kernel {
    function: KernelFn,
    threads: u32,
    smem: u32,
    per_sm: u32,
}

/// The packet's native block-scaled FP8 decode GEMM object and its shared split scratch. Routes
/// run serially on the engine stream, so one partial buffer and one set of per-tile counters
/// (self-resetting) serve every plan.
pub(super) struct Fp8Block {
    be: Arc<CudaBackend>,
    _module: Arc<DecodeModule>,
    kernels: BTreeMap<(u32, u32), Kernel>,
    reduce: KernelFn,
    partial: DeviceMem,
    count: DeviceMem,
    plan: Vec<[u32; 5]>,
}

pub(super) struct Plan {
    owner: Arc<Fp8Block>,
    kernel: (u32, u32),
    grid: u32,
    shape: [u32; 5],
    fused: bool,
    input: u64,
    weight: u64,
    a_scale: u64,
    w_scale: u64,
    maps: Box<[TensorMap; 2]>,
}

fn tile_rows(m: u32) -> u32 {
    if m <= 8 {
        8
    } else if m <= 16 {
        16
    } else {
        32
    }
}

/// `fp8b_smem_bytes` of op_gemm_fp8_block_sm90.cuh; the kernel traps on less.
fn smem_bytes(mt: u32, stages: u32, nc: u32) -> u32 {
    stages * (64 * nc * GROUP + (mt * GROUP).next_multiple_of(1024))
        + stages * mt * 4
        + stages * 16
        + 16
        + 1024
}

fn consumers(variant: u32) -> u32 {
    if variant & 1 != 0 {
        1
    } else {
        2
    }
}

impl Fp8Block {
    /// The declared object with a kernel and scratch for every planned shape; `None` with
    /// `PLOW_NV_FP8_BLOCK_NATIVE=0`.
    pub(super) fn load(
        be: &Arc<CudaBackend>,
        assets: &Path,
        object: &SegmentObject,
        profile: &str,
    ) -> Result<Option<Arc<Self>>> {
        let plan = object
            .decode_plan
            .clone()
            .ok_or_else(|| RuntimeError::Rejected("native FP8 block object has no plan".into()))?;
        if !crate::config::RuntimeConfig::get().nv.fp8_block_native {
            return Ok(None);
        }
        let image = std::fs::read(assets.join(&object.file))
            .map_err(|e| RuntimeError::Rejected(format!("native FP8 block object: {e}")))?;
        if profile != "sm90a"
            || plow_asset::cubin::inspect(&image).is_none_or(|i| i.sm != 90)
            || object.sha256.as_deref()
                != Some(plow_asset::decode_objects::image_sha256(&image).as_str())
            || plow_asset::cubin::global_u32(&image, "plow_gemm_fp8_block_abi") != Some(1)
            || plow_asset::cubin::global_u32(&image, "plow_gemm_fp8_block_max_rows")
                != Some(NATIVE_DECODE_FP8_BLOCK_MAX_ROWS)
        {
            return Err(RuntimeError::Rejected(
                "native FP8 block object hash or ABI mismatch".into(),
            ));
        }
        let used = &plan;
        let mut partial_bytes = 4u64;
        let mut tiles = 1u64;
        for &[m, n, _, _, splits] in used {
            if splits > 1 {
                partial_bytes =
                    partial_bytes.max(u64::from(splits) * u64::from(m) * u64::from(n) * 4);
            }
            tiles = tiles.max(u64::from(n / 64));
        }
        let module = DecodeModule::load(be, &image)?;
        let mut kernels = BTreeMap::new();
        for &[m, _, _, cfg, _] in used {
            let mt = tile_rows(m);
            if kernels.contains_key(&(mt, cfg)) {
                continue;
            }
            let (stages, variant) = fp8_block_cfg(cfg)
                .ok_or_else(|| RuntimeError::Rejected("native FP8 block cfg".into()))?;
            let nc = consumers(variant);
            let function = be.get_function(
                &module,
                &format!("plow_gemm_fp8_block_m{mt}_s{stages}_v{variant}"),
            )?;
            let smem = smem_bytes(mt, stages, nc);
            be.set_max_dynamic_smem(function, smem)?;
            let threads = 128 * (nc + 1);
            let per_sm = be.occupancy_blocks_per_sm(function, threads, smem as usize)?;
            if per_sm == 0 {
                return Err(RuntimeError::Rejected(
                    "native FP8 block kernel does not fit an SM".into(),
                ));
            }
            kernels.insert(
                (mt, cfg),
                Kernel {
                    function,
                    threads,
                    smem,
                    per_sm,
                },
            );
        }
        let reduce = be.get_function(&module, "plow_gemm_fp8_block_reduce")?;
        let partial = be.alloc(0, partial_bytes)?;
        let count = be.alloc(0, tiles * 4)?;
        be.memset_d8(count.base, 0, (tiles * 4) as usize)?;
        tracing::info!(
            shapes = used.len(),
            kernels = kernels.len(),
            partial_bytes,
            "native FP8 block decode object loaded"
        );
        Ok(Some(Arc::new(Self {
            be: Arc::clone(be),
            _module: module,
            kernels,
            reduce,
            partial,
            count,
            plan,
        })))
    }

    /// A plan for one block-FP8 projection, or `None` when the object has no plan for its shape.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plan(
        self: &Arc<Self>,
        m: u32,
        n: u32,
        k: u32,
        input: u64,
        weight: u64,
        a_scale: u64,
        w_scale: u64,
    ) -> Result<Option<Plan>> {
        let Some(&[_, _, _, cfg, splits]) = self.plan.iter().find(|p| p[..3] == [m, n, k]) else {
            return Ok(None);
        };
        let reject = |what: &str| RuntimeError::Rejected(format!("native FP8 block plan: {what}"));
        let mt = tile_rows(m);
        let kernel = self
            .kernels
            .get(&(mt, cfg))
            .ok_or_else(|| reject("kernel not loaded"))?;
        let (_, variant) = fp8_block_cfg(cfg).ok_or_else(|| reject("cfg"))?;
        let nc = consumers(variant);
        if splits > 1 && u64::from(splits) * u64::from(m) * u64::from(n) * 4 > self.partial.len {
            return Err(reject("split scratch exceeds owner"));
        }
        if u64::from(n / 64) * 4 > self.count.len || input % 16 != 0 || weight % 16 != 0 {
            return Err(reject("counters or operand alignment"));
        }
        let items = n / (64 * nc) * splits;
        let grid = items.min(self.be.sm_count() * kernel.per_sm);
        let encode = |base: u64, rows: u32, box_rows: u32| {
            self.be
                .encode_tmap_tiled(
                    TMAP_UINT8,
                    base,
                    &[u64::from(k), u64::from(rows)],
                    &[u64::from(k)],
                    &[GROUP, box_rows],
                    &[1, 1],
                    0,
                    SWIZZLE_128B,
                    L2_PROMOTION_256B,
                    0,
                )
                .map(TensorMap)
        };
        // x rows past `m` are out of the map's bounds: the TMA zero-fills them.
        let maps = Box::new([encode(weight, n, 64 * nc)?, encode(input, m, mt)?]);
        Ok(Some(Plan {
            owner: Arc::clone(self),
            kernel: (mt, cfg),
            grid,
            shape: [m, m.next_multiple_of(4), n, k, splits],
            fused: variant & 4 != 0,
            input,
            weight,
            a_scale,
            w_scale,
            maps,
        }))
    }
}

impl Plan {
    pub(super) fn run(
        &self,
        input: u64,
        weight: u64,
        output: u64,
        stream: &CudaStream,
    ) -> Result<()> {
        if (input, weight) != (self.input, self.weight) {
            return Err(RuntimeError::Rejected(
                "native FP8 block plan operands changed".into(),
            ));
        }
        let owner = &self.owner;
        let kernel = &owner.kernels[&self.kernel];
        let [mut m, mut m4, mut n, mut k, mut splits] = self.shape;
        let (mut out, mut partial, mut count) = (output, owner.partial.base, owner.count.base);
        let (mut a_scale, mut w_scale) = (self.a_scale, self.w_scale);
        let [w_map, x_map] = &*self.maps;
        let mut params = [
            (&mut out as *mut u64).cast(),
            (&mut partial as *mut u64).cast(),
            (&mut count as *mut u64).cast(),
            (w_map as *const TensorMap).cast_mut().cast(),
            (x_map as *const TensorMap).cast_mut().cast(),
            (&mut a_scale as *mut u64).cast(),
            (&mut w_scale as *mut u64).cast(),
            (&mut m as *mut u32).cast(),
            (&mut m4 as *mut u32).cast(),
            (&mut n as *mut u32).cast(),
            (&mut k as *mut u32).cast(),
            (&mut splits as *mut u32).cast(),
        ];
        owner.be.launch_kernel(
            kernel.function,
            self.grid,
            kernel.threads,
            kernel.smem,
            &mut params,
            Some(stream),
        )?;
        if splits > 1 && !self.fused {
            let mut params = [
                (&mut out as *mut u64).cast(),
                (&mut partial as *mut u64).cast(),
                (&mut m as *mut u32).cast(),
                (&mut n as *mut u32).cast(),
                (&mut splits as *mut u32).cast(),
            ];
            let blocks = (u64::from(m) * u64::from(n)).div_ceil(256).min(264) as u32;
            owner
                .be
                .launch_kernel(owner.reduce, blocks, 256, 0, &mut params, Some(stream))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smem_matches_the_kernel_and_fits_hopper() {
        // fp8b_smem_bytes(8, 4, 2) and (32, 6, 2) of op_gemm_fp8_block_sm90.cuh.
        assert_eq!(
            smem_bytes(8, 4, 2),
            4 * (16384 + 1024) + 128 + 64 + 16 + 1024
        );
        assert_eq!(
            smem_bytes(32, 6, 1),
            6 * (8192 + 4096) + 768 + 96 + 16 + 1024
        );
        for &[m, _, _, cfg, _] in &plow_asset::segment_roles::NATIVE_DECODE_FP8_BLOCK_SHAPES {
            let (stages, variant) = fp8_block_cfg(cfg).unwrap();
            assert!(smem_bytes(tile_rows(m), stages, consumers(variant)) <= 227 * 1024);
        }
        assert_eq!(
            [tile_rows(1), tile_rows(9), tile_rows(17), tile_rows(32)],
            [8, 16, 32, 32]
        );
    }
}
