//! Native gfx942 block-scale FP8 projection through AITER's pre-shuffled assembly.
//!
//! The BF16 twin of this module is [`super::amd_gemm_lt`], and the shape of the two is deliberately
//! the same: a hash-pinned object, geometry validated at load, refused by name when anything does
//! not match. What differs is the numerics. GLM-5.3's checkpoint declares
//! `activation_scheme: dynamic` over a `[128,128]` `weight_scale_inv` grid, so W8A8 with a
//! per-token per-128-group activation scale IS this model's reference serving arithmetic — the
//! BF16 route plow ships today is *more* precise than the reference, not less.
//!
//! # Why assembly and not hipBLASLt
//!
//! hipBLASLt on gfx942 carries per-TENSOR FP8 scaling only; a `[128,128]` grid needs either CK's
//! `a8w8_blockscale_*` instances or these assembly objects. AITER ships both for gfx942, but its
//! own gfx942 tuning table (`a8w8_blockscale_tuned_gemm_glm5_1.csv`) selects CK at every M=8192
//! shape and the only `bpreshuffle` tuned tables in the tree are gfx950 — so the throughput of
//! these objects at our shapes is a measurement, not a quotation. See
//! `blk_matches_fp64_and_reports_throughput`.
//!
//! # ABI
//!
//! Transcribed from `/workspace/aiter/csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`
//! (`KernelArgs` + the launch at the bottom) and checked against the object's own
//! `NT_AMDGPU_METADATA`: 256-byte kernarg, 256 threads, 64 KiB static LDS, no private segment.
//! Every argument occupies a 16-byte slot. `ldc` is in BYTES; `lda`/`ldb` are in elements.
//! `A` is `[M,K]` e4m3-fnuz row-major, `B` is `[N,K]` e4m3-fnuz pre-shuffled by
//! [`super::amd::shuffle_moe_weight_16x32`] (AITER's `shuffle_weight(layout=(16,16))`),
//! `a_scale` is `[K/128][M]` f32 (group-major — the transposed layout AITER's own test passes),
//! `b_scale` is `[N/128][K/128]` f32 (the checkpoint's grid verbatim), and `bias` must be a real
//! `[N]` f32 pointer even when it is zero.
//!
//! Only the GPU harness below calls this yet; the emit half (FP8 weight declaration, the B
//! pre-shuffle at bind, and the activation-quant packet) is not built. Measured on one MI300X at
//! M=8192: 706-783 TF/s at the four pinned shapes, 84-94% of AITER's CK table; W8A8 rel-L2 vs
//! FP64 3.6-3.8e-2.
#![allow(dead_code)]

use std::path::Path;

use super::device_api::EngineDevice;
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::Module;
use crate::{Result, RuntimeError};

/// The six pre-shuffled tiles AITER ships for gfx942. All are `tile_n = 128`; the entry is
/// `(file, sha256, tile_m)`. `tile_n` is a constant of the family, not a per-object field:
/// `validate_inputs` in AITER's launcher refuses `N % 128 != 0` for every one of them.
const OBJECTS: [(&str, &str, u32); 6] = [
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_32x128.co",
        "3d1acfd1e5bf6f16816334d8aede37f93af0cbb3f0d6030dddb1deacf7e38d8f",
        32,
    ),
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_48x128.co",
        "f9a85f3cca7df1d2fea8b73f717a71674595b650bb4d9bafbb45b1298f62d841",
        48,
    ),
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_64x128.co",
        "b3b0814cdfc6be1cc838ba7e9065aa763dc2dcd72eb0c26afda31efab2884736",
        64,
    ),
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_80x128.co",
        "9f79c4151eab0d216595010507d671945ef8c1abc5e8271f026d21b155a1c5f1",
        80,
    ),
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_96x128.co",
        "6cda76fcdafd257f73d9cb9cbcf22b4730077b8d41c5bc9c6b28942799c9e5af",
        96,
    ),
    (
        "fp8gemm_bf16_blockscale_BpreShuffle_128x128.co",
        "de905e406509c62f78d0760a36e762f7de89ded5d39fd3f98062eb58a8078b99",
        128,
    ),
];

const TILE_N: u32 = 128;
/// The checkpoint's `weight_block_size`, and the group length of the activation quant.
const BLOCK_K: u32 = 128;
const WORKGROUP: u32 = 256;
const LDS_BYTES: u32 = 65536;
const KERNARG_BYTES: u32 = 256;
/// File offset of `KERNARG_SIZE` in the kernel descriptor (`.kd` at 0x12c0, field at +8).
const KD_KERNARG_SIZE: usize = 0x12c8;

/// Kernel-name prefix; the suffix is `{tile_m}x128E`, Itanium-mangled as `aiter::<name>`.
fn symbol(tile_m: u32) -> String {
    let stem = format!("fp8gemm_bf16_blockscale_BpreShuffle_{tile_m}x128");
    format!("_ZN5aiter{}{stem}E", stem.len())
}

/// AITER's own heuristic, restricted to `splitK = 1`.
///
/// `get_heuristic_fp8_kernel` minimises the number of CU rounds
/// `ceil(tiles / num_cu)`, breaking ties on spare CUs and then on the tile's compute-to-memory
/// ratio `tile_m*tile_n/(tile_m+tile_n)` — which is monotone in `tile_m` at fixed `tile_n`, so the
/// tie-break always prefers the larger tile. `splitK > 1` multiplies the tile count and therefore
/// the round count, so it can only win when one round is not yet full; at the prefill row counts
/// this route accepts (`M >= 2048`, `N >= 128`) it never is. Keeping `splitK = 1` is not just a
/// simplification: `splitK > 1` accumulates atomically into `C` and would oblige this route to zero
/// the output first, which a native segment has no packet to do.
fn tile_choice(m: u32, n: u32, n_cu: u32) -> Option<usize> {
    if m == 0 || n == 0 || !n.is_multiple_of(TILE_N) || n_cu == 0 {
        return None;
    }
    let tiles_n = n / TILE_N;
    let mut best: Option<(u32, u32, usize)> = None;
    for (index, &(_, _, tile_m)) in OBJECTS.iter().enumerate() {
        let tiles = m.div_ceil(tile_m).checked_mul(tiles_n)?;
        let round = tiles.div_ceil(n_cu);
        let spare = round * n_cu - tiles;
        // Ordering is (fewest rounds, then most spare CUs, then largest tile_m).
        if best.is_none_or(|(r, s, _)| round < r || (round == r && spare >= s)) {
            best = Some((round, spare, index));
        }
    }
    best.map(|(_, _, index)| index)
}

/// A qualified projection: the pinned tile plus the geometry the packet fixed.
#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    n: u32,
    k: u32,
    rows: u32,
    capacity: u32,
    index: usize,
}

impl Route {
    /// Qualify one projection. Returns `None` rather than an error so callers can fall back to the
    /// BF16 arm; everything that is structurally wrong (rather than merely unpinned) is an error.
    pub fn qualify(capacity: u32, n: u32, k: u32, n_cu: u32) -> Result<Option<Self>> {
        // The shapes this route is pinned to, per rank, TP8. Every one is a whole checkpoint
        // block-FP8 tensor or a block-row-aligned slice of one, so the `[N/128,K/128]` grid is the
        // checkpoint's own and no offline requantization is involved. `q_absorb` (4096x2048) is
        // deliberately ABSENT even though its geometry fits: the prep builds it as
        // `einsum(kv_b_nope, q_b_nope)`, a product of two dequantized matrices with no grid of its
        // own. The indexer's `wq_b` has the same 4096x2048 shape and IS a checkpoint tensor, so the
        // shape alone cannot decide — the emitter names the eligible instructions, and this list is
        // the second gate. The shared expert is left to the fused-MoE fold (as expert 257), and
        // indexer wk (128x6144) is out of scope until measured.
        const PINNED: [(u32, u32); 4] = [
            (2048, 6144), // q_a_proj
            (512, 6144),  // kv_a_latent
            (6144, 2048), // o_proj
            (4096, 2048), // indexer wq_b
        ];
        if capacity < 2048 || capacity > 8192 {
            return Ok(None);
        }
        if !PINNED.contains(&(n, k)) {
            return Ok(None);
        }
        // Geometry the object itself requires. These are errors, not fallbacks: a pinned shape that
        // fails them means the table above and the kernel family have diverged.
        if !n.is_multiple_of(TILE_N) || !k.is_multiple_of(BLOCK_K) {
            return Err(RuntimeError::Device(format!(
                "block-scale FP8 projection {n}x{k} needs N%{TILE_N}=0 and K%{BLOCK_K}=0"
            )));
        }
        let index = tile_choice(capacity, n, n_cu).ok_or_else(|| {
            RuntimeError::Device(format!(
                "block-scale FP8 projection {capacity}x{n}x{k}: no tile covers it"
            ))
        })?;
        Ok(Some(Self {
            n,
            k,
            rows: capacity,
            capacity,
            index,
        }))
    }

    /// Narrow to the live rows of this chunk. The tile is re-chosen, because the row count is what
    /// picks it and a 464-row tail wants a different tile from an 8192-row chunk.
    pub fn rebase(&mut self, rows: u32, n_cu: u32) -> Result<()> {
        if rows == 0 || rows > self.capacity {
            return Err(RuntimeError::Device(format!(
                "block-scale FP8 chunk of {rows} rows exceeds the {} it was emitted for",
                self.capacity
            )));
        }
        self.index = tile_choice(rows, self.n, n_cu)
            .ok_or_else(|| RuntimeError::Device("block-scale FP8 rebase found no tile".into()))?;
        self.rows = rows;
        Ok(())
    }

    fn grid(&self) -> [u32; 3] {
        [
            self.n / TILE_N,
            self.rows.div_ceil(OBJECTS[self.index].2),
            1,
        ]
    }
}

/// The kernel's 256-byte argument block. Each field owns a 16-byte slot; the padding is the
/// object's own (`.name: pad` in its metadata), not alignment slack, so it is spelled out.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Args {
    c: u64,
    _p0: u64,
    a: u64,
    _p1: u64,
    b: u64,
    _p2: u64,
    a_scale: u64,
    _p3: u64,
    b_scale: u64,
    _p4: u64,
    bias: u64,
    _p5: u64,
    m: u32,
    _p6: [u32; 3],
    n: u32,
    _p7: [u32; 3],
    k: u32,
    _p8: [u32; 3],
    lda: u32,
    _p9: [u32; 3],
    ldb: u32,
    _p10: [u32; 3],
    /// BYTES per output row, not elements — AITER passes `stride(0) * sizeof(uint16_t)`.
    ldc: u32,
    _p11: [u32; 3],
    ks: u32,
    _p12: [u32; 3],
    scale_m: u32,
    _p13: [u32; 3],
    scale_n: u32,
    _p14: [u32; 3],
    scale_k: u32,
    _p15: [u32; 3],
}
const _: () = assert!(std::mem::size_of::<Args>() == KERNARG_BYTES as usize);

/// `[out, a, b, a_scale, b_scale, bias]` device addresses, in that order.
pub(super) fn arguments(route: Route, tensors: [u64; 6]) -> Args {
    let [c, a, b, a_scale, b_scale, bias] = tensors;
    Args {
        c,
        a,
        b,
        a_scale,
        b_scale,
        bias,
        m: route.rows,
        n: route.n,
        k: route.k,
        lda: route.k,
        ldb: route.k,
        ldc: route.n * 2,
        ks: 1,
        // `block_shape_m` is 1: one activation scale per token per K-group.
        scale_m: route.rows,
        scale_n: route.n.div_ceil(BLOCK_K),
        scale_k: route.k.div_ceil(BLOCK_K),
        ..Default::default()
    }
}

pub(super) struct GemmBlk {
    kernels: Vec<HsaKernel>,
}

impl GemmBlk {
    pub fn load(be: &HsaBackend, dir: &Path, modules: &mut Vec<Module>) -> Result<Self> {
        let mut kernels = Vec::with_capacity(OBJECTS.len());
        for &(file, hash, tile_m) in &OBJECTS {
            let path = dir.join(file);
            let mut image = std::fs::read(&path)
                .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
            if plow_asset::decode_objects::image_sha256(&image) != hash {
                return Err(RuntimeError::Device(format!(
                    "{file} does not match the qualified block-scale FP8 ABI"
                )));
            }
            // Same omission as the AITER MoE objects: HIP reads the kernarg size from metadata,
            // ROCr from the descriptor, and these descriptors leave it zero. All six place
            // `.kd` at 0x12c0.
            let field = &mut image[KD_KERNARG_SIZE..KD_KERNARG_SIZE + 4];
            if field != [0; 4] {
                return Err(RuntimeError::Device(format!(
                    "{file} descriptor differs from the qualified ABI"
                )));
            }
            field.copy_from_slice(&KERNARG_BYTES.to_le_bytes());
            let module = EngineDevice::module_load(be, &image)?;
            let kernel = EngineDevice::get_function(be, &module, &symbol(tile_m))?;
            if kernel.kernarg_size() != KERNARG_BYTES
                || kernel.private_segment_size() != 0
                || HsaBackend::kernel_lds_bytes(&kernel) != LDS_BYTES
            {
                return Err(RuntimeError::Device(format!(
                    "{file} resource ABI mismatch: kernarg={} private={} lds={}",
                    kernel.kernarg_size(),
                    kernel.private_segment_size(),
                    HsaBackend::kernel_lds_bytes(&kernel),
                )));
            }
            modules.push(module);
            kernels.push(kernel);
        }
        tracing::info!(tiles = OBJECTS.len(), "block-scale FP8 projections loaded");
        Ok(Self { kernels })
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, tensors: [u64; 6]) -> Result<()> {
        let args = arguments(route, tensors);
        be.launch_3d(
            self.kernels[route.index],
            route.grid(),
            WORKGROUP as u16,
            bytemuck::bytes_of(&args),
        )
    }
}

/// gfx942's FP8 is e4m3 **fnuz**: bias 8, no infinities, `0x80` is the only NaN, max 240. The
/// checkpoint's bytes are OCP e4m3 (bias 7, max 448), so the same byte pattern denotes half the
/// value — which is why every consumer of a checkpoint grid on this chip doubles the scale
/// (`pack_moe` in `runtime/amd/moe_aiter_adapter.hip` does exactly that). These two functions are
/// the host side of that convention; the device side is `__builtin_amdgcn_cvt_pk_fp8_f32`.
#[cfg(test)]
fn fnuz_decode(b: u8) -> f64 {
    if b == 0 || b == 0x80 {
        return 0.0;
    }
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((b >> 3) & 0xf);
    let man = f64::from(b & 7);
    if exp == 0 {
        sign * (man / 8.0) * 2f64.powi(-7)
    } else {
        sign * (1.0 + man / 8.0) * 2f64.powi(exp - 8)
    }
}

#[cfg(test)]
fn fnuz_encode(x: f32) -> u8 {
    if !x.is_finite() || x == 0.0 {
        return 0;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= 240.0 {
        return sign | 0x7f;
    }
    let bits = a.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let man = bits & 0x7f_ffff;
    if exp + 8 >= 1 {
        let (mut m, rem) = (man >> 20, man & 0xf_ffff);
        let mut code_exp = (exp + 8) as u32;
        if rem > 0x8_0000 || (rem == 0x8_0000 && m & 1 == 1) {
            m += 1;
        }
        if m == 8 {
            m = 0;
            code_exp += 1;
        }
        if code_exp > 15 {
            return sign | 0x7f;
        }
        sign | ((code_exp as u8) << 3) | (m as u8)
    } else {
        // Subnormal: value = m/8 * 2^-7. m == 8 has rounded up into the smallest normal.
        let m = (f64::from(a) * 1024.0).round() as u32;
        if m == 0 {
            0
        } else {
            sign | (m.min(8) as u8)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blk_route_pins_the_checkpoint_shapes_and_refuses_the_absorbed_one() {
        // Every pinned shape qualifies at the prefill row counts the emitter can produce.
        for (n, k) in [(2048, 6144), (512, 6144), (6144, 2048), (4096, 2048)] {
            let route = Route::qualify(8192, n, k, 304).unwrap().unwrap();
            assert_eq!((route.n, route.k, route.rows), (n, k, 8192));
        }
        // Shapes the emitter must never hand over: k_rope and the indexer's weights_proj are
        // narrower than one tile, q_rope is a non-block-aligned slice, the shared expert (256x6144,
        // 6144x256) belongs to the MoE fold, indexer wk is out of scope, and the row buckets below
        // 2048 belong to the small-tile arm.
        for (n, k) in [
            (64, 6144),
            (32, 6144),
            (512, 2048),
            (256, 6144),
            (6144, 256),
            (128, 6144),
        ] {
            assert!(
                Route::qualify(8192, n, k, 304).unwrap().is_none(),
                "{n}x{k}"
            );
        }
        for rows in [1, 128, 512, 2047, 8193] {
            assert!(
                Route::qualify(rows, 2048, 6144, 304).unwrap().is_none(),
                "{rows}"
            );
        }
    }

    #[test]
    fn blk_rebase_tracks_live_rows_and_reselects_the_tile() {
        let mut route = Route::qualify(8192, 6144, 2048, 304).unwrap().unwrap();
        // 8192 rows x 48 N-tiles: the 128-row tile is 3072 groups = 11 rounds of 304 CUs, the
        // 96-row tile 4128 = 14. Largest tile wins outright.
        assert_eq!(OBJECTS[route.index].2, 128);
        assert_eq!(route.grid(), [48, 64, 1]);
        route.rebase(464, 304).unwrap();
        assert_eq!(route.rows, 464);
        // 464 rows: the 32-row tile is 15x48 = 720 groups (3 rounds), 128 gives 4x48 = 192 (1
        // round) — one round, so the tie-break on tile size takes the widest that still fits.
        assert_eq!(
            route.grid(),
            [48, 464u32.div_ceil(OBJECTS[route.index].2), 1]
        );
        assert!(route.rebase(0, 304).is_err());
        assert!(route.rebase(8193, 304).is_err());
    }

    #[test]
    fn blk_arguments_carry_the_aiter_abi() {
        let mut route = Route::qualify(8192, 6144, 2048, 304).unwrap().unwrap();
        route.rebase(4096, 304).unwrap();
        let args = arguments(route, [1, 2, 3, 4, 5, 6]);
        let raw = bytemuck::bytes_of(&args);
        // Pointer slots are 16 bytes apart; the scalars start at 96 and are 16 bytes apart too.
        for (slot, want) in [1u64, 2, 3, 4, 5, 6].into_iter().enumerate() {
            let at = slot * 16;
            assert_eq!(
                u64::from_le_bytes(raw[at..at + 8].try_into().unwrap()),
                want
            );
        }
        for (slot, want) in [4096u32, 6144, 2048, 2048, 2048, 6144 * 2, 1, 4096, 48, 16]
            .into_iter()
            .enumerate()
        {
            let at = 96 + slot * 16;
            assert_eq!(
                u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()),
                want,
                "scalar slot {slot}"
            );
        }
    }

    #[test]
    fn blk_fnuz_codec_round_trips_every_representable_byte() {
        for b in 0..=255u8 {
            if b == 0x80 {
                continue;
            }
            let v = fnuz_decode(b);
            assert_eq!(
                fnuz_encode(v as f32),
                if b == 0 { 0 } else { b },
                "byte {b:#04x}"
            );
        }
        assert_eq!(fnuz_encode(1000.0), 0x7f);
        assert_eq!(fnuz_decode(0x7f), 240.0);
        assert_eq!(fnuz_encode(0.0), 0);
    }

    /// One deterministic pseudo-Gaussian stream, so the two arms see identical inputs across runs.
    #[cfg(feature = "hsa")]
    fn gaussian(seed: &mut u64) -> f32 {
        let mut sum = 0i64;
        for _ in 0..4 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            sum += ((*seed >> 33) & 0xffff) as i64 - 0x8000;
        }
        sum as f32 / 65536.0
    }

    #[cfg(feature = "hsa")]
    fn bf16(x: f32) -> u16 {
        (x.to_bits() >> 16) as u16
    }

    #[cfg(feature = "hsa")]
    fn unbf16(x: u16) -> f64 {
        f64::from(f32::from_bits(u32::from(x) << 16))
    }

    /// Known-answer + numerics screen for the four pinned prefill shapes, and the throughput
    /// measurement that decides whether this route is worth an emit.
    ///
    /// Three numbers per shape:
    ///   * **rel-L2 vs FP64 over the quantized operands** — the kernel's own error. Anything above
    ///     the bf16 output rounding floor (~2e-3) means the ABI or a layout is wrong.
    ///   * **rel-L2 vs FP64 over the BF16 operands** — what serving would actually change, i.e. the
    ///     W8A8 quantization floor. The MoE route's documented floor at the same block size is
    ///     4.0-4.4e-2.
    ///   * **µs and TF/s** at M=8192, against AITER's own gfx942 CK table (838-873 TF/s).
    #[test]
    #[cfg(feature = "hsa")]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR (fp8gemm_blockscale)"]
    fn blk_matches_fp64_and_reports_throughput() {
        let dir = std::env::var("PLOW_TEST_AITER_DIR").expect("PLOW_TEST_AITER_DIR");
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let blk = GemmBlk::load(&be, Path::new(&dir), &mut modules).unwrap();
        let n_cu = 304;
        let m = 8192usize;
        let probe = [0usize, 1, 1023, 4096, 8191];

        for (label, n, k) in [
            ("q_a_proj", 2048usize, 6144usize),
            ("kv_a_latent", 512, 6144),
            ("o_proj", 6144, 2048),
            ("indexer_wq_b", 4096, 2048),
        ] {
            let mut seed = 0x5eed_0000_0000_0001 ^ (n as u64) << 20 ^ k as u64;
            // Activation: bf16, per-row scaled so rows differ in magnitude the way a normed
            // hidden state does not, which is the harder case for a per-row-per-group scale.
            let a_bf: Vec<u16> = (0..m * k)
                .map(|i| bf16(gaussian(&mut seed) * (0.5 + (i / k % 7) as f32 * 0.3)))
                .collect();
            let w_bf: Vec<u16> = (0..n * k)
                .map(|_| bf16(gaussian(&mut seed) * 0.05))
                .collect();

            // Activation quant: 128-element groups along K, scale = amax/240, scales group-major.
            let gk = k / BLOCK_K as usize;
            let mut a_q = vec![0u8; m * k];
            let mut a_s = vec![0f32; gk * m];
            for row in 0..m {
                for g in 0..gk {
                    let span = &a_bf[row * k + g * 128..row * k + g * 128 + 128];
                    let amax = span
                        .iter()
                        .fold(1e-10f32, |acc, &v| acc.max(unbf16(v).abs() as f32));
                    let s = amax / 240.0;
                    a_s[g * m + row] = s;
                    for (j, &v) in span.iter().enumerate() {
                        a_q[row * k + g * 128 + j] = fnuz_encode(unbf16(v) as f32 / s);
                    }
                }
            }
            // Weight quant: the checkpoint's [128,128] grid, expressed directly in fnuz.
            let (gn, mut w_q, mut w_s) = (n / 128, vec![0u8; n * k], vec![0f32; (n / 128) * gk]);
            for bn in 0..gn {
                for bg in 0..gk {
                    let amax = (0..128).flat_map(|r| (0..128).map(move |c| (r, c))).fold(
                        1e-10f32,
                        |acc, (r, c)| {
                            acc.max(unbf16(w_bf[(bn * 128 + r) * k + bg * 128 + c]).abs() as f32)
                        },
                    );
                    let s = amax / 240.0;
                    w_s[bn * gk + bg] = s;
                    for r in 0..128 {
                        for c in 0..128 {
                            let at = (bn * 128 + r) * k + bg * 128 + c;
                            w_q[at] = fnuz_encode(unbf16(w_bf[at]) as f32 / s);
                        }
                    }
                }
            }
            let w_shuffled = super::super::amd::shuffle_moe_weight_16x32(&w_q, n, k).unwrap();

            let upload = |bytes: &[u8]| {
                let mem = EngineDevice::alloc(&be, bytes.len() as u64).unwrap();
                EngineDevice::upload(&be, &mem, 0, bytes).unwrap();
                mem
            };
            let d_a = upload(&a_q);
            let d_b = upload(&w_shuffled);
            let d_as = upload(bytemuck::cast_slice(&a_s));
            let d_bs = upload(bytemuck::cast_slice(&w_s));
            let d_bias = upload(bytemuck::cast_slice(&vec![0f32; n]));
            let d_c = EngineDevice::alloc(&be, (m * n * 2) as u64).unwrap();

            let route = Route::qualify(m as u32, n as u32, k as u32, n_cu)
                .unwrap()
                .unwrap_or_else(|| panic!("{label}: {n}x{k} is not a pinned shape"));
            let ptrs = [
                d_c.base,
                d_a.base,
                d_b.base,
                d_as.base,
                d_bs.base,
                d_bias.base,
            ];
            blk.enqueue(&be, route, ptrs).unwrap();
            be.synchronize().unwrap();

            let mut c = vec![0u8; m * n * 2];
            EngineDevice::download(&be, &d_c, 0, &mut c).unwrap();
            let c: &[u16] = bytemuck::cast_slice(&c);

            let (mut e_q, mut e_b, mut norm) = (0f64, 0f64, 0f64);
            for &row in &probe {
                for col in 0..n {
                    let (mut acc_q, mut acc_b) = (0f64, 0f64);
                    for g in 0..gk {
                        let (mut dot, wsc) = (0f64, f64::from(w_s[(col / 128) * gk + g]));
                        for j in 0..128 {
                            let at_a = row * k + g * 128 + j;
                            let at_w = col * k + g * 128 + j;
                            dot += fnuz_decode(a_q[at_a]) * fnuz_decode(w_q[at_w]);
                            acc_b += unbf16(a_bf[at_a]) * unbf16(w_bf[at_w]);
                        }
                        acc_q += dot * f64::from(a_s[g * m + row]) * wsc;
                    }
                    let got = unbf16(c[row * n + col]);
                    e_q += (got - acc_q).powi(2);
                    e_b += (got - acc_b).powi(2);
                    norm += acc_b * acc_b;
                }
            }
            let (rel_q, rel_b) = ((e_q / norm).sqrt(), (e_b / norm).sqrt());

            for _ in 0..3 {
                blk.enqueue(&be, route, ptrs).unwrap();
            }
            be.synchronize().unwrap();
            let iters = 20;
            let started = std::time::Instant::now();
            for _ in 0..iters {
                blk.enqueue(&be, route, ptrs).unwrap();
            }
            be.synchronize().unwrap();
            let us = started.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
            let tflops = 2.0 * (m * n * k) as f64 / (us * 1e6);
            println!(
                "{label:14} {m}x{n}x{k} tile_m={:3} {us:8.1} us {tflops:7.1} TF/s \
                 rel-L2 vs FP64(fp8) {rel_q:.3e} vs FP64(bf16) {rel_b:.3e}",
                OBJECTS[route.index].2
            );
            assert!(
                rel_q < 5e-3,
                "{label}: kernel disagrees with its own operands: {rel_q:.3e}"
            );
            assert!(
                rel_b < 8e-2,
                "{label}: W8A8 floor {rel_b:.3e} is worse than the MoE route's"
            );
        }
    }

    #[test]
    fn blk_symbols_match_the_object_names() {
        assert_eq!(
            symbol(128),
            "_ZN5aiter43fp8gemm_bf16_blockscale_BpreShuffle_128x128E"
        );
        assert_eq!(
            symbol(32),
            "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E"
        );
    }
}
