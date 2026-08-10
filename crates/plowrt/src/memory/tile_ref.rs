//! CPU reference tiler — the **oracle** for the weight-loader layout contract.
//!
//! Correctness over speed: this is what every GPU tiling kernel is validated
//! against, bit-exactly. Tiling is a pure byte permutation (plus zero padding at
//! the far edge), so the bar is bit-equality, not a tolerance.
//!
//! # The layout contract
//!
//! Source: a checkpoint tensor `[N, K]` row-major (HuggingFace `Linear.weight`
//! convention: `N = out_features`, `K = in_features`), element size
//! `elem_bytes`. Source byte offset of element `(n, k)`:
//!
//! ```text
//! src_off(n, k) = (n * K + k) * elem_bytes
//! ```
//!
//! Destination: the arena region for that tensor, in the compiled `(BN, BK)`
//! tiling. Per `plow_asset::WeightTiling` (`block_iteration = "n_major_k_inner"`,
//! `within_block_layout = "n_outer_k_inner"`, `padding_policy = "zero_extend"`):
//!
//! ```text
//! grid_cols = ceil(K / BK)
//! tr = n / BN            n_local = n % BN
//! tc = k / BK            k_local = k % BK
//! dst_off(n, k) = (tr * grid_cols + tc) * BN * BK * elem_bytes
//!               + (n_local * BK + k_local) * elem_bytes
//! ```
//!
//! Total size is `ceil(N/BN) * ceil(K/BK) * BN * BK * elem_bytes` — the **padded**
//! size. When `BN ∤ N` or `BK ∤ K` this is strictly larger than `N * K *
//! elem_bytes`; the surplus positions are zero-filled.
//!
//! # Caveat — the consumers do not read this layout
//!
//! `runtime/nvidia/gemma_sm120.cu:179` and `runtime/amd/op_gemm.h:5` both index
//! the weight as **plain row-major `[N, K]` with row stride `K`**. For them
//! `(BN, BK)` is a shared-memory staging tile shape, not a global byte layout.
//! [`Tiling::is_identity`] reports when the tiled layout coincides with
//! row-major (`BN >= N && BK >= K`, i.e. a single tile); for the real Qwen3-4B
//! `(128, 64)` it does not. See the design notes.

/// The `(BN, BK)` byte-layout parameters for one tensor, from
/// `plow_asset::WeightTiling`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tiling {
    pub bn: usize,
    pub bk: usize,
    pub elem_bytes: usize,
}

/// Why a tiling request is not well-formed.
#[derive(Debug, PartialEq, Eq)]
pub enum TileError {
    /// A tiling parameter was zero.
    BadParam(&'static str),
    /// Source buffer is not exactly `N * K * elem_bytes`. Names the tensor so a
    /// short/misnamed checkpoint tensor hard-fails loudly rather than being
    /// silently zero-padded.
    ShortSource {
        tensor: String,
        expected: usize,
        got: usize,
    },
    /// Destination buffer is smaller than the padded tiled size.
    ShortDest {
        tensor: String,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for TileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TileError::BadParam(p) => write!(f, "tiling parameter '{p}' must be non-zero"),
            TileError::ShortSource {
                tensor,
                expected,
                got,
            } => write!(
                f,
                "tensor '{tensor}': checkpoint gave {got} bytes, layout needs exactly {expected}"
            ),
            TileError::ShortDest {
                tensor,
                expected,
                got,
            } => write!(
                f,
                "tensor '{tensor}': arena region is {got} bytes, tiled layout needs {expected}"
            ),
        }
    }
}

impl std::error::Error for TileError {}

impl Tiling {
    pub fn new(bn: i64, bk: i64, elem_bytes: u32) -> Result<Self, TileError> {
        if bn <= 0 {
            return Err(TileError::BadParam("bn"));
        }
        if bk <= 0 {
            return Err(TileError::BadParam("bk"));
        }
        if elem_bytes == 0 {
            return Err(TileError::BadParam("elem_bytes"));
        }
        Ok(Tiling {
            bn: bn as usize,
            bk: bk as usize,
            elem_bytes: elem_bytes as usize,
        })
    }

    /// Number of tile columns for a `K`-wide tensor: `ceil(K / BK)`.
    pub fn grid_cols(&self, k: usize) -> usize {
        k.div_ceil(self.bk)
    }

    /// Number of tile rows for an `N`-tall tensor: `ceil(N / BN)`.
    pub fn grid_rows(&self, n: usize) -> usize {
        n.div_ceil(self.bn)
    }

    /// Padded byte size of the tiled form — what the arena region must hold.
    /// Equals `n * k * elem_bytes` iff `BN | N` and `BK | K`.
    pub fn tiled_bytes(&self, n: usize, k: usize) -> usize {
        self.grid_rows(n) * self.grid_cols(k) * self.bn * self.bk * self.elem_bytes
    }

    /// Unpadded logical byte size — what the checkpoint holds, and what the
    /// compiler currently reserves in the arena (`schedule::memory` sizes a
    /// `Persistent` buffer from `op_in_bytes` = `N * K * elem`).
    pub fn logical_bytes(&self, n: usize, k: usize) -> usize {
        n * k * self.elem_bytes
    }

    /// `true` when the tiled layout is byte-identical to row-major: a single
    /// tile that exactly covers the tensor. Any other case is a real
    /// permutation, and a loader that DMAs raw bytes produces wrong results.
    pub fn is_identity(&self, n: usize, k: usize) -> bool {
        self.grid_rows(n) == 1 && self.grid_cols(k) == 1 && self.bk == k
    }

    /// Byte offset of element `(n_idx, k_idx)` in the tiled arena region.
    #[inline]
    pub fn dst_off(&self, n_idx: usize, k_idx: usize, k: usize) -> usize {
        let grid_cols = self.grid_cols(k);
        let (tr, n_local) = (n_idx / self.bn, n_idx % self.bn);
        let (tc, k_local) = (k_idx / self.bk, k_idx % self.bk);
        let tile_ordinal = tr * grid_cols + tc;
        (tile_ordinal * self.bn * self.bk + n_local * self.bk + k_local) * self.elem_bytes
    }

    /// Byte offset of element `(n_idx, k_idx)` in the row-major source.
    #[inline]
    pub fn src_off(&self, n_idx: usize, k_idx: usize, k: usize) -> usize {
        (n_idx * k + k_idx) * self.elem_bytes
    }
}

/// Permute a row-major `[n, k]` tensor into its tiled arena form.
///
/// `dst` must be at least [`Tiling::tiled_bytes`]; the padded region is zeroed.
/// `tensor` names the weight for error reporting — a short source hard-fails
/// naming it, because a silently zero-padded weight still generates fluent text.
pub fn tile_rowmajor(
    tensor: &str,
    src: &[u8],
    n: usize,
    k: usize,
    t: &Tiling,
    dst: &mut [u8],
) -> Result<(), TileError> {
    let need_src = t.logical_bytes(n, k);
    if src.len() != need_src {
        return Err(TileError::ShortSource {
            tensor: tensor.to_string(),
            expected: need_src,
            got: src.len(),
        });
    }
    let need_dst = t.tiled_bytes(n, k);
    if dst.len() < need_dst {
        return Err(TileError::ShortDest {
            tensor: tensor.to_string(),
            expected: need_dst,
            got: dst.len(),
        });
    }
    // zero_extend: every byte of the padded region starts at zero, so positions
    // past N or K stay zero without a second pass.
    dst[..need_dst].fill(0);

    let eb = t.elem_bytes;
    // Copy the longest contiguous run available: within one tile, a fixed n_idx
    // spans `min(bk, k - k0)` consecutive elements in BOTH layouts.
    for n_idx in 0..n {
        for tc in 0..t.grid_cols(k) {
            let k0 = tc * t.bk;
            let run = t.bk.min(k - k0);
            let s = t.src_off(n_idx, k0, k);
            let d = t.dst_off(n_idx, k0, k);
            dst[d..d + run * eb].copy_from_slice(&src[s..s + run * eb]);
        }
    }
    Ok(())
}

/// Inverse permutation: tiled arena form → row-major `[n, k]`. Exists so tests
/// can prove `untile(tile(x)) == x` (a permutation must round-trip) and so a
/// GPU-tiled buffer read back from the device can be compared against the
/// checkpoint in its original layout.
pub fn untile_to_rowmajor(
    tensor: &str,
    src: &[u8],
    n: usize,
    k: usize,
    t: &Tiling,
    dst: &mut [u8],
) -> Result<(), TileError> {
    let need_src = t.tiled_bytes(n, k);
    if src.len() < need_src {
        return Err(TileError::ShortSource {
            tensor: tensor.to_string(),
            expected: need_src,
            got: src.len(),
        });
    }
    let need_dst = t.logical_bytes(n, k);
    if dst.len() < need_dst {
        return Err(TileError::ShortDest {
            tensor: tensor.to_string(),
            expected: need_dst,
            got: dst.len(),
        });
    }
    let eb = t.elem_bytes;
    for n_idx in 0..n {
        for tc in 0..t.grid_cols(k) {
            let k0 = tc * t.bk;
            let run = t.bk.min(k - k0);
            let s = t.dst_off(n_idx, k0, k);
            let d = t.src_off(n_idx, k0, k);
            dst[d..d + run * eb].copy_from_slice(&src[s..s + run * eb]);
        }
    }
    Ok(())
}

/// Tile only output rows `[n_lo, n_hi)` of a tensor — the tensor-parallel N-shard
/// path. Produces exactly the bytes device `d` needs, reading only that row range
/// from the checkpoint, so a sharded load never materializes a full host copy per
/// device.
///
/// The permutation is **N-shard-local**: rows `[n_lo, n_hi)` are re-indexed to
/// `[0, n_hi - n_lo)` before tiling, which is only equal to a slice of the full
/// tiled buffer when `n_lo` is a multiple of `BN`. Callers must shard on a `BN`
/// boundary; this is checked.
pub fn tile_rowmajor_nshard(
    tensor: &str,
    src_rows: &[u8],
    n_lo: usize,
    n_hi: usize,
    k: usize,
    t: &Tiling,
    dst: &mut [u8],
) -> Result<(), TileError> {
    if n_lo % t.bn != 0 {
        return Err(TileError::BadParam("n_lo must be a multiple of bn"));
    }
    tile_rowmajor(tensor, src_rows, n_hi - n_lo, k, t, dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_pattern(n: usize, k: usize) -> Vec<u8> {
        // Distinct 16-bit value per (n, k) so any misplaced element is visible.
        let mut v = Vec::with_capacity(n * k * 2);
        for i in 0..n {
            for j in 0..k {
                let e = ((i * k + j) as u32 & 0xffff) as u16;
                v.extend_from_slice(&e.to_le_bytes());
            }
        }
        v
    }

    /// The layout formula, spelled out by hand on a tiny case. If the docstring
    /// in `plow_asset::WeightTiling` and this test ever disagree, one is wrong.
    #[test]
    fn index_math_matches_manifest_formula_by_hand() {
        // N=4, K=4, BN=2, BK=2 -> 2x2 grid of 2x2 tiles, 4 tiles.
        let t = Tiling::new(2, 2, 2).unwrap();
        let (n, k) = (4usize, 4usize);
        assert_eq!(t.grid_rows(n), 2);
        assert_eq!(t.grid_cols(k), 2);
        assert_eq!(t.tiled_bytes(n, k), 4 * 2 * 2 * 2);

        // tile(0,0) holds (n,k) in {0,1}x{0,1} at element ordinals 0..4.
        assert_eq!(t.dst_off(0, 0, k), 0);
        assert_eq!(t.dst_off(0, 1, k), 2);
        assert_eq!(t.dst_off(1, 0, k), 4);
        assert_eq!(t.dst_off(1, 1, k), 6);
        // tile(0,1) = ordinal 1 -> base 1*2*2*2 = 8 bytes. Holds k in {2,3}.
        assert_eq!(t.dst_off(0, 2, k), 8);
        assert_eq!(t.dst_off(1, 3, k), 14);
        // tile(1,0) = ordinal 2 -> base 16. Holds n in {2,3}, k in {0,1}.
        assert_eq!(t.dst_off(2, 0, k), 16);
        // tile(1,1) = ordinal 3 -> base 24.
        assert_eq!(t.dst_off(3, 3, k), 30);

        // And the produced bytes agree element-for-element with the formula.
        let src = bf16_pattern(n, k);
        let mut dst = vec![0u8; t.tiled_bytes(n, k)];
        tile_rowmajor("t", &src, n, k, &t, &mut dst).unwrap();
        for i in 0..n {
            for j in 0..k {
                let s = t.src_off(i, j, k);
                let d = t.dst_off(i, j, k);
                assert_eq!(&dst[d..d + 2], &src[s..s + 2], "element ({i},{j})");
            }
        }
    }

    /// A permutation must round-trip exactly.
    #[test]
    fn tile_then_untile_is_bit_exact_identity() {
        for (n, k, bn, bk) in [
            (4096usize, 2560usize, 128i64, 64i64), // Qwen3-4B q_proj, real (bn,bk)
            (2560, 9728, 128, 64),                 // down_proj
            (1024, 2560, 128, 64),                 // k_proj
            (150, 100, 128, 64),                   // ragged both axes
        ] {
            let t = Tiling::new(bn, bk, 2).unwrap();
            let src = bf16_pattern(n, k);
            let mut tiled = vec![0u8; t.tiled_bytes(n, k)];
            tile_rowmajor("w", &src, n, k, &t, &mut tiled).unwrap();
            let mut back = vec![0u8; t.logical_bytes(n, k)];
            untile_to_rowmajor("w", &tiled, n, k, &t, &mut back).unwrap();
            assert_eq!(back, src, "round-trip {n}x{k} bn={bn} bk={bk}");
        }
    }

    /// NEGATIVE CONTROL: the round-trip test above must be able to fail.
    /// Perturbing a single byte of the tiled buffer must be detected.
    #[test]
    fn negative_control_single_byte_perturbation_is_detected() {
        let (n, k) = (1024usize, 2560usize);
        let t = Tiling::new(128, 64, 2).unwrap();
        let src = bf16_pattern(n, k);
        let mut tiled = vec![0u8; t.tiled_bytes(n, k)];
        tile_rowmajor("w", &src, n, k, &t, &mut tiled).unwrap();

        let victim = t.dst_off(700, 1337, k);
        tiled[victim] ^= 0x01;

        let mut back = vec![0u8; t.logical_bytes(n, k)];
        untile_to_rowmajor("w", &tiled, n, k, &t, &mut back).unwrap();
        assert_ne!(back, src, "a flipped bit must break the round-trip");
        // and it must surface at exactly the element that was perturbed
        assert_eq!(
            back[t.src_off(700, 1337, k)],
            src[t.src_off(700, 1337, k)] ^ 0x01
        );
    }

    /// NEGATIVE CONTROL: tiling is NOT the identity at the real (128, 64).
    /// If it were, a raw DMA would accidentally pass and the whole exercise
    /// would be moot. Prove the permutation actually moves bytes.
    #[test]
    fn negative_control_tiling_differs_from_rowmajor() {
        let (n, k) = (4096usize, 2560usize);
        let t = Tiling::new(128, 64, 2).unwrap();
        assert!(!t.is_identity(n, k));
        let src = bf16_pattern(n, k);
        let mut tiled = vec![0u8; t.tiled_bytes(n, k)];
        tile_rowmajor("q_proj.weight", &src, n, k, &t, &mut tiled).unwrap();
        assert_ne!(
            tiled, src,
            "tiled form must differ from row-major, else a raw DMA would pass by accident"
        );
        // The very first divergence: element (0, 64) is at byte 128 row-major
        // but lands in tile ordinal 1, at byte 128*64*2 = 16384.
        assert_eq!(t.src_off(0, 64, k), 128);
        assert_eq!(t.dst_off(0, 64, k), 16384);
    }

    /// Single-tile-wide layouts DO coincide with row-major. Documents the
    /// boundary of `is_identity` so nobody generalizes from a passing toy case.
    #[test]
    fn identity_only_when_one_tile_covers_k_and_n() {
        let t = Tiling::new(128, 64, 2).unwrap();
        assert!(t.is_identity(128, 64));
        assert!(t.is_identity(1, 64)); // N < BN, zero-padded but rows contiguous
        assert!(!t.is_identity(128, 128)); // two tile columns
        assert!(!t.is_identity(256, 64)); // two tile rows -> still contiguous...
    }

    /// zero_extend: ragged edges pad with zeros, and the padded size exceeds the
    /// logical size. This is the arena-overflow hazard — `MemEntry::reserved` is
    /// sized from `N * K * elem`, not from this.
    #[test]
    fn zero_extend_pads_and_grows_the_footprint() {
        let (n, k) = (130usize, 100usize);
        let t = Tiling::new(128, 64, 2).unwrap();
        assert_eq!(t.logical_bytes(n, k), 130 * 100 * 2); // 26_000
        assert_eq!(t.tiled_bytes(n, k), 2 * 2 * 128 * 64 * 2); // 65_536
        assert!(t.tiled_bytes(n, k) > t.logical_bytes(n, k));

        let src = bf16_pattern(n, k);
        let mut dst = vec![0xAAu8; t.tiled_bytes(n, k)];
        tile_rowmajor("w", &src, n, k, &t, &mut dst).unwrap();
        // Element (129, 99) is the last real one; (129, 100) is padding.
        assert_eq!(
            &dst[t.dst_off(129, 100, k)..t.dst_off(129, 100, k) + 2],
            &[0, 0]
        );
        // A whole padded row (n = 200 >= 130) is zero.
        assert_eq!(
            &dst[t.dst_off(200, 0, k)..t.dst_off(200, 0, k) + 2],
            &[0, 0]
        );
    }

    /// A short / misnamed tensor hard-fails and NAMES the tensor.
    #[test]
    fn short_source_hard_fails_naming_the_tensor() {
        let t = Tiling::new(128, 64, 2).unwrap();
        let (n, k) = (128usize, 64usize);
        let short = vec![0u8; t.logical_bytes(n, k) - 2];
        let mut dst = vec![0u8; t.tiled_bytes(n, k)];
        let err = tile_rowmajor(
            "model.layers.0.self_attn.q_proj.weight",
            &short,
            n,
            k,
            &t,
            &mut dst,
        )
        .unwrap_err();
        match &err {
            TileError::ShortSource {
                tensor,
                expected,
                got,
            } => {
                assert_eq!(tensor, "model.layers.0.self_attn.q_proj.weight");
                assert_eq!(*expected, 16384);
                assert_eq!(*got, 16382);
            }
            other => panic!("wrong error: {other:?}"),
        }
        assert!(err.to_string().contains("q_proj.weight"), "{err}");
    }

    /// A dest sized to the LOGICAL bytes (what `MemEntry::reserved` gives) is
    /// rejected when the tensor needs padding — this is the check that stops a
    /// silent overrun into the neighbouring weight.
    #[test]
    fn dest_sized_to_reserved_is_rejected_when_padding_is_needed() {
        let t = Tiling::new(128, 64, 2).unwrap();
        let (n, k) = (130usize, 100usize);
        let src = bf16_pattern(n, k);
        let mut dst = vec![0u8; t.logical_bytes(n, k)]; // == MemEntry::reserved
        let err = tile_rowmajor("gate_proj.weight", &src, n, k, &t, &mut dst).unwrap_err();
        assert!(matches!(err, TileError::ShortDest { .. }), "{err:?}");
    }

    /// TP: sharding on a BN boundary makes the shard's tiled bytes exactly the
    /// corresponding byte range of the full tiled buffer — so each device can be
    /// filled from its own checkpoint row range with no full host copy.
    #[test]
    fn n_shard_on_bn_boundary_equals_a_slice_of_the_full_tiling() {
        let (n, k) = (4096usize, 2560usize);
        let t = Tiling::new(128, 64, 2).unwrap();
        let src = bf16_pattern(n, k);
        let mut full = vec![0u8; t.tiled_bytes(n, k)];
        tile_rowmajor("w", &src, n, k, &t, &mut full).unwrap();

        let shards = 4usize;
        let rows = n / shards; // 1024, a multiple of BN=128
        for d in 0..shards {
            let (lo, hi) = (d * rows, (d + 1) * rows);
            let row_bytes = &src[lo * k * 2..hi * k * 2];
            let mut shard = vec![0u8; t.tiled_bytes(hi - lo, k)];
            tile_rowmajor_nshard("w", row_bytes, lo, hi, k, &t, &mut shard).unwrap();
            let base = t.dst_off(lo, 0, k);
            assert_eq!(
                shard,
                &full[base..base + shard.len()],
                "shard {d} rows [{lo},{hi}) must equal the full tiling's byte range"
            );
        }
    }

    /// Real checkpoint bytes: Qwen3-4B `layers.0.self_attn.q_proj.weight`,
    /// bf16 `[4096, 2560]`, at the compiler's real `(bn, bk) = (128, 64)`.
    /// Skipped (not failed) when the checkpoint is not staged.
    #[test]
    #[cfg(feature = "hub")]
    fn real_qwen3_q_proj_tiles_bit_exactly() {
        let path = std::path::Path::new("/root/models/Qwen3-4B/model-00001-of-00003.safetensors");
        if !path.exists() {
            eprintln!("SKIP: {} not staged", path.display());
            return;
        }
        let file = std::fs::File::open(path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let st = safetensors::SafeTensors::deserialize(&mmap).unwrap();
        let name = "model.layers.0.self_attn.q_proj.weight";
        let tv = st.tensor(name).unwrap();
        assert_eq!(tv.shape(), &[4096, 2560]);
        let src = tv.data();
        let (n, k) = (4096usize, 2560usize);

        let t = Tiling::new(128, 64, 2).unwrap();
        assert_eq!(t.logical_bytes(n, k), src.len());
        // Qwen3-4B divides evenly: no zero padding for this tensor.
        assert_eq!(t.tiled_bytes(n, k), t.logical_bytes(n, k));

        let mut tiled = vec![0u8; t.tiled_bytes(n, k)];
        tile_rowmajor(name, src, n, k, &t, &mut tiled).unwrap();

        // It is a real permutation of real weights, not a copy.
        assert_ne!(&tiled[..], src, "tiled must differ from on-disk row-major");

        // Every element lands where the manifest formula says.
        for &(i, j) in &[
            (0usize, 0usize),
            (0, 64),
            (1, 0),
            (127, 63),
            (128, 0),
            (4095, 2559),
        ] {
            let s = t.src_off(i, j, k);
            let d = t.dst_off(i, j, k);
            assert_eq!(&tiled[d..d + 2], &src[s..s + 2], "element ({i},{j})");
        }

        // Bit-exact round-trip on the real bytes.
        let mut back = vec![0u8; t.logical_bytes(n, k)];
        untile_to_rowmajor(name, &tiled, n, k, &t, &mut back).unwrap();
        assert_eq!(&back[..], src, "round-trip on real checkpoint bytes");

        // NEGATIVE CONTROL on real data: perturb one weight, detection required.
        let mut bad = tiled.clone();
        bad[t.dst_off(2000, 1000, k)] ^= 0x01;
        let mut back2 = vec![0u8; t.logical_bytes(n, k)];
        untile_to_rowmajor(name, &bad, n, k, &t, &mut back2).unwrap();
        assert_ne!(&back2[..], src, "perturbed real tensor must be detected");
    }

    /// NEGATIVE CONTROL for the TP claim: off a BN boundary it is rejected,
    /// because the re-indexing is not a slice of the full tiling.
    #[test]
    fn negative_control_n_shard_off_bn_boundary_is_rejected() {
        let t = Tiling::new(128, 64, 2).unwrap();
        let k = 2560usize;
        let src = bf16_pattern(64, k);
        let mut dst = vec![0u8; t.tiled_bytes(64, k)];
        let err = tile_rowmajor_nshard("w", &src, 64, 128, k, &t, &mut dst).unwrap_err();
        assert_eq!(err, TileError::BadParam("n_lo must be a multiple of bn"));
    }
}
