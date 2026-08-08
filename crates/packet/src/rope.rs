//! RoPE cos/sin table generation — shared by the compiler (`devgen`) and the
//! runtime (`plowrt`).
//!
//! # Why this lives here
//!
//! These tables used to be expanded by the compiler and shipped verbatim in the
//! device blob's init section. At the default `--max-ctx 131072` that is ~403 MB
//! of a Gemma-4 `model.pkt` — the tables dominate the file, the load-time PCIe
//! traffic, and (once resident) a chunk of HBM. They are a pure function of a
//! handful of config scalars, so the blob now carries a [`GenTensor`] recipe and
//! the runtime materialises the bytes at bind time.
//!
//! # The one rule: this stays host-side Rust
//!
//! Every value here is computed in **f64** and rounded to f32 exactly once, at
//! the store. Compiler and runtime therefore produce bit-identical tables as
//! long as both call *this* code. A GPU-side reimplementation would not: `__cosf`
//! is an SFU approximation, and the resulting drift is the worst failure mode in
//! this stack — the model still produces fluent text, just subtly wrong text.
//! If you are tempted to move this into a kernel, don't.

/// RoPE frequency scaling. Gemma/Qwen use plain `theta^(-2i/hd)`; Llama-3.1 rescales the low
/// frequencies (long wavelengths) by `factor` with a smooth transition band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeScale {
    None,
    Llama3 {
        factor: f64,
        low: f64,
        high: f64,
        orig: f64,
    },
}

/// Llama-3.1 inv_freq rescaling: low frequencies (long wavelengths) are divided by `factor`, high
/// frequencies pass through, with a smooth interpolation between. Mirrors HF
/// `_compute_llama3_parameters`. [`RopeScale::None`] returns inv unchanged (Gemma/Qwen).
fn scale_inv_freq(inv: f64, scale: RopeScale) -> f64 {
    match scale {
        RopeScale::None => inv,
        RopeScale::Llama3 {
            factor,
            low,
            high,
            orig,
        } => {
            let low_wl = orig / low; // long-wavelength threshold
            let high_wl = orig / high; // short-wavelength threshold
            let wl = 2.0 * std::f64::consts::PI / inv;
            if wl > low_wl {
                inv / factor
            } else if wl < high_wl {
                inv
            } else {
                let smooth = (orig / wl - low) / (high - low);
                (1.0 - smooth) * inv / factor + smooth * inv
            }
        }
    }
}

/// RoPE cos/sin tables, laid out as the kernel indexes them: `cos[pos * (hd/2) + j]`.
///
/// The partial rotary is the trap. `_compute_proportional_rope_parameters` gives
///   rope_angles = int(frac * hd // 2)                       (= 64 for frac=0.25, hd=512)
///   inv_freq[i] = theta^(-2i / hd)   for i < rope_angles
///   inv_freq[i] = 0                  for rope_angles <= i < hd/2     <- NoPE, 192 of them
/// Note the exponent divides by the FULL hd (512), not by the rotary width (128) — and the
/// table is hd/2 wide, not rope_angles wide, so `emb = cat(freqs, freqs)` is hd wide and
/// `rotate_half` pairs (i, i+hd/2) = (i, i+256). A zero frequency gives cos=1, sin=0, so
/// those dims pass through EXACTLY unchanged. Getting any of this wrong still produces
/// fluent text — which is why ONE function owns the table and nothing else recomputes it.
pub fn rope_tables(t: u32, hd: u32, theta: f64, frac: f64, scale: RopeScale) -> (Vec<u8>, Vec<u8>) {
    let h2 = (hd / 2) as usize;
    let rope_angles = (frac * (hd as f64) / 2.0) as usize;
    let mut cos = Vec::with_capacity(t as usize * h2 * 4);
    let mut sin = Vec::with_capacity(t as usize * h2 * 4);
    for p in 0..t as usize {
        for j in 0..h2 {
            let (c, s) = if j < rope_angles {
                let inv = scale_inv_freq(1.0 / theta.powf(2.0 * j as f64 / hd as f64), scale);
                let a = p as f64 * inv;
                (a.cos(), a.sin())
            } else {
                (1.0, 0.0) // NoPE: identity
            };
            cos.extend_from_slice(&(c as f32).to_le_bytes());
            sin.extend_from_slice(&(s as f32).to_le_bytes());
        }
    }
    (cos, sin)
}

/// DSA indexer interleaved-RoPE cos/sin: `[ctx][index_dim/2]` where the first `rope_hd/2` angles are
/// the qk_rope=`rope_hd` freqs (theta) and the rest are identity (cos=1, sin=0). A HD=index_dim GPT-J
/// interleaved RoPE with this table rotates the first `rope_hd` dims of each 128-d index head and
/// passes the remaining dims — matching HF (the indexer reuses the main qk_rope RoPE on its first
/// qk_rope_head_dim dims). Verified equivalent to HF in scripts/glm52_indexer_oracle.py. [GLM52-DSA]
pub fn rope_tables_idx(t: u32, rope_hd: u32, index_dim: u32, theta: f64) -> (Vec<u8>, Vec<u8>) {
    let h2 = (index_dim / 2) as usize; // table stride per position (64)
    let real = (rope_hd / 2) as usize; // real rope angles (32); the qk_rope denominator is rope_hd
    let mut cos = Vec::with_capacity(t as usize * h2 * 4);
    let mut sin = Vec::with_capacity(t as usize * h2 * 4);
    for p in 0..t as usize {
        for j in 0..h2 {
            let (c, s) = if j < real {
                let inv = 1.0 / theta.powf(2.0 * j as f64 / rope_hd as f64);
                let a = p as f64 * inv;
                (a.cos(), a.sin())
            } else {
                (1.0, 0.0)
            };
            cos.extend_from_slice(&(c as f32).to_le_bytes());
            sin.extend_from_slice(&(s as f32).to_le_bytes());
        }
    }
    (cos, sin)
}

// --- blob-side recipe -------------------------------------------------------

/// [`GenTensor::kind`]: half-split RoPE cosine, from [`rope_tables`].
pub const GEN_ROPE_COS: u32 = 0;
/// [`GenTensor::kind`]: half-split RoPE sine, from [`rope_tables`].
pub const GEN_ROPE_SIN: u32 = 1;
/// [`GenTensor::kind`]: DSA interleaved-indexer cosine, from [`rope_tables_idx`].
pub const GEN_ROPE_IDX_COS: u32 = 2;
/// [`GenTensor::kind`]: DSA interleaved-indexer sine, from [`rope_tables_idx`].
pub const GEN_ROPE_IDX_SIN: u32 = 3;
/// [`GenTensor::kind`]: a 128-byte CUtensorMap over another tensor (sm_90a TMA prefill
/// GEMM, `PLOW_NV_TMA_GEMM`). bf16, rank-2 K-major, 128B swizzle, box {64, `scale`} —
/// the exact `tma_ws_gemm_bf16.cu` recipe. Field reuse: `ctx` = rows (globalDim[1]),
/// `hd` = K elements (globalDim[0]), `aux` = TARGET tensor handle, `scale` = box rows.
///
/// Unlike the RoPE kinds this is NOT a pure host function of scalars: the blob carries a
/// zero placeholder ([`GenTensor::generate`]) and the ENGINE re-encodes it in place once
/// the target's device address exists (`exec/gpu.rs`, after the upload loop). An engine
/// that only runs `generate()` (AMD) serves zeros — harmless, nothing dispatches TMA there.
pub const GEN_TMAP_BF16: u32 = 4;
/// [`GenTensor::kind`]: as [`GEN_TMAP_BF16`] but over an e4m3 (u8) K-major tensor —
/// box {128, `scale`} so the inner box stays one 128-byte swizzle atom. Same placeholder
/// contract; the engine encodes with `CU_TENSOR_MAP_DATA_TYPE_UINT8`.
pub const GEN_TMAP_E4M3: u32 = 5;
/// [`GenTensor::kind`]: a 256-byte K/V tensor-map PAIR for the flash-prefill TMA stager
/// (K's rank-3 map at +0, V's at +128 — FLASH_PREFILL has ONE spare t[] slot, so the pair
/// rides in one buffer). Each map: bf16, rank-3 {hd, ring_rows, n_kv_head}, 128B swizzle,
/// box {64, BKV=32, 1}. Field reuse: `ctx` = ring rows, `hd` = head_dim, `aux` = K tensor
/// handle, `scale` = V tensor handle, `frac` = n_kv_head (f64 carrying a small integer).
/// Same zero-placeholder contract as the other TMAP kinds.
pub const GEN_TMAP_KV_PAIR: u32 = 6;

/// [`GenTensor::scale`]: no inv_freq rescaling (Gemma / Qwen / GLM).
pub const ROPE_SCALE_NONE: u32 = 0;
/// [`GenTensor::scale`]: Llama-3.1 smooth low-frequency rescaling.
pub const ROPE_SCALE_LLAMA3: u32 = 1;

/// A recipe for one tensor the runtime materialises at bind time instead of
/// reading from the blob's init section. Mirrors `PlowGenTensor` in
/// `runtime/common/dev_blob.h`; locked by `packet/tests/dev_abi.rs`.
///
/// Fields are a flat union across every [`GEN_ROPE_COS`]-family kind rather than
/// a tagged variant, because this is a `#[repr(C)]` wire record the C runtime
/// casts directly. Slots a kind does not use are zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenTensor {
    /// Index into the blob's tensor decl table.
    pub tensor: u32,
    /// One of the `GEN_*` constants.
    pub kind: u32,
    /// Rows: context length the table spans.
    pub ctx: u32,
    /// Head dim for the `ROPE` kinds; `index_dim` for the `ROPE_IDX` kinds.
    pub hd: u32,
    /// `rope_hd` for the `ROPE_IDX` kinds; 0 otherwise.
    pub aux: u32,
    /// One of the `ROPE_SCALE_*` constants.
    pub scale: u32,
    pub theta: f64,
    /// Partial-rotary fraction. 1.0 = fully rotated.
    pub frac: f64,
    /// [`RopeScale::Llama3`] parameters; all 0 when `scale == ROPE_SCALE_NONE`.
    pub factor: f64,
    pub low: f64,
    pub high: f64,
    pub orig: f64,
}

const _: () = assert!(size_of::<GenTensor>() == 72);

impl GenTensor {
    /// Byte size of the table this recipe produces — `ctx` rows of `hd/2` f32 for the
    /// RoPE kinds, a fixed 128-byte descriptor for [`GEN_TMAP_BF16`].
    /// Lets a caller declare the tensor without expanding it.
    pub fn byte_len(&self) -> u64 {
        match self.kind {
            GEN_TMAP_BF16 | GEN_TMAP_E4M3 => 128,
            GEN_TMAP_KV_PAIR => 256,
            _ => self.ctx as u64 * (self.hd as u64 / 2) * 4,
        }
    }

    /// Materialise this recipe's bytes. The result is bit-identical to what the
    /// compiler would have baked into the init section.
    ///
    /// Returns `None` for an unknown `kind` — a blob from a newer compiler, which
    /// the caller must reject loudly rather than serve with a zeroed table.
    pub fn generate(&self) -> Option<Vec<u8>> {
        // Placeholder, not the descriptor: encoding needs the target's DEVICE address
        // (cuTensorMapEncodeTiled is host-driver work at bind time). The CUDA engine
        // overwrites these bytes after the upload loop; see GEN_TMAP_BF16's doc.
        if self.kind == GEN_TMAP_BF16 || self.kind == GEN_TMAP_E4M3 {
            return Some(vec![0u8; 128]);
        }
        if self.kind == GEN_TMAP_KV_PAIR {
            return Some(vec![0u8; 256]);
        }
        let scale = match self.scale {
            ROPE_SCALE_LLAMA3 => RopeScale::Llama3 {
                factor: self.factor,
                low: self.low,
                high: self.high,
                orig: self.orig,
            },
            _ => RopeScale::None,
        };
        let (cos, sin) = match self.kind {
            GEN_ROPE_COS | GEN_ROPE_SIN => {
                rope_tables(self.ctx, self.hd, self.theta, self.frac, scale)
            }
            GEN_ROPE_IDX_COS | GEN_ROPE_IDX_SIN => {
                rope_tables_idx(self.ctx, self.aux, self.hd, self.theta)
            }
            _ => return None,
        };
        Some(match self.kind {
            GEN_ROPE_COS | GEN_ROPE_IDX_COS => cos,
            _ => sin,
        })
    }

    /// The `(cos, sin)` recipe pair for a [`rope_tables`] table. `tensor` is left
    /// 0 — [`crate::devbuild::Builder::tensor_gen`] fills in the real handle.
    pub fn rope_pair(ctx: u32, hd: u32, theta: f64, frac: f64, scale: RopeScale) -> [GenTensor; 2] {
        let (skind, factor, low, high, orig) = match scale {
            RopeScale::None => (ROPE_SCALE_NONE, 0.0, 0.0, 0.0, 0.0),
            RopeScale::Llama3 {
                factor,
                low,
                high,
                orig,
            } => (ROPE_SCALE_LLAMA3, factor, low, high, orig),
        };
        let base = GenTensor {
            tensor: 0,
            kind: GEN_ROPE_COS,
            ctx,
            hd,
            aux: 0,
            scale: skind,
            theta,
            frac,
            factor,
            low,
            high,
            orig,
        };
        [
            base,
            GenTensor {
                kind: GEN_ROPE_SIN,
                ..base
            },
        ]
    }

    /// The recipe for a [`GEN_TMAP_E4M3`] descriptor over `target`, a `[rows][k]` e4m3
    /// K-major tensor.
    pub fn tmap_e4m3(target: u32, rows: u32, k: u32, box_rows: u32) -> GenTensor {
        GenTensor {
            kind: GEN_TMAP_E4M3,
            ..Self::tmap_bf16(target, rows, k, box_rows)
        }
    }

    /// The recipe for a [`GEN_TMAP_KV_PAIR`]: rank-3 maps over the K and V cache tensors
    /// (`[n_kv_head][ring_rows][hd]` bf16 each).
    pub fn tmap_kv_pair(
        k_target: u32,
        v_target: u32,
        ring_rows: u32,
        hd: u32,
        n_kv_head: u32,
    ) -> GenTensor {
        GenTensor {
            tensor: 0,
            kind: GEN_TMAP_KV_PAIR,
            ctx: ring_rows,
            hd,
            aux: k_target,
            scale: v_target,
            theta: 0.0,
            frac: n_kv_head as f64,
            factor: 0.0,
            low: 0.0,
            high: 0.0,
            orig: 0.0,
        }
    }

    /// The recipe for a [`GEN_TMAP_BF16`] descriptor over `target`, a `[rows][k]` bf16
    /// K-major tensor. `tensor` is left 0 — `Builder::tensor_gen` fills the real handle.
    pub fn tmap_bf16(target: u32, rows: u32, k: u32, box_rows: u32) -> GenTensor {
        GenTensor {
            tensor: 0,
            kind: GEN_TMAP_BF16,
            ctx: rows,
            hd: k,
            aux: target,
            scale: box_rows,
            theta: 0.0,
            frac: 0.0,
            factor: 0.0,
            low: 0.0,
            high: 0.0,
            orig: 0.0,
        }
    }

    /// The `(cos, sin)` recipe pair for a [`rope_tables_idx`] table.
    pub fn rope_idx_pair(ctx: u32, rope_hd: u32, index_dim: u32, theta: f64) -> [GenTensor; 2] {
        let base = GenTensor {
            tensor: 0,
            kind: GEN_ROPE_IDX_COS,
            ctx,
            hd: index_dim,
            aux: rope_hd,
            scale: ROPE_SCALE_NONE,
            theta,
            frac: 1.0,
            factor: 0.0,
            low: 0.0,
            high: 0.0,
            orig: 0.0,
        };
        [
            base,
            GenTensor {
                kind: GEN_ROPE_IDX_SIN,
                ..base
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recipe must reproduce `rope_tables` byte-for-byte — that equality is the
    /// entire safety argument for not shipping the expanded table.
    #[test]
    fn recipe_roundtrips_rope_tables() {
        // Gemma-4 full layers: partial rotary, so this also covers the NoPE tail.
        let (cos, sin) = rope_tables(512, 512, 1_000_000.0, 0.25, RopeScale::None);
        let [gc, gs] = GenTensor::rope_pair(512, 512, 1_000_000.0, 0.25, RopeScale::None);
        assert_eq!(gc.generate().unwrap(), cos);
        assert_eq!(gs.generate().unwrap(), sin);
    }

    #[test]
    fn recipe_roundtrips_llama3_scaling() {
        let scale = RopeScale::Llama3 {
            factor: 8.0,
            low: 1.0,
            high: 4.0,
            orig: 8192.0,
        };
        let (cos, sin) = rope_tables(256, 128, 500_000.0, 1.0, scale);
        let [gc, gs] = GenTensor::rope_pair(256, 128, 500_000.0, 1.0, scale);
        assert_eq!(gc.generate().unwrap(), cos);
        assert_eq!(gs.generate().unwrap(), sin);
    }

    #[test]
    fn recipe_roundtrips_idx_tables() {
        let (cos, sin) = rope_tables_idx(256, 64, 128, 8_000_000.0);
        let [gc, gs] = GenTensor::rope_idx_pair(256, 64, 128, 8_000_000.0);
        assert_eq!(gc.generate().unwrap(), cos);
        assert_eq!(gs.generate().unwrap(), sin);
    }

    /// The NoPE tail must be exact identity, not merely close: those dims pass
    /// through unrotated and a 1-ulp error there is a silent accuracy bug.
    #[test]
    fn nope_tail_is_exact_identity() {
        let (cos, sin) = rope_tables(4, 512, 1_000_000.0, 0.25, RopeScale::None);
        let c: Vec<f32> = cos
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let s: Vec<f32> = sin
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for p in 0..4 {
            for j in 64..256 {
                assert_eq!(c[p * 256 + j], 1.0);
                assert_eq!(s[p * 256 + j], 0.0);
            }
        }
    }

    #[test]
    fn unknown_kind_is_rejected() {
        let g = GenTensor {
            kind: 99,
            ..Default::default()
        };
        assert!(g.generate().is_none());
    }

    #[test]
    fn tmap_recipe_is_a_128_byte_placeholder() {
        let g = GenTensor::tmap_bf16(7, 512, 3840, 128);
        assert_eq!(g.byte_len(), 128);
        assert_eq!((g.aux, g.ctx, g.hd, g.scale), (7, 512, 3840, 128));
        // The placeholder must parse-pass everywhere (devblob.rs rejects None) but be
        // all-zero: the ENGINE overwrites it at bind time, and a zero descriptor fed to
        // TMA would fault rather than compute — never silently serve.
        assert_eq!(g.generate().unwrap(), vec![0u8; 128]);
    }
}
