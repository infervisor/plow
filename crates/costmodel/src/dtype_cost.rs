//! Bridge between `nn_graph::DType` and the costmodel's compute/SRAM parameters.
//!
//! The costmodel historically worked with a single global `elem_bytes` + `MmaDtype`.
//! With per-op mixed precision, each op may have different weight and activation
//! dtypes. This module provides the conversion logic and an asymmetric working-set
//! calculation where the A-operand (activation, always in compute precision) and
//! B-operand (weight, possibly block-quantized) have different byte costs.

use crate::tile::TileShape;
use hwspec::MmaDtype;
use nn_graph::DType;

/// Per-op cost parameters derived from the op's weight and compute dtypes.
/// Passed to dtype-aware cost methods on [`CostModel`](crate::CostModel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostParams {
    /// Bytes per activation element (A-operand, in compute precision).
    pub activation_elem: u64,
    /// Bytes per weight element (B-operand). For standard dtypes this equals
    /// the element size; for block-quant types this is the *amortized* per-element
    /// cost from `DType::tile_bytes()` (includes packed scales).
    pub weight_elem: u64,
    /// Which MMA throughput rate to use for compute cost.
    pub mma_dtype: MmaDtype,
    /// Whether the weight is a block-quantized format (affects SRAM staging
    /// granularity — must stage in whole blocks, not individual elements).
    pub block_quant: bool,
    /// Block size for block-quant (e.g. 32 for Q4_0). 1 for non-block dtypes.
    pub block_size: u64,
    /// Whether the weight is MX FP4 (native Blackwell 4-bit path).
    /// Distinct from block_quant (GGUF dequant) — this uses hardware FP4 tensor cores.
    pub native_fp4: bool,
}

impl CostParams {
    /// Derive cost parameters from the weight dtype and compute dtype of an op.
    ///
    /// - `weight_dt`: The on-disk/in-memory dtype of the weight tensor (e.g. Q4_K, BF16, FP8)
    /// - `compute_dt`: The dtype the MMA actually runs at (e.g. BF16 after dequant)
    pub fn from_dtypes(weight_dt: DType, compute_dt: DType) -> Self {
        let activation_elem = elem_bytes_of(compute_dt);
        let mma_dtype = to_mma_dtype(compute_dt);

        if weight_dt.is_block_quant() {
            // For block-quant, amortized bytes per element from a reference tile
            // (use 256 elements as a reasonable amortization length — always ≥ block_size).
            let ref_count = 256u64;
            let total = weight_dt.tile_bytes(ref_count);
            let weight_elem = total.div_ceil(ref_count).max(1);
            CostParams {
                activation_elem,
                weight_elem,
                mma_dtype,
                block_quant: true,
                block_size: weight_dt.block_size() as u64,
                native_fp4: false,
            }
        } else {
            CostParams {
                activation_elem,
                weight_elem: elem_bytes_of(weight_dt),
                mma_dtype,
                block_quant: false,
                block_size: 1,
                native_fp4: weight_dt == DType::F4,
            }
        }
    }

    /// Default BF16 params (backward compatible with the old uniform `elem_bytes = 2`).
    pub fn bf16() -> Self {
        CostParams {
            activation_elem: 2,
            weight_elem: 2,
            mma_dtype: MmaDtype::Bf16,
            block_quant: false,
            block_size: 1,
            native_fp4: false,
        }
    }

    /// Asymmetric working-set bytes for a GEMM tile: A-operand uses activation
    /// element size, B-operand uses weight element size, both times buffering.
    ///
    /// ```text
    /// A: BM × BK × activation_elem × buffering
    /// B: BK × BN × weight_elem × buffering
    /// total = A + B
    /// ```
    pub fn working_set_bytes(self, tile: TileShape, buffering: u64) -> u64 {
        let a_bytes = (tile.bm * tile.bk) as u64 * self.activation_elem * buffering;
        let b_bytes = if self.block_quant {
            // For block-quant, use DType::tile_bytes on the actual element count
            // but we don't have the DType here — use the amortized weight_elem.
            (tile.bk * tile.bn) as u64 * self.weight_elem * buffering
        } else {
            (tile.bk * tile.bn) as u64 * self.weight_elem * buffering
        };
        a_bytes + b_bytes
    }

    /// Symmetric shortcut: average element size (for backward compatibility in
    /// callers that haven't been migrated yet).
    pub fn avg_elem_bytes(self) -> u64 {
        (self.activation_elem + self.weight_elem + 1) / 2
    }
}

/// Map a standard (non-block-quant) `DType` to its element byte size.
fn elem_bytes_of(dt: DType) -> u64 {
    match dt {
        DType::F32 | DType::I32 => 4,
        DType::I64 => 8,
        DType::BF16 | DType::F16 => 2,
        DType::F8E4M3 | DType::F8E5M2 | DType::I8 | DType::U8 => 1,
        DType::F4 | DType::Bool => 1, // 0.5 bytes, but SRAM staging rounds up to 1
        // Block-quant types as activation doesn't make sense; fall back to 2.
        _ => 2,
    }
}

/// Map a compute `DType` to the hardware `MmaDtype` that selects throughput.
pub fn to_mma_dtype(dt: DType) -> MmaDtype {
    match dt {
        DType::F16 => MmaDtype::Fp16,
        DType::BF16 => MmaDtype::Bf16,
        DType::F8E4M3 | DType::F8E5M2 => MmaDtype::Fp8,
        DType::F4 => MmaDtype::Fp4,
        DType::I8 | DType::U8 => MmaDtype::Int8,
        // Block-quant as compute → dequant target is bf16
        dt if dt.is_block_quant() => MmaDtype::Bf16,
        // F32 accumulator still uses bf16 tensor cores
        DType::F32 => MmaDtype::Bf16,
        _ => MmaDtype::Bf16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_symmetric() {
        let p = CostParams::from_dtypes(DType::BF16, DType::BF16);
        assert_eq!(p.activation_elem, 2);
        assert_eq!(p.weight_elem, 2);
        assert_eq!(p.mma_dtype, MmaDtype::Bf16);
        assert!(!p.block_quant);
    }

    #[test]
    fn fp8_halves_weight_bytes() {
        let p = CostParams::from_dtypes(DType::F8E4M3, DType::BF16);
        assert_eq!(p.activation_elem, 2); // compute still bf16 activations
        assert_eq!(p.weight_elem, 1); // fp8 = 1 byte
        assert_eq!(p.mma_dtype, MmaDtype::Bf16);
    }

    #[test]
    fn q4_k_block_quant() {
        let p = CostParams::from_dtypes(DType::Q4_K, DType::BF16);
        assert_eq!(p.activation_elem, 2);
        // Q4_K: 256 elements = 144 bytes → 144/256 ≈ 0.56, ceil → 1
        assert_eq!(p.weight_elem, 1);
        assert_eq!(p.mma_dtype, MmaDtype::Bf16);
        assert!(p.block_quant);
        assert_eq!(p.block_size, 256);
    }

    #[test]
    fn q8_0_block_quant() {
        let p = CostParams::from_dtypes(DType::Q8_0, DType::BF16);
        assert_eq!(p.activation_elem, 2);
        // Q8_0: 32 elements = 34 bytes → for 256 elems = 8 blocks × 34 = 272
        // 272 / 256 = ceil → 2
        assert_eq!(p.weight_elem, 2);
        assert!(p.block_quant);
        assert_eq!(p.block_size, 32);
    }

    #[test]
    fn asymmetric_working_set() {
        let p = CostParams::from_dtypes(DType::F8E4M3, DType::BF16);
        let tile = TileShape::new(128, 256, 64);
        let ws = p.working_set_bytes(tile, 2);
        // A: 128 * 64 * 2 * 2 = 32768
        // B: 64 * 256 * 1 * 2 = 32768
        // total = 65536
        assert_eq!(ws, 32768 + 32768);

        // Compare with symmetric bf16: both A and B at 2 bytes
        let p_bf16 = CostParams::bf16();
        let ws_bf16 = p_bf16.working_set_bytes(tile, 2);
        // A: 128 * 64 * 2 * 2 = 32768
        // B: 64 * 256 * 2 * 2 = 65536
        // total = 98304
        assert_eq!(ws_bf16, 32768 + 65536);
        assert!(ws < ws_bf16, "fp8 weights should use less SRAM");
    }

    #[test]
    fn to_mma_dtype_mapping() {
        assert_eq!(to_mma_dtype(DType::BF16), MmaDtype::Bf16);
        assert_eq!(to_mma_dtype(DType::F16), MmaDtype::Fp16);
        assert_eq!(to_mma_dtype(DType::F8E4M3), MmaDtype::Fp8);
        assert_eq!(to_mma_dtype(DType::F4), MmaDtype::Fp4);
        assert_eq!(to_mma_dtype(DType::I8), MmaDtype::Int8);
        // Block-quant computes via dequant → bf16
        assert_eq!(to_mma_dtype(DType::Q4_0), MmaDtype::Bf16);
        assert_eq!(to_mma_dtype(DType::Q6_K), MmaDtype::Bf16);
    }

    #[test]
    fn mx_fp4_native_path() {
        let p = CostParams::from_dtypes(DType::F4, DType::BF16);
        assert_eq!(p.activation_elem, 2);
        assert_eq!(p.weight_elem, 1); // 0.5 bytes rounds up to 1
        assert_eq!(p.mma_dtype, MmaDtype::Bf16); // compute still in bf16
        assert!(!p.block_quant); // NOT block-quant (native hardware path)
        assert!(p.native_fp4); // triggers VARIANT_FP4
    }
}
