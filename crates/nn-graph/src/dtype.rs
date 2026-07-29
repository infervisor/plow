//! Element data types.
//!
//! Covers IEEE/bfloat scalars AND GGUF block-quantized formats. Block-quant
//! types are "one tensor" in the graph — scales are packed inline with data,
//! not separate graph edges. The [`DType::tile_bytes`] method accounts for
//! both data and scale overhead so the cost model sizes SRAM staging correctly.

use std::fmt;

/// Element data type carried per-tensor through the graph.
///
/// Block-quantized variants (Q4_0 through IQ4_NL) match the GGUF spec's type
/// enum. The graph treats them as opaque storage types on weight edges —
/// `Op::Linear` topology is unchanged, only the physical byte layout differs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[allow(non_camel_case_types)]
pub enum DType {
    // --- IEEE / bfloat scalars ------------------------------------------------
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    /// Blackwell / MX microscaling FP4 (nibble-packed, 32-element groups with
    /// one fp8 scale each). Treated as a scalar type for hardware that natively
    /// accelerates it; falls back to dequant-to-f16 otherwise.
    F4,
    I32,
    I64,
    I8,
    U8,
    Bool,

    // --- GGUF block-quantized types ------------------------------------------
    // Each stores `block_size` logical elements in `block_bytes` bytes (data +
    // scales + optional min/zero interleaved). The kernel unpacks in SRAM.

    /// 4-bit quantization, block_size=32, 18 bytes/block (2 scale + 16 data).
    Q4_0,
    /// 4-bit with min, block_size=32, 20 bytes/block (2 scale + 2 min + 16 data).
    Q4_1,
    /// 5-bit, block_size=32, 22 bytes/block.
    Q5_0,
    /// 5-bit with min, block_size=32, 24 bytes/block.
    Q5_1,
    /// 8-bit, block_size=32, 34 bytes/block (2 scale + 32 data).
    Q8_0,
    /// 8-bit with sum, block_size=32, 36 bytes/block.
    Q8_1,

    // --- GGUF k-quant types (super-blocks of 256 elements) -------------------
    /// 2-bit k-quant, 256 elements, 84 bytes/super-block.
    Q2_K,
    /// 3-bit k-quant, 256 elements, 110 bytes/super-block.
    Q3_K,
    /// 4-bit k-quant, 256 elements, 144 bytes/super-block.
    Q4_K,
    /// 5-bit k-quant, 256 elements, 176 bytes/super-block.
    Q5_K,
    /// 6-bit k-quant, 256 elements, 210 bytes/super-block.
    Q6_K,

    // --- Importance-matrix quant (imatrix) ------------------------------------
    /// 4-bit non-linear lookup, block_size=32, 18 bytes/block.
    IQ4_NL,
}

impl DType {
    /// Bytes per scalar element for non-block types. Returns `None` for
    /// block-quantized types (use [`DType::tile_bytes`] instead).
    pub fn byte_size(self) -> Option<u32> {
        Some(match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F8E4M3 | DType::F8E5M2 | DType::I8 | DType::U8 | DType::Bool => 1,
            DType::F4 => return None, // nibble-packed
            DType::I64 => 8,
            _ => return None, // block-quant
        })
    }

    /// Whether this is a block-quantized (GGUF) type.
    pub fn is_block_quant(self) -> bool {
        matches!(
            self,
            DType::Q4_0
                | DType::Q4_1
                | DType::Q5_0
                | DType::Q5_1
                | DType::Q8_0
                | DType::Q8_1
                | DType::Q2_K
                | DType::Q3_K
                | DType::Q4_K
                | DType::Q5_K
                | DType::Q6_K
                | DType::IQ4_NL
        )
    }

    /// Logical elements per block (1 for scalar types).
    pub fn block_size(self) -> u32 {
        match self {
            // k-quant super-blocks
            DType::Q2_K | DType::Q3_K | DType::Q4_K | DType::Q5_K | DType::Q6_K => 256,
            // Standard GGUF blocks
            DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Q8_0
            | DType::Q8_1 | DType::IQ4_NL => 32,
            // MX FP4: 32-element groups
            DType::F4 => 32,
            // Scalar types
            _ => 1,
        }
    }

    /// Bytes per block (including data + scales + metadata). For scalar types
    /// this equals `byte_size().unwrap()`.
    pub fn block_bytes(self) -> u32 {
        match self {
            // Scalar types
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F8E4M3 | DType::F8E5M2 | DType::I8 | DType::U8 | DType::Bool => 1,
            DType::I64 => 8,
            // MX FP4: 32 elements × 0.5B + 1B scale = 17 bytes per 32-element group
            DType::F4 => 17,
            // GGUF standard blocks (block_size = 32)
            DType::Q4_0 => 18,   // 2B scale + 16B data
            DType::Q4_1 => 20,   // 2B scale + 2B min + 16B data
            DType::Q5_0 => 22,   // 2B scale + 4B high-bits + 16B low-nibbles
            DType::Q5_1 => 24,   // 2B scale + 2B min + 4B high + 16B low
            DType::Q8_0 => 34,   // 2B scale + 32B data
            DType::Q8_1 => 36,   // 2B scale + 2B sum + 32B data
            DType::IQ4_NL => 18, // 2B scale + 16B data (with non-linear lookup)
            // GGUF k-quant super-blocks (block_size = 256)
            DType::Q2_K => 84,
            DType::Q3_K => 110,
            DType::Q4_K => 144,
            DType::Q5_K => 176,
            DType::Q6_K => 210,
        }
    }

    /// Total bytes to store `count` logical elements in this dtype, including
    /// block overhead (scales, mins). This is what SRAM staging / HBM budget
    /// should use — it accounts for the packed block structure.
    ///
    /// For scalar types: `count * byte_size`.
    /// For block-quant: `ceil(count / block_size) * block_bytes`.
    pub fn tile_bytes(self, count: u64) -> u64 {
        let bs = self.block_size() as u64;
        let bb = self.block_bytes() as u64;
        if bs == 1 {
            // Scalar type
            count * bb
        } else {
            // Block-quantized: round up to full blocks
            count.div_ceil(bs) * bb
        }
    }

    /// Effective bytes per element (for cost-model bandwidth estimates).
    /// For block-quant types this is `block_bytes / block_size` as a ratio.
    pub fn effective_bits_per_element(self) -> f64 {
        (self.block_bytes() as f64 * 8.0) / self.block_size() as f64
    }

    /// The target compute dtype when this type is dequantized at load time
    /// (the fallback path when hardware doesn't natively accelerate this format).
    pub fn dequant_target(self) -> DType {
        match self {
            // Block-quant → BF16 (the safe default for modern GPUs)
            d if d.is_block_quant() => DType::BF16,
            // MX FP4 → F16 (Blackwell native, else dequant to fp16)
            DType::F4 => DType::F16,
            // FP8 variants dequant to BF16
            DType::F8E4M3 | DType::F8E5M2 => DType::BF16,
            // Already a compute type
            _ => self,
        }
    }

    /// Map from a GGUF type enum integer (from the file header) to our DType.
    /// Returns `None` for unsupported/unknown GGUF type ids.
    pub fn from_gguf_type(gguf_type: u32) -> Option<DType> {
        Some(match gguf_type {
            0 => DType::F32,
            1 => DType::F16,
            2 => DType::Q4_0,
            3 => DType::Q4_1,
            6 => DType::Q5_0,
            7 => DType::Q5_1,
            8 => DType::Q8_0,
            9 => DType::Q8_1,
            10 => DType::Q2_K,
            11 => DType::Q3_K,
            12 => DType::Q4_K,
            13 => DType::Q5_K,
            14 => DType::Q6_K,
            // 15 => Q8_K (rarely used, skip for now)
            // 16 => IQ2_XXS
            // ...
            20 => DType::IQ4_NL,
            28 => DType::BF16,
            30 => DType::F8E4M3, // GGUF "F8_E4M3" (unreleased spec extension)
            _ => return None,
        })
    }

    /// Map to the safetensors dtype string (for manifest / weight-loading).
    pub fn safetensors_name(self) -> Option<&'static str> {
        Some(match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::F8E4M3 => "F8_E4M3",
            DType::F8E5M2 => "F8_E5M2",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::I8 => "I8",
            DType::U8 => "U8",
            DType::Bool => "BOOL",
            _ => return None, // block-quant types don't exist in safetensors
        })
    }

    /// Parse from a safetensors dtype string.
    pub fn from_safetensors_name(s: &str) -> Option<DType> {
        Some(match s {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F8_E4M3" => DType::F8E4M3,
            "F8_E5M2" => DType::F8E5M2,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "I8" => DType::I8,
            "U8" => DType::U8,
            "BOOL" => DType::Bool,
            _ => return None,
        })
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F8E4M3 => write!(f, "f8e4m3"),
            DType::F8E5M2 => write!(f, "f8e5m2"),
            DType::F4 => write!(f, "f4"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::I8 => write!(f, "i8"),
            DType::U8 => write!(f, "u8"),
            DType::Bool => write!(f, "bool"),
            DType::Q4_0 => write!(f, "q4_0"),
            DType::Q4_1 => write!(f, "q4_1"),
            DType::Q5_0 => write!(f, "q5_0"),
            DType::Q5_1 => write!(f, "q5_1"),
            DType::Q8_0 => write!(f, "q8_0"),
            DType::Q8_1 => write!(f, "q8_1"),
            DType::Q2_K => write!(f, "q2_k"),
            DType::Q3_K => write!(f, "q3_k"),
            DType::Q4_K => write!(f, "q4_k"),
            DType::Q5_K => write!(f, "q5_k"),
            DType::Q6_K => write!(f, "q6_k"),
            DType::IQ4_NL => write!(f, "iq4_nl"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_tile_bytes() {
        assert_eq!(DType::BF16.tile_bytes(1024), 2048);
        assert_eq!(DType::F32.tile_bytes(1024), 4096);
        assert_eq!(DType::F8E4M3.tile_bytes(1024), 1024);
    }

    #[test]
    fn block_quant_tile_bytes() {
        // Q4_K: 256-element super-blocks, 144 bytes each
        // 1024 elements = 4 blocks × 144 = 576 bytes
        assert_eq!(DType::Q4_K.tile_bytes(1024), 576);
        // 256 elements = 1 block × 144
        assert_eq!(DType::Q4_K.tile_bytes(256), 144);
        // 257 elements = 2 blocks × 144 (rounds up)
        assert_eq!(DType::Q4_K.tile_bytes(257), 288);

        // Q8_0: 32-element blocks, 34 bytes each
        // 1024 elements = 32 blocks × 34 = 1088 bytes
        assert_eq!(DType::Q8_0.tile_bytes(1024), 1088);
    }

    #[test]
    fn effective_bits() {
        // BF16: 16 bits per element
        assert!((DType::BF16.effective_bits_per_element() - 16.0).abs() < 0.01);
        // Q4_K: 144*8/256 = 4.5 bits per element
        assert!((DType::Q4_K.effective_bits_per_element() - 4.5).abs() < 0.01);
        // Q8_0: 34*8/32 = 8.5 bits per element
        assert!((DType::Q8_0.effective_bits_per_element() - 8.5).abs() < 0.01);
    }

    #[test]
    fn gguf_type_roundtrip() {
        assert_eq!(DType::from_gguf_type(12), Some(DType::Q4_K));
        assert_eq!(DType::from_gguf_type(0), Some(DType::F32));
        assert_eq!(DType::from_gguf_type(1), Some(DType::F16));
        assert_eq!(DType::from_gguf_type(28), Some(DType::BF16));
        assert_eq!(DType::from_gguf_type(99), None);
    }

    #[test]
    fn safetensors_roundtrip() {
        for dt in [DType::F32, DType::F16, DType::BF16, DType::F8E4M3, DType::I32] {
            let name = dt.safetensors_name().unwrap();
            assert_eq!(DType::from_safetensors_name(name), Some(dt));
        }
        // Block-quant types have no safetensors representation
        assert_eq!(DType::Q4_K.safetensors_name(), None);
    }

    #[test]
    fn dequant_targets() {
        assert_eq!(DType::Q4_K.dequant_target(), DType::BF16);
        assert_eq!(DType::Q8_0.dequant_target(), DType::BF16);
        assert_eq!(DType::F4.dequant_target(), DType::F16);
        assert_eq!(DType::BF16.dequant_target(), DType::BF16); // identity
    }

    #[test]
    fn mx_fp4_tile_bytes() {
        // MX FP4: 32-element groups, 17 bytes per group
        // 1024 elements = 32 groups × 17 = 544 bytes
        assert_eq!(DType::F4.tile_bytes(1024), 544);
    }
}
