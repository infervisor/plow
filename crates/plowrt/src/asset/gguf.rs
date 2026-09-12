use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use gguf_rs_lib::format::metadata::Metadata;
use gguf_rs_lib::format::types::GGUFTensorType;
use gguf_rs_lib::reader::file_reader::GGUFFileReader;

use crate::ops::linear::Q8Matrix;
use crate::{Result, RuntimeError};

pub struct GgufFile {
    path: PathBuf,
    map: memmap2::Mmap,
    catalog: GGUFFileReader<BufReader<File>>,
}

pub struct GgufTensor<'a> {
    pub name: &'a str,
    pub dimensions: &'a [u64],
    pub dtype: GGUFTensorType,
    pub bytes: &'a [u8],
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|source| RuntimeError::Io {
            path: path.to_owned(),
            source,
        })?;
        let catalog_file = file.try_clone().map_err(|source| RuntimeError::Io {
            path: path.to_owned(),
            source,
        })?;
        let catalog = GGUFFileReader::new(BufReader::new(catalog_file)).map_err(|error| {
            RuntimeError::Rejected(format!("invalid GGUF {}: {error}", path.display()))
        })?;
        // SAFETY: the mapping is retained for the lifetime of this object and Plow treats model
        // assets as immutable after load.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| RuntimeError::Io {
            path: path.to_owned(),
            source,
        })?;
        Ok(Self {
            path: path.to_owned(),
            map,
            catalog,
        })
    }

    pub fn metadata(&self) -> &Metadata {
        self.catalog.metadata()
    }

    pub fn tensor_count(&self) -> usize {
        self.catalog.tensor_count()
    }

    pub fn tensors(&self) -> impl Iterator<Item = GgufTensor<'_>> {
        self.catalog.tensor_infos().iter().map(|info| {
            self.tensor_from_info(info)
                .expect("validated GGUF descriptor must remain in mapped file")
        })
    }

    pub fn tensor(&self, name: &str) -> Result<GgufTensor<'_>> {
        let info = self
            .catalog
            .get_tensor_info(name)
            .ok_or_else(|| RuntimeError::Rejected(format!("GGUF tensor {name:?} is missing")))?;
        self.tensor_from_info(info)
    }

    fn tensor_from_info<'a>(
        &'a self,
        info: &'a gguf_rs_lib::tensor::TensorInfo,
    ) -> Result<GgufTensor<'a>> {
        let start = self
            .catalog
            .tensor_data_offset()
            .checked_add(info.data_offset())
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or_else(|| RuntimeError::Rejected("GGUF tensor offset overflows usize".into()))?;
        let length = usize::try_from(info.checked_expected_data_size().map_err(|error| {
            RuntimeError::Rejected(format!("invalid GGUF tensor {:?}: {error}", info.name()))
        })?)
        .map_err(|_| RuntimeError::Rejected("GGUF tensor size overflows usize".into()))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| RuntimeError::Rejected("GGUF tensor range overflows usize".into()))?;
        let bytes = self.map.get(start..end).ok_or_else(|| {
            RuntimeError::Rejected(format!(
                "GGUF tensor {:?} exceeds {}",
                info.name(),
                self.path.display()
            ))
        })?;
        Ok(GgufTensor {
            name: info.name(),
            dimensions: info.shape().dims(),
            dtype: info.tensor_type(),
            bytes,
        })
    }
}

impl<'a> GgufTensor<'a> {
    pub fn f16_values(&self) -> Result<Vec<f32>> {
        if self.dtype != GGUFTensorType::F16 {
            return Err(RuntimeError::Rejected(format!(
                "GGUF tensor {:?} is not F16",
                self.name
            )));
        }
        let elements = tensor_elements(self.dimensions)?;
        if self.bytes.len() != elements.checked_mul(2).unwrap_or(usize::MAX) {
            return Err(RuntimeError::Rejected(format!(
                "GGUF F16 tensor {:?} payload size mismatch",
                self.name
            )));
        }
        Ok(self
            .bytes
            .chunks_exact(2)
            .map(|bytes| f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
            .collect())
    }

    pub fn f32_values(&self) -> Result<Vec<f32>> {
        if self.dtype != GGUFTensorType::F32 {
            return Err(RuntimeError::Rejected(format!(
                "GGUF tensor {:?} is not F32",
                self.name
            )));
        }
        let elements = tensor_elements(self.dimensions)?;
        if self.bytes.len() != elements.checked_mul(4).unwrap_or(usize::MAX) {
            return Err(RuntimeError::Rejected(format!(
                "GGUF F32 tensor {:?} payload size mismatch",
                self.name
            )));
        }
        Ok(self
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect())
    }

    pub fn q8_0_matvec(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.q8_0_matmul(input, 1, output)
    }

    pub fn q8_0_matrix(&self) -> Result<Q8Matrix<'a>> {
        if self.dtype != GGUFTensorType::Q8_0 || self.dimensions.len() != 2 {
            return Err(RuntimeError::Rejected(format!(
                "GGUF tensor {:?} is not a Q8_0 matrix",
                self.name
            )));
        }
        let k = usize::try_from(self.dimensions[0])
            .map_err(|_| RuntimeError::Rejected("Q8_0 K overflows usize".into()))?;
        let n = usize::try_from(self.dimensions[1])
            .map_err(|_| RuntimeError::Rejected("Q8_0 N overflows usize".into()))?;
        Q8Matrix::new(self.bytes, k, n)
    }

    /// Multiplies row-major `input[m, k]` by this GGUF matrix `[k, n]`.
    pub fn q8_0_matmul(&self, input: &[f32], m: usize, output: &mut [f32]) -> Result<()> {
        self.q8_0_matrix()?.matmul(input, m, output)
    }
}

fn tensor_elements(dimensions: &[u64]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |count, &dimension| {
        usize::try_from(dimension)
            .ok()
            .and_then(|dimension| count.checked_mul(dimension))
            .ok_or_else(|| RuntimeError::Rejected("GGUF tensor shape overflows usize".into()))
    })
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = u32::from(value >> 15) << 31;
    let exponent = i32::from((value >> 10) & 0x1f);
    let fraction = u32::from(value & 0x03ff);
    let bits = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let shift = fraction.leading_zeros() - 21;
            let normalized = (fraction << shift) & 0x03ff;
            sign | ((127 - 14 - shift) << 23) | (normalized << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((u32::try_from(exponent - 15 + 127).unwrap()) << 23) | (fraction << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::{f16_to_f32, GgufTensor};
    use gguf_rs_lib::format::types::GGUFTensorType;

    #[test]
    fn q8_0_matvec_uses_signed_values_and_per_block_half_scale() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x3800u16.to_le_bytes()); // 0.5
        bytes.extend((0..32).map(|value| (value as i8 - 16) as u8));
        bytes.extend_from_slice(&0xbc00u16.to_le_bytes()); // -1.0
        bytes.extend((0..32).map(|value| (16 - value as i8) as u8));
        let tensor = GgufTensor {
            name: "weight",
            dimensions: &[32, 2],
            dtype: GGUFTensorType::Q8_0,
            bytes: &bytes,
        };
        let input: Vec<_> = (0..32).map(|value| value as f32 / 8.0 - 2.0).collect();
        let mut output = [0.0; 2];
        tensor.q8_0_matvec(&input, &mut output).unwrap();
        let expected0 = (0..32)
            .map(|index| 0.5 * f32::from(index as i8 - 16) * input[index])
            .sum::<f32>();
        let expected1 = (0..32)
            .map(|index| -f32::from(16 - index as i8) * input[index])
            .sum::<f32>();
        assert_eq!(output, [expected0, expected1]);
        assert_eq!(f16_to_f32(1), 2f32.powi(-24));
        assert!(f16_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn q8_0_matmul_uses_row_major_activations() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
        bytes.extend(0..32u8);
        let tensor = GgufTensor {
            name: "weight",
            dimensions: &[32, 1],
            dtype: GGUFTensorType::Q8_0,
            bytes: &bytes,
        };
        let input: Vec<_> = (0..64).map(|value| value as f32).collect();
        let mut output = [0.0; 2];
        tensor.q8_0_matmul(&input, 2, &mut output).unwrap();
        assert_eq!(output[0], (0..32).map(|i| (i * i) as f32).sum::<f32>());
        assert_eq!(
            output[1],
            (0..32).map(|i| (i * (i + 32)) as f32).sum::<f32>()
        );
    }

    #[test]
    fn f32_values_decode_little_endian_payload() {
        let bytes: Vec<_> = [1.25f32, -2.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let tensor = GgufTensor {
            name: "values",
            dimensions: &[2],
            dtype: GGUFTensorType::F32,
            bytes: &bytes,
        };
        assert_eq!(tensor.f32_values().unwrap(), [1.25, -2.5]);
    }

    #[test]
    fn f16_values_decode_signed_and_subnormal_payload() {
        let bytes: Vec<_> = [0xbc00u16, 1]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let tensor = GgufTensor {
            name: "values",
            dimensions: &[2],
            dtype: GGUFTensorType::F16,
            bytes: &bytes,
        };
        assert_eq!(tensor.f16_values().unwrap(), [-1.0, 2f32.powi(-24)]);
    }
}
