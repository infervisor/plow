use crate::{Result, RuntimeError};

#[derive(Clone, Copy)]
pub struct Q8Matrix<'a> {
    bytes: &'a [u8],
    k: usize,
    n: usize,
}

pub struct Q8LinearPlan<'a> {
    weight: Q8Matrix<'a>,
    bias_f32_le: &'a [u8],
}

impl<'a> Q8Matrix<'a> {
    pub fn new(bytes: &'a [u8], k: usize, n: usize) -> Result<Self> {
        if k == 0 || n == 0 || !k.is_multiple_of(32) {
            return Err(rejected(format!("invalid Q8_0 matrix [{k}, {n}]")));
        }
        let expected = k
            .checked_div(32)
            .and_then(|blocks| blocks.checked_mul(34))
            .and_then(|row_bytes| row_bytes.checked_mul(n))
            .ok_or_else(|| rejected("Q8_0 matrix size overflows"))?;
        if bytes.len() != expected {
            return Err(rejected(format!(
                "Q8_0 payload has {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        Ok(Self { bytes, k, n })
    }

    pub fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    pub fn k(self) -> usize {
        self.k
    }

    pub fn n(self) -> usize {
        self.n
    }

    pub fn matmul(self, input: &[f32], m: usize, output: &mut [f32]) -> Result<()> {
        let input_len = m
            .checked_mul(self.k)
            .ok_or_else(|| rejected("Q8_0 input size overflows"))?;
        let output_len = m
            .checked_mul(self.n)
            .ok_or_else(|| rejected("Q8_0 output size overflows"))?;
        if input.len() != input_len || output.len() != output_len {
            return Err(rejected(format!(
                "Q8_0 matmul shape mismatch: matrix [{}, {}], M={m}, input {}, output {}",
                self.k,
                self.n,
                input.len(),
                output.len()
            )));
        }
        let row_bytes = (self.k / 32) * 34;
        for input_row in 0..m {
            let input = &input[input_row * self.k..(input_row + 1) * self.k];
            for output_row in 0..self.n {
                let mut sum = 0.0f64;
                let weights = &self.bytes[output_row * row_bytes..(output_row + 1) * row_bytes];
                for (block_index, block) in weights.chunks_exact(34).enumerate() {
                    let scale = f64::from(half_to_f32(u16::from_le_bytes([block[0], block[1]])));
                    let input = &input[block_index * 32..(block_index + 1) * 32];
                    for (&quantized, &activation) in block[2..].iter().zip(input) {
                        sum += scale * f64::from(quantized as i8) * f64::from(activation);
                    }
                }
                output[input_row * self.n + output_row] = sum as f32;
            }
        }
        Ok(())
    }
}

impl<'a> Q8LinearPlan<'a> {
    pub fn new(weight: Q8Matrix<'a>, bias_f32_le: &'a [u8]) -> Result<Self> {
        if weight.n.checked_mul(4) != Some(bias_f32_le.len()) {
            return Err(rejected("linear bias size mismatch"));
        }
        Ok(Self {
            weight,
            bias_f32_le,
        })
    }

    pub fn weight(&self) -> Q8Matrix<'a> {
        self.weight
    }

    pub fn bias_bytes(&self) -> &'a [u8] {
        self.bias_f32_le
    }

    pub fn matmul(&self, input: &[f32], m: usize, output: &mut [f32]) -> Result<()> {
        self.weight.matmul(input, m, output)?;
        for row in output.chunks_exact_mut(self.weight.n) {
            for (value, bytes) in row.iter_mut().zip(self.bias_f32_le.chunks_exact(4)) {
                *value += f32::from_le_bytes(bytes.try_into().unwrap());
            }
        }
        Ok(())
    }
}

fn half_to_f32(value: u16) -> f32 {
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

fn rejected(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("invalid linear operation: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_linear_adds_bias_to_every_row() {
        let mut weights = Vec::new();
        for row in 0..2 {
            weights.extend_from_slice(&0x3800u16.to_le_bytes());
            weights.extend((0..32).map(|column| (row * 3 + column) as i8 as u8));
        }
        let matrix = Q8Matrix::new(&weights, 32, 2).unwrap();
        let bias: Vec<_> = [1.25f32, -2.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let plan = Q8LinearPlan::new(matrix, &bias).unwrap();
        let input: Vec<_> = (0..64).map(|index| (index % 32) as f32 / 32.0).collect();
        let mut without_bias = [0.0; 4];
        matrix.matmul(&input, 2, &mut without_bias).unwrap();
        let mut output = [0.0; 4];
        plan.matmul(&input, 2, &mut output).unwrap();
        for row in 0..2 {
            assert_eq!(output[row * 2], without_bias[row * 2] + 1.25);
            assert_eq!(output[row * 2 + 1], without_bias[row * 2 + 1] - 2.5);
        }
    }
}
