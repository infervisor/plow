use crate::{Result, RuntimeError};

#[derive(Clone, Copy)]
pub struct LayerNormPlan<'a> {
    width: usize,
    epsilon: f32,
    weight_f32_le: &'a [u8],
    bias_f32_le: &'a [u8],
}

impl<'a> LayerNormPlan<'a> {
    pub fn new(
        width: usize,
        epsilon: f32,
        weight_f32_le: &'a [u8],
        bias_f32_le: &'a [u8],
    ) -> Result<Self> {
        if width == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(rejected("invalid layer normalization geometry"));
        }
        let bytes = width
            .checked_mul(4)
            .ok_or_else(|| rejected("layer normalization size overflows"))?;
        if weight_f32_le.len() != bytes || bias_f32_le.len() != bytes {
            return Err(rejected("layer normalization parameter size mismatch"));
        }
        Ok(Self {
            width,
            epsilon,
            weight_f32_le,
            bias_f32_le,
        })
    }

    pub fn width(self) -> usize {
        self.width
    }

    pub fn epsilon(self) -> f32 {
        self.epsilon
    }

    pub fn weight_bytes(self) -> &'a [u8] {
        self.weight_f32_le
    }

    pub fn bias_bytes(self) -> &'a [u8] {
        self.bias_f32_le
    }

    pub fn run(self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() || !input.len().is_multiple_of(self.width) {
            return Err(rejected("layer normalization input shape mismatch"));
        }
        for (input, output) in input
            .chunks_exact(self.width)
            .zip(output.chunks_exact_mut(self.width))
        {
            let mean = input.iter().map(|&value| f64::from(value)).sum::<f64>() / self.width as f64;
            let variance = input
                .iter()
                .map(|&value| {
                    let centered = f64::from(value) - mean;
                    centered * centered
                })
                .sum::<f64>()
                / self.width as f64;
            let scale = (variance + f64::from(self.epsilon)).sqrt().recip();
            for index in 0..self.width {
                let weight = decode(self.weight_f32_le, index);
                let bias = decode(self.bias_f32_le, index);
                output[index] =
                    (((f64::from(input[index]) - mean) * scale) as f32).mul_add(weight, bias);
            }
        }
        Ok(())
    }
}

fn decode(bytes: &[u8], index: usize) -> f32 {
    f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
}

fn rejected(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!(
        "invalid normalization operation: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_uses_population_variance_and_affine_parameters() {
        let weight: Vec<_> = [2.0f32, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let bias: Vec<_> = [1.0f32, -1.0, 0.5, -0.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let plan = LayerNormPlan::new(4, 1e-5, &weight, &bias).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        plan.run(&input, &mut output).unwrap();
        let inv = (1.25f32 + 1e-5).sqrt().recip();
        for index in 0..4 {
            let expected = (input[index] - 2.5) * inv * [2.0, 3.0, 4.0, 5.0][index]
                + [1.0, -1.0, 0.5, -0.5][index];
            assert!((output[index] - expected).abs() < 2e-6);
        }
    }
}
