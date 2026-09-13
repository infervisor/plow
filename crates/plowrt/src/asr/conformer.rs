use crate::ops::linear::Q8Matrix;
use crate::ops::norm::LayerNormPlan;
use crate::{Result, RuntimeError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionMask {
    Full,
    ChunkedLimited {
        chunk_size: usize,
        left_chunks: usize,
    },
}

#[derive(Clone, Copy)]
pub struct FeedForwardPlan<'a> {
    pub norm: LayerNormPlan<'a>,
    pub expand: Q8Matrix<'a>,
    pub contract: Q8Matrix<'a>,
}

#[derive(Clone, Copy)]
pub struct RelativeAttentionPlan<'a> {
    pub norm: LayerNormPlan<'a>,
    pub query: Q8Matrix<'a>,
    pub key: Q8Matrix<'a>,
    pub value: Q8Matrix<'a>,
    pub position: Q8Matrix<'a>,
    pub output: Q8Matrix<'a>,
    pub bias_u_f32_le: &'a [u8],
    pub bias_v_f32_le: &'a [u8],
    pub position_f32_le: &'a [u8],
    pub position_count: usize,
    pub position_center: usize,
    pub heads: usize,
    pub mask: AttentionMask,
}

#[derive(Clone, Copy)]
pub struct ConvolutionPlan<'a> {
    pub norm: LayerNormPlan<'a>,
    pub pointwise_in: Q8Matrix<'a>,
    pub depthwise_f16_le: &'a [u8],
    pub kernel: usize,
    pub channel_norm: LayerNormPlan<'a>,
    pub pointwise_out: Q8Matrix<'a>,
}

#[derive(Clone, Copy)]
pub struct ConformerBlockPlan<'a> {
    width: usize,
    pub feed_forward1: FeedForwardPlan<'a>,
    pub attention: RelativeAttentionPlan<'a>,
    pub convolution: ConvolutionPlan<'a>,
    pub feed_forward2: FeedForwardPlan<'a>,
    pub output_norm: LayerNormPlan<'a>,
}

pub struct CpuConformerBlock<'a> {
    plan: ConformerBlockPlan<'a>,
}

pub struct ConformerEncoderPlan<'a> {
    width: usize,
    blocks: Vec<ConformerBlockPlan<'a>>,
}

pub struct CpuConformerEncoder<'a> {
    plan: ConformerEncoderPlan<'a>,
}

impl<'a> ConformerBlockPlan<'a> {
    pub fn new(
        width: usize,
        feed_forward1: FeedForwardPlan<'a>,
        attention: RelativeAttentionPlan<'a>,
        convolution: ConvolutionPlan<'a>,
        feed_forward2: FeedForwardPlan<'a>,
        output_norm: LayerNormPlan<'a>,
    ) -> Result<Self> {
        if width == 0 {
            return Err(rejected("block width is zero"));
        }
        validate_feed_forward(feed_forward1, width, "feed_forward1")?;
        validate_feed_forward(feed_forward2, width, "feed_forward2")?;
        validate_attention(attention, width)?;
        validate_convolution(convolution, width)?;
        if output_norm.width() != width {
            return Err(rejected("output normalization width mismatch"));
        }
        Ok(Self {
            width,
            feed_forward1,
            attention,
            convolution,
            feed_forward2,
            output_norm,
        })
    }

    pub fn width(self) -> usize {
        self.width
    }
}

impl<'a> CpuConformerBlock<'a> {
    pub fn new(plan: ConformerBlockPlan<'a>) -> Self {
        Self { plan }
    }

    pub fn run(&self, input: &[f32], frames: usize) -> Result<Vec<f32>> {
        let width = self.plan.width;
        if frames == 0 || frames.checked_mul(width) != Some(input.len()) {
            return Err(rejected("block input shape mismatch"));
        }
        let mut residual = input.to_vec();
        self.feed_forward(&mut residual, frames, self.plan.feed_forward1)?;
        self.attention(&mut residual, frames)?;
        self.convolution(&mut residual, frames)?;
        self.feed_forward(&mut residual, frames, self.plan.feed_forward2)?;
        let mut output = vec![0.0; residual.len()];
        self.plan.output_norm.run(&residual, &mut output)?;
        Ok(output)
    }

    fn feed_forward(
        &self,
        residual: &mut [f32],
        frames: usize,
        plan: FeedForwardPlan<'a>,
    ) -> Result<()> {
        let mut normalized = vec![0.0; residual.len()];
        plan.norm.run(residual, &mut normalized)?;
        let hidden = plan.expand.n();
        let expanded_len = frames
            .checked_mul(hidden)
            .ok_or_else(|| rejected("feed-forward workspace size overflows"))?;
        let mut expanded = vec![0.0; expanded_len];
        plan.expand.matmul(&normalized, frames, &mut expanded)?;
        expanded.iter_mut().for_each(|value| *value = silu(*value));
        let mut output = vec![0.0; residual.len()];
        plan.contract.matmul(&expanded, frames, &mut output)?;
        for (residual, output) in residual.iter_mut().zip(output) {
            *residual = output.mul_add(0.5, *residual);
        }
        Ok(())
    }

    fn attention(&self, residual: &mut [f32], frames: usize) -> Result<()> {
        let plan = self.plan.attention;
        let width = self.plan.width;
        let head_width = width / plan.heads;
        let mut normalized = vec![0.0; residual.len()];
        plan.norm.run(residual, &mut normalized)?;
        let mut query = vec![0.0; residual.len()];
        let mut key = vec![0.0; residual.len()];
        let mut value = vec![0.0; residual.len()];
        plan.query.matmul(&normalized, frames, &mut query)?;
        plan.key.matmul(&normalized, frames, &mut key)?;
        plan.value.matmul(&normalized, frames, &mut value)?;

        let relative_count = frames
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| rejected("relative position extent overflows"))?;
        let position_end = plan
            .position_center
            .checked_add(frames)
            .ok_or_else(|| rejected("relative position extent overflows"))?;
        let table_end = plan
            .position_count
            .checked_add(1)
            .ok_or_else(|| rejected("relative position table size overflows"))?;
        if frames > plan.position_center || position_end > table_end {
            return Err(rejected("input exceeds relative position table"));
        }
        let position_start = plan.position_center - frames;
        let position_bytes = position_start
            .checked_mul(width)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| rejected("relative position offset overflows"))?;
        let position_len = relative_count
            .checked_mul(width)
            .ok_or_else(|| rejected("relative position size overflows"))?;
        let mut position_input = vec![0.0; position_len];
        for (index, output) in position_input.iter_mut().enumerate() {
            *output = decode_f32(plan.position_f32_le, position_bytes / 4 + index);
        }
        let mut position = vec![0.0; position_len];
        plan.position
            .matmul(&position_input, relative_count, &mut position)?;

        let mut context = vec![0.0; residual.len()];
        let score_scale = (head_width as f32).sqrt().recip();
        let mut scores = vec![0.0; frames];
        for head in 0..plan.heads {
            let bias_base = head * head_width;
            for query_frame in 0..frames {
                let query_base = query_frame * width + bias_base;
                let mut maximum = f32::NEG_INFINITY;
                for key_frame in 0..frames {
                    if !allowed(plan.mask, query_frame, key_frame) {
                        scores[key_frame] = f32::NEG_INFINITY;
                        continue;
                    }
                    let key_base = key_frame * width + bias_base;
                    let position_index = frames - 1 + key_frame - query_frame;
                    let position_base = position_index * width + bias_base;
                    let mut content = 0.0f64;
                    let mut relative = 0.0f64;
                    for column in 0..head_width {
                        let q = query[query_base + column];
                        let u = decode_f32(plan.bias_u_f32_le, bias_base + column);
                        let v = decode_f32(plan.bias_v_f32_le, bias_base + column);
                        content += f64::from(key[key_base + column]) * f64::from(q + u);
                        relative += f64::from(position[position_base + column]) * f64::from(q + v);
                    }
                    let score = ((content + relative) as f32) * score_scale;
                    scores[key_frame] = score;
                    maximum = maximum.max(score);
                }
                let mut denominator = 0.0f64;
                for score in &mut scores {
                    if score.is_finite() {
                        *score = (*score - maximum).exp();
                        denominator += f64::from(*score);
                    } else {
                        *score = 0.0;
                    }
                }
                if denominator == 0.0 {
                    return Err(rejected("attention row is fully masked"));
                }
                let inverse = denominator.recip() as f32;
                for column in 0..head_width {
                    let mut sum = 0.0f64;
                    for key_frame in 0..frames {
                        let value_index = key_frame * width + bias_base + column;
                        sum +=
                            f64::from(scores[key_frame] * inverse) * f64::from(value[value_index]);
                    }
                    context[query_base + column] = sum as f32;
                }
            }
        }
        let mut output = vec![0.0; residual.len()];
        plan.output.matmul(&context, frames, &mut output)?;
        for (residual, output) in residual.iter_mut().zip(output) {
            *residual += output;
        }
        Ok(())
    }

    fn convolution(&self, residual: &mut [f32], frames: usize) -> Result<()> {
        let plan = self.plan.convolution;
        let width = self.plan.width;
        let mut normalized = vec![0.0; residual.len()];
        plan.norm.run(residual, &mut normalized)?;
        let pointwise_len = residual
            .len()
            .checked_mul(2)
            .ok_or_else(|| rejected("convolution workspace size overflows"))?;
        let mut pointwise = vec![0.0; pointwise_len];
        plan.pointwise_in
            .matmul(&normalized, frames, &mut pointwise)?;
        let mut gated = vec![0.0; residual.len()];
        for frame in 0..frames {
            let input = &pointwise[frame * width * 2..(frame + 1) * width * 2];
            let output = &mut gated[frame * width..(frame + 1) * width];
            for channel in 0..width {
                output[channel] = input[channel] * sigmoid(input[width + channel]);
            }
        }
        let mut depthwise = vec![0.0; residual.len()];
        for frame in 0..frames {
            for channel in 0..width {
                let mut sum = 0.0f32;
                for tap in 0..plan.kernel {
                    let Some(source_frame) = (frame + tap + 1).checked_sub(plan.kernel) else {
                        continue;
                    };
                    let weight = decode_f16(plan.depthwise_f16_le, channel * plan.kernel + tap);
                    sum = weight.mul_add(gated[source_frame * width + channel], sum);
                }
                depthwise[frame * width + channel] = sum;
            }
        }
        let mut activated = vec![0.0; residual.len()];
        plan.channel_norm.run(&depthwise, &mut activated)?;
        activated.iter_mut().for_each(|value| *value = silu(*value));
        let mut output = vec![0.0; residual.len()];
        plan.pointwise_out.matmul(&activated, frames, &mut output)?;
        for (residual, output) in residual.iter_mut().zip(output) {
            *residual += output;
        }
        Ok(())
    }
}

impl<'a> ConformerEncoderPlan<'a> {
    pub fn new(blocks: Vec<ConformerBlockPlan<'a>>) -> Result<Self> {
        let width = blocks
            .first()
            .map(|block| block.width())
            .ok_or_else(|| rejected("encoder has no blocks"))?;
        if blocks.iter().any(|block| block.width() != width) {
            return Err(rejected("encoder block widths differ"));
        }
        Ok(Self { width, blocks })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn blocks(&self) -> &[ConformerBlockPlan<'a>] {
        &self.blocks
    }
}

impl<'a> CpuConformerEncoder<'a> {
    pub fn new(plan: ConformerEncoderPlan<'a>) -> Self {
        Self { plan }
    }

    pub fn run(&self, input: &[f32], frames: usize) -> Result<Vec<f32>> {
        if frames.checked_mul(self.plan.width) != Some(input.len()) {
            return Err(rejected("encoder input shape mismatch"));
        }
        let mut output = input.to_vec();
        for &block in &self.plan.blocks {
            output = CpuConformerBlock::new(block).run(&output, frames)?;
        }
        Ok(output)
    }
}

fn validate_feed_forward(plan: FeedForwardPlan<'_>, width: usize, name: &str) -> Result<()> {
    if plan.norm.width() != width
        || plan.expand.k() != width
        || plan.contract.k() != plan.expand.n()
        || plan.contract.n() != width
    {
        return Err(rejected(format!("{name} geometry mismatch")));
    }
    Ok(())
}

fn validate_attention(plan: RelativeAttentionPlan<'_>, width: usize) -> Result<()> {
    let parameter_bytes = width
        .checked_mul(4)
        .ok_or_else(|| rejected("attention parameter size overflows"))?;
    if plan.norm.width() != width
        || plan.heads == 0
        || !width.is_multiple_of(plan.heads)
        || [plan.query, plan.key, plan.value, plan.position, plan.output]
            .iter()
            .any(|matrix| matrix.k() != width || matrix.n() != width)
        || plan.bias_u_f32_le.len() != parameter_bytes
        || plan.bias_v_f32_le.len() != parameter_bytes
        || plan
            .position_count
            .checked_mul(width)
            .and_then(|n| n.checked_mul(4))
            != Some(plan.position_f32_le.len())
        || plan.position_center >= plan.position_count
    {
        return Err(rejected("relative attention geometry mismatch"));
    }
    if let AttentionMask::ChunkedLimited {
        chunk_size,
        left_chunks: _,
    } = plan.mask
    {
        if chunk_size == 0 {
            return Err(rejected("attention chunk size is zero"));
        }
    }
    Ok(())
}

fn validate_convolution(plan: ConvolutionPlan<'_>, width: usize) -> Result<()> {
    let doubled_width = width
        .checked_mul(2)
        .ok_or_else(|| rejected("convolution width overflows"))?;
    let depthwise_bytes = width
        .checked_mul(plan.kernel)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| rejected("depthwise convolution size overflows"))?;
    if plan.norm.width() != width
        || plan.pointwise_in.k() != width
        || plan.pointwise_in.n() != doubled_width
        || plan.kernel == 0
        || plan.depthwise_f16_le.len() != depthwise_bytes
        || plan.channel_norm.width() != width
        || plan.pointwise_out.k() != width
        || plan.pointwise_out.n() != width
    {
        return Err(rejected("convolution geometry mismatch"));
    }
    Ok(())
}

fn allowed(mask: AttentionMask, query: usize, key: usize) -> bool {
    match mask {
        AttentionMask::Full => true,
        AttentionMask::ChunkedLimited {
            chunk_size,
            left_chunks,
        } => {
            let query_chunk = query / chunk_size;
            let key_chunk = key / chunk_size;
            key_chunk <= query_chunk && query_chunk - key_chunk <= left_chunks
        }
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn decode_f32(bytes: &[u8], index: usize) -> f32 {
    f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
}

fn decode_f16(bytes: &[u8], index: usize) -> f32 {
    half_to_f32(u16::from_le_bytes(
        bytes[index * 2..index * 2 + 2].try_into().unwrap(),
    ))
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
    RuntimeError::Rejected(format!("invalid Conformer block: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_mask_includes_lookahead_inside_current_chunk() {
        let mask = AttentionMask::ChunkedLimited {
            chunk_size: 4,
            left_chunks: 2,
        };
        assert!(allowed(mask, 0, 3));
        assert!(!allowed(mask, 3, 4));
        assert!(allowed(mask, 11, 4));
        assert!(!allowed(mask, 12, 3));
    }

    #[test]
    fn relative_shift_selects_center_plus_key_minus_query() {
        let frames = 4;
        let indices: Vec<_> = (0..frames)
            .flat_map(|query| (0..frames).map(move |key| frames - 1 + key - query))
            .collect();
        assert_eq!(indices, [3, 4, 5, 6, 2, 3, 4, 5, 1, 2, 3, 4, 0, 1, 2, 3]);
    }

    #[test]
    fn zero_weight_block_runs_all_stages_and_finishes_with_layer_norm() {
        let width = 32;
        let matrix = |rows: usize| {
            let mut bytes = Vec::with_capacity(rows * 34);
            for _ in 0..rows {
                bytes.extend_from_slice(&0u16.to_le_bytes());
                bytes.extend_from_slice(&[0; 32]);
            }
            bytes
        };
        let matrix32 = matrix(width);
        let matrix64 = matrix(width * 2);
        let affine = |value: f32| {
            (0..width)
                .flat_map(|_| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let weight = affine(1.0);
        let bias = affine(0.0);
        let norm = LayerNormPlan::new(width, 1e-5, &weight, &bias).unwrap();
        let q32 = Q8Matrix::new(&matrix32, width, width).unwrap();
        let q64 = Q8Matrix::new(&matrix64, width, width * 2).unwrap();
        let feed_forward = FeedForwardPlan {
            norm,
            expand: q32,
            contract: q32,
        };
        let positions = vec![0; 9 * width * 4];
        let attention = RelativeAttentionPlan {
            norm,
            query: q32,
            key: q32,
            value: q32,
            position: q32,
            output: q32,
            bias_u_f32_le: &bias,
            bias_v_f32_le: &bias,
            position_f32_le: &positions,
            position_count: 9,
            position_center: 4,
            heads: 4,
            mask: AttentionMask::Full,
        };
        let depthwise = vec![0; width * 3 * 2];
        let convolution = ConvolutionPlan {
            norm,
            pointwise_in: q64,
            depthwise_f16_le: &depthwise,
            kernel: 3,
            channel_norm: norm,
            pointwise_out: q32,
        };
        let plan = ConformerBlockPlan::new(
            width,
            feed_forward,
            attention,
            convolution,
            feed_forward,
            norm,
        )
        .unwrap();
        let input: Vec<_> = (0..width * 2).map(|index| index as f32 / 7.0).collect();
        let output = CpuConformerBlock::new(plan).run(&input, 2).unwrap();
        let mut expected = vec![0.0; input.len()];
        norm.run(&input, &mut expected).unwrap();
        assert_eq!(output, expected);
    }
}
