use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode,
    MTLResourceOptions, MTLSize,
};

use crate::asr::conformer::{AttentionMask, ConformerBlockPlan, ConformerEncoderPlan};
use crate::ops::linear::Q8Matrix;
use crate::ops::norm::LayerNormPlan;
use crate::{Result, RuntimeError};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

struct Q8Weight {
    buffer: Buffer,
    k: usize,
    n: usize,
}

struct NormWeight {
    weight: Buffer,
    bias: Buffer,
}

struct BlockWeights {
    norm_ff1: NormWeight,
    ff1_expand: Q8Weight,
    ff1_contract: Q8Weight,
    norm_attention: NormWeight,
    query: Q8Weight,
    key: Q8Weight,
    value: Q8Weight,
    position: Q8Weight,
    attention_out: Q8Weight,
    bias_u: Buffer,
    bias_v: Buffer,
    norm_conv: NormWeight,
    pointwise_in: Q8Weight,
    depthwise: Buffer,
    channel_norm: NormWeight,
    pointwise_out: Q8Weight,
    norm_ff2: NormWeight,
    ff2_expand: Q8Weight,
    ff2_contract: Q8Weight,
    norm_out: NormWeight,
}

struct Pipelines {
    q8_32: Pipeline,
    q8_64: Pipeline,
    norm: Pipeline,
    silu: Pipeline,
    scaled_add: Pipeline,
    glu: Pipeline,
    depthwise: Pipeline,
    attention: Pipeline,
}

pub struct MetalConformerBlock {
    inner: MetalConformerExecutor,
}

pub struct MetalConformerEncoder {
    inner: MetalConformerExecutor,
}

struct MetalConformerExecutor {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: Pipelines,
    weights: Vec<BlockWeights>,
    frames: usize,
    width: usize,
    heads: usize,
    chunk_size: usize,
    left_chunks: usize,
    kernel: usize,
    residual: Buffer,
    d0: Buffer,
    d1: Buffer,
    query: Buffer,
    key: Buffer,
    value: Buffer,
    large: Buffer,
    position_input: Buffer,
    position_output: Buffer,
}

impl MetalConformerBlock {
    pub fn new(plan: &ConformerBlockPlan<'_>, frames: usize) -> Result<Self> {
        Ok(Self {
            inner: MetalConformerExecutor::new(std::slice::from_ref(plan), frames)?,
        })
    }

    pub fn run(&mut self, input: &[f32], output: &mut [f32]) -> Result<f64> {
        self.inner.run(input, output)
    }
}

impl MetalConformerEncoder {
    pub fn new(plan: &ConformerEncoderPlan<'_>, frames: usize) -> Result<Self> {
        Ok(Self {
            inner: MetalConformerExecutor::new(plan.blocks(), frames)?,
        })
    }

    pub fn run(&mut self, input: &[f32], output: &mut [f32]) -> Result<f64> {
        self.inner.run(input, output)
    }
}

impl MetalConformerExecutor {
    fn new(plans: &[ConformerBlockPlan<'_>], frames: usize) -> Result<Self> {
        if frames == 0 {
            return Err(rejected("frame count is zero"));
        }
        let plan = plans
            .first()
            .ok_or_else(|| rejected("encoder has no blocks"))?;
        let width = plan.width();
        let AttentionMask::ChunkedLimited {
            chunk_size,
            left_chunks,
        } = plan.attention.mask
        else {
            return Err(rejected("only chunk-limited attention is supported"));
        };
        let attention_window = left_chunks
            .checked_add(1)
            .and_then(|chunks| chunks.checked_mul(chunk_size))
            .ok_or_else(|| rejected("attention window overflows"))?;
        if attention_window > 64 {
            return Err(rejected("Metal fused attention window exceeds 64 frames"));
        }
        if plans.iter().any(|block| {
            block.width() != width
                || block.attention.heads != plan.attention.heads
                || block.attention.mask != plan.attention.mask
                || block.attention.position_count != plan.attention.position_count
                || block.attention.position_center != plan.attention.position_center
                || block.attention.position_f32_le != plan.attention.position_f32_le
                || block.convolution.kernel != plan.convolution.kernel
        }) {
            return Err(rejected("encoder blocks have incompatible geometry"));
        }
        for value in [frames, width, plan.attention.heads, chunk_size, left_chunks] {
            require_u32(value, "block geometry")?;
        }
        let relative_count = frames
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| rejected("relative position extent overflows"))?;
        require_u32(relative_count, "relative position extent")?;
        if frames > plan.attention.position_center {
            return Err(rejected("frame count exceeds position center"));
        }
        let position_start = plan.attention.position_center - frames;
        let position_byte_start = position_start
            .checked_mul(width)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| rejected("position offset overflows"))?;
        let position_bytes = relative_count
            .checked_mul(width)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| rejected("position size overflows"))?;
        let position_byte_end = position_byte_start
            .checked_add(position_bytes)
            .ok_or_else(|| rejected("position range overflows"))?;
        let position_slice = plan
            .attention
            .position_f32_le
            .get(position_byte_start..position_byte_end)
            .ok_or_else(|| rejected("position range exceeds table"))?;

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| RuntimeError::Device("Metal device unavailable".into()))?;
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        let source = format!(
            "{}\n{}",
            include_str!("../../../../../runtime/apple/q8.metal"),
            include_str!("../../../../../runtime/apple/asr_conformer.metal")
        );
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .map_err(|error| RuntimeError::Device(format!("compile Conformer Metal: {error}")))?;
        let pipelines = Pipelines {
            q8_32: pipeline(&device, &library, "q8_0_gemm")?,
            q8_64: pipeline(&device, &library, "q8_0_gemm64")?,
            norm: pipeline(&device, &library, "conformer_layer_norm")?,
            silu: pipeline(&device, &library, "conformer_silu")?,
            scaled_add: pipeline(&device, &library, "conformer_scaled_add")?,
            glu: pipeline(&device, &library, "conformer_glu")?,
            depthwise: pipeline(&device, &library, "conformer_depthwise_causal")?,
            attention: pipeline(&device, &library, "conformer_relative_attention_fused")?,
        };
        if pipelines.q8_32.maxTotalThreadsPerThreadgroup() < 256
            || pipelines.q8_64.maxTotalThreadsPerThreadgroup() < 256
            || pipelines.norm.threadExecutionWidth() != 32
            || pipelines.attention.threadExecutionWidth() != 32
            || pipelines.attention.maxTotalThreadsPerThreadgroup() < 128
        {
            return Err(RuntimeError::Device(
                "Metal device does not support Conformer launch geometry".into(),
            ));
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| RuntimeError::Device("create Metal queue".into()))?;
        let weights = plans
            .iter()
            .map(|plan| BlockWeights::new(&device, plan))
            .collect::<Result<Vec<_>>>()?;
        let elements = frames
            .checked_mul(width)
            .ok_or_else(|| rejected("block buffer size overflows"))?;
        let doubled_width = width
            .checked_mul(2)
            .ok_or_else(|| rejected("block width overflows"))?;
        let large_width = plans.iter().fold(doubled_width, |maximum, plan| {
            maximum
                .max(plan.feed_forward1.expand.n())
                .max(plan.feed_forward2.expand.n())
        });
        let large_elements = frames
            .checked_mul(large_width)
            .ok_or_else(|| rejected("large workspace size overflows"))?;
        let position_elements = relative_count
            .checked_mul(width)
            .ok_or_else(|| rejected("position workspace size overflows"))?;
        for (value, label) in [
            (elements, "block element count"),
            (large_elements, "large workspace element count"),
            (position_elements, "position element count"),
        ] {
            require_u32(value, label)?;
        }
        Ok(Self {
            queue,
            pipelines,
            weights,
            frames,
            width,
            heads: plan.attention.heads,
            chunk_size,
            left_chunks,
            kernel: plan.convolution.kernel,
            residual: allocate_f32(&device, elements, "Conformer residual")?,
            d0: allocate_f32(&device, elements, "Conformer D scratch")?,
            d1: allocate_f32(&device, elements, "Conformer D scratch")?,
            query: allocate_f32(&device, elements, "Conformer query")?,
            key: allocate_f32(&device, elements, "Conformer key")?,
            value: allocate_f32(&device, elements, "Conformer value")?,
            large: allocate_f32(&device, large_elements, "Conformer large scratch")?,
            position_input: copy_buffer(&device, position_slice, "Conformer positions")?,
            position_output: allocate_f32(
                &device,
                position_elements,
                "Conformer projected positions",
            )?,
        })
    }

    pub fn run(&mut self, input: &[f32], output: &mut [f32]) -> Result<f64> {
        let elements = self
            .frames
            .checked_mul(self.width)
            .ok_or_else(|| rejected("block shape overflows"))?;
        if input.len() != elements || output.len() != elements {
            return Err(rejected("run shape mismatch"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.residual.contents().as_ptr().cast::<f32>(),
                input.len(),
            );
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| RuntimeError::Device("create Metal command buffer".into()))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| RuntimeError::Device("create Metal compute encoder".into()))?;
        for weights in &self.weights {
            self.encode_block(&encoder, weights, elements);
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(RuntimeError::Device(format!(
                "Conformer Metal dispatch failed: {:?}",
                command.error()
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.residual.contents().as_ptr().cast::<f32>(),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok((command.GPUEndTime() - command.GPUStartTime()) * 1e6)
    }

    fn encode_block(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &BlockWeights,
        elements: usize,
    ) {
        self.encode_feed_forward(
            encoder,
            &weights.norm_ff1,
            &weights.ff1_expand,
            &weights.ff1_contract,
        );
        self.dispatch_norm(encoder, &self.residual, &weights.norm_attention, &self.d0);
        self.dispatch_q8(encoder, &self.d0, &weights.query, self.frames, &self.query);
        self.dispatch_q8(encoder, &self.d0, &weights.key, self.frames, &self.key);
        self.dispatch_q8(encoder, &self.d0, &weights.value, self.frames, &self.value);
        self.dispatch_q8(
            encoder,
            &self.position_input,
            &weights.position,
            self.frames * 2 - 1,
            &self.position_output,
        );
        self.dispatch_attention(encoder, weights);
        self.dispatch_q8(
            encoder,
            &self.d1,
            &weights.attention_out,
            self.frames,
            &self.d0,
        );
        self.dispatch_scaled_add(encoder, &self.residual, &self.d0, &self.residual, 1.0);

        self.dispatch_norm(encoder, &self.residual, &weights.norm_conv, &self.d0);
        self.dispatch_q8(
            encoder,
            &self.d0,
            &weights.pointwise_in,
            self.frames,
            &self.large,
        );
        self.dispatch_glu(encoder);
        self.dispatch_depthwise(encoder, weights);
        self.dispatch_norm(encoder, &self.d0, &weights.channel_norm, &self.d1);
        self.dispatch_silu(encoder, &self.d1, &self.d1, elements);
        self.dispatch_q8(
            encoder,
            &self.d1,
            &weights.pointwise_out,
            self.frames,
            &self.d0,
        );
        self.dispatch_scaled_add(encoder, &self.residual, &self.d0, &self.residual, 1.0);
        self.encode_feed_forward(
            encoder,
            &weights.norm_ff2,
            &weights.ff2_expand,
            &weights.ff2_contract,
        );
        self.dispatch_norm(encoder, &self.residual, &weights.norm_out, &self.residual);
    }

    fn encode_feed_forward(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        norm: &NormWeight,
        expand: &Q8Weight,
        contract: &Q8Weight,
    ) {
        self.dispatch_norm(encoder, &self.residual, norm, &self.d0);
        self.dispatch_q8(encoder, &self.d0, expand, self.frames, &self.large);
        self.dispatch_silu(encoder, &self.large, &self.large, self.frames * expand.n);
        self.dispatch_q8(encoder, &self.large, contract, self.frames, &self.d1);
        self.dispatch_scaled_add(encoder, &self.residual, &self.d1, &self.residual, 0.5);
    }

    fn dispatch_q8(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Buffer,
        weight: &Q8Weight,
        rows: usize,
        output: &Buffer,
    ) {
        let tile_rows = select_tile_rows(rows, weight.n);
        encoder.setComputePipelineState(if tile_rows == 64 {
            &self.pipelines.q8_64
        } else {
            &self.pipelines.q8_32
        });
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&weight.buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
            let shape = [rows as u32, weight.n as u32, weight.k as u32, 0u32];
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), 16, 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows.div_ceil(tile_rows) * weight.n.div_ceil(64),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }

    fn dispatch_norm(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Buffer,
        norm: &NormWeight,
        output: &Buffer,
    ) {
        encoder.setComputePipelineState(&self.pipelines.norm);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&norm.weight), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&norm.bias), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
            let shape = [self.frames as u32, self.width as u32];
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), 8, 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.frames,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    fn dispatch_silu(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Buffer,
        output: &Buffer,
        count: usize,
    ) {
        encoder.setComputePipelineState(&self.pipelines.silu);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
            let count = count as u32;
            encoder.setBytes_length_atIndex(NonNull::from(&count).cast(), 4, 2);
        }
        dispatch_elements(encoder, count, 256);
    }

    fn dispatch_scaled_add(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Buffer,
        other: &Buffer,
        output: &Buffer,
        scale: f32,
    ) {
        let count = self.frames * self.width;
        encoder.setComputePipelineState(&self.pipelines.scaled_add);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(other), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
            let count = count as u32;
            encoder.setBytes_length_atIndex(NonNull::from(&count).cast(), 4, 3);
            encoder.setBytes_length_atIndex(NonNull::from(&scale).cast(), 4, 4);
        }
        dispatch_elements(encoder, count, 256);
    }

    fn dispatch_glu(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        encoder.setComputePipelineState(&self.pipelines.glu);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.large), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.d1), 0, 1);
            let shape = [self.frames as u32, self.width as u32];
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), 8, 2);
        }
        dispatch_elements(encoder, self.frames * self.width, 256);
    }

    fn dispatch_depthwise(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &BlockWeights,
    ) {
        encoder.setComputePipelineState(&self.pipelines.depthwise);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.d1), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&weights.depthwise), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.d0), 0, 2);
            let shape = [self.frames as u32, self.width as u32, self.kernel as u32];
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), 12, 3);
        }
        dispatch_elements(encoder, self.frames * self.width, 256);
    }

    fn dispatch_attention(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &BlockWeights,
    ) {
        encoder.setComputePipelineState(&self.pipelines.attention);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.query), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.key), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.value), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.position_output), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&weights.bias_u), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&weights.bias_v), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&self.d1), 0, 6);
            let shape = [
                self.frames as u32,
                self.width as u32,
                self.heads as u32,
                self.chunk_size as u32,
                self.left_chunks as u32,
            ];
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), 20, 7);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.frames * self.heads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
}

impl BlockWeights {
    fn new(device: &ProtocolObject<dyn MTLDevice>, plan: &ConformerBlockPlan<'_>) -> Result<Self> {
        Ok(Self {
            norm_ff1: NormWeight::new(device, plan.feed_forward1.norm)?,
            ff1_expand: Q8Weight::new(device, plan.feed_forward1.expand)?,
            ff1_contract: Q8Weight::new(device, plan.feed_forward1.contract)?,
            norm_attention: NormWeight::new(device, plan.attention.norm)?,
            query: Q8Weight::new(device, plan.attention.query)?,
            key: Q8Weight::new(device, plan.attention.key)?,
            value: Q8Weight::new(device, plan.attention.value)?,
            position: Q8Weight::new(device, plan.attention.position)?,
            attention_out: Q8Weight::new(device, plan.attention.output)?,
            bias_u: copy_buffer(device, plan.attention.bias_u_f32_le, "attention bias u")?,
            bias_v: copy_buffer(device, plan.attention.bias_v_f32_le, "attention bias v")?,
            norm_conv: NormWeight::new(device, plan.convolution.norm)?,
            pointwise_in: Q8Weight::new(device, plan.convolution.pointwise_in)?,
            depthwise: copy_buffer(
                device,
                plan.convolution.depthwise_f16_le,
                "depthwise weight",
            )?,
            channel_norm: NormWeight::new(device, plan.convolution.channel_norm)?,
            pointwise_out: Q8Weight::new(device, plan.convolution.pointwise_out)?,
            norm_ff2: NormWeight::new(device, plan.feed_forward2.norm)?,
            ff2_expand: Q8Weight::new(device, plan.feed_forward2.expand)?,
            ff2_contract: Q8Weight::new(device, plan.feed_forward2.contract)?,
            norm_out: NormWeight::new(device, plan.output_norm)?,
        })
    }
}

impl Q8Weight {
    fn new(device: &ProtocolObject<dyn MTLDevice>, matrix: Q8Matrix<'_>) -> Result<Self> {
        require_u32(matrix.k(), "Q8 K")?;
        require_u32(matrix.n(), "Q8 N")?;
        Ok(Self {
            buffer: copy_buffer(device, matrix.bytes(), "Q8 weight")?,
            k: matrix.k(),
            n: matrix.n(),
        })
    }
}

impl NormWeight {
    fn new(device: &ProtocolObject<dyn MTLDevice>, plan: LayerNormPlan<'_>) -> Result<Self> {
        Ok(Self {
            weight: copy_buffer(device, plan.weight_bytes(), "normalization weight")?,
            bias: copy_buffer(device, plan.bias_bytes(), "normalization bias")?,
        })
    }
}

fn pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Pipeline> {
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| RuntimeError::Device(format!("Metal function {name} missing")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| RuntimeError::Device(format!("create {name} pipeline: {error}")))
}

fn select_tile_rows(rows: usize, columns: usize) -> usize {
    if rows >= 192 || columns >= 2048 {
        64
    } else {
        32
    }
}

fn dispatch_elements(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    count: usize,
    threads: usize,
) {
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

fn copy_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    label: &str,
) -> Result<Buffer> {
    unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(bytes.as_ptr().cast_mut().cast()).unwrap(),
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| RuntimeError::Oom(label.into()))
}

fn allocate_f32(
    device: &ProtocolObject<dyn MTLDevice>,
    elements: usize,
    label: &str,
) -> Result<Buffer> {
    let bytes = elements
        .checked_mul(4)
        .ok_or_else(|| rejected(format!("{label} size overflows")))?;
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| RuntimeError::Oom(label.into()))
}

fn require_u32(value: usize, label: &str) -> Result<()> {
    u32::try_from(value)
        .map(|_| ())
        .map_err(|_| rejected(format!("{label} exceeds u32")))
}

fn rejected(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("invalid Metal Conformer block: {}", message.into()))
}
