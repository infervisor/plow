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

use crate::asr::frontend::LogMelFeatures;
use crate::asr::subsampling::{
    Activation, Conv2dSpec, ConvKind, FeatureShape, Padding, SubsamplingPlan,
};
use crate::{Result, RuntimeError};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

struct ConvBuffers {
    weight: Buffer,
    bias: Buffer,
}

pub struct MetalSubsampler {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    conv_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pointwise_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    specs: Vec<Conv2dSpec>,
    layers: Vec<ConvBuffers>,
    shapes: Vec<FeatureShape>,
    input: Buffer,
    scratch: [Buffer; 2],
}

impl MetalSubsampler {
    pub fn new(plan: &SubsamplingPlan<'_>, frames: usize) -> Result<Self> {
        let shapes = plan.shapes(frames)?;
        for (index, stage) in plan.stages().iter().enumerate() {
            let supported = match stage.spec {
                Conv2dSpec {
                    kernel: 3,
                    stride: 2,
                    padding:
                        Padding {
                            before: 2,
                            after: 1,
                        },
                    kind: ConvKind::Standard | ConvKind::Depthwise,
                    ..
                } => true,
                Conv2dSpec {
                    kernel: 1,
                    stride: 1,
                    padding:
                        Padding {
                            before: 0,
                            after: 0,
                        },
                    kind: ConvKind::Standard,
                    ..
                } => true,
                _ => false,
            };
            if !supported {
                return Err(rejected(format!("unsupported stage {index}")));
            }
            for value in [
                stage.spec.input_channels,
                stage.spec.output_channels,
                shapes[index].frames,
                shapes[index].width,
            ] {
                if u32::try_from(value).is_err() {
                    return Err(rejected(format!("stage {index} geometry exceeds u32")));
                }
            }
        }

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| RuntimeError::Device("Metal device unavailable".into()))?;
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!(
                    "../../../../../runtime/apple/asr_subsampling.metal"
                )),
                Some(&options),
            )
            .map_err(|error| RuntimeError::Device(format!("compile ASR Metal library: {error}")))?;
        let pipeline = |name: &str| -> Result<_> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| RuntimeError::Device(format!("Metal function {name} missing")))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| RuntimeError::Device(format!("create {name} pipeline: {error}")))
        };
        let conv_pipeline = pipeline("asr_causal_conv3_f16")?;
        let pointwise_pipeline = pipeline("asr_pointwise_f16")?;
        if pointwise_pipeline.threadExecutionWidth() != 32
            || pointwise_pipeline.maxTotalThreadsPerThreadgroup() < 256
        {
            return Err(RuntimeError::Device(
                "ASR pointwise kernel requires 32-wide SIMD and 256 threads".into(),
            ));
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| RuntimeError::Device("create Metal queue".into()))?;
        let layers = plan
            .stages()
            .iter()
            .map(|stage| {
                Ok(ConvBuffers {
                    weight: copy_buffer(&device, stage.weights_f16_le, "ASR convolution weight")?,
                    bias: copy_buffer(&device, stage.bias_f32_le, "ASR convolution bias")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let input_elements = checked_elements(shapes[0])?;
        let scratch_elements = shapes
            .iter()
            .skip(1)
            .copied()
            .map(checked_elements)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .ok_or_else(|| rejected("missing output shape"))?;
        Ok(Self {
            queue,
            conv_pipeline,
            pointwise_pipeline,
            specs: plan.stages().iter().map(|stage| stage.spec).collect(),
            layers,
            shapes,
            input: allocate(&device, input_elements, "ASR feature input")?,
            scratch: [
                allocate(&device, scratch_elements, "ASR subsampling scratch")?,
                allocate(&device, scratch_elements, "ASR subsampling scratch")?,
            ],
        })
    }

    pub fn output_shape(&self) -> [usize; 3] {
        let shape = self.shapes[self.shapes.len() - 1];
        [shape.frames, shape.width, shape.channels]
    }

    pub fn run(&mut self, input: &LogMelFeatures, output: &mut [f32]) -> Result<f64> {
        let input_shape = self.shapes[0];
        let output_shape = self.shapes[self.shapes.len() - 1];
        if input.frames != input_shape.frames
            || input.bins != input_shape.width
            || input.values.len() != checked_elements(input_shape)?
            || output.len() != checked_elements(output_shape)?
        {
            return Err(rejected("run shape mismatch"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.values.as_ptr(),
                self.input.contents().as_ptr().cast::<f32>(),
                input.values.len(),
            );
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| RuntimeError::Device("create Metal command buffer".into()))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| RuntimeError::Device("create Metal compute encoder".into()))?;

        for index in 0..self.specs.len() {
            let source = if index == 0 {
                &self.input
            } else {
                &self.scratch[(index - 1) % 2]
            };
            let destination = &self.scratch[index % 2];
            let spec = self.specs[index];
            if spec.kernel == 3 {
                self.dispatch_conv(
                    &encoder,
                    &self.layers[index],
                    source,
                    destination,
                    self.shapes[index],
                    self.shapes[index + 1],
                    spec,
                );
            } else {
                self.dispatch_pointwise(
                    &encoder,
                    &self.layers[index],
                    source,
                    destination,
                    self.shapes[index + 1],
                    spec,
                );
            }
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(RuntimeError::Device(format!(
                "ASR Metal subsampling failed: {:?}",
                command.error()
            )));
        }
        let final_buffer = &self.scratch[(self.specs.len() - 1) % 2];
        let source = final_buffer.contents().as_ptr().cast::<f32>();
        for frame in 0..output_shape.frames {
            for channel in 0..output_shape.channels {
                for column in 0..output_shape.width {
                    let source_index =
                        (frame * output_shape.width + column) * output_shape.channels + channel;
                    let output_index =
                        (frame * output_shape.channels + channel) * output_shape.width + column;
                    // SAFETY: both indices are bounded by the validated output element count.
                    output[output_index] = unsafe { *source.add(source_index) };
                }
            }
        }
        Ok((command.GPUEndTime() - command.GPUStartTime()) * 1e6)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_conv(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        layer: &ConvBuffers,
        input: &Buffer,
        output: &Buffer,
        input_shape: FeatureShape,
        output_shape: FeatureShape,
        spec: Conv2dSpec,
    ) {
        encoder.setComputePipelineState(&self.conv_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&layer.weight), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&layer.bias), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
            let shape = [
                input_shape.frames as u32,
                input_shape.width as u32,
                spec.input_channels as u32,
                spec.output_channels as u32,
                (spec.kind == ConvKind::Depthwise) as u32,
                (spec.activation == Activation::Relu) as u32,
            ];
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), 24, 4);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: output_shape.frames * output_shape.width * output_shape.channels,
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

    fn dispatch_pointwise(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        layer: &ConvBuffers,
        input: &Buffer,
        output: &Buffer,
        output_shape: FeatureShape,
        spec: Conv2dSpec,
    ) {
        encoder.setComputePipelineState(&self.pointwise_pipeline);
        let rows = output_shape.frames * output_shape.width;
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&layer.weight), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&layer.bias), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
            let shape = [
                rows as u32,
                spec.output_channels as u32,
                spec.input_channels as u32,
                (spec.activation == Activation::Relu) as u32,
            ];
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), 16, 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows.div_ceil(32) * spec.output_channels.div_ceil(64),
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
}

fn checked_elements(shape: FeatureShape) -> Result<usize> {
    shape
        .frames
        .checked_mul(shape.width)
        .and_then(|count| count.checked_mul(shape.channels))
        .ok_or_else(|| rejected("feature buffer size overflows"))
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

fn allocate(device: &ProtocolObject<dyn MTLDevice>, count: usize, label: &str) -> Result<Buffer> {
    let bytes = count
        .checked_mul(4)
        .ok_or_else(|| rejected(format!("{label} size overflows")))?;
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| RuntimeError::Oom(label.into()))
}

fn rejected(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("invalid Metal ASR subsampler: {}", message.into()))
}
