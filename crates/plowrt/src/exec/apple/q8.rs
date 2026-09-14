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

use crate::ops::linear::{Q8LinearPlan, Q8Matrix};
use crate::{Result, RuntimeError};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct Q8Gemv {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    weight: Buffer,
    input: Buffer,
    output: Buffer,
    n: usize,
    k: usize,
    simdgroups: usize,
    rows_per_simd: usize,
}

impl Q8Gemv {
    pub fn new(matrix: Q8Matrix<'_>) -> Result<Self> {
        Self::with_layout(matrix, 2, 4)
    }

    pub fn with_simdgroups(matrix: Q8Matrix<'_>, simdgroups: usize) -> Result<Self> {
        Self::with_layout(matrix, simdgroups, 1)
    }

    pub fn with_layout(
        matrix: Q8Matrix<'_>,
        simdgroups: usize,
        rows_per_simd: usize,
    ) -> Result<Self> {
        let (k, n) = (matrix.k(), matrix.n());
        require_device_shape(k, n)?;
        if !matches!(simdgroups, 1 | 2 | 4 | 8) {
            return Err(RuntimeError::Rejected(
                "Q8_0 Metal SIMD-group count must be 1, 2, 4 or 8".into(),
            ));
        }
        if !matches!(rows_per_simd, 1 | 4) {
            return Err(RuntimeError::Rejected(
                "Q8_0 Metal rows per SIMD must be 1 or 4".into(),
            ));
        }
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| RuntimeError::Device("Metal device unavailable".into()))?;
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("../../../../../runtime/apple/q8.metal")),
                Some(&options),
            )
            .map_err(|error| RuntimeError::Device(format!("compile Q8 Metal library: {error}")))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(if rows_per_simd == 1 {
                "q8_0_gemv"
            } else {
                "q8_0_gemv4"
            }))
            .ok_or_else(|| RuntimeError::Device("Q8_0 Metal function missing".into()))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| RuntimeError::Device(format!("create Q8_0 pipeline: {error}")))?;
        if pipeline.threadExecutionWidth() != 32 || pipeline.maxTotalThreadsPerThreadgroup() < 256 {
            return Err(RuntimeError::Device(
                "Q8_0 kernel requires 32-wide SIMD and 256 threads".into(),
            ));
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| RuntimeError::Device("create Metal queue".into()))?;
        let weight = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(matrix.bytes().as_ptr().cast_mut().cast()).unwrap(),
                matrix.bytes().len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| RuntimeError::Oom("Q8_0 weight buffer".into()))?;
        let input_bytes = k
            .checked_mul(4)
            .ok_or_else(|| RuntimeError::Rejected("Q8_0 input size overflows".into()))?;
        let output_bytes = n
            .checked_mul(4)
            .ok_or_else(|| RuntimeError::Rejected("Q8_0 output size overflows".into()))?;
        let input = device
            .newBufferWithLength_options(input_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| RuntimeError::Oom("Q8_0 input buffer".into()))?;
        let output = device
            .newBufferWithLength_options(output_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| RuntimeError::Oom("Q8_0 output buffer".into()))?;
        Ok(Self {
            queue,
            pipeline,
            weight,
            input,
            output,
            n,
            k,
            simdgroups,
            rows_per_simd,
        })
    }

    pub fn run(&mut self, input: &[f32], output: &mut [f32]) -> Result<f64> {
        if input.len() != self.k || output.len() != self.n {
            return Err(RuntimeError::Rejected(
                "Metal Q8_0 GEMV shape mismatch".into(),
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input.contents().as_ptr().cast::<f32>(),
                self.k,
            );
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| RuntimeError::Device("create Metal command buffer".into()))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| RuntimeError::Device("create Metal compute encoder".into()))?;
        encoder.setComputePipelineState(&self.pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.weight), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.output), 0, 2);
            let shape = [
                u32::try_from(self.n).unwrap(),
                u32::try_from(self.k).unwrap(),
                u32::try_from(self.simdgroups).unwrap(),
            ];
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), 12, 3);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.n.div_ceil(self.simdgroups * self.rows_per_simd),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.simdgroups * 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(RuntimeError::Device(format!(
                "Q8_0 Metal dispatch failed: {:?}",
                command.error()
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.output.contents().as_ptr().cast::<f32>(),
                output.as_mut_ptr(),
                self.n,
            );
        }
        Ok((command.GPUEndTime() - command.GPUStartTime()) * 1e6)
    }
}

pub struct Q8Gemm {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    weight: Buffer,
    bias: Option<Buffer>,
    input: Buffer,
    output: Buffer,
    m: usize,
    n: usize,
    k: usize,
    tile_rows: usize,
}

impl Q8Gemm {
    pub fn new(matrix: Q8Matrix<'_>, m: usize) -> Result<Self> {
        let tile_rows = select_tile_rows(m, matrix.n());
        Self::build(matrix, None, m, tile_rows)
    }

    pub fn with_tile_rows(matrix: Q8Matrix<'_>, m: usize, tile_rows: usize) -> Result<Self> {
        Self::build(matrix, None, m, tile_rows)
    }

    pub fn linear(plan: &Q8LinearPlan<'_>, m: usize) -> Result<Self> {
        let matrix = plan.weight();
        Self::build(
            matrix,
            Some(plan.bias_bytes()),
            m,
            select_tile_rows(m, matrix.n()),
        )
    }

    pub fn linear_with_tile_rows(
        plan: &Q8LinearPlan<'_>,
        m: usize,
        tile_rows: usize,
    ) -> Result<Self> {
        Self::build(plan.weight(), Some(plan.bias_bytes()), m, tile_rows)
    }

    fn build(
        matrix: Q8Matrix<'_>,
        bias: Option<&[u8]>,
        m: usize,
        tile_rows: usize,
    ) -> Result<Self> {
        let (k, n) = (matrix.k(), matrix.n());
        require_device_shape(k, n)?;
        if m == 0 || u32::try_from(m).is_err() {
            return Err(RuntimeError::Rejected(format!(
                "invalid Metal Q8_0 GEMM M={m}"
            )));
        }
        if !matches!(tile_rows, 32 | 64) {
            return Err(RuntimeError::Rejected(
                "Metal Q8_0 GEMM tile rows must be 32 or 64".into(),
            ));
        }
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| RuntimeError::Device("Metal device unavailable".into()))?;
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("../../../../../runtime/apple/q8.metal")),
                Some(&options),
            )
            .map_err(|error| RuntimeError::Device(format!("compile Q8 Metal library: {error}")))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(if tile_rows == 64 {
                "q8_0_gemm64"
            } else {
                "q8_0_gemm"
            }))
            .ok_or_else(|| RuntimeError::Device("Q8_0 GEMM Metal function missing".into()))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| RuntimeError::Device(format!("create Q8_0 GEMM pipeline: {error}")))?;
        if pipeline.threadExecutionWidth() != 32 || pipeline.maxTotalThreadsPerThreadgroup() < 256 {
            return Err(RuntimeError::Device(
                "Q8_0 GEMM kernel requires 32-wide SIMD and 256 threads".into(),
            ));
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| RuntimeError::Device("create Metal queue".into()))?;
        let weight = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(matrix.bytes().as_ptr().cast_mut().cast()).unwrap(),
                matrix.bytes().len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| RuntimeError::Oom("Q8_0 weight buffer".into()))?;
        let bias = match bias {
            Some(bytes) => Some(
                unsafe {
                    device.newBufferWithBytes_length_options(
                        NonNull::new(bytes.as_ptr().cast_mut().cast()).unwrap(),
                        bytes.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| RuntimeError::Oom("Q8_0 bias buffer".into()))?,
            ),
            None => None,
        };
        let input_bytes = m
            .checked_mul(k)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| RuntimeError::Rejected("Q8_0 GEMM input size overflows".into()))?;
        let output_bytes = m
            .checked_mul(n)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| RuntimeError::Rejected("Q8_0 GEMM output size overflows".into()))?;
        let input = device
            .newBufferWithLength_options(input_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| RuntimeError::Oom("Q8_0 GEMM input buffer".into()))?;
        let output = device
            .newBufferWithLength_options(output_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| RuntimeError::Oom("Q8_0 GEMM output buffer".into()))?;
        Ok(Self {
            queue,
            pipeline,
            weight,
            bias,
            input,
            output,
            m,
            n,
            k,
            tile_rows,
        })
    }

    pub fn run(&mut self, input: &[f32], output: &mut [f32]) -> Result<f64> {
        if input.len() != self.m * self.k || output.len() != self.m * self.n {
            return Err(RuntimeError::Rejected(
                "Metal Q8_0 GEMM shape mismatch".into(),
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input.contents().as_ptr().cast::<f32>(),
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
        encoder.setComputePipelineState(&self.pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.weight), 0, 1);
            encoder.setBuffer_offset_atIndex(self.bias.as_deref().or(Some(&self.output)), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.output), 0, 3);
            let shape = [
                u32::try_from(self.m).unwrap(),
                u32::try_from(self.n).unwrap(),
                u32::try_from(self.k).unwrap(),
                self.bias.is_some() as u32,
            ];
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), 16, 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.m.div_ceil(self.tile_rows) * self.n.div_ceil(64),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(RuntimeError::Device(format!(
                "Q8_0 GEMM Metal dispatch failed: {:?}",
                command.error()
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.output.contents().as_ptr().cast::<f32>(),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok((command.GPUEndTime() - command.GPUStartTime()) * 1e6)
    }
}

fn select_tile_rows(m: usize, n: usize) -> usize {
    if m >= 192 || n >= 2048 {
        64
    } else {
        32
    }
}

fn require_device_shape(k: usize, n: usize) -> Result<()> {
    if u32::try_from(k).is_err() || u32::try_from(n).is_err() {
        return Err(RuntimeError::Rejected(format!(
            "invalid Metal Q8_0 matrix [{k}, {n}]"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Q8Gemm, Q8Gemv};
    use crate::ops::linear::{Q8LinearPlan, Q8Matrix};

    #[test]
    fn q8_0_tail_rows_match_scalar_oracle() {
        let mut bytes = Vec::new();
        for row in 0..3 {
            bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
            bytes.extend((0..32).map(|column| (row * 7 + column) as i8 as u8));
        }
        let matrix = Q8Matrix::new(&bytes, 32, 3).unwrap();
        let input: Vec<_> = (0..32).map(|index| index as f32 / 32.0).collect();
        let mut expected = [0.0; 3];
        matrix.matmul(&input, 1, &mut expected).unwrap();
        let mut actual = [0.0; 3];
        Q8Gemv::with_layout(matrix, 8, 4)
            .unwrap()
            .run(&input, &mut actual)
            .unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
        }
    }

    #[test]
    fn q8_0_gemm_tail_matches_scalar_oracle() {
        let mut bytes = Vec::new();
        for row in 0..3 {
            bytes.extend_from_slice(&0x3800u16.to_le_bytes());
            bytes.extend((0..32).map(|column| (row * 7 + column) as i8 as u8));
        }
        let matrix = Q8Matrix::new(&bytes, 32, 3).unwrap();
        let input: Vec<_> = (0..33 * 32)
            .map(|index| (index % 47) as f32 / 47.0 - 0.5)
            .collect();
        let mut expected = vec![0.0; 33 * 3];
        matrix.matmul(&input, 33, &mut expected).unwrap();
        for tile_rows in [32, 64] {
            let mut actual = vec![0.0; 33 * 3];
            Q8Gemm::with_tile_rows(matrix, 33, tile_rows)
                .unwrap()
                .run(&input, &mut actual)
                .unwrap();
            for (actual, expected) in actual.into_iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-3, "{actual} vs {expected}");
            }
        }

        let bias: Vec<_> = [0.25f32, -1.5, 3.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let linear = Q8LinearPlan::new(matrix, &bias).unwrap();
        linear.matmul(&input, 33, &mut expected).unwrap();
        for tile_rows in [32, 64] {
            let mut actual = vec![0.0; 33 * 3];
            Q8Gemm::linear_with_tile_rows(&linear, 33, tile_rows)
                .unwrap()
                .run(&input, &mut actual)
                .unwrap();
            for (actual, expected) in actual.into_iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-3, "{actual} vs {expected}");
            }
        }
    }
}
