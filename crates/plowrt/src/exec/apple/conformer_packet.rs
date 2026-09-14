use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};

use super::Buf;
use crate::{Result, RuntimeError};

const MSL: &str = concat!(
    include_str!("../../../../../runtime/apple/q8.metal"),
    "\n",
    include_str!("../../../../../runtime/apple/asr_conformer.metal")
);

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub(super) struct ConformerPipelines {
    q8_32: Pipeline,
    q8_64: Pipeline,
    norm: Pipeline,
    silu: Pipeline,
    scaled_add: Pipeline,
    glu: Pipeline,
    depthwise: Pipeline,
    attention: Pipeline,
}

pub(super) fn supports(insts: &[DevInst64]) -> bool {
    insts
        .iter()
        .any(|inst| inst.op == DevOp::RelativeAttentionF32 as u16)
        && insts.iter().all(|inst| match DevOp::from_u16(inst.op) {
            Some(DevOp::Q8GemmF32) => {
                nonzero(&inst.i[..3])
                    && inst.i[2].is_multiple_of(32)
                    && inst.i[3] <= 1
                    && product_fits_u32(inst.i[0], inst.i[1])
                    && byte_offset(inst.i[4], inst.i[2]).is_some()
                    && tensors(inst, &[0, 1, 2])
            }
            Some(DevOp::LayerNormF32) => {
                nonzero(&inst.i[..2])
                    && inst.i[2] == 0
                    && inst.fj[0] == 1e-5f32.to_bits()
                    && tensors(inst, &[0, 1, 2, 3])
            }
            Some(DevOp::ScaledAddF32) => inst.i[0] != 0 && tensors(inst, &[0, 1, 2]),
            Some(DevOp::SiluF32) => inst.i[0] != 0 && tensors(inst, &[0, 1]),
            Some(DevOp::GluF32) => nonzero(&inst.i[..2]) && tensors(inst, &[0, 1]),
            Some(DevOp::CausalDepthwiseConv1dF32) => {
                nonzero(&inst.i[..3]) && tensors(inst, &[0, 1, 2])
            }
            Some(DevOp::RelativeAttentionF32) => {
                nonzero(&inst.i[..4])
                    && inst.i[1].is_multiple_of(inst.i[2])
                    && inst.i[4]
                        .checked_add(1)
                        .and_then(|chunks| inst.i[3].checked_mul(chunks))
                        .is_some_and(|window| window <= 64)
                    && tensors(inst, &[0, 1, 2, 3, 4, 5, 6])
            }
            _ => false,
        })
}

fn nonzero(values: &[u32]) -> bool {
    values.iter().all(|&value| value != 0)
}

fn tensors(inst: &DevInst64, slots: &[usize]) -> bool {
    slots.iter().all(|&slot| inst.t[slot] != TENSOR_NONE16)
}

fn product_fits_u32(lhs: u32, rhs: u32) -> bool {
    lhs.checked_mul(rhs).is_some()
}

fn byte_offset(row: u32, width: u32) -> Option<usize> {
    (row as usize)
        .checked_mul(width as usize)?
        .checked_mul(std::mem::size_of::<f32>())
}

impl ConformerPipelines {
    pub(super) fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        options: &MTLCompileOptions,
    ) -> Result<Self> {
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(MSL), Some(options))
            .map_err(|error| RuntimeError::Device(format!("compile Conformer kernels: {error}")))?;
        let pipelines = Self {
            q8_32: pipeline(device, &library, "q8_0_gemm")?,
            q8_64: pipeline(device, &library, "q8_0_gemm64")?,
            norm: pipeline(device, &library, "conformer_layer_norm")?,
            silu: pipeline(device, &library, "conformer_silu")?,
            scaled_add: pipeline(device, &library, "conformer_scaled_add")?,
            glu: pipeline(device, &library, "conformer_glu")?,
            depthwise: pipeline(device, &library, "conformer_depthwise_causal")?,
            attention: pipeline(device, &library, "conformer_relative_attention_fused")?,
        };
        for (pipeline, threads) in [
            (&pipelines.q8_32, 256),
            (&pipelines.q8_64, 256),
            (&pipelines.norm, 32),
            (&pipelines.silu, 256),
            (&pipelines.scaled_add, 256),
            (&pipelines.glu, 256),
            (&pipelines.depthwise, 256),
            (&pipelines.attention, 128),
        ] {
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < threads
            {
                return Err(RuntimeError::Device(
                    "Metal device does not support Conformer launch geometry".into(),
                ));
            }
        }
        Ok(pipelines)
    }

    pub(super) fn encode(
        &self,
        insts: &[DevInst64],
        buffers: &[Buf],
        command: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<()> {
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| RuntimeError::Device("metal: no Conformer encoder".into()))?;
        for inst in insts {
            match DevOp::from_u16(inst.op) {
                Some(DevOp::Q8GemmF32) => self.encode_q8(&encoder, inst, buffers),
                Some(DevOp::LayerNormF32) => self.encode_norm(&encoder, inst, buffers),
                Some(DevOp::ScaledAddF32) => self.encode_scaled_add(&encoder, inst, buffers),
                Some(DevOp::GluF32) => self.encode_glu(&encoder, inst, buffers),
                Some(DevOp::CausalDepthwiseConv1dF32) => {
                    self.encode_depthwise(&encoder, inst, buffers)
                }
                Some(DevOp::RelativeAttentionF32) => self.encode_attention(&encoder, inst, buffers),
                Some(DevOp::SiluF32) => self.encode_silu(&encoder, inst, buffers),
                _ => {
                    return Err(RuntimeError::Device(
                        "invalid Conformer fast program".into(),
                    ))
                }
            }
        }
        encoder.endEncoding();
        Ok(())
    }

    fn encode_q8(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        let (m, n, k) = (inst.i[0] as usize, inst.i[1] as usize, inst.i[2] as usize);
        let tile_rows = if m >= 192 || n >= 2048 { 64 } else { 32 };
        encoder.setComputePipelineState(if tile_rows == 64 {
            &self.q8_64
        } else {
            &self.q8_32
        });
        // `supports` checked this offset before selecting the fast route.
        bind(
            encoder,
            buffers,
            inst.t[1],
            byte_offset(inst.i[4], inst.i[2]).unwrap(),
            0,
        );
        bind(encoder, buffers, inst.t[2], 0, 1);
        bind(
            encoder,
            buffers,
            if inst.t[3] == TENSOR_NONE16 {
                inst.t[0]
            } else {
                inst.t[3]
            },
            0,
            2,
        );
        bind(encoder, buffers, inst.t[0], 0, 3);
        let shape = [
            m as u32,
            n as u32,
            k as u32,
            u32::from(inst.t[3] != TENSOR_NONE16),
        ];
        bytes(encoder, &shape, 4);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            size(m.div_ceil(tile_rows) * n.div_ceil(64)),
            size(256),
        );
        if inst.i[3] == 1 {
            encoder.setComputePipelineState(&self.silu);
            bind(encoder, buffers, inst.t[0], 0, 0);
            bind(encoder, buffers, inst.t[0], 0, 1);
            let count = (m * n) as u32;
            bytes(encoder, &count, 2);
            dispatch_elements(encoder, m * n);
        }
    }

    fn encode_norm(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.norm);
        bind(encoder, buffers, inst.t[1], 0, 0);
        bind(encoder, buffers, inst.t[2], 0, 1);
        bind(encoder, buffers, inst.t[3], 0, 2);
        bind(encoder, buffers, inst.t[0], 0, 3);
        let shape = [inst.i[0], inst.i[1]];
        bytes(encoder, &shape, 4);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(size(inst.i[0] as usize), size(32));
    }

    fn encode_scaled_add(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.scaled_add);
        bind(encoder, buffers, inst.t[1], 0, 0);
        bind(encoder, buffers, inst.t[2], 0, 1);
        bind(encoder, buffers, inst.t[0], 0, 2);
        bytes(encoder, &inst.i[0], 3);
        bytes(encoder, &f32::from_bits(inst.fj[0]), 4);
        dispatch_elements(encoder, inst.i[0] as usize);
    }

    fn encode_glu(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.glu);
        bind(encoder, buffers, inst.t[1], 0, 0);
        bind(encoder, buffers, inst.t[0], 0, 1);
        let shape = [inst.i[0], inst.i[1]];
        bytes(encoder, &shape, 2);
        dispatch_elements(encoder, inst.i[0] as usize * inst.i[1] as usize);
    }

    fn encode_depthwise(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.depthwise);
        bind(encoder, buffers, inst.t[1], 0, 0);
        bind(encoder, buffers, inst.t[2], 0, 1);
        bind(encoder, buffers, inst.t[0], 0, 2);
        let shape = [inst.i[0], inst.i[1], inst.i[2]];
        bytes(encoder, &shape, 3);
        dispatch_elements(encoder, inst.i[0] as usize * inst.i[1] as usize);
    }

    fn encode_attention(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.attention);
        for (index, slot) in [1, 2, 3, 4, 5, 6, 0].into_iter().enumerate() {
            bind(encoder, buffers, inst.t[slot], 0, index);
        }
        let shape = [inst.i[0], inst.i[1], inst.i[2], inst.i[3], inst.i[4]];
        bytes(encoder, &shape, 7);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            size(inst.i[0] as usize * inst.i[2] as usize),
            size(128),
        );
    }

    fn encode_silu(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.silu);
        bind(encoder, buffers, inst.t[1], 0, 0);
        bind(encoder, buffers, inst.t[0], 0, 1);
        bytes(encoder, &inst.i[0], 2);
        dispatch_elements(encoder, inst.i[0] as usize);
    }
}

fn pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Pipeline> {
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| RuntimeError::Device(format!("Metal function {name} is missing")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| RuntimeError::Device(format!("create {name} pipeline: {error}")))
}

fn bind(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffers: &[Buf],
    handle: u16,
    offset: usize,
    index: usize,
) {
    unsafe { encoder.setBuffer_offset_atIndex(Some(&buffers[handle as usize]), offset, index) };
}

fn bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new(value as *const T as *mut c_void).unwrap(),
            std::mem::size_of::<T>(),
            index,
        )
    };
}

fn dispatch_elements(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, count: usize) {
    encoder.dispatchThreads_threadsPerThreadgroup(size(count), size(256));
}

fn size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(op: DevOp) -> DevInst64 {
        DevInst64 {
            op: op as u16,
            ..DevInst64::default()
        }
    }

    #[test]
    fn recognizes_only_supported_conformer_programs() {
        let mut q8 = inst(DevOp::Q8GemmF32);
        q8.i[..3].copy_from_slice(&[10, 64, 64]);
        let mut norm = inst(DevOp::LayerNormF32);
        norm.i[..2].copy_from_slice(&[10, 64]);
        norm.fj[0] = 1e-5f32.to_bits();
        let mut attention = inst(DevOp::RelativeAttentionF32);
        attention.i[..5].copy_from_slice(&[10, 64, 4, 4, 1]);

        assert!(supports(&[q8, norm, attention]));
        norm.fj[0] = 1e-6f32.to_bits();
        assert!(!supports(&[q8, norm, attention]));
        norm.fj[0] = 1e-5f32.to_bits();
        q8.i[0] = u32::MAX;
        assert!(!supports(&[q8, norm, attention]));
        assert!(!supports(&[q8]));
    }
}
