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

const MSL: &str = include_str!("../../../../../runtime/apple/asr_rnnt.metal");
const THREADS: usize = 256;
const SIMD_GROUPS: usize = 8;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub(super) struct RnntPipelines {
    q8: Pipeline,
    embed: Pipeline,
    scaled_add: Pipeline,
    lstm: Pipeline,
    broadcast_add: Pipeline,
    relu: Pipeline,
    argmax: Pipeline,
}

pub(super) fn supports(insts: &[DevInst64]) -> bool {
    let predictor = insts
        .iter()
        .any(|inst| inst.op == DevOp::LstmCellF32 as u16);
    let joint = insts.iter().any(|inst| inst.op == DevOp::ArgmaxF32 as u16);
    (predictor || joint)
        && insts.iter().all(|inst| match DevOp::from_u16(inst.op) {
            Some(DevOp::Q8GemmF32) => {
                nonzero(&inst.i[..3])
                    && inst.i[2].is_multiple_of(32)
                    && inst.i[1].is_multiple_of(4)
                    && inst.i[1] <= u32::MAX - (SIMD_GROUPS as u32 * 4 - 1)
                    && inst.i[3] == 0
                    && inst.i[4] == 0
                    && inst.t[3] != TENSOR_NONE16
                    && product_fits_u32(inst.i[0], inst.i[1])
                    && tensors(inst, &[0, 1, 2, 3])
            }
            Some(DevOp::EmbedF16F32) => nonzero(&inst.i[..2]) && tensors(inst, &[0, 1, 2]),
            Some(DevOp::ScaledAddF32) => inst.i[0] != 0 && tensors(inst, &[0, 1, 2]),
            Some(DevOp::LstmCellF32) => inst.i[0] != 0 && tensors(inst, &[0, 1, 2, 3]),
            Some(DevOp::BroadcastAddF32) => {
                nonzero(&inst.i[..2])
                    && product_fits_u32(inst.i[0], inst.i[1])
                    && tensors(inst, &[0, 1, 2])
            }
            Some(DevOp::ReluF32) => inst.i[0] != 0 && tensors(inst, &[0, 1]),
            Some(DevOp::ArgmaxF32) => nonzero(&inst.i[..2]) && tensors(inst, &[0, 1]),
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

impl RnntPipelines {
    pub(super) fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        options: &MTLCompileOptions,
    ) -> Result<Self> {
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(MSL), Some(options))
            .map_err(|error| RuntimeError::Device(format!("compile RNNT kernels: {error}")))?;
        let pipelines = Self {
            q8: pipeline(device, &library, "rnnt_q8_gemv4")?,
            embed: pipeline(device, &library, "rnnt_embed_f16_f32")?,
            scaled_add: pipeline(device, &library, "rnnt_scaled_add")?,
            lstm: pipeline(device, &library, "rnnt_lstm_cell")?,
            broadcast_add: pipeline(device, &library, "rnnt_broadcast_add")?,
            relu: pipeline(device, &library, "rnnt_relu")?,
            argmax: pipeline(device, &library, "rnnt_argmax")?,
        };
        for pipeline in [
            &pipelines.q8,
            &pipelines.embed,
            &pipelines.scaled_add,
            &pipelines.lstm,
            &pipelines.broadcast_add,
            &pipelines.relu,
            &pipelines.argmax,
        ] {
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < THREADS
            {
                return Err(RuntimeError::Device(
                    "Metal device does not support RNNT launch geometry".into(),
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
            .ok_or_else(|| RuntimeError::Device("metal: no RNNT encoder".into()))?;
        for inst in insts {
            match DevOp::from_u16(inst.op) {
                Some(DevOp::Q8GemmF32) => self.encode_q8(&encoder, inst, buffers),
                Some(DevOp::EmbedF16F32) => self.encode_embed(&encoder, inst, buffers),
                Some(DevOp::ScaledAddF32) => self.encode_scaled_add(&encoder, inst, buffers),
                Some(DevOp::LstmCellF32) => self.encode_lstm(&encoder, inst, buffers),
                Some(DevOp::BroadcastAddF32) => self.encode_broadcast_add(&encoder, inst, buffers),
                Some(DevOp::ReluF32) => self.encode_relu(&encoder, inst, buffers),
                Some(DevOp::ArgmaxF32) => self.encode_argmax(&encoder, inst, buffers),
                _ => return Err(RuntimeError::Device("invalid RNNT fast program".into())),
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
        encoder.setComputePipelineState(&self.q8);
        bind(encoder, buffers, inst.t[1], 0, 0);
        bind(encoder, buffers, inst.t[2], 0, 1);
        bind(encoder, buffers, inst.t[3], 0, 2);
        bind(encoder, buffers, inst.t[0], 0, 3);
        let shape = [inst.i[0], inst.i[1], inst.i[2], SIMD_GROUPS as u32];
        bytes(encoder, &shape, 4);
        let groups_per_row = (inst.i[1] as usize).div_ceil(SIMD_GROUPS * 4);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            size(inst.i[0] as usize * groups_per_row),
            size(THREADS),
        );
    }

    fn encode_embed(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.embed);
        bind_slots(encoder, buffers, inst, &[0, 1, 2]);
        bytes(encoder, &[inst.i[0], inst.i[1]], 3);
        dispatch(encoder, inst.i[1] as usize);
    }

    fn encode_scaled_add(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.scaled_add);
        bind_slots(encoder, buffers, inst, &[0, 1, 2]);
        bytes(encoder, &inst.i[0], 3);
        bytes(encoder, &f32::from_bits(inst.fj[0]), 4);
        dispatch(encoder, inst.i[0] as usize);
    }

    fn encode_lstm(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.lstm);
        bind_slots(encoder, buffers, inst, &[0, 1, 2, 3]);
        bytes(encoder, &inst.i[0], 4);
        dispatch(encoder, inst.i[0] as usize);
    }

    fn encode_broadcast_add(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.broadcast_add);
        bind_slots(encoder, buffers, inst, &[0, 1, 2]);
        bytes(encoder, &[inst.i[0], inst.i[1]], 3);
        dispatch(encoder, inst.i[0] as usize * inst.i[1] as usize);
    }

    fn encode_relu(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.relu);
        bind_slots(encoder, buffers, inst, &[0, 1]);
        bytes(encoder, &inst.i[0], 2);
        dispatch(encoder, inst.i[0] as usize);
    }

    fn encode_argmax(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        inst: &DevInst64,
        buffers: &[Buf],
    ) {
        encoder.setComputePipelineState(&self.argmax);
        bind_slots(encoder, buffers, inst, &[0, 1]);
        bytes(encoder, &[inst.i[0], inst.i[1]], 2);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(size(inst.i[0] as usize), size(THREADS));
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

fn bind_slots(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffers: &[Buf],
    inst: &DevInst64,
    slots: &[usize],
) {
    for (index, &slot) in slots.iter().enumerate() {
        bind(encoder, buffers, inst.t[slot], 0, index);
    }
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

fn dispatch(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, count: usize) {
    encoder.dispatchThreads_threadsPerThreadgroup(size(count), size(THREADS));
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
    fn recognizes_predictor_and_joint_programs() {
        let mut q8 = inst(DevOp::Q8GemmF32);
        q8.i[..3].copy_from_slice(&[1, 2560, 640]);
        let mut lstm = inst(DevOp::LstmCellF32);
        lstm.i[0] = 640;
        assert!(supports(&[q8, lstm]));

        let mut argmax = inst(DevOp::ArgmaxF32);
        argmax.i[..2].copy_from_slice(&[1, 13088]);
        assert!(supports(&[q8, argmax]));

        q8.i[2] = 639;
        assert!(!supports(&[q8, lstm]));
        assert!(!supports(&[q8]));
    }
}
